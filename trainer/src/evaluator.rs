//! Checkpoint evaluation on the CPU with the Rust engine and runtime LoRA,
//! while the GPU keeps training: WER on a slice of the validation split, the
//! sample recordings, and after a run the whole test split.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::JoinHandle;
use std::time::Instant;

use cactus_whistle::{Spec, Whistle};
use cactus_whistle_common::{dataset, load, load_lora, wer};
use serde_json::{Value, json};

use crate::config::Sweep;
use crate::state::append_event;

pub enum Job {
    /// Score a checkpoint written at `step`.
    Checkpoint {
        run_dir: PathBuf,
        step: u64,
        adapter: PathBuf,
    },
    /// Score a run's best adapter on the test split.
    Test { run_dir: PathBuf, adapter: PathBuf },
}

struct Item {
    audio: String,
    reference: Option<String>,
    source: String,
    lang: Option<String>,
    samples: Vec<f32>,
}

pub struct Evaluator {
    /// Checkpoints are scored (validation clips or samples configured).
    pub checkpoints: bool,
    /// The best adapter is scored on the test split after each run.
    pub tests: bool,
    tx: Option<Sender<Job>>,
    handle: Option<JoinHandle<()>>,
    pub pending: Arc<AtomicUsize>,
}

impl Evaluator {
    pub fn start(sweep: &Sweep) -> Result<Evaluator, String> {
        let blob = std::fs::read(&sweep.eval.base_cact)
            .map_err(|e| format!("{}: {e}", sweep.eval.base_cact.display()))?;
        Whistle::open(&blob).map_err(|e| format!("{}: {e}", sweep.eval.base_cact.display()))?;
        let spec = parse_spec(&sweep.eval.spec)?;
        let test_spec = parse_spec(&sweep.eval.test_spec)?;
        let mut val = Vec::new();
        if sweep.eval.clips > 0 {
            for c in dataset::read_jsonl(&sweep.data.validation, Some(sweep.eval.clips))? {
                let samples = load(c.path.to_str().unwrap_or_default())?;
                val.push(Item {
                    audio: c.audio,
                    reference: Some(c.text),
                    source: c.source,
                    lang: c.language,
                    samples,
                });
            }
        }
        let mut samples = Vec::new();
        for s in &sweep.eval.sample_list {
            samples.push(Item {
                audio: s
                    .audio
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("")
                    .to_string(),
                reference: s.text.clone(),
                source: "sample".into(),
                lang: None,
                samples: load(s.audio.to_str().unwrap_or_default())?,
            });
        }
        let checkpoints = !val.is_empty() || !samples.is_empty();
        let test_path = if sweep.eval.test_at_end {
            sweep.data.test.clone()
        } else {
            None
        };
        let (threads, lang) = (sweep.eval.threads.max(1), sweep.eval.lang.clone());
        let (tx, rx) = channel::<Job>();
        let pending = Arc::new(AtomicUsize::new(0));
        let p2 = pending.clone();
        let tests = test_path.is_some();
        let handle = std::thread::spawn(move || {
            // The E-cores of a hybrid CPU; the trainer has the P-cores.
            crate::affinity::efficiency_cores();
            let ctx = Ctx {
                blob,
                spec,
                threads,
                lang,
            };
            worker(rx, ctx, val, samples, test_path, test_spec, p2)
        });
        Ok(Evaluator {
            checkpoints,
            tests,
            tx: Some(tx),
            handle: Some(handle),
            pending,
        })
    }

    pub fn submit(&self, job: Job) {
        if let Some(tx) = &self.tx {
            self.pending.fetch_add(1, Ordering::SeqCst);
            let _ = tx.send(job);
        }
    }

    /// Finish the queued jobs.
    pub fn finish(mut self) {
        drop(self.tx.take());
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn parse_spec(s: &str) -> Result<Spec, String> {
    match s {
        "guide" => Ok(Spec::Guide),
        "fast" => Ok(Spec::GuideFast),
        "c" => Ok(Spec::CEngine),
        o => Err(format!("eval spec must be guide, fast or c, not {o}")),
    }
}

struct Ctx {
    blob: Vec<u8>,
    spec: Spec,
    threads: usize,
    lang: Option<String>,
}

/// Transcribe `items` with the adapter on every thread; hypotheses and
/// detected languages in order.
fn transcribe_all(
    ctx: &Ctx,
    spec: Spec,
    adapter: &Path,
    items: &[Item],
) -> Result<Vec<(String, String)>, String> {
    let probe = Whistle::open(&ctx.blob).map_err(|e| e.to_string())?;
    let (a, tok) = load_lora(&probe, adapter.to_str().unwrap_or_default())?;
    let lora = a.lora(&tok);
    let mut check = Whistle::open(&ctx.blob).map_err(|e| e.to_string())?;
    check
        .set_lora(Some(&lora))
        .map_err(|e| format!("{}: {e}", adapter.display()))?;
    let next = AtomicUsize::new(0);
    let out: Vec<std::sync::Mutex<(String, String)>> =
        items.iter().map(|_| Default::default()).collect();
    std::thread::scope(|sc| {
        for _ in 0..ctx.threads.min(items.len()).max(1) {
            sc.spawn(|| {
                let Ok(mut w) = Whistle::open(&ctx.blob) else {
                    return;
                };
                if w.set_lora(Some(&lora)).is_err() {
                    return;
                }
                w.set_spec(spec);
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(it) = items.get(i) else { break };
                    let code = ctx.lang.as_deref().or(it.lang.as_deref());
                    let lang = code.and_then(|c| w.lang_index(c));
                    let hyp = w.transcribe(&it.samples, lang).unwrap_or_default();
                    *out[i].lock().unwrap() = (hyp, w.lang_code(w.lang).to_string());
                }
            });
        }
    });
    Ok(out.into_iter().map(|m| m.into_inner().unwrap()).collect())
}

fn report(items: &[Item], hyps: &[(String, String)]) -> wer::Report {
    let mut r = wer::Report::default();
    for (it, (h, _)) in items.iter().zip(hyps) {
        if let Some(reference) = &it.reference {
            r.add(&it.source, reference, h);
        }
    }
    r
}

fn wer_json(r: &wer::Report) -> Value {
    let by: serde_json::Map<String, Value> = r
        .by_source
        .iter()
        .map(|(k, c)| (k.clone(), json!(c.wer())))
        .collect();
    json!({"wer": r.all.wer(), "strict": r.all.strict(), "clips": r.all.clips, "by_source": by})
}

fn worker(
    rx: Receiver<Job>,
    ctx: Ctx,
    val: Vec<Item>,
    samples: Vec<Item>,
    test_path: Option<PathBuf>,
    test_spec: Spec,
    pending: Arc<AtomicUsize>,
) {
    for job in rx {
        match job {
            Job::Checkpoint {
                run_dir,
                step,
                adapter,
            } => {
                let t0 = Instant::now();
                append_event(
                    &run_dir,
                    json!({"event": "evaluating", "what": "checkpoint", "step": step,
                           "clips": val.len() + samples.len()}),
                );
                if !samples.is_empty() {
                    match transcribe_all(&ctx, ctx.spec, &adapter, &samples) {
                        Ok(h) => {
                            let items: Vec<Value> = samples
                                .iter()
                                .zip(&h)
                                .map(|(s, (hyp, lang))| json!({"audio": s.audio, "ref": s.reference, "hyp": hyp, "lang": lang}))
                                .collect();
                            append_event(
                                &run_dir,
                                json!({"event": "samples", "step": step, "items": items}),
                            );
                        }
                        Err(e) => append_event(
                            &run_dir,
                            json!({"event": "log", "msg": format!("sample eval failed: {e}")}),
                        ),
                    }
                }
                if !val.is_empty() {
                    match transcribe_all(&ctx, ctx.spec, &adapter, &val) {
                        Ok(h) => {
                            let mut v = wer_json(&report(&val, &h));
                            v["event"] = "wer".into();
                            v["step"] = step.into();
                            v["secs"] = t0.elapsed().as_secs_f64().into();
                            append_event(&run_dir, v);
                        }
                        Err(e) => append_event(
                            &run_dir,
                            json!({"event": "log", "msg": format!("WER eval failed: {e}")}),
                        ),
                    }
                }
                append_event(
                    &run_dir,
                    json!({"event": "evaluated", "what": "checkpoint", "step": step,
                           "secs": t0.elapsed().as_secs_f64()}),
                );
            }
            Job::Test { run_dir, adapter } => {
                let Some(path) = &test_path else {
                    pending.fetch_sub(1, Ordering::SeqCst);
                    continue;
                };
                append_event(
                    &run_dir,
                    json!({"event": "log", "msg": "scoring the best adapter on the test split"}),
                );
                let t0 = Instant::now();
                let items: Result<Vec<Item>, String> =
                    dataset::read_jsonl(path, None).and_then(|clips| {
                        clips
                            .into_iter()
                            .map(|c| {
                                Ok(Item {
                                    samples: load(c.path.to_str().unwrap_or_default())?,
                                    audio: c.audio,
                                    reference: Some(c.text),
                                    source: c.source,
                                    lang: c.language,
                                })
                            })
                            .collect()
                    });
                let res = items.and_then(|items| {
                    append_event(
                        &run_dir,
                        json!({"event": "evaluating", "what": "test", "clips": items.len()}),
                    );
                    let h = transcribe_all(&ctx, test_spec, &adapter, &items)?;
                    let hyp_path = run_dir.join("test_hyp.jsonl");
                    let lines: Vec<String> = items
                        .iter()
                        .zip(&h)
                        .map(|(it, (hyp, lang))| json!({"audio": it.audio, "ref": it.reference, "hyp": hyp, "lang": lang}).to_string())
                        .collect();
                    let _ = std::fs::write(&hyp_path, lines.join("\n") + "\n");
                    Ok(report(&items, &h))
                });
                match res {
                    Ok(r) => {
                        let mut v = wer_json(&r);
                        v["event"] = "test".into();
                        v["secs"] = t0.elapsed().as_secs_f64().into();
                        append_event(&run_dir, v);
                    }
                    Err(e) => append_event(
                        &run_dir,
                        json!({"event": "log", "msg": format!("test eval failed: {e}")}),
                    ),
                }
                append_event(
                    &run_dir,
                    json!({"event": "evaluated", "what": "test", "secs": t0.elapsed().as_secs_f64()}),
                );
            }
        }
        pending.fetch_sub(1, Ordering::SeqCst);
    }
}
