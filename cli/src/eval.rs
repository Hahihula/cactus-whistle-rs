//! `whistle eval MODEL.cact SPLIT.jsonl --out HYP.jsonl [options]`:
//! transcribe a dataset split (the fine-tuning JSONL format: `audio` relative
//! to the file, `text`, `language`) on every core and write one
//! `{"audio", "ref", "hyp", "lang"}` line per clip, in input order.

use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use cactus_whistle::{Spec, Whistle};
use cactus_whistle_cli::{load, load_lora};

pub const USAGE: &str = "\
whistle eval MODEL.cact SPLIT.jsonl --out HYP.jsonl [options]
  SPLIT.jsonl        {\"audio\": path relative to the file, \"text\": ..., \"language\": ...}
  --out HYP.jsonl    one {\"audio\", \"ref\", \"hyp\", \"lang\"} line per clip, in input order
  --lang xx|auto|row force a language, detect, or use each row's language (default)
  --lora ADAPTER     run with a LoRA adapter on its base model
  --lora-strength X  scale the adapter's update (default 1)
  --spec guide|fast|c
  --limit N          only the first N clips
  --threads N        worker threads (default: all cores)";

struct Row {
    audio: String,
    text: String,
    lang: Option<String>,
}

pub fn run(args: &[String]) {
    if args.len() < 2 {
        eprintln!("{USAGE}");
        std::process::exit(2);
    }
    let (model_path, split) = (&args[0], &args[1]);
    let (mut out, mut lang_arg, mut lora_path, mut strength) =
        (None::<String>, "row".to_string(), None::<String>, 1.0f32);
    let (mut spec, mut limit, mut threads) = (Spec::DEFAULT, usize::MAX, 0usize);
    let mut it = args[2..].iter();
    while let Some(a) = it.next() {
        let v = it.next().unwrap_or_else(|| {
            eprintln!("{a} needs a value\n\n{USAGE}");
            std::process::exit(2)
        });
        match a.as_str() {
            "--out" => out = Some(v.clone()),
            "--lang" => lang_arg = v.clone(),
            "--lora" => lora_path = Some(v.clone()),
            "--lora-strength" => strength = v.parse().expect("lora strength"),
            "--limit" => limit = v.parse().expect("limit"),
            "--threads" => threads = v.parse().expect("threads"),
            "--spec" => {
                spec = match v.as_str() {
                    "guide" => Spec::Guide,
                    "fast" => Spec::GuideFast,
                    "c" => Spec::CEngine,
                    o => panic!("unknown spec {o}"),
                }
            }
            _ => {
                eprintln!("unknown argument {a}\n\n{USAGE}");
                std::process::exit(2);
            }
        }
    }
    let Some(out) = out else {
        eprintln!("--out is required\n\n{USAGE}");
        std::process::exit(2);
    };

    let text = std::fs::read_to_string(split).unwrap_or_else(|e| {
        eprintln!("{split}: {e}");
        std::process::exit(1)
    });
    let root = std::path::Path::new(split)
        .parent()
        .unwrap_or(std::path::Path::new("."));
    let rows: Vec<Row> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .take(limit)
        .enumerate()
        .map(|(i, l)| {
            let v: serde_json::Value = serde_json::from_str(l).unwrap_or_else(|e| {
                eprintln!("{split}:{}: {e}", i + 1);
                std::process::exit(1)
            });
            let s = |k: &str| v.get(k).and_then(|x| x.as_str()).map(str::to_string);
            Row {
                audio: s("audio").expect("row without audio"),
                text: s("text").unwrap_or_default(),
                lang: s("language"),
            }
        })
        .collect();

    let blob = crate::read(model_path);
    let probe = Whistle::open(&blob).unwrap_or_else(|e| {
        eprintln!("cannot open {model_path}: {e}");
        std::process::exit(1)
    });
    let adapter = lora_path.as_deref().map(|p| {
        load_lora(&probe, p).unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(1)
        })
    });
    let lora = adapter.as_ref().map(|(a, tok)| a.lora(tok));
    let threads = if threads == 0 {
        std::thread::available_parallelism().map_or(1, |n| n.get())
    } else {
        threads
    };
    let open = || {
        let mut w = Whistle::open(&blob).expect("open");
        if let Some(l) = &lora {
            w.set_lora(Some(l)).unwrap_or_else(|e| {
                eprintln!("{}: {e}", lora_path.as_deref().unwrap_or(""));
                std::process::exit(1)
            });
            w.lora_strength = strength;
        }
        w.set_spec(spec);
        w
    };
    // Resolve a forced language once, against the (possibly extended) model.
    let forced = match lang_arg.as_str() {
        "row" | "auto" => None,
        code => Some(open().lang_index(code).unwrap_or_else(|| {
            eprintln!("the model has no language {code}");
            std::process::exit(2)
        })),
    };
    eprintln!(
        "eval  {} clips  {threads} threads  {}{}",
        rows.len(),
        spec.name(),
        lora_path
            .as_deref()
            .map(|p| format!("  lora {p} x{strength}"))
            .unwrap_or_default()
    );

    let t0 = Instant::now();
    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let mut hyps: Vec<(String, String)> = vec![Default::default(); rows.len()];
    let slots: Vec<std::sync::Mutex<&mut (String, String)>> =
        hyps.iter_mut().map(std::sync::Mutex::new).collect();
    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| {
                let mut w = open();
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(r) = rows.get(i) else { break };
                    let lang = match lang_arg.as_str() {
                        "auto" => None,
                        "row" => r.lang.as_deref().and_then(|c| w.lang_index(c)),
                        _ => forced,
                    };
                    let path = root.join(&r.audio);
                    let hyp = load(path.to_str().unwrap_or_default())
                        .map_err(|e| e.to_string())
                        .and_then(|x| w.transcribe(&x, lang).map_err(|e| e.to_string()));
                    let hyp = hyp.unwrap_or_else(|e| {
                        eprintln!("{}: {e}", r.audio);
                        String::new()
                    });
                    **slots[i].lock().unwrap() = (hyp, w.lang_code(w.lang).to_string());
                    let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                    if n.is_multiple_of(50) || n == rows.len() {
                        let s = t0.elapsed().as_secs_f64();
                        eprintln!(
                            "  {n}/{}  {s:.0} s  eta {:.0} s",
                            rows.len(),
                            s / n as f64 * (rows.len() - n) as f64
                        );
                    }
                }
            });
        }
    });
    drop(slots);

    let mut f = std::io::BufWriter::new(std::fs::File::create(&out).unwrap_or_else(|e| {
        eprintln!("{out}: {e}");
        std::process::exit(1)
    }));
    for (r, (hyp, lang)) in rows.iter().zip(&hyps) {
        let line = serde_json::json!({"audio": r.audio, "ref": r.text, "hyp": hyp, "lang": lang});
        writeln!(f, "{line}").expect("write");
    }
    eprintln!("wrote {out}  ({:.0} s)", t0.elapsed().as_secs_f64());
}
