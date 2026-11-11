//! What the TUI shows, rebuilt from the runs' `events.jsonl` files: the same
//! code serves `run` (which writes them) and `watch` (which only reads).
//!
//! Event sources: the trainer (`log`, `start`, `step`, `eval`,
//! `early_stop`, `stopped`, `done`, `error`), the runner (`status`, `stderr`,
//! `built`) and the evaluator (`wer`, `samples`, `test`). The runner adds `t`
//! (Unix seconds) to every event.

use std::collections::{BTreeMap, VecDeque};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use serde_json::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Queued,
    Running,
    Done,
    Failed,
    Stalled,
    Stopped,
}

impl Status {
    pub fn parse(s: &str) -> Status {
        match s {
            "running" => Status::Running,
            "done" => Status::Done,
            "failed" => Status::Failed,
            "stalled" => Status::Stalled,
            "stopped" => Status::Stopped,
            _ => Status::Queued,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Status::Queued => "queued",
            Status::Running => "running",
            Status::Done => "done",
            Status::Failed => "failed",
            Status::Stalled => "stalled",
            Status::Stopped => "stopped",
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct StepPt {
    pub step: f64,
    pub epoch: f64,
    pub loss: f64,
    pub loss_max: f64,
    pub lr: f64,
    pub grad_norm: f64,
    pub clipped: f64,
    pub secs: f64,
    pub t: f64,
}

#[derive(Clone, Debug, Default)]
pub struct EvalPt {
    pub step: f64,
    pub val_loss: f64,
}

#[derive(Clone, Debug, Default)]
pub struct WerPt {
    pub step: f64,
    pub wer: f64,
    pub strict: f64,
    pub clips: usize,
    pub by_source: BTreeMap<String, f64>,
}

#[derive(Clone, Debug, Default)]
pub struct SampleOut {
    pub audio: String,
    pub reference: Option<String>,
    pub hyp: String,
    pub lang: String,
}

#[derive(Clone, Debug)]
pub struct RunState {
    pub name: String,
    pub dir: PathBuf,
    pub status: Status,
    pub status_msg: String,
    pub settings: Value,
    pub start: Option<Value>,
    pub steps: Vec<StepPt>,
    pub evals: Vec<EvalPt>,
    pub wers: Vec<WerPt>,
    pub samples: Vec<(f64, Vec<SampleOut>)>,
    pub test: Option<WerPt>,
    pub done: Option<Value>,
    pub error: Option<String>,
    pub early_stop: Option<f64>,
    /// The evaluator scores this run's checkpoints.
    pub evaluates: bool,
    /// Checkpoint steps the evaluator has finished.
    pub evaluated: Vec<f64>,
    /// What the evaluator is on: ("checkpoint" | "test", step, started at, clips).
    pub busy: Option<(String, f64, f64, usize)>,
    /// Seconds per checkpoint evaluation, most recent last.
    pub eval_secs: Vec<f64>,
    pub test_state: TestState,
    pub log: VecDeque<String>,
    pub first_t: Option<f64>,
    pub last_t: f64,
    offset: u64,
}

fn f(v: &Value, k: &str) -> f64 {
    v.get(k).and_then(Value::as_f64).unwrap_or(0.0)
}

fn wer_pt(v: &Value) -> WerPt {
    WerPt {
        step: f(v, "step"),
        wer: f(v, "wer"),
        strict: f(v, "strict"),
        clips: f(v, "clips") as usize,
        by_source: v
            .get("by_source")
            .and_then(Value::as_object)
            .map(|m| {
                m.iter()
                    .map(|(k, x)| (k.clone(), x.as_f64().unwrap_or(0.0)))
                    .collect()
            })
            .unwrap_or_default(),
    }
}

impl RunState {
    pub fn new(name: &str, dir: PathBuf) -> RunState {
        RunState {
            name: name.into(),
            dir,
            status: Status::Queued,
            status_msg: String::new(),
            settings: Value::Null,
            start: None,
            steps: Vec::new(),
            evals: Vec::new(),
            wers: Vec::new(),
            samples: Vec::new(),
            test: None,
            done: None,
            error: None,
            early_stop: None,
            evaluates: false,
            evaluated: Vec::new(),
            busy: None,
            eval_secs: Vec::new(),
            test_state: TestState::None,
            log: VecDeque::new(),
            first_t: None,
            last_t: 0.0,
            offset: 0,
        }
    }

    fn push_log(&mut self, s: String) {
        if self.log.len() >= 500 {
            self.log.pop_front();
        }
        self.log.push_back(s);
    }

    pub fn apply(&mut self, v: &Value) {
        let t = f(v, "t");
        if t > 0.0 {
            self.first_t.get_or_insert(t);
            self.last_t = t;
        }
        let ev = v.get("event").and_then(Value::as_str).unwrap_or("");
        match ev {
            "status" => {
                self.status = Status::parse(v.get("status").and_then(Value::as_str).unwrap_or(""));
                if let Some(e) = v.get("evaluates").and_then(Value::as_bool) {
                    self.evaluates = e;
                }
                self.status_msg = v
                    .get("msg")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                if self.status == Status::Running && v.get("resume") != Some(&Value::Bool(true)) {
                    // A fresh start: forget an earlier attempt's curves.
                    self.steps.clear();
                    self.evals.clear();
                    self.wers.clear();
                    self.samples.clear();
                    self.done = None;
                    self.error = None;
                    self.early_stop = None;
                    self.evaluated.clear();
                    self.busy = None;
                    self.test_state = TestState::None;
                }
                self.push_log(format!("[{}] {}", self.status.name(), self.status_msg));
            }
            "log" | "stderr" => {
                let msg = v.get("msg").and_then(Value::as_str).unwrap_or("");
                self.push_log(if ev == "stderr" {
                    format!("  {msg}")
                } else {
                    msg.to_string()
                });
            }
            "start" => {
                self.start = Some(v.clone());
                self.push_log(format!(
                    "start: {} train clips, {} steps",
                    f(v, "train_clips"),
                    f(v, "steps")
                ));
            }
            "step" => {
                // A resumed run repeats nothing, but a restarted one may: keep steps increasing.
                let s = f(v, "step");
                self.steps.retain(|p| p.step < s);
                self.steps.push(StepPt {
                    step: s,
                    epoch: f(v, "epoch"),
                    loss: f(v, "loss"),
                    loss_max: f(v, "loss_max"),
                    lr: f(v, "lr"),
                    grad_norm: f(v, "grad_norm"),
                    clipped: f(v, "clipped"),
                    secs: f(v, "secs"),
                    t,
                });
            }
            "eval" => {
                let s = f(v, "step");
                self.evals.retain(|p| p.step < s);
                self.evals.push(EvalPt {
                    step: s,
                    val_loss: f(v, "val_loss"),
                });
            }
            "wer" => {
                let p = wer_pt(v);
                self.wers.retain(|x| x.step != p.step);
                self.wers.push(p);
                self.wers.sort_by(|a, b| a.step.total_cmp(&b.step));
            }
            "samples" => {
                let step = f(v, "step");
                let items = v
                    .get("items")
                    .and_then(Value::as_array)
                    .map(|a| {
                        a.iter()
                            .map(|x| SampleOut {
                                audio: x.get("audio").and_then(Value::as_str).unwrap_or("").into(),
                                reference: x.get("ref").and_then(Value::as_str).map(str::to_string),
                                hyp: x.get("hyp").and_then(Value::as_str).unwrap_or("").into(),
                                lang: x.get("lang").and_then(Value::as_str).unwrap_or("").into(),
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                self.samples.retain(|(s, _)| *s != step);
                self.samples.push((step, items));
                self.samples.sort_by(|a, b| a.0.total_cmp(&b.0));
            }
            "test" => {
                self.test_state = TestState::Done;
                self.test = Some(wer_pt(v));
                self.push_log(format!("test WER {:.1}%", 100.0 * f(v, "wer")));
            }
            "early_stop" => self.early_stop = Some(f(v, "step")),
            "stopped" => self.push_log(format!("stopped at step {}, state saved", f(v, "step"))),
            "done" => {
                self.done = Some(v.clone());
                self.push_log("training finished".into());
            }
            "error" => {
                let msg = v
                    .get("msg")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                for l in msg.lines() {
                    self.push_log(format!("ERROR {l}"));
                }
                self.error = Some(msg);
            }
            "evaluating" => {
                let what = v
                    .get("what")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                if what == "test" {
                    self.test_state = TestState::Running;
                }
                self.busy = Some((what, f(v, "step"), t, f(v, "clips") as usize));
            }
            "evaluated" => {
                let what = v.get("what").and_then(Value::as_str).unwrap_or("");
                if what == "checkpoint" {
                    self.evaluated.push(f(v, "step"));
                    self.eval_secs.push(f(v, "secs"));
                } else if what == "test" && self.test_state != TestState::Done {
                    self.test_state = TestState::Failed;
                }
                self.busy = None;
            }
            "test_queued" => self.test_state = TestState::Queued,
            "built" => self.push_log(format!(
                "built {}",
                v.get("path").and_then(Value::as_str).unwrap_or("")
            )),
            _ => {}
        }
    }

    /// Read events appended since the last call.
    pub fn poll(&mut self) {
        if self.settings.is_null()
            && let Ok(s) = std::fs::read_to_string(self.dir.join("config.json"))
        {
            self.settings = serde_json::from_str(&s).unwrap_or(Value::Null);
        }
        let Ok(mut file) = std::fs::File::open(self.dir.join("events.jsonl")) else {
            return;
        };
        let Ok(len) = file.metadata().map(|m| m.len()) else {
            return;
        };
        if len < self.offset {
            // Rewritten (a forced re-run): start over.
            let (name, dir) = (self.name.clone(), self.dir.clone());
            *self = RunState::new(&name, dir);
        }
        if len == self.offset || file.seek(SeekFrom::Start(self.offset)).is_err() {
            return;
        }
        let mut buf = Vec::new();
        if file.read_to_end(&mut buf).is_err() {
            return;
        }
        // Only whole lines; a half-written one waits for the next poll.
        let end = buf.iter().rposition(|&b| b == b'\n').map_or(0, |i| i + 1);
        for line in buf[..end].split(|&b| b == b'\n') {
            if let Ok(v) = serde_json::from_slice::<Value>(line) {
                self.apply(&v);
            }
        }
        self.offset += end as u64;
    }

    pub fn total_steps(&self) -> f64 {
        self.start.as_ref().map_or(0.0, |s| f(s, "steps"))
    }

    pub fn step(&self) -> f64 {
        self.steps.last().map_or(0.0, |p| p.step)
    }

    pub fn best_eval(&self) -> Option<&EvalPt> {
        self.evals
            .iter()
            .min_by(|a, b| a.val_loss.total_cmp(&b.val_loss))
    }

    /// Seconds per step, from the recent `step` events (not the first,
    /// which includes compilation).
    pub fn secs_per_step(&self) -> Option<f64> {
        let pts: Vec<&StepPt> = self.steps.iter().skip(1).rev().take(10).collect();
        if pts.len() < 2 {
            return None;
        }
        let (a, b) = (pts.last()?, pts.first()?);
        if b.t > a.t && b.step > a.step {
            Some((b.t - a.t) / (b.step - a.step))
        } else {
            Some(pts.iter().map(|p| p.secs).sum::<f64>() / pts.len() as f64)
        }
    }

    /// Checkpoints written but not yet scored by the evaluator.
    pub fn checkpoints_pending(&self) -> usize {
        if !self.evaluates {
            return 0;
        }
        self.evals
            .iter()
            .filter(|e| !self.evaluated.contains(&e.step))
            .count()
    }

    /// The evaluator still has work for this run (checkpoints or the test split).
    pub fn evaluating(&self) -> bool {
        self.checkpoints_pending() > 0
            || matches!(self.test_state, TestState::Queued | TestState::Running)
    }

    /// Rough seconds until the evaluator is done with this run's checkpoints
    /// (the test split is not included: its length is only known once it runs).
    pub fn eval_eta_secs(&self, now: f64) -> Option<f64> {
        let n = self.checkpoints_pending();
        if n == 0 || self.eval_secs.is_empty() {
            return None;
        }
        let recent: Vec<f64> = self.eval_secs.iter().rev().take(5).copied().collect();
        let per = recent.iter().sum::<f64>() / recent.len() as f64;
        let into = match &self.busy {
            Some((w, _, since, _)) if w == "checkpoint" => (now - since).clamp(0.0, per),
            _ => 0.0,
        };
        Some(per * n as f64 - into)
    }

    pub fn eta_secs(&self) -> Option<f64> {
        if self.status != Status::Running {
            return None;
        }
        Some(self.secs_per_step()? * (self.total_steps() - self.step()).max(0.0))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TestState {
    None,
    Queued,
    Running,
    Done,
    Failed,
}

pub fn now_secs() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64())
}

#[derive(Clone, Debug, Default)]
pub struct Gpu {
    pub util: f64,
    pub mem_used: f64,
    pub mem_total: f64,
    pub temp: f64,
}

pub struct SweepState {
    pub dir: PathBuf,
    pub runs: Vec<RunState>,
    pub gpu: Option<Gpu>,
    pub runner_msg: String,
}

impl SweepState {
    /// The runs of the sweep in `dir`: in `sweep.json` order if the runner
    /// wrote one, else every subdirectory with an `events.jsonl`.
    pub fn open(dir: &Path) -> SweepState {
        let mut names: Vec<String> = std::fs::read_to_string(dir.join("sweep.json"))
            .ok()
            .and_then(|s| serde_json::from_str::<Value>(&s).ok())
            .and_then(|v| {
                v.get("runs").and_then(Value::as_array).map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_str().map(str::to_string))
                        .collect()
                })
            })
            .unwrap_or_default();
        if names.is_empty()
            && let Ok(rd) = std::fs::read_dir(dir)
        {
            names = rd
                .filter_map(Result::ok)
                .filter(|e| e.path().join("events.jsonl").exists())
                .filter_map(|e| e.file_name().to_str().map(str::to_string))
                .collect();
            names.sort();
        }
        SweepState {
            dir: dir.to_path_buf(),
            runs: names
                .iter()
                .map(|n| RunState::new(n, dir.join(n)))
                .collect(),
            gpu: None,
            runner_msg: String::new(),
        }
    }

    pub fn poll(&mut self) {
        for r in &mut self.runs {
            r.poll();
        }
    }
}

/// Append one event (with `t`) to a run's `events.jsonl`. Lines are written
/// whole under a lock, so the runner and the evaluator can share the file.
pub fn append_event(dir: &Path, mut v: Value) {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    if let Some(m) = v.as_object_mut()
        && !m.contains_key("t")
    {
        let t = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0.0, |d| d.as_secs_f64());
        m.insert("t".into(), t.into());
    }
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let line = format!("{v}\n");
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join("events.jsonl"))
    {
        use std::io::Write;
        let _ = file.write_all(line.as_bytes());
    }
}
