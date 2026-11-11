//! Runs a sweep: one trainer process (`whistle-train train`) at a time on the GPU, its JSON
//! events appended to `<run>/events.jsonl`, checkpoints handed to the
//! evaluator. A run whose trainer stays silent for `stall_minutes` is killed;
//! the GPU is checked before every run (a CUDA crash can leave it unusable).

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::mpsc::{RecvTimeoutError, channel};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use crate::config::{Run, Sweep};
use crate::evaluator::{Evaluator, Job};
use crate::state::append_event;

/// Shared with the TUI: ask the runner to stop, see what it is doing.
#[derive(Default)]
pub struct Control {
    pub stop: AtomicBool,
    pub child: AtomicI32,
    pub finished: AtomicBool,
    pub msg: std::sync::Mutex<String>,
}

impl Control {
    pub fn say(&self, s: impl Into<String>) {
        let s = s.into();
        if let Ok(mut m) = self.msg.lock() {
            *m = s;
        }
    }

    /// SIGTERM the trainer (it saves its state and exits) and stop the sweep.
    pub fn request_stop(&self) {
        self.stop.store(true, Ordering::SeqCst);
        terminate(self.child.load(Ordering::SeqCst));
    }
}

/// Ask process `pid` to stop: SIGTERM, which the trainer answers by saving
/// its state. (Elsewhere the runner kills the process once it sees `stop`.)
fn terminate(pid: i32) {
    #[cfg(unix)]
    if pid > 0 {
        // SAFETY: plain kill(2) on our own child's pid.
        unsafe { libc::kill(pid, libc::SIGTERM) };
    }
    #[cfg(not(unix))]
    let _ = pid;
}

fn gpu_ok() -> bool {
    Command::new("nvidia-smi")
        .arg("-L")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

/// Did an earlier attempt at this run finish?
fn finished(dir: &Path) -> bool {
    std::fs::read_to_string(dir.join("events.jsonl")).is_ok_and(|s| {
        s.lines().any(|l| {
            serde_json::from_str::<Value>(l)
                .is_ok_and(|v| v["event"] == "status" && v["status"] == "done")
        })
    })
}

/// Make the 4-bit base model the evaluator runs adapters on, if missing.
pub fn ensure_eval_base(sweep: &Sweep, ctl: &Control) -> Result<(), String> {
    let out = &sweep.eval.base_cact;
    if out.exists() {
        return Ok(());
    }
    ctl.say(format!(
        "building the 4-bit evaluation base {}",
        out.display()
    ));
    sweep.build_cact(None, out)
}

pub fn run_sweep(
    sweep: &Sweep,
    only: &[String],
    force: bool,
    ctl: Arc<Control>,
) -> Result<(), String> {
    let result = run_inner(sweep, only, force, &ctl);
    match &result {
        Ok(()) => ctl.say("sweep finished"),
        Err(e) => ctl.say(format!("sweep stopped: {e}")),
    }
    ctl.finished.store(true, Ordering::SeqCst);
    result
}

fn run_inner(
    sweep: &Sweep,
    only: &[String],
    force: bool,
    ctl: &Arc<Control>,
) -> Result<(), String> {
    let out = &sweep.out_dir;
    std::fs::create_dir_all(out).map_err(|e| format!("{}: {e}", out.display()))?;
    let runs: Vec<Run> = sweep
        .runs()
        .into_iter()
        .filter(|r| only.is_empty() || only.contains(&r.name))
        .collect();
    let names: Vec<&str> = runs.iter().map(|r| r.name.as_str()).collect();
    std::fs::write(out.join("sweep.json"), json!({"runs": names}).to_string())
        .map_err(|e| e.to_string())?;
    for r in &runs {
        let dir = out.join(&r.name);
        std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        if !dir.join("events.jsonl").exists() {
            append_event(&dir, json!({"event": "status", "status": "queued"}));
        }
    }
    ensure_eval_base(sweep, ctl)?;
    ctl.say("loading evaluation clips");
    let eval = Evaluator::start(sweep)?;

    let mut failure = None;
    for run in &runs {
        if ctl.stop.load(Ordering::SeqCst) {
            break;
        }
        let dir = out.join(&run.name);
        if finished(&dir) && !force {
            continue;
        }
        if force && dir.join("events.jsonl").exists() {
            // Keep the earlier attempt next to the new one.
            let stamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_secs());
            let old = out.join(format!("{}.{stamp}.old", run.name));
            std::fs::rename(&dir, &old).map_err(|e| e.to_string())?;
            std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        }
        if !gpu_ok() {
            append_event(
                &dir,
                json!({"event": "status", "status": "failed", "msg": "nvidia-smi cannot see the GPU"}),
            );
            failure = Some("nvidia-smi cannot see the GPU".to_string());
            break;
        }
        match run_one(sweep, run, &dir, &eval, ctl) {
            Ok(()) => {}
            Err(e) if ctl.stop.load(Ordering::SeqCst) => {
                failure = Some(e);
                break;
            }
            Err(e) => {
                if !sweep.keep_going {
                    failure = Some(e);
                    break;
                }
            }
        }
    }
    if !ctl.stop.load(Ordering::SeqCst) {
        ctl.say("finishing evaluations");
    }
    eval.finish();
    failure.map_or(Ok(()), Err)
}

fn run_one(
    sweep: &Sweep,
    run: &Run,
    dir: &Path,
    eval: &Evaluator,
    ctl: &Arc<Control>,
) -> Result<(), String> {
    let resume = dir.join("state.safetensors").exists();
    let config = sweep.trainer_config(run);
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_string_pretty(&config).unwrap_or_default(),
    )
    .map_err(|e| e.to_string())?;
    append_event(
        dir,
        json!({"event": "status", "status": "running", "resume": resume, "evaluates": eval.checkpoints,
               "msg": if resume { "resuming from the saved state" } else { "starting" }}),
    );
    ctl.say(format!("training {}", run.name));

    // The trainer is this program (`whistle-train train`), in a process of
    // its own: a CUDA failure can't take the dashboard and evaluator with it.
    let exe =
        std::env::current_exe().map_err(|e| format!("cannot find whistle-train itself: {e}"))?;
    let mut cmd = Command::new(&exe);
    cmd.arg("train").arg(dir.join("config.json"));
    if resume {
        cmd.arg("--resume");
    }
    let mut child = cmd
        .current_dir(dir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("{}: {e}", exe.display()))?;
    ctl.child.store(child.id() as i32, Ordering::SeqCst);

    let (tx, rx) = channel::<String>();
    let stdout = child.stdout.take().expect("piped");
    let tx2 = tx.clone();
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if tx2.send(line).is_err() {
                break;
            }
        }
    });
    let stderr = child.stderr.take().expect("piped");
    let dir2: PathBuf = dir.to_path_buf();
    std::thread::spawn(move || {
        let mut log = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(dir2.join("stderr.txt"))
            .ok();
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            if let Some(f) = log.as_mut() {
                use std::io::Write;
                let _ = writeln!(f, "{line}");
            }
            if !line.trim().is_empty() {
                append_event(&dir2, json!({"event": "stderr", "msg": line}));
            }
        }
    });
    drop(tx);

    let stall = Duration::from_secs_f64(sweep.stall_minutes * 60.0);
    let mut last = Instant::now();
    let mut stalled = false;
    let mut done = false;
    loop {
        match rx.recv_timeout(Duration::from_secs(1)) {
            Ok(line) => {
                last = Instant::now();
                let Ok(v) = serde_json::from_str::<Value>(&line) else {
                    append_event(dir, json!({"event": "log", "msg": line}));
                    continue;
                };
                if v["event"] == "eval"
                    && let (Some(step), Some(ckpt)) = (v["step"].as_u64(), v["checkpoint"].as_str())
                {
                    eval.submit(Job::Checkpoint {
                        run_dir: dir.to_path_buf(),
                        step,
                        adapter: ckpt.into(),
                    });
                }
                done |= v["event"] == "done";
                append_event(dir, v);
            }
            Err(RecvTimeoutError::Timeout) => {
                #[cfg(not(unix))]
                if ctl.stop.load(Ordering::SeqCst) {
                    let _ = child.kill();
                }
                if last.elapsed() > stall && !stalled {
                    stalled = true;
                    append_event(
                        dir,
                        json!({"event": "log", "msg": format!("no output for {} min, killing the trainer", sweep.stall_minutes)}),
                    );
                    terminate(child.id() as i32);
                    let t0 = Instant::now();
                    while t0.elapsed() < Duration::from_secs(60) {
                        if child.try_wait().ok().flatten().is_some() {
                            break;
                        }
                        std::thread::sleep(Duration::from_millis(500));
                    }
                    let _ = child.kill();
                }
            }
            Err(RecvTimeoutError::Disconnected) => break,
        }
    }
    let status = child.wait().map_err(|e| e.to_string())?;
    ctl.child.store(0, Ordering::SeqCst);
    let code = status.code();
    let (st, msg) = if stalled {
        (
            "stalled",
            format!("silent for {} min, killed", sweep.stall_minutes),
        )
    } else if code == Some(3) || (ctl.stop.load(Ordering::SeqCst) && !done) {
        ("stopped", "state saved; run again to resume".to_string())
    } else if code == Some(0) && done {
        ("done", String::new())
    } else {
        ("failed", format!("trainer exited with {status}"))
    };
    if st == "done" {
        let best = dir.join("adapter_best.safetensors");
        if eval.tests {
            append_event(dir, json!({"event": "test_queued"}));
            eval.submit(Job::Test {
                run_dir: dir.to_path_buf(),
                adapter: best.clone(),
            });
        }
        if sweep.build {
            ctl.say(format!("building tuned_{}.cact", run.name));
            let cact = dir.join(format!("tuned_{}.cact", run.name));
            match sweep.build_cact(Some(&best), &cact) {
                Ok(()) => append_event(
                    dir,
                    json!({"event": "built", "path": cact.display().to_string()}),
                ),
                Err(e) => append_event(
                    dir,
                    json!({"event": "log", "msg": format!("building {} failed: {e}", cact.display())}),
                ),
            }
        }
    }
    append_event(dir, json!({"event": "status", "status": st, "msg": msg}));
    if st == "done" {
        Ok(())
    } else {
        Err(format!("{}: {st} {msg}", run.name))
    }
}
