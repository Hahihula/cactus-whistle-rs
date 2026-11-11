//! `whistle-train`: LoRA fine-tuning of Cactus Whistle with a dashboard.
//!
//!     whistle-train run SWEEP.toml [--headless] [--only NAME,...] [--force]
//!     whistle-train watch RUNS_DIR
//!     whistle-train example > sweep.toml

mod affinity;
mod config;
mod evaluator;
mod hints;
mod runner;
mod state;
mod train;
mod tui;

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

const USAGE: &str = "\
whistle-train: LoRA fine-tuning of Cactus Whistle

  whistle-train run SWEEP.toml [options]   train the sweep's runs one after another, with the dashboard
      --headless        no dashboard, progress lines on stderr (watch it from elsewhere)
      --only A,B        only these runs
      --force           re-run finished runs (the earlier attempt is kept as NAME.<time>.old)
  whistle-train watch RUNS_DIR             the dashboard for a sweep running elsewhere (--headless)
  whistle-train example                    print an example sweep file

Finished runs are skipped and stopped ones resume, so `run` can be repeated after a crash or a stop.
Training runs on the GPU (burn: CUDA, or wgpu when built with --features wgpu); evaluation runs the
Rust engine with runtime LoRA on the CPU.";

const EXAMPLE: &str = include_str!("../sweep.example.toml");

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("run") if args.len() >= 2 => run(&args[1], &args[2..]),
        Some("watch") if args.len() == 2 => {
            let dir = Path::new(&args[1]);
            if !dir.is_dir() {
                die(&format!("{} is not a directory", dir.display()));
            }
            if let Err(e) = tui::run(state::SweepState::open(dir), None) {
                die(&e.to_string());
            }
        }
        Some("example") => print!("{EXAMPLE}"),
        Some("train") => std::process::exit(train::main(&args[1..])),
        Some("parity") => train::parity::run(&args[1..]).unwrap_or_else(|e| die(&e)),
        Some("-h" | "--help" | "help") => println!("{USAGE}"),
        _ => {
            eprintln!("{USAGE}");
            std::process::exit(2);
        }
    }
}

fn die(msg: &str) -> ! {
    eprintln!("whistle-train: {msg}");
    std::process::exit(1)
}

fn run(path: &str, opts: &[String]) {
    let (mut headless, mut force, mut only) = (false, false, Vec::new());
    let mut it = opts.iter();
    while let Some(o) = it.next() {
        match o.as_str() {
            "--headless" => headless = true,
            "--force" => force = true,
            "--only" => {
                only = it
                    .next()
                    .unwrap_or_else(|| die("--only needs run names"))
                    .split(',')
                    .map(str::to_string)
                    .collect()
            }
            other => die(&format!("unknown option {other}\n\n{USAGE}")),
        }
    }
    let sweep = config::Sweep::load(Path::new(path)).unwrap_or_else(|e| die(&e));
    let names: Vec<String> = sweep.runs().into_iter().map(|r| r.name).collect();
    if let Some(bad) = only.iter().find(|n| !names.contains(n)) {
        die(&format!(
            "no run named {bad}; the sweep has {}",
            names.join(", ")
        ));
    }
    match &sweep.data.template {
        Some(t) if !t.exists() => die(&format!("[data] template {} does not exist", t.display())),
        None => die(
            "[data] template is missing: the published whistle.cact (tokenizer, mel filterbank)",
        ),
        _ => {}
    }
    let ctl = Arc::new(runner::Control::default());

    // SIGINT / SIGTERM: stop the trainer cleanly (it saves its state).
    #[cfg(unix)]
    {
        static CTL: std::sync::OnceLock<Arc<runner::Control>> = std::sync::OnceLock::new();
        let _ = CTL.set(ctl.clone());
        extern "C" fn on_signal(_: libc::c_int) {
            if let Some(c) = CTL.get() {
                c.request_stop();
            }
        }
        // SAFETY: installing a handler that only does atomic stores and kill(2).
        unsafe {
            libc::signal(libc::SIGTERM, on_signal as *const () as libc::sighandler_t);
            if headless {
                libc::signal(libc::SIGINT, on_signal as *const () as libc::sighandler_t);
            }
        }
    }

    let out = sweep.out_dir.clone();
    let c2 = ctl.clone();
    let worker = std::thread::spawn(move || runner::run_sweep(&sweep, &only, force, c2));
    if headless {
        headless_progress(&out, &ctl);
    } else {
        // Wait for sweep.json so the dashboard lists the runs.
        for _ in 0..50 {
            if out.join("sweep.json").exists() {
                break;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        if let Err(e) = tui::run(state::SweepState::open(&out), Some(ctl.clone())) {
            ctl.request_stop();
            eprintln!("whistle-train: dashboard failed: {e}");
        }
        if !ctl.finished.load(Ordering::SeqCst) {
            eprintln!(
                "whistle-train: stopping; the trainer saves its state (run again to resume)..."
            );
        }
    }
    match worker.join() {
        Ok(Ok(())) => eprintln!(
            "whistle-train: {}",
            ctl.msg.lock().map(|m| m.clone()).unwrap_or_default()
        ),
        Ok(Err(e)) => {
            eprintln!("whistle-train: {e}");
            std::process::exit(1)
        }
        Err(_) => std::process::exit(1),
    }
}

fn headless_progress(out: &Path, ctl: &runner::Control) {
    let mut st = state::SweepState::open(out);
    let mut last: std::collections::HashMap<String, String> = Default::default();
    let mut last_msg = String::new();
    while !ctl.finished.load(Ordering::SeqCst) {
        std::thread::sleep(Duration::from_secs(2));
        if st.runs.is_empty() {
            st = state::SweepState::open(out);
        }
        st.poll();
        let msg = ctl.msg.lock().map(|m| m.clone()).unwrap_or_default();
        if msg != last_msg {
            eprintln!("{msg}");
            last_msg = msg;
        }
        for r in &st.runs {
            if r.status == state::Status::Queued {
                continue;
            }
            let line = format!(
                "{} [{}]  step {}/{}  loss {}  best val {}  WER {}{}",
                r.name,
                r.status.name(),
                r.step(),
                r.total_steps(),
                r.steps
                    .last()
                    .map_or("-".into(), |s| format!("{:.4}", s.loss)),
                r.best_eval()
                    .map_or("-".into(), |e| format!("{:.4}", e.val_loss)),
                r.wers
                    .last()
                    .map_or("-".into(), |w| format!("{:.1}%", 100.0 * w.wer)),
                r.test
                    .as_ref()
                    .map_or(String::new(), |w| format!("  test {:.1}%", 100.0 * w.wer)),
            );
            if last.get(&r.name) != Some(&line) {
                eprintln!("{line}");
                last.insert(r.name.clone(), line);
            }
        }
    }
}
