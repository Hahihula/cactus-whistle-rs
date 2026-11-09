//! Plain-language advice from a run's numbers, for people who don't read loss
//! curves every day. Each rule says what it sees and what to try.

use crate::state::{Gpu, RunState, Status, TestState};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    Info,
    Warn,
    Bad,
}

pub struct Hint {
    pub level: Level,
    pub text: String,
}

fn hint(level: Level, text: impl Into<String>) -> Hint {
    Hint {
        level,
        text: text.into(),
    }
}

fn setting(r: &RunState, k: &str) -> Option<f64> {
    r.settings.get(k).and_then(|v| v.as_f64())
}

fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

pub fn hints(r: &RunState, gpu: Option<&Gpu>) -> Vec<Hint> {
    let mut out = Vec::new();

    if let Some(e) = &r.error {
        let first = e.lines().rev().find(|l| !l.trim().is_empty()).unwrap_or(e);
        let mut h = format!("The trainer failed: {first}");
        if e.contains("RESOURCE_EXHAUSTED") || e.contains("out of memory") {
            h += "  The GPU ran out of memory: lower batch_size.";
        } else if e.contains("loss became") {
            h += "  Lower lr (try half), or keep grad_clip at 1.0.";
        }
        out.push(hint(Level::Bad, h));
    }
    match r.status {
        Status::Stalled => out.push(hint(
            Level::Bad,
            "The trainer stopped printing and was killed. If the GPU is gone (nvidia-smi fails), reboot before the next run.",
        )),
        Status::Stopped => out.push(hint(Level::Info, "Stopped with its state saved: run the sweep again to resume it.")),
        _ => {}
    }

    // What is still running comes first.
    if r.status == Status::Done {
        let n = r.checkpoints_pending();
        if n > 0 {
            out.push(hint(
                Level::Info,
                format!(
                    "Training is finished, but the evaluator is still scoring {n} checkpoint(s) on the CPU, so WER and samples keep arriving. \
                     It runs slower than training when checkpoints are frequent (fewer evals_per_epoch or [eval] clips makes it keep up)."
                ),
            ));
        }
        match r.test_state {
            TestState::Queued => out.push(hint(
                Level::Info,
                "After the checkpoints, the best adapter will be scored on the whole test split; its WER appears in the test column.",
            )),
            TestState::Running => out.push(hint(
                Level::Info,
                "Training is finished; the best adapter is being scored on the whole test split on the CPU. Its WER appears in the test column when done.",
            )),
            TestState::Failed => out.push(hint(Level::Warn, "Scoring the test split failed; see the Log tab.")),
            _ => {}
        }
    }

    if let Some(s) = &r.start {
        let ns: Vec<&str> = s["not_single"]
            .as_array()
            .map(|a| a.iter().filter_map(|x| x.as_str()).collect())
            .unwrap_or_default();
        if !ns.is_empty() {
            out.push(hint(
                Level::Warn,
                format!("These tokens did not become single tokens: {}. They still work, split into pieces.", ns.join(" ")),
            ));
        }
        let (n, found) = (
            s["tokens"].as_f64().unwrap_or(0.0),
            s["tokens_found"].as_f64().unwrap_or(0.0),
        );
        if n > found {
            out.push(hint(
                Level::Info,
                format!(
                    "{} of the {} added tokens never appear in the training transcripts, so their embeddings stay untrained.",
                    n - found,
                    n
                ),
            ));
        }
        let skipped = s["skipped"].as_f64().unwrap_or(0.0) + s["too_long"].as_f64().unwrap_or(0.0);
        if skipped > 0.0 {
            out.push(hint(
                Level::Info,
                format!("{skipped} clips were skipped (empty, over 30 s, or over 320 tokens)."),
            ));
        }
        let new_langs = s["new_languages"].as_array().map_or(0, Vec::len);
        if new_langs > 0 && setting(r, "lr").is_some_and(|lr| lr < 1e-3) {
            out.push(hint(
                Level::Info,
                "This run adds a language. Cactus's guide notes a new language needs more steps or a higher learning rate than adapting a known one (their example used 1e-2).",
            ));
        }
    }

    // Overfitting: validation rising after its best while training falls.
    if let Some(best) = r.best_eval()
        && let Some(last) = r.evals.last()
        && last.step > best.step
        && last.val_loss > best.val_loss * 1.02
    {
        let after = r.evals.iter().filter(|e| e.step > best.step).count();
        out.push(hint(
            if after >= 2 { Level::Warn } else { Level::Info },
            format!(
                "Validation loss is {:.1}% above its best ({:.4} at step {}) for {} evaluation(s): the model is starting to memorise the training set. \
                 adapter_best.safetensors keeps the best one. Next time try fewer epochs, more weight_decay, or lora_dropout around 0.1.",
                100.0 * (last.val_loss / best.val_loss - 1.0),
                best.val_loss,
                best.step,
                after
            ),
        ));
    }
    if let (Some(last_step), Some(last_eval)) = (r.steps.last(), r.evals.last())
        && last_eval.val_loss > 0.0
        && last_step.loss < 0.5 * last_eval.val_loss
        && r.evals.len() >= 2
    {
        out.push(hint(
            Level::Info,
            format!(
                "Training loss ({:.3}) is well below validation loss ({:.3}): a gap this size means the adapter fits the training clips much better than new ones.",
                last_step.loss, last_eval.val_loss
            ),
        ));
    }
    if let Some(step) = r.early_stop {
        out.push(hint(
            Level::Info,
            format!("Early stopping ended the run at step {step}: validation stopped improving."),
        ));
    }

    // Instability.
    if r.steps.len() >= 8 {
        let recent = &r.steps[r.steps.len() - 8..];
        let mut losses: Vec<f64> = recent.iter().map(|p| p.loss).collect();
        let med = median(&mut losses);
        let spikes = recent
            .iter()
            .filter(|p| p.loss_max > 3.0 * med && med > 0.0)
            .count();
        if spikes >= 3 {
            out.push(hint(
                Level::Warn,
                "The loss spikes often (single batches at 3x the typical loss). Some spikes are hard clips; frequent ones suggest the learning rate is too high.",
            ));
        }
        let clipped = recent.iter().map(|p| p.clipped).sum::<f64>() / recent.len() as f64;
        if clipped > 0.5 && setting(r, "grad_clip").unwrap_or(1.0) > 0.0 {
            out.push(hint(
                Level::Warn,
                format!(
                    "Gradients were clipped on {:.0}% of recent steps: the updates want to be bigger than grad_clip allows, often a sign lr is high.",
                    100.0 * clipped
                ),
            ));
        }
        let first = r.steps.iter().skip(1).take(4).map(|p| p.loss).sum::<f64>() / 4.0;
        let now = recent.iter().map(|p| p.loss).sum::<f64>() / recent.len() as f64;
        let progress = r.step() / r.total_steps().max(1.0);
        if progress > 0.3 && now > 0.9 * first {
            out.push(hint(
                Level::Warn,
                "The training loss has barely moved. Check the transcripts match the audio, or raise lr.",
            ));
        }
    }

    // The engine's WER against the loss.
    if r.wers.len() >= 3 && r.evals.len() >= 3 {
        let w: Vec<f64> = r.wers.iter().rev().take(3).map(|x| x.wer).collect();
        let v: Vec<f64> = r.evals.iter().rev().take(3).map(|x| x.val_loss).collect();
        if w[0] > w[2] * 1.03 && v[0] < v[2] {
            out.push(hint(
                Level::Warn,
                "Validation loss still falls but the engine's WER rises: the model grows more confident on the reference text without transcribing better. \
                 Listen to the samples before trusting the loss.",
            ));
        }
    }
    if let Some((_, items)) = r.samples.last() {
        let detected: Vec<&str> = items.iter().map(|s| s.lang.as_str()).collect();
        if let Some(l) = r.settings.get("language").and_then(|v| v.as_str())
            && !detected.is_empty()
            && detected.iter().any(|d| *d != l)
        {
            out.push(hint(
                Level::Info,
                format!(
                    "Some samples are detected as another language than {l} ({}). Early in training that's normal; late, the adapter may need more data or steps.",
                    detected.join(", ")
                ),
            ));
        }
    }

    if let Some(g) = gpu
        && g.mem_total > 0.0
        && g.mem_used / g.mem_total > 0.95
        && r.status == Status::Running
    {
        out.push(hint(
            Level::Info,
            "GPU memory is nearly full; a bigger batch_size would not fit.",
        ));
    }
    out.sort_by_key(|h| std::cmp::Reverse(h.level));
    out
}
