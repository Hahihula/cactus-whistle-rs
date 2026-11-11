//! The LoRA trainer (burn), run by `whistle-train train CONFIG.json [--resume]`
//! as a child process of the sweep runner. A port of the earlier JAX trainer,
//! itself a re-implementation of `needle whistle finetune` (cactus-needle
//! 3.2.8) with the knobs the CLI doesn't expose:
//!
//!   optimiser   AdamW (weight decay separately for the adapter and the new
//!               vocabulary rows), a learning-rate multiplier for the rows,
//!               warmup, cosine / linear / constant schedule and its floor,
//!               gradient clipping
//!   adapter     rank, alpha, rank dropout
//!   evaluation  validation N times per epoch, best checkpoint kept, early stop
//!   safety      resumable state saved with every checkpoint; SIGTERM saves and
//!               stops (exit code 3)
//!   data        speed perturbation, SpecAugment, background noise
//!
//! With the defaults it trains like `needle whistle finetune`: AdamW (weight
//! decay 1e-4), 5% warmup, cosine decay to 0, clip 1.0, the 15 projection
//! groups, CQ-W4 straight-through weights and A8 activations.
//!
//! One JSON object per line on stdout:
//!
//!   {"event": "log", "msg"}                       progress while loading
//!   {"event": "start", ...}                       data, vocabulary, schedule
//!   {"event": "step", "step", "epoch", "loss", "loss_max", "lr", "grad_norm", "clipped", "secs"}
//!   {"event": "eval", "step", "epoch", "val_loss", "best", "checkpoint"}
//!   {"event": "early_stop", "step", "patience"}
//!   {"event": "stopped", "step"}                  after SIGTERM, state saved
//!   {"event": "done", "best_step", "best_val", "adapter", "steps"}
//!   {"event": "error", "msg"}
//!
//! Files in out_dir: checkpoints/step_NNNNNN.safetensors (one per evaluation),
//! adapter_best.safetensors, adapter_last.safetensors, state.safetensors.

pub mod data;
pub mod features;
pub mod model;
pub mod parity;
pub mod quant;

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::Instant;

use burn::tensor::{Device, Tensor, TensorData};
use serde::Deserialize;
use serde_json::{Value, json};

use cactus_whistle_common::checkpoint::Checkpoint;
use cactus_whistle_common::export::{lora_param, template};
use cactus_whistle_common::spm::{Encoder, token_pieces};

use data::{Augment, Batch, Clip, Noise, Rng};
use features::Frontend;
use model::{GROUPS, Train, Whistle};

fn emit(event: &str, mut fields: Value) {
    let m = fields.as_object_mut().expect("object");
    m.insert("event".into(), event.into());
    println!("{fields}");
}

fn log(msg: impl Into<String>) {
    emit("log", json!({"msg": msg.into()}));
}

#[derive(Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub train: String,
    pub validation: String,
    pub out_dir: String,
    pub base: String,
    /// A published `.cact`: tokenizer, mel filterbank, Hadamard permutations.
    pub template: String,
    pub tokens: String,
    pub language: String,
    pub epochs: usize,
    pub batch_size: usize,
    pub lr: f64,
    pub warmup: f64,
    pub schedule: String,
    pub min_lr_ratio: f64,
    pub weight_decay: f64,
    pub rows_weight_decay: f64,
    pub rows_lr_mult: f64,
    pub grad_clip: f64,
    pub lora_rank: usize,
    pub lora_alpha: f64,
    pub lora_dropout: f64,
    pub seed: u64,
    pub evals_per_epoch: usize,
    pub early_stop: usize,
    pub log_every: usize,
    pub limit_train: usize,
    pub limit_validation: usize,
    pub max_steps: usize,
    pub load_threads: usize,
    pub augment: Augment,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            train: String::new(),
            validation: String::new(),
            out_dir: String::new(),
            base: "checkpoints/whistle.safetensors".into(),
            template: String::new(),
            tokens: String::new(),
            language: String::new(),
            epochs: 3,
            batch_size: 16,
            lr: 1e-4,
            warmup: 0.05,
            schedule: "cosine".into(),
            min_lr_ratio: 0.0,
            weight_decay: 1e-4,
            rows_weight_decay: 1e-4,
            rows_lr_mult: 1.0,
            grad_clip: 1.0,
            lora_rank: 16,
            lora_alpha: 32.0,
            lora_dropout: 0.0,
            seed: 0,
            evals_per_epoch: 1,
            early_stop: 0,
            log_every: 0,
            limit_train: 0,
            limit_validation: 0,
            max_steps: 0,
            load_threads: 8,
            augment: Augment::default(),
        }
    }
}

/// Learning rate at optimiser step `i` (0-based), as optax's schedules.
pub struct Schedule {
    lr: f64,
    floor: f64,
    warmup: usize,
    total: usize,
    kind: String,
}

impl Schedule {
    pub fn new(cfg: &Config, total: usize) -> Result<Schedule, String> {
        let warmup = if cfg.warmup > 0.0 {
            ((cfg.warmup * total as f64).round() as usize)
                .max(1)
                .min(total.saturating_sub(1))
        } else {
            0
        };
        if !["cosine", "linear", "constant"].contains(&cfg.schedule.as_str()) {
            return Err(format!(
                "schedule must be cosine, linear or constant, not {}",
                cfg.schedule
            ));
        }
        Ok(Schedule {
            lr: cfg.lr,
            floor: cfg.lr * cfg.min_lr_ratio,
            warmup,
            total,
            kind: cfg.schedule.clone(),
        })
    }

    pub fn at(&self, i: usize) -> f64 {
        if i < self.warmup {
            return self.lr * i as f64 / self.warmup as f64;
        }
        let span = self.total.saturating_sub(self.warmup).max(1) as f64;
        let p = ((i - self.warmup) as f64 / span).min(1.0);
        match self.kind.as_str() {
            "cosine" => {
                self.floor + (self.lr - self.floor) * 0.5 * (1.0 + (std::f64::consts::PI * p).cos())
            }
            "linear" => self.lr + (self.floor - self.lr) * p,
            _ => self.lr,
        }
    }
}

static STOP: AtomicBool = AtomicBool::new(false);

#[cfg(unix)]
extern "C" fn on_term(_: libc::c_int) {
    STOP.store(true, Ordering::SeqCst);
}

/// SIGTERM (the runner) and SIGINT (Ctrl+C reaches the whole process group
/// in a terminal) both mean: save the state and stop.
fn install_sigterm() {
    #[cfg(unix)]
    // SAFETY: the handler only stores to an atomic.
    unsafe {
        libc::signal(libc::SIGTERM, on_term as *const () as libc::sighandler_t);
        libc::signal(libc::SIGINT, on_term as *const () as libc::sighandler_t);
    }
}

// --------------------------------------------------------------- tensors

fn to_vec<const D: usize>(t: &Tensor<D>) -> Vec<f32> {
    t.clone()
        .without_autodiff()
        .into_data()
        .convert::<f32>()
        .try_to_vec::<f32>()
        .expect("f32 data")
}

fn tensor<const D: usize>(v: Vec<f32>, shape: [usize; D], d: &Device) -> Tensor<D> {
    Tensor::from_data(TensorData::new(v, shape), d)
}

/// AdamW state of one tensor.
struct Moments<const D: usize> {
    m: Tensor<D>,
    v: Tensor<D>,
}

impl<const D: usize> Moments<D> {
    fn zeros(like: &Tensor<D>) -> Self {
        let z = like.clone().without_autodiff().zeros_like();
        Moments { m: z.clone(), v: z }
    }
}

/// optax.adamw: b1 0.9, b2 0.999, eps 1e-8, decoupled weight decay times lr.
fn adamw<const D: usize>(
    p: &Tensor<D>,
    g: Tensor<D>,
    st: &mut Moments<D>,
    t: usize,
    lr: f64,
    wd: f64,
) -> Tensor<D> {
    let (b1, b2, eps) = (0.9f64, 0.999f64, 1e-8f64);
    st.m = st.m.clone().mul_scalar(b1) + g.clone().mul_scalar(1.0 - b1);
    st.v = st.v.clone().mul_scalar(b2) + g.square().mul_scalar(1.0 - b2);
    let mhat = st.m.clone().div_scalar(1.0 - b1.powi(t as i32));
    let vhat = st.v.clone().div_scalar(1.0 - b2.powi(t as i32));
    let p0 = p.clone().without_autodiff();
    let upd = mhat / vhat.sqrt().add_scalar(eps) + p0.clone().mul_scalar(wd);
    (p0 - upd.mul_scalar(lr)).autodiff().require_grad()
}

// --------------------------------------------------------------- files

fn json_meta(m: &[(&str, Value)]) -> HashMap<String, String> {
    m.iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

fn write_safetensors(
    path: &Path,
    tensors: Vec<(String, Vec<usize>, Vec<f32>)>,
    meta: HashMap<String, String>,
) -> Result<(), String> {
    let bytes: Vec<(String, Vec<usize>, Vec<u8>)> = tensors
        .into_iter()
        .map(|(n, s, v)| (n, s, v.iter().flat_map(|x| x.to_le_bytes()).collect()))
        .collect();
    let views: Vec<(String, safetensors::tensor::TensorView)> = bytes
        .iter()
        .map(|(n, s, b)| {
            (
                n.clone(),
                safetensors::tensor::TensorView::new(safetensors::Dtype::F32, s.clone(), b)
                    .unwrap(),
            )
        })
        .collect();
    let tmp = path.with_extension("tmp");
    safetensors::serialize_to_file(views, Some(meta), &tmp)
        .map_err(|e| format!("{}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| format!("{}: {e}", path.display()))
}

/// Named f32 tensors (shape, values) and the string metadata of a file.
type Tensors = (
    BTreeMap<String, (Vec<usize>, Vec<f32>)>,
    HashMap<String, String>,
);

fn read_safetensors(path: &Path) -> Result<Tensors, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let (_, h) = safetensors::SafeTensors::read_metadata(&bytes).map_err(|e| e.to_string())?;
    let meta = h.metadata().clone().unwrap_or_default();
    let st = safetensors::SafeTensors::deserialize(&bytes).map_err(|e| e.to_string())?;
    let mut out = BTreeMap::new();
    for (n, t) in st.tensors() {
        let v = t
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        out.insert(n, (t.shape().to_vec(), v));
    }
    Ok((out, meta))
}

// --------------------------------------------------------------- main

pub fn main(args: &[String]) -> i32 {
    install_sigterm();
    // Kernel dispatch is single-threaded CPU work: keep it on the P-cores.
    crate::affinity::performance_cores();
    let Some(path) = args.first() else {
        eprintln!("usage: whistle-train train CONFIG.json [--resume]");
        return 2;
    };
    let resume = args[1..].iter().any(|a| a == "--resume");
    let res = std::panic::catch_unwind(|| run(path, resume));
    match res {
        Ok(Ok(code)) => code,
        Ok(Err(e)) => {
            emit("error", json!({"msg": e}));
            1
        }
        Err(p) => {
            let msg = p
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "panic".into());
            emit("error", json!({"msg": format!("trainer crashed: {msg}")}));
            1
        }
    }
}

struct Data {
    clips: Vec<Clip>,
    skipped: usize,
    too_long: usize,
    minutes: f64,
    found: usize,
}

#[allow(clippy::too_many_arguments)]
fn load_split(
    path: &str,
    cfg: &Config,
    limit: usize,
    enc: &Encoder,
    languages: &[String],
    text: usize,
    max_tokens: usize,
    tokens: &[String],
) -> Result<Data, String> {
    let rows = data::read_rows(Path::new(path), &cfg.language, limit)?;
    let name = Path::new(path)
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    let n = rows.len();
    let t0 = Instant::now();
    let audio = data::read_audio(&rows, cfg.load_threads, &|d| {
        log(format!(
            "{name}: {d}/{n} clips read ({:.0} s)",
            t0.elapsed().as_secs_f64()
        ))
    })?;
    let mut d = Data {
        clips: Vec::new(),
        skipped: 0,
        too_long: 0,
        minutes: 0.0,
        found: 0,
    };
    let mut seen = vec![false; tokens.len()];
    for ((_, txt, lang), a) in rows.iter().zip(audio) {
        let a = a.unwrap_or_default();
        if txt.is_empty() || a.is_empty() || a.len() > data::MAX_SECONDS * data::RATE {
            d.skipped += 1;
            continue;
        }
        let li = languages
            .iter()
            .position(|l| l == lang)
            .ok_or_else(|| format!("no training clips for {lang}"))?;
        let ids = data::encode(enc, 2, (text + li) as u32, 1, txt);
        if ids.len() > max_tokens + 1 {
            d.too_long += 1;
            continue;
        }
        for (k, t) in tokens.iter().enumerate() {
            if !seen[k] && txt.contains(t.as_str()) {
                seen[k] = true;
            }
        }
        d.minutes += a.len() as f64 / data::RATE as f64 / 60.0;
        d.clips.push(Clip { samples: a, ids });
    }
    d.found = seen.iter().filter(|&&s| s).count();
    Ok(d)
}

fn run(path: &str, resume: bool) -> Result<i32, String> {
    let raw = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let cfg: Config = serde_json::from_str(&raw).map_err(|e| format!("{path}: {e}"))?;
    if cfg.template.is_empty() {
        return Err("no template .cact ([data] template in the sweep file)".into());
    }
    if cfg.epochs == 0 || cfg.batch_size == 0 || cfg.lora_rank == 0 {
        return Err("epochs, batch_size and lora_rank must be at least 1".into());
    }
    let out = PathBuf::from(&cfg.out_dir);
    std::fs::create_dir_all(out.join("checkpoints"))
        .map_err(|e| format!("{}: {e}", out.display()))?;

    log(format!("loading base {}", cfg.base));
    let ck = Checkpoint::load(&cfg.base)?;
    let mcfg = ck.config.clone();
    let tblob = std::fs::read(&cfg.template).map_err(|e| format!("{}: {e}", cfg.template))?;
    let tc =
        cactus_whistle::cact::Cact::open(&tblob).map_err(|e| format!("{}: {e}", cfg.template))?;
    let tpl = template(&tc)?;
    let base_enc = Encoder::new(tpl.tokenizer, mcfg.text)?;
    if base_enc.len() != mcfg.text {
        return Err("the template's tokenizer doesn't match the checkpoint".into());
    }

    // Vocabulary: added pieces from the tokens file, languages from the clips.
    let tokens: Vec<String> = if cfg.tokens.is_empty() {
        Vec::new()
    } else {
        let t = std::fs::read_to_string(&cfg.tokens).map_err(|e| format!("{}: {e}", cfg.tokens))?;
        let t: Vec<String> = t
            .trim_start_matches('\u{feff}')
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .collect();
        if let Some(bad) = t.iter().find(|t| t.starts_with("<|") && t.ends_with("|>")) {
            return Err(format!(
                "{}: {bad} looks like a language tag; a new language gets its tag from the clips' \"language\" code",
                cfg.tokens
            ));
        }
        t
    };
    let pieces = token_pieces(&base_enc, &tokens);
    let train_rows = data::read_rows(Path::new(&cfg.train), &cfg.language, cfg.limit_train)?;
    let val_rows = data::read_rows(
        Path::new(&cfg.validation),
        &cfg.language,
        cfg.limit_validation,
    )?;
    let mut languages = mcfg.languages.clone();
    let mut new: Vec<String> = train_rows
        .iter()
        .map(|r| r.2.clone())
        .filter(|l| !languages.contains(l))
        .collect();
    new.sort();
    new.dedup();
    languages.extend(new.iter().cloned());
    if let Some(l) = val_rows
        .iter()
        .map(|r| &r.2)
        .find(|l| !languages.contains(l))
    {
        return Err(format!("no training clips for {l}"));
    }
    let pr: Vec<&str> = pieces.iter().map(String::as_str).collect();
    let nl: Vec<&str> = new.iter().map(String::as_str).collect();
    let ext = cactus_whistle::tokenizer::extend(tpl.tokenizer, mcfg.text, &pr, &nl)
        .ok_or("cannot extend the tokenizer")?;
    let text = mcfg.text + pieces.len();
    let enc = Encoder::new(&ext, text)?;
    let not_single: Vec<&String> = tokens
        .iter()
        .filter(|t| {
            [t.to_string(), format!(" {t}")]
                .iter()
                .any(|x| enc.encode(x).len() != 1)
        })
        .collect();

    log("reading clips");
    let tr = load_split(
        &cfg.train,
        &cfg,
        cfg.limit_train,
        &enc,
        &languages,
        text,
        mcfg.max_tokens,
        &tokens,
    )?;
    let va = load_split(
        &cfg.validation,
        &cfg,
        cfg.limit_validation,
        &enc,
        &languages,
        text,
        mcfg.max_tokens,
        &[],
    )?;
    if tr.clips.is_empty() || va.clips.is_empty() {
        return Err("no usable clips in the train or validation split".into());
    }

    // Frozen weights and everything derived from them live on the plain
    // device; only the trained values carry autodiff (mixed operations treat
    // plain tensors as constants), so evaluation runs without a graph.
    let device = Device::default();
    let adev = device.clone().autodiff();
    let backend = format!("{device:?}");
    log(format!("loading the model on {backend}"));
    let (p1, p2) = tpl.perms();
    let model = Whistle::load(&ck, (&p1, &p2), pieces.len(), new.len(), &device)?;
    let fe = Frontend::new(tpl.mel());

    // Trainable values: A ~ N(0, 1) / rank, B = 0, rows from the base embedding.
    let rank = cfg.lora_rank;
    let scale = (cfg.lora_alpha / rank as f64) as f32;
    let mut init = Rng::stream(cfg.seed, 0x1A2B);
    let mut a = Vec::new();
    let mut b = Vec::new();
    for &(l, i, o) in &model.lora_dims {
        let av: Vec<f32> = (0..l * i * rank)
            .map(|_| (init.normal() / rank as f64) as f32)
            .collect();
        a.push(tensor(av, [l, i, rank], &adev).require_grad());
        b.push(Tensor::<3>::zeros([l, rank, o], &adev).require_grad());
    }
    let d = mcfg.dim;
    let emb = ck.get("embedding/embedding")?;
    let mut rows = Vec::with_capacity((pieces.len() + new.len()) * d);
    for p in &pieces {
        let ids = base_enc.encode(&p.replace('\u{2581}', " "));
        let mut r = vec![0f32; d];
        for &id in &ids {
            for (x, v) in r
                .iter_mut()
                .zip(&emb.data[id as usize * d..(id as usize + 1) * d])
            {
                *x += v / ids.len() as f32;
            }
        }
        rows.extend(r);
    }
    let n_tags = mcfg.languages.len();
    let mut tag = vec![0f32; d];
    for k in 0..n_tags {
        for (x, v) in tag
            .iter_mut()
            .zip(&emb.data[(mcfg.text + k) * d..(mcfg.text + k + 1) * d])
        {
            *x += v / n_tags as f32;
        }
    }
    for _ in &new {
        rows.extend(&tag);
    }
    let n_rows = pieces.len() + new.len();
    let mut train = Train {
        a,
        b,
        rows: tensor(rows, [n_rows, d], &adev).require_grad(),
    };
    let mut mom_a: Vec<Moments<3>> = train.a.iter().map(Moments::zeros).collect();
    let mut mom_b: Vec<Moments<3>> = train.b.iter().map(Moments::zeros).collect();
    let mut mom_r = Moments::zeros(&train.rows);

    let bs = cfg.batch_size;
    let count = tr.clips.len();
    let spe = count.div_ceil(bs);
    let mut total = cfg.epochs * spe;
    if cfg.max_steps > 0 {
        total = total.min(cfg.max_steps);
    }
    let sched = Schedule::new(&cfg, total)?;

    // Resume.
    let state_path = out.join("state.safetensors");
    let mut step = 0usize;
    let mut best_val = f64::INFINITY;
    let mut best_step = 0usize;
    let mut bad = 0usize;
    let mut evals: Vec<Value> = Vec::new();
    if resume && state_path.exists() {
        let (t, meta) = read_safetensors(&state_path)?;
        let get = |n: &str| t.get(n).cloned().ok_or_else(|| format!("state has no {n}"));
        for g in 0..GROUPS.len() {
            let (sa, va_) = get(&format!("a/{g}"))?;
            let (sb, vb) = get(&format!("b/{g}"))?;
            let sh3 = |s: &[usize]| [s[0], s[1], s[2]];
            train.a[g] = tensor(va_, sh3(&sa), &adev).require_grad();
            train.b[g] = tensor(vb, sh3(&sb), &adev).require_grad();
            mom_a[g] = Moments {
                m: tensor(get(&format!("ma/{g}"))?.1, sh3(&sa), &device),
                v: tensor(get(&format!("va/{g}"))?.1, sh3(&sa), &device),
            };
            mom_b[g] = Moments {
                m: tensor(get(&format!("mb/{g}"))?.1, sh3(&sb), &device),
                v: tensor(get(&format!("vb/{g}"))?.1, sh3(&sb), &device),
            };
        }
        let (sr, vr) = get("rows")?;
        train.rows = tensor(vr, [sr[0], sr[1]], &adev).require_grad();
        mom_r = Moments {
            m: tensor(get("mr")?.1, [sr[0], sr[1]], &device),
            v: tensor(get("vr")?.1, [sr[0], sr[1]], &device),
        };
        let num = |k: &str| {
            meta.get(k)
                .and_then(|v| v.parse::<f64>().ok())
                .unwrap_or(0.0)
        };
        step = num("step") as usize;
        best_val = meta
            .get("best_val")
            .and_then(|v| v.parse().ok())
            .unwrap_or(f64::INFINITY);
        best_step = num("best_step") as usize;
        bad = num("bad") as usize;
        evals = meta
            .get("evals")
            .and_then(|v| serde_json::from_str(v).ok())
            .unwrap_or_default();
        log(format!("resumed at step {step}"));
    }

    let per_epoch = cfg.evals_per_epoch.clamp(1, spe);
    let eval_points: std::collections::HashSet<usize> = (1..=per_epoch)
        .map(|k| ((k * spe) as f64 / per_epoch as f64).round() as usize % spe)
        .collect();
    let log_every = if cfg.log_every > 0 {
        cfg.log_every
    } else {
        (total / 200).max(1)
    };
    emit(
        "start",
        json!({
            "backend": backend, "train_clips": count, "val_clips": va.clips.len(), "minutes": tr.minutes,
            "skipped": tr.skipped + va.skipped, "too_long": tr.too_long + va.too_long,
            "tokens": tokens.len(), "tokens_found": tr.found, "not_single": not_single,
            "pieces": pieces.len(), "languages": languages, "new_languages": new,
            "lora_groups": GROUPS.len(), "rows": n_rows, "rank": rank, "scale": scale,
            "steps": total, "steps_per_epoch": spe, "warmup": sched.warmup, "eval_every": spe / per_epoch,
            "resumed_from": if step > 0 { json!(step) } else { Value::Null }, "trainer": "burn",
        }),
    );

    let adapter_tensors = |train: &Train| -> Vec<(String, Vec<usize>, Vec<f32>)> {
        let mut v = Vec::new();
        for (g, (site, proj)) in GROUPS.iter().enumerate() {
            let p = lora_param(*site, *proj);
            v.push((
                format!("lora/{p}/A"),
                train.a[g].dims().to_vec(),
                to_vec(&train.a[g]),
            ));
            v.push((
                format!("lora/{p}/B"),
                train.b[g].dims().to_vec(),
                to_vec(&train.b[g]),
            ));
        }
        v.push((
            "rows".into(),
            train.rows.dims().to_vec(),
            to_vec(&train.rows),
        ));
        v
    };
    let adapter_meta = json_meta(&[
        ("scale", json!(scale)),
        ("base", json!(cfg.base)),
        ("rank", json!(rank)),
        ("seed", json!(cfg.seed)),
        ("pieces", json!(pieces)),
        ("languages", json!(languages)),
    ]);

    let aug = cfg.augment.clone();
    let noise = if !aug.noise_dir.is_empty() && aug.noise_prob > 0.0 {
        Some(Noise::load(&aug.noise_dir)?)
    } else {
        None
    };
    let use_aug = aug.any();

    // Validation batches, built once.
    let val_batches: Vec<Batch> = (0..va.clips.len())
        .step_by(bs)
        .map(|i| {
            let idx: Vec<usize> = (i..(i + bs).min(va.clips.len())).collect();
            data::batch(&va.clips, &idx, &fe, None, mcfg.pad)
        })
        .collect();

    // WHISTLE_PROFILE=1: sync after each phase and print where a step's time goes.
    let profile = std::env::var("WHISTLE_PROFILE").is_ok();
    let prof = std::cell::RefCell::new(Vec::<(&str, f64)>::new());
    let mark = |name: &'static str, t: &Tensor<3>, since: &mut Instant| {
        if profile {
            let _ = t.clone().without_autodiff().sum().into_scalar::<f32>();
            prof.borrow_mut()
                .push((name, since.elapsed().as_secs_f64() * 1e3));
            *since = Instant::now();
        }
    };
    let forward = |train: &Train, b: &Batch, keep: Option<&[Tensor<3>]>| -> Tensor<1> {
        let mut t = Instant::now();
        let w = model.effective(train, scale, keep);
        let emb = model.embedding(&train.rows);
        mark("lora+quant", &w[14], &mut t);
        let feats = tensor(b.feats.clone(), [b.b, b.t, fe.mels], &device);
        let (encd, lens) = model.encode(feats, &b.frames, &w);
        mark("encoder", &encd, &mut t);
        let logits = model.decode(encd, &lens, &b.input, b.s, &w, &emb);
        mark("decoder", &logits, &mut t);
        model.loss(logits, &b.target)
    };

    let evaluate = |train: &Train,
                    step: usize,
                    best_val: &mut f64,
                    best_step: &mut usize,
                    bad: &mut usize,
                    evals: &mut Vec<Value>|
     -> Result<(), String> {
        let frozen = Train {
            a: train
                .a
                .iter()
                .map(|t| t.clone().without_autodiff())
                .collect(),
            b: train
                .b
                .iter()
                .map(|t| t.clone().without_autodiff())
                .collect(),
            rows: train.rows.clone().without_autodiff(),
        };
        let (mut sum, mut n) = (0f64, 0usize);
        for b in &val_batches {
            let l = to_vec(&forward(&frozen, b, None))[0] as f64;
            if std::env::var("WHISTLE_MEM").is_ok() {
                eprintln!(
                    "eval b{} t{} s{} {:?}",
                    b.b,
                    b.t,
                    b.s,
                    device.memory_pool_usage()
                );
            }
            device.memory_cleanup();
            sum += l * b.clips as f64;
            n += b.clips;
        }
        let val = sum / n as f64;
        let tensors = adapter_tensors(train);
        let ckpt = out
            .join("checkpoints")
            .join(format!("step_{step:06}.safetensors"));
        write_safetensors(&ckpt, tensors.clone(), adapter_meta.clone())?;
        write_safetensors(
            &out.join("adapter_last.safetensors"),
            tensors.clone(),
            adapter_meta.clone(),
        )?;
        let is_best = val < *best_val;
        if is_best {
            *best_val = val;
            *best_step = step;
            *bad = 0;
            write_safetensors(
                &out.join("adapter_best.safetensors"),
                tensors,
                adapter_meta.clone(),
            )?;
        } else {
            *bad += 1;
        }
        evals.push(json!({"step": step, "val_loss": val}));
        emit(
            "eval",
            json!({"step": step, "epoch": step as f64 / spe as f64, "val_loss": val, "best": is_best,
                   "checkpoint": ckpt.display().to_string()}),
        );
        Ok(())
    };

    let save_state = |train: &Train,
                      ma: &[Moments<3>],
                      mb: &[Moments<3>],
                      mr: &Moments<2>,
                      step: usize,
                      best_val: f64,
                      best_step: usize,
                      bad: usize,
                      evals: &[Value]|
     -> Result<(), String> {
        let mut t = Vec::new();
        for g in 0..GROUPS.len() {
            t.push((
                format!("a/{g}"),
                train.a[g].dims().to_vec(),
                to_vec(&train.a[g]),
            ));
            t.push((
                format!("b/{g}"),
                train.b[g].dims().to_vec(),
                to_vec(&train.b[g]),
            ));
            t.push((format!("ma/{g}"), ma[g].m.dims().to_vec(), to_vec(&ma[g].m)));
            t.push((format!("va/{g}"), ma[g].v.dims().to_vec(), to_vec(&ma[g].v)));
            t.push((format!("mb/{g}"), mb[g].m.dims().to_vec(), to_vec(&mb[g].m)));
            t.push((format!("vb/{g}"), mb[g].v.dims().to_vec(), to_vec(&mb[g].v)));
        }
        t.push((
            "rows".into(),
            train.rows.dims().to_vec(),
            to_vec(&train.rows),
        ));
        t.push(("mr".into(), mr.m.dims().to_vec(), to_vec(&mr.m)));
        t.push(("vr".into(), mr.v.dims().to_vec(), to_vec(&mr.v)));
        let meta: HashMap<String, String> = [
            ("step", step.to_string()),
            ("best_val", best_val.to_string()),
            ("best_step", best_step.to_string()),
            ("bad", bad.to_string()),
            ("evals", Value::from(evals.to_vec()).to_string()),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
        write_safetensors(&state_path, t, meta)
    };

    // Training batches are built ahead on worker threads, in step order.
    let mut order_rng = Rng::stream(cfg.seed, 0x0BDE);
    let epoch0 = step / spe;
    for _ in 0..epoch0 {
        order_rng.permutation(count);
    }
    let mut plan: Vec<(usize, Vec<usize>)> = Vec::new();
    {
        let mut r = order_rng.clone();
        let mut s = epoch0 * spe;
        'outer: for _ in epoch0..cfg.epochs {
            let order = r.permutation(count);
            for chunk in order.chunks(bs) {
                if s >= total {
                    break 'outer;
                }
                if s >= step {
                    plan.push((s, chunk.to_vec()));
                }
                s += 1;
            }
        }
    }
    // Reserved device memory above which the allocator's free pages go back
    // (WHISTLE_MEM_GB, default 8).
    let mem_limit = (std::env::var("WHISTLE_MEM_GB")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(8.0)
        * 1e9) as u64;
    log("starting (the first steps compile GPU kernels)");
    let workers = cfg.load_threads.clamp(1, 6);
    let code = std::thread::scope(|scope| -> Result<i32, String> {
        let (tx, rx) = mpsc::sync_channel::<(usize, Batch)>(workers * 2);
        let plan = &plan;
        let clips = &tr.clips;
        let fe = &fe;
        let aug = &aug;
        let noise = noise.as_ref();
        let pad = mcfg.pad;
        let seed = cfg.seed;
        for w in 0..workers {
            let tx = tx.clone();
            scope.spawn(move || {
                for (s, idx) in plan.iter().skip(w).step_by(workers) {
                    if STOP.load(Ordering::SeqCst) {
                        break;
                    }
                    let mut rng = Rng::stream(seed, 1_000_000 + *s as u64);
                    let b = if use_aug {
                        data::batch(clips, idx, fe, Some((aug, &mut rng, noise)), pad)
                    } else {
                        data::batch(clips, idx, fe, None, pad)
                    };
                    if tx.send((*s, b)).is_err() {
                        break;
                    }
                }
            });
        }
        drop(tx);

        let mut pending: BTreeMap<usize, Batch> = BTreeMap::new();
        let (mut win_loss, mut win_norm, mut win_clip) =
            (Vec::<f64>::new(), Vec::<f64>::new(), 0usize);
        let mut t_win = Instant::now();
        let mut drop_rng = Rng::stream(cfg.seed, 0xD50);
        for _ in 0..step {
            drop_rng.next();
        }
        let mut finished = step >= total;
        while !finished {
            let batch = loop {
                if let Some(b) = pending.remove(&step) {
                    break b;
                }
                match rx.recv() {
                    Ok((s, b)) => {
                        pending.insert(s, b);
                    }
                    Err(_) => return Err("the batch workers stopped".into()),
                }
            };
            // Rank dropout masks, [L, 1, r] per group, scaled by 1/keep.
            let keep = if cfg.lora_dropout > 0.0 {
                let kp = 1.0 - cfg.lora_dropout;
                let mut r = Rng::stream(cfg.seed, 2_000_000 + step as u64);
                Some(
                    model
                        .lora_dims
                        .iter()
                        .map(|&(l, _, _)| {
                            let m: Vec<f32> = (0..l * rank)
                                .map(|_| if r.f64() < kp { (1.0 / kp) as f32 } else { 0.0 })
                                .collect();
                            tensor(m, [l, 1, rank], &device)
                        })
                        .collect::<Vec<_>>(),
                )
            } else {
                None
            };
            let loss = forward(&train, &batch, keep.as_deref());
            let mut tb = Instant::now();
            let mut grads = loss.backward();
            if profile {
                let g = train.b[0].grad(&grads).unwrap();
                mark("backward", &g, &mut tb);
            }
            let ga: Vec<Tensor<3>> = train
                .a
                .iter()
                .map(|t| t.grad_remove(&mut grads).expect("grad A"))
                .collect();
            let gb: Vec<Tensor<3>> = train
                .b
                .iter()
                .map(|t| t.grad_remove(&mut grads).expect("grad B"))
                .collect();
            let gr = train.rows.grad_remove(&mut grads).expect("grad rows");
            let mut sq = gr.clone().square().sum();
            for g in ga.iter().chain(&gb) {
                sq = sq + g.clone().square().sum();
            }
            let stats = to_vec(&Tensor::cat(vec![loss.detach(), sq.sqrt()], 0));
            let (lv, norm) = (stats[0] as f64, stats[1] as f64);
            if !lv.is_finite() {
                // Where did it come from? Recompute without autodiff and look.
                let fz = Train {
                    a: train
                        .a
                        .iter()
                        .map(|t| t.clone().without_autodiff())
                        .collect(),
                    b: train
                        .b
                        .iter()
                        .map(|t| t.clone().without_autodiff())
                        .collect(),
                    rows: train.rows.clone().without_autodiff(),
                };
                let bad = |t: &Tensor<3>| {
                    t.clone().is_nan().any().into_scalar::<bool>()
                        || t.clone().is_inf().any().into_scalar::<bool>()
                };
                let w = model.effective(&fz, scale, keep.as_deref());
                let wbad: Vec<usize> = (0..w.len()).filter(|&g| bad(&w[g])).collect();
                let feats = tensor(batch.feats.clone(), [batch.b, batch.t, fe.mels], &device);
                let fbad = bad(&feats);
                let (encd, lens) = model.encode(feats, &batch.frames, &w);
                let ebad = bad(&encd);
                let emb = model.embedding(&fz.rows);
                let logits = model.decode(encd, &lens, &batch.input, batch.s, &w, &emb);
                let lbad = bad(&logits);
                return Err(format!(
                    "loss became {lv} at step {} (non-finite: weights {wbad:?}, features {fbad}, encoder {ebad}, logits {lbad}; batch t {} s {} frames {:?} encoder lengths {lens:?})",
                    step + 1,
                    batch.t,
                    batch.s,
                    batch.frames
                ));
            }
            if !norm.is_finite() {
                let bad = |t: &Tensor<3>| !t.clone().is_finite().all().into_scalar::<bool>();
                let ba: Vec<usize> = (0..ga.len()).filter(|&g| bad(&ga[g])).collect();
                let bb: Vec<usize> = (0..gb.len()).filter(|&g| bad(&gb[g])).collect();
                let br = !gr.clone().is_finite().all().into_scalar::<bool>();
                return Err(format!(
                    "gradient became {norm} at step {} (loss {lv}; non-finite A {ba:?} B {bb:?} rows {br}; batch t {} s {} frames {:?})",
                    step + 1,
                    batch.t,
                    batch.s,
                    batch.frames
                ));
            }
            let f = if cfg.grad_clip > 0.0 && norm > cfg.grad_clip {
                cfg.grad_clip / norm
            } else {
                1.0
            };
            let lr = sched.at(step);
            let t = step + 1;
            for g in 0..GROUPS.len() {
                train.a[g] = adamw(
                    &train.a[g],
                    ga[g].clone().mul_scalar(f),
                    &mut mom_a[g],
                    t,
                    lr,
                    cfg.weight_decay,
                );
                train.b[g] = adamw(
                    &train.b[g],
                    gb[g].clone().mul_scalar(f),
                    &mut mom_b[g],
                    t,
                    lr,
                    cfg.weight_decay,
                );
            }
            train.rows = adamw(
                &train.rows,
                gr.mul_scalar(f),
                &mut mom_r,
                t,
                lr * cfg.rows_lr_mult,
                cfg.rows_weight_decay,
            );
            // Batches differ in shape, so the allocator's pages grow with the
            // longest one seen; hand them back when they hold a lot.
            let reserved = device.memory_pool_usage().map_or(0, |u| u.bytes_reserved);
            if std::env::var("WHISTLE_MEM").is_ok() {
                eprintln!(
                    "step {step} b{} t{} s{} reserved {:.2} GB",
                    batch.b,
                    batch.t,
                    batch.s,
                    reserved as f64 / 1e9
                );
            }
            if reserved > mem_limit {
                device.memory_cleanup();
            }
            if profile {
                let _ = to_vec(&train.rows);
                let p: Vec<String> = prof
                    .borrow()
                    .iter()
                    .map(|(n, ms)| format!("{n} {ms:.0}"))
                    .collect();
                eprintln!(
                    "step {step} t{} s{}: {} | optimizer+sync {:.0} ms",
                    batch.t,
                    batch.s,
                    p.join(", "),
                    tb.elapsed().as_secs_f64() * 1e3
                );
                prof.borrow_mut().clear();
            }
            step += 1;
            win_loss.push(lv);
            win_norm.push(norm);
            win_clip += (cfg.grad_clip > 0.0 && norm > cfg.grad_clip) as usize;
            if step.is_multiple_of(log_every) || step == total {
                let n = win_loss.len() as f64;
                emit(
                    "step",
                    json!({"step": step, "epoch": step as f64 / spe as f64,
                           "loss": win_loss.iter().sum::<f64>() / n,
                           "loss_max": win_loss.iter().cloned().fold(f64::MIN, f64::max),
                           "lr": sched.at(step), "grad_norm": win_norm.iter().sum::<f64>() / n,
                           "clipped": win_clip as f64 / n, "secs": t_win.elapsed().as_secs_f64() / n}),
                );
                win_loss.clear();
                win_norm.clear();
                win_clip = 0;
                t_win = Instant::now();
            }
            if eval_points.contains(&(step % spe)) || step == total {
                evaluate(
                    &train,
                    step,
                    &mut best_val,
                    &mut best_step,
                    &mut bad,
                    &mut evals,
                )?;
                save_state(
                    &train, &mom_a, &mom_b, &mom_r, step, best_val, best_step, bad, &evals,
                )?;
                if cfg.early_stop > 0 && bad >= cfg.early_stop {
                    emit(
                        "early_stop",
                        json!({"step": step, "patience": cfg.early_stop}),
                    );
                    finished = true;
                }
            }
            if STOP.load(Ordering::SeqCst) {
                save_state(
                    &train, &mom_a, &mom_b, &mom_r, step, best_val, best_step, bad, &evals,
                )?;
                emit("stopped", json!({"step": step}));
                return Ok(3);
            }
            if step >= total {
                finished = true;
            }
        }
        STOP.store(true, Ordering::SeqCst);
        drop(rx);
        if evals.last().and_then(|e| e["step"].as_u64()) != Some(step as u64) {
            evaluate(
                &train,
                step,
                &mut best_val,
                &mut best_step,
                &mut bad,
                &mut evals,
            )?;
            save_state(
                &train, &mom_a, &mom_b, &mom_r, step, best_val, best_step, bad, &evals,
            )?;
        }
        emit(
            "done",
            json!({"best_step": best_step, "best_val": best_val,
                   "adapter": out.join("adapter_best.safetensors").display().to_string(), "steps": step}),
        );
        Ok(0)
    })?;
    Ok(code)
}
