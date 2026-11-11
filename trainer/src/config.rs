//! The sweep file (TOML). Paths are relative to the file.
//!
//! ```toml
//! out_dir = "runs_v4"
//!
//! [data]
//! train = "dataset_v2/train.jsonl"
//! validation = "dataset_v2/validation.jsonl"
//! test = "dataset_v2/test.jsonl"           # optional
//! tokens = "tokens.txt"
//! language = "cs"
//! base = "checkpoints/whistle.safetensors"
//! template = "model/whistle.cact"          # the published model: tokenizer, mel, permutations
//!
//! [eval]                                   # the Rust engine, on the CPU, per checkpoint
//! base_cact = "model/whistle_w4.cact"      # built from base + template if missing
//! clips = 200
//! samples = ["cz_jirka.wav", { audio = "pub.wav", text = "reference, if known" }]
//!
//! [defaults]                               # trainer settings for every run
//! epochs = 5
//! lr = 2e-3
//!
//! [[run]]
//! name = "wd1e-2"
//! weight_decay = 1e-2
//! ```

use std::path::{Path, PathBuf};

use serde::Deserialize;
use toml::Table;

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Data {
    pub train: PathBuf,
    pub validation: PathBuf,
    #[serde(default)]
    pub test: Option<PathBuf>,
    #[serde(default)]
    pub tokens: Option<PathBuf>,
    #[serde(default)]
    pub language: Option<String>,
    pub base: PathBuf,
    /// A published Whistle `.cact` (whistle.cact): models built from `base`
    /// copy its tokenizer, mel filterbank and Hadamard permutations.
    #[serde(default)]
    pub template: Option<PathBuf>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum SampleSpec {
    Path(PathBuf),
    WithText {
        audio: PathBuf,
        text: Option<String>,
    },
}

#[derive(Debug, Clone)]
pub struct Sample {
    pub audio: PathBuf,
    pub text: Option<String>,
}

fn d_clips() -> usize {
    200
}
fn d_threads() -> usize {
    std::thread::available_parallelism().map_or(2, |n| (n.get() / 3).max(1))
}
fn d_spec() -> String {
    "fast".into()
}
fn d_test_spec() -> String {
    "guide".into()
}
fn d_true() -> bool {
    true
}
fn d_stall() -> f64 {
    30.0
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Eval {
    /// 4-bit base model for runtime-LoRA evaluation.
    pub base_cact: PathBuf,
    /// Validation clips decoded per checkpoint for WER (0 = off).
    #[serde(default = "d_clips")]
    pub clips: usize,
    /// CPU threads for evaluation (the trainer needs some too).
    #[serde(default = "d_threads")]
    pub threads: usize,
    /// Decoding during training: fast (greedy) or guide (5-beam).
    #[serde(default = "d_spec")]
    pub spec: String,
    /// Recordings transcribed at every checkpoint.
    #[serde(default)]
    samples: Vec<SampleSpec>,
    /// Language to force; default: detect.
    #[serde(default)]
    pub lang: Option<String>,
    /// WER of the best adapter on the whole test split after each run.
    #[serde(default = "d_true")]
    pub test_at_end: bool,
    #[serde(default = "d_test_spec")]
    pub test_spec: String,
    #[serde(skip)]
    pub sample_list: Vec<Sample>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sweep {
    pub out_dir: PathBuf,
    /// Build `tuned_<run>.cact` from the best adapter after each run.
    #[serde(default)]
    pub build: bool,
    /// Kill a run whose trainer printed nothing for this many minutes.
    #[serde(default = "d_stall")]
    pub stall_minutes: f64,
    /// Continue with the next run after one fails.
    #[serde(default)]
    pub keep_going: bool,
    pub data: Data,
    pub eval: Eval,
    #[serde(default)]
    pub defaults: Table,
    #[serde(default, rename = "run")]
    pub runs: Vec<Table>,
}

/// One run: its name and the trainer config (defaults merged with the run's
/// own settings).
#[derive(Debug, Clone)]
pub struct Run {
    pub name: String,
    pub settings: Table,
}

fn merge(base: &mut Table, over: &Table) {
    for (k, v) in over {
        match (base.get_mut(k), v) {
            (Some(toml::Value::Table(b)), toml::Value::Table(o)) => merge(b, o),
            _ => {
                base.insert(k.clone(), v.clone());
            }
        }
    }
}

impl Sweep {
    pub fn load(path: &Path) -> Result<Sweep, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut s: Sweep = toml::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        let path = std::path::absolute(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let root = path.parent().unwrap_or(Path::new("/")).to_path_buf();
        let abs = |p: &PathBuf| {
            if p.is_absolute() {
                p.clone()
            } else {
                root.join(p)
            }
        };
        s.out_dir = abs(&s.out_dir);
        s.data.train = abs(&s.data.train);
        s.data.validation = abs(&s.data.validation);
        s.data.test = s.data.test.as_ref().map(abs);
        s.data.tokens = s.data.tokens.as_ref().map(abs);
        s.data.base = abs(&s.data.base);
        s.data.template = s.data.template.as_ref().map(abs);
        s.eval.base_cact = abs(&s.eval.base_cact);
        s.eval.sample_list = s
            .eval
            .samples
            .iter()
            .map(|x| match x {
                SampleSpec::Path(p) => Sample {
                    audio: abs(p),
                    text: None,
                },
                SampleSpec::WithText { audio, text } => Sample {
                    audio: abs(audio),
                    text: text.clone(),
                },
            })
            .collect();
        if s.runs.is_empty() {
            return Err(format!("{}: no [[run]] entries", path.display()));
        }
        let mut seen = std::collections::HashSet::new();
        for r in &s.runs {
            match r.get("name").and_then(|v| v.as_str()) {
                Some(n) if !n.is_empty() && !n.contains('/') => {
                    if !seen.insert(n.to_string()) {
                        return Err(format!("run name {n} appears twice"));
                    }
                }
                _ => return Err("every [[run]] needs a name (no '/')".into()),
            }
        }
        Ok(s)
    }

    /// Writes `out`: `base` as a 4-bit `.cact`, with `adapter` merged if given.
    pub fn build_cact(&self, adapter: Option<&Path>, out: &Path) -> Result<(), String> {
        use cactus_whistle_common::{checkpoint::Checkpoint, export, lora::Adapter};
        let template = self
            .data
            .template
            .as_ref()
            .ok_or("[data] template is needed to build models: the published whistle.cact")?;
        let tpl = std::fs::read(template).map_err(|e| format!("{}: {e}", template.display()))?;
        let ck = Checkpoint::load(&self.data.base.display().to_string())?;
        let adapter = match adapter {
            Some(a) => Some(Adapter::load(&a.display().to_string())?),
            None => None,
        };
        let built = export::build(&ck, adapter.as_ref(), &tpl)?;
        if let Some(p) = out.parent() {
            std::fs::create_dir_all(p).map_err(|e| format!("{}: {e}", p.display()))?;
        }
        let tmp = out.with_extension("cact.tmp");
        std::fs::write(&tmp, &built.blob).map_err(|e| format!("{}: {e}", tmp.display()))?;
        std::fs::rename(&tmp, out).map_err(|e| format!("{}: {e}", out.display()))
    }

    pub fn runs(&self) -> Vec<Run> {
        self.runs
            .iter()
            .map(|r| {
                let mut settings = self.defaults.clone();
                merge(&mut settings, r);
                let name = settings
                    .remove("name")
                    .and_then(|v| v.as_str().map(str::to_string))
                    .unwrap_or_default();
                Run { name, settings }
            })
            .collect()
    }

    /// The JSON config the trainer (`whistle-train train`) reads for `run`.
    pub fn trainer_config(&self, run: &Run) -> serde_json::Value {
        let mut v = serde_json::to_value(&run.settings).unwrap_or_default();
        let m = v.as_object_mut().expect("table");
        let p = |p: &Path| serde_json::Value::String(p.display().to_string());
        m.insert("train".into(), p(&self.data.train));
        m.insert("validation".into(), p(&self.data.validation));
        m.insert("base".into(), p(&self.data.base));
        if let Some(t) = &self.data.template {
            m.insert("template".into(), p(t));
        }
        m.insert("out_dir".into(), p(&self.out_dir.join(&run.name)));
        if let Some(t) = &self.data.tokens {
            m.insert("tokens".into(), p(t));
        }
        if let Some(l) = &self.data.language {
            m.insert("language".into(), l.clone().into());
        }
        v
    }
}
