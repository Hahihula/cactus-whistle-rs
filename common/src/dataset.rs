//! The fine-tuning dataset format: JSONL, one clip per line,
//! `{"audio": "clips/0001.wav", "text": "...", "language": "cs"}` with `audio`
//! relative to the JSONL file (Cactus's `whistle finetune` format).

use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct Clip {
    /// As written in the file (relative to it).
    pub audio: String,
    /// `audio` resolved against the file's directory.
    pub path: PathBuf,
    pub text: String,
    pub language: Option<String>,
    /// `source` if the line has one, else the audio file name's prefix
    /// before the first `_` (`cv_common_voice_cs_1.wav` -> `cv`).
    pub source: String,
}

/// Read a JSONL split. `limit` keeps the first N clips.
pub fn read_jsonl(path: &Path, limit: Option<usize>) -> Result<Vec<Clip>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let root = path.parent().unwrap_or(Path::new("."));
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        if limit.is_some_and(|n| out.len() >= n) {
            break;
        }
        let v: serde_json::Value =
            serde_json::from_str(line).map_err(|e| format!("{}:{}: {e}", path.display(), i + 1))?;
        let s = |k: &str| v.get(k).and_then(|x| x.as_str()).map(str::to_string);
        let audio = s("audio").ok_or_else(|| format!("{}:{}: no audio", path.display(), i + 1))?;
        let source = s("source").unwrap_or_else(|| {
            let stem = Path::new(&audio)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("");
            stem.split('_').next().unwrap_or("").to_string()
        });
        out.push(Clip {
            path: root.join(&audio),
            audio,
            text: s("text").unwrap_or_default(),
            language: s("language"),
            source,
        });
    }
    Ok(out)
}
