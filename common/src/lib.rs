//! Desktop helpers shared by the `whistle` CLI and the `whistle-train`
//! fine-tuning tool. The engine (`cactus-whistle`) stays `no_std`; everything
//! that needs files, threads or parsing lives here.

pub mod audio;
pub mod dataset;
pub mod lora;
pub mod wer;

pub use audio::load;

use cactus_whistle::Whistle;

/// The model's language codes, in tag order.
pub fn lang_codes<'a>(w: &Whistle<'a>) -> Vec<&'a str> {
    (0..w.n_langs()).map(|i| w.lang_code(i)).collect()
}

/// Resolve `--lang xx` against the model's own languages (a fine-tune can
/// add some): `None` or `auto` detects.
pub fn resolve_lang(w: &Whistle, code: Option<&str>) -> Result<Option<usize>, String> {
    match code {
        None | Some("auto") => Ok(None),
        Some(c) => w.lang_index(c).map(Some).ok_or_else(|| {
            format!(
                "the model has no language {c}; it has {}",
                lang_codes(w).join(", ")
            )
        }),
    }
}

/// Read the adapter at `path` for model `w`: the adapter and the extended
/// tokenizer the engine's [`cactus_whistle::Lora`] borrows (see
/// [`lora::Adapter::lora`]).
pub fn load_lora(w: &Whistle, path: &str) -> Result<(lora::Adapter, Vec<u8>), String> {
    let adapter = lora::Adapter::load(path)?;
    let tokenizer = adapter.tokenizer(w).map_err(|e| format!("{path}: {e}"))?;
    Ok((adapter, tokenizer))
}
