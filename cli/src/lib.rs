//! Helpers shared by the `whistle` binary and the tests.

pub mod audio;

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
