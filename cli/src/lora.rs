//! Reading a LoRA adapter written by Cactus's fine-tuning
//! (`needle whistle finetune --out adapter.safetensors`).
//!
//! Tensors: `lora/<param path>/kernel/A` `[layers, in, rank]` and `.../B`
//! `[layers, rank, out]` per projection, `rows` `[pieces + languages, d]`.
//! Metadata (JSON strings): `scale` (alpha / rank), `rank`, `pieces` (the
//! added tokenizer pieces), `languages` (every language, base ones first).

use cactus_whistle::lora::{LoraWeight, Proj, Site};
use cactus_whistle::{Lora, Whistle};
use safetensors::{Dtype, SafeTensors};

/// One projection's update, owned.
pub struct Update {
    pub site: Site,
    pub proj: Proj,
    pub layers: usize,
    pub din: usize,
    pub rank: usize,
    pub dout: usize,
    pub a: Vec<f32>,
    pub b: Vec<f32>,
}

/// An adapter file's contents.
pub struct Adapter {
    pub scale: f32,
    pub rank: usize,
    pub pieces: Vec<String>,
    pub languages: Vec<String>,
    pub updates: Vec<Update>,
    pub rows: Vec<f32>,
    pub d: usize,
}

fn site_proj(name: &str) -> Option<(Site, Proj)> {
    let site = if name.starts_with("lora/encoder/") && name.contains("/self_attn/") {
        Site::Encoder
    } else if name.starts_with("lora/stack/") && name.contains("/self_attn/") {
        Site::DecoderSelf
    } else if name.starts_with("lora/stack/") && name.contains("/cross_attn/") {
        Site::Cross
    } else {
        return None;
    };
    let proj = [
        ("/q_proj/", Proj::Q),
        ("/k_proj/", Proj::K),
        ("/v_proj/", Proj::V),
        ("/gate_proj/", Proj::Gate),
        ("/out_proj/", Proj::Out),
    ]
    .into_iter()
    .find(|(k, _)| name.contains(k))?
    .1;
    Some((site, proj))
}

fn f32s(st: &SafeTensors, name: &str) -> Result<(Vec<usize>, Vec<f32>), String> {
    let t = st.tensor(name).map_err(|e| format!("{name}: {e}"))?;
    let data = t.data();
    let v: Vec<f32> = match t.dtype() {
        Dtype::F32 => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        Dtype::F16 => data
            .chunks_exact(2)
            .map(|c| cactus_whistle::math::f16(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        other => return Err(format!("{name}: dtype {other:?}, expected F32 or F16")),
    };
    Ok((t.shape().to_vec(), v))
}

fn meta<T: serde::de::DeserializeOwned>(
    m: &std::collections::HashMap<String, String>,
    key: &str,
) -> Result<T, String> {
    let s = m.get(key).ok_or_else(|| format!("metadata has no {key}"))?;
    serde_json::from_str(s).map_err(|e| format!("metadata {key}: {e}"))
}

impl Adapter {
    pub fn load(path: &str) -> Result<Adapter, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
        let (_, header) = SafeTensors::read_metadata(&bytes).map_err(|e| format!("{path}: {e}"))?;
        let st = SafeTensors::deserialize(&bytes).map_err(|e| format!("{path}: {e}"))?;
        let m = header
            .metadata()
            .clone()
            .ok_or_else(|| format!("{path}: no metadata; not a Whistle adapter"))?;
        let languages: Option<Vec<String>> = meta(&m, "languages")?;
        let languages = languages
            .ok_or_else(|| format!("{path} is a Needle (not Whistle) adapter: no languages"))?;
        let scale: f64 = meta(&m, "scale")?;
        let rank: usize = meta(&m, "rank")?;
        let pieces: Vec<String> = meta::<Option<Vec<String>>>(&m, "pieces")?.unwrap_or_default();

        let mut names: Vec<String> = st.names().into_iter().map(str::to_string).collect();
        names.sort();
        let mut updates = Vec::new();
        for name in names.iter().filter(|n| n.ends_with("/A")) {
            let Some((site, proj)) = site_proj(name) else {
                return Err(format!(
                    "{path}: {name} is a projection this engine can't adapt"
                ));
            };
            let bname = format!("{}/B", &name[..name.len() - 2]);
            let (sa, a) = f32s(&st, name)?;
            let (sb, b) = f32s(&st, &bname)?;
            if sa.len() != 3 || sb.len() != 3 || sa[0] != sb[0] || sa[2] != sb[1] {
                return Err(format!("{path}: {name} {sa:?} and B {sb:?} don't match"));
            }
            updates.push(Update {
                site,
                proj,
                layers: sa[0],
                din: sa[1],
                rank: sa[2],
                dout: sb[2],
                a,
                b,
            });
        }
        let (rows, d) = if names.iter().any(|n| n == "rows") {
            let (shape, rows) = f32s(&st, "rows")?;
            if shape.len() != 2 {
                return Err(format!("{path}: rows has shape {shape:?}"));
            }
            (rows, shape[1])
        } else {
            (Vec::new(), 0)
        };
        Ok(Adapter {
            scale: scale as f32,
            rank,
            pieces,
            languages,
            updates,
            rows,
            d,
        })
    }

    /// Languages the adapter adds to `w`'s, after checking the base ones
    /// match.
    pub fn added_languages<'s>(&'s self, w: &Whistle) -> Result<Vec<&'s str>, String> {
        let n = (0..)
            .take_while(|&i| !w.base_lang_code(i).is_empty())
            .count();
        let base: Vec<&str> = (0..n).map(|i| w.base_lang_code(i)).collect();
        if self.languages.len() < n || self.languages[..n] != base {
            return Err(format!(
                "the adapter was trained on a model with languages {} but this model has {}; \
                 pass the base model it was fine-tuned from",
                self.languages.join(", "),
                base.join(", ")
            ));
        }
        let added: Vec<&str> = self.languages[n..].iter().map(String::as_str).collect();
        let want = (self.pieces.len() + added.len()) * w.d;
        if self.rows.len() != want {
            return Err(format!(
                "the adapter has {} vocabulary rows for {} pieces and {} languages",
                self.rows.len() / self.d.max(1),
                self.pieces.len(),
                added.len()
            ));
        }
        Ok(added)
    }

    /// The tokenizer of `w` extended with the adapter's pieces and languages.
    pub fn tokenizer(&self, w: &Whistle) -> Result<Vec<u8>, String> {
        let added = self.added_languages(w)?;
        let pieces: Vec<&str> = self.pieces.iter().map(String::as_str).collect();
        cactus_whistle::tokenizer::extend(w.tokenizer_blob(), w.base_text_pieces(), &pieces, &added)
            .ok_or_else(|| "cannot extend the model's tokenizer".to_string())
    }

    /// The engine's view of the adapter, borrowing from `self` and
    /// `tokenizer` (from [`Adapter::tokenizer`]).
    pub fn lora<'s>(&'s self, tokenizer: &'s [u8]) -> Lora<'s> {
        let mut weights: [[Option<LoraWeight<'s>>; 5]; 3] = Default::default();
        for u in &self.updates {
            weights[u.site as usize][u.proj as usize] = Some(LoraWeight {
                a: &u.a,
                b: &u.b,
                layers: u.layers,
                din: u.din,
                rank: u.rank,
                dout: u.dout,
            });
        }
        Lora {
            scale: self.scale,
            weights,
            rows: &self.rows,
            pieces: self.pieces.len(),
            tokenizer,
        }
    }
}
