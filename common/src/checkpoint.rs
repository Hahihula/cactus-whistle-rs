//! The float Whistle checkpoint (`checkpoints/whistle.safetensors`, Cactus's
//! training format): every parameter as a flat `/`-separated name, the model
//! geometry as JSON in the `config` metadata string.

use std::collections::BTreeMap;

use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;

/// A dense f32 array, row-major.
#[derive(Clone, Debug, PartialEq)]
pub struct Array {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Array {
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Array {
        assert_eq!(shape.iter().product::<usize>(), data.len());
        Array { shape, data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Slice `i` of the leading axis (a layer of a stacked parameter).
    pub fn index(&self, i: usize) -> Array {
        let inner: usize = self.shape[1..].iter().product();
        Array::new(
            self.shape[1..].to_vec(),
            self.data[i * inner..(i + 1) * inner].to_vec(),
        )
    }

    /// The last two axes swapped (a `[.., in, out]` kernel as `[.., out, in]`).
    pub fn transpose_last(&self) -> Array {
        let n = self.shape.len();
        assert!(n >= 2);
        let (r, c) = (self.shape[n - 2], self.shape[n - 1]);
        let lead = self.data.len() / (r * c);
        let mut out = vec![0f32; self.data.len()];
        for l in 0..lead {
            let (src, dst) = (
                &self.data[l * r * c..],
                &mut out[l * r * c..(l + 1) * r * c],
            );
            for i in 0..r {
                for j in 0..c {
                    dst[j * r + i] = src[i * c + j];
                }
            }
        }
        let mut shape = self.shape.clone();
        shape.swap(n - 2, n - 1);
        Array::new(shape, out)
    }
}

/// `WhistleConfig` as Cactus saves it (fields this crate doesn't need are ignored).
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(default)]
pub struct Config {
    pub text: usize,
    pub languages: Vec<String>,
    pub dim: usize,
    pub kv_heads: usize,
    pub qk_dim: usize,
    pub v_dim: usize,
    pub enc_layers: usize,
    pub dec_layers: usize,
    pub conv_kernel: usize,
    pub taps: usize,
    pub lanes: usize,
    pub enc_lanes: usize,
    pub sites: Vec<usize>,
    pub orders: Vec<usize>,
    pub slots: usize,
    pub hashes: usize,
    pub stem: usize,
    pub stem_stages: usize,
    pub rope: f32,
    pub mels: usize,
    pub rate: usize,
    pub fft: usize,
    pub win: usize,
    pub hop: usize,
    pub max_tokens: usize,
    pub pad: u32,
    pub eos: u32,
    pub bos: u32,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            text: 8192,
            languages: ["en", "de", "fr", "es", "it", "nl", "pl"]
                .map(String::from)
                .to_vec(),
            dim: 512,
            kv_heads: 2,
            qk_dim: 48,
            v_dim: 64,
            enc_layers: 8,
            dec_layers: 8,
            conv_kernel: 9,
            taps: 3,
            lanes: 4,
            enc_lanes: 4,
            sites: vec![3, 7],
            orders: vec![2, 3],
            slots: 18432,
            hashes: 2,
            stem: 128,
            stem_stages: 3,
            rope: 100000.0,
            mels: 80,
            rate: 16000,
            fft: 512,
            win: 400,
            hop: 160,
            max_tokens: 320,
            pad: 0,
            eos: 1,
            bos: 2,
        }
    }
}

impl Config {
    pub fn heads(&self) -> usize {
        self.dim / self.v_dim
    }
    pub fn vocab(&self) -> usize {
        self.text + self.languages.len()
    }
    /// Engram hash heads per order and the width of one table row.
    pub fn engram_geometry(&self) -> (usize, usize) {
        let heads = if self.hashes > 0 {
            self.hashes
        } else {
            (self.dim / (self.orders.len() * 128)).max(1)
        };
        (heads, self.dim / (self.orders.len() * heads))
    }
}

/// A loaded checkpoint.
pub struct Checkpoint {
    pub config: Config,
    pub params: BTreeMap<String, Array>,
}

pub(crate) fn tensor_f32(st: &SafeTensors, name: &str) -> Result<Array, String> {
    let t = st.tensor(name).map_err(|e| format!("{name}: {e}"))?;
    let data = t.data();
    let v: Vec<f32> = match t.dtype() {
        Dtype::F32 => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        Dtype::F16 => data
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        Dtype::BF16 => data
            .chunks_exact(2)
            .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        other => return Err(format!("{name}: dtype {other:?}, expected a float")),
    };
    Ok(Array::new(t.shape().to_vec(), v))
}

impl Checkpoint {
    pub fn load(path: &str) -> Result<Checkpoint, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
        let (_, header) = SafeTensors::read_metadata(&bytes).map_err(|e| format!("{path}: {e}"))?;
        let config = match header.metadata().as_ref().and_then(|m| m.get("config")) {
            Some(s) if !s.is_empty() => {
                serde_json::from_str(s).map_err(|e| format!("{path}: config: {e}"))?
            }
            _ => {
                return Err(format!(
                    "{path}: no config metadata; not a Whistle checkpoint"
                ));
            }
        };
        let st = SafeTensors::deserialize(&bytes).map_err(|e| format!("{path}: {e}"))?;
        let mut params = BTreeMap::new();
        for name in st.names() {
            params.insert(name.to_string(), tensor_f32(&st, name)?);
        }
        if !params.contains_key("stem/w") || !params.contains_key("encoder/final_norm/scale") {
            return Err(format!(
                "{path} is not a Whistle checkpoint (no stem or encoder)"
            ));
        }
        Ok(Checkpoint { config, params })
    }

    pub fn get(&self, name: &str) -> Result<&Array, String> {
        self.params
            .get(name)
            .ok_or_else(|| format!("checkpoint has no {name}"))
    }
}
