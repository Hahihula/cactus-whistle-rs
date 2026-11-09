//! Runtime LoRA: run a base model with a fine-tuned adapter without merging
//! it into a new `.cact` first.
//!
//! An adapter from Cactus's tooling (`needle whistle finetune`) has
//!
//! - low-rank updates `W + scale * A @ B` for the attention projections (Q, K,
//!   V, gate and output) of the encoder self-attention, the decoder
//!   self-attention and the cross-attention, stacked over layers;
//! - embedding rows for the tokens and languages it adds. New tokens come
//!   after the base model's text pieces, new languages after its language
//!   tags, so token ids from the first new piece on change.
//!
//! The engine adds `strength * scale * (x @ A) @ B` in float next to each
//! int8 matmul, so the update is not quantised. `needle whistle build` instead
//! merges it into the weights before the 4-bit export (the numerics training
//! simulates), so results are close to, but not the same as, a built model.
//!
//! Everything is borrowed: parse the adapter file on the host (the CLI does
//! it for `.safetensors`) and pass slices.

/// Where a projection is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Site {
    /// Encoder self-attention (`encoder/layers/block/self_attn`).
    Encoder = 0,
    /// Decoder self-attention (`stack/layers/block/self_attn`).
    DecoderSelf = 1,
    /// Decoder cross-attention (`stack/layers/block/cross_attn`).
    Cross = 2,
}

/// Which projection of an attention block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Proj {
    Q = 0,
    K = 1,
    V = 2,
    Gate = 3,
    Out = 4,
}

pub const SITES: [Site; 3] = [Site::Encoder, Site::DecoderSelf, Site::Cross];
pub const PROJS: [Proj; 5] = [Proj::Q, Proj::K, Proj::V, Proj::Gate, Proj::Out];

/// Largest adapter rank the engine takes.
pub const MAX_RANK: usize = 128;

/// One projection's update for every layer: `a` is `[layers][din][rank]`,
/// `b` is `[layers][rank][dout]`, row-major (the adapter file's layout).
#[derive(Clone, Copy, Debug)]
pub struct LoraWeight<'a> {
    pub a: &'a [f32],
    pub b: &'a [f32],
    pub layers: usize,
    pub din: usize,
    pub rank: usize,
    pub dout: usize,
}

impl LoraWeight<'_> {
    /// Shapes agree with the slices.
    pub fn valid(&self) -> bool {
        self.rank > 0
            && self.rank <= MAX_RANK
            && self.a.len() == self.layers * self.din * self.rank
            && self.b.len() == self.layers * self.rank * self.dout
    }

    /// `y[t] += scale * (x[t] @ A[layer]) @ B[layer]` for `rows` rows. Plain
    /// multiply-adds, which vectorise (`fma` is a library call on targets
    /// without hardware FMA); the result may differ in the last bits by platform.
    pub(crate) fn apply(&self, layer: usize, scale: f32, x: &[f32], rows: usize, y: &mut [f32]) {
        let (din, r, dout) = (self.din, self.rank, self.dout);
        let a = &self.a[layer * din * r..(layer + 1) * din * r];
        let b = &self.b[layer * r * dout..(layer + 1) * r * dout];
        let mut u = [0f32; MAX_RANK];
        for t in 0..rows {
            let u = &mut u[..r];
            u.fill(0.0);
            for (i, &xi) in x[t * din..(t + 1) * din].iter().enumerate() {
                for (uk, &ak) in u.iter_mut().zip(&a[i * r..(i + 1) * r]) {
                    *uk += xi * ak;
                }
            }
            let yt = &mut y[t * dout..(t + 1) * dout];
            for (k, &uk) in u.iter().enumerate() {
                let uk = uk * scale;
                for (yo, &bo) in yt.iter_mut().zip(&b[k * dout..(k + 1) * dout]) {
                    *yo += uk * bo;
                }
            }
        }
    }
}

/// An adapter, ready for [`crate::Whistle::set_lora`].
#[derive(Clone, Debug)]
pub struct Lora<'a> {
    /// The adapter's `alpha / rank`. [`crate::Whistle::lora_strength`]
    /// multiplies it.
    pub scale: f32,
    /// Updates by `[site][proj]`; `None` leaves a projection as it is.
    pub weights: [[Option<LoraWeight<'a>>; 5]; 3],
    /// Embedding rows `[pieces + languages][d_model]`: the added text pieces,
    /// then the added language tags.
    pub rows: &'a [f32],
    /// How many of `rows` are text pieces.
    pub pieces: usize,
    /// The extended tokenizer (base pieces, added pieces, base language tags,
    /// added tags), as built by [`crate::tokenizer::extend`].
    pub tokenizer: &'a [u8],
}

impl<'a> Lora<'a> {
    pub fn weight(&self, site: Site, proj: Proj) -> Option<&LoraWeight<'a>> {
        self.weights[site as usize][proj as usize].as_ref()
    }
}
