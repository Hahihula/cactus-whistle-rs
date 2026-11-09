// Started as a port of the C engine's Whistle forward pass in andrisgauracs/whistle-esp32
// (Apache-2.0), and changed since. See README.md, Acknowledgements.

//! Whistle speech recognition over a `.cact` v4 blob.
//!
//! ```text
//! audio --log-mel--> [frames, 80] --stem--> [frames/8, 512]
//!       --encoder (8 non-causal blocks, mHC lanes)--> [frames/8, 512]
//!       --decoder (causal blocks + gated cross attention)--> tokens
//! ```
//!
//! A port of `nd_whistle.c` from andrisgauracs/whistle-esp32, which was
//! checked tensor by tensor against the official engine. Function names and
//! evaluation order follow the C code so the two can be diffed directly.

use alloc::rc::Rc;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use crate::cact::{Cact, CactError, DT_CQ, DT_FP16, DT_FP32, DT_RAW, Tensor, rd_u32};
use crate::math::{expf, f16, fma, logf, rms_unit, sigm, silu, softmax, sqrtf};
use crate::prof;
use crate::quant::{Q8Act, dequant_row, q8_gemm, q8_ok};
use crate::tokenizer::Tokenizer;

/// Languages of the shipped Whistle model, in tag order. A fine-tuned model
/// can add languages: use [`Whistle::lang_code`] and [`Whistle::lang_index`],
/// which read the model's own list.
pub const LANGS: [&str; 7] = ["en", "de", "fr", "es", "it", "nl", "pl"];
/// Number of languages of the shipped model; see [`LANGS`].
pub const N_LANGS: usize = 7;
/// Most languages a model may have (one bit each in [`Whistle::lang_mask`]).
pub const MAX_LANGS: usize = 32;
use crate::ladder::MAX_LAYERS;
use crate::lora::{Lora, Proj, Site};
const SINKHORN: usize = 20;
const EG_HIST: usize = 16;
const EG_SEED: u32 = 0x9E37_79B9;
const EG_PRIME: u32 = 0x0100_0193;
const BOS_ID: u32 = 2;
const EOS_ID: u32 = 1;
// Front end as the C engine (andrisgauracs/whistle-esp32) runs it.
const LOG_EPS: f32 = 6e-7;
const NORM_EPS: f32 = 1e-3;
// Front end as Cactus's porting guide describes the official engine
// (cactuscompute.com/blog/porting-whistle, section 3).
const LOG_EPS_BLOG: f32 = 5.960_464_5e-8; // 2^-24
const NORM_EPS_BLOG: f32 = 1e-5;
const STEM_CHUNK: usize = 8;
/// Beam search: proposals per live beam and the token limit (the guide).
const BEAM_PROPOSALS: usize = 6;
const MAX_TOKENS: usize = 319;
const PI: f32 = core::f32::consts::PI;

// Per-layer tensor offsets, canon order (tools/whistle_cact.py).
const D_NORM_IN: usize = 0;
const D_Q: usize = 1;
const D_K: usize = 2;
const D_V: usize = 3;
const D_QT: usize = 4;
const D_KT: usize = 5;
const D_VT: usize = 6;
const D_QN: usize = 7;
const D_KN: usize = 8;
const D_GATE: usize = 9;
const D_OUT: usize = 10;
const D_POST: usize = 11;
const D_AGATE: usize = 12;
const D_PRE_HADA: usize = 13;
const D_MLP: usize = 14;
const D_N: usize = D_MLP + 13;

const X_NORM: usize = 0;
const X_Q: usize = 1;
const X_K: usize = 2;
const X_V: usize = 3;
const X_GATE: usize = 4;
const X_OUT: usize = 5;
const X_QN: usize = 6;
const X_KN: usize = 7;
const X_POST: usize = 8;
const X_GATE_S: usize = 9;
const X_N: usize = 10;

const E_PRE_HADA0: usize = 0;
const E_NORM_IN: usize = 1;
const E_POST: usize = 2;
const E_CONV_NORM: usize = 3;
const E_CONV_OUT_NORM: usize = 4;
const E_PRE_HADA: usize = 5;
const E_Q: usize = 6;
const E_K: usize = 7;
const E_V: usize = 8;
const E_GATE: usize = 9;
const E_OUT: usize = 10;
const E_QN: usize = 11;
const E_KN: usize = 12;
const E_AGATE: usize = 13;
const E_PW1: usize = 14;
const E_PW2: usize = 15;
const E_DW: usize = 16;
const E_MLP0: usize = 17;
const E_MLP: usize = E_MLP0 + 13;
const E_N: usize = E_MLP + 13;

const M_D1: usize = 0;
const M_D2: usize = 1;
const M_B2: usize = 2;
const M_D3: usize = 3;
const M_D4: usize = 4;
const M_W1A: usize = 5;
const M_W1B: usize = 6;
const M_CV: usize = 11;
const M_CU: usize = 12;

const H_A_PRE: usize = 0;
const H_A_POST: usize = 1;
const H_A_RES: usize = 2;
const H_B_PRE: usize = 3;
const H_B_POST: usize = 4;
const H_B_RES: usize = 5;
const H_PHI_PRE: usize = 6;
const H_PHI_POST: usize = 7;
const H_PHI_RES: usize = 8;

const G_TABLES: usize = 0;
const G_KEY: usize = 1;
const G_VALUE: usize = 2;
const G_TAPS: usize = 3;

const S_W: usize = 0;
const S_DW1: usize = 1;
const S_PW1: usize = 2;
const S_DW2: usize = 3;
const S_PW2: usize = 4;
const S_OUT: usize = 5;

/// Which behaviour to follow (see the README's measurements against Cactus's
/// own engine).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Spec {
    /// As Cactus's porting guide describes the official engine: its front
    /// end, positional signal on the cross-attention memory, 5-beam search
    /// (beams batched). Closest to the official engine: 31.6% WER on the
    /// depth-4 model, official 31.3%. The default.
    Guide,
    /// The guide's front end with greedy decoding: about half the decoding
    /// time of [`Spec::Guide`] on microcontrollers, ~3 WER points worse.
    GuideFast,
    /// As the C engine (andrisgauracs/whistle-esp32) runs it: peak-normalised
    /// front end, positional signal on the encoder input, greedy decoding.
    /// Kept for comparison with that engine.
    CEngine,
}

impl Spec {
    /// The default: [`Spec::Guide`], or [`Spec::GuideFast`] with the `fast`
    /// feature, or [`Spec::CEngine`] with the `c-engine` feature.
    pub const DEFAULT: Spec = if cfg!(feature = "c-engine") {
        Spec::CEngine
    } else if cfg!(feature = "fast") {
        Spec::GuideFast
    } else {
        Spec::Guide
    };

    pub fn name(self) -> &'static str {
        match self {
            Spec::Guide => "guide spec, 5-beam search",
            Spec::GuideFast => "guide spec, greedy (fast)",
            Spec::CEngine => "C engine behaviour, greedy",
        }
    }
}

/// The language tags of a tokenizer with `vocab` pieces: the trailing
/// `<|xx|>` pieces. Returns the first tag's id and the number of tags.
fn lang_tags(tok: &Tokenizer, vocab: usize) -> Option<(u32, usize)> {
    if tok.len() != vocab {
        return None;
    }
    let is_tag = |id: usize| {
        let s = tok.piece(id as u32);
        s.len() > 4 && s.starts_with(b"<|") && s.ends_with(b"|>") && core::str::from_utf8(s).is_ok()
    };
    let n = (0..vocab).rev().take_while(|&id| is_tag(id)).count();
    if n == 0 || n > MAX_LANGS || n >= vocab {
        return None;
    }
    Some(((vocab - n) as u32, n))
}

/// Where a token's embedding row is when an adapter extends the vocabulary.
enum Row {
    Base(usize),
    Lora(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    Cact(CactError),
    NotV4Whistle,
    Unsupported,
    Layout,
    Tokenizer,
    BadDepth,
    ClipTooShort,
    BadLanguage,
    /// The adapter doesn't fit the model (shapes, layers or vocabulary).
    BadLora,
}

impl From<CactError> for Error {
    fn from(e: CactError) -> Self {
        Error::Cact(e)
    }
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::Cact(e) => write!(f, "invalid .cact file: {e}"),
            Error::NotV4Whistle => {
                f.write_str("not a Whistle model (.cact v4 with the audio tensors)")
            }
            Error::Unsupported => {
                f.write_str("model uses a configuration this engine doesn't support")
            }
            Error::Layout => f.write_str("model tensors don't have the expected shapes"),
            Error::Tokenizer => f.write_str("model's tokenizer tensor is invalid"),
            Error::BadDepth => {
                f.write_str("decoder depth must be between 2 and the model's layer count")
            }
            Error::ClipTooShort => f.write_str("audio too short to transcribe"),
            Error::BadLanguage => f.write_str("language index is not one of the model's languages"),
            Error::BadLora => f.write_str(
                "adapter doesn't fit this model (fine-tuned from another base, or a different shape)",
            ),
        }
    }
}

impl core::error::Error for Error {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Error::Cact(e) => Some(e),
            _ => None,
        }
    }
}

/// Hadamard MLP weights, expanded from fp16 once per call.
struct MlpW {
    a: [[f32; 256]; 3],
    b: [[f32; 1024]; 3],
    d1: [f32; 512],
    d2: [f32; 512],
    b2: [f32; 512],
    d3: [f32; 512],
    d4: [f32; 512],
    cv: [f32; 512 * 8],
    cu: [f32; 8 * 512],
}

/// Decoder state for one clip: cross-attention K/V and the self-attention
/// caches. Dropped at the end of a decode.
#[derive(Clone)]
pub struct Session {
    enc_t: usize,
    // Shared by every beam: projected once per clip.
    cross_k: Rc<Vec<f32>>, // [layer][nh][T][qk]
    cross_v: Rc<Vec<f32>>, // [layer][nh][T][vd]
    // Per layer, grown one position at a time: a beam clone copies only the
    // positions used so far, not all max_seq (~1.2 MB for 4 layers).
    k_cache: Vec<Vec<f32>>, // [layer][pos][nkv*qk]
    v_cache: Vec<Vec<f32>>, // [layer][pos][nkv*vd]
    qkv_hist: Vec<f32>,     // [layer][taps-1][q|k|v raw]
    eg_hist: Vec<f32>,      // [site][EG_HIST][d]
    hist: [u32; 8],
    sc: Vec<f32>,
}

impl Session {
    /// A placeholder sharing the cross-attention memory, with empty caches.
    fn clone_empty(&self) -> Session {
        Session {
            enc_t: self.enc_t,
            cross_k: self.cross_k.clone(),
            cross_v: self.cross_v.clone(),
            k_cache: Vec::new(),
            v_cache: Vec::new(),
            qkv_hist: Vec::new(),
            eg_hist: Vec::new(),
            hist: [0; 8],
            sc: Vec::new(),
        }
    }
}

pub struct Whistle<'a> {
    pub c: Cact<'a>,
    t: Vec<Tensor>,
    p: Vec<&'a [u8]>,
    /// fp16 payloads as aligned u16 slices (empty for other dtypes).
    h16: Vec<&'a [u16]>,
    pub tok: Tokenizer<'a>,

    pub d: usize,
    nh: usize,
    nkv: usize,
    qk: usize,
    vd: usize,
    lanes: usize,
    taps: usize,
    pub vocab: usize,
    /// Token id of the first language tag; ids below it are text.
    lang_base: u32,
    /// Languages of the model (tags `lang_base..vocab`).
    n_langs: usize,
    max_seq: usize,
    pub dec_layers: usize,
    pub enc_layers: usize,
    n_sites: usize,
    sites: [usize; 4],
    mels: usize,
    n_fft: usize,
    hop: usize,
    rope_inv: [f32; 32],
    fft_cos: [f32; 256],
    fft_sin: [f32; 256],
    hann: [f32; 512],
    /// Symmetric Hann over `win` (÷ (win - 1)), the guide's window.
    hann_sym: [f32; 512],
    fb_lo: [u16; 128],
    fb_hi: [u16; 128],
    perm1: [usize; 512],
    perm2: [usize; 512],

    i_emb: usize,
    i_dec: usize,
    i_dec_mhc: usize,
    i_engram: usize,
    i_dec_final: usize,
    i_cross: usize,
    i_pe_gate: usize,
    i_enc: usize,
    i_enc_mhc: usize,
    i_enc_final: usize,
    i_stem: usize,
    i_melfb: usize,

    pub depth: usize,
    active: [usize; MAX_LAYERS],

    /// Language of the last decode (index into the model's languages; see
    /// [`Whistle::lang_code`]).
    pub lang: usize,
    /// The detection's scores for each language (the first
    /// [`Whistle::n_langs`] entries).
    pub lang_logit: [f32; MAX_LANGS],
    /// Mean log-prob of the last transcript.
    pub score: f32,
    /// Bit i enables language i for detection; 0 enables all.
    pub lang_mask: u32,
    /// Front end per the porting guide (99th-percentile RMS gain to 0.1,
    /// symmetric Hann, log floor 2^-24, std epsilon 1e-5) instead of the C
    /// engine's (peak gain to 1, periodic Hann, 6e-7, 1e-3).
    pub front_blog: bool,
    /// Add the gated sinusoid to the cross-attention memory (the guide:
    /// sines in the first half of the width, cosines in the second) instead
    /// of to the encoder input interleaved, as the C engine does.
    pub pe_memory: bool,
    /// Beam width: 1 is greedy (the C engine), 5 is the official engine.
    pub beams: usize,

    i_tok: usize,
    /// The model's own vocabulary, before an adapter extends it.
    base_vocab: usize,
    base_lang_base: u32,
    base_n_langs: usize,
    lora: Option<&'a Lora<'a>>,
    /// Adapter layer of each decoder layer in the file (a trimmed model keeps
    /// only some of the layers the adapter was trained on).
    lora_dec: [usize; MAX_LAYERS],
    /// Multiplies the adapter's update (not its vocabulary rows): 0 is the
    /// base model, 1 the adapter as trained.
    pub lora_strength: f32,
}

impl<'a> Whistle<'a> {
    // ------------------------------------------------------------ helpers

    /// Element k of a small fp16 or fp32 tensor.
    #[inline]
    fn hv(&self, i: usize, k: usize) -> f32 {
        let h = self.h16[i];
        if !h.is_empty() {
            return f16(h[k]);
        }
        let b = self.p[i];
        if self.t[i].dtype == DT_FP32 {
            f32::from_bits(rd_u32(b, k * 4))
        } else {
            f16(u16::from_le_bytes([b[2 * k], b[2 * k + 1]]))
        }
    }

    /// The first `out.len()` elements of a small fp16/fp32 tensor. The fp16
    /// case is one tight loop over an aligned u16 view: element by element
    /// through `hv` it cost ~115 cycles per value on the LX7, and loading an
    /// MLP's 14.6K values happens ~80 times per transcription.
    fn hvec(&self, i: usize, out: &mut [f32]) {
        let h = self.h16[i];
        if h.len() >= out.len() {
            for (o, &v) in out.iter_mut().zip(h) {
                *o = f16(v);
            }
        } else {
            for (k, v) in out.iter_mut().enumerate() {
                *v = self.hv(i, k);
            }
        }
    }

    /// ZCRMSNorm: (1 + scale) * x / rms(x).
    fn zc(&self, x: &[f32], si: usize, out: &mut [f32]) {
        rms_unit(x, out);
        let h = self.h16[si];
        if h.len() >= out.len() {
            for (v, &w) in out.iter_mut().zip(h) {
                *v *= 1.0 + f16(w);
            }
        } else {
            for (k, v) in out.iter_mut().enumerate() {
                *v *= 1.0 + self.hv(si, k);
            }
        }
    }

    /// Half-split rotary on one head at position `pos`.
    fn rope(&self, z: &mut [f32], pos: usize) {
        let half = z.len() / 2;
        for i in 0..half {
            let a = pos as f32 * self.rope_inv[i];
            let (c, s) = (libm::cosf(a), libm::sinf(a));
            let (z1, z2) = (z[i], z[i + half]);
            z[i] = z1 * c - z2 * s;
            z[i + half] = z2 * c + z1 * s;
        }
    }

    // ------------------------------------------------------------ matmuls

    /// Prepare an input for every tensor that reduces over it.
    fn act(&self, ti: usize, x: &[f32], rows: usize) -> Q8Act {
        let _s = prof::span(prof::LIN);
        let _p = prof::span(prof::PREP);
        Q8Act::prepare(&self.t[ti], x, rows)
    }

    fn gemm_rows(&self, ti: usize, a: &Q8Act, r0: usize, nrows: usize, y: &mut [f32]) {
        let _s = prof::span(prof::LIN);
        q8_gemm(&self.c, &self.t[ti], self.p[ti], a, r0, nrows, y);
    }

    fn gemm(&self, ti: usize, a: &Q8Act, y: &mut [f32]) {
        self.gemm_rows(ti, a, 0, self.t[ti].shape[0] as usize, y);
    }

    /// Y[T][out] = X[T][in] @ W^T.
    fn lin(&self, ti: usize, x: &[f32], rows: usize) -> Vec<f32> {
        let a = self.act(ti, x, rows);
        let mut y = vec![0f32; rows * self.t[ti].shape[0] as usize];
        self.gemm(ti, &a, &mut y);
        y
    }

    // ---------------------------------------------------- Hadamard MLP

    fn mlp_load(&self, base: usize) -> alloc::boxed::Box<MlpW> {
        let _s = prof::span(prof::MLP_LOAD);
        let mut m = alloc::boxed::Box::new(MlpW {
            a: [[0.0; 256]; 3],
            b: [[0.0; 1024]; 3],
            d1: [0.0; 512],
            d2: [0.0; 512],
            b2: [0.0; 512],
            d3: [0.0; 512],
            d4: [0.0; 512],
            cv: [0.0; 512 * 8],
            cu: [0.0; 8 * 512],
        });
        for s in 0..3 {
            self.hvec(base + M_W1A + 2 * s, &mut m.a[s]);
            self.hvec(base + M_W1B + 2 * s, &mut m.b[s]);
        }
        self.hvec(base + M_D1, &mut m.d1);
        self.hvec(base + M_D2, &mut m.d2);
        self.hvec(base + M_B2, &mut m.b2);
        self.hvec(base + M_D3, &mut m.d3);
        self.hvec(base + M_D4, &mut m.d4);
        self.hvec(base + M_CV, &mut m.cv);
        self.hvec(base + M_CU, &mut m.cu);
        m
    }

    fn mlp_row(&self, m: &MlpW, x: &[f32], out: &mut [f32]) {
        let _s = prof::span(prof::KRON);
        let mut c8 = [0f32; 8];
        let mut cond = [1f32; 512];
        let mut z = [0f32; 512];
        let mut y = [0f32; 512];
        for k in 0..512 {
            let xk = x[k];
            let cv = &m.cv[k * 8..k * 8 + 8];
            for r in 0..8 {
                c8[r] = fma(xk, cv[r], c8[r]);
            }
        }
        softmax(&mut c8);
        for r in 0..8 {
            let cr = c8[r];
            let cu = &m.cu[r * 512..r * 512 + 512];
            for k in 0..512 {
                cond[k] = fma(cr, cu[k], cond[k]);
            }
        }
        for k in 0..512 {
            z[k] = m.d1[k] * x[k];
        }
        kron(&z, &m.a[0], &m.b[0], &mut y);
        for k in 0..512 {
            z[k] = silu(m.d2[k] * cond[k] * y[self.perm1[k]] + m.b2[k]);
        }
        kron(&z, &m.a[1], &m.b[1], &mut y);
        for k in 0..512 {
            z[k] = m.d3[k] * y[self.perm2[k]];
        }
        kron(&z, &m.a[2], &m.b[2], &mut y);
        for k in 0..512 {
            out[k] = m.d4[k] * y[k];
        }
    }

    /// x[T][d] += scale * MLP(zc(x, norm)).
    fn mlp_residual(&self, x: &mut [f32], rows: usize, norm_i: usize, base: usize, scale: f32) {
        let _s = prof::span(prof::MLP);
        let m = self.mlp_load(base);
        let d = self.d;
        let xs = RowsPtr(x.as_mut_ptr());
        let kernel = |ra: usize, rb: usize| {
            let mut n = [0f32; 512];
            let mut o = [0f32; 512];
            for t in ra..rb {
                // SAFETY: each worker owns rows [ra, rb) of x.
                let row = unsafe { xs.row(t, d) };
                self.zc(row, norm_i, &mut n[..d]);
                self.mlp_row(&m, &n, &mut o);
                for k in 0..d {
                    row[k] += scale * o[k];
                }
            }
        };
        crate::par::rows(rows, 1, &kernel);
    }

    // ---------------------------------------------------------------- mHC

    /// One mHC step over T rows: mix lanes into u, run the block, scatter
    /// back. `layer` indexes the parameters; `slot` (position in the active
    /// stack) picks the lane offsets.
    fn mhc_step(
        &self,
        mhc: usize,
        layer: usize,
        slot: usize,
        lanes: &mut [f32],
        rows: usize,
        block: &mut dyn FnMut(&Self, usize, &mut [f32], usize),
    ) {
        let span = prof::span(prof::MHC);
        let n = self.lanes;
        let d = self.d;
        let nd = n * d;
        let mut nx = vec![0f32; rows * nd];
        let mut pre = vec![0f32; rows * n];
        let mut post = vec![0f32; rows * n];
        let mut res = vec![0f32; rows * n * n];
        let mut u = vec![0f32; rows * d];
        let a_pre = self.hv(mhc + H_A_PRE, layer);
        let a_post = self.hv(mhc + H_A_POST, layer);
        let a_res = self.hv(mhc + H_A_RES, layer);
        let hot = slot % n;

        for t in 0..rows {
            rms_unit(&lanes[t * nd..(t + 1) * nd], &mut nx[t * nd..(t + 1) * nd]);
        }
        {
            let a = self.act(mhc + H_PHI_PRE, &nx, rows);
            self.gemm_rows(mhc + H_PHI_PRE, &a, layer * n, n, &mut pre);
            self.gemm_rows(mhc + H_PHI_POST, &a, layer * n, n, &mut post);
            self.gemm_rows(mhc + H_PHI_RES, &a, layer * n * n, n * n, &mut res);
        }
        let mx = prof::span(prof::MIX);
        for t in 0..rows {
            let l = &lanes[t * nd..(t + 1) * nd];
            let ut = &mut u[t * d..(t + 1) * d];
            for j in 0..n {
                let off = if j == hot { 4.0 } else { -4.0 };
                let h = sigm(a_pre * pre[t * n + j] + self.hv(mhc + H_B_PRE, layer * n + j) + off);
                for c in 0..d {
                    ut[c] = fma(h, l[j * d + c], ut[c]);
                }
            }
        }
        let u0 = u.clone();
        drop(mx);
        drop(span);
        block(self, layer, &mut u, rows);
        let _s = prof::span(prof::MHC);

        // One frame per work item: its rows of res and lanes are its own.
        let (rp, lp) = (RowsPtr(res.as_mut_ptr()), RowsPtr(lanes.as_mut_ptr()));
        let (post, u, u0) = (&post[..], &u[..], &u0[..]);
        let kernel = |ta: usize, tb: usize| {
            let mut mix = vec![0f32; nd];
            for t in ta..tb {
                // SAFETY: rows t of res and lanes belong to this work item.
                let (r, l) = unsafe { (rp.row(t, n * n), lp.row(t, nd)) };
                for (k, v) in r.iter_mut().enumerate() {
                    *v = a_res * *v + self.hv(mhc + H_B_RES, layer * n * n + k);
                }
                let sk = prof::span(prof::SINKHORN);
                sinkhorn(r, n);
                drop(sk);
                let _mx = prof::span(prof::MIX);
                for j in 0..n {
                    let off = if j == hot { 0.0 } else { -4.0 };
                    let hp = 2.0
                        * sigm(
                            a_post * post[t * n + j] + self.hv(mhc + H_B_POST, layer * n + j) + off,
                        );
                    for c in 0..d {
                        let mut s = 0f32;
                        for k in 0..n {
                            s = fma(r[j * n + k], l[k * d + c], s);
                        }
                        mix[j * d + c] = fma(hp, u[t * d + c] - u0[t * d + c], s);
                    }
                }
                l.copy_from_slice(&mix);
            }
        };
        crate::par::rows(rows, 1, &kernel);
    }

    fn lanes_out(&self, lanes: &[f32], rows: usize, norm_i: usize, out: &mut [f32]) {
        let (n, d) = (self.lanes, self.d);
        let mut m = [0f32; 512];
        for t in 0..rows {
            for c in 0..d {
                let mut s = 0f32;
                for j in 0..n {
                    s += lanes[(t * n + j) * d + c];
                }
                m[c] = s / n as f32;
            }
            self.zc(&m[..d], norm_i, &mut out[t * d..(t + 1) * d]);
        }
    }

    // ---------------------------------------------------------- front end

    fn fft512(&self, re: &mut [f32; 512], im: &mut [f32; 512]) {
        let n = 512usize;
        let mut j = 0usize;
        for i in 1..n {
            let mut bit = n >> 1;
            while j & bit != 0 {
                j ^= bit;
                bit >>= 1;
            }
            j ^= bit;
            if i < j {
                re.swap(i, j);
                im.swap(i, j);
            }
        }
        let mut len = 2;
        while len <= n {
            let step = n / len;
            let h = len / 2;
            let mut i = 0;
            while i < n {
                for j in 0..h {
                    let (wr, wi) = (self.fft_cos[j * step], self.fft_sin[j * step]);
                    let (ur, ui) = (re[i + j], im[i + j]);
                    let vr = re[i + j + h] * wr - im[i + j + h] * wi;
                    let vi = re[i + j + h] * wi + im[i + j + h] * wr;
                    re[i + j] = ur + vr;
                    im[i + j] = ui + vi;
                    re[i + j + h] = ur - vr;
                    im[i + j + h] = ui - vi;
                }
                i += len;
            }
            len <<= 1;
        }
    }

    /// Normalised log-mel features, [n / hop][mels].
    pub fn logmel(&self, x: &[f32]) -> Vec<f32> {
        let n = x.len();
        let m_frames = n / self.hop;
        let nb = self.n_fft / 2 + 1;
        let mels = self.mels;
        let fb = self.p[self.i_melfb];
        let mut out = vec![0f32; m_frames * mels];
        let blog = self.front_blog;
        let gain = if blog {
            level_gain_blog(x)
        } else {
            peak_gain(x)
        };
        let hann = if blog { &self.hann_sym } else { &self.hann };
        let (log_eps, norm_eps) = if blog {
            (LOG_EPS_BLOG, NORM_EPS_BLOG)
        } else {
            (LOG_EPS, NORM_EPS)
        };
        let mut re = [0f32; 512];
        let mut im = [0f32; 512];
        let mut pw = [0f32; 257];
        for t in 0..m_frames {
            for k in 0..self.n_fft {
                let s = (t * self.hop + k) as isize - (self.n_fft / 2) as isize;
                re[k] = if s >= 0 && (s as usize) < n {
                    sample(x[s as usize], blog) * gain * hann[k]
                } else {
                    0.0
                };
                im[k] = 0.0;
            }
            self.fft512(&mut re, &mut im);
            for k in 0..nb {
                pw[k] = re[k] * re[k] + im[k] * im[k];
            }
            for b in 0..mels {
                let mut s = 0f32;
                for k in self.fb_lo[b] as usize..self.fb_hi[b] as usize {
                    s += pw[k] * f32::from_bits(rd_u32(fb, (k * mels + b) * 4));
                }
                out[t * mels + b] = logf(s + log_eps);
            }
        }
        // Normalise each mel channel over the clip (sample std, ddof = 1).
        for b in 0..mels {
            let mut s = 0f32;
            for t in 0..m_frames {
                s += out[t * mels + b];
            }
            s /= m_frames.max(1) as f32;
            let mut q = 0f32;
            for t in 0..m_frames {
                let dd = out[t * mels + b] - s;
                q += dd * dd;
            }
            let sd = if m_frames > 1 {
                sqrtf(q / (m_frames - 1) as f32)
            } else {
                0.0
            };
            for t in 0..m_frames {
                out[t * mels + b] = (out[t * mels + b] - s) / (sd + norm_eps);
            }
        }
        out
    }

    // --------------------------------------------------------------- stem

    /// [M][mels] log-mel -> [T][d], in chunks of STEM_CHUNK encoder frames so
    /// the first stage never has to exist for the whole clip at once.
    fn stem(&self, lm: &[f32], m_frames: usize, out: &mut [f32]) -> usize {
        let c_ch = self.t[self.i_stem + S_W].shape[1] as usize; // 128
        let f0 = self.mels;
        let f1 = f0.div_ceil(2);
        let f2 = f1.div_ceil(2);
        let f3 = f2.div_ceil(2);
        let t1 = m_frames.div_ceil(2);
        let t2 = t1.div_ceil(2);
        let t3 = t2.div_ceil(2);
        let rows1 = 4 * STEM_CHUNK + 4;
        let mut k0 = vec![0f32; 9 * c_ch];
        let mut k1 = vec![0f32; 9 * c_ch];
        let mut k2 = vec![0f32; 9 * c_ch];
        let mut b1 = vec![0f32; rows1 * f1 * c_ch];
        let mut b2 = vec![0f32; rows1 * f1 * c_ch];
        let mut flat = vec![0f32; STEM_CHUNK * c_ch * f3];
        self.hvec(self.i_stem + S_W, &mut k0);
        self.hvec(self.i_stem + S_DW1, &mut k1);
        self.hvec(self.i_stem + S_DW2, &mut k2);

        let mut a = 0;
        while a < t3 {
            let b = (a + STEM_CHUNK).min(t3);
            let r2a = if a > 0 { 2 * a - 1 } else { 0 };
            let r2b = (2 * b).min(t2);
            let r1a = if r2a > 0 { 2 * r2a - 1 } else { 0 };
            let r1b = (2 * r2b).min(t1);
            let (n1, n2, n3) = (r1b - r1a, r2b - r2a, b - a);

            let cv = prof::span(prof::CONV);
            conv_rows(&k0, lm, 0, m_frames, f0, 1, c_ch, r1a, n1, &mut b1, true);
            conv_rows(&k1, &b1, r1a, t1, f1, c_ch, c_ch, r2a, n2, &mut b2, false);
            drop(cv);
            let mut p1 = self.lin(self.i_stem + S_PW1, &b2[..n2 * f2 * c_ch], n2 * f2);
            let sl = prof::span(prof::SILU);
            p1.iter_mut().for_each(|v| *v = silu(*v));
            drop(sl);
            let cv = prof::span(prof::CONV);
            conv_rows(&k2, &p1, r2a, t2, f2, c_ch, c_ch, a, n3, &mut b2, false);
            drop(cv);
            let mut p2 = self.lin(self.i_stem + S_PW2, &b2[..n3 * f3 * c_ch], n3 * f3);
            let sl = prof::span(prof::SILU);
            p2.iter_mut().for_each(|v| *v = silu(*v));
            drop(sl);
            for t in 0..n3 {
                for f in 0..f3 {
                    for c in 0..c_ch {
                        flat[t * c_ch * f3 + c * f3 + f] = p2[(t * f3 + f) * c_ch + c];
                    }
                }
            }
            let o = self.lin(self.i_stem + S_OUT, &flat[..n3 * c_ch * f3], n3);
            out[a * self.d..(a + n3) * self.d].copy_from_slice(&o);
            a += STEM_CHUNK;
        }

        // Sinusoidal position, gated by sigmoid(pe_gate): here on the encoder
        // input as the C engine does, or later on the cross-attention memory
        // (pe_memory, the guide).
        let g = sigm(self.hv(self.i_pe_gate, 0));
        let d = self.d;
        for t in 0..if self.pe_memory { 0 } else { t3 } {
            let mut c = 0;
            while c < d {
                let div = libm::expf(-(c as f32) * logf(10000.0) / d as f32);
                out[t * d + c] += g * libm::sinf(t as f32 * div);
                out[t * d + c + 1] += g * libm::cosf(t as f32 * div);
                c += 2;
            }
        }
        t3
    }

    // ------------------------------------------------------------ encoder

    fn enc_attention(&self, l: usize, xn: &[f32], rows: usize, out: &mut [f32]) {
        let (nh, nkv, qk, vd) = (self.nh, self.nkv, self.qk, self.vd);
        let rep = nh / nkv;
        let mut q = vec![0f32; rows * nh * qk];
        let mut k = vec![0f32; rows * nkv * qk];
        let mut v = vec![0f32; rows * nkv * vd];
        let mut o = vec![0f32; rows * nh * vd];
        let mut g = vec![0f32; rows * nh * vd];
        let mut tmp = [0f32; 64];
        let scale = 1.0 / sqrtf(qk as f32);

        let layer = (l - self.i_enc) / E_N;
        let a = self.act(l + E_Q, xn, rows);
        self.gemm(l + E_Q, &a, &mut q);
        self.gemm(l + E_K, &a, &mut k);
        self.gemm(l + E_V, &a, &mut v);
        self.lora_add(Site::Encoder, Proj::Q, layer, xn, rows, &mut q);
        self.lora_add(Site::Encoder, Proj::K, layer, xn, rows, &mut k);
        self.lora_add(Site::Encoder, Proj::V, layer, xn, rows, &mut v);
        for t in 0..rows {
            for h in 0..nh {
                let z = &mut q[(t * nh + h) * qk..(t * nh + h + 1) * qk];
                self.zc(z, l + E_QN, &mut tmp[..qk]);
                z.copy_from_slice(&tmp[..qk]);
                self.rope(z, t);
            }
            for h in 0..nkv {
                let z = &mut k[(t * nkv + h) * qk..(t * nkv + h + 1) * qk];
                self.zc(z, l + E_KN, &mut tmp[..qk]);
                z.copy_from_slice(&tmp[..qk]);
                self.rope(z, t);
            }
        }
        let attn = prof::span(prof::ATTN);
        {
            // One query frame per work item; each writes only its row of o.
            let (q, k, v) = (&q[..], &k[..], &v[..]);
            let op = RowsPtr(o.as_mut_ptr());
            let kernel = |ta: usize, tb: usize| {
                let mut sc = vec![0f32; rows];
                for t in ta..tb {
                    // SAFETY: row t of o is owned by this work item.
                    let ot = unsafe { op.row(t, nh * vd) };
                    for h in 0..nh {
                        let qh = &q[(t * nh + h) * qk..(t * nh + h + 1) * qk];
                        let kh = h / rep;
                        for s in 0..rows {
                            let ks = &k[(s * nkv + kh) * qk..(s * nkv + kh + 1) * qk];
                            let mut acc = 0f32;
                            for e in 0..qk {
                                acc = fma(qh[e], ks[e], acc);
                            }
                            sc[s] = acc * scale;
                        }
                        softmax(&mut sc);
                        let oh = &mut ot[h * vd..(h + 1) * vd];
                        oh.fill(0.0);
                        for s in 0..rows {
                            let vs = &v[(s * nkv + kh) * vd..(s * nkv + kh + 1) * vd];
                            for e in 0..vd {
                                oh[e] = fma(sc[s], vs[e], oh[e]);
                            }
                        }
                    }
                }
            };
            crate::par::rows(rows, 1, &kernel);
        }
        drop(attn);
        self.gemm(l + E_GATE, &a, &mut g);
        self.lora_add(Site::Encoder, Proj::Gate, layer, xn, rows, &mut g);
        for (ov, gv) in o.iter_mut().zip(g.iter()) {
            *ov *= sigm(*gv);
        }
        let mut y = self.lin(l + E_OUT, &o, rows);
        self.lora_add(Site::Encoder, Proj::Out, layer, &o, rows, &mut y);
        out[..y.len()].copy_from_slice(&y);
    }

    fn enc_block(&self, layer: usize, x: &mut [f32], rows: usize) {
        let l = self.i_enc + layer * E_N;
        let d = self.d;
        let mut xn = vec![0f32; rows * d];
        let mut o = vec![0f32; rows * 2 * d];
        let mut g = vec![0f32; rows * d];
        let mut tmp = [0f32; 512];
        let mut dw = vec![0f32; 9 * d];

        self.mlp_residual(x, rows, l + E_PRE_HADA0, l + E_MLP0, 0.5);

        for t in 0..rows {
            self.zc(
                &x[t * d..(t + 1) * d],
                l + E_NORM_IN,
                &mut xn[t * d..(t + 1) * d],
            );
        }
        self.enc_attention(l, &xn, rows, &mut o);
        let gate = sigm(self.hv(l + E_AGATE, 0));
        for t in 0..rows {
            self.zc(&o[t * d..(t + 1) * d], l + E_POST, &mut tmp[..d]);
            for c in 0..d {
                x[t * d + c] += gate * tmp[c];
            }
        }

        // Conv module: pw1 -> GLU -> 9-tap depthwise -> norm -> SiLU -> pw2.
        for t in 0..rows {
            self.zc(
                &x[t * d..(t + 1) * d],
                l + E_CONV_NORM,
                &mut xn[t * d..(t + 1) * d],
            );
        }
        let o = self.lin(l + E_PW1, &xn, rows); // [T][2d]
        for t in 0..rows {
            for c in 0..d {
                g[t * d + c] = o[t * 2 * d + c] * sigm(o[t * 2 * d + d + c]);
            }
        }
        self.hvec(l + E_DW, &mut dw);
        for t in 0..rows {
            for c in 0..d {
                let mut s = 0f32;
                for j in 0..9 {
                    let tt = t as isize + j as isize - 4;
                    if tt >= 0 && (tt as usize) < rows {
                        s = fma(dw[j * d + c], g[tt as usize * d + c], s);
                    }
                }
                xn[t * d + c] = s;
            }
        }
        for t in 0..rows {
            self.zc(&xn[t * d..(t + 1) * d], l + E_CONV_OUT_NORM, &mut tmp[..d]);
            for c in 0..d {
                xn[t * d + c] = silu(tmp[c]);
            }
        }
        let o = self.lin(l + E_PW2, &xn, rows);
        for (xv, ov) in x.iter_mut().zip(o.iter()) {
            *xv += *ov;
        }

        self.mlp_residual(x, rows, l + E_PRE_HADA, l + E_MLP, 0.5);
    }

    /// Encoder frames for `n` samples (one per 80 ms).
    pub fn frames(&self, n: usize) -> usize {
        let m = n / self.hop;
        m.div_ceil(2).div_ceil(2).div_ceil(2)
    }

    /// 16 kHz mono float samples -> encoder output, [frames][d].
    pub fn encode(&self, x: &[f32]) -> Result<(Vec<f32>, usize), Error> {
        let m_frames = x.len() / self.hop;
        if m_frames == 0 {
            return Err(Error::ClipTooShort);
        }
        let sp = prof::span(prof::MEL);
        let lm = self.logmel(x);
        drop(sp);
        let sp = prof::span(prof::STEM);
        let mut s = vec![0f32; self.frames(x.len()) * self.d];
        let t = self.stem(&lm, m_frames, &mut s);
        drop(lm);
        drop(sp);
        let _sp = prof::span(prof::ENCODER);
        let (n, d) = (self.lanes, self.d);
        let mut lanes = vec![0f32; t * n * d];
        lanes_init(&s, t, n, d, &mut lanes);
        drop(s);
        for i in 0..self.enc_layers {
            self.mhc_step(
                self.i_enc_mhc,
                i,
                i,
                &mut lanes,
                t,
                &mut |w, layer, u, rows| w.enc_block(layer, u, rows),
            );
        }
        let mut out = vec![0f32; t * d];
        self.lanes_out(&lanes, t, self.i_enc_final, &mut out);
        Ok((out, t))
    }

    // ------------------------------------------------------------ decoder

    /// Engram lookups for a batch of beams at `pos`. Per beam: the n-gram
    /// hash over its own token history and the taps over its own value
    /// history; the key/value projections run as one batched matmul.
    /// `ek`/`ev` hold [beam][site][512].
    fn engram_batch(
        &self,
        toks: &[u32],
        pos: usize,
        ss: &mut [Session],
        ek: &mut [f32],
        ev: &mut [f32],
    ) {
        let _s = prof::span(prof::ENGRAM);
        let bsz = ss.len();
        let h = &self.c.h;
        let heads = (h.engram_tables / h.num_orders) as usize;
        let stride = if h.engram_seed_heads != 0 {
            h.engram_seed_heads as usize
        } else {
            heads
        };
        let sub = h.engram_sub_dim as usize;
        let ntab = h.engram_tables as usize;
        let slots = h.engram_slots as usize;
        let d = self.d;
        let inn = ntab * sub;
        let mut slot = vec![[0usize; 16]; bsz];
        let mut ok = vec![[false; 16]; bsz];
        let mut row = [0f32; 256];

        for (b, s) in ss.iter_mut().enumerate() {
            s.hist[pos & 7] = toks[b];
            let mut col = 0;
            for oi in 0..h.num_orders as usize {
                for hh in 0..heads {
                    let mut acc = EG_SEED.wrapping_mul((oi * stride + hh + 1) as u32);
                    for j in 0..h.orders[oi] as usize {
                        let u = if pos >= j { s.hist[(pos - j) & 7] } else { 0 };
                        acc = (acc ^ u).wrapping_mul(EG_PRIME);
                    }
                    acc ^= acc >> 15;
                    slot[b][col] = acc as usize % slots;
                    ok[b][col] = pos + 1 >= h.orders[oi] as usize;
                    col += 1;
                }
            }
        }
        let mut e = vec![0f32; bsz * inn];
        let mut kv = vec![0f32; bsz * d];
        let mut vraw = vec![0f32; bsz * d];
        for si in 0..self.n_sites {
            let g = self.i_engram + si * 4;
            let tt_t = &self.t[g + G_TABLES];
            for b in 0..bsz {
                for tt in 0..ntab {
                    let dst = &mut e[b * inn + tt * sub..b * inn + (tt + 1) * sub];
                    if ok[b][tt] {
                        dequant_row(
                            &self.c,
                            tt_t,
                            self.p[g + G_TABLES],
                            tt * slots + slot[b][tt],
                            &mut row,
                            dst,
                        );
                    } else {
                        dst.fill(0.0);
                    }
                }
            }
            let a = self.act(g + G_KEY, &e, bsz);
            self.gemm(g + G_KEY, &a, &mut kv);
            self.gemm(g + G_VALUE, &a, &mut vraw);
            for (b, s) in ss.iter_mut().enumerate() {
                let o = b * 4 * 512 + si * 512;
                ek[o..o + d].copy_from_slice(&kv[b * d..(b + 1) * d]);
                let hb = (si * EG_HIST + (pos & (EG_HIST - 1))) * d;
                s.eg_hist[hb..hb + d].copy_from_slice(&vraw[b * d..(b + 1) * d]);
                let evs = &mut ev[o..o + d];
                evs.fill(0.0);
                for j in 0..h.engram_conv_taps as usize {
                    let sh = j * h.engram_dilation as usize;
                    if sh > pos {
                        break;
                    }
                    let src = (si * EG_HIST + ((pos - sh) & (EG_HIST - 1))) * d;
                    for c in 0..d {
                        evs[c] = fma(self.hv(g + G_TAPS, j * d + c), s.eg_hist[src + c], evs[c]);
                    }
                }
            }
        }
    }

    /// One decoder block for a batch of beams, x = [beam][d]. The matmuls run
    /// once for all beams (each weight row unpacked once); the Q/K/V conv
    /// history, the self-attention cache and the attention itself are per
    /// beam. Each row is quantised and dotted on its own, so a beam's result
    /// is bit-identical to running it alone.
    #[allow(clippy::too_many_arguments)]
    fn dec_block_batch(
        &self,
        layer: usize,
        x: &mut [f32],
        ss: &mut [Session],
        pos: usize,
        ek: &[f32],
        ev: &[f32],
    ) {
        let bsz = ss.len();
        let l = self.i_dec + layer * D_N;
        let xi = self.i_cross + layer * X_N;
        let (d, nh, nkv, qk, vd) = (self.d, self.nh, self.nkv, self.qk, self.vd);
        let rep = nh / nkv;
        let (nq, nk, nv) = (nh * qk, nkv * qk, nkv * vd);
        let nraw = nq + nk + nv;
        let no = nh * vd;
        let mut tmp = [0f32; 64];
        let mut tmpd = [0f32; 512];
        let scale = 1.0 / sqrtf(qk as f32);
        // This position's slot in each beam's cache (grown on first use; the
        // language detection writes position 0 before the pass rewrites it).
        for s in ss.iter_mut() {
            if s.k_cache[layer].len() < (pos + 1) * nk {
                s.k_cache[layer].resize((pos + 1) * nk, 0.0);
            }
            if s.v_cache[layer].len() < (pos + 1) * nv {
                s.v_cache[layer].resize((pos + 1) * nv, 0.0);
            }
        }

        // Engram injection at its sites.
        for si in 0..self.n_sites {
            if self.sites[si] != layer {
                continue;
            }
            for b in 0..bsz {
                let xb = &mut x[b * d..(b + 1) * d];
                let eo = b * 4 * 512 + si * 512;
                let mut a = [0f32; 512];
                let mut bb = [0f32; 512];
                rms_unit(xb, &mut a[..d]);
                rms_unit(&ek[eo..eo + d], &mut bb[..d]);
                let mut dot = 0f32;
                for c in 0..d {
                    dot += a[c] * bb[c];
                }
                let alpha = sigm(dot / sqrtf(d as f32));
                for c in 0..d {
                    xb[c] += alpha * ev[eo + c];
                }
            }
        }

        // Self attention with a causal 3-tap conv on Q, K, V.
        let mut xn = vec![0f32; bsz * d];
        for b in 0..bsz {
            self.zc(
                &x[b * d..(b + 1) * d],
                l + D_NORM_IN,
                &mut xn[b * d..(b + 1) * d],
            );
        }
        let mut raw = vec![0f32; bsz * nraw];
        let mut gt = vec![0f32; bsz * no];
        {
            let a = self.act(l + D_Q, &xn, bsz);
            let (mut tq, mut tk, mut tv) = (
                vec![0f32; bsz * nq],
                vec![0f32; bsz * nk],
                vec![0f32; bsz * nv],
            );
            self.gemm(l + D_Q, &a, &mut tq);
            self.gemm(l + D_K, &a, &mut tk);
            self.gemm(l + D_V, &a, &mut tv);
            self.gemm(l + D_GATE, &a, &mut gt);
            // Before the Q/K/V conv: the adapter updates the projection kernels.
            self.lora_add(Site::DecoderSelf, Proj::Q, layer, &xn, bsz, &mut tq);
            self.lora_add(Site::DecoderSelf, Proj::K, layer, &xn, bsz, &mut tk);
            self.lora_add(Site::DecoderSelf, Proj::V, layer, &xn, bsz, &mut tv);
            self.lora_add(Site::DecoderSelf, Proj::Gate, layer, &xn, bsz, &mut gt);
            for b in 0..bsz {
                let r = &mut raw[b * nraw..(b + 1) * nraw];
                r[..nq].copy_from_slice(&tq[b * nq..(b + 1) * nq]);
                r[nq..nq + nk].copy_from_slice(&tk[b * nk..(b + 1) * nk]);
                r[nq + nk..].copy_from_slice(&tv[b * nv..(b + 1) * nv]);
            }
        }
        let mut o = vec![0f32; bsz * no];
        let mut q = [0f32; 384];
        for (b, s) in ss.iter_mut().enumerate() {
            let rawb = &mut raw[b * nraw..(b + 1) * nraw];
            let hist0 = layer * (self.taps - 1) * nraw;
            let mut conv = [0f32; 384 + 96 + 128];
            for e in 0..nraw {
                let (ti, ci, width) = if e < nq {
                    (l + D_QT, e, nq)
                } else if e < nq + nk {
                    (l + D_KT, e - nq, nk)
                } else {
                    (l + D_VT, e - nq - nk, nv)
                };
                let mut acc = self.hv(ti, ci) * rawb[e];
                for j in 1..self.taps {
                    if pos >= j {
                        acc += self.hv(ti, j * width + ci)
                            * s.qkv_hist[hist0 + ((pos - j) % (self.taps - 1)) * nraw + e];
                    }
                }
                conv[e] = acc;
            }
            let hp = hist0 + (pos % (self.taps - 1)) * nraw;
            s.qkv_hist[hp..hp + nraw].copy_from_slice(&rawb[..nraw]);
            rawb.copy_from_slice(&conv[..nraw]);
            for h in 0..nh {
                self.zc(
                    &rawb[h * qk..(h + 1) * qk],
                    l + D_QN,
                    &mut q[h * qk..(h + 1) * qk],
                );
                self.rope(&mut q[h * qk..(h + 1) * qk], pos);
            }
            for h in 0..nkv {
                self.zc(
                    &rawb[nq + h * qk..nq + (h + 1) * qk],
                    l + D_KN,
                    &mut tmp[..qk],
                );
                self.rope(&mut tmp[..qk], pos);
                let kc = pos * nk + h * qk;
                s.k_cache[layer][kc..kc + qk].copy_from_slice(&tmp[..qk]);
            }
            let vc = pos * nv;
            s.v_cache[layer][vc..vc + nv].copy_from_slice(&rawb[nq + nk..nraw]);
            let attn = prof::span(prof::ATTN);
            let ob = &mut o[b * no..(b + 1) * no];
            for h in 0..nh {
                let kh = h / rep;
                for si in 0..=pos {
                    let ks = si * nk + kh * qk;
                    let mut acc = 0f32;
                    for e in 0..qk {
                        acc = fma(q[h * qk + e], s.k_cache[layer][ks + e], acc);
                    }
                    s.sc[si] = acc * scale;
                }
                softmax(&mut s.sc[..=pos]);
                for e in 0..vd {
                    let mut acc = 0f32;
                    for si in 0..=pos {
                        acc = fma(s.sc[si], s.v_cache[layer][si * nv + kh * vd + e], acc);
                    }
                    ob[h * vd + e] = acc;
                }
            }
            drop(attn);
        }
        for (ov, gv) in o.iter_mut().zip(gt.iter()) {
            *ov *= sigm(*gv);
        }
        let mut t2 = self.lin(l + D_OUT, &o, bsz);
        self.lora_add(Site::DecoderSelf, Proj::Out, layer, &o, bsz, &mut t2);
        let g = sigm(self.hv(l + D_AGATE, 0));
        for b in 0..bsz {
            self.zc(&t2[b * d..(b + 1) * d], l + D_POST, &mut tmpd[..d]);
            for c in 0..d {
                x[b * d + c] += g * tmpd[c];
            }
        }

        // Gated cross attention over the clip (K, V precomputed per layer,
        // shared by every beam).
        for b in 0..bsz {
            self.zc(
                &x[b * d..(b + 1) * d],
                xi + X_NORM,
                &mut xn[b * d..(b + 1) * d],
            );
        }
        let mut cq = vec![0f32; bsz * nq];
        {
            let a = self.act(xi + X_Q, &xn, bsz);
            self.gemm(xi + X_Q, &a, &mut cq);
            self.gemm(xi + X_GATE, &a, &mut gt);
            self.lora_add(Site::Cross, Proj::Q, layer, &xn, bsz, &mut cq);
            self.lora_add(Site::Cross, Proj::Gate, layer, &xn, bsz, &mut gt);
        }
        for (b, s) in ss.iter_mut().enumerate() {
            let et = s.enc_t;
            let attn = prof::span(prof::ATTN);
            let ob = &mut o[b * no..(b + 1) * no];
            for h in 0..nh {
                let k0 = (layer * nh + h) * et * qk;
                let v0 = (layer * nh + h) * et * vd;
                self.zc(
                    &cq[b * nq + h * qk..b * nq + (h + 1) * qk],
                    xi + X_QN,
                    &mut tmp[..qk],
                );
                for si in 0..et {
                    let mut acc = 0f32;
                    for e in 0..qk {
                        acc = fma(tmp[e], s.cross_k[k0 + si * qk + e], acc);
                    }
                    s.sc[si] = acc * scale;
                }
                softmax(&mut s.sc[..et]);
                for e in 0..vd {
                    let mut acc = 0f32;
                    for si in 0..et {
                        acc = fma(s.sc[si], s.cross_v[v0 + si * vd + e], acc);
                    }
                    ob[h * vd + e] = acc;
                }
            }
            drop(attn);
        }
        for (ov, gv) in o.iter_mut().zip(gt.iter()) {
            *ov *= sigm(*gv);
        }
        let mut t2 = self.lin(xi + X_OUT, &o, bsz);
        self.lora_add(Site::Cross, Proj::Out, layer, &o, bsz, &mut t2);
        let g = sigm(self.hv(xi + X_GATE_S, 0));
        for b in 0..bsz {
            self.zc(&t2[b * d..(b + 1) * d], xi + X_POST, &mut tmpd[..d]);
            for c in 0..d {
                x[b * d + c] += g * tmpd[c];
            }
        }

        self.mlp_residual(&mut x[..bsz * d], bsz, l + D_PRE_HADA, l + D_MLP, 1.0);
    }

    /// Precompute cross-attention K/V for every active layer.
    pub fn session(&self, enc: &[f32], t: usize) -> Session {
        let _s = prof::span(prof::CROSS_KV);
        // The guide: memory = enc + sigmoid(pe_gate) * [sin(t w_i), cos(t w_i)],
        // w_i = exp(-i ln 10000 / 255), i < 256, sines first, cosines second.
        let mem;
        let enc = if self.pe_memory {
            let g = sigm(self.hv(self.i_pe_gate, 0));
            let (d, h) = (self.d, self.d / 2);
            let mut m = enc[..t * d].to_vec();
            for tt in 0..t {
                for i in 0..h {
                    let w = libm::expf(-(i as f32) * logf(10000.0) / (h - 1) as f32);
                    let a = tt as f32 * w;
                    m[tt * d + i] += g * libm::sinf(a);
                    m[tt * d + h + i] += g * libm::cosf(a);
                }
            }
            mem = m;
            &mem[..]
        } else {
            enc
        };
        let (nh, qk, vd, l_all) = (self.nh, self.qk, self.vd, self.dec_layers);
        let nraw = nh * qk + self.nkv * (qk + vd);
        let mut s = Session {
            enc_t: t,
            cross_k: Rc::new(Vec::new()),
            cross_v: Rc::new(Vec::new()),
            k_cache: vec![Vec::new(); l_all],
            v_cache: vec![Vec::new(); l_all],
            qkv_hist: vec![0f32; l_all * (self.taps - 1) * nraw],
            eg_hist: vec![0f32; self.n_sites * EG_HIST * self.d],
            hist: [0; 8],
            sc: vec![0f32; t.max(self.max_seq)],
        };
        let mut tmp = [0f32; 64];
        let mut cross_k = vec![0f32; l_all * nh * t * qk];
        let mut cross_v = vec![0f32; l_all * nh * t * vd];
        let mut k = vec![0f32; t * nh * qk];
        let mut v = vec![0f32; t * nh * vd];
        for i in 0..self.depth {
            let l = self.active[i];
            let xi = self.i_cross + l * X_N;
            let a = self.act(xi + X_K, enc, t);
            self.gemm(xi + X_K, &a, &mut k);
            self.gemm(xi + X_V, &a, &mut v);
            self.lora_add(Site::Cross, Proj::K, l, enc, t, &mut k);
            self.lora_add(Site::Cross, Proj::V, l, enc, t, &mut v);
            for si in 0..t {
                for h in 0..nh {
                    let kb = (si * nh + h) * qk;
                    self.zc(&k[kb..kb + qk], xi + X_KN, &mut tmp[..qk]);
                    let ck = ((l * nh + h) * t + si) * qk;
                    cross_k[ck..ck + qk].copy_from_slice(&tmp[..qk]);
                    let cv = ((l * nh + h) * t + si) * vd;
                    let vb = (si * nh + h) * vd;
                    cross_v[cv..cv + vd].copy_from_slice(&v[vb..vb + vd]);
                }
            }
        }
        s.cross_k = Rc::new(cross_k);
        s.cross_v = Rc::new(cross_v);
        s
    }

    /// Feed one token at `pos`; returns the final hidden state.
    pub fn dec_step(&self, tok: u32, pos: usize, s: &mut Session) -> [f32; 512] {
        let h = self.dec_step_batch(&[tok], pos, core::slice::from_mut(s));
        let mut hid = [0f32; 512];
        hid[..self.d].copy_from_slice(&h);
        hid
    }

    /// Feed one token per beam at `pos` (all beams are at the same
    /// position); returns the final hidden states, [beam][d].
    pub fn dec_step_batch(&self, toks: &[u32], pos: usize, ss: &mut [Session]) -> Vec<f32> {
        let bsz = ss.len();
        let e = &self.t[self.i_emb];
        let d = self.d;
        let mut x = vec![0f32; bsz * d];
        let mut row = [0f32; 512];
        let sd = sqrtf(d as f32);
        for b in 0..bsz {
            let xb = &mut x[b * d..(b + 1) * d];
            match self.row_of(toks[b] as usize) {
                Row::Base(i) => dequant_row(&self.c, e, self.p[self.i_emb], i, &mut row, xb),
                Row::Lora(j) => {
                    let rows = self.lora.map_or(&[][..], |l| l.rows);
                    xb.copy_from_slice(&rows[j * d..(j + 1) * d]);
                }
            }
            for v in xb.iter_mut() {
                *v *= sd;
            }
        }
        let mut ek = vec![0f32; bsz * 4 * 512];
        let mut ev = vec![0f32; bsz * 4 * 512];
        self.engram_batch(toks, pos, ss, &mut ek, &mut ev);
        let mut lanes = vec![0f32; bsz * self.lanes * d];
        lanes_init(&x, bsz, self.lanes, d, &mut lanes);
        for i in 0..self.depth {
            self.mhc_step(
                self.i_dec_mhc,
                self.active[i],
                i,
                &mut lanes,
                bsz,
                &mut |w, layer, u, _| w.dec_block_batch(layer, u, ss, pos, &ek, &ev),
            );
        }
        let mut hid = vec![0f32; bsz * d];
        self.lanes_out(&lanes, bsz, self.i_dec_final, &mut hid);
        hid
    }

    /// Tied output head: vocab logits for a hidden state.
    pub fn logits(&self, hid: &[f32]) -> Vec<f32> {
        self.logits_batch(&hid[..self.d], 1)
    }

    /// Logits for `bsz` hidden states at once, [beam][vocab].
    pub fn logits_batch(&self, hid: &[f32], bsz: usize) -> Vec<f32> {
        let _s = prof::span(prof::HEAD);
        let base = self.lin(self.i_emb, &hid[..bsz * self.d], bsz);
        let Some(l) = self.lora else {
            return base;
        };
        // Tied head over the extended vocabulary: the adapter's rows give the
        // logits of the tokens and languages it adds.
        let (d, bv, v) = (self.d, self.base_vocab, self.vocab);
        let mut out = vec![0f32; bsz * v];
        for b in 0..bsz {
            let h = &hid[b * d..(b + 1) * d];
            for id in 0..v {
                out[b * v + id] = match self.row_of(id) {
                    Row::Base(i) => base[b * bv + i],
                    Row::Lora(j) => {
                        let r = &l.rows[j * d..(j + 1) * d];
                        r.iter().zip(h).map(|(&a, &x)| a * x).sum()
                    }
                };
            }
        }
        out
    }

    /// One greedy pass in `lang` from BOS. Returns ids and the mean log-prob.
    fn decode_pass(
        &self,
        lang: usize,
        s: &mut Session,
        cap: usize,
        budget: usize,
    ) -> (Vec<u32>, f32) {
        let mut ids: Vec<u32> = Vec::new();
        let mut sum = 0f32;
        let mut pos = 0usize;
        self.dec_step(BOS_ID, pos, s);
        pos += 1;
        let mut tok = self.lang_base + lang as u32;
        while pos < self.max_seq {
            let hid = self.dec_step(tok, pos, s);
            pos += 1;
            let lg = self.logits(&hid);
            let mut mx = lg[0];
            for &v in &lg[1..] {
                if v > mx {
                    mx = v;
                }
            }
            let mut se = 0f32;
            for &v in &lg {
                se += expf(v - mx);
            }
            let lse = mx + logf(se);
            let mut best = 0usize;
            for k in 1..self.vocab {
                if lg[k] > lg[best] {
                    best = k;
                }
            }
            sum += lg[best] - lse;
            if best as u32 == EOS_ID || ids.len() >= cap || ids.len() >= budget {
                break;
            }
            ids.push(best as u32);
            tok = best as u32;
            let k = tail_loop(&ids);
            if k > 0 {
                ids.truncate(ids.len() - 2 * k);
                break;
            }
        }
        let score = sum / (ids.len() + 1) as f32;
        (ids, score)
    }

    /// Greedy transcription. `lang` = None detects the language from the
    /// decoder's ranking of the language tags after BOS.
    pub fn decode(&mut self, enc: &[f32], t: usize, lang: Option<usize>, cap: usize) -> Vec<u32> {
        // Never more than ~6 tokens per second of audio (80 ms frames).
        let budget = 8 + t / 2;
        let mut s = self.session(enc, t);
        let _s = prof::span(prof::DECODER);
        let lang = match lang {
            Some(l) => {
                assert!(l < self.n_langs, "language index {l} out of range");
                l
            }
            None => {
                let hid = self.dec_step(BOS_ID, 0, &mut s);
                let lg = self.logits(&hid);
                let mask = if self.lang_mask != 0 {
                    self.lang_mask
                } else {
                    u32::MAX
                };
                let mut best: Option<usize> = None;
                for k in 0..self.n_langs {
                    self.lang_logit[k] = lg[self.lang_base as usize + k];
                    if mask & (1 << k) != 0
                        && best.is_none_or(|b| self.lang_logit[k] > self.lang_logit[b])
                    {
                        best = Some(k);
                    }
                }
                best.unwrap_or(0)
            }
        };
        self.lang = lang;
        let (ids, score) = if self.beams > 1 {
            self.beam_search(lang, &mut s, cap.min(MAX_TOKENS))
        } else {
            self.decode_pass(lang, &mut s, cap, budget)
        };
        self.score = score;
        ids
    }

    /// Beam search as the porting guide describes the official engine
    /// (section 6): every live beam proposes its six best text tokens (ids
    /// below 8192, </s> among them), scored as the beam's running log
    /// probability plus the token's, normalised over the text tokens only.
    /// Proposals are taken best first until `beams` beams are live; one that
    /// ends the transcript (</s>, a loop, or the token limit) finishes
    /// instead and is scored as its log probability over its token count.
    /// Stops once `beams` transcripts have finished or no beam is left.
    fn beam_search(&self, lang: usize, s0: &mut Session, cap: usize) -> (Vec<u32>, f32) {
        struct Beam {
            s: Session,
            toks: Vec<u32>,
            score: f32,
            next: u32,
        }
        let nb = self.beams;
        let vocab = self.vocab;
        self.dec_step(BOS_ID, 0, s0);
        let mut live = vec![Beam {
            s: s0.clone(),
            toks: Vec::new(),
            score: 0.0,
            next: self.lang_base + lang as u32,
        }];
        let mut done: Vec<(f32, Vec<u32>)> = Vec::new();
        let mut pos = 1;
        while !live.is_empty() && done.len() < nb && pos < self.max_seq {
            // All live beams through the decoder at once.
            let toks: Vec<u32> = live.iter().map(|b| b.next).collect();
            let mut ss: Vec<Session> = live
                .iter_mut()
                .map(|b| core::mem::replace(&mut b.s, s0.clone_empty()))
                .collect();
            let hid = self.dec_step_batch(&toks, pos, &mut ss);
            let lg = self.logits_batch(&hid, live.len());
            for (b, s) in live.iter_mut().zip(ss) {
                b.s = s;
            }
            let mut props: Vec<(f32, usize, u32)> = Vec::with_capacity(live.len() * BEAM_PROPOSALS);
            for (bi, b) in live.iter().enumerate() {
                let text = &lg[bi * vocab..bi * vocab + self.lang_base as usize];
                let mut mx = text[0];
                for &v in &text[1..] {
                    mx = mx.max(v);
                }
                let mut se = 0f32;
                for &v in text {
                    se += expf(v - mx);
                }
                let lse = mx + logf(se);
                let (best, n) = top_k::<BEAM_PROPOSALS>(text);
                for &k in &best[..n] {
                    props.push((b.score + text[k] - lse, bi, k as u32));
                }
            }
            pos += 1;
            props.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(core::cmp::Ordering::Equal));
            let mut next: Vec<Beam> = Vec::new();
            for (score, bi, tok) in props {
                if next.len() == nb {
                    break;
                }
                let parent = &live[bi];
                let mut toks = parent.toks.clone();
                if tok == EOS_ID {
                    let n = toks.len() + 1;
                    done.push((score / n as f32, toks));
                    continue;
                }
                toks.push(tok);
                let n = toks.len();
                if let Some(unit) = loop_unit(&toks) {
                    toks.truncate(n - 3 * unit);
                    done.push((score / n as f32, toks));
                    continue;
                }
                if n >= cap {
                    done.push((score / n as f32, toks));
                    continue;
                }
                next.push(Beam {
                    s: parent.s.clone(),
                    toks,
                    score,
                    next: tok,
                });
            }
            live = next;
        }
        let best = done
            .into_iter()
            .chain(
                live.into_iter()
                    .map(|b| (b.score / (b.toks.len().max(1)) as f32, b.toks)),
            )
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));
        match best {
            Some((score, toks)) => (toks, score),
            None => (Vec::new(), 0.0),
        }
    }

    /// Ids to text, with every leading space removed.
    /// Number of languages the model knows (7 for the shipped model; a
    /// fine-tune can add more).
    pub fn n_langs(&self) -> usize {
        self.n_langs
    }

    /// Code of language `i` (`"en"`, `"cs"`, ...), read from the model's
    /// `<|xx|>` tag; `""` if `i` is out of range.
    pub fn lang_code(&self, i: usize) -> &'a str {
        if i >= self.n_langs {
            return "";
        }
        let piece = self.tok.piece(self.lang_base + i as u32);
        // Validated as UTF-8 `<|..|>` when the model was opened.
        core::str::from_utf8(&piece[2..piece.len() - 2]).unwrap_or("")
    }

    /// Index of the language with code `code`, if the model has it.
    pub fn lang_index(&self, code: &str) -> Option<usize> {
        (0..self.n_langs).find(|&i| self.lang_code(i) == code)
    }

    pub fn detok(&self, ids: &[u32]) -> String {
        let s = self.tok.decode(ids, true);
        String::from(s.trim_start_matches(' '))
    }

    /// Full pipeline: samples -> (text, language index).
    pub fn transcribe(&mut self, x: &[f32], lang: Option<usize>) -> Result<String, Error> {
        if lang.is_some_and(|l| l >= self.n_langs) {
            return Err(Error::BadLanguage);
        }
        let (enc, t) = self.encode(x)?;
        let ids = self.decode(&enc, t, lang, 320);
        Ok(self.detok(&ids))
    }

    // --------------------------------------------------------------- open

    /// Apply a [`Spec`] preset (front end, positional placement, beam width).
    /// The three fields can also be set one by one.
    pub fn set_spec(&mut self, spec: Spec) {
        let guide = spec != Spec::CEngine;
        self.front_blog = guide;
        self.pe_memory = guide;
        self.beams = if spec == Spec::Guide { 5 } else { 1 };
    }

    /// Use only `depth` decoder layers, chosen by [`crate::ladder::order`].
    /// A model cut with [`crate::ladder::trim`] needs no depth setting.
    pub fn set_depth(&mut self, depth: usize) -> Result<(), Error> {
        let l = self.dec_layers;
        if depth < 2 || depth > l {
            return Err(Error::BadDepth);
        }
        let order = crate::ladder::order(l);
        self.active[..depth].copy_from_slice(&order[..depth]);
        self.active[..depth].sort_unstable();
        self.depth = depth;
        Ok(())
    }

    /// Point a tensor's payload at a copy (e.g. the output head in PSRAM,
    /// which streams faster than the flash mapping).
    pub fn emb_bytes(&self) -> &'a [u8] {
        self.p[self.i_emb]
    }

    pub fn set_emb_bytes(&mut self, b: &'a [u8]) {
        assert_eq!(b.len(), self.p[self.i_emb].len());
        self.p[self.i_emb] = b;
    }

    // ------------------------------------------------------------- LoRA

    /// The model's own tokenizer blob (before an adapter extends it), for
    /// [`crate::tokenizer::extend`].
    pub fn tokenizer_blob(&self) -> &'a [u8] {
        self.p[self.i_tok]
    }

    /// Number of text pieces of the model's own vocabulary (8192 for the
    /// shipped model); an adapter's pieces go after them.
    pub fn base_text_pieces(&self) -> usize {
        self.base_lang_base as usize
    }

    /// Codes of the model's own languages, before an adapter adds any.
    pub fn base_lang_code(&self, i: usize) -> &'a str {
        if i >= self.base_n_langs {
            return "";
        }
        let piece = Tokenizer::new(self.p[self.i_tok])
            .map(|t| t.piece(self.base_lang_base + i as u32))
            .unwrap_or(&[]);
        if piece.len() < 4 {
            return "";
        }
        core::str::from_utf8(&piece[2..piece.len() - 2]).unwrap_or("")
    }

    /// Engine tensor of projection `proj` at `site` in layer 0 and the stride
    /// to the next layer.
    fn proj_tensor(&self, site: Site, proj: Proj) -> (usize, usize) {
        let p = proj as usize;
        match site {
            Site::Encoder => (self.i_enc + [E_Q, E_K, E_V, E_GATE, E_OUT][p], E_N),
            Site::DecoderSelf => (self.i_dec + [D_Q, D_K, D_V, D_GATE, D_OUT][p], D_N),
            Site::Cross => (self.i_cross + [X_Q, X_K, X_V, X_GATE, X_OUT][p], X_N),
        }
    }

    /// Run with an adapter (`None`: the base model again). Checks that it fits
    /// the model: projection shapes, layer counts (an adapter trained on 8
    /// decoder layers runs on a model trimmed to fewer, as on the ladder), and
    /// that its tokenizer extends this model's.
    pub fn set_lora(&mut self, lora: Option<&'a Lora<'a>>) -> Result<(), Error> {
        let Some(l) = lora else {
            self.lora = None;
            self.tok = Tokenizer::new(self.p[self.i_tok]).ok_or(Error::Tokenizer)?;
            self.vocab = self.base_vocab;
            self.lang_base = self.base_lang_base;
            self.n_langs = self.base_n_langs;
            return Ok(());
        };
        let mut dec_layers = None;
        for site in crate::lora::SITES {
            for proj in crate::lora::PROJS {
                let Some(w) = l.weight(site, proj) else {
                    continue;
                };
                let (ti, _) = self.proj_tensor(site, proj);
                let t = &self.t[ti];
                if !w.valid() || w.din != t.shape[1] as usize || w.dout != t.shape[0] as usize {
                    return Err(Error::BadLora);
                }
                if site == Site::Encoder {
                    if w.layers != self.enc_layers {
                        return Err(Error::BadLora);
                    }
                } else if *dec_layers.get_or_insert(w.layers) != w.layers {
                    return Err(Error::BadLora);
                }
            }
        }
        let mut map = [0usize; MAX_LAYERS];
        match dec_layers {
            Some(n) if n == self.dec_layers => {
                for (i, m) in map.iter_mut().enumerate() {
                    *m = i;
                }
            }
            Some(n) if n > self.dec_layers && n <= MAX_LAYERS => {
                // A trimmed model keeps the ladder's layers, in order.
                let keep = crate::ladder::layers_for(n, self.dec_layers);
                map[..keep.len()].copy_from_slice(&keep);
            }
            Some(_) => return Err(Error::BadLora),
            None => {}
        }

        // Vocabulary: base text, added pieces, base tags, added tags.
        let d = self.d;
        if l.rows.len() % d != 0 || l.pieces > l.rows.len() / d {
            return Err(Error::BadLora);
        }
        let added_langs = l.rows.len() / d - l.pieces;
        let vocab = self.base_vocab + l.rows.len() / d;
        let tok = Tokenizer::new(l.tokenizer).ok_or(Error::BadLora)?;
        let (lang_base, n_langs) = lang_tags(&tok, vocab).ok_or(Error::BadLora)?;
        if lang_base != self.base_lang_base + l.pieces as u32
            || n_langs != self.base_n_langs + added_langs
        {
            return Err(Error::BadLora);
        }
        let base_tok = Tokenizer::new(self.p[self.i_tok]).ok_or(Error::Tokenizer)?;
        for i in 0..self.base_n_langs {
            if tok.piece(lang_base + i as u32) != base_tok.piece(self.base_lang_base + i as u32) {
                return Err(Error::BadLora);
            }
        }
        self.lora = Some(l);
        self.lora_dec = map;
        self.tok = tok;
        self.vocab = vocab;
        self.lang_base = lang_base;
        self.n_langs = n_langs;
        Ok(())
    }

    /// The adapter in use, if any.
    pub fn lora(&self) -> Option<&'a Lora<'a>> {
        self.lora
    }

    /// `y += strength * scale * (x @ A) @ B` for one projection, if the
    /// adapter has it. `layer` is the layer in the file.
    fn lora_add(
        &self,
        site: Site,
        proj: Proj,
        layer: usize,
        x: &[f32],
        rows: usize,
        y: &mut [f32],
    ) {
        let Some(l) = self.lora else {
            return;
        };
        let Some(w) = l.weight(site, proj) else {
            return;
        };
        let scale = l.scale * self.lora_strength;
        if scale == 0.0 {
            return;
        }
        let layer = if site == Site::Encoder {
            layer
        } else {
            self.lora_dec[layer]
        };
        let _s = prof::span(prof::LIN);
        w.apply(layer, scale, x, rows, y);
    }

    /// Where token `id`'s embedding row is.
    fn row_of(&self, id: usize) -> Row {
        let Some(l) = self.lora else {
            return Row::Base(id);
        };
        let (text, p, tags) = (self.base_lang_base as usize, l.pieces, self.base_n_langs);
        if id < text {
            Row::Base(id)
        } else if id < text + p {
            Row::Lora(id - text)
        } else if id < text + p + tags {
            Row::Base(id - p)
        } else {
            Row::Lora(id - text - tags)
        }
    }

    pub fn open(blob: &'a [u8]) -> Result<Self, Error> {
        let c = Cact::open(blob)?;
        let h = c.h;
        let nt = c.n as usize;
        let mut t = Vec::with_capacity(nt);
        let mut p = Vec::with_capacity(nt);
        for i in 0..nt {
            let ti = c.tensor(i as u32)?;
            let data = c
                .data(&ti)
                .ok_or(Error::Cact(CactError::BadRecord(i as u32)))?;
            if ti.dtype == DT_CQ && !q8_ok(&ti) {
                return Err(Error::Unsupported);
            }
            t.push(ti);
            p.push(data);
        }
        let d = h.d_model as usize;
        let (nh, nkv, qk, vd) = (
            h.num_heads as usize,
            h.num_kv_heads as usize,
            h.qk_head_dim as usize,
            h.v_head_dim as usize,
        );
        let lanes = h.mhc_lanes as usize;
        let taps = h.qkv_conv_taps as usize;
        let dec_layers = h.num_layers as usize;
        let n_sites = h.num_sites as usize;
        if d != 512
            || lanes > 8
            || taps < 2
            || dec_layers > MAX_LAYERS
            || n_sites > 4
            || qk > 64
            || nh * qk > 384
            || nh * vd > 512
            || nkv * qk > 96
            || nkv * vd > 128
        {
            return Err(Error::Unsupported);
        }
        let mut x = 0usize;
        let i_emb = x;
        x += 1;
        let i_dec = x;
        x += dec_layers * D_N;
        let i_dec_mhc = x;
        x += 9;
        let i_perm1 = x;
        x += 1;
        let i_perm2 = x;
        x += 1;
        let i_engram = x;
        x += n_sites * 4;
        let i_dec_final = x;
        x += 1;
        let i_audio_cfg = x;
        x += 1;
        let i_cross = x;
        x += dec_layers * X_N;
        let i_pe_gate = x;
        x += 1;
        if i_audio_cfg >= nt || t[i_audio_cfg].dtype != DT_FP32 || t[i_audio_cfg].shape[0] != 13 {
            return Err(Error::NotV4Whistle);
        }
        let cfg = |k: usize| f32::from_bits(rd_u32(p[i_audio_cfg], k * 4)) as usize;
        let enc_layers = cfg(3);
        let mels = cfg(8);
        let n_fft = cfg(10);
        let win = cfg(11);
        let hop = cfg(12);
        let i_enc = x;
        x += enc_layers * E_N;
        let i_enc_mhc = x;
        x += 9;
        let i_enc_final = x;
        x += 1;
        let i_stem = x;
        x += 6;
        let i_melfb = x;
        x += 1;
        let i_tok = x;
        x += 1;
        if x != nt
            || t[i_tok].dtype != DT_RAW
            || n_fft != 512
            || win > n_fft
            || mels > 128
            || t[i_melfb].shape[0] as usize != n_fft / 2 + 1
            || t[i_melfb].shape[1] as usize != mels
        {
            return Err(Error::Layout);
        }
        let tok = Tokenizer::new(p[i_tok]).ok_or(Error::Tokenizer)?;
        // The language tags are the last pieces, `<|en|>`, `<|de|>`, ...: the
        // shipped model has 7 after 8192 text pieces, and a fine-tune that adds
        // tokens or languages moves and extends them.
        let vocab = h.vocab_size as usize;
        if t[i_emb].shape[0] as usize != vocab {
            return Err(Error::Tokenizer);
        }
        let (lang_base, n_langs) = lang_tags(&tok, vocab).ok_or(Error::Tokenizer)?;
        let h16 = t
            .iter()
            .zip(&p)
            .map(|(ti, &b)| {
                // SAFETY: any bit pattern is a valid u16; only taken when the
                // payload is 2-byte aligned (cact payloads are 64-byte aligned).
                let (pre, mid, _) = unsafe { b.align_to::<u16>() };
                if ti.dtype == DT_FP16 && pre.is_empty() {
                    mid
                } else {
                    &[][..]
                }
            })
            .collect();

        let mut w = Whistle {
            c,
            t,
            p,
            h16,
            tok,
            d,
            nh,
            nkv,
            qk,
            vd,
            lanes,
            taps,
            vocab,
            lang_base,
            n_langs,
            max_seq: h.max_seq_len as usize,
            dec_layers,
            enc_layers,
            n_sites,
            sites: [0; 4],
            mels,
            n_fft,
            hop,
            rope_inv: [0.0; 32],
            fft_cos: [0.0; 256],
            fft_sin: [0.0; 256],
            hann: [0.0; 512],
            hann_sym: [0.0; 512],
            fb_lo: [0; 128],
            fb_hi: [0; 128],
            perm1: [0; 512],
            perm2: [0; 512],
            i_emb,
            i_dec,
            i_dec_mhc,
            i_engram,
            i_dec_final,
            i_cross,
            i_pe_gate,
            i_enc,
            i_enc_mhc,
            i_enc_final,
            i_stem,
            i_melfb,
            i_tok,
            base_vocab: vocab,
            base_lang_base: lang_base,
            base_n_langs: n_langs,
            lora: None,
            lora_dec: [0; MAX_LAYERS],
            lora_strength: 1.0,
            depth: 0,
            active: [0; MAX_LAYERS],
            lang: 0,
            lang_logit: [0.0; MAX_LANGS],
            score: 0.0,
            lang_mask: 0,
            front_blog: false,
            pe_memory: false,
            beams: 1,
        };
        for i in 0..(qk / 2).min(32) {
            w.rope_inv[i] = libm::powf(h.rope_theta, -((2 * i) as f32) / qk as f32);
        }
        for i in 0..n_sites {
            w.sites[i] = h.sites[i] as usize;
        }
        for i in 0..256 {
            let a = -2.0 * core::f64::consts::PI * i as f64 / 512.0;
            w.fft_cos[i] = libm::cos(a) as f32;
            w.fft_sin[i] = libm::sin(a) as f32;
        }
        let off = (n_fft - win) / 2;
        for i in 0..512 {
            w.hann[i] = if i >= off && i < off + win {
                0.5 - 0.5 * libm::cosf(2.0 * PI * (i - off) as f32 / win as f32)
            } else {
                0.0
            };
            w.hann_sym[i] = if i >= off && i < off + win {
                0.5 - 0.5 * libm::cosf(2.0 * PI * (i - off) as f32 / (win - 1) as f32)
            } else {
                0.0
            };
        }
        let nb = n_fft / 2 + 1;
        for b in 0..mels {
            let mut lo = nb as u16;
            let mut hi = 0u16;
            for k in 0..nb {
                if f32::from_bits(rd_u32(w.p[i_melfb], (k * mels + b) * 4)) != 0.0 {
                    lo = lo.min(k as u16);
                    hi = (k + 1) as u16;
                }
            }
            w.fb_lo[b] = lo.min(hi);
            w.fb_hi[b] = hi;
        }
        for i in 0..512 {
            w.perm1[i] = w.hv(i_perm1, i) as usize;
            w.perm2[i] = w.hv(i_perm2, i) as usize;
        }
        w.set_depth(dec_layers)?;
        w.set_spec(Spec::DEFAULT);
        Ok(w)
    }
}

/// The C engine's level: scale the clip to a peak of 1.
fn peak_gain(x: &[f32]) -> f32 {
    let mut peak = 0f32;
    for &v in x {
        if v.abs() > peak {
            peak = v.abs();
        }
    }
    if peak > 1e-9 { 1.0 / peak } else { 1.0 }
}

/// The guide's level gain: RMS of 320-sample blocks, the one at index
/// floor(0.99 * (blocks - 1)) after sorting (a selection, not an
/// interpolated percentile), scaled to 0.1 if above 1e-6.
fn level_gain_blog(x: &[f32]) -> f32 {
    let blocks = x.len() / 320;
    if blocks == 0 {
        return 1.0;
    }
    let mut rms: Vec<f32> = (0..blocks)
        .map(|b| {
            let mut e = 0f32;
            for &v in &x[b * 320..(b + 1) * 320] {
                let v = sample(v, true);
                e = fma(v, v, e);
            }
            sqrtf(e / 320.0)
        })
        .collect();
    rms.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    let loud = rms[(0.99 * (blocks - 1) as f32) as usize];
    if loud > 1e-6 { 0.1 / loud } else { 1.0 }
}

/// The guide: samples that are NaN, infinite or of magnitude 65536 or more
/// read as zero (only in the blog front end).
#[inline(always)]
fn sample(v: f32, blog: bool) -> f32 {
    // Written so that NaN also maps to 0.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if blog && !(v.abs() < 65536.0) { 0.0 } else { v }
}

/// Rows of an activation matrix handed to parallel workers, each of which
/// owns a disjoint range of rows.
#[derive(Clone, Copy)]
struct RowsPtr(*mut f32);
// SAFETY: workers touch disjoint rows only.
unsafe impl Sync for RowsPtr {}
unsafe impl Send for RowsPtr {}

impl RowsPtr {
    /// # Safety
    /// Row `t` (of `d` floats) is in bounds and owned by the caller.
    #[inline(always)]
    #[allow(clippy::mut_from_ref)]
    unsafe fn row<'x>(self, t: usize, d: usize) -> &'x mut [f32] {
        unsafe { core::slice::from_raw_parts_mut(self.0.add(t * d), d) }
    }
}

/// kron(z, a, b) = a^T Z b, with Z = z viewed as 16x32.
pub fn kron(z: &[f32; 512], a: &[f32; 256], b: &[f32; 1024], out: &mut [f32; 512]) {
    // Register-blocked: 8 outputs stay in FPU registers for the whole sum
    // instead of being loaded and stored for every multiply-add (that ran at
    // ~10 cycles per MAC on the LX7), with fused multiply-adds (3.8 cycles
    // per MAC; separate mul.s + add.s stall on each other).
    let mut tmp = [0f32; 16 * 32];
    for i in 0..16 {
        for lb in (0..32).step_by(8) {
            let mut acc = [0f32; 8];
            for j in 0..32 {
                let zz = z[i * 32 + j];
                let bj: &[f32; 8] = b[j * 32 + lb..j * 32 + lb + 8].try_into().unwrap();
                for q in 0..8 {
                    acc[q] = fma(zz, bj[q], acc[q]);
                }
            }
            tmp[i * 32 + lb..i * 32 + lb + 8].copy_from_slice(&acc);
        }
    }
    for k in 0..16 {
        for lb in (0..32).step_by(8) {
            let mut acc = [0f32; 8];
            for i in 0..16 {
                let aa = a[i * 16 + k];
                let ti: &[f32; 8] = tmp[i * 32 + lb..i * 32 + lb + 8].try_into().unwrap();
                for q in 0..8 {
                    acc[q] = fma(aa, ti[q], acc[q]);
                }
            }
            out[k * 32 + lb..k * 32 + lb + 8].copy_from_slice(&acc);
        }
    }
}

fn sinkhorn(m: &mut [f32], n: usize) {
    for _ in 0..SINKHORN {
        for i in 0..n {
            let mut mx = m[i * n];
            for j in 1..n {
                mx = mx.max(m[i * n + j]);
            }
            let mut s = 0f32;
            for j in 0..n {
                s += expf(m[i * n + j] - mx);
            }
            let s = mx + logf(s);
            for j in 0..n {
                m[i * n + j] -= s;
            }
        }
        for j in 0..n {
            let mut mx = m[j];
            for i in 1..n {
                mx = mx.max(m[i * n + j]);
            }
            let mut s = 0f32;
            for i in 0..n {
                s += expf(m[i * n + j] - mx);
            }
            let s = mx + logf(s);
            for i in 0..n {
                m[i * n + j] -= s;
            }
        }
    }
    for v in m[..n * n].iter_mut() {
        *v = libm::expf(*v);
    }
}

fn lanes_init(x: &[f32], rows: usize, n: usize, d: usize, lanes: &mut [f32]) {
    for t in 0..rows {
        for j in 0..n {
            lanes[(t * n + j) * d..(t * n + j + 1) * d].copy_from_slice(&x[t * d..(t + 1) * d]);
        }
    }
}

/// 3x3 conv, stride 2 on both axes, zero padding, over [T][F][C], for output
/// rows [o0, o0 + on). `x` holds global rows from `i0`. Kernel [9][C], index
/// kt*3 + kf. `in_c` 1 fans one input channel out to C; otherwise depthwise.
#[allow(clippy::too_many_arguments)]
fn conv_rows(
    k: &[f32],
    x: &[f32],
    i0: usize,
    t_all: usize,
    f: usize,
    in_c: usize,
    c_ch: usize,
    o0: usize,
    on: usize,
    out: &mut [f32],
    act: bool,
) {
    let fo_n = f.div_ceil(2);
    debug_assert!(c_ch.is_multiple_of(8));
    let row_len = fo_n * c_ch;
    assert!(out.len() >= on * row_len);
    let op = RowsPtr(out.as_mut_ptr());
    // One output row (time step) per work item; blocks of 8 channels stay in
    // FPU registers across the 9 taps.
    let kernel = |ra: usize, rb: usize| {
        for r in ra..rb {
            // SAFETY: row r of out is owned by this work item.
            let orow = unsafe { op.row(r, row_len) };
            let to = o0 + r;
            for fo in 0..fo_n {
                for cb in (0..c_ch).step_by(8) {
                    let mut acc = [0f32; 8];
                    for a in 0..3 {
                        let tt = 2 * to as isize + a as isize - 1;
                        if tt < 0 || tt >= t_all as isize {
                            continue;
                        }
                        for b in 0..3 {
                            let ff = 2 * fo as isize + b as isize - 1;
                            if ff < 0 || ff >= f as isize {
                                continue;
                            }
                            let ko = (a * 3 + b) * c_ch + cb;
                            let kk: &[f32; 8] = k[ko..ko + 8].try_into().unwrap();
                            let xi = ((tt as usize - i0) * f + ff as usize) * in_c;
                            if in_c == 1 {
                                let xv = x[xi];
                                for q in 0..8 {
                                    acc[q] = fma(xv, kk[q], acc[q]);
                                }
                            } else {
                                let xs: &[f32; 8] = x[xi + cb..xi + cb + 8].try_into().unwrap();
                                for q in 0..8 {
                                    acc[q] = fma(xs[q], kk[q], acc[q]);
                                }
                            }
                        }
                    }
                    if act {
                        for v in acc.iter_mut() {
                            *v = silu(*v);
                        }
                    }
                    let o = fo * c_ch + cb;
                    orow[o..o + 8].copy_from_slice(&acc);
                }
            }
        }
    };
    crate::par::rows(on, 1, &kernel);
}

/// The `k` largest entries of `x`, best first.
fn top_k<const K: usize>(x: &[f32]) -> ([usize; K], usize) {
    // A sorted array, largest first; ties keep the lower index first.
    let mut best = [0usize; K];
    let mut n = 0;
    for (i, &v) in x.iter().enumerate() {
        if n < K || v > x[best[n - 1]] {
            let at = best[..n].iter().position(|&j| v > x[j]).unwrap_or(n);
            n = (n + 1).min(K);
            best.copy_within(at..n - 1, at + 1);
            best[at] = i;
        }
    }
    (best, n)
}

/// The guide's loop rule for the beam: the same unit of up to eight tokens
/// repeated four times at the end. Returns the unit length.
fn loop_unit(ids: &[u32]) -> Option<usize> {
    let n = ids.len();
    (1..=8).find(|&k| n >= 4 * k && (0..3 * k).all(|j| ids[n - 1 - j] == ids[n - 1 - j - k]))
}

/// Length k of a run at the end of `ids`: the last k tokens repeated three
/// times back to back, for some k <= 4. Returns 0 when there is none.
fn tail_loop(ids: &[u32]) -> usize {
    let n = ids.len();
    for k in 1..=4 {
        if n < 3 * k {
            break;
        }
        let mut j = 0;
        while j < 2 * k && ids[n - 1 - j] == ids[n - 1 - j - k] {
            j += 1;
        }
        if j == 2 * k {
            return k;
        }
    }
    0
}

/// Find the speech in a clip: 20 ms windows whose RMS is `min_db` over the
/// noise floor (`floor_rms`, or the 10th-percentile window when <= 0).
/// Returns the sample range with `margin_ms` either side, or None for silence.
pub fn find_speech(
    x: &[f32],
    mut floor_rms: f32,
    min_db: f32,
    margin_ms: usize,
) -> Option<(usize, usize)> {
    const W: usize = 320;
    let n = x.len();
    let nw = n / W;
    if nw < 5 {
        return None;
    }
    let r: Vec<f32> = (0..nw)
        .map(|i| {
            let e: f32 = x[i * W..(i + 1) * W].iter().map(|v| v * v).sum();
            sqrtf(e / W as f32)
        })
        .collect();
    if floor_rms <= 0.0 {
        let mut s = r.clone();
        s.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        floor_rms = s[nw / 10];
    }
    let thr = floor_rms.max(1e-7) * libm::powf(10.0, min_db / 20.0);
    let (mut loud, mut first, mut last) = (0, 0, 0);
    for (i, &v) in r.iter().enumerate() {
        if v > thr {
            if loud == 0 {
                first = i;
            }
            last = i;
            loud += 1;
        }
    }
    if loud < 5 {
        return None;
    }
    let m = margin_ms * 16;
    let s0 = (first * W).saturating_sub(m);
    let s1 = ((last + 1) * W + m).min(n);
    Some((s0, s1))
}

#[cfg(test)]
mod tests {
    extern crate std;
    use std::vec::Vec;

    use super::top_k;

    /// The insertion-sort version top_k replaced.
    fn reference(x: &[f32], k: usize) -> Vec<usize> {
        let mut best: Vec<usize> = Vec::new();
        for (i, &v) in x.iter().enumerate() {
            if best.len() < k || v > x[best[best.len() - 1]] {
                let at = best.iter().position(|&j| v > x[j]).unwrap_or(best.len());
                best.insert(at, i);
                best.truncate(k);
            }
        }
        best
    }

    #[test]
    fn top_k_matches_the_reference() {
        let mut seed = 12345u32;
        for len in [1usize, 3, 6, 7, 50, 8199] {
            let x: Vec<f32> = (0..len)
                .map(|_| {
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    // Few distinct values, so ties are common.
                    (seed >> 28) as f32
                })
                .collect();
            let (best, n) = top_k::<6>(&x);
            assert_eq!(&best[..n], &reference(&x, 6)[..], "len {len}");
        }
    }
}
