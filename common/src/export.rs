//! Writing a `.cact` model from a float checkpoint, optionally with a LoRA
//! adapter merged in: Rust's `needle whistle build`.
//!
//! The tensors, their order and the header follow Cactus's exporter
//! (`needle/model/export.py`, cactus-needle 3.2.8). A few constants of the
//! format are not derived from the checkpoint but copied from a published
//! `.cact` (the *template*): the base tokenizer, the Hadamard-MLP permutations
//! and the mel filterbank. They are the same for every Whistle model.

use cactus_whistle::cact::{Cact, DT_CQ, DT_FP16, DT_FP32, DT_RAW, TAG4};
use cactus_whistle::lora::{Proj, Site};

use crate::checkpoint::{Array, Checkpoint, Config};
use crate::cq;
use crate::lora::Adapter;

const ALIGN: usize = 64;
const WEIGHT_BITS: u32 = 4;
const AUDIO_MANIFEST_MAGIC: f32 = 3246.0;
const ENGRAM_CONV_TAPS: u32 = 4;
const HADAMARD: [&str; 13] = [
    "d1", "d2", "b2", "d3", "d4", "w1a", "w1b", "w2a", "w2b", "w3a", "w3b", "cond_v", "cond_u",
];
const PROJ: [&str; 5] = ["q_proj", "k_proj", "v_proj", "gate_proj", "out_proj"];

struct Tensor {
    name: String,
    dtype: u8,
    shape: Vec<usize>,
    blob: Vec<u8>,
    group: u32,
    bits: u32,
}

fn fp16(name: impl Into<String>, a: &Array, shape: Vec<usize>) -> Tensor {
    assert_eq!(shape.iter().product::<usize>(), a.len());
    let blob = a
        .data
        .iter()
        .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
        .collect();
    Tensor {
        name: name.into(),
        dtype: DT_FP16,
        shape,
        blob,
        group: 0,
        bits: 0,
    }
}

fn fp32(name: impl Into<String>, data: &[u8], shape: Vec<usize>) -> Tensor {
    Tensor {
        name: name.into(),
        dtype: DT_FP32,
        shape,
        blob: data.to_vec(),
        group: 0,
        bits: 0,
    }
}

/// CQ tensor of a `[rows, cols]` matrix.
fn q(name: impl Into<String>, data: &[f32], rows: usize, cols: usize) -> Tensor {
    let p = cq::pack(data, rows, cols, WEIGHT_BITS, cq::GROUP);
    let mut blob = p.indices;
    blob.extend(p.norms.iter().flat_map(|n| n.to_le_bytes()));
    Tensor {
        name: name.into(),
        dtype: DT_CQ,
        shape: vec![rows, cols],
        blob,
        group: cq::GROUP as u32,
        bits: WEIGHT_BITS,
    }
}

/// CQ of a `[in, out]` kernel (stored transposed, `[out, in]`).
fn q_kernel(name: impl Into<String>, k: &Array) -> Tensor {
    let t = k.transpose_last();
    q(name, &t.data, t.shape[0], t.shape[1])
}

/// The base model's parameters with an adapter merged in.
struct Params<'a> {
    ck: &'a Checkpoint,
    merged: std::collections::BTreeMap<String, Array>,
}

impl Params<'_> {
    fn get(&self, name: &str) -> Result<&Array, String> {
        match self.merged.get(name) {
            Some(a) => Ok(a),
            None => self.ck.get(name),
        }
    }
    /// Layer `i` of a stacked parameter.
    fn at(&self, name: &str, i: usize) -> Result<Array, String> {
        Ok(self.get(name)?.index(i))
    }
}

pub fn lora_param(site: Site, proj: Proj) -> String {
    let prefix = match site {
        Site::Encoder => "encoder/layers/block/self_attn",
        Site::DecoderSelf => "stack/layers/block/self_attn",
        Site::Cross => "stack/layers/block/cross_attn",
    };
    format!("{prefix}/{}/kernel", PROJ[proj as usize])
}

/// `W += scale * A @ B` per layer, and the new vocabulary rows spliced into the
/// embedding: `[base text | new pieces | base tags | new tags]`.
fn merge<'a>(ck: &'a Checkpoint, adapter: &Adapter) -> Result<(Params<'a>, Config), String> {
    let base = &ck.config;
    let mut merged = std::collections::BTreeMap::new();
    for u in &adapter.updates {
        let name = lora_param(u.site, u.proj);
        let w = ck.get(&name)?;
        if w.shape != [u.layers, u.din, u.dout] {
            return Err(format!(
                "adapter {name} is [{}, {}, {}] but the checkpoint has {:?}",
                u.layers, u.din, u.dout, w.shape
            ));
        }
        let mut w = w.clone();
        let scale = adapter.scale;
        for l in 0..u.layers {
            let a = &u.a[l * u.din * u.rank..(l + 1) * u.din * u.rank];
            let b = &u.b[l * u.rank * u.dout..(l + 1) * u.rank * u.dout];
            let wl = &mut w.data[l * u.din * u.dout..(l + 1) * u.din * u.dout];
            let mut row = vec![0f32; u.dout];
            for i in 0..u.din {
                row.fill(0.0);
                for r in 0..u.rank {
                    let av = a[i * u.rank + r];
                    for (o, bv) in row.iter_mut().zip(&b[r * u.dout..(r + 1) * u.dout]) {
                        *o += av * bv;
                    }
                }
                for (wv, d) in wl[i * u.dout..(i + 1) * u.dout].iter_mut().zip(&row) {
                    *wv += scale * d;
                }
            }
        }
        merged.insert(name, w);
    }

    let n_base = base.languages.len();
    if adapter.languages.len() < n_base || adapter.languages[..n_base] != base.languages[..] {
        return Err(format!(
            "the adapter's languages {:?} don't start with the base model's {:?}",
            adapter.languages, base.languages
        ));
    }
    let pieces = adapter.pieces.len();
    let new_langs = adapter.languages.len() - n_base;
    let d = base.dim;
    if adapter.rows.len() != (pieces + new_langs) * d {
        return Err(format!(
            "the adapter has {} vocabulary rows, expected {} pieces + {} languages",
            adapter.rows.len() / d.max(1),
            pieces,
            new_langs
        ));
    }
    let emb = ck.get("embedding/embedding")?;
    let (text, rows) = (base.text, &adapter.rows);
    let mut e = Vec::with_capacity(emb.len() + rows.len());
    e.extend_from_slice(&emb.data[..text * d]);
    e.extend_from_slice(&rows[..pieces * d]);
    e.extend_from_slice(&emb.data[text * d..]);
    e.extend_from_slice(&rows[pieces * d..]);
    let vocab = e.len() / d;
    merged.insert("embedding/embedding".into(), Array::new(vec![vocab, d], e));

    let mut config = base.clone();
    config.text = base.text + pieces;
    config.languages = adapter.languages.clone();
    Ok((Params { ck, merged }, config))
}

/// What a template `.cact` contributes: its tokenizer (RAW tensor), the
/// Hadamard-MLP permutations (`hada_p1`, `hada_p2`, FP32 `[hada_n]`) and the
/// mel filterbank (FP32 `[bins, mels]`), found by type, shape and order.
pub struct Template<'a> {
    pub tokenizer: &'a [u8],
    pub hada_p1: &'a [u8],
    pub hada_p2: &'a [u8],
    pub mel: (&'a [u8], Vec<usize>),
}

impl Template<'_> {
    fn floats(b: &[u8]) -> Vec<f32> {
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
    /// The two permutations as indices.
    pub fn perms(&self) -> (Vec<usize>, Vec<usize>) {
        let p = |b: &[u8]| Self::floats(b).into_iter().map(|v| v as usize).collect();
        (p(self.hada_p1), p(self.hada_p2))
    }
    /// The mel filterbank, row-major `[bins, mels]`.
    pub fn mel(&self) -> Vec<f32> {
        Self::floats(self.mel.0)
    }
}

pub fn template<'a>(c: &Cact<'a>) -> Result<Template<'a>, String> {
    let mut tokenizer = None;
    let mut fp32s = Vec::new();
    for i in 0..c.n {
        let t = c.tensor(i).map_err(|e| e.to_string())?;
        let data = c.data(&t).ok_or("template: truncated tensor")?;
        match t.dtype {
            DT_RAW => tokenizer = Some(data),
            DT_FP32 => fp32s.push((t, data)),
            _ => {}
        }
    }
    // FP32 tensors in Cactus's order: hada_p1, hada_p2, audio.manifest, mel.
    let hada: Vec<_> = fp32s
        .iter()
        .filter(|(t, _)| t.ndim == 1 && t.shape[0] == c.h.hada_n)
        .collect();
    let mel = fp32s.iter().find(|(t, _)| t.ndim == 2);
    match (tokenizer, hada.as_slice(), mel) {
        (Some(tok), [p1, p2], Some((m, md))) => Ok(Template {
            tokenizer: tok,
            hada_p1: p1.1,
            hada_p2: p2.1,
            mel: (md, vec![m.shape[0] as usize, m.shape[1] as usize]),
        }),
        _ => Err("the template .cact lacks the tokenizer, the Hadamard permutations or the mel filterbank".into()),
    }
}

/// Builds a `.cact` (W4A8, all decoder layers) from `ck`, merging `adapter` if
/// given. `template` is a published Whistle `.cact` (e.g. `whistle.cact`).
pub fn build(
    ck: &Checkpoint,
    adapter: Option<&Adapter>,
    template_blob: &[u8],
) -> Result<Built, String> {
    let tc = Cact::open(template_blob).map_err(|e| format!("template: {e}"))?;
    let tpl = template(&tc)?;
    let (p, cfg) = match adapter {
        Some(a) => merge(ck, a)?,
        None => (
            Params {
                ck,
                merged: Default::default(),
            },
            ck.config.clone(),
        ),
    };
    let base_tok = cactus_whistle::tokenizer::Tokenizer::new(tpl.tokenizer)
        .ok_or("template: unreadable tokenizer")?;
    if base_tok.len() != ck.config.vocab() {
        return Err(format!(
            "the template's tokenizer has {} pieces, the checkpoint {} (text {} + {} languages): use the published base whistle.cact as template",
            base_tok.len(),
            ck.config.vocab(),
            ck.config.text,
            ck.config.languages.len()
        ));
    }
    let tokenizer = match adapter {
        Some(a) => {
            let pieces: Vec<&str> = a.pieces.iter().map(String::as_str).collect();
            let langs: Vec<&str> = a.languages[ck.config.languages.len()..]
                .iter()
                .map(String::as_str)
                .collect();
            cactus_whistle::tokenizer::extend(tpl.tokenizer, ck.config.text, &pieces, &langs)
                .ok_or("cannot extend the template's tokenizer")?
        }
        None => tpl.tokenizer.to_vec(),
    };

    let mut ts: Vec<Tensor> = Vec::new();
    let d = cfg.dim;
    let emb = p.get("embedding/embedding")?;
    ts.push(q("embedding", &emb.data, emb.shape[0], d));

    let b = "stack/layers/block";
    let sa = "stack/layers/block/self_attn";
    let ha = "stack/layers/block/hadamard_mlp";
    for i in 0..cfg.dec_layers {
        let l = format!("layer{i:02}");
        let s = |n: &str| -> Result<Array, String> { p.at(n, i) };
        let f = |name: String, a: Array| fp16(name, &a, a.shape.clone());
        ts.push(f(
            format!("{l}.norm_in"),
            s(&format!("{b}/ZCRMSNorm_0/scale"))?,
        ));
        for n in ["q_proj", "k_proj", "v_proj"] {
            ts.push(q_kernel(
                format!("{l}.{n}"),
                &s(&format!("{sa}/{n}/kernel"))?,
            ));
        }
        if cfg.taps > 0 {
            for n in ["q_taps", "k_taps", "v_taps"] {
                ts.push(f(format!("{l}.{n}"), s(&format!("{sa}/{n}"))?));
            }
        }
        ts.push(f(format!("{l}.q_norm"), s(&format!("{sa}/q_norm/scale"))?));
        ts.push(f(format!("{l}.k_norm"), s(&format!("{sa}/k_norm/scale"))?));
        for n in ["gate_proj", "out_proj"] {
            ts.push(q_kernel(
                format!("{l}.{n}"),
                &s(&format!("{sa}/{n}/kernel"))?,
            ));
        }
        ts.push(f(
            format!("{l}.post_norm"),
            s(&format!("{b}/post_attn_norm/scale"))?,
        ));
        ts.push(fp16(
            format!("{l}.attn_gate"),
            &s(&format!("{b}/attn_gate"))?,
            vec![1],
        ));
        ts.push(f(
            format!("{l}.pre_hada"),
            s(&format!("{b}/pre_hada_norm/scale"))?,
        ));
        for n in HADAMARD {
            ts.push(f(format!("{l}.{n}"), s(&format!("{ha}/{n}"))?));
        }
    }

    for n in [
        "mhc_a_pre",
        "mhc_a_post",
        "mhc_a_res",
        "mhc_b_pre",
        "mhc_b_post",
        "mhc_b_res",
    ] {
        let a = p.get(&format!("stack/{n}"))?;
        ts.push(fp16(n, a, a.shape.clone()));
    }
    for n in ["mhc_phi_pre", "mhc_phi_post", "mhc_phi_res"] {
        ts.push(phi(n, p.get(&format!("stack/{n}"))?));
    }
    ts.push(fp32("hada_p1", tpl.hada_p1, vec![tpl.hada_p1.len() / 4]));
    ts.push(fp32("hada_p2", tpl.hada_p2, vec![tpl.hada_p2.len() / 4]));

    let (heads, sub_dim) = cfg.engram_geometry();
    let tables = cfg.orders.len() * heads;
    for s in 0..cfg.sites.len() {
        let e = format!("engrams_{s}");
        let t = p.get(&format!("{e}/embedding"))?;
        ts.push(q(
            format!("engram{s}.tables"),
            &t.data,
            tables * cfg.slots,
            sub_dim,
        ));
        ts.push(q_kernel(
            format!("engram{s}.key_proj"),
            p.get(&format!("{e}/key_proj/kernel"))?,
        ));
        ts.push(q_kernel(
            format!("engram{s}.value_proj"),
            p.get(&format!("{e}/value_proj/kernel"))?,
        ));
        let taps = p.get(&format!("{e}/taps"))?;
        ts.push(fp16(format!("engram{s}.taps"), taps, taps.shape.clone()));
    }
    let fin = p.get("stack/final_norm/scale")?;
    ts.push(fp16("final_norm", fin, fin.shape.clone()));
    if ck
        .params
        .keys()
        .any(|k| k.starts_with("heads") || k.contains("probes"))
    {
        return Err("the checkpoint has probe heads, which this exporter doesn't write".into());
    }

    // Audio: cross attention, encoder, stem, mel.
    let manifest: Vec<u8> = [
        AUDIO_MANIFEST_MAGIC,
        1.0,
        cfg.text as f32,
        cfg.enc_layers as f32,
        cfg.enc_lanes as f32,
        cfg.conv_kernel as f32,
        cfg.stem as f32,
        cfg.stem_stages as f32,
        cfg.mels as f32,
        cfg.rate as f32,
        cfg.fft as f32,
        cfg.win as f32,
        cfg.hop as f32,
    ]
    .iter()
    .flat_map(|v: &f32| v.to_le_bytes())
    .collect();
    ts.push(fp32("audio.manifest", &manifest, vec![13]));
    let ca = "stack/layers/block/cross_attn";
    for i in 0..cfg.dec_layers {
        let l = format!("layer{i:02}");
        let s = |n: &str| -> Result<Array, String> { p.at(n, i) };
        let a = s(&format!("{b}/cross_norm/scale"))?;
        ts.push(fp16(format!("{l}.cross_norm"), &a, a.shape.clone()));
        for n in PROJ {
            ts.push(q_kernel(
                format!("{l}.cross.{n}"),
                &s(&format!("{ca}/{n}/kernel"))?,
            ));
        }
        for (out, n) in [
            ("cross_q_norm", format!("{ca}/q_norm/scale")),
            ("cross_k_norm", format!("{ca}/k_norm/scale")),
            ("post_cross_norm", format!("{b}/post_cross_norm/scale")),
        ] {
            let a = s(&n)?;
            ts.push(fp16(format!("{l}.{out}"), &a, a.shape.clone()));
        }
        ts.push(fp16(
            format!("{l}.cross_gate"),
            &s(&format!("{b}/cross_gate"))?,
            vec![1],
        ));
    }
    ts.push(fp16("pe_gate", p.get("pe_gate")?, vec![1]));

    let e = "encoder/layers/block";
    for i in 0..cfg.enc_layers {
        let l = format!("encoder{i:02}");
        let s = |n: &str| -> Result<Array, String> { p.at(n, i) };
        for n in [
            "pre_hada_norm_0",
            "ZCRMSNorm_0",
            "post_attn_norm",
            "conv_norm",
            "conv_out_norm",
            "pre_hada_norm",
        ] {
            let a = s(&format!("{e}/{n}/scale"))?;
            ts.push(fp16(format!("{l}.{n}"), &a, a.shape.clone()));
        }
        for n in PROJ {
            ts.push(q_kernel(
                format!("{l}.{n}"),
                &s(&format!("{e}/self_attn/{n}/kernel"))?,
            ));
        }
        for n in ["q_norm", "k_norm"] {
            let a = s(&format!("{e}/self_attn/{n}/scale"))?;
            ts.push(fp16(format!("{l}.{n}"), &a, a.shape.clone()));
        }
        ts.push(fp16(
            format!("{l}.attn_gate"),
            &s(&format!("{e}/attn_gate"))?,
            vec![1],
        ));
        ts.push(q_kernel(
            format!("{l}.pw1"),
            &s(&format!("{e}/pw1/kernel"))?,
        ));
        ts.push(q_kernel(
            format!("{l}.pw2"),
            &s(&format!("{e}/pw2/kernel"))?,
        ));
        let dw = s(&format!("{e}/dw"))?;
        let k = cfg.conv_kernel;
        ts.push(fp16(format!("{l}.dw"), &dw, vec![k, dw.len() / k]));
        for mlp in ["hadamard_mlp_0", "hadamard_mlp"] {
            for n in HADAMARD {
                let a = s(&format!("{e}/{mlp}/{n}"))?;
                ts.push(fp16(format!("{l}.{mlp}.{n}"), &a, a.shape.clone()));
            }
        }
    }
    for n in [
        "mhc_a_pre",
        "mhc_a_post",
        "mhc_a_res",
        "mhc_b_pre",
        "mhc_b_post",
        "mhc_b_res",
    ] {
        let a = p.get(&format!("encoder/{n}"))?;
        ts.push(fp16(format!("encoder.{n}"), a, a.shape.clone()));
    }
    for n in ["mhc_phi_pre", "mhc_phi_post", "mhc_phi_res"] {
        ts.push(phi(
            &format!("encoder.{n}"),
            p.get(&format!("encoder/{n}"))?,
        ));
    }
    let a = p.get("encoder/final_norm/scale")?;
    ts.push(fp16("encoder.final_norm", a, a.shape.clone()));
    let w = p.get("stem/w")?;
    ts.push(fp16("stem.w", w, vec![9, w.len() / 9]));
    for i in 1..cfg.stem_stages {
        let dw = p.get(&format!("stem/dw_{i}"))?;
        ts.push(fp16(format!("stem.dw_{i}"), dw, vec![9, dw.len() / 9]));
        ts.push(q_kernel(
            format!("stem.pw_{i}"),
            p.get(&format!("stem/pw_{i}/kernel"))?,
        ));
    }
    ts.push(q_kernel("stem.out", p.get("stem/out/kernel")?));
    ts.push(fp32("mel", tpl.mel.0, tpl.mel.1.clone()));
    ts.push(Tensor {
        name: "tokenizer".into(),
        dtype: DT_RAW,
        shape: vec![],
        blob: tokenizer,
        group: 0,
        bits: 0,
    });

    Ok(Built {
        blob: pack(&cfg, &ts),
        names: ts.into_iter().map(|t| t.name).collect(),
        languages: cfg.languages,
    })
}

/// A written model: the `.cact` bytes, its tensor names in order and its languages.
pub struct Built {
    pub blob: Vec<u8>,
    pub names: Vec<String>,
    pub languages: Vec<String>,
}

/// `[L, n*C, lanes]` mHC projection, stored as `[L * lanes, n*C]`.
fn phi(name: &str, a: &Array) -> Tensor {
    let (l, nc, lanes) = (a.shape[0], a.shape[1], a.shape[2]);
    let t = a.transpose_last();
    q(name, &t.data, l * lanes, nc)
}

fn pack(cfg: &Config, ts: &[Tensor]) -> Vec<u8> {
    let (heads, sub_dim) = cfg.engram_geometry();
    let hada_n = cfg.dim.next_power_of_two();
    let mut words: Vec<u32> = vec![
        TAG4,
        ts.len() as u32,
        cq::CODEBOOK.len() as u32,
        0, // kv_window
        8, // kv_bits
        cfg.vocab() as u32,
        0, // out_vocab
        cfg.dim as u32,
        cfg.heads() as u32,
        cfg.kv_heads as u32,
        cfg.dec_layers as u32,
        cfg.qk_dim as u32,
        cfg.v_dim as u32,
        cfg.max_tokens as u32,
        hada_n as u32,
        cfg.lanes as u32,
        0, // sliding_window
        0, // global mask lo
        0, // global mask hi
        cfg.taps as u32,
        cfg.slots as u32,
        sub_dim as u32,
        (cfg.orders.len() * heads) as u32,
        ENGRAM_CONV_TAPS,
        *cfg.orders.iter().max().unwrap_or(&0) as u32,
        0, // engram seed heads
        cfg.orders.len() as u32,
    ];
    words.extend((0..4).map(|i| cfg.orders.get(i).copied().unwrap_or(0) as u32));
    words.push(cfg.sites.len() as u32);
    words.extend((0..16).map(|i| cfg.sites.get(i).copied().unwrap_or(0) as u32));
    let mut buf: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    buf.extend(cfg.rope.to_le_bytes());
    buf.extend(cq::CODEBOOK.iter().flat_map(|v| v.to_le_bytes()));

    let align = |n: usize| n.div_ceil(ALIGN) * ALIGN;
    let mut pos = buf.len() + ts.len() * 44;
    let mut offsets = Vec::with_capacity(ts.len());
    for t in ts {
        pos = align(pos);
        offsets.push(pos);
        pos += t.blob.len();
    }
    for (t, &off) in ts.iter().zip(&offsets) {
        buf.push(t.dtype);
        buf.push(t.shape.len() as u8);
        buf.extend(0u16.to_le_bytes());
        for i in 0..4 {
            buf.extend((t.shape.get(i).copied().unwrap_or(0) as u32).to_le_bytes());
        }
        buf.extend((off as u64).to_le_bytes());
        buf.extend((t.blob.len() as u64).to_le_bytes());
        buf.extend(t.group.to_le_bytes());
        buf.extend(t.bits.to_le_bytes());
    }
    for (t, &off) in ts.iter().zip(&offsets) {
        buf.resize(off, 0);
        buf.extend_from_slice(&t.blob);
    }
    buf
}
