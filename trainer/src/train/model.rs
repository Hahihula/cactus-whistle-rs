//! Whistle's training forward pass in burn, numerically the model Cactus
//! fine-tunes (`needle/model/whistle.py`, cactus-needle 3.2.8, Apache-2.0)
//! with `quant=True`: CQ-W4 weights, A8 activations, queries and KV.
//!
//! Only the LoRA matrices and the new vocabulary rows train; every other
//! weight is frozen, so frozen kernels are fake-quantised once at load, and
//! the parts of the graph that don't depend on trainable values (the audio
//! stem, the Engram n-gram memories) run without autodiff.
//!
//! Padding never changes valid positions (attention, convolutions and norms
//! are masked or per position), so a batch is cut to its longest clip and
//! transcript instead of 30 s and 320 tokens.

use burn::tensor::activation::{log_softmax, sigmoid, silu, softmax};
use burn::tensor::module::conv2d;
use burn::tensor::ops::ConvOptions;
use burn::tensor::{Device, Int, Tensor, TensorData};

use cactus_whistle::lora::{Proj, Site};
use cactus_whistle_common::checkpoint::{Array, Checkpoint, Config};
use cactus_whistle_common::export::lora_param;

use super::quant::{Cq, aq, fake_kernel, fake_rows};

/// The 15 LoRA groups, in adapter order.
pub const GROUPS: [(Site, Proj); 15] = {
    use Proj::*;
    use Site::*;
    [
        (Encoder, Q),
        (Encoder, K),
        (Encoder, V),
        (Encoder, Gate),
        (Encoder, Out),
        (DecoderSelf, Q),
        (DecoderSelf, K),
        (DecoderSelf, V),
        (DecoderSelf, Gate),
        (DecoderSelf, Out),
        (Cross, Q),
        (Cross, K),
        (Cross, V),
        (Cross, Gate),
        (Cross, Out),
    ]
};

fn group(site: Site, proj: Proj) -> usize {
    site as usize * 5 + proj as usize
}

const NEG: f32 = -1e9;

// ------------------------------------------------------------ helpers

fn t1(a: &[f32], d: &Device) -> Tensor<1> {
    Tensor::from_data(TensorData::new(a.to_vec(), [a.len()]), d)
}
fn t2(a: Vec<f32>, r: usize, c: usize, d: &Device) -> Tensor<2> {
    Tensor::from_data(TensorData::new(a, [r, c]), d)
}

/// `x @ w` over the last axis.
fn lin(x: Tensor<3>, w: Tensor<2>) -> Tensor<3> {
    let [b, t, i] = x.dims();
    let o = w.dims()[1];
    x.reshape([b * t, i]).matmul(w).reshape([b, t, o])
}

/// ZCRMSNorm: `(1 + scale) * x / rms(x)`, `one_plus` already `1 + scale`.
fn zcrms<const D: usize>(x: Tensor<D>, one_plus: &Tensor<1>) -> Tensor<D> {
    let ms = x.clone().square().mean_dim(D - 1).add_scalar(1e-6);
    x.div(ms.sqrt()).mul(one_plus.clone().unsqueeze::<D>())
}

fn rms_unit<const D: usize>(x: Tensor<D>) -> Tensor<D> {
    let ms = x.clone().square().mean_dim(D - 1).add_scalar(1e-6);
    x.div(ms.sqrt())
}

/// `x` shifted `j` steps later along axis 1 (zeros in front).
fn shift_right(x: Tensor<3>, j: usize) -> Tensor<3> {
    if j == 0 {
        return x;
    }
    let [b, t, c] = x.dims();
    if j >= t {
        return x.zeros_like();
    }
    let z = Tensor::<3>::zeros([b, j, c], &x.device());
    Tensor::cat(vec![z, x.narrow(1, 0, t - j)], 1)
}

/// Rotary embedding over the head dim (half split), `cos`/`sin` `[T, hd/2]`.
fn rope(x: Tensor<4>, cos: &Tensor<2>, sin: &Tensor<2>) -> Tensor<4> {
    let hd = x.dims()[3];
    let h = hd / 2;
    let (c, s) = (cos.clone().unsqueeze::<4>(), sin.clone().unsqueeze::<4>());
    let x1 = x.clone().narrow(3, 0, h);
    let x2 = x.narrow(3, h, h);
    Tensor::cat(
        vec![
            x1.clone() * c.clone() - x2.clone() * s.clone(),
            x2 * c + x1 * s,
        ],
        3,
    )
}

fn rope_tables(hd: usize, t: usize, theta: f32, d: &Device) -> (Tensor<2>, Tensor<2>) {
    let h = hd / 2;
    let (mut c, mut s) = (Vec::with_capacity(t * h), Vec::with_capacity(t * h));
    for p in 0..t {
        for i in 0..h {
            let f = 1.0 / theta.powf((2 * i) as f32 / hd as f32);
            let a = p as f32 * f;
            c.push(a.cos());
            s.push(a.sin());
        }
    }
    (t2(c, t, h, d), t2(s, t, h, d))
}

/// Multi-head attention with grouped KV heads: `q` `[B,H,Sq,d]`, `k`
/// `[B,Hk,Sk,d]`, `v` `[B,Hk,Sk,dv]`, `bias` broadcast to the scores. Returns
/// `[B, Sq, H*dv]`.
fn attend(q: Tensor<4>, k: Tensor<4>, v: Tensor<4>, bias: Tensor<4>) -> Tensor<3> {
    let [b, h, sq, d] = q.dims();
    let [_, hk, sk, dv] = v.dims();
    let (k, v) = if hk == h {
        (k, v)
    } else {
        let g = h / hk;
        (
            k.unsqueeze_dim::<5>(2)
                .expand([b, hk, g, sk, d])
                .reshape([b, h, sk, d]),
            v.unsqueeze_dim::<5>(2)
                .expand([b, hk, g, sk, dv])
                .reshape([b, h, sk, dv]),
        )
    };
    let scores = q
        .matmul(k.swap_dims(2, 3))
        .mul_scalar(1.0 / (d as f32).sqrt())
        + bias;
    softmax(scores, 3)
        .matmul(v)
        .swap_dims(1, 2)
        .reshape([b, sq, h * dv])
}

// ------------------------------------------------------------ weights

struct Mlp {
    cond_v: Tensor<2>,
    cond_u: Tensor<2>,
    m1: Tensor<2>,
    d2: Tensor<1>,
    b2: Tensor<1>,
    m2: Tensor<2>,
}

struct Attn {
    q_norm: Tensor<1>,
    k_norm: Tensor<1>,
}

struct EncLayer {
    pre_hada_0: Tensor<1>,
    mlp0: Mlp,
    norm_in: Tensor<1>,
    attn: Attn,
    post_attn: Tensor<1>,
    attn_gate: f32,
    conv_norm: Tensor<1>,
    pw1: Tensor<2>,
    dw: Vec<Tensor<1>>,
    conv_out_norm: Tensor<1>,
    pw2: Tensor<2>,
    pre_hada: Tensor<1>,
    mlp: Mlp,
}

struct DecLayer {
    norm_in: Tensor<1>,
    attn: Attn,
    taps: [Vec<Tensor<1>>; 3],
    post_attn: Tensor<1>,
    attn_gate: f32,
    cross_norm: Tensor<1>,
    cross: Attn,
    post_cross: Tensor<1>,
    cross_gate: f32,
    pre_hada: Tensor<1>,
    mlp: Mlp,
}

struct Mhc {
    /// `[nC, n + n + n*n]`: phi_pre | phi_post | phi_res, quantised.
    phi: Vec<Tensor<2>>,
    a: Vec<[f32; 3]>,
    /// b_pre + pre_off, b_post + post_off, b_res, per layer.
    b_pre: Vec<Tensor<1>>,
    b_post: Vec<Tensor<1>>,
    b_res: Vec<Tensor<2>>,
    final_norm: Tensor<1>,
}

struct Engram {
    tables: Tensor<2>,
    key: Tensor<2>,
    value: Tensor<2>,
    taps: Vec<Tensor<1>>,
}

/// The frozen model on the device.
pub struct Whistle {
    pub cfg: Config,
    device: Device,
    cq: Cq,
    stem_w: Vec<Tensor<4>>,
    stem_pw: Vec<Tensor<4>>,
    stem_out: Tensor<2>,
    enc: Vec<EncLayer>,
    enc_mhc: Mhc,
    dec: Vec<DecLayer>,
    dec_mhc: Mhc,
    engrams: Vec<Engram>,
    pe_gate: f32,
    /// Quantised base embedding, `[text + langs, d]`.
    emb: Tensor<2>,
    /// Float LoRA target kernels per group, `[L, in, out]`.
    lora_base: Vec<Tensor<3>>,
    pub lora_dims: Vec<(usize, usize, usize)>,
    /// Added vocabulary: pieces, new languages.
    pub pieces: usize,
    pub new_langs: usize,
}

/// The trained values.
#[derive(Clone)]
pub struct Train {
    pub a: Vec<Tensor<3>>,
    pub b: Vec<Tensor<3>>,
    pub rows: Tensor<2>,
}

fn get<'a>(ck: &'a Checkpoint, name: &str) -> Result<&'a Array, String> {
    ck.get(name)
}

fn one_plus(a: &Array, d: &Device) -> Tensor<1> {
    t1(&a.data.iter().map(|v| 1.0 + v).collect::<Vec<_>>(), d)
}

fn layer_one_plus(ck: &Checkpoint, name: &str, l: usize, d: &Device) -> Result<Tensor<1>, String> {
    Ok(one_plus(&get(ck, name)?.index(l), d))
}

fn scalar(ck: &Checkpoint, name: &str, l: usize) -> Result<f32, String> {
    Ok(get(ck, name)?.data[l])
}

/// A frozen `[in, out]` kernel, CQ-W4 fake-quantised.
fn qkernel(a: &Array, d: &Device) -> Tensor<2> {
    let (i, o) = (a.shape[0], a.shape[1]);
    t2(fake_kernel(&a.data, 1, i, o), i, o, d)
}

/// The Hadamard MLP folded into two matrices:
/// `silu(d2 * cond * (x @ M1) + b2) @ M2` with `M1 = diag(d1) (Wa⊗Wb)[:, p1]`,
/// `M2 = (Wa⊗Wb)[:, p2] diag(d3) (Wa⊗Wb) diag(d4)`.
fn mlp(
    ck: &Checkpoint,
    prefix: &str,
    l: usize,
    p1: &[usize],
    p2: &[usize],
    d: &Device,
) -> Result<Mlp, String> {
    let g = |n: &str| -> Result<Array, String> { Ok(get(ck, &format!("{prefix}/{n}"))?.index(l)) };
    let n = p1.len();
    let kron = |a: &Array, b: &Array| -> Vec<f64> {
        let (na, nb) = (a.shape[0], b.shape[0]);
        let mut k = vec![0f64; n * n];
        for i in 0..na {
            for j in 0..nb {
                for kk in 0..na {
                    for ll in 0..nb {
                        k[(i * nb + j) * n + kk * nb + ll] =
                            a.data[i * na + kk] as f64 * b.data[j * nb + ll] as f64;
                    }
                }
            }
        }
        k
    };
    let (d1, d3, d4) = (g("d1")?, g("d3")?, g("d4")?);
    let k1 = kron(&g("w1a")?, &g("w1b")?);
    let k2 = kron(&g("w2a")?, &g("w2b")?);
    let k3 = kron(&g("w3a")?, &g("w3b")?);
    let mut m1 = vec![0f32; n * n];
    for r in 0..n {
        for c in 0..n {
            m1[r * n + c] = (d1.data[r] as f64 * k1[r * n + p1[c]]) as f32;
        }
    }
    // A = K2[:, p2] diag(d3); M2 = A K3 diag(d4).
    let mut a = vec![0f64; n * n];
    for r in 0..n {
        for c in 0..n {
            a[r * n + c] = k2[r * n + p2[c]] * d3.data[c] as f64;
        }
    }
    let mut m2 = vec![0f64; n * n];
    for r in 0..n {
        for k in 0..n {
            let av = a[r * n + k];
            if av == 0.0 {
                continue;
            }
            let row = &k3[k * n..(k + 1) * n];
            for (o, kv) in m2[r * n..(r + 1) * n].iter_mut().zip(row) {
                *o += av * kv;
            }
        }
    }
    for r in 0..n {
        for c in 0..n {
            m2[r * n + c] *= d4.data[c] as f64;
        }
    }
    let cv = g("cond_v")?;
    let cu = g("cond_u")?;
    Ok(Mlp {
        cond_v: t2(cv.data.clone(), cv.shape[0], cv.shape[1], d),
        cond_u: t2(cu.data.clone(), cu.shape[0], cu.shape[1], d),
        m1: t2(m1, n, n, d),
        d2: t1(&g("d2")?.data, d),
        b2: t1(&g("b2")?.data, d),
        m2: t2(m2.iter().map(|&v| v as f32).collect(), n, n, d),
    })
}

fn mhc(ck: &Checkpoint, prefix: &str, layers: usize, d: &Device) -> Result<Mhc, String> {
    let phi = |n: &str| get(ck, &format!("{prefix}/{n}"));
    let (pre, post, res) = (
        phi("mhc_phi_pre")?,
        phi("mhc_phi_post")?,
        phi("mhc_phi_res")?,
    );
    let n = pre.shape[2];
    let nc = pre.shape[1];
    let w = n + n + n * n;
    let mut phis = Vec::new();
    let mut a = Vec::new();
    let (mut b_pre, mut b_post, mut b_res) = (Vec::new(), Vec::new(), Vec::new());
    for l in 0..layers {
        let mut cat = vec![0f32; nc * w];
        for (src, off, k) in [(pre, 0, n), (post, n, n), (res, 2 * n, n * n)] {
            let q = fake_kernel(&src.index(l).data, 1, nc, k);
            for r in 0..nc {
                cat[r * w + off..r * w + off + k].copy_from_slice(&q[r * k..(r + 1) * k]);
            }
        }
        phis.push(t2(cat, nc, w, d));
        a.push([
            scalar(ck, &format!("{prefix}/mhc_a_pre"), l)?,
            scalar(ck, &format!("{prefix}/mhc_a_post"), l)?,
            scalar(ck, &format!("{prefix}/mhc_a_res"), l)?,
        ]);
        let lane = l % n;
        let bp = get(ck, &format!("{prefix}/mhc_b_pre"))?.index(l);
        let bq = get(ck, &format!("{prefix}/mhc_b_post"))?.index(l);
        let br = get(ck, &format!("{prefix}/mhc_b_res"))?.index(l);
        b_pre.push(t1(
            &(0..n)
                .map(|i| bp.data[i] + if i == lane { 4.0 } else { -4.0 })
                .collect::<Vec<_>>(),
            d,
        ));
        b_post.push(t1(
            &(0..n)
                .map(|i| bq.data[i] + if i == lane { 0.0 } else { -4.0 })
                .collect::<Vec<_>>(),
            d,
        ));
        b_res.push(t2(br.data.clone(), n, n, d));
    }
    Ok(Mhc {
        phi: phis,
        a,
        b_pre,
        b_post,
        b_res,
        final_norm: one_plus(get(ck, &format!("{prefix}/final_norm/scale"))?, d),
    })
}

/// `[3, 3, 1, C]` (HWIO) depthwise kernel as `[C, 1, 3, 3]` (OIHW).
fn stem_kernel(a: &Array, d: &Device) -> Tensor<4> {
    let c = a.shape[3];
    let mut w = vec![0f32; c * 9];
    for h in 0..3 {
        for x in 0..3 {
            for o in 0..c {
                w[o * 9 + h * 3 + x] = a.data[(h * 3 + x) * c + o];
            }
        }
    }
    Tensor::from_data(TensorData::new(w, [c, 1, 3, 3]), d)
}

impl Whistle {
    /// The frozen model, from the float checkpoint and the Hadamard
    /// permutations (`hada_p1`/`hada_p2` of a `.cact`), for a vocabulary with
    /// `pieces` added text pieces and `new_langs` added languages.
    pub fn load(
        ck: &Checkpoint,
        perms: (&[usize], &[usize]),
        pieces: usize,
        new_langs: usize,
        device: &Device,
    ) -> Result<Whistle, String> {
        let cfg = ck.config.clone();
        let d = device;
        let (p1, p2) = perms;
        if p1.len() != cfg.dim || p2.len() != cfg.dim || !cfg.dim.is_power_of_two() {
            return Err("the Hadamard MLP permutations don't match the model width".into());
        }

        let stem = |n: &str| get(ck, &format!("stem/{n}"));
        let mut stem_w = vec![stem_kernel(stem("w")?, d)];
        let mut stem_pw = Vec::new();
        for i in 1..cfg.stem_stages {
            stem_w.push(stem_kernel(stem(&format!("dw_{i}"))?, d));
            let k = stem(&format!("pw_{i}/kernel"))?;
            let (ci, co) = (k.shape[0], k.shape[1]);
            let q = fake_kernel(&k.data, 1, ci, co);
            // [in, out] -> [out, in, 1, 1]
            let mut w = vec![0f32; ci * co];
            for i2 in 0..ci {
                for o in 0..co {
                    w[o * ci + i2] = q[i2 * co + o];
                }
            }
            stem_pw.push(Tensor::from_data(TensorData::new(w, [co, ci, 1, 1]), d));
        }
        let stem_out = qkernel(stem("out/kernel")?, d);

        let e = "encoder/layers/block";
        let mut enc = Vec::new();
        for l in 0..cfg.enc_layers {
            let dw = get(ck, &format!("{e}/dw"))?.index(l);
            let c = cfg.dim;
            enc.push(EncLayer {
                pre_hada_0: layer_one_plus(ck, &format!("{e}/pre_hada_norm_0/scale"), l, d)?,
                mlp0: mlp(ck, &format!("{e}/hadamard_mlp_0"), l, p1, p2, d)?,
                norm_in: layer_one_plus(ck, &format!("{e}/ZCRMSNorm_0/scale"), l, d)?,
                attn: Attn {
                    q_norm: layer_one_plus(ck, &format!("{e}/self_attn/q_norm/scale"), l, d)?,
                    k_norm: layer_one_plus(ck, &format!("{e}/self_attn/k_norm/scale"), l, d)?,
                },
                post_attn: layer_one_plus(ck, &format!("{e}/post_attn_norm/scale"), l, d)?,
                attn_gate: scalar(ck, &format!("{e}/attn_gate"), l)?,
                conv_norm: layer_one_plus(ck, &format!("{e}/conv_norm/scale"), l, d)?,
                pw1: qkernel(&get(ck, &format!("{e}/pw1/kernel"))?.index(l), d),
                dw: (0..cfg.conv_kernel)
                    .map(|j| t1(&dw.data[j * c..(j + 1) * c], d))
                    .collect(),
                conv_out_norm: layer_one_plus(ck, &format!("{e}/conv_out_norm/scale"), l, d)?,
                pw2: qkernel(&get(ck, &format!("{e}/pw2/kernel"))?.index(l), d),
                pre_hada: layer_one_plus(ck, &format!("{e}/pre_hada_norm/scale"), l, d)?,
                mlp: mlp(ck, &format!("{e}/hadamard_mlp"), l, p1, p2, d)?,
            });
        }
        let enc_mhc = mhc(ck, "encoder", cfg.enc_layers, d)?;

        let b = "stack/layers/block";
        let mut dec = Vec::new();
        for l in 0..cfg.dec_layers {
            let taps = |n: &str| -> Result<Vec<Tensor<1>>, String> {
                let a = get(ck, &format!("{b}/self_attn/{n}"))?.index(l);
                let w = a.shape[1];
                Ok((0..a.shape[0])
                    .map(|j| t1(&a.data[j * w..(j + 1) * w], d))
                    .collect())
            };
            dec.push(DecLayer {
                norm_in: layer_one_plus(ck, &format!("{b}/ZCRMSNorm_0/scale"), l, d)?,
                attn: Attn {
                    q_norm: layer_one_plus(ck, &format!("{b}/self_attn/q_norm/scale"), l, d)?,
                    k_norm: layer_one_plus(ck, &format!("{b}/self_attn/k_norm/scale"), l, d)?,
                },
                taps: [taps("q_taps")?, taps("k_taps")?, taps("v_taps")?],
                post_attn: layer_one_plus(ck, &format!("{b}/post_attn_norm/scale"), l, d)?,
                attn_gate: scalar(ck, &format!("{b}/attn_gate"), l)?,
                cross_norm: layer_one_plus(ck, &format!("{b}/cross_norm/scale"), l, d)?,
                cross: Attn {
                    q_norm: layer_one_plus(ck, &format!("{b}/cross_attn/q_norm/scale"), l, d)?,
                    k_norm: layer_one_plus(ck, &format!("{b}/cross_attn/k_norm/scale"), l, d)?,
                },
                post_cross: layer_one_plus(ck, &format!("{b}/post_cross_norm/scale"), l, d)?,
                cross_gate: scalar(ck, &format!("{b}/cross_gate"), l)?,
                pre_hada: layer_one_plus(ck, &format!("{b}/pre_hada_norm/scale"), l, d)?,
                mlp: mlp(ck, &format!("{b}/hadamard_mlp"), l, p1, p2, d)?,
            });
        }
        let dec_mhc = mhc(ck, "stack", cfg.dec_layers, d)?;

        let (_, sub) = cfg.engram_geometry();
        let mut engrams = Vec::new();
        for s in 0..cfg.sites.len() {
            let g = |n: &str| get(ck, &format!("engrams_{s}/{n}"));
            let t = g("embedding")?;
            let rows = t.len() / sub;
            let taps = g("taps")?;
            let w = taps.shape[1];
            engrams.push(Engram {
                tables: t2(fake_rows(&t.data, rows, sub), rows, sub, d),
                key: qkernel(g("key_proj/kernel")?, d),
                value: qkernel(g("value_proj/kernel")?, d),
                taps: (0..taps.shape[0])
                    .map(|j| t1(&taps.data[j * w..(j + 1) * w], d))
                    .collect(),
            });
        }

        let emb = get(ck, "embedding/embedding")?;
        let v0 = emb.shape[0];
        if v0 != cfg.vocab() {
            return Err(format!(
                "embedding has {v0} rows, config says {}",
                cfg.vocab()
            ));
        }
        let emb = t2(fake_rows(&emb.data, v0, cfg.dim), v0, cfg.dim, d);

        let mut lora_base = Vec::new();
        let mut lora_dims = Vec::new();
        for (site, proj) in GROUPS {
            let a = get(ck, &lora_param(site, proj))?;
            let (l, i, o) = (a.shape[0], a.shape[1], a.shape[2]);
            lora_base.push(Tensor::from_data(
                TensorData::new(a.data.clone(), [l, i, o]),
                d,
            ));
            lora_dims.push((l, i, o));
        }

        Ok(Whistle {
            pe_gate: get(ck, "pe_gate")?.data[0],
            cfg,
            device: device.clone(),
            cq: Cq::new(d),
            stem_w,
            stem_pw,
            stem_out,
            enc,
            enc_mhc,
            dec,
            dec_mhc,
            engrams,
            emb,
            lora_base,
            lora_dims,
            pieces,
            new_langs,
        })
    }

    /// Effective (merged, quantised) LoRA kernels `[L, in, out]` per group.
    /// `keep` is the rank-dropout mask per group, `[L, 1, r]`, already scaled.
    pub fn effective(&self, tr: &Train, scale: f32, keep: Option<&[Tensor<3>]>) -> Vec<Tensor<3>> {
        (0..GROUPS.len())
            .map(|g| {
                let a = match keep {
                    Some(k) => tr.a[g].clone() * k[g].clone(),
                    None => tr.a[g].clone(),
                };
                let w = self.lora_base[g].clone() + a.matmul(tr.b[g].clone()).mul_scalar(scale);
                self.cq.ste_kernels(w)
            })
            .collect()
    }

    /// Like [`Whistle::effective`], but with the quantised values given
    /// (`[L, in, out]` per group, computed elsewhere), straight-through.
    pub fn effective_given(&self, tr: &Train, scale: f32, q: &[Tensor<3>]) -> Vec<Tensor<3>> {
        (0..GROUPS.len())
            .map(|g| {
                let w = self.lora_base[g].clone()
                    + tr.a[g].clone().matmul(tr.b[g].clone()).mul_scalar(scale);
                let wd = w.clone().detach();
                w + (q[g].clone() - wd).detach()
            })
            .collect()
    }

    /// The quantised embedding with the trained rows spliced in:
    /// `[base text | pieces | base tags | new tags]`.
    pub fn embedding(&self, rows: &Tensor<2>) -> Tensor<2> {
        let text = self.cfg.text;
        let v0 = self.cfg.vocab();
        if rows.dims()[0] == 0 {
            return self.emb.clone();
        }
        let rq = self.cq.ste_rows(rows.clone());
        let mut parts = vec![self.emb.clone().narrow(0, 0, text)];
        if self.pieces > 0 {
            parts.push(rq.clone().narrow(0, 0, self.pieces));
        }
        parts.push(self.emb.clone().narrow(0, text, v0 - text));
        if self.new_langs > 0 {
            parts.push(rq.narrow(0, self.pieces, self.new_langs));
        }
        Tensor::cat(parts, 0)
    }

    // -------------------------------------------------------- encoder

    /// The audio stem on `[B, T, mels]` log-mel features with `frames` valid
    /// frames per clip: `[B, T', d]` and the valid lengths. No gradient.
    pub fn stem(&self, feats: Tensor<3>, frames: &[usize]) -> (Tensor<3>, Vec<usize>) {
        let d = &self.device;
        let [b, _, _] = feats.dims();
        let mut x = feats.unsqueeze_dim::<4>(1);
        let mut lens: Vec<usize> = frames.to_vec();
        let mask = |x: Tensor<4>, lens: &[usize]| {
            let t = x.dims()[2];
            let m: Vec<f32> = lens
                .iter()
                .flat_map(|&l| (0..t).map(move |i| if i < l { 1.0 } else { 0.0 }))
                .collect();
            x * Tensor::<4>::from_data(TensorData::new(m, [b, 1, t, 1]), d)
        };
        for (i, w) in self.stem_w.iter().enumerate() {
            if i > 0 {
                x = silu(x);
            }
            let groups = if i == 0 { 1 } else { w.dims()[0] };
            x = conv2d(
                x,
                w.clone(),
                None,
                ConvOptions::new([2, 2], [1, 1], [1, 1], groups),
            );
            lens.iter_mut().for_each(|l| *l = l.div_ceil(2));
            x = mask(x, &lens);
            if i > 0 {
                x = conv2d(
                    x,
                    self.stem_pw[i - 1].clone(),
                    None,
                    ConvOptions::new([1, 1], [0, 0], [1, 1], 1),
                );
            }
        }
        let [_, c, t, f] = x.dims();
        let x = silu(x).swap_dims(1, 2).reshape([b, t, c * f]);
        (lin(aq(x), self.stem_out.clone()), lens)
    }

    fn mlp(&self, m: &Mlp, x: Tensor<3>) -> Tensor<3> {
        let cond = lin(
            softmax(lin(x.clone(), m.cond_v.clone()), 2),
            m.cond_u.clone(),
        )
        .add_scalar(1.0);
        let z = lin(x, m.m1.clone());
        let h = silu(z * m.d2.clone().unsqueeze::<3>() * cond + m.b2.clone().unsqueeze::<3>());
        lin(h, m.m2.clone())
    }

    /// One mHC lane stack: `block(layer, u)` is the residual block.
    fn lanes(
        &self,
        x: Tensor<3>,
        mhc: &Mhc,
        mut block: impl FnMut(usize, Tensor<3>) -> Tensor<3>,
    ) -> Tensor<3> {
        let [b, t, c] = x.dims();
        let n = mhc.b_res[0].dims()[0];
        let mut x4 = x.unsqueeze_dim::<4>(2).expand([b, t, n, c]);
        for l in 0..mhc.phi.len() {
            let nx = aq(rms_unit(x4.clone().reshape([b, t, n * c])));
            let p = lin(nx, mhc.phi[l].clone());
            let [a_pre, a_post, a_res] = mhc.a[l];
            let hpre = sigmoid(
                p.clone().narrow(2, 0, n).mul_scalar(a_pre) + mhc.b_pre[l].clone().unsqueeze::<3>(),
            );
            let u = (hpre.unsqueeze_dim::<4>(3) * x4.clone())
                .sum_dim(2)
                .reshape([b, t, c]);
            let y = block(l, u.clone()) - u;
            let hpost = sigmoid(
                p.clone().narrow(2, n, n).mul_scalar(a_post)
                    + mhc.b_post[l].clone().unsqueeze::<3>(),
            )
            .mul_scalar(2.0);
            let lk = p
                .narrow(2, 2 * n, n * n)
                .mul_scalar(a_res)
                .reshape([b, t, n, n])
                + mhc.b_res[l].clone().unsqueeze::<4>();
            // Sinkhorn: 20 rounds of row then column normalisation in the log
            // domain, as Cactus does. (Dividing by sums in the exp domain is a
            // quarter of the operations, but its backward pass overflows on
            // some batches.)
            let mut lk = lk;
            for _ in 0..20 {
                lk = log_softmax(lk, 3);
                lk = log_softmax(lk, 2);
            }
            let hres = lk.exp();
            x4 = hres.matmul(x4) + hpost.unsqueeze_dim::<4>(3) * y.unsqueeze_dim::<4>(2);
        }
        zcrms(x4.mean_dim(2).reshape([b, t, c]), &mhc.final_norm)
    }

    /// Self attention (`site` Encoder or DecoderSelf) of layer `l`.
    #[allow(clippy::too_many_arguments)]
    fn self_attn(
        &self,
        x: Tensor<3>,
        site: Site,
        l: usize,
        w: &[Tensor<3>],
        norms: &Attn,
        taps: Option<&[Vec<Tensor<1>>; 3]>,
        bias: Tensor<4>,
        rope_t: &(Tensor<2>, Tensor<2>),
    ) -> Tensor<3> {
        let c = &self.cfg;
        let [b, t, _] = x.dims();
        let wl = |p: Proj| {
            let g = group(site, p);
            let (_, i, o) = self.lora_dims[g];
            w[g].clone().narrow(0, l, 1).reshape([i, o])
        };
        let x = aq(x);
        let mut q = lin(x.clone(), wl(Proj::Q));
        let mut k = lin(x.clone(), wl(Proj::K));
        let mut v = lin(x.clone(), wl(Proj::V));
        if let Some([qt, kt, vt]) = taps {
            let tap = |z: Tensor<3>, taps: &[Tensor<1>]| {
                let mut acc = z.clone() * taps[0].clone().unsqueeze::<3>();
                for (j, tw) in taps.iter().enumerate().skip(1) {
                    acc = acc + shift_right(z.clone(), j) * tw.clone().unsqueeze::<3>();
                }
                acc
            };
            q = tap(q, qt);
            k = tap(k, kt);
            v = tap(v, vt);
        }
        let (h, hk) = (c.heads(), c.kv_heads);
        let q = q.reshape([b, t, h, c.qk_dim]).swap_dims(1, 2);
        let k = k.reshape([b, t, hk, c.qk_dim]).swap_dims(1, 2);
        let v = v.reshape([b, t, hk, c.v_dim]).swap_dims(1, 2);
        let q = rope(zcrms(q, &norms.q_norm), &rope_t.0, &rope_t.1);
        let k = rope(zcrms(k, &norms.k_norm), &rope_t.0, &rope_t.1);
        let out = attend(aq(q), aq(k), aq(v), bias);
        let out = out * sigmoid(lin(x, wl(Proj::Gate)));
        lin(aq(out), wl(Proj::Out))
    }

    /// Encoder output `[B, T, d]` (lanes merged, final norm) and valid lengths.
    pub fn encode(
        &self,
        feats: Tensor<3>,
        frames: &[usize],
        w: &[Tensor<3>],
    ) -> (Tensor<3>, Vec<usize>) {
        let d = &self.device;
        let (x, lens) = self.stem(feats, frames);
        let [b, t, c] = x.dims();
        let (x, t) = if t % 2 == 1 {
            (Tensor::cat(vec![x, Tensor::zeros([b, 1, c], d)], 1), t + 1)
        } else {
            (x, t)
        };
        let seen: Vec<f32> = lens
            .iter()
            .flat_map(|&l| (0..t).map(move |i| if i < l { 1.0 } else { 0.0 }))
            .collect();
        let valid = Tensor::<3>::from_data(TensorData::new(seen.clone(), [b, t, 1]), d);
        let bias = Tensor::<4>::from_data(
            TensorData::new(
                seen.iter()
                    .map(|&s| if s > 0.0 { 0.0 } else { NEG })
                    .collect::<Vec<_>>(),
                [b, 1, 1, t],
            ),
            d,
        );
        let rope_t = rope_tables(self.cfg.qk_dim, t, self.cfg.rope, d);
        let k = self.cfg.conv_kernel;
        let enc = self.lanes(x, &self.enc_mhc, |l, x| {
            let ly = &self.enc[l];
            let x = x.clone() + self.mlp(&ly.mlp0, zcrms(x, &ly.pre_hada_0)).mul_scalar(0.5);
            let h = zcrms(x.clone(), &ly.norm_in);
            let h = self.self_attn(
                h,
                Site::Encoder,
                l,
                w,
                &ly.attn,
                None,
                bias.clone(),
                &rope_t,
            );
            let x = x + zcrms(h, &ly.post_attn).mul_scalar(sig(ly.attn_gate));
            let h = lin(aq(zcrms(x.clone(), &ly.conv_norm)), ly.pw1.clone());
            let h = h.clone().narrow(2, 0, c) * sigmoid(h.narrow(2, c, c)) * valid.clone();
            // Depthwise conv along time, "SAME" padding.
            let pad = Tensor::<3>::zeros([b, k / 2, c], d);
            let hp = Tensor::cat(vec![pad.clone(), h, pad], 1);
            let mut acc = hp.clone().narrow(1, 0, t) * ly.dw[0].clone().unsqueeze::<3>();
            for j in 1..k {
                acc = acc + hp.clone().narrow(1, j, t) * ly.dw[j].clone().unsqueeze::<3>();
            }
            let h = silu(zcrms(acc, &ly.conv_out_norm));
            let x = x + lin(aq(h), ly.pw2.clone());
            x.clone() + self.mlp(&ly.mlp, zcrms(x, &ly.pre_hada)).mul_scalar(0.5)
        });
        (enc, lens)
    }

    // -------------------------------------------------------- decoder

    /// Engram keys and values per site for `tokens` `[B, S]` (no gradient).
    fn engram_kv(&self, tokens: &[u32], b: usize, s: usize) -> Vec<(Tensor<3>, Tensor<3>)> {
        let d = &self.device;
        let c = &self.cfg;
        let (heads, sub) = c.engram_geometry();
        let orders = &c.orders;
        let nt = orders.len() * heads;
        let mut idx = vec![0i64; b * s * nt];
        let mut ok = vec![0f32; b * s * nt];
        for bi in 0..b {
            let u = &tokens[bi * s..(bi + 1) * s];
            for t in 0..s {
                for (oi, &order) in orders.iter().enumerate() {
                    for h in 0..heads {
                        let k = oi * heads + h;
                        let seed = 0x9E37_79B9u32.wrapping_mul((oi * heads + h + 1) as u32);
                        let mut acc = seed;
                        for j in 0..order {
                            let v = if t >= j { u[t - j] } else { 0 };
                            acc = (acc ^ v).wrapping_mul(0x0100_0193);
                        }
                        acc ^= acc >> 15;
                        let i = (bi * s + t) * nt + k;
                        idx[i] = (k * c.slots) as i64 + (acc % c.slots as u32) as i64;
                        ok[i] = if t + 1 >= order { 1.0 } else { 0.0 };
                    }
                }
            }
        }
        let idx = Tensor::<2, Int>::from_data(TensorData::new(idx, [b, s * nt]), d);
        let ok = Tensor::<3>::from_data(TensorData::new(ok, [b, s * nt, 1]), d);
        let dil = *orders.iter().max().unwrap_or(&1);
        self.engrams
            .iter()
            .map(|e| {
                let f = burn::tensor::module::embedding(e.tables.clone(), idx.clone()) * ok.clone();
                let f = aq(f.reshape([b, s, nt * sub]));
                let k = lin(f.clone(), e.key.clone());
                let v = lin(f, e.value.clone());
                let mut acc = v.clone() * e.taps[0].clone().unsqueeze::<3>();
                for (j, tw) in e.taps.iter().enumerate().skip(1) {
                    acc = acc + shift_right(v.clone(), j * dil) * tw.clone().unsqueeze::<3>();
                }
                (rms_unit(k), acc)
            })
            .collect()
    }

    /// Cross attention of decoder layer `l` over the memory.
    fn cross_attn(
        &self,
        x: Tensor<3>,
        l: usize,
        w: &[Tensor<3>],
        kv: &(Tensor<4>, Tensor<4>),
        bias: Tensor<4>,
    ) -> Tensor<3> {
        let c = &self.cfg;
        let [b, s, _] = x.dims();
        let wl = |p: Proj| {
            let g = group(Site::Cross, p);
            let (_, i, o) = self.lora_dims[g];
            w[g].clone().narrow(0, l, 1).reshape([i, o])
        };
        let x = aq(x);
        let q = lin(x.clone(), wl(Proj::Q))
            .reshape([b, s, c.heads(), c.qk_dim])
            .swap_dims(1, 2);
        let q = aq(zcrms(q, &self.dec[l].cross.q_norm));
        let out = attend(q, kv.0.clone(), kv.1.clone(), bias);
        let out = out * sigmoid(lin(x, wl(Proj::Gate)));
        lin(aq(out), wl(Proj::Out))
    }

    /// Logits `[B, S, V]` of `tokens` (`[B, S]`, row-major) given the encoder output.
    pub fn decode(
        &self,
        enc: Tensor<3>,
        enc_lens: &[usize],
        tokens: &[u32],
        s: usize,
        w: &[Tensor<3>],
        emb: &Tensor<2>,
    ) -> Tensor<3> {
        let d = &self.device;
        let c = &self.cfg;
        let [b, t, dm] = enc.dims();
        // Memory: encoder output plus gated sinusoids.
        let half = dm / 2;
        let mut pe = Vec::with_capacity(t * dm);
        for p in 0..t {
            let mut row = vec![0f32; dm];
            for i in 0..half {
                let a = p as f32 * (-(10000f32.ln()) / (half - 1) as f32 * i as f32).exp();
                row[i] = a.sin();
                row[half + i] = a.cos();
            }
            pe.extend(row);
        }
        let pe = t2(pe, t, dm, d).mul_scalar(sig(self.pe_gate));
        let mem = aq(enc + pe.unsqueeze::<3>());
        let mem_bias = Tensor::<4>::from_data(
            TensorData::new(
                enc_lens
                    .iter()
                    .flat_map(|&l| (0..t).map(move |i| if i < l { 0.0 } else { NEG }))
                    .collect::<Vec<f32>>(),
                [b, 1, 1, t],
            ),
            d,
        );
        // Cross K/V per layer (they depend on the cross LoRA, not on tokens).
        let kv: Vec<(Tensor<4>, Tensor<4>)> = (0..c.dec_layers)
            .map(|l| {
                let wl = |p: Proj| {
                    let g = group(Site::Cross, p);
                    let (_, i, o) = self.lora_dims[g];
                    w[g].clone().narrow(0, l, 1).reshape([i, o])
                };
                let k = lin(mem.clone(), wl(Proj::K))
                    .reshape([b, t, c.heads(), c.qk_dim])
                    .swap_dims(1, 2);
                let v = lin(mem.clone(), wl(Proj::V))
                    .reshape([b, t, c.heads(), c.v_dim])
                    .swap_dims(1, 2);
                (aq(zcrms(k, &self.dec[l].cross.k_norm)), aq(v))
            })
            .collect();

        let ids = Tensor::<2, Int>::from_data(
            TensorData::new(tokens.iter().map(|&v| v as i64).collect::<Vec<_>>(), [b, s]),
            d,
        );
        let x = burn::tensor::module::embedding(emb.clone(), ids).mul_scalar((dm as f32).sqrt());
        let engram = self.engram_kv(tokens, b, s);
        let mut causal = vec![0f32; s * s];
        for i in 0..s {
            for j in i + 1..s {
                causal[i * s + j] = NEG;
            }
        }
        let causal = Tensor::<4>::from_data(TensorData::new(causal, [1, 1, s, s]), d);
        let rope_t = rope_tables(c.qk_dim, s, c.rope, d);
        let x = self.lanes(x, &self.dec_mhc, |l, x| {
            let ly = &self.dec[l];
            let x = match c.sites.iter().position(|&site| site == l) {
                Some(site) => {
                    let (ek, ev) = &engram[site];
                    let alpha = sigmoid(
                        (rms_unit(x.clone()) * ek.clone())
                            .sum_dim(2)
                            .mul_scalar(1.0 / (dm as f32).sqrt()),
                    );
                    x + alpha * ev.clone()
                }
                None => x,
            };
            let h = zcrms(x.clone(), &ly.norm_in);
            let h = self.self_attn(
                h,
                Site::DecoderSelf,
                l,
                w,
                &ly.attn,
                Some(&ly.taps),
                causal.clone(),
                &rope_t,
            );
            let x = x + zcrms(h, &ly.post_attn).mul_scalar(sig(ly.attn_gate));
            let h = zcrms(x.clone(), &ly.cross_norm);
            let h = self.cross_attn(h, l, w, &kv[l], mem_bias.clone());
            let x = x + zcrms(h, &ly.post_cross).mul_scalar(sig(ly.cross_gate));
            x.clone() + self.mlp(&ly.mlp, zcrms(x, &ly.pre_hada))
        });
        lin(aq(x), emb.clone().transpose())
    }

    /// Mean cross-entropy of `targets` (`[B, S]`, pad 0 ignored) under `logits`.
    pub fn loss(&self, logits: Tensor<3>, targets: &[u32]) -> Tensor<1> {
        let d = &self.device;
        let [b, s, _] = logits.dims();
        let mask: Vec<f32> = targets
            .iter()
            .map(|&t| if t != self.cfg.pad { 1.0 } else { 0.0 })
            .collect();
        let count = mask.iter().sum::<f32>().max(1.0);
        let idx = Tensor::<3, Int>::from_data(
            TensorData::new(
                targets.iter().map(|&v| v as i64).collect::<Vec<_>>(),
                [b, s, 1],
            ),
            d,
        );
        let mask = Tensor::<3>::from_data(TensorData::new(mask, [b, s, 1]), d);
        let picked = log_softmax(logits, 2).gather(2, idx);
        (picked * mask).sum().mul_scalar(-1.0 / count)
    }
}

fn sig(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
}
