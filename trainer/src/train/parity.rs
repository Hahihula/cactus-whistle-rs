//! `whistle-train parity REF.safetensors CHECKPOINT TEMPLATE`: compares the
//! burn model with values dumped from needle's JAX model on one batch (the
//! features, the encoder output, the loss and every gradient). A development
//! check of the port; the reference comes from `tools/train_parity/ref_model.py`.

use std::collections::HashMap;

use burn::tensor::{Device, Tensor, TensorData};
use safetensors::SafeTensors;

use cactus_whistle_common::checkpoint::Checkpoint;
use cactus_whistle_common::export::{lora_param, template};

use super::features::Frontend;
use super::model::{GROUPS, Train, Whistle};

fn f32s(st: &SafeTensors, n: &str) -> (Vec<usize>, Vec<f32>) {
    let t = st.tensor(n).unwrap_or_else(|e| panic!("{n}: {e}"));
    let v = match t.dtype() {
        safetensors::Dtype::F32 => t
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        safetensors::Dtype::I32 => t
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
            .collect(),
        d => panic!("{n}: {d:?}"),
    };
    (t.shape().to_vec(), v)
}

/// Max abs difference and the relative L2 error.
fn compare(name: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{name}: lengths");
    let (mut maxd, mut num, mut den) = (0f64, 0f64, 0f64);
    for (&g, &w) in got.iter().zip(want) {
        let d = (g as f64 - w as f64).abs();
        maxd = maxd.max(d);
        num += d * d;
        den += (w as f64) * (w as f64);
    }
    println!(
        "{name:<48} max|d| {maxd:10.3e}  rel {:10.3e}",
        (num / den.max(1e-30)).sqrt()
    );
}

fn to_vec<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    t.without_autodiff()
        .into_data()
        .convert::<f32>()
        .try_into_vec::<f32>()
        .unwrap()
}

pub fn run(args: &[String]) -> Result<(), String> {
    // Without activation quantisation compare with JAX's quant=False values.
    let pre = if std::env::var("NOQUANT").is_ok() {
        "nq/"
    } else {
        ""
    };
    let [reference, ck, tpl] = args else {
        return Err(
            "usage: whistle-train parity REF.safetensors CHECKPOINT.safetensors TEMPLATE.cact"
                .into(),
        );
    };
    let bytes = std::fs::read(reference).map_err(|e| format!("{reference}: {e}"))?;
    let (_, header) = SafeTensors::read_metadata(&bytes).map_err(|e| e.to_string())?;
    let meta: HashMap<String, String> = header.metadata().clone().unwrap_or_default();
    let st = SafeTensors::deserialize(&bytes).map_err(|e| e.to_string())?;
    let scale: f32 = meta["scale"].parse().unwrap();
    let pieces: Vec<String> = serde_json::from_str(&meta["pieces"]).unwrap();

    let ck = Checkpoint::load(ck)?;
    let tblob = std::fs::read(tpl).map_err(|e| format!("{tpl}: {e}"))?;
    let tc = cactus_whistle::cact::Cact::open(&tblob).map_err(|e| e.to_string())?;
    let t = template(&tc)?;
    let (p1, p2) = t.perms();
    let fe = Frontend::new(t.mel());

    if std::env::var("NOQUANT").is_ok() {
        super::quant::ACT_QUANT.store(false, std::sync::atomic::Ordering::Relaxed);
        println!("activation quantisation OFF (compare with loss_noquant)");
    }
    let device = Device::default();
    let adev = device.clone().autodiff();
    let t0 = std::time::Instant::now();
    let model = Whistle::load(&ck, (&p1, &p2), pieces.len(), 1, &device)?;
    println!("loaded in {:.1} s", t0.elapsed().as_secs_f64());

    // Batch.
    let (ws, wav) = f32s(&st, "wav");
    let (_, lengths) = f32s(&st, "lengths");
    let (ts, toks) = f32s(&st, "tokens");
    let (b, n) = (ws[0], ws[1]);
    let lens: Vec<usize> = lengths.iter().map(|&v| v as usize).collect();

    // Features: compare the valid frames.
    let (fs, want_feats) = f32s(&st, "features");
    let (tf, m) = (fs[1], fs[2]);
    let frames: Vec<usize> = lens.iter().map(|&l| Frontend::frames(l)).collect();
    let t_crop = frames.iter().max().unwrap().div_ceil(64) * 64;
    let mut feats = vec![0f32; b * t_crop * m];
    let (mut got_v, mut want_v) = (Vec::new(), Vec::new());
    for i in 0..b {
        let x = &wav[i * n..i * n + lens[i]];
        fe.compute(x, &mut feats[i * t_crop * m..(i + 1) * t_crop * m], t_crop);
        for t in 0..frames[i] {
            got_v.extend_from_slice(&feats[(i * t_crop + t) * m..(i * t_crop + t + 1) * m]);
            want_v.extend_from_slice(&want_feats[(i * tf + t) * m..(i * tf + t + 1) * m]);
        }
    }
    compare("features (valid frames)", &got_v, &want_v);
    // From here on use JAX's features, to test the model alone.
    let mut jf = vec![0f32; b * t_crop * m];
    for i in 0..b {
        for t in 0..t_crop.min(tf) {
            jf[(i * t_crop + t) * m..(i * t_crop + t + 1) * m]
                .copy_from_slice(&want_feats[(i * tf + t) * m..(i * tf + t + 1) * m]);
        }
    }

    // Trainable values.
    let mut a = Vec::new();
    let mut bm = Vec::new();
    for (site, proj) in GROUPS {
        let p = lora_param(site, proj);
        let (sa, va) = f32s(&st, &format!("lora/{p}/A"));
        let (sb, vb) = f32s(&st, &format!("lora/{p}/B"));
        a.push(
            Tensor::<3>::from_data(TensorData::new(va, [sa[0], sa[1], sa[2]]), &adev)
                .require_grad(),
        );
        bm.push(
            Tensor::<3>::from_data(TensorData::new(vb, [sb[0], sb[1], sb[2]]), &adev)
                .require_grad(),
        );
    }
    let (rs, rv) = f32s(&st, "rows");
    let rows = Tensor::<2>::from_data(TensorData::new(rv, [rs[0], rs[1]]), &adev).require_grad();
    let tr = Train { a, b: bm, rows };

    // Quantised weights: on the device, and exactly on the CPU.
    let w_gpu = model.effective(&tr, scale, None);
    let mut qcpu = Vec::new();
    for (g, (site, proj)) in GROUPS.iter().enumerate() {
        let base = ck.get(&lora_param(*site, *proj))?;
        let (l, i, o) = (base.shape[0], base.shape[1], base.shape[2]);
        let av = to_vec(tr.a[g].clone());
        let bv = to_vec(tr.b[g].clone());
        let r = av.len() / (l * i);
        let mut w = base.data.clone();
        for li in 0..l {
            for ii in 0..i {
                for oo in 0..o {
                    let mut acc = 0f32;
                    for k in 0..r {
                        acc += av[(li * i + ii) * r + k] * bv[(li * r + k) * o + oo];
                    }
                    w[(li * i + ii) * o + oo] += scale * acc;
                }
            }
        }
        let q = super::quant::fake_kernel(&w, l, i, o);
        qcpu.push(Tensor::<3>::from_data(
            TensorData::new(q, [l, i, o]),
            &device,
        ));
    }
    let (_, want_q) = f32s(&st, "q/encoder_q_proj");
    compare(
        "merged CQ-W4 encoder q_proj (device)",
        &to_vec(w_gpu[0].clone()),
        &want_q,
    );
    compare(
        "merged CQ-W4 encoder q_proj (CPU)",
        &to_vec(qcpu[0].clone()),
        &want_q,
    );
    let w = if std::env::var("GPUQ").is_ok() {
        w_gpu
    } else {
        println!("using the CPU-quantised LoRA weights (GPUQ=1 for the device ones)");
        model.effective_given(&tr, scale, &qcpu)
    };
    let emb = model.embedding(&tr.rows);
    let (_, want_e) = f32s(&st, "q/embedding");
    compare("CQ-W4 embedding (with rows)", &to_vec(emb.clone()), &want_e);

    // Encoder.
    let feats_t = Tensor::<3>::from_data(TensorData::new(jf, [b, t_crop, m]), &device);
    let (enc, enc_lens) = model.encode(feats_t, &frames, &w);
    let (es, want_enc) = f32s(&st, &format!("{pre}enc"));
    let te = enc.dims()[1];
    let ev = to_vec(enc.clone());
    let (mut g2, mut w2) = (Vec::new(), Vec::new());
    for i in 0..b {
        for t in 0..enc_lens[i] {
            g2.extend_from_slice(&ev[(i * te + t) * es[2]..(i * te + t + 1) * es[2]]);
            w2.extend_from_slice(&want_enc[(i * es[1] + t) * es[2]..(i * es[1] + t + 1) * es[2]]);
        }
    }
    println!("encoder lengths {enc_lens:?}");
    compare("encoder output (valid frames)", &g2, &w2);

    // Decoder and loss.
    let s_full = ts[1] - 1;
    let used = (0..b)
        .map(|i| {
            (0..ts[1])
                .rev()
                .find(|&j| toks[i * ts[1] + j] != 0.0)
                .unwrap_or(0)
                + 1
        })
        .max()
        .unwrap();
    let s = (used - 1).div_ceil(16) * 16;
    let s = s.min(s_full);
    let mut input = vec![0u32; b * s];
    let mut target = vec![0u32; b * s];
    for i in 0..b {
        for j in 0..s {
            input[i * s + j] = toks[i * ts[1] + j] as u32;
            target[i * s + j] = toks[i * ts[1] + j + 1] as u32;
        }
    }
    let logits = model.decode(enc, &enc_lens, &input, s, &w, &emb);
    let (ls, want_logits) = f32s(&st, &format!("{pre}logits_head"));
    let lv = to_vec(logits.clone().narrow(1, 0, ls[1]));
    compare("logits (first 24 positions)", &lv, &want_logits);
    let loss = model.loss(logits, &target);
    let (_, want_loss) = f32s(&st, "loss");
    let lval = to_vec(loss.clone())[0];
    println!(
        "loss {lval:.6}  (JAX {:.6}, without activation quantisation {:.6})",
        want_loss[0],
        f32s(&st, "loss_noquant").1[0]
    );
    let t1 = std::time::Instant::now();
    let grads = loss.backward();
    println!("backward {:.2} s", t1.elapsed().as_secs_f64());
    for (g, (site, proj)) in GROUPS.iter().enumerate() {
        let p = lora_param(*site, *proj);
        let ga = tr.a[g].grad(&grads).unwrap();
        let gb = tr.b[g].grad(&grads).unwrap();
        compare(
            &format!("grad {p} A"),
            &to_vec(ga),
            &f32s(&st, &format!("{pre}grad/{p}/A")).1,
        );
        compare(
            &format!("grad {p} B"),
            &to_vec(gb),
            &f32s(&st, &format!("{pre}grad/{p}/B")).1,
        );
    }
    let gr = tr.rows.grad(&grads).unwrap();
    compare(
        "grad rows",
        &to_vec(gr),
        &f32s(&st, &format!("{pre}grad/rows")).1,
    );
    Ok(())
}
