//! Fake quantisation as Cactus trains it (`needle/model/quantize.py`): the
//! forward pass sees what the int8/W4 engine computes, the gradient passes
//! straight through.

use burn::tensor::{Device, FloatDType, Tensor, TensorData};

use cactus_whistle_common::cq;

/// Activation quantisation on (training) or off (Cactus's `quant=False`).
pub static ACT_QUANT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// `x + stop_gradient(q(x) - x)`: per-row symmetric int8 over the last axis
/// (`fake_quant(x, x.shape[-1], 8)`). Used for activations, queries and KV.
pub fn aq<const D: usize>(x: Tensor<D>) -> Tensor<D> {
    if !ACT_QUANT.load(std::sync::atomic::Ordering::Relaxed) {
        return x;
    }
    let xd = x.clone().detach();
    let s = xd
        .clone()
        .abs()
        .max_dim(D - 1)
        .div_scalar(127.0)
        .clamp_min(1e-30);
    let q = xd
        .clone()
        .div(s.clone())
        .round()
        .clamp(-128.0, 127.0)
        .mul(s);
    x + (q - xd).detach()
}

/// The codebook rotation and thresholds on the device.
pub struct Cq {
    h: Tensor<2>,
    lo: f32,
    /// (midpoint between levels k and k+1, level step).
    steps: Vec<(f32, f32)>,
}

impl Cq {
    pub fn new(device: &Device) -> Cq {
        let g = cq::GROUP;
        let mut h = vec![0f32; g * g];
        for (i, row) in h.chunks_exact_mut(g).enumerate() {
            row[i] = 1.0;
            cq::hadamard(row);
        }
        let cb = cq::codebook(4);
        Cq {
            h: Tensor::from_data(TensorData::new(h, [g, g]), device),
            lo: cb[0],
            steps: cb
                .windows(2)
                .map(|w| ((w[0] + w[1]) / 2.0, w[1] - w[0]))
                .collect(),
        }
    }

    /// CQ-W4 of a `[rows, cols]` matrix along its rows (cols a multiple of 128),
    /// straight-through.
    pub fn ste_rows(&self, w: Tensor<2>) -> Tensor<2> {
        let wd = w.clone().detach();
        let [n, d] = wd.dims();
        let g = cq::GROUP;
        assert_eq!(d % g, 0, "CQ needs whole groups of {g}");
        let rot = wd.clone().reshape([n * d / g, g]).matmul(self.h.clone());
        let norm = rot.clone().square().sum_dim(1).sqrt();
        let normq = norm.clone().cast(FloatDType::F16).cast(FloatDType::F32);
        let unit = rot / norm.clamp_min(1e-12);
        let mut lvl = unit.zeros_like().add_scalar(self.lo);
        for &(mid, step) in &self.steps {
            lvl = lvl + unit.clone().greater_elem(mid).float().mul_scalar(step);
        }
        let deq = (lvl * normq).matmul(self.h.clone()).reshape([n, d]);
        w + (deq - wd).detach()
    }

    /// CQ-W4 of stacked `[L, in, out]` kernels along `in`, straight-through.
    pub fn ste_kernels(&self, w: Tensor<3>) -> Tensor<3> {
        let [l, i, o] = w.dims();
        let t = w.swap_dims(1, 2).reshape([l * o, i]);
        self.ste_rows(t).reshape([l, o, i]).swap_dims(1, 2)
    }
}

/// CQ-W4 fake quantisation on the CPU of a `[.., in, out]` kernel along `in`
/// (frozen weights, quantised once).
pub fn fake_kernel(data: &[f32], lead: usize, din: usize, dout: usize) -> Vec<f32> {
    let mut out = vec![0f32; data.len()];
    for l in 0..lead {
        let k = &data[l * din * dout..(l + 1) * din * dout];
        let mut t = vec![0f32; din * dout];
        for i in 0..din {
            for o in 0..dout {
                t[o * din + i] = k[i * dout + o];
            }
        }
        let q = cq::fake_quant(&t, dout, din, 4);
        let dst = &mut out[l * din * dout..(l + 1) * din * dout];
        for i in 0..din {
            for o in 0..dout {
                dst[i * dout + o] = q[o * din + i];
            }
        }
    }
    out
}

/// CQ-W4 fake quantisation on the CPU along the last axis (`[rows, cols]`).
pub fn fake_rows(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    cq::fake_quant(data, rows, cols, 4)
}
