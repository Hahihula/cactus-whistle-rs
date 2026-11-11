//! Log-mel features on the CPU, as Whistle's `features` computes them on the
//! device (`jax.scipy.signal.stft`, boundary zeros, not padded, symmetric
//! Hann window, power through the mel filterbank, `log(. + 2^-24)`, then each
//! clip normalised over its valid frames, sample std).

use std::sync::Arc;

use rustfft::num_complex::Complex32;
use rustfft::{Fft, FftPlanner};

pub const HOP: usize = 160;
pub const WIN: usize = 400;
pub const NFFT: usize = 512;
pub const BINS: usize = NFFT / 2 + 1;

pub struct Frontend {
    window: Vec<f32>,
    /// `[BINS, mels]`.
    mel: Vec<f32>,
    pub mels: usize,
    fft: Arc<dyn Fft<f32>>,
}

impl Frontend {
    /// `mel` is the `[257, mels]` filterbank (the `mel` tensor of a `.cact`).
    pub fn new(mel: Vec<f32>) -> Frontend {
        assert_eq!(mel.len() % BINS, 0);
        let mels = mel.len() / BINS;
        // np.hanning: symmetric.
        let window = (0..WIN)
            .map(|n| {
                (0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / (WIN - 1) as f64).cos())
                    as f32
            })
            .collect();
        Frontend {
            window,
            mel,
            mels,
            fft: FftPlanner::new().plan_fft_forward(NFFT),
        }
    }

    /// Valid frames of a clip of `n` samples.
    pub fn frames(n: usize) -> usize {
        n / HOP
    }

    /// Normalised features of `x` into `out` (`[t_out, mels]`, zero past the
    /// valid frames).
    pub fn compute(&self, x: &[f32], out: &mut [f32], t_out: usize) {
        let m = self.mels;
        let valid = Self::frames(x.len()).min(t_out);
        out[..t_out * m].fill(0.0);
        if valid == 0 {
            return;
        }
        let mut buf = vec![Complex32::new(0.0, 0.0); NFFT];
        let mut scratch = vec![Complex32::new(0.0, 0.0); self.fft.get_inplace_scratch_len()];
        let mut power = [0f32; BINS];
        for t in 0..valid {
            // Frame t covers samples [t*HOP - WIN/2, t*HOP + WIN/2) of the zero-padded clip.
            for (k, c) in buf.iter_mut().enumerate() {
                let v = if k < WIN {
                    let s = (t * HOP + k) as isize - (WIN / 2) as isize;
                    if s >= 0 && (s as usize) < x.len() {
                        x[s as usize] * self.window[k]
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };
                *c = Complex32::new(v, 0.0);
            }
            self.fft.process_with_scratch(&mut buf, &mut scratch);
            for (p, c) in power.iter_mut().zip(&buf) {
                *p = c.re * c.re + c.im * c.im;
            }
            let row = &mut out[t * m..(t + 1) * m];
            row.fill(0.0);
            for (b, &p) in power.iter().enumerate() {
                let w = &self.mel[b * m..(b + 1) * m];
                for (r, &wv) in row.iter_mut().zip(w) {
                    *r += p * wv;
                }
            }
            for r in row.iter_mut() {
                *r = (*r + 2f32.powi(-24)).ln();
            }
        }
        // Per mel channel over the valid frames: mean, sample std.
        let n = valid as f32;
        for j in 0..m {
            let mean = (0..valid).map(|t| out[t * m + j]).sum::<f32>() / n;
            let var = (0..valid)
                .map(|t| (out[t * m + j] - mean).powi(2))
                .sum::<f32>()
                / (valid.max(2) - 1) as f32;
            let std = var.sqrt() + 1e-5;
            for t in 0..valid {
                out[t * m + j] = (out[t * m + j] - mean) / std;
            }
        }
    }
}
