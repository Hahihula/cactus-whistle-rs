//! Codebook quantisation (CQ), the weight format of `.cact` files, as Cactus's
//! exporter writes it.
//!
//! Each row is cut into groups of `group` (128) weights. A group is rotated by
//! the normalised Walsh-Hadamard matrix `H`, its L2 norm kept as f16, and every
//! element of the unit vector replaced by the index of the nearest codebook
//! level. `w ≈ (codebook[idx] * norm) @ H`.

/// Lloyd-Max levels of a unit Gaussian for 2, 3 and 4 bits, divided by
/// sqrt(128): `cb2[4] | cb3[8] | cb4[16]`, as every `.cact` header stores them.
pub const CODEBOOK: [f32; 28] = [
    // 2 bits
    f32::from_bits(0xbe085e47),
    f32::from_bits(0xbd237060),
    f32::from_bits(0x3d23fc09),
    f32::from_bits(0x3e08a9db),
    // 3 bits
    f32::from_bits(0xbe41cdbb),
    f32::from_bits(0xbdf16db0),
    f32::from_bits(0xbd87797d),
    f32::from_bits(0xbcac246c),
    f32::from_bits(0x3cb50d39),
    f32::from_bits(0x3d88c69c),
    f32::from_bits(0x3df23363),
    f32::from_bits(0x3e4261c1),
    // 4 bits
    f32::from_bits(0xbe7547c2),
    f32::from_bits(0xbe390b02),
    f32::from_bits(0xbe105747),
    f32::from_bits(0xbddf41f9),
    f32::from_bits(0xbda6f045),
    f32::from_bits(0xbd67898a),
    f32::from_bits(0xbd06eb34),
    f32::from_bits(0xbc25be0a),
    f32::from_bits(0x3c5040f6),
    f32::from_bits(0x3d124277),
    f32::from_bits(0x3d738a80),
    f32::from_bits(0x3dae086f),
    f32::from_bits(0x3de6e514),
    f32::from_bits(0x3e142b6e),
    f32::from_bits(0x3e3ca926),
    f32::from_bits(0x3e781699),
];

pub const GROUP: usize = 128;

pub fn codebook(bits: u32) -> &'static [f32] {
    match bits {
        2 => &CODEBOOK[0..4],
        3 => &CODEBOOK[4..12],
        4 => &CODEBOOK[12..28],
        _ => panic!("CQ codebooks exist for 2, 3 and 4 bits, not {bits}"),
    }
}

/// `x @ H` for one group, H = Walsh(n) / sqrt(n) in Sylvester order (the fast
/// transform; rows of `H` are its columns, it is symmetric).
pub fn hadamard(x: &mut [f32]) {
    let n = x.len();
    debug_assert!(n.is_power_of_two());
    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(2 * h) {
            for j in i..i + h {
                let (a, b) = (x[j], x[j + h]);
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
        h *= 2;
    }
    let s = 1.0 / (n as f32).sqrt();
    for v in x {
        *v *= s;
    }
}

/// Index of the level nearest to `v` (ties to the lower one, as Cactus's
/// `searchsorted` rule has it).
#[inline]
pub fn nearest(cb: &[f32], v: f32) -> usize {
    let pos = cb.partition_point(|&c| c < v).clamp(1, cb.len() - 1);
    let (l, r) = (cb[pos - 1], cb[pos]);
    if (v - l).abs() <= (r - v).abs() {
        pos - 1
    } else {
        pos
    }
}

/// A `[rows, cols]` matrix quantised along its rows: packed indices
/// (`rows * in_pad * bits / 8` bytes, 8 indices per `bits` bytes, LSB first)
/// and one f16 norm per group (`rows * in_pad / group`).
pub struct Packed {
    pub indices: Vec<u8>,
    pub norms: Vec<half::f16>,
}

pub fn pack(w: &[f32], rows: usize, cols: usize, bits: u32, group: usize) -> Packed {
    assert_eq!(w.len(), rows * cols);
    let cb = codebook(bits);
    let in_pad = cols.div_ceil(group) * group;
    let groups = in_pad / group;
    let mut indices = vec![0u8; rows * in_pad * bits as usize / 8];
    let mut norms = Vec::with_capacity(rows * groups);
    let mut idx = vec![0u8; in_pad];
    let mut g = vec![0f32; group];
    for r in 0..rows {
        let row = &w[r * cols..(r + 1) * cols];
        for k in 0..groups {
            g.fill(0.0);
            let lo = k * group;
            let hi = (lo + group).min(cols);
            g[..hi - lo].copy_from_slice(&row[lo..hi]);
            hadamard(&mut g);
            let norm = g.iter().map(|v| v * v).sum::<f32>().sqrt();
            let inv = norm.max(1e-12);
            for (i, v) in g.iter().enumerate() {
                idx[lo + i] = nearest(cb, v / inv) as u8;
            }
            norms.push(half::f16::from_f32(norm));
        }
        let out =
            &mut indices[r * in_pad * bits as usize / 8..(r + 1) * in_pad * bits as usize / 8];
        for (c, chunk) in idx.chunks_exact(8).enumerate() {
            let mut word = 0u64;
            for (i, &v) in chunk.iter().enumerate() {
                word |= (v as u64) << (i as u32 * bits);
            }
            for b in 0..bits as usize {
                out[c * bits as usize + b] = (word >> (8 * b)) as u8;
            }
        }
    }
    Packed { indices, norms }
}

/// The weights a packed matrix stands for (`[rows, cols]`).
pub fn unpack(p: &Packed, rows: usize, cols: usize, bits: u32, group: usize) -> Vec<f32> {
    let cb = codebook(bits);
    let in_pad = cols.div_ceil(group) * group;
    let groups = in_pad / group;
    let row_bytes = in_pad * bits as usize / 8;
    let mask = (1u64 << bits) - 1;
    let mut out = vec![0f32; rows * cols];
    let mut g = vec![0f32; in_pad];
    for r in 0..rows {
        let bytes = &p.indices[r * row_bytes..(r + 1) * row_bytes];
        for (c, chunk) in bytes.chunks_exact(bits as usize).enumerate() {
            let mut word = 0u64;
            for (b, &v) in chunk.iter().enumerate() {
                word |= (v as u64) << (8 * b);
            }
            for i in 0..8 {
                g[c * 8 + i] = cb[((word >> (i as u32 * bits)) & mask) as usize];
            }
        }
        for k in 0..groups {
            let norm = p.norms[r * groups + k].to_f32();
            let s = &mut g[k * group..(k + 1) * group];
            for v in s.iter_mut() {
                *v *= norm;
            }
            hadamard(s);
        }
        out[r * cols..(r + 1) * cols].copy_from_slice(&g[..cols]);
    }
    out
}

/// Quantise and dequantise (what the engine will compute with).
pub fn fake_quant(w: &[f32], rows: usize, cols: usize, bits: u32) -> Vec<f32> {
    unpack(&pack(w, rows, cols, bits, GROUP), rows, cols, bits, GROUP)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hadamard_is_its_own_inverse() {
        let mut x: Vec<f32> = (0..128).map(|i| (i as f32 * 0.37).sin()).collect();
        let orig = x.clone();
        hadamard(&mut x);
        hadamard(&mut x);
        for (a, b) in x.iter().zip(&orig) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn nearest_breaks_ties_low() {
        let cb = [-1.0, 0.0, 1.0];
        assert_eq!(nearest(&cb, 0.5), 1);
        assert_eq!(nearest(&cb, 0.51), 2);
        assert_eq!(nearest(&cb, -5.0), 0);
        assert_eq!(nearest(&cb, 5.0), 2);
    }

    #[test]
    fn round_trip_is_close() {
        let (rows, cols) = (3, 200);
        let mut seed = 12345u64;
        let w: Vec<f32> = (0..rows * cols)
            .map(|_| {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5
            })
            .collect();
        let q = fake_quant(&w, rows, cols, 4);
        let err: f32 = w.iter().zip(&q).map(|(a, b)| (a - b).powi(2)).sum::<f32>() / w.len() as f32;
        let pow: f32 = w.iter().map(|a| a * a).sum::<f32>() / w.len() as f32;
        assert!(err / pow < 0.02, "relative error {}", err / pow);
    }
}

#[cfg(test)]
mod template {
    #[test]
    fn codebook_matches_a_published_model() {
        let Ok(blob) = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../model/whistle.cact"
        )) else {
            return;
        };
        let c = cactus_whistle::cact::Cact::open(&blob).unwrap();
        assert_eq!(c.codebook, super::CODEBOOK);
    }
}
