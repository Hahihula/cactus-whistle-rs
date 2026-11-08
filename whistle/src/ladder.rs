// `trim` is a port of tools/whistle_strip.py from andrisgauracs/whistle-esp32
// (Apache-2.0). See README.md, Acknowledgements.

//! Whistle's laddered decoder: choosing and cutting decoder depth.
//!
//! The decoder is trained as nested sub-models (Needle 3's ladder): depth
//! `k` uses the layers `order(L)[..k]`, both endpoints first, then the middle
//! of the largest gap. [`crate::Whistle::set_depth`] picks a depth at run
//! time; [`trim`] writes a smaller `.cact` that only holds the layers a depth
//! uses, so it fits a 16 MB flash chip (16.9 MB full, 14.5 MB at depth 4).

use alloc::vec::Vec;

use crate::cact::{Cact, CactError, DT_CQ, DT_FP16, HDR4_BYTES, HDR4_WORDS, REC_SIZE, rd_u32};

/// Most decoder layers a model may have.
pub const MAX_LAYERS: usize = 16;

/// Ladder order of `layers` decoder layers (needle `architecture.
/// _ladder_layer_order`): 0, L-1, then repeatedly the midpoint of the
/// widest gap between chosen layers, leftmost on ties. Only the first
/// `layers` entries are meaningful.
pub fn order(layers: usize) -> [usize; MAX_LAYERS] {
    assert!((2..=MAX_LAYERS).contains(&layers), "2..=16 decoder layers");
    let mut order = [0usize; MAX_LAYERS];
    let mut sel = [0usize; MAX_LAYERS];
    order[1] = layers - 1;
    sel[1] = layers - 1;
    let (mut n, mut ns) = (2, 2);
    while n < layers {
        let (mut best, mut bl) = (0, 0);
        for i in 0..ns - 1 {
            let gap = sel[i + 1] - sel[i];
            if gap > 1 && gap > best {
                best = gap;
                bl = i;
            }
        }
        let mid = (sel[bl] + sel[bl + 1]) / 2;
        order[n] = mid;
        n += 1;
        for j in (bl + 2..=ns).rev() {
            sel[j] = sel[j - 1];
        }
        sel[bl + 1] = mid;
        ns += 1;
    }
    order
}

/// The layers depth `depth` keeps, ascending.
pub fn layers_for(layers: usize, depth: usize) -> Vec<usize> {
    let mut keep: Vec<usize> = order(layers)[..depth].to_vec();
    keep.sort_unstable();
    keep
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrimError {
    Cact(CactError),
    /// Depth outside 2..=num_layers.
    BadDepth,
    /// The directory doesn't have the tensor count the header implies.
    Schema,
    /// A tensor's payload runs past the blob.
    Truncated,
}

impl core::fmt::Display for TrimError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            TrimError::Cact(e) => write!(f, "invalid .cact file: {e}"),
            TrimError::BadDepth => {
                f.write_str("depth must be between 2 and the model's layer count")
            }
            TrimError::Schema => f.write_str("tensor count doesn't match the header"),
            TrimError::Truncated => f.write_str("a tensor runs past the end of the file"),
        }
    }
}

impl core::error::Error for TrimError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            TrimError::Cact(e) => Some(e),
            _ => None,
        }
    }
}

impl From<CactError> for TrimError {
    fn from(e: CactError) -> Self {
        TrimError::Cact(e)
    }
}

/// A trimmed model.
pub struct Trimmed {
    /// The new `.cact` blob.
    pub blob: Vec<u8>,
    /// Decoder layers kept (indices into the original stack).
    pub layers: Vec<usize>,
    /// Layers of the engram sites kept (original indices).
    pub sites: Vec<usize>,
}

// Tensors per group, in blob order (whistle_cact.py SCHEMA).
const DEC_LAYER: usize = 27; // self-attention block + Hadamard MLP
const CROSS_LAYER: usize = 10;
const MHC: usize = 9; // a_pre a_post a_res b_pre b_post b_res phi_pre phi_post phi_res
const ENGRAM: usize = 4;
const ALIGN: usize = 64;

/// Rewrites a `.cact` to hold only the decoder layers depth `depth` uses,
/// the way the training code's `ladder_slice` slices the parameters:
///
/// - the kept layers' decoder and cross-attention tensors;
/// - their rows of the decoder's mHC tensors (for CQ tensors: packed rows,
///   then the matching group norms);
/// - the engram sites on kept layers, renumbered;
/// - the header's layer count and sites.
///
/// Everything else (encoder, stem, filter bank, embedding, tokenizer) is
/// copied byte for byte. The engine needs no depth setting for the result:
/// a layer's mHC lane offsets come from its position in the stack.
///
/// The output is byte-identical to `tools/whistle_strip.py` from
/// whistle-esp32.
pub fn trim(blob: &[u8], depth: usize) -> Result<Trimmed, TrimError> {
    let c = Cact::open(blob)?;
    let h = c.h;
    let l_all = h.num_layers as usize;
    if depth < 2 || depth > l_all || l_all > MAX_LAYERS {
        return Err(TrimError::BadDepth);
    }
    let keep = layers_for(l_all, depth);
    let all_sites = &h.sites[..h.num_sites as usize];
    let sites: Vec<(usize, usize)> = all_sites
        .iter()
        .enumerate()
        .map(|(s, &l)| (s, l as usize))
        .filter(|(_, l)| keep.contains(l))
        .collect();

    // Directory positions of the decoder groups.
    let i_layer = |l: usize| 1 + l * DEC_LAYER;
    let i_mhc = 1 + l_all * DEC_LAYER;
    let i_hada = i_mhc + MHC;
    let i_engram = |s: usize| i_hada + 2 + s * ENGRAM;
    let i_final = i_engram(all_sites.len());
    let i_audio = i_final + 1;
    let i_cross = |l: usize| i_audio + 1 + l * CROSS_LAYER;
    let i_encoder = i_cross(l_all);
    if i_encoder >= c.n as usize {
        return Err(TrimError::Schema);
    }

    // (record, payload) of every output tensor, in order.
    let mut out: Vec<(crate::cact::Tensor, Vec<u8>)> = Vec::new();
    let payload = |i: usize| -> Result<(crate::cact::Tensor, &[u8]), TrimError> {
        let t = c.tensor(i as u32)?;
        Ok((t, c.data(&t).ok_or(TrimError::Truncated)?))
    };
    let copy = |i: usize, out: &mut Vec<_>| -> Result<(), TrimError> {
        let (t, d) = payload(i)?;
        out.push((t, d.to_vec()));
        Ok(())
    };

    copy(0, &mut out)?; // embedding
    for &l in &keep {
        for i in i_layer(l)..i_layer(l) + DEC_LAYER {
            copy(i, &mut out)?;
        }
    }
    for k in 0..MHC {
        let (t, d) = payload(i_mhc + k)?;
        let rows: Vec<usize> = if k >= 6 {
            // phi_*: n or n*n rows per layer.
            let per = t.shape[0] as usize / l_all;
            keep.iter()
                .flat_map(|&l| (0..per).map(move |r| l * per + r))
                .collect()
        } else {
            keep.clone()
        };
        out.push(slice_rows(&t, d, &rows));
    }
    copy(i_hada, &mut out)?;
    copy(i_hada + 1, &mut out)?;
    for &(s, _) in &sites {
        for i in i_engram(s)..i_engram(s) + ENGRAM {
            copy(i, &mut out)?;
        }
    }
    copy(i_final, &mut out)?;
    copy(i_audio, &mut out)?;
    for &l in &keep {
        for i in i_cross(l)..i_cross(l) + CROSS_LAYER {
            copy(i, &mut out)?;
        }
    }
    for i in i_encoder..c.n as usize {
        copy(i, &mut out)?;
    }

    // Header: same geometry, fewer tensors, layers and sites.
    let mut w = [0u32; HDR4_WORDS];
    for (k, v) in w.iter_mut().enumerate() {
        *v = rd_u32(blob, 4 * k);
    }
    w[1] = out.len() as u32;
    w[10] = depth as u32;
    w[31] = sites.len() as u32;
    for k in 0..16 {
        w[32 + k] = sites.get(k).map_or(0, |&(_, l)| {
            keep.iter().position(|&x| x == l).unwrap() as u32
        });
    }
    let cb = &blob[HDR4_BYTES..HDR4_BYTES + 4 * h.codebook_len as usize];
    let data_off = (HDR4_BYTES + cb.len() + out.len() * REC_SIZE).next_multiple_of(ALIGN);

    let mut b: Vec<u8> = Vec::with_capacity(blob.len());
    for v in w {
        b.extend_from_slice(&v.to_le_bytes());
    }
    b.extend_from_slice(cb);
    let mut off = data_off;
    for (t, d) in &out {
        b.push(t.dtype);
        b.push(t.ndim);
        b.extend_from_slice(&[0, 0]);
        for s in t.shape {
            b.extend_from_slice(&s.to_le_bytes());
        }
        b.extend_from_slice(&(off as u64).to_le_bytes());
        b.extend_from_slice(&(d.len() as u64).to_le_bytes());
        b.extend_from_slice(&t.group.to_le_bytes());
        b.extend_from_slice(&t.bits.to_le_bytes());
        off = (off + d.len()).next_multiple_of(ALIGN);
    }
    b.resize(data_off, 0);
    for (k, (_, d)) in out.iter().enumerate() {
        b.extend_from_slice(d);
        if k + 1 < out.len() {
            b.resize(b.len().next_multiple_of(ALIGN), 0);
        }
    }
    Ok(Trimmed {
        blob: b,
        layers: keep,
        sites: sites.iter().map(|&(_, l)| l).collect(),
    })
}

/// Tensor `t` restricted to `rows` of its leading axis.
fn slice_rows(t: &crate::cact::Tensor, d: &[u8], rows: &[usize]) -> (crate::cact::Tensor, Vec<u8>) {
    let mut r = *t;
    r.shape[0] = rows.len() as u32;
    let mut v = Vec::new();
    if t.dtype == DT_CQ {
        let (out, rb, nb) = (t.shape[0] as usize, t.row_bytes(), t.groups() * 2);
        for &k in rows {
            v.extend_from_slice(&d[k * rb..(k + 1) * rb]);
        }
        for &k in rows {
            v.extend_from_slice(&d[out * rb + k * nb..out * rb + (k + 1) * nb]);
        }
    } else {
        let item = if t.dtype == DT_FP16 { 2 } else { 4 };
        let per = item
            * t.shape[1..t.ndim as usize]
                .iter()
                .map(|&s| s as usize)
                .product::<usize>();
        for &k in rows {
            v.extend_from_slice(&d[k * per..(k + 1) * per]);
        }
    }
    (r, v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ladder_order_matches_needle() {
        // whistle_cact.py ladder_order(8) and the official engine's depths.
        assert_eq!(order(8)[..8], [0, 7, 3, 5, 1, 2, 4, 6]);
        assert_eq!(layers_for(8, 4), [0, 3, 5, 7]);
        assert_eq!(order(2)[..2], [0, 1]);
    }
}
