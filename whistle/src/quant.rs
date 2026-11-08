// Started as a port of the C engine's CQ / int8 kernels in andrisgauracs/whistle-esp32
// (Apache-2.0), and changed since. See README.md, Acknowledgements.

//! Cactus-Quants (CQ) kernels.
//!
//! A CQ tensor stores, per group of `group` weights, one codebook index per
//! weight plus an fp16 group norm. The logical weights are
//!
//! ```text
//! w_group = (codebook[idx] * norm) @ H,   H = Walsh(group) / sqrt(group)
//! ```
//!
//! H is symmetric and orthonormal, so `<w, x> = <cb[idx] * norm, H x>`: the
//! activation is transformed once and the weights are never materialised.
//!
//! Whistle's matrices are all 2- or 4-bit with 128-wide groups, so every
//! matmul runs the int8 path: codebook levels map exactly onto int8, each
//! activation group gets one int8 scale, and the float norm is applied after
//! each 128-long integer dot. Those dots are the one SIMD hot spot: the
//! engine computes them in portable Rust, and a platform can plug in a
//! vector unit with [`set_kernel`].

use alloc::vec;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicPtr, Ordering};

use crate::cact::{Cact, Tensor};
use crate::math::{f16, fwht, sqrtf};

/// 16-byte aligned chunk, so int8 rows can feed 128-bit vector loads.
#[derive(Clone, Copy)]
#[repr(C, align(16))]
pub struct A16(pub [i8; 16]);

/// A growable, 16-byte aligned int8 buffer.
pub struct AlignedI8 {
    buf: Vec<A16>,
    len: usize,
}

impl AlignedI8 {
    pub fn zeroed(len: usize) -> Self {
        AlignedI8 {
            buf: vec![A16([0; 16]); len.div_ceil(16)],
            len,
        }
    }
    #[inline]
    pub fn as_slice(&self) -> &[i8] {
        // SAFETY: A16 is 16 plain i8s with no padding.
        unsafe { core::slice::from_raw_parts(self.buf.as_ptr() as *const i8, self.len) }
    }
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [i8] {
        // SAFETY: as above, and we hold the only reference.
        unsafe { core::slice::from_raw_parts_mut(self.buf.as_mut_ptr() as *mut i8, self.len) }
    }
}

/// A platform's int8 dot-product kernel, e.g. on a SIMD unit, installed
/// with [`set_kernel`]. Results must equal [`dot128`] exactly (integer math),
/// so transcripts don't depend on the kernel.
#[derive(Clone, Copy, Debug)]
pub struct Kernel {
    /// One weight row against several activation rows: for each activation
    /// row `r` in `0..rows` and group `g` in `0..groups`, writes
    /// `out[r * groups + g] = sum(w[128 * g + i] * x[r * 128 * groups + 128 * g + i] for i in 0..128)`.
    /// `out.len() == rows * groups`, at most 256 (one call covers a whole
    /// tile of activations, so the call itself costs nothing measurable).
    ///
    /// # Safety
    ///
    /// The engine guarantees: `w` has `128 * groups` and `x` has
    /// `128 * groups * rows` readable bytes, both 16-byte aligned, and the
    /// call happens on a thread where [`Kernel::here`] returned `true`.
    pub dot_rows:
        unsafe fn(w: *const i8, x: *const i8, groups: usize, rows: usize, out: &mut [i32]),
    /// Whether `dot_rows` can run on the calling thread or core (on chips
    /// where only some cores have the vector unit). Asked once per batch of
    /// rows; where it says no, the engine uses its portable loop.
    pub here: fn() -> bool,
}

static DOT_ROWS: AtomicPtr<()> = AtomicPtr::new(core::ptr::null_mut());
static HERE: AtomicPtr<()> = AtomicPtr::new(core::ptr::null_mut());

/// Installs a dot-product kernel for all following matmuls (`None`: back to
/// the portable loop). Set it before opening the model, on the core that
/// will run it; it applies to every thread.
///
/// Process-wide on purpose: a kernel describes the hardware (which cores have
/// a vector unit), not a model, and every kernel must give the same results
/// as the portable loop, so two models sharing it can't affect each other.
pub fn set_kernel(kernel: Option<Kernel>) {
    match kernel {
        Some(k) => {
            HERE.store(k.here as *mut (), Ordering::Release);
            DOT_ROWS.store(k.dot_rows as *mut (), Ordering::Release);
        }
        None => DOT_ROWS.store(core::ptr::null_mut(), Ordering::Release),
    }
}

/// The installed kernel, if any.
pub fn kernel() -> Option<Kernel> {
    let dot = DOT_ROWS.load(Ordering::Acquire);
    if dot.is_null() {
        return None;
    }
    // SAFETY: both pointers were stored by `set_kernel` from functions of
    // exactly these types (HERE before DOT_ROWS, so HERE is set too).
    unsafe {
        Some(Kernel {
            dot_rows: core::mem::transmute::<
                *mut (),
                unsafe fn(*const i8, *const i8, usize, usize, &mut [i32]),
            >(dot),
            here: core::mem::transmute::<*mut (), fn() -> bool>(HERE.load(Ordering::Acquire)),
        })
    }
}

/// The installed kernel if it can run on the calling thread.
#[inline]
fn kernel_here() -> Option<Kernel> {
    kernel().filter(|k| (k.here)())
}

/// sum(a[i] * b[i]) over 128 int8s: the portable dot product, and the
/// reference a [`Kernel`] must match. |sum| <= 128 * 128 * 128 fits an i32.
///
/// Fixed-size arrays matter: with plain slices LLVM keeps a bounds check per
/// element and only part of the loop gets vectorised.
#[inline(always)]
pub fn dot128(a: &[i8; 128], b: &[i8; 128]) -> i32 {
    let mut r = 0i32;
    for i in 0..128 {
        r += a[i] as i32 * b[i] as i32;
    }
    r
}

/// [`Kernel::dot_rows`] with the installed kernel if `kernel` is `Some`,
/// else portably: `out[r * groups + g]` = group `g` of `w` against group
/// `g` of activation row `r` of `x`.
#[inline(always)]
fn dot_rows(kernel: Option<Kernel>, w: &[i8], x: &[i8], groups: usize, out: &mut [i32]) {
    let rows = out.len() / groups;
    let row = 128 * groups;
    assert!(w.len() >= row && x.len() >= row * rows && out.len() == rows * groups);
    match kernel {
        // SAFETY: the weight row and the activation rows start 16-byte
        // aligned (AlignedI8 / A16 buffers at multiples of 128 bytes), the
        // lengths are checked above, and `kernel` came from kernel_here() on
        // this thread.
        Some(k) => unsafe { (k.dot_rows)(w.as_ptr(), x.as_ptr(), groups, rows, out) },
        None => {
            for (r, d) in out.chunks_exact_mut(groups).enumerate() {
                let xr = &x[r * row..(r + 1) * row];
                for (gi, d) in d.iter_mut().enumerate() {
                    *d = dot128(group128(w, gi), group128(xr, gi));
                }
            }
        }
    }
}

/// Group `gi` of an int8 row as a fixed-size array.
#[inline(always)]
fn group128(x: &[i8], gi: usize) -> &[i8; 128] {
    x[gi * 128..gi * 128 + 128].try_into().unwrap()
}

/// Round half away from zero (truncating float->int conversion, cheap on an Xtensa LX7).
#[inline]
pub fn q8(v: f32) -> i8 {
    (v + if v >= 0.0 { 0.5 } else { -0.5 }) as i32 as i8
}

/// [`q8`] for values known to lie within +-127.5: the same result without
/// the saturating cast's range checks, which cost more than the rest of the
/// loop on an Xtensa LX7.
///
/// The only caller scales a group by 127 / its largest magnitude after
/// checking that magnitude is finite, so `|v| <= 127 * (1 + epsilon)`.
#[inline(always)]
fn q8_bounded(v: f32) -> i8 {
    let r = v + if v >= 0.0 { 0.5 } else { -0.5 };
    debug_assert!(r.abs() < 128.5);
    // SAFETY: r is finite and within +-127.5 (see above), so inside i32.
    unsafe { r.to_int_unchecked::<i32>() as i8 }
}

/// True when the int8 path applies to `t`.
pub fn q8_ok(t: &Tensor) -> bool {
    t.group == 128 && (t.bits == 2 || t.bits == 4) && t.in_pad() <= 2048
}

/// T activation rows in the Hadamard domain as int8, one scale per 128-group.
/// Prepare once and run every tensor that reduces over the same input.
pub struct Q8Act {
    pub xq: AlignedI8, // [T][in_pad]
    pub sx: Vec<f32>,  // [T][in_pad / 128]
    pub t: usize,
    pub in_pad: usize,
}

/// Activation tile for q8_gemm, as in the C engine.
const TILE_BYTES: usize = 32 * 1024;
/// Most group dots of one weight row against one tile: TILE_BYTES / 128, or
/// 16 for a single 2048-wide activation row.
const MAX_TILE_DOTS: usize = TILE_BYTES / 128;

/// 1/sqrt(128), the exact f32 of the C literal (one ulp off flips near-tie transcripts).
#[allow(clippy::excessive_precision)]
const INV_SQRT_128: f32 = 0.088_388_347_648_318_45;

impl Q8Act {
    /// From T rows of `x` (row stride `t.shape[1]`) for tensors shaped like `t`.
    pub fn prepare(t: &Tensor, x: &[f32], rows: usize) -> Self {
        debug_assert!(q8_ok(t));
        let inn = t.shape[1] as usize;
        let in_pad = t.in_pad();
        let ng = in_pad / 128;
        let mut xq = AlignedI8::zeroed(rows * in_pad);
        let mut sx = vec![0f32; rows * ng];
        let qp = SyncPtrI8(xq.as_mut_slice().as_mut_ptr());
        let sp = SyncPtr(sx.as_mut_ptr());
        // One activation row per work item: it owns its int8 row and scales.
        let kernel = |ka: usize, kb: usize| {
            let mut blk = [0f32; 128];
            for k in ka..kb {
                let src = &x[k * inn..(k + 1) * inn];
                for g in 0..ng {
                    // The tail group of a row is zero padded.
                    let col0 = g * 128;
                    let n = inn.saturating_sub(col0).min(128);
                    blk[..n].copy_from_slice(&src[col0..col0 + n]);
                    blk[n..].fill(0.0);
                    crate::math::fwht128(&mut blk);
                    // The largest magnitude, as an integer max over the bits
                    // of |v|: for non-negative floats the bit patterns order
                    // like the values, and NaN and infinity sort above every
                    // finite value, so the check below catches them too.
                    let mut m_bits = 0u32;
                    for v in blk.iter_mut() {
                        *v *= INV_SQRT_128;
                        m_bits = m_bits.max(v.to_bits() & 0x7FFF_FFFF);
                    }
                    let m = f32::from_bits(m_bits);
                    // SAFETY: group g of row k belongs to this work item.
                    let qg = unsafe { qp.group(k * in_pad + g * 128) };
                    if !m.is_finite() {
                        // NaN or infinity in the activations (broken input):
                        // a zero group, so q8_bounded only ever sees values
                        // within +-127.5.
                        qg.fill(0);
                        unsafe { sp.write(k * ng + g, 1.0) };
                        continue;
                    }
                    let s = if m > 0.0 { m / 127.0 } else { 1.0 };
                    let inv = 1.0 / s;
                    for j in 0..128 {
                        qg[j] = q8_bounded(blk[j] * inv);
                    }
                    unsafe { sp.write(k * ng + g, s) };
                }
            }
        };
        crate::par::rows(rows, 1, &kernel);
        Q8Act {
            xq,
            sx,
            t: rows,
            in_pad,
        }
    }
}

/// Packed indices then fp16 norms; norms for row `r`, group `g`.
#[inline]
fn norm_at(blob: &[u8], norms_off: usize, idx: usize) -> f32 {
    let o = norms_off + idx * 2;
    f16(u16::from_le_bytes([blob[o], blob[o + 1]]))
}

/// `Y[k][r - r0] = W[r] . x_k` for rows `[r0, r0 + nrows)`, Y stride `nrows`.
pub fn q8_gemm(
    c: &Cact,
    t: &Tensor,
    blob: &[u8],
    a: &Q8Act,
    r0: usize,
    nrows: usize,
    y: &mut [f32],
) {
    let in_pad = a.in_pad;
    let ng = in_pad / 128;
    let rowb = t.row_bytes();
    let norms_off = t.shape[0] as usize * rowb;
    let cb = c.codebook(t.bits);
    let nl = 1usize << t.bits;

    let mut mx = 0f32;
    for &v in &cb[..nl] {
        if v.abs() > mx {
            mx = v.abs();
        }
    }
    let sw = 127.0 / mx;
    let mut lv = [0i8; 16];
    for i in 0..nl {
        lv[i] = q8(cb[i] * sw);
    }
    let inv_sw = 1.0 / sw;

    // byte -> its int8 levels (LSB-first index packing), as whole aligned
    // words: with [i8; N] entries the Xtensa backend loads each level byte by
    // byte and reassembles the word, ~6x slower per packed byte.
    let mut tab2 = [0u32; 256];
    let mut tab4 = [0u16; 256];
    for i in 0..256 {
        if t.bits == 2 {
            tab2[i] = u32::from_le_bytes([
                lv[i & 3] as u8,
                lv[(i >> 2) & 3] as u8,
                lv[(i >> 4) & 3] as u8,
                lv[(i >> 6) & 3] as u8,
            ]);
        } else {
            tab4[i] = u16::from_le_bytes([lv[i & 15] as u8, lv[i >> 4] as u8]);
        }
    }

    // Every weight row reads every activation row, so the activations are
    // walked in tiles of TILE_BYTES that stay in the data cache. Each output is still computed exactly as without tiling.
    let tile = (TILE_BYTES / in_pad).max(1);
    let xq = a.xq.as_slice();
    let y = SyncPtr(y.as_mut_ptr());
    let mut k0 = 0;
    while k0 < a.t {
        let k1 = (k0 + tile).min(a.t);
        // Rows [ra, rb): expand each packed row to int8 once, then dot it
        // with every activation row of the tile.
        let kernel = |ra: usize, rb: usize| {
            let mut wbuf = [A16([0; 16]); 2048 / 16];
            let wp = wbuf.as_mut_ptr();
            let fast = kernel_here();
            let mut nrm = [0f32; 16];
            // One weight row against the whole tile; ng * (k1 - k0) * 128
            // is at most TILE_BYTES (or one 2048-wide row).
            let mut dots = [0i32; MAX_TILE_DOTS];
            let tile_dots = &mut dots[..ng * (k1 - k0)];
            for r in ra..rb {
                let tr = r + r0;
                let p = &blob[tr * rowb..(tr + 1) * rowb];
                // SAFETY: wbuf is 2048 bytes, 16-byte aligned, and
                // in_pad <= 2048 (q8_ok); u32/u16 views of it are aligned.
                // SAFETY: as above; `unpack` checks the source alignment.
                let w32 = unsafe { core::slice::from_raw_parts_mut(wp as *mut u32, in_pad / 4) };
                if t.bits == 2 {
                    unpack2(p, &tab2, w32);
                } else {
                    unpack4(p, &tab4, w32);
                }
                // SAFETY: as above, now read as the int8 row.
                let w: &[i8] = unsafe { core::slice::from_raw_parts(wp as *const i8, in_pad) };
                for gi in 0..ng {
                    nrm[gi] = norm_at(blob, norms_off, tr * ng + gi) * inv_sw;
                }
                dot_rows(fast, w, &xq[k0 * in_pad..k1 * in_pad], ng, tile_dots);
                for k in k0..k1 {
                    let sx = &a.sx[k * ng..(k + 1) * ng];
                    let d = &tile_dots[(k - k0) * ng..(k - k0 + 1) * ng];
                    let mut acc = 0f32;
                    for gi in 0..ng {
                        acc += nrm[gi] * sx[gi] * d[gi] as f32;
                    }
                    // SAFETY: each row r is written by exactly one worker.
                    unsafe { y.write(k * nrows + r, acc) };
                }
            }
        };
        // Aim for chunks of ~8K int8 MACs-worth of decode + dots per steal.
        let chunk = (64 / (k1 - k0)).clamp(1, 16);
        crate::par::rows_int8(nrows, chunk, k1 - k0 <= crate::par::light_rows(), &kernel);
        k0 = k1;
    }
}

/// 2-bit row: each packed byte becomes four int8 levels (one u32). Four
/// bytes per step, read as one aligned word, so the table loads overlap
/// instead of each waiting on the byte load before it (LX7 load-use stalls).
#[inline(always)]
fn unpack2(p: &[u8], tab: &[u32; 256], w: &mut [u32]) {
    let n = p.len().min(w.len());
    if !(p.as_ptr() as usize).is_multiple_of(4) || !n.is_multiple_of(4) {
        for (d, &b) in w.iter_mut().zip(p) {
            *d = tab[b as usize];
        }
        return;
    }
    let src = p.as_ptr() as *const u32;
    let dst = w.as_mut_ptr();
    for i in 0..n / 4 {
        // SAFETY: i < n / 4 words of an aligned source, 4 * i + 3 < n <= w.len().
        unsafe {
            let v = src.add(i).read();
            let (a, b, c, d) = (
                tab[(v & 0xFF) as usize],
                tab[(v >> 8 & 0xFF) as usize],
                tab[(v >> 16 & 0xFF) as usize],
                tab[(v >> 24) as usize],
            );
            dst.add(4 * i).write(a);
            dst.add(4 * i + 1).write(b);
            dst.add(4 * i + 2).write(c);
            dst.add(4 * i + 3).write(d);
        }
    }
}

/// 4-bit row: each packed byte becomes two int8 levels (one u16); two of
/// those make one u32 store. Four packed bytes per step, as in `unpack2`.
#[inline(always)]
fn unpack4(p: &[u8], tab: &[u16; 256], w: &mut [u32]) {
    let n = p.len().min(2 * w.len());
    if !(p.as_ptr() as usize).is_multiple_of(4) || !n.is_multiple_of(4) {
        for (i, &b) in p[..n].iter().enumerate() {
            let s = (i & 1) * 16;
            w[i / 2] = (w[i / 2] & !(0xFFFF << s)) | (tab[b as usize] as u32) << s;
        }
        return;
    }
    let src = p.as_ptr() as *const u32;
    let dst = w.as_mut_ptr();
    for i in 0..n / 4 {
        // SAFETY: as in unpack2; 2 * i + 1 < n / 2 <= w.len().
        unsafe {
            let v = src.add(i).read();
            let (a, b, c, d) = (
                tab[(v & 0xFF) as usize],
                tab[(v >> 8 & 0xFF) as usize],
                tab[(v >> 16 & 0xFF) as usize],
                tab[(v >> 24) as usize],
            );
            dst.add(2 * i).write(a as u32 | (b as u32) << 16);
            dst.add(2 * i + 1).write(c as u32 | (d as u32) << 16);
        }
    }
}

/// The int8 activation buffer shared by `Q8Act::prepare`'s row workers.
#[derive(Clone, Copy)]
struct SyncPtrI8(*mut i8);
// SAFETY: workers only write disjoint 128-byte groups.
unsafe impl Sync for SyncPtrI8 {}
unsafe impl Send for SyncPtrI8 {}

impl SyncPtrI8 {
    /// # Safety
    /// `o..o + 128` is in bounds and owned by the caller.
    #[inline(always)]
    #[allow(clippy::mut_from_ref)]
    unsafe fn group<'x>(self, o: usize) -> &'x mut [i8; 128] {
        unsafe { &mut *(self.0.add(o) as *mut [i8; 128]) }
    }
}

/// A raw output pointer shared by the row workers, which write disjoint rows.
#[derive(Clone, Copy)]
struct SyncPtr(*mut f32);
// SAFETY: workers only write disjoint elements (see q8_gemm).
unsafe impl Sync for SyncPtr {}
unsafe impl Send for SyncPtr {}

impl SyncPtr {
    /// # Safety
    /// `i` is in bounds and no other worker writes it.
    #[inline(always)]
    unsafe fn write(self, i: usize, v: f32) {
        unsafe { *self.0.add(i) = v };
    }
}

/// Reconstruct dequantised row `row` of `t` into `w` (`shape[1]` floats).
/// `scratch` holds `in_pad` floats. Used for the embedding and engram tables.
pub fn dequant_row(
    c: &Cact,
    t: &Tensor,
    blob: &[u8],
    row: usize,
    scratch: &mut [f32],
    w: &mut [f32],
) {
    let g = t.group as usize;
    let ng = t.groups();
    let rowb = t.row_bytes();
    let norms_off = t.shape[0] as usize * rowb;
    let cb = c.codebook(t.bits);
    let bits = t.bits as usize;
    let mask = (1u32 << bits) - 1;
    let p = &blob[row * rowb..];
    let scale = 1.0 / sqrtf(g as f32);
    for gi in 0..ng {
        let blk = &mut scratch[gi * g..(gi + 1) * g];
        let norm = norm_at(blob, norms_off, row * ng + gi);
        let bit_off = gi * g * bits;
        for (j, v) in blk.iter_mut().enumerate() {
            let b = bit_off + j * bits;
            let byt = b >> 3;
            let sh = b & 7;
            let mut x = p[byt] as u32;
            if sh + bits > 8 {
                x |= (p[byt + 1] as u32) << 8;
            }
            *v = cb[((x >> sh) & mask) as usize] * norm;
        }
        fwht(blk);
        for v in blk.iter_mut() {
            *v *= scale;
        }
    }
    let n = t.shape[1] as usize;
    w[..n].copy_from_slice(&scratch[..n]);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(inn: u32) -> Tensor {
        Tensor {
            dtype: crate::cact::DT_CQ,
            ndim: 2,
            shape: [1, inn, 0, 0],
            offset: 0,
            nbytes: 0,
            group: 128,
            bits: 4,
        }
    }

    #[test]
    fn non_finite_activations_give_zero_groups() {
        // Two 128-wide groups per row: one finite, one broken.
        let t = tensor(256);
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut x = [0.25f32; 256];
            x[200] = bad;
            let q = Q8Act::prepare(&t, &x, 1);
            let xq = q.xq.as_slice();
            assert!(xq[..128].iter().any(|&v| v != 0), "finite group was zeroed");
            assert!(
                xq[128..256].iter().all(|&v| v == 0),
                "broken group not zeroed ({bad})"
            );
            assert_eq!(q.sx[1], 1.0);
        }
    }

    #[test]
    fn finite_groups_use_the_full_int8_range() {
        let t = tensor(128);
        let x: [f32; 128] = core::array::from_fn(|i| i as f32 - 64.0);
        let q = Q8Act::prepare(&t, &x, 1);
        let peak =
            q.xq.as_slice()
                .iter()
                .map(|&v| (v as i32).abs())
                .max()
                .unwrap();
        assert_eq!(peak, 127);
    }
}
