//! Per-phase time counters, like the C engine's `nd_wprof`.
//!
//! Off until [`set_clock`] installs a microsecond clock; then every span adds
//! its duration to a counter. The first [`PHASES`] counters are the pipeline
//! stages, the rest break them down and overlap them.

use core::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

pub const MEL: usize = 0;
pub const STEM: usize = 1;
pub const ENCODER: usize = 2;
pub const CROSS_KV: usize = 3;
pub const DECODER: usize = 4;
pub const LIN: usize = 5;
pub const MLP: usize = 6;
pub const ATTN: usize = 7;
pub const MHC: usize = 8;
pub const HEAD: usize = 9;
pub const ENGRAM: usize = 10;
pub const CONV: usize = 11;
pub const SILU: usize = 12;
pub const KRON: usize = 13;
pub const MLP_LOAD: usize = 14;
pub const SINKHORN: usize = 15;
pub const MIX: usize = 16;
pub const PREP: usize = 17;
pub const COUNT: usize = 18;
pub const PHASES: usize = 5;
pub const NAMES: [&str; COUNT] =
    ["mel", "stem", "encoder", "cross_kv", "decoder", "~matmul", "~mlp", "~attn", "~mhc", "~head", "~engram",
     "~conv", "~silu", "~kron", "~mlp_load", "~sinkhorn", "~mix", "~q8_prep"];

static CLOCK: AtomicUsize = AtomicUsize::new(0);
static US: [AtomicU32; COUNT] = [const { AtomicU32::new(0) }; COUNT];

/// Install a free-running microsecond clock (wrapping is fine).
pub fn set_clock(f: fn() -> u32) {
    CLOCK.store(f as usize, Ordering::Relaxed);
}

#[inline]
fn now() -> Option<u32> {
    let f = CLOCK.load(Ordering::Relaxed);
    if f == 0 {
        return None;
    }
    // SAFETY: only ever stored from a `fn() -> u32` in set_clock.
    let f: fn() -> u32 = unsafe { core::mem::transmute(f) };
    Some(f())
}

pub fn reset() {
    for c in &US {
        c.store(0, Ordering::Relaxed);
    }
}

/// Microseconds per counter since the last [`reset`].
pub fn read() -> [u32; COUNT] {
    core::array::from_fn(|i| US[i].load(Ordering::Relaxed))
}

/// Adds the time until it is dropped to counter `slot`.
pub struct Span {
    slot: usize,
    t0: Option<u32>,
}

#[inline]
pub fn span(slot: usize) -> Span {
    Span { slot, t0: now() }
}

impl Drop for Span {
    #[inline]
    fn drop(&mut self) {
        if let (Some(t0), Some(t1)) = (self.t0, now()) {
            US[self.slot].fetch_add(t1.wrapping_sub(t0), Ordering::Relaxed);
        }
    }
}
