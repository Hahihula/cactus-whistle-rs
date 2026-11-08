//! `no_std` inference engine for [Cactus Whistle](https://huggingface.co/Cactus-Compute/whistle),
//! a multilingual speech recognition model, sized to run on microcontrollers as
//! well as on a desktop.
//!
//! The weights are read in place from the `.cact` blob (on a microcontroller, a
//! memory-mapped flash partition); activations are heap allocated.
//!
//! ```ignore
//! let mut w = cactus_whistle::Whistle::open(blob)?;
//! let text = w.transcribe(&samples_16khz, None)?;
//! ```
#![no_std]
#![cfg_attr(target_arch = "xtensa", feature(core_intrinsics))]
#![cfg_attr(target_arch = "xtensa", allow(internal_features))]

// Kernels index several arrays per loop on purpose: the loops mirror the C
// engine so the two can be diffed, and evaluation order affects the floats.
#![allow(clippy::needless_range_loop, clippy::type_complexity)]

extern crate alloc;

pub mod cact;
pub mod ladder;
pub mod math;
pub mod model;
pub mod par;
pub mod prof;
pub mod quant;
pub mod tokenizer;

/// Entry points for on-device micro-benchmarks of internal kernels.
#[doc(hidden)]
pub mod bench {
    pub use crate::model::kron;
}

pub use model::{find_speech, Error, Session, Spec, Whistle, LANGS, N_LANGS};
