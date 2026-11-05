//! Platform-independent random index draws.
//!
//! `rand`'s `gen_range` over `usize` samples with the pointer width (`u64` on
//! 64-bit targets, `u32` on 32-bit ones such as wasm32), so a seeded stream
//! of `usize` draws differs between platforms. Every seeded index draw in the
//! crate goes through [`index`], which samples a `u64` instead: identical on
//! all targets, and identical to the previous `usize` draws on 64-bit
//! targets (where `usize` and `u64` share the sampling algorithm).

use rand::Rng;

/// Uniform index in `0..n` (`n > 0`), the same on 32- and 64-bit targets.
#[inline]
pub(crate) fn index<R: Rng + ?Sized>(rng: &mut R, n: usize) -> usize {
    rng.gen_range(0..n as u64) as usize
}
