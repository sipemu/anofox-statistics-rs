//! Kullback-Leibler divergence.

use super::binning::{
    binned_proportions, category_proportions, validate_proportions, zero_fill, Binning,
};
use super::{ln_base, DivergenceResult};
use crate::error::Result;

const METHOD: &str = "Kullback-Leibler divergence";

/// `KL(p || q)` from proportions `p` (reference) and `q` (current):
/// evidently's [`zero_fill`] of each vector, then both are renormalized to sum
/// to 1 (as `scipy.stats.entropy`), then `sum_i p_i log(p_i / q_i)`, in base
/// `base` (default `e`).
///
/// Errors when the vectors are empty, differ in length, contain negative or
/// non-finite values, or sum to zero, and when `base` is not a finite
/// positive number other than 1.
///
/// R: `philentropy::KL(rbind(P, Q), unit = "log")` on the zero-filled,
/// renormalized `P`, `Q` (`unit = "log2"` for `base = Some(2.0)`).
pub fn kl_divergence_from_proportions(p: &[f64], q: &[f64], base: Option<f64>) -> Result<f64> {
    validate_proportions(p, q)?;
    let ln_b = ln_base(base)?;
    let normalize = |v: &[f64]| {
        let v = zero_fill(v);
        let s: f64 = v.iter().sum();
        v.into_iter().map(|x| x / s).collect::<Vec<_>>()
    };
    let (p, q) = (normalize(p), normalize(q));
    let kl: f64 = p.iter().zip(&q).map(|(&pi, &qi)| pi * (pi / qi).ln()).sum();
    Ok(kl / ln_b)
}

/// Kullback-Leibler divergence `KL(p || q)` of the bin proportions `q` of a
/// numeric current sample from the proportions `p` of the reference sample.
///
/// The proportions are zero-filled ([`zero_fill`]) and renormalized before
/// `sum_i p_i log(p_i / q_i)` is taken, in base `base` (default `e`). With
/// [`Binning::Auto`] this is evidently's `"kl_div"` test, which flags drift
/// at `KL >= 0.1`. KL is not symmetric; [`psi`](super::psi) is its
/// symmetrized form (without the renormalization).
///
/// R equivalent (with `p`, `q` the bin proportions and `fill` the zero fill):
/// `P <- fill(p); P <- P / sum(P); Q <- fill(q); Q <- Q / sum(Q);
/// philentropy::KL(rbind(P, Q), unit = "log")`.
///
/// # Example
/// ```
/// use anofox_statistics::drift::{kl_divergence, Binning};
/// let reference: Vec<f64> = (0..50).map(|i| 2.0 * (0.37 * i as f64).sin()).collect();
/// let current: Vec<f64> = (0..40).map(|i| 2.0 * (0.53 * i as f64).sin() + 0.5).collect();
/// let r = kl_divergence(&reference, &current, Binning::Auto, None).unwrap();
/// assert!((r.statistic - 0.264782319603).abs() < 1e-10);
/// ```
pub fn kl_divergence(
    reference: &[f64],
    current: &[f64],
    binning: Binning,
    base: Option<f64>,
) -> Result<DivergenceResult> {
    let (p, q) = binned_proportions(reference, current, binning)?;
    Ok(DivergenceResult::new(
        METHOD,
        kl_divergence_from_proportions(&p, &q, base)?,
        p.len(),
        reference.len(),
        current.len(),
    ))
}

/// `KL(p || q)` between the category distributions of two samples of labels
/// (one bin per category seen in either sample).
pub fn kl_divergence_categorical<T: Ord>(
    reference: &[T],
    current: &[T],
    base: Option<f64>,
) -> Result<DivergenceResult> {
    let (p, q) = category_proportions(reference, current)?;
    Ok(DivergenceResult::new(
        METHOD,
        kl_divergence_from_proportions(&p, &q, base)?,
        p.len(),
        reference.len(),
        current.len(),
    ))
}
