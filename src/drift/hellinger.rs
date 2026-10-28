//! Hellinger distance.

use super::binning::{
    binned_proportions, category_proportions, sturges_distinct_proportions, validate_proportions,
    Binning,
};
use super::DivergenceResult;
use crate::error::Result;

const METHOD: &str = "Hellinger distance";

/// Hellinger distance `sqrt(1 - sum_i sqrt(p_i q_i))` from proportions `p`
/// (reference) and `q` (current), in `[0, 1]`. The proportions are used as
/// given (no zero fill, no renormalization); the sum is capped at 1, as in
/// evidently.
///
/// Errors when the vectors are empty, differ in length, contain negative or
/// non-finite values, or sum to zero.
///
/// R: `philentropy::distance(rbind(p, q), method = "hellinger") / 2`
/// (philentropy's Hellinger distance is scaled by 2).
pub fn hellinger_from_proportions(p: &[f64], q: &[f64]) -> Result<f64> {
    validate_proportions(p, q)?;
    let bc: f64 = p.iter().zip(q).map(|(&pi, &qi)| (pi * qi).sqrt()).sum();
    Ok((1.0 - bc.min(1.0)).sqrt())
}

/// Hellinger distance `sqrt(1 - sum_i sqrt(p_i q_i))` between the bin
/// proportions of a numeric reference and current sample, in `[0, 1]`.
/// evidently's `"hellinger"` test flags drift at `>= 0.1`.
///
/// Binning: [`Binning::Auto`] copies evidently's numeric Hellinger rule,
/// which differs from the other divergences: always equal-width Sturges bins
/// over the pooled range, but with the bin count `ceil(log2(k) + 1)` taken
/// from the number `k` of *distinct* pooled values (evidently calls
/// `histogram_bin_edges(unique(ref, cur), "sturges")`), not from the pooled
/// sample size as [`Binning::Sturges`] does. Without ties the two agree. The
/// other [`Binning`] variants behave as in [`binned_proportions`]; use
/// [`hellinger_categorical`] (or [`Binning::Categorical`]) for evidently's
/// categorical case.
///
/// R equivalent (numeric, evidently's binning):
/// ```text
/// k  <- unique(c(x, y)); br <- seq(min(k), max(k), length.out = nclass.Sturges(k) + 1)
/// p  <- tabulate(findInterval(x, br, rightmost.closed = TRUE), length(br) - 1) / length(x)
/// q  <- tabulate(findInterval(y, br, rightmost.closed = TRUE), length(br) - 1) / length(y)
/// sqrt(1 - min(1, sum(sqrt(p * q))))
/// ```
/// `philentropy::distance(rbind(p, q), method = "hellinger")` returns twice
/// this value.
///
/// # Example
/// ```
/// use anofox_statistics::drift::{hellinger, Binning};
/// let reference: Vec<f64> = (0..50).map(|i| 2.0 * (0.37 * i as f64).sin()).collect();
/// let current: Vec<f64> = (0..40).map(|i| 2.0 * (0.53 * i as f64).sin() + 0.5).collect();
/// let r = hellinger(&reference, &current, Binning::Auto).unwrap();
/// assert!((r.statistic - 0.262287362723).abs() < 1e-10);
/// ```
pub fn hellinger(reference: &[f64], current: &[f64], binning: Binning) -> Result<DivergenceResult> {
    let (p, q) = match binning {
        Binning::Auto => sturges_distinct_proportions(reference, current)?,
        b => binned_proportions(reference, current, b)?,
    };
    Ok(DivergenceResult::new(
        METHOD,
        hellinger_from_proportions(&p, &q)?,
        p.len(),
        reference.len(),
        current.len(),
    ))
}

/// Hellinger distance between the category distributions of two samples of
/// labels (one bin per category seen in either sample), evidently's
/// categorical case.
///
/// R: `sqrt(1 - sum(sqrt(p * q)))` on the category proportions.
pub fn hellinger_categorical<T: Ord>(reference: &[T], current: &[T]) -> Result<DivergenceResult> {
    let (p, q) = category_proportions(reference, current)?;
    Ok(DivergenceResult::new(
        METHOD,
        hellinger_from_proportions(&p, &q)?,
        p.len(),
        reference.len(),
        current.len(),
    ))
}
