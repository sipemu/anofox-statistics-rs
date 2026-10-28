//! Population Stability Index.

use super::binning::{
    binned_proportions, category_proportions, validate_proportions, Binning, ZeroFill,
};
use super::DivergenceResult;
use crate::error::Result;

const METHOD: &str = "Population Stability Index";

/// PSI from proportions `p` (reference) and `q` (current):
/// `sum_i (P_i - Q_i) ln(P_i / Q_i)` after the zero fill `fill` of each
/// vector ([`ZeroFill::Evidently`] is evidently's
/// [`zero_fill`](super::zero_fill)). The proportions are not renormalized.
///
/// Errors when the vectors are empty, differ in length, contain negative or
/// non-finite values, or sum to zero, and when a constant fill is not finite
/// and positive.
///
/// R: `philentropy::distance(rbind(P, Q), method = "jeffreys", unit = "log")`
/// on the zero-filled `P`, `Q`.
pub fn psi_from_proportions(p: &[f64], q: &[f64], fill: ZeroFill) -> Result<f64> {
    validate_proportions(p, q)?;
    let (p, q) = (fill.apply(p)?, fill.apply(q)?);
    Ok(p.iter()
        .zip(&q)
        .map(|(&pi, &qi)| (pi - qi) * (pi / qi).ln())
        .sum())
}

/// Population Stability Index of a numeric current sample against a
/// reference sample.
///
/// `PSI = sum_i (p_i - q_i) ln(p_i / q_i)` over the bin proportions `p` of the
/// reference and `q` of the current sample, after the zero fill `fill`
/// ([`ZeroFill::Evidently`] for evidently's rule, [`ZeroFill::Constant`] for
/// a fixed epsilon).
/// PSI is the symmetric (Jeffreys) divergence `KL(p||q) + KL(q||p)`. A common
/// reading in credit scoring: `< 0.1` stable, `0.1-0.25` moderate shift,
/// `> 0.25` significant shift; evidently flags drift at `PSI >= 0.1`.
///
/// With [`Binning::Auto`] and [`ZeroFill::Evidently`] this is evidently's
/// `"psi"` test; use
/// [`Binning::Quantile`]`(10)` for the classic credit-scoring decile PSI.
///
/// R equivalent (with `p`, `q` the bin proportions and `fill` the zero
/// fill): `sum((fill(p) - fill(q)) * log(fill(p) / fill(q)))`, i.e.
/// `philentropy::distance(rbind(P, Q), method = "jeffreys", unit = "log")`.
///
/// # Example
/// ```
/// use anofox_statistics::drift::{psi, Binning, ZeroFill};
/// let reference: Vec<f64> = (0..50).map(|i| 2.0 * (0.37 * i as f64).sin()).collect();
/// let current: Vec<f64> = (0..40).map(|i| 2.0 * (0.53 * i as f64).sin() + 0.5).collect();
/// let r = psi(&reference, &current, Binning::Auto, ZeroFill::Evidently).unwrap();
/// assert!((r.statistic - 0.570692693084).abs() < 1e-10);
/// assert_eq!(r.n_bins, 8);
/// ```
pub fn psi(
    reference: &[f64],
    current: &[f64],
    binning: Binning,
    fill: ZeroFill,
) -> Result<DivergenceResult> {
    let (p, q) = binned_proportions(reference, current, binning)?;
    Ok(DivergenceResult::new(
        METHOD,
        psi_from_proportions(&p, &q, fill)?,
        p.len(),
        reference.len(),
        current.len(),
    ))
}

/// PSI between the category distributions of two samples of labels (one bin
/// per category seen in either sample), after the zero fill `fill`.
///
/// # Example
/// ```
/// use anofox_statistics::drift::{psi_categorical, ZeroFill};
/// let r = psi_categorical(&["a", "a", "b"], &["a", "b", "b"], ZeroFill::Evidently).unwrap();
/// assert!((r.statistic - (1.0f64 / 3.0) * 2.0f64.ln() * 2.0).abs() < 1e-12);
/// ```
pub fn psi_categorical<T: Ord>(
    reference: &[T],
    current: &[T],
    fill: ZeroFill,
) -> Result<DivergenceResult> {
    let (p, q) = category_proportions(reference, current)?;
    Ok(DivergenceResult::new(
        METHOD,
        psi_from_proportions(&p, &q, fill)?,
        p.len(),
        reference.len(),
        current.len(),
    ))
}
