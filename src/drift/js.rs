//! Jensen-Shannon distance.

use super::binning::{binned_proportions, category_proportions, validate_proportions, Binning};
use super::{ln_base, DivergenceResult};
use crate::error::Result;

const METHOD: &str = "Jensen-Shannon distance";

/// `a ln(a / b)`, with `0 ln(0 / b) = 0` (scipy `rel_entr`).
fn rel_entr(a: f64, b: f64) -> f64 {
    if a > 0.0 {
        a * (a / b).ln()
    } else {
        0.0
    }
}

/// Jensen-Shannon *distance* from proportions `p` (reference) and `q`
/// (current), as `scipy.spatial.distance.jensenshannon`: both vectors are
/// normalized to sum to 1, `m = (p + q) / 2`, and the result is
/// `sqrt((KL(p || m) + KL(q || m)) / 2)` in base `base` (default `e`). No zero
/// fill is needed. Bounded by `sqrt(ln 2)` (natural log) or 1 (base 2).
///
/// Errors when the vectors are empty, differ in length, contain negative or
/// non-finite values, or sum to zero, and when `base` is not a finite
/// positive number other than 1.
///
/// R: `sqrt(philentropy::JSD(rbind(p, q), unit = "log"))` (`philentropy::JSD`
/// returns the divergence and defaults to `unit = "log2"`).
pub fn jensen_shannon_from_proportions(p: &[f64], q: &[f64], base: Option<f64>) -> Result<f64> {
    validate_proportions(p, q)?;
    let ln_b = ln_base(base)?;
    let (sp, sq): (f64, f64) = (p.iter().sum(), q.iter().sum());
    let js: f64 = p
        .iter()
        .zip(q)
        .map(|(&pi, &qi)| {
            let (pi, qi) = (pi / sp, qi / sq);
            let m = (pi + qi) / 2.0;
            rel_entr(pi, m) + rel_entr(qi, m)
        })
        .sum::<f64>()
        / 2.0;
    // Rounding can leave a tiny negative value for identical inputs.
    Ok((js / ln_b).max(0.0).sqrt())
}

/// Jensen-Shannon distance between the bin proportions of a numeric
/// reference and current sample: the square root of the Jensen-Shannon
/// divergence, in base `base` (default `e`).
///
/// Symmetric and bounded (`<= sqrt(ln 2)`, or `<= 1` in base 2). With
/// [`Binning::Auto`] this is evidently's `"jensenshannon"` test, which flags
/// drift at a distance `>= 0.1`; evidently uses it by default for categorical
/// columns of large datasets. The proportions are not zero-filled.
///
/// R equivalent (with `p`, `q` the bin proportions):
/// `sqrt(philentropy::JSD(rbind(p, q), unit = "log"))`.
///
/// # Example
/// ```
/// use anofox_statistics::drift::{jensen_shannon, Binning};
/// let reference: Vec<f64> = (0..50).map(|i| 2.0 * (0.37 * i as f64).sin()).collect();
/// let current: Vec<f64> = (0..40).map(|i| 2.0 * (0.53 * i as f64).sin() + 0.5).collect();
/// let r = jensen_shannon(&reference, &current, Binning::Auto, None).unwrap();
/// assert!((r.statistic - 0.257847900615).abs() < 1e-10);
/// ```
pub fn jensen_shannon(
    reference: &[f64],
    current: &[f64],
    binning: Binning,
    base: Option<f64>,
) -> Result<DivergenceResult> {
    let (p, q) = binned_proportions(reference, current, binning)?;
    Ok(DivergenceResult::new(
        METHOD,
        jensen_shannon_from_proportions(&p, &q, base)?,
        p.len(),
        reference.len(),
        current.len(),
    ))
}

/// Jensen-Shannon distance between the category distributions of two
/// samples of labels (one bin per category seen in either sample).
pub fn jensen_shannon_categorical<T: Ord>(
    reference: &[T],
    current: &[T],
    base: Option<f64>,
) -> Result<DivergenceResult> {
    let (p, q) = category_proportions(reference, current)?;
    Ok(DivergenceResult::new(
        METHOD,
        jensen_shannon_from_proportions(&p, &q, base)?,
        p.len(),
        reference.len(),
        current.len(),
    ))
}
