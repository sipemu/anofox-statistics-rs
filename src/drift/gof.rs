//! Goodness-of-fit drift tests of the current category counts against the
//! reference distribution (evidently's `"chisquare"` and `"g_test"`).

use super::{category_counts, OrdF64};
use crate::categorical::ChiSquareResult;
use crate::error::{Result, StatError};
use crate::utils::finite::ensure_finite;
use statrs::distribution::{ChiSquared, ContinuousCDF};

/// Which goodness-of-fit statistic [`gof_drift`] computes.
#[derive(Clone, Copy)]
enum Gof {
    Pearson,
    LogLikelihood,
}

fn gof_drift<T: Ord>(reference: &[T], current: &[T], kind: Gof) -> Result<ChiSquareResult> {
    if reference.is_empty() || current.is_empty() {
        return Err(StatError::EmptyData);
    }
    let (r, c) = category_counts(reference, current);
    let k = r.len();
    let scale = current.len() as f64 / reference.len() as f64;
    let expected: Vec<f64> = r.iter().map(|&v| v as f64 * scale).collect();
    let mut statistic = 0.0;
    for (&o, &e) in c.iter().zip(&expected) {
        let o = o as f64;
        statistic += match kind {
            // A category absent from the reference (e = 0, o > 0) makes the
            // statistic infinite, as in scipy.
            Gof::Pearson if e == 0.0 => f64::INFINITY,
            Gof::Pearson => (o - e).powi(2) / e,
            Gof::LogLikelihood if o == 0.0 => 0.0,
            Gof::LogLikelihood if e == 0.0 => f64::INFINITY,
            Gof::LogLikelihood => 2.0 * o * (o / e).ln(),
        };
    }
    let df = (k - 1) as f64;
    let p_value = if k == 1 {
        1.0
    } else if statistic.is_infinite() {
        0.0
    } else {
        ChiSquared::new(df).unwrap().sf(statistic)
    };
    let residuals = c
        .iter()
        .zip(&expected)
        .map(|(&o, &e)| {
            if e > 0.0 {
                (o as f64 - e) / e.sqrt()
            } else {
                f64::INFINITY
            }
        })
        .collect();
    let method = match kind {
        Gof::Pearson => "Chi-squared goodness-of-fit drift test",
        Gof::LogLikelihood => "G goodness-of-fit drift test",
    };
    Ok(ChiSquareResult {
        statistic,
        df,
        p_value,
        expected: vec![expected],
        residuals: Some(vec![residuals]),
        method: method.to_string(),
    })
}

/// Pearson chi-squared goodness-of-fit test of the current category counts
/// against the reference distribution.
///
/// Over the `k` sorted categories seen in either sample, the observed counts
/// `o` are the current counts and the expected counts are the reference
/// counts scaled to the current size, `e = r * n / m`.
/// `X^2 = sum (o - e)^2 / e` has `k - 1` degrees of freedom. A category that
/// is absent from the reference but present in the current sample gives
/// `X^2 = Inf` and `p = 0`. A single category gives `X^2 = 0`, `p = 1`.
/// `expected` and `residuals` (`(o - e) / sqrt(e)`) are in category order.
///
/// This is evidently's `"chisquare"` drift test, the default for
/// categorical columns with more than two categories of a reference with up
/// to 1000 rows (drift at `p < 0.05`). Unlike [`chisq_test`](crate::chisq_test)
/// it treats the reference as the known distribution, not as a second
/// sample.
///
/// Errors on an empty sample.
///
/// R equivalent: `chisq.test(o, p = r / sum(r))`.
///
/// # Example
/// ```
/// use anofox_statistics::drift::chisq_drift;
/// let r = chisq_drift(&["a", "a", "b", "b"], &["a", "b", "b", "b"]).unwrap();
/// assert_eq!(r.statistic, 1.0);
/// assert_eq!(r.df, 1.0);
/// ```
pub fn chisq_drift<T: Ord>(reference: &[T], current: &[T]) -> Result<ChiSquareResult> {
    gof_drift(reference, current, Gof::Pearson)
}

/// [`chisq_drift`] for numeric labels: one category per distinct value.
/// Errors additionally on non-finite values.
pub fn chisq_drift_numeric(reference: &[f64], current: &[f64]) -> Result<ChiSquareResult> {
    ensure_finite("reference", reference)?;
    ensure_finite("current", current)?;
    chisq_drift(&OrdF64::codes(reference), &OrdF64::codes(current))
}

/// G (log-likelihood ratio) goodness-of-fit test of the current category
/// counts against the reference distribution.
///
/// As [`chisq_drift`], with `G = 2 sum o ln(o / e)` (terms with `o = 0` are
/// 0) in place of Pearson's statistic, on `k - 1` degrees of freedom.
///
/// This is evidently's `"g_test"` drift test (drift at `p < 0.1`). evidently
/// passes the unscaled reference counts as expected counts, which only
/// agrees with the observed total when both samples have the same size
/// (recent scipy rejects the call otherwise); here the expected counts are
/// always scaled to the current size, which gives evidently's value for
/// equal sizes.
///
/// Errors on an empty sample.
///
/// R equivalent: `e <- sum(o) * r / sum(r); G <- 2 * sum(o[o > 0] * log(o[o > 0] / e[o > 0]));
/// pchisq(G, length(o) - 1, lower.tail = FALSE)`.
///
/// # Example
/// ```
/// use anofox_statistics::drift::g_drift;
/// let r = g_drift(&["a", "a", "b", "b"], &["a", "b", "b", "b"]).unwrap();
/// let g = 2.0 * (1.0 * (1.0f64 / 2.0).ln() + 3.0 * (3.0f64 / 2.0).ln());
/// assert!((r.statistic - g).abs() < 1e-12);
/// ```
pub fn g_drift<T: Ord>(reference: &[T], current: &[T]) -> Result<ChiSquareResult> {
    gof_drift(reference, current, Gof::LogLikelihood)
}

/// [`g_drift`] for numeric labels: one category per distinct value. Errors
/// additionally on non-finite values.
pub fn g_drift_numeric(reference: &[f64], current: &[f64]) -> Result<ChiSquareResult> {
    ensure_finite("reference", reference)?;
    ensure_finite("current", current)?;
    g_drift(&OrdF64::codes(reference), &OrdF64::codes(current))
}
