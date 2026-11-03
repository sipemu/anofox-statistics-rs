//! Two-proportion z test on category labels (evidently's `"z"` drift test).

use super::{category_counts, OrdF64};
use crate::error::{Result, StatError};
use crate::parametric::Alternative;
use crate::utils::dist::chisq_sf;
use crate::utils::finite::ensure_finite;
use statrs::distribution::{ContinuousCDF, Normal};

/// Result of [`z_test`].
#[derive(Debug, Clone, PartialEq)]
pub struct ZTestResult {
    /// `z = (p1 - p2) / sqrt(P (1 - P) (1/m + 1/n))`, `0` when only one
    /// category is seen.
    pub statistic: f64,
    /// Normal p-value for the alternative.
    pub p_value: f64,
    /// Shares `(p1, p2)` of the reference and the current sample that are
    /// not the smallest category.
    pub estimate: [f64; 2],
    /// Number of categories seen in either sample.
    pub n_categories: usize,
    /// Size of the reference sample.
    pub n_reference: usize,
    /// Size of the current sample.
    pub n_current: usize,
}

/// Pooled two-proportion z test of a reference against a current sample of
/// category labels, without continuity correction.
///
/// Each label is coded `0` if it equals the smallest category seen in either
/// sample and `1` otherwise, so a binary column compares the share of its
/// larger label. `p1` and `p2` are the shares of `1`s in the reference and
/// the current sample, `P` the pooled share, and
/// `z = (p1 - p2) / sqrt(P (1 - P) (1/m + 1/n))`. When both samples hold a
/// single category, `z = 0` and `p = 1`.
///
/// This is evidently's `"z"` drift test (two-sided, drift at `p < 0.05`),
/// the default test for binary columns of a reference with up to 1000 rows.
///
/// Errors on an empty sample.
///
/// R equivalent: `prop.test(c(x1, x2), c(m, n), correct = FALSE)`, with
/// `z = sign(p1 - p2) * sqrt(statistic)`.
///
/// # Example
/// ```
/// use anofox_statistics::drift::z_test;
/// use anofox_statistics::Alternative;
/// let r = z_test(&["a", "a", "b", "b"], &["b", "b", "b", "b"], Alternative::TwoSided).unwrap();
/// assert!((r.statistic + 1.6329931618554523).abs() < 1e-12);
/// assert_eq!(r.estimate, [0.5, 1.0]);
/// ```
pub fn z_test<T: Ord>(
    reference: &[T],
    current: &[T],
    alternative: Alternative,
) -> Result<ZTestResult> {
    if reference.is_empty() || current.is_empty() {
        return Err(StatError::EmptyData);
    }
    let (r, c) = category_counts(reference, current);
    let (m, n) = (reference.len(), current.len());
    let (mf, nf) = (m as f64, n as f64);
    // Share of labels other than the smallest category.
    let p1 = (m - r[0]) as f64 / mf;
    let p2 = (n - c[0]) as f64 / nf;
    let (statistic, p_value) = if r.len() == 1 {
        (0.0, 1.0)
    } else {
        let pooled = (p1 * mf + p2 * nf) / (mf + nf);
        let z = (p1 - p2) / (pooled * (1.0 - pooled) * (1.0 / mf + 1.0 / nf)).sqrt();
        let norm = Normal::new(0.0, 1.0).unwrap();
        let p = match alternative {
            // As R's prop.test: X^2 = z^2 on one degree of freedom.
            Alternative::TwoSided => chisq_sf(z * z, 1.0),
            Alternative::Less => norm.cdf(z),
            Alternative::Greater => norm.sf(z),
        };
        (z, p.min(1.0))
    };
    Ok(ZTestResult {
        statistic,
        p_value,
        estimate: [p1, p2],
        n_categories: r.len(),
        n_reference: m,
        n_current: n,
    })
}

/// [`z_test`] for numeric labels: one category per distinct value. Errors
/// additionally on non-finite values.
pub fn z_test_numeric(
    reference: &[f64],
    current: &[f64],
    alternative: Alternative,
) -> Result<ZTestResult> {
    ensure_finite("reference", reference)?;
    ensure_finite("current", current)?;
    z_test(
        &OrdF64::codes(reference),
        &OrdF64::codes(current),
        alternative,
    )
}
