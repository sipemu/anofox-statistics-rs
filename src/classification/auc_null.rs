//! Null distribution of the ROC AUC of a random classifier.

use crate::error::{Result, StatError};
use crate::nonparametric::wilcoxon_dist::{
    cdf_ge, mann_whitney_exact_feasible, mann_whitney_pmf, quantile,
};
use statrs::distribution::{ContinuousCDF, Normal};

fn check_sizes(n_pos: usize, n_neg: usize) -> Result<()> {
    if n_pos == 0 || n_neg == 0 {
        return Err(StatError::InvalidParameter(
            "the AUC needs at least one positive and one negative case".into(),
        ));
    }
    Ok(())
}

/// Mean `0.5` and standard deviation `sqrt((n_pos + n_neg + 1) / (12 n_pos n_neg))`
/// of the null AUC.
fn null_sd(n_pos: usize, n_neg: usize) -> f64 {
    let (a, b) = (n_pos as f64, n_neg as f64);
    ((a + b + 1.0) / (12.0 * a * b)).sqrt()
}

/// Quantile of the ROC AUC of a random classifier.
///
/// With continuous scores that do not depend on the labels, the AUC of
/// `n_pos` positive and `n_neg` negative cases is `U / (n_pos n_neg)`, with
/// `U` the Mann-Whitney statistic under the null. This returns the smallest
/// AUC `a` with `P(AUC_random <= a) >= prob`, exactly from the Mann-Whitney
/// distribution for `n_pos * n_neg <= 10000` and from its normal
/// approximation (mean 0.5, variance `(n_pos + n_neg + 1) / (12 n_pos n_neg)`,
/// clipped to `[0, 1]`) beyond.
///
/// evidently's model-based drift detectors (text content drift, the
/// embedding `model` method) flag drift when the domain classifier's AUC
/// exceeds the 95% quantile of the AUCs of random scores, which they
/// simulate (100 or 1000 draws of `np.random.rand`); this is the quantity
/// those simulations estimate.
///
/// Errors if a class is empty or `prob` is outside `[0, 1]`.
///
/// R equivalent: `qwilcox(prob, n_pos, n_neg) / (n_pos * n_neg)`.
///
/// # Example
/// ```
/// use anofox_statistics::classification::auc_null_quantile;
/// // qwilcox(0.95, 10, 10) = 72
/// assert_eq!(auc_null_quantile(10, 10, 0.95).unwrap(), 0.72);
/// ```
pub fn auc_null_quantile(n_pos: usize, n_neg: usize, prob: f64) -> Result<f64> {
    check_sizes(n_pos, n_neg)?;
    if !(0.0..=1.0).contains(&prob) {
        return Err(StatError::InvalidParameter(format!(
            "prob must be in [0, 1], got {}",
            prob
        )));
    }
    let pairs = (n_pos * n_neg) as f64;
    if mann_whitney_exact_feasible(n_pos, n_neg) {
        return Ok(quantile(&mann_whitney_pmf(n_pos, n_neg), prob) as f64 / pairs);
    }
    let z = Normal::new(0.0, 1.0).unwrap().inverse_cdf(prob);
    Ok((0.5 + z * null_sd(n_pos, n_neg)).clamp(0.0, 1.0))
}

/// Probability that a random classifier reaches at least the AUC `auc`:
/// `P(AUC_random >= auc)`, the one-sided p-value of an observed AUC against
/// random scores.
///
/// Exact from the Mann-Whitney distribution for `n_pos * n_neg <= 10000`;
/// beyond, the normal approximation with continuity correction (as
/// `wilcox.test(..., exact = FALSE)`), `1 - Phi((U - 0.5 - n_pos n_neg / 2) /
/// sd_U)` with `U = auc * n_pos * n_neg`.
///
/// Errors if a class is empty or `auc` is outside `[0, 1]`.
///
/// R equivalent: `pwilcox(ceiling(u) - 1, n_pos, n_neg, lower.tail = FALSE)`,
/// `u = auc * n_pos * n_neg`.
///
/// # Example
/// ```
/// use anofox_statistics::classification::auc_null_p_value;
/// assert_eq!(auc_null_p_value(0.0, 3, 4).unwrap(), 1.0);
/// // only 1 of the choose(7, 3) = 35 rankings puts all positives first
/// assert!((auc_null_p_value(1.0, 3, 4).unwrap() - 1.0 / 35.0).abs() < 1e-15);
/// ```
pub fn auc_null_p_value(auc: f64, n_pos: usize, n_neg: usize) -> Result<f64> {
    check_sizes(n_pos, n_neg)?;
    if !(0.0..=1.0).contains(&auc) {
        return Err(StatError::InvalidParameter(format!(
            "auc must be in [0, 1], got {}",
            auc
        )));
    }
    let pairs = (n_pos * n_neg) as f64;
    // An AUC computed from scores is a multiple of 1 / (2 pairs); remove the
    // rounding of the division before taking the ceiling.
    let u = (auc * pairs * 2.0).round() / 2.0;
    let u = if (auc * pairs - u).abs() < 1e-7 {
        u
    } else {
        auc * pairs
    };
    if mann_whitney_exact_feasible(n_pos, n_neg) {
        return Ok(cdf_ge(&mann_whitney_pmf(n_pos, n_neg), u));
    }
    let z = (u - 0.5 - pairs / 2.0) / (null_sd(n_pos, n_neg) * pairs);
    Ok(Normal::new(0.0, 1.0).unwrap().sf(z))
}
