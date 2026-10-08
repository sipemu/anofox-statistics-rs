use crate::error::{Result, StatError};
use crate::nonparametric::hodges_lehmann::{mann_whitney_estimate_ci, wilcoxon_estimate_ci};
use crate::nonparametric::ranks::rank_with_ties;
use crate::nonparametric::wilcoxon_dist::{
    cdf_ge, cdf_le, mann_whitney_exact_feasible, mann_whitney_pmf, wilcoxon_exact_feasible,
    wilcoxon_pmf,
};
use crate::parametric::Alternative;
use crate::utils::finite::{ensure_finite_param, ensure_no_nan};
use statrs::distribution::{ContinuousCDF, Normal};

/// Compute tie correction factor: sum(t^3 - t) for all tie groups.
fn tie_correction(tie_sizes: &[usize]) -> f64 {
    tie_sizes
        .iter()
        .map(|&t| {
            let t = t as f64;
            t * t * t - t
        })
        .sum()
}

/// Compute p-value from z-score using standard normal based on alternative hypothesis.
///
/// Uses the survival function (`sf`) directly instead of `1 - cdf(...)` so that
/// extremely small p-values do not underflow to 0 for large |z|.
fn compute_p_value(z: f64, alternative: &Alternative) -> f64 {
    let normal = Normal::new(0.0, 1.0).unwrap();
    match alternative {
        Alternative::TwoSided => 2.0 * normal.sf(z.abs()),
        Alternative::Less => normal.cdf(z),
        Alternative::Greater => normal.sf(z),
    }
}

/// Confidence interval result
#[derive(Debug, Clone)]
pub struct ConfidenceInterval {
    /// Lower bound of the confidence interval
    pub lower: f64,
    /// Upper bound of the confidence interval
    pub upper: f64,
    /// Confidence level (e.g., 0.95 for 95%)
    pub conf_level: f64,
}

/// Result of Mann-Whitney U test
#[derive(Debug, Clone)]
pub struct MannWhitneyResult {
    /// The U statistic (for first sample, matching R's wilcox.test)
    pub statistic: f64,
    /// The p-value
    pub p_value: f64,
    /// Hodges-Lehmann estimate of location shift (difference in medians)
    pub estimate: Option<f64>,
    /// Confidence interval for the location shift
    pub conf_int: Option<ConfidenceInterval>,
    /// Null hypothesis value (location shift under H0)
    pub null_value: f64,
}

/// Result of Wilcoxon Signed-Rank test
#[derive(Debug, Clone)]
pub struct WilcoxonResult {
    /// The V statistic (sum of positive ranks)
    pub statistic: f64,
    /// The p-value
    pub p_value: f64,
    /// Hodges-Lehmann estimate (pseudo-median of differences)
    pub estimate: Option<f64>,
    /// Confidence interval for the pseudo-median
    pub conf_int: Option<ConfidenceInterval>,
    /// Null hypothesis value (median difference under H0)
    pub null_value: f64,
}

/// Perform Mann-Whitney U test (Wilcoxon rank-sum test) for two independent samples.
///
/// # Arguments
/// * `x` - First sample
/// * `y` - Second sample
/// * `alternative` - Alternative hypothesis (TwoSided, Less, Greater)
/// * `continuity_correction` - Whether to apply continuity correction (only for normal approximation)
/// * `exact` - Whether to compute exact p-value (recommended for small samples without ties).
///   The exact null distribution is only used without ties and for
///   `n1 * n2 <= 10_000`; larger samples always use the normal approximation
///   (R's automatic rule is `n1 < 50 && n2 < 50`).
/// * `conf_level` - If Some, compute confidence interval at this level (e.g., 0.95)
/// * `mu` - If Some, test for location shift equal to this value instead of 0
///
/// # Estimate and confidence interval
/// As R's `wilcox.test(x, y, conf.int = TRUE)`: with the exact distribution
/// the estimate is the median of the pairwise differences `x_i - y_j` and the
/// interval their order statistics; otherwise (normal approximation) the
/// estimate and limits are the shifts at which the standardised rank-sum
/// statistic crosses 0 and the normal quantiles, found with R's root finder
/// (tolerance 1e-4). One-sided alternatives give one-sided intervals.
///
/// # Complexity
/// `O(n log n)` time and `O(n)` memory with `n = n1 + n2` (the pairwise
/// differences are not materialised on the normal-approximation path; the
/// exact path is limited to small samples).
///
/// # Returns
/// * `MannWhitneyResult` containing U statistic, p-value, and optionally estimate/CI
pub fn mann_whitney_u(
    x: &[f64],
    y: &[f64],
    alternative: Alternative,
    continuity_correction: bool,
    exact: bool,
    conf_level: Option<f64>,
    mu: Option<f64>,
) -> Result<MannWhitneyResult> {
    if x.is_empty() {
        return Err(StatError::EmptyData);
    }
    if y.is_empty() {
        return Err(StatError::EmptyData);
    }
    // ±Inf ranks as an extreme value (as in R); NaN cannot be ranked.
    ensure_no_nan("x", x)?;
    ensure_no_nan("y", y)?;
    if let Some(m) = mu {
        ensure_finite_param("mu", m)?;
    }

    let nx = x.len();
    let ny = y.len();
    let n = nx + ny;

    // Apply location shift if mu is specified
    let mu_shift = mu.unwrap_or(0.0);
    let y_shifted: Vec<f64> = y.iter().map(|yi| yi + mu_shift).collect();
    let y_use = if mu.is_some() { &y_shifted } else { y };

    // Combine samples and rank
    let mut combined: Vec<f64> = Vec::with_capacity(n);
    combined.extend_from_slice(x);
    combined.extend_from_slice(y_use);

    let (ranks, tie_sizes) = rank_with_ties(&combined)?;

    // Sum of ranks for first sample
    let r1: f64 = ranks[..nx].iter().sum();

    // U statistic for first sample
    // U1 = R1 - n1(n1+1)/2
    let u1 = r1 - (nx * (nx + 1)) as f64 / 2.0;

    // Expected value and variance under null
    let nx_f = nx as f64;
    let ny_f = ny as f64;
    let n_f = n as f64;

    let mu = nx_f * ny_f / 2.0;

    // Variance with tie correction: Var(U) = (n1*n2/12) * (n + 1 - sum(t^3 - t)/(n*(n-1)))
    let tc = tie_correction(&tie_sizes);
    let sigma_sq = (nx_f * ny_f / 12.0) * ((n_f + 1.0) - tc / (n_f * (n_f - 1.0)));
    let sigma = sigma_sq.sqrt();

    // Compute p-value
    let has_ties = tie_sizes.iter().any(|&t| t > 1);
    // Exact distribution only without ties and for small samples; larger
    // samples always use the normal approximation (bounded cost).
    let use_exact = exact && !has_ties && mann_whitney_exact_feasible(nx, ny);
    let p_value = if use_exact {
        mann_whitney_exact_p(nx, ny, u1, &alternative)
    } else if sigma_sq <= 0.0 {
        // Every observation is tied: Var(U) = 0 and U equals its null
        // expectation, so there is no evidence of a shift. R returns NaN here
        // (0/0) for the two-sided test and 1 for one-sided tests;
        // scipy.stats.mannwhitneyu returns 1. We return 1.
        1.0
    } else {
        // Normal approximation with optional continuity correction
        let correction = if continuity_correction { 0.5 } else { 0.0 };
        let z = match alternative {
            Alternative::TwoSided => {
                // R: CORRECTION = sign(z) * 0.5, i.e. no correction when the
                // statistic equals its null expectation (z = 0, p = 1).
                if u1 > mu {
                    (u1 - mu - correction) / sigma
                } else if u1 < mu {
                    (u1 - mu + correction) / sigma
                } else {
                    0.0
                }
            }
            Alternative::Less => (u1 - mu + correction) / sigma,
            Alternative::Greater => (u1 - mu - correction) / sigma,
        };
        compute_p_value(z, &alternative)
    };

    // Compute Hodges-Lehmann estimate and confidence interval if requested
    let (estimate, conf_int) = if let Some(level) = conf_level {
        if !(0.0 < level && level < 1.0) {
            return Err(StatError::InvalidParameter(
                "conf_level must be between 0 and 1".to_string(),
            ));
        }
        let (est, ci) =
            mann_whitney_estimate_ci(x, y, level, alternative, continuity_correction, use_exact);
        (Some(est), Some(ci))
    } else {
        (None, None)
    };

    Ok(MannWhitneyResult {
        statistic: u1,
        p_value,
        estimate,
        conf_int,
        null_value: mu_shift,
    })
}

/// Perform Wilcoxon Signed-Rank test for paired samples.
///
/// # Arguments
/// * `x` - First sample
/// * `y` - Second sample (must be same length as x)
/// * `alternative` - Alternative hypothesis (TwoSided, Less, Greater)
/// * `continuity_correction` - Whether to apply continuity correction (only for normal approximation)
/// * `exact` - Whether to compute exact p-value (recommended for small samples without ties).
///   The exact null distribution is only used without ties and for at most
///   300 non-zero differences; larger samples always use the normal
///   approximation (R's automatic rule is `n < 50`).
/// * `conf_level` - If Some, compute confidence interval at this level (e.g., 0.95)
/// * `mu` - If Some, test if median difference equals this value instead of 0
///
/// # Estimate and confidence interval
/// As R's `wilcox.test(x, y, paired = TRUE, conf.int = TRUE)`, computed on the
/// differences `x - y` (not shifted by `mu`, zeros kept): with the exact
/// distribution (no ties, no zero differences) the (pseudo)median of the Walsh
/// averages and their order statistics; otherwise the shifts at which the
/// standardised signed-rank statistic crosses 0 and the normal quantiles
/// (R's root finder, tolerance 1e-4). One-sided alternatives give one-sided
/// intervals.
///
/// # Complexity
/// `O(n log n)` time and `O(n)` memory (the `n(n+1)/2` Walsh averages are not
/// materialised on the normal-approximation path; the exact path is limited to
/// small samples).
///
/// # Returns
/// * `WilcoxonResult` containing V statistic, p-value, and optionally estimate/CI
pub fn wilcoxon_signed_rank(
    x: &[f64],
    y: &[f64],
    alternative: Alternative,
    continuity_correction: bool,
    exact: bool,
    conf_level: Option<f64>,
    mu: Option<f64>,
) -> Result<WilcoxonResult> {
    let n = x.len();

    if n == 0 {
        return Err(StatError::EmptyData);
    }

    if n != y.len() {
        return Err(StatError::InvalidParameter(format!(
            "Wilcoxon signed-rank test requires equal length samples, got {} and {}",
            n,
            y.len()
        )));
    }

    // ±Inf ranks as an extreme value (as in R); NaN cannot be ranked.
    ensure_no_nan("x", x)?;
    ensure_no_nan("y", y)?;
    if let Some(m) = mu {
        ensure_finite_param("mu", m)?;
    }

    // Apply mu shift if specified (null hypothesis: median difference = mu)
    let mu_shift = mu.unwrap_or(0.0);

    // Compute differences. Like R, drop undefined differences (Inf - Inf =
    // NaN) and then the zeros.
    let defined: Vec<f64> = x
        .iter()
        .zip(y.iter())
        .map(|(xi, yi)| xi - yi - mu_shift)
        .filter(|d| !d.is_nan())
        .collect();
    if defined.is_empty() {
        return Err(StatError::InsufficientData { needed: 1, got: 0 });
    }
    let n_defined = defined.len();
    let diffs: Vec<f64> = defined.into_iter().filter(|&d| d != 0.0).collect();

    let n_nonzero = diffs.len();

    if n_nonzero == 0 {
        // All differences are zero - return 0 statistic with p-value 1
        return Ok(WilcoxonResult {
            statistic: 0.0,
            p_value: 1.0,
            estimate: None,
            conf_int: None,
            null_value: mu_shift,
        });
    }

    // Rank absolute differences
    let abs_diffs: Vec<f64> = diffs.iter().map(|d| d.abs()).collect();
    let (ranks, tie_sizes) = rank_with_ties(&abs_diffs)?;

    // V = sum of ranks where difference is positive
    let v: f64 = diffs
        .iter()
        .zip(ranks.iter())
        .filter(|(&d, _)| d > 0.0)
        .map(|(_, &r)| r)
        .sum();

    // Normal approximation
    let n_f = n_nonzero as f64;

    // Expected value under null: E(V) = n(n+1)/4
    let mu = n_f * (n_f + 1.0) / 4.0;

    // Variance with tie correction: Var(V) = n(n+1)(2n+1)/24 - sum(t^3 - t)/48
    let tc = tie_correction(&tie_sizes);
    let sigma_sq = n_f * (n_f + 1.0) * (2.0 * n_f + 1.0) / 24.0 - tc / 48.0;
    let sigma = sigma_sq.sqrt();

    // Compute p-value
    let has_ties = tie_sizes.iter().any(|&t| t > 1);
    // Exact distribution only without ties and for small samples; larger
    // samples always use the normal approximation (bounded cost).
    let use_exact = exact && !has_ties && wilcoxon_exact_feasible(n_nonzero);
    let p_value = if use_exact {
        wilcoxon_exact_p(n_nonzero, v, &alternative)
    } else {
        // Normal approximation with optional continuity correction
        let correction = if continuity_correction { 0.5 } else { 0.0 };
        let z = match alternative {
            Alternative::TwoSided => {
                // R: CORRECTION = sign(z) * 0.5, i.e. no correction when the
                // statistic equals its null expectation (z = 0, p = 1).
                if v > mu {
                    (v - mu - correction) / sigma
                } else if v < mu {
                    (v - mu + correction) / sigma
                } else {
                    0.0
                }
            }
            Alternative::Less => (v - mu + correction) / sigma,
            Alternative::Greater => (v - mu - correction) / sigma,
        };
        compute_p_value(z, &alternative)
    };

    // Compute Hodges-Lehmann estimate and confidence interval if requested
    let (estimate, conf_int) = if let Some(level) = conf_level {
        if !(0.0 < level && level < 1.0) {
            return Err(StatError::InvalidParameter(
                "conf_level must be between 0 and 1".to_string(),
            ));
        }
        // As in R, the interval is computed on the unshifted differences
        // x - y (including zeros); the exact interval requires no zeros.
        let raw: Vec<f64> = x
            .iter()
            .zip(y.iter())
            .map(|(xi, yi)| xi - yi)
            .filter(|d| !d.is_nan())
            .collect();
        let no_zeros = n_nonzero == n_defined;
        let (est, ci) = wilcoxon_estimate_ci(
            &raw,
            level,
            alternative,
            continuity_correction,
            use_exact && no_zeros,
        );
        (Some(est), Some(ci))
    } else {
        (None, None)
    };

    Ok(WilcoxonResult {
        statistic: v,
        p_value,
        estimate,
        conf_int,
        null_value: mu_shift,
    })
}

// ============================================
// Exact p-value computation
// ============================================

/// Exact Mann-Whitney p-value from the null pmf of U (R's `pwilcox`).
fn mann_whitney_exact_p(n1: usize, n2: usize, u: f64, alternative: &Alternative) -> f64 {
    let pmf = mann_whitney_pmf(n1, n2);
    match alternative {
        Alternative::TwoSided => {
            // R wilcox.test: use the tail on the side of the observed U
            // (P(U >= u) above the centre n1*n2/2, P(U <= u) otherwise),
            // double it and cap at 1.
            let p = if 2.0 * u > (n1 * n2) as f64 {
                cdf_ge(&pmf, u)
            } else {
                cdf_le(&pmf, u)
            };
            (2.0 * p).min(1.0)
        }
        Alternative::Less => cdf_le(&pmf, u),
        Alternative::Greater => cdf_ge(&pmf, u),
    }
}

/// Exact signed-rank p-value from the null pmf of V (R's `psignrank`).
fn wilcoxon_exact_p(n: usize, v: f64, alternative: &Alternative) -> f64 {
    let pmf = wilcoxon_pmf(n);
    let max_v = (n * (n + 1) / 2) as f64;
    match alternative {
        Alternative::TwoSided => {
            let p_lower = cdf_le(&pmf, v);
            let p_upper = cdf_le(&pmf, max_v - v);
            2.0 * p_lower.min(p_upper).min(0.5)
        }
        // V is small when negative differences dominate
        Alternative::Less => cdf_le(&pmf, v),
        // V is large when positive differences dominate
        Alternative::Greater => cdf_ge(&pmf, v),
    }
}

// ============================================
// Rank-biserial effect sizes
// ============================================

/// Rank-biserial correlation from a Mann-Whitney U statistic:
/// `r = 1 - 2 * u1 / (n1 * n2)`.
///
/// `u1` is the U statistic of the first sample, i.e.
/// [`MannWhitneyResult::statistic`] (R's `W`), which counts the pairs with
/// `x > y` (ties count 0.5). Equivalently `r = P(y > x) - P(x > y)`.
///
/// **Sign convention:** `r > 0` when the second sample tends to be *larger*,
/// `r < 0` when the first sample tends to be larger; `r` lies in `[-1, 1]`.
/// R's `effectsize::rank_biserial(x, y)` uses the opposite sign
/// (`2 * u1 / (n1 * n2) - 1`).
///
/// Returns NaN when `n1 * n2 == 0`.
pub fn rank_biserial_from_u(u1: f64, n1: usize, n2: usize) -> f64 {
    let nn = n1 as f64 * n2 as f64;
    if nn == 0.0 {
        return f64::NAN;
    }
    1.0 - 2.0 * u1 / nn
}

/// Rank-biserial correlation effect size for the Mann-Whitney U test,
/// `r = 1 - 2 * U1 / (n1 * n2)` (see [`rank_biserial_from_u`] for the sign
/// convention: positive when `y` tends to be larger than `x`).
///
/// `mu` shifts `y` exactly as in [`mann_whitney_u`] (the location shift under
/// the null), so the result is consistent with that test's statistic.
///
/// # Examples
/// ```
/// use anofox_statistics::rank_biserial;
///
/// let x = [5.1, 4.9, 6.2, 5.8, 6.05, 5.5, 5.3, 6.1];
/// let y = [6.5, 7.1, 6.8, 7.4, 6.0, 7.9, 6.6, 7.2, 6.9, 7.05];
/// // R: 1 - 2 * wilcox.test(x, y)$statistic / (8 * 10)
/// assert!((rank_biserial(&x, &y, None).unwrap() - 0.925).abs() < 1e-12);
/// ```
pub fn rank_biserial(x: &[f64], y: &[f64], mu: Option<f64>) -> Result<f64> {
    if x.is_empty() || y.is_empty() {
        return Err(StatError::EmptyData);
    }
    ensure_no_nan("x", x)?;
    ensure_no_nan("y", y)?;
    if let Some(m) = mu {
        ensure_finite_param("mu", m)?;
    }
    let shift = mu.unwrap_or(0.0);
    let mut u1 = 0.0;
    for &xi in x {
        for &yj in y {
            let yj = yj + shift;
            if xi > yj {
                u1 += 1.0;
            } else if xi == yj {
                u1 += 0.5;
            }
        }
    }
    Ok(rank_biserial_from_u(u1, x.len(), y.len()))
}

/// Matched-pairs rank-biserial correlation for the Wilcoxon signed-rank test:
/// `r = (T- - T+) / (T+ + T-)`, where `T+` / `T-` are the sums of the ranks of
/// `|d|` for positive / negative differences `d = x - y - mu` (zero differences
/// dropped, average ranks for ties, as in [`wilcoxon_signed_rank`]).
///
/// The sign convention matches [`rank_biserial`]: `r > 0` when `y` tends to be
/// larger than `x`. R's `effectsize::rank_biserial(x, y, paired = TRUE)` uses
/// the opposite sign (`(T+ - T-) / (T+ + T-)`).
///
/// Returns `Ok(NaN)` when every difference is zero.
pub fn matched_pairs_rank_biserial(x: &[f64], y: &[f64], mu: Option<f64>) -> Result<f64> {
    if x.is_empty() {
        return Err(StatError::EmptyData);
    }
    if x.len() != y.len() {
        return Err(StatError::InvalidParameter(format!(
            "paired samples must have equal length, got {} and {}",
            x.len(),
            y.len()
        )));
    }
    ensure_no_nan("x", x)?;
    ensure_no_nan("y", y)?;
    if let Some(m) = mu {
        ensure_finite_param("mu", m)?;
    }
    let shift = mu.unwrap_or(0.0);
    // As in wilcoxon_signed_rank: drop undefined (Inf - Inf) and zero differences.
    let diffs: Vec<f64> = x
        .iter()
        .zip(y)
        .map(|(a, b)| a - b - shift)
        .filter(|&d| d != 0.0 && !d.is_nan())
        .collect();
    if diffs.is_empty() {
        return Ok(f64::NAN);
    }
    let abs: Vec<f64> = diffs.iter().map(|d| d.abs()).collect();
    let (ranks, _) = rank_with_ties(&abs)?;
    let (mut t_pos, mut t_neg) = (0.0, 0.0);
    for (d, r) in diffs.iter().zip(&ranks) {
        if *d > 0.0 {
            t_pos += r;
        } else {
            t_neg += r;
        }
    }
    Ok((t_neg - t_pos) / (t_pos + t_neg))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::excessive_precision, clippy::approx_constant)]
    fn test_mann_whitney_exact_two_sided_both_orders() {
        // R: wilcox.test(x, y, exact = TRUE): W = 58, p = 0.3153781203316808
        let x = [1.83, 0.50, 1.62, 2.48, 1.68, 1.88, 1.55, 3.06, 1.30];
        let y = [0.878, 0.647, 0.598, 2.05, 1.06, 1.29, 1.07, 3.14, 1.28, 4.1];
        let r = mann_whitney_u(&x, &y, Alternative::TwoSided, true, true, None, None).unwrap();
        assert_eq!(r.statistic, 58.0);
        assert!(
            (r.p_value - 0.31537812033168078).abs() < 1e-12,
            "{}",
            r.p_value
        );
        let r = mann_whitney_u(&y, &x, Alternative::TwoSided, true, true, None, None).unwrap();
        assert_eq!(r.statistic, 32.0);
        assert!(
            (r.p_value - 0.31537812033168078).abs() < 1e-12,
            "{}",
            r.p_value
        );
        // centre of the distribution: p capped at 1
        let r = mann_whitney_u(
            &[1.0, 4.0],
            &[2.0, 3.0],
            Alternative::TwoSided,
            true,
            true,
            None,
            None,
        )
        .unwrap();
        assert_eq!(r.p_value, 1.0);
    }

    #[test]
    fn test_rank_biserial() {
        let x = [5.1, 4.9, 6.2, 5.8, 6.05, 5.5, 5.3, 6.1];
        let y = [6.5, 7.1, 6.8, 7.4, 6.0, 7.9, 6.6, 7.2, 6.9, 7.05];
        // R: W <- wilcox.test(x, y, exact = FALSE)$statistic; 1 - 2*W/(8*10)
        let r = rank_biserial(&x, &y, None).unwrap();
        assert!((r - 0.925).abs() < 1e-12);
        assert!((rank_biserial(&y, &x, None).unwrap() + 0.925).abs() < 1e-12);
        // consistent with the test statistic, also with ties and a shift
        let mw =
            mann_whitney_u(&x, &y, Alternative::TwoSided, true, false, None, Some(-1.0)).unwrap();
        let r = rank_biserial(&x, &y, Some(-1.0)).unwrap();
        assert!((r - rank_biserial_from_u(mw.statistic, 8, 10)).abs() < 1e-12);
        assert_eq!(rank_biserial(&[1.0, 1.0], &[1.0], None).unwrap(), 0.0);
        assert!(rank_biserial(&[], &[1.0], None).is_err());
    }

    #[test]
    #[allow(clippy::excessive_precision)]
    fn test_matched_pairs_rank_biserial() {
        let x = [5.1, 4.9, 6.2, 5.8, 6.05, 5.5, 5.3, 6.1];
        let y = [6.5, 7.1, 6.8, 7.4, 6.0, 7.9, 6.6, 7.2];
        // R: d <- x - y; d <- d[d != 0]; r <- rank(abs(d));
        //    (sum(r[d < 0]) - sum(r[d > 0])) / sum(r)
        let r = matched_pairs_rank_biserial(&x, &y, None).unwrap();
        assert!((r - 0.94444444444444442).abs() < 1e-12, "{r}");
        assert!(matched_pairs_rank_biserial(&x, &x, None).unwrap().is_nan());
        assert!(matched_pairs_rank_biserial(&x, &y[..7], None).is_err());
    }

    #[test]
    fn test_mann_whitney_all_tied_p_is_one() {
        // Var(U) = 0: scipy.stats.mannwhitneyu -> p = 1 (R: NaN two-sided, 1 one-sided)
        let x = [5.0; 6];
        let y = [5.0; 7];
        for alt in [
            Alternative::TwoSided,
            Alternative::Less,
            Alternative::Greater,
        ] {
            let r = mann_whitney_u(&x, &y, alt, true, false, None, None).unwrap();
            assert_eq!(r.p_value, 1.0);
            assert_eq!(r.statistic, 21.0);
        }
    }

    #[test]
    fn test_mann_whitney_u_at_null_expectation_no_correction() {
        // R: wilcox.test(c(1, 4), c(2, 3), exact = FALSE)$p.value == 1
        let r = mann_whitney_u(
            &[1.0, 4.0],
            &[2.0, 3.0],
            Alternative::TwoSided,
            true,
            false,
            None,
            None,
        )
        .unwrap();
        assert_eq!(r.p_value, 1.0);
    }

    #[test]
    fn test_wilcoxon_v_at_null_expectation_no_correction() {
        // R: wilcox.test(c(1,2,3,4), c(4,3,2,1), paired = TRUE, exact = FALSE)$p.value == 1
        let r = wilcoxon_signed_rank(
            &[1.0, 2.0, 3.0, 4.0],
            &[4.0, 3.0, 2.0, 1.0],
            Alternative::TwoSided,
            true,
            false,
            None,
            None,
        )
        .unwrap();
        assert_eq!(r.p_value, 1.0);
    }
}
