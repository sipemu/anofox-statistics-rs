use crate::error::{Result, StatError};
use crate::nonparametric::ranks::rank_with_ties;
use crate::parametric::Alternative;
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
/// * `exact` - Whether to compute exact p-value (recommended for small samples without ties)
/// * `conf_level` - If Some, compute confidence interval at this level (e.g., 0.95)
/// * `mu` - If Some, test for location shift equal to this value instead of 0
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
    let p_value = if exact && !has_ties {
        // Exact p-value using enumeration
        mann_whitney_exact_p(nx, ny, u1 as usize, &alternative)
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
        let (est, ci) = mann_whitney_estimate_ci(x, y, level, exact && !has_ties)?;
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
/// * `exact` - Whether to compute exact p-value (recommended for small samples without ties)
/// * `conf_level` - If Some, compute confidence interval at this level (e.g., 0.95)
/// * `mu` - If Some, test if median difference equals this value instead of 0
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

    // Apply mu shift if specified (null hypothesis: median difference = mu)
    let mu_shift = mu.unwrap_or(0.0);

    // Compute differences and filter out zeros
    let diffs: Vec<f64> = x
        .iter()
        .zip(y.iter())
        .map(|(xi, yi)| xi - yi - mu_shift)
        .filter(|&d| d != 0.0)
        .collect();

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
    let p_value = if exact && !has_ties {
        // Exact p-value using enumeration
        wilcoxon_exact_p(n_nonzero, v as usize, &alternative)
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
        let (est, ci) = wilcoxon_estimate_ci(&diffs, level, exact && !has_ties)?;
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

/// Compute exact p-value for Mann-Whitney U test using dynamic programming.
///
/// Uses the recursion: count(n1, n2, u) = count(n1-1, n2, u-n2) + count(n1, n2-1, u)
fn mann_whitney_exact_p(n1: usize, n2: usize, u: usize, alternative: &Alternative) -> f64 {
    let total = mann_whitney_count_all(n1, n2);

    match alternative {
        Alternative::TwoSided => {
            // R wilcox.test: use the tail on the side of the observed U
            // (P(U >= u) above the centre n1*n2/2, P(U <= u) otherwise),
            // double it and cap at 1.
            let p = if 2 * u > n1 * n2 {
                mann_whitney_count_ge(n1, n2, u) as f64 / total as f64
            } else {
                mann_whitney_count_le(n1, n2, u) as f64 / total as f64
            };
            (2.0 * p).min(1.0)
        }
        Alternative::Less => {
            // P(U <= u)
            mann_whitney_count_le(n1, n2, u) as f64 / total as f64
        }
        Alternative::Greater => {
            // P(U >= u)
            mann_whitney_count_ge(n1, n2, u) as f64 / total as f64
        }
    }
}

/// Count total number of Mann-Whitney configurations: C(n1+n2, n1)
fn mann_whitney_count_all(n1: usize, n2: usize) -> u64 {
    binomial(n1 + n2, n1)
}

/// Count configurations with U <= u using recursion with memoization
fn mann_whitney_count_le(n1: usize, n2: usize, u: usize) -> u64 {
    mann_whitney_count_le_recursive(n1, n2, u)
}

/// Recursive count with memoization for Mann-Whitney U <= u
fn mann_whitney_count_le_recursive(n1: usize, n2: usize, max_u: usize) -> u64 {
    use std::collections::HashMap;

    fn count(
        n1: usize,
        n2: usize,
        u: usize,
        memo: &mut HashMap<(usize, usize, usize), u64>,
    ) -> u64 {
        if n1 == 0 {
            return 1; // Only one way: all n2 elements at the end
        }
        if n2 == 0 {
            return 1; // Only one way: all n1 elements
        }

        if let Some(&result) = memo.get(&(n1, n2, u)) {
            return result;
        }

        // Either last position goes to sample 1 (contributes n2 to U) or sample 2 (contributes 0)
        let mut result = count(n1, n2 - 1, u, memo); // Last position to sample 2
        if u >= n2 {
            result += count(n1 - 1, n2, u - n2, memo); // Last position to sample 1
        }

        memo.insert((n1, n2, u), result);
        result
    }

    let mut memo = HashMap::new();
    count(n1, n2, max_u, &mut memo)
}

/// Count configurations with U >= u
fn mann_whitney_count_ge(n1: usize, n2: usize, u: usize) -> u64 {
    let total = mann_whitney_count_all(n1, n2);
    let less = if u > 0 {
        mann_whitney_count_le_recursive(n1, n2, u - 1)
    } else {
        0
    };
    total - less
}

/// Compute exact p-value for Wilcoxon signed-rank test.
///
/// Uses the fact that V can range from 0 to n(n+1)/2, with equal probability
/// for each configuration of signs.
fn wilcoxon_exact_p(n: usize, v: usize, alternative: &Alternative) -> f64 {
    let total = 1u64 << n; // 2^n possible sign combinations
    let max_v = n * (n + 1) / 2;

    match alternative {
        Alternative::TwoSided => {
            let v_symmetric = max_v - v;
            let p_lower = wilcoxon_count_le(n, v) as f64 / total as f64;
            let p_upper = wilcoxon_count_le(n, v_symmetric) as f64 / total as f64;
            2.0 * p_lower.min(p_upper).min(0.5)
        }
        Alternative::Less => {
            // V is small when negative differences dominate
            wilcoxon_count_le(n, v) as f64 / total as f64
        }
        Alternative::Greater => {
            // V is large when positive differences dominate
            wilcoxon_count_ge(n, v) as f64 / total as f64
        }
    }
}

/// Count sign combinations with V <= v using dynamic programming
fn wilcoxon_count_le(n: usize, v: usize) -> u64 {
    // dp[j] = number of ways to get sum <= j using ranks 1..i
    let max_v = n * (n + 1) / 2;
    let v = v.min(max_v);

    // dp[j] = count of ways to achieve exactly sum j
    let mut dp = vec![0u64; v + 1];
    dp[0] = 1;

    for rank in 1..=n {
        // Process in reverse to avoid using updated values
        for j in (rank..=v).rev() {
            dp[j] += dp[j - rank];
        }
    }

    // Sum up all counts for V <= v
    dp.iter().sum()
}

/// Count sign combinations with V >= v
fn wilcoxon_count_ge(n: usize, v: usize) -> u64 {
    let total = 1u64 << n;
    let less = if v > 0 {
        wilcoxon_count_le(n, v - 1)
    } else {
        0
    };
    total - less
}

/// Compute binomial coefficient C(n, k)
fn binomial(n: usize, k: usize) -> u64 {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }
    let k = k.min(n - k); // Take advantage of symmetry
    let mut result = 1u64;
    for i in 0..k {
        result = result * (n - i) as u64 / (i + 1) as u64;
    }
    result
}

// ============================================
// Hodges-Lehmann estimates and confidence intervals
// ============================================

/// Compute Hodges-Lehmann estimate and confidence interval for Mann-Whitney U test.
///
/// The estimate is the median of all pairwise differences (y_j - x_i).
/// The confidence interval is based on order statistics of these differences.
fn mann_whitney_estimate_ci(
    x: &[f64],
    y: &[f64],
    conf_level: f64,
    exact: bool,
) -> Result<(f64, ConfidenceInterval)> {
    let nx = x.len();
    let ny = y.len();
    let n_pairs = nx * ny;

    // Compute all pairwise differences: x_i - y_j (to match R's convention)
    let mut diffs: Vec<f64> = Vec::with_capacity(n_pairs);
    for xi in x {
        for yi in y {
            diffs.push(xi - yi);
        }
    }
    diffs.sort_by(|a, b| a.total_cmp(b));

    // Hodges-Lehmann estimate: median of pairwise differences
    let estimate = if n_pairs % 2 == 0 {
        (diffs[n_pairs / 2 - 1] + diffs[n_pairs / 2]) / 2.0
    } else {
        diffs[n_pairs / 2]
    };

    // Confidence interval based on critical values
    let alpha = 1.0 - conf_level;
    let (k_lower, k_upper) = if exact && nx <= 20 && ny <= 20 {
        mann_whitney_ci_exact_k(nx, ny, alpha)
    } else {
        mann_whitney_ci_approx_k(nx, ny, alpha)
    };

    let lower = if k_lower > 0 {
        diffs[k_lower - 1]
    } else {
        diffs[0]
    };
    let upper = if k_upper <= n_pairs {
        diffs[k_upper - 1]
    } else {
        diffs[n_pairs - 1]
    };

    Ok((
        estimate,
        ConfidenceInterval {
            lower,
            upper,
            conf_level,
        },
    ))
}

/// Find exact k values for Mann-Whitney CI using enumeration.
/// Returns (k_lower, k_upper) as 0-based indices into the sorted differences array.
/// Matches R's qwilcox approach: find smallest k such that P(U <= k) >= alpha/2.
fn mann_whitney_ci_exact_k(n1: usize, n2: usize, alpha: f64) -> (usize, usize) {
    let total = mann_whitney_count_all(n1, n2) as f64;
    let n_pairs = n1 * n2;

    // Find qwilcox(alpha/2, n1, n2) - smallest k such that P(U <= k) >= alpha/2
    let mut wci_lo = 0;
    for k in 0..n_pairs {
        let p = mann_whitney_count_le_recursive(n1, n2, k) as f64 / total;
        if p >= alpha / 2.0 {
            wci_lo = k;
            break;
        }
    }

    // wci_hi = n_pairs - wci_lo
    let wci_hi = n_pairs - wci_lo;

    // R uses: lower = diffs[wci_lo] (1-based), upper = diffs[wci_hi + 1] (1-based)
    // In 0-based: lower = diffs[wci_lo - 1], upper = diffs[wci_hi]
    // But wci_lo from qwilcox is already the correct 1-based index
    // So in 0-based: lower = wci_lo - 1, upper = wci_hi (which is wci_hi+1 - 1)
    let k_lower = if wci_lo > 0 { wci_lo } else { 1 };
    let k_upper = wci_hi + 1;

    (k_lower, k_upper)
}

/// Approximate k values for Mann-Whitney CI using normal approximation
fn mann_whitney_ci_approx_k(n1: usize, n2: usize, alpha: f64) -> (usize, usize) {
    let n1_f = n1 as f64;
    let n2_f = n2 as f64;
    let n_pairs = n1 * n2;

    let mu = n1_f * n2_f / 2.0;
    let sigma = (n1_f * n2_f * (n1_f + n2_f + 1.0) / 12.0).sqrt();

    let normal = Normal::new(0.0, 1.0).unwrap();
    let z = normal.inverse_cdf(1.0 - alpha / 2.0);

    let k_lower = ((mu - z * sigma).floor() as usize).max(1);
    let k_upper = ((mu + z * sigma).ceil() as usize + 1).min(n_pairs);

    (k_lower, k_upper)
}

/// Compute Hodges-Lehmann estimate and CI for Wilcoxon signed-rank test.
///
/// The estimate is the median of Walsh averages (d_i + d_j) / 2 for i <= j.
fn wilcoxon_estimate_ci(
    diffs: &[f64],
    conf_level: f64,
    exact: bool,
) -> Result<(f64, ConfidenceInterval)> {
    let n = diffs.len();
    let n_walsh = n * (n + 1) / 2;

    // Compute all Walsh averages: (d_i + d_j) / 2 for i <= j
    let mut walsh: Vec<f64> = Vec::with_capacity(n_walsh);
    for i in 0..n {
        for j in i..n {
            walsh.push((diffs[i] + diffs[j]) / 2.0);
        }
    }
    walsh.sort_by(|a, b| a.total_cmp(b));

    // Hodges-Lehmann estimate: median of Walsh averages
    let estimate = if n_walsh % 2 == 0 {
        (walsh[n_walsh / 2 - 1] + walsh[n_walsh / 2]) / 2.0
    } else {
        walsh[n_walsh / 2]
    };

    // Confidence interval based on critical values
    let alpha = 1.0 - conf_level;
    let (k_lower, k_upper) = if exact && n <= 20 {
        wilcoxon_ci_exact_k(n, alpha)
    } else {
        wilcoxon_ci_approx_k(n, alpha)
    };

    let lower = if k_lower > 0 {
        walsh[k_lower - 1]
    } else {
        walsh[0]
    };
    let upper = if k_upper <= n_walsh {
        walsh[k_upper - 1]
    } else {
        walsh[n_walsh - 1]
    };

    Ok((
        estimate,
        ConfidenceInterval {
            lower,
            upper,
            conf_level,
        },
    ))
}

/// Find exact k values for Wilcoxon CI.
/// Returns (k_lower, k_upper) as 1-based indices into the sorted Walsh averages array.
/// Matches R's qsignrank approach: find smallest k such that P(V <= k) >= alpha/2.
fn wilcoxon_ci_exact_k(n: usize, alpha: f64) -> (usize, usize) {
    let total = (1u64 << n) as f64;
    let n_walsh = n * (n + 1) / 2;

    // Find qsignrank(alpha/2, n) - smallest k such that P(V <= k) >= alpha/2
    let mut k_lo = 0;
    for k in 0..=n_walsh {
        let p = wilcoxon_count_le(n, k) as f64 / total;
        if p >= alpha / 2.0 {
            k_lo = k;
            break;
        }
    }

    // k_hi = n_walsh - k_lo (by symmetry)
    let k_hi = n_walsh - k_lo;

    // R uses: lower = walsh[k_lo] (1-based), upper = walsh[k_hi + 1] (1-based)
    // So in 1-based terms: k_lower = k_lo, k_upper = k_hi + 1
    let k_lower = if k_lo > 0 { k_lo } else { 1 };
    let k_upper = k_hi + 1;

    (k_lower, k_upper)
}

/// Approximate k values for Wilcoxon CI using normal approximation
fn wilcoxon_ci_approx_k(n: usize, alpha: f64) -> (usize, usize) {
    let n_f = n as f64;
    let n_walsh = n * (n + 1) / 2;

    let mu = n_f * (n_f + 1.0) / 4.0;
    let sigma = (n_f * (n_f + 1.0) * (2.0 * n_f + 1.0) / 24.0).sqrt();

    let normal = Normal::new(0.0, 1.0).unwrap();
    let z = normal.inverse_cdf(1.0 - alpha / 2.0);

    let k_lower = ((mu - z * sigma).floor() as usize).max(1);
    let k_upper = ((mu + z * sigma).ceil() as usize + 1).min(n_walsh);

    (k_lower, k_upper)
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
    let shift = mu.unwrap_or(0.0);
    let diffs: Vec<f64> = x
        .iter()
        .zip(y)
        .map(|(a, b)| a - b - shift)
        .filter(|&d| d != 0.0)
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
