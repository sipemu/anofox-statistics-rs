//! Non-parametric TOST equivalence tests using Wilcoxon procedures.
//!
//! These tests are robust alternatives to t-test based TOST when normality
//! assumptions may not hold.

use crate::equivalence::{EquivalenceBounds, OneSidedTestResult, TostResult};
use crate::error::{Result, StatError};
use crate::nonparametric::ranks::rank_with_ties;
use crate::utils::finite::ensure_finite;
use crate::utils::select::{kth_pairwise_difference, kth_walsh_average, median_by};
use statrs::distribution::{ContinuousCDF, Normal};

/// Perform TOST for paired samples using Wilcoxon signed-rank test.
///
/// Tests whether the median difference is practically equivalent to zero
/// using the Hodges-Lehmann estimator and shifted Wilcoxon tests.
///
/// # Arguments
/// * `x` - First sample
/// * `y` - Second sample (must be same length as x)
/// * `bounds` - Equivalence bounds for the median difference (Raw or Symmetric only)
/// * `alpha` - Significance level (default: 0.05)
///
/// # Returns
/// * `TostResult` containing test statistics, p-values, and equivalence conclusion
///
/// # Example
/// ```
/// use anofox_statistics::equivalence::{tost_wilcoxon_paired, EquivalenceBounds};
///
/// let before = vec![10.0, 12.0, 11.0, 13.0, 10.5, 11.5, 12.5, 10.0];
/// let after = vec![10.1, 11.9, 11.1, 13.1, 10.4, 11.6, 12.4, 10.1];
/// let bounds = EquivalenceBounds::Symmetric { delta: 0.5 };
///
/// let result = tost_wilcoxon_paired(&before, &after, &bounds, 0.05).unwrap();
/// println!("Median difference equivalent to zero: {}", result.equivalent);
/// ```
///
/// # R equivalent
/// `TOSTER::wilcox_TOST(x, y, paired = TRUE, low_eqbound, high_eqbound)`
///
/// # Complexity
/// O(n log^2 n) expected time and O(n) memory: the Hodges-Lehmann estimate
/// and CI select order statistics of the n(n+1)/2 Walsh averages without
/// materialising them.
pub fn tost_wilcoxon_paired(
    x: &[f64],
    y: &[f64],
    bounds: &EquivalenceBounds,
    alpha: f64,
) -> Result<TostResult> {
    validate_inputs_paired(x, y, alpha)?;

    let n = x.len();

    // Compute differences
    let diffs: Vec<f64> = x.iter().zip(y.iter()).map(|(xi, yi)| xi - yi).collect();

    // Get bounds
    let (lower_bound, upper_bound) = bounds_for_nonparametric(bounds)?;

    // Compute Hodges-Lehmann estimator (pseudo-median of differences)
    let estimate = hodges_lehmann_one_sample(&diffs);

    // Perform shifted Wilcoxon tests for TOST
    // Lower test: shift diffs by -lower_bound, test if median > 0
    let diffs_lower: Vec<f64> = diffs.iter().map(|d| d - lower_bound).collect();
    let (stat_lower, p_lower) = wilcoxon_signed_rank_test(&diffs_lower, true)?;

    // Upper test: shift diffs by -upper_bound, test if median < 0
    let diffs_upper: Vec<f64> = diffs.iter().map(|d| d - upper_bound).collect();
    let (stat_upper, p_upper) = wilcoxon_signed_rank_test(&diffs_upper, false)?;

    // TOST p-value
    let tost_p = p_lower.max(p_upper);

    // Confidence interval using Walsh averages (approximate)
    let ci = hodges_lehmann_ci(&diffs, alpha)?;

    // Equivalence established if CI within bounds
    let equivalent = ci.0 >= lower_bound && ci.1 <= upper_bound;

    Ok(TostResult {
        estimate,
        ci,
        bounds: (lower_bound, upper_bound),
        lower_test: OneSidedTestResult {
            hypothesis: format!("H0: median diff <= {:.4}", lower_bound),
            statistic: stat_lower,
            p_value: p_lower,
            rejected: p_lower < alpha,
        },
        upper_test: OneSidedTestResult {
            hypothesis: format!("H0: median diff >= {:.4}", upper_bound),
            statistic: stat_upper,
            p_value: p_upper,
            rejected: p_upper < alpha,
        },
        tost_p_value: tost_p,
        equivalent,
        alpha,
        n,
        df: None,
        method: "Wilcoxon signed-rank TOST".to_string(),
    })
}

/// Perform TOST for two independent samples using Wilcoxon rank-sum test.
///
/// Tests whether the location shift between groups is practically equivalent
/// to zero using the Hodges-Lehmann estimator and shifted Mann-Whitney tests.
///
/// # Arguments
/// * `x` - First sample
/// * `y` - Second sample
/// * `bounds` - Equivalence bounds for the location shift (Raw or Symmetric only)
/// * `alpha` - Significance level (default: 0.05)
///
/// # Returns
/// * `TostResult` containing test statistics, p-values, and equivalence conclusion
///
/// # Example
/// ```
/// use anofox_statistics::equivalence::{tost_wilcoxon_two_sample, EquivalenceBounds};
///
/// let group1 = vec![10.0, 12.0, 11.0, 13.0, 10.5, 11.5, 12.5, 10.0];
/// let group2 = vec![10.2, 11.8, 11.1, 13.1, 10.4, 11.6, 12.4, 10.1];
/// let bounds = EquivalenceBounds::Symmetric { delta: 0.5 };
///
/// let result = tost_wilcoxon_two_sample(&group1, &group2, &bounds, 0.05).unwrap();
/// println!("Location shift equivalent to zero: {}", result.equivalent);
/// ```
///
/// # R equivalent
/// `TOSTER::wilcox_TOST(x, y, paired = FALSE, low_eqbound, high_eqbound)`
///
/// # Complexity
/// O(N log^2 N) expected time and O(N) memory for `N = nx + ny`: the
/// Hodges-Lehmann estimate and CI select order statistics of the nx*ny
/// pairwise differences without materialising them.
pub fn tost_wilcoxon_two_sample(
    x: &[f64],
    y: &[f64],
    bounds: &EquivalenceBounds,
    alpha: f64,
) -> Result<TostResult> {
    validate_inputs_two_sample(x, y, alpha)?;

    let nx = x.len();
    let ny = y.len();

    // Get bounds
    let (lower_bound, upper_bound) = bounds_for_nonparametric(bounds)?;

    // Compute Hodges-Lehmann estimator (median of pairwise differences)
    let estimate = hodges_lehmann_two_sample(x, y);

    // Perform shifted Mann-Whitney tests for TOST
    // Lower test: shift y by lower_bound, test if x > y_shifted
    let y_lower: Vec<f64> = y.iter().map(|yi| yi + lower_bound).collect();
    let (stat_lower, p_lower) = mann_whitney_test(x, &y_lower, true)?;

    // Upper test: shift y by upper_bound, test if x < y_shifted
    let y_upper: Vec<f64> = y.iter().map(|yi| yi + upper_bound).collect();
    let (stat_upper, p_upper) = mann_whitney_test(x, &y_upper, false)?;

    // TOST p-value
    let tost_p = p_lower.max(p_upper);

    // Confidence interval using Hodges-Lehmann approach
    let ci = hodges_lehmann_ci_two_sample(x, y, alpha)?;

    // Equivalence established if CI within bounds
    let equivalent = ci.0 >= lower_bound && ci.1 <= upper_bound;

    Ok(TostResult {
        estimate,
        ci,
        bounds: (lower_bound, upper_bound),
        lower_test: OneSidedTestResult {
            hypothesis: format!("H0: location shift <= {:.4}", lower_bound),
            statistic: stat_lower,
            p_value: p_lower,
            rejected: p_lower < alpha,
        },
        upper_test: OneSidedTestResult {
            hypothesis: format!("H0: location shift >= {:.4}", upper_bound),
            statistic: stat_upper,
            p_value: p_upper,
            rejected: p_upper < alpha,
        },
        tost_p_value: tost_p,
        equivalent,
        alpha,
        n: nx + ny,
        df: None,
        method: "Wilcoxon rank-sum TOST".to_string(),
    })
}

/// Ascending sorted copy.
fn sorted_asc(data: &[f64]) -> Vec<f64> {
    let mut v = data.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    v
}

/// Descending sorted copy.
fn sorted_desc(data: &[f64]) -> Vec<f64> {
    let mut v = data.to_vec();
    v.sort_by(|a, b| b.total_cmp(a));
    v
}

/// Compute Hodges-Lehmann estimator for one sample (pseudo-median).
/// This is the median of all Walsh averages (xi + xj) / 2 for i <= j.
///
/// The Walsh averages are not materialised: order statistics are selected
/// in a sorted-rows matrix. O(n log^2 n) expected time, O(n) memory.
fn hodges_lehmann_one_sample(data: &[f64]) -> f64 {
    let n = data.len();
    let s = sorted_asc(data);
    median_by(n * (n + 1) / 2, |k| kth_walsh_average(&s, k))
}

/// Compute Hodges-Lehmann estimator for two samples.
/// This is the median of all pairwise differences (xi - yj).
///
/// O((nx + ny) log^2) expected time, O(nx + ny) memory (no nx*ny buffer).
fn hodges_lehmann_two_sample(x: &[f64], y: &[f64]) -> f64 {
    let yd = sorted_desc(y);
    median_by(x.len() * y.len(), |k| kth_pairwise_difference(x, &yd, k))
}

/// Compute confidence interval for Hodges-Lehmann estimator (one sample).
///
/// O(n log^2 n) expected time, O(n) memory.
fn hodges_lehmann_ci(data: &[f64], alpha: f64) -> Result<(f64, f64)> {
    let n = data.len();
    let s = sorted_asc(data);
    let n_walsh = n * (n + 1) / 2;

    // Use normal approximation for CI indices
    let n_f = n as f64;
    let mu = n_f * (n_f + 1.0) / 4.0;
    let sigma = (n_f * (n_f + 1.0) * (2.0 * n_f + 1.0) / 24.0).sqrt();

    let normal = Normal::new(0.0, 1.0)
        .map_err(|e| StatError::InvalidParameter(format!("Failed to create normal: {}", e)))?;

    // For (1-2*alpha) CI
    let z = normal.inverse_cdf(1.0 - alpha);

    let k_lower = ((mu - z * sigma).floor() as usize).min(n_walsh - 1);
    let k_upper = ((mu + z * sigma).ceil() as usize).min(n_walsh - 1);

    Ok((
        kth_walsh_average(&s, k_lower),
        kth_walsh_average(&s, k_upper),
    ))
}

/// Compute confidence interval for Hodges-Lehmann estimator (two samples).
///
/// O((nx + ny) log^2) expected time, O(nx + ny) memory.
fn hodges_lehmann_ci_two_sample(x: &[f64], y: &[f64], alpha: f64) -> Result<(f64, f64)> {
    let nx = x.len();
    let ny = y.len();
    let yd = sorted_desc(y);

    let n_diffs = nx * ny;
    let nx_f = nx as f64;
    let ny_f = ny as f64;

    // Normal approximation for Mann-Whitney CI indices
    let mu = nx_f * ny_f / 2.0;
    let sigma = (nx_f * ny_f * (nx_f + ny_f + 1.0) / 12.0).sqrt();

    let normal = Normal::new(0.0, 1.0)
        .map_err(|e| StatError::InvalidParameter(format!("Failed to create normal: {}", e)))?;

    // For (1-2*alpha) CI
    let z = normal.inverse_cdf(1.0 - alpha);

    let k_lower = ((mu - z * sigma).floor() as usize).min(n_diffs - 1);
    let k_upper = ((mu + z * sigma).ceil() as usize).min(n_diffs - 1);

    Ok((
        kth_pairwise_difference(x, &yd, k_lower),
        kth_pairwise_difference(x, &yd, k_upper),
    ))
}

/// Perform Wilcoxon signed-rank test (one-sided).
/// Returns (V statistic, p-value).
/// If `greater` is true, tests H_a: median > 0; otherwise H_a: median < 0.
fn wilcoxon_signed_rank_test(diffs: &[f64], greater: bool) -> Result<(f64, f64)> {
    // Filter out zeros
    let nonzero: Vec<f64> = diffs.iter().copied().filter(|&d| d != 0.0).collect();
    let n = nonzero.len();

    if n == 0 {
        return Ok((0.0, 1.0));
    }

    // Compute absolute values and signs
    let abs_vals: Vec<f64> = nonzero.iter().map(|d| d.abs()).collect();
    let (ranks, tie_sizes) = rank_with_ties(&abs_vals)?;

    // V = sum of ranks where difference is positive
    let v: f64 = nonzero
        .iter()
        .zip(ranks.iter())
        .filter(|(&d, _)| d > 0.0)
        .map(|(_, &r)| r)
        .sum();

    // Normal approximation
    let n_f = n as f64;
    let mu = n_f * (n_f + 1.0) / 4.0;

    // Tie correction
    let tc: f64 = tie_sizes
        .iter()
        .map(|&t| {
            let t = t as f64;
            t * t * t - t
        })
        .sum();
    let sigma_sq = n_f * (n_f + 1.0) * (2.0 * n_f + 1.0) / 24.0 - tc / 48.0;
    let sigma = sigma_sq.sqrt();

    // z-score with continuity correction
    let z = if greater {
        (v - mu - 0.5) / sigma
    } else {
        (v - mu + 0.5) / sigma
    };

    let normal = Normal::new(0.0, 1.0)
        .map_err(|e| StatError::InvalidParameter(format!("Failed to create normal: {}", e)))?;

    let p = if greater { normal.sf(z) } else { normal.cdf(z) };

    Ok((v, p))
}

/// Perform Mann-Whitney U test (one-sided).
/// Returns (U statistic, p-value).
/// If `greater` is true, tests H_a: x > y; otherwise H_a: x < y.
fn mann_whitney_test(x: &[f64], y: &[f64], greater: bool) -> Result<(f64, f64)> {
    let nx = x.len();
    let ny = y.len();
    let n = nx + ny;

    // Combine and rank
    let mut combined: Vec<f64> = Vec::with_capacity(n);
    combined.extend_from_slice(x);
    combined.extend_from_slice(y);

    let (ranks, tie_sizes) = rank_with_ties(&combined)?;

    // Sum of ranks for x
    let r1: f64 = ranks[..nx].iter().sum();

    // U statistic for x
    let u1 = r1 - (nx * (nx + 1)) as f64 / 2.0;

    // Normal approximation
    let nx_f = nx as f64;
    let ny_f = ny as f64;
    let n_f = n as f64;

    let mu = nx_f * ny_f / 2.0;

    // Tie correction
    let tc: f64 = tie_sizes
        .iter()
        .map(|&t| {
            let t = t as f64;
            t * t * t - t
        })
        .sum();
    let sigma_sq = (nx_f * ny_f / 12.0) * ((n_f + 1.0) - tc / (n_f * (n_f - 1.0)));
    let sigma = sigma_sq.sqrt();

    // z-score with continuity correction
    let z = if greater {
        (u1 - mu - 0.5) / sigma
    } else {
        (u1 - mu + 0.5) / sigma
    };

    let normal = Normal::new(0.0, 1.0)
        .map_err(|e| StatError::InvalidParameter(format!("Failed to create normal: {}", e)))?;

    let p = if greater { normal.sf(z) } else { normal.cdf(z) };

    Ok((u1, p))
}

/// Convert bounds for non-parametric tests.
fn bounds_for_nonparametric(bounds: &EquivalenceBounds) -> Result<(f64, f64)> {
    match bounds {
        EquivalenceBounds::Raw { lower, upper } => Ok((*lower, *upper)),
        EquivalenceBounds::Symmetric { delta } => Ok((-*delta, *delta)),
        EquivalenceBounds::CohenD { .. } => Err(StatError::InvalidParameter(
            "Cohen's d bounds not applicable for Wilcoxon TOST; use Raw or Symmetric bounds"
                .to_string(),
        )),
    }
}

/// Validate inputs for paired Wilcoxon test.
fn validate_inputs_paired(x: &[f64], y: &[f64], alpha: f64) -> Result<()> {
    if x.is_empty() || y.is_empty() {
        return Err(StatError::EmptyData);
    }

    if x.len() != y.len() {
        return Err(StatError::InvalidParameter(format!(
            "Paired samples must have equal length: {} vs {}",
            x.len(),
            y.len()
        )));
    }

    if x.len() < 2 {
        return Err(StatError::InsufficientData {
            needed: 2,
            got: x.len(),
        });
    }

    if !(0.0 < alpha && alpha < 1.0) {
        return Err(StatError::InvalidParameter(format!(
            "alpha must be between 0 and 1, got {}",
            alpha
        )));
    }

    // Equivalence bounds are in data units: ±Inf makes the shifted tests and
    // the Hodges-Lehmann interval meaningless.
    ensure_finite("x", x)?;
    ensure_finite("y", y)?;

    Ok(())
}

/// Validate inputs for two-sample Wilcoxon test.
fn validate_inputs_two_sample(x: &[f64], y: &[f64], alpha: f64) -> Result<()> {
    if x.is_empty() || y.is_empty() {
        return Err(StatError::EmptyData);
    }

    if x.len() < 2 {
        return Err(StatError::InsufficientData {
            needed: 2,
            got: x.len(),
        });
    }

    if y.len() < 2 {
        return Err(StatError::InsufficientData {
            needed: 2,
            got: y.len(),
        });
    }

    if !(0.0 < alpha && alpha < 1.0) {
        return Err(StatError::InvalidParameter(format!(
            "alpha must be between 0 and 1, got {}",
            alpha
        )));
    }

    // Equivalence bounds are in data units: ±Inf makes the shifted tests and
    // the Hodges-Lehmann interval meaningless.
    ensure_finite("x", x)?;
    ensure_finite("y", y)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sorted(mut v: Vec<f64>) -> Vec<f64> {
        v.sort_by(|a, b| a.total_cmp(b));
        v
    }

    fn naive_median(v: &[f64]) -> f64 {
        let n = v.len();
        if n % 2 == 0 {
            (v[n / 2 - 1] + v[n / 2]) / 2.0
        } else {
            v[n / 2]
        }
    }

    fn naive_walsh(d: &[f64]) -> Vec<f64> {
        let mut w = Vec::new();
        for i in 0..d.len() {
            for j in i..d.len() {
                w.push((d[i] + d[j]) / 2.0);
            }
        }
        sorted(w)
    }

    fn naive_diffs(x: &[f64], y: &[f64]) -> Vec<f64> {
        sorted(
            x.iter()
                .flat_map(|a| y.iter().map(move |b| a - b))
                .collect(),
        )
    }

    fn naive_ci(sorted_vals: &[f64], mu: f64, sigma: f64, alpha: f64) -> (f64, f64) {
        let z = Normal::new(0.0, 1.0).unwrap().inverse_cdf(1.0 - alpha);
        let m = sorted_vals.len() - 1;
        let kl = ((mu - z * sigma).floor() as usize).min(m);
        let ku = ((mu + z * sigma).ceil() as usize).min(m);
        (sorted_vals[kl], sorted_vals[ku])
    }

    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 11) as f64 / (1u64 << 53) as f64
    }

    #[test]
    fn test_hodges_lehmann_matches_naive() {
        let mut s = 31u64;
        for &n in &[3usize, 8, 31, 120] {
            for &levels in &[0usize, 4] {
                let mut draw = || {
                    let u = lcg(&mut s);
                    if levels == 0 {
                        u * 3.0
                    } else {
                        (u * levels as f64).floor() * 0.25
                    }
                };
                let x: Vec<f64> = (0..n).map(|_| draw()).collect();
                let y: Vec<f64> = (0..n + 3).map(|_| draw()).collect();

                let w = naive_walsh(&x);
                assert_eq!(hodges_lehmann_one_sample(&x), naive_median(&w));
                let nf = n as f64;
                let mu = nf * (nf + 1.0) / 4.0;
                let sd = (nf * (nf + 1.0) * (2.0 * nf + 1.0) / 24.0).sqrt();
                assert_eq!(
                    hodges_lehmann_ci(&x, 0.05).unwrap(),
                    naive_ci(&w, mu, sd, 0.05)
                );

                let d = naive_diffs(&x, &y);
                assert_eq!(hodges_lehmann_two_sample(&x, &y), naive_median(&d));
                let (a, b) = (n as f64, (n + 3) as f64);
                let sd2 = (a * b * (a + b + 1.0) / 12.0).sqrt();
                assert_eq!(
                    hodges_lehmann_ci_two_sample(&x, &y, 0.05).unwrap(),
                    naive_ci(&d, a * b / 2.0, sd2, 0.05)
                );
            }
        }
    }

    /// n = 100_000 per group (5e9 Walsh averages / 1e10 differences, 40-80 GB
    /// when materialised).
    #[test]
    #[ignore]
    fn test_wilcoxon_tost_large_n() {
        let mut s = 77u64;
        let n = 100_000;
        let x: Vec<f64> = (0..n).map(|_| lcg(&mut s) * 10.0).collect();
        let y: Vec<f64> = (0..n).map(|_| lcg(&mut s) * 10.0 + 0.01).collect();
        let bounds = EquivalenceBounds::Symmetric { delta: 0.5 };
        let p = tost_wilcoxon_paired(&x, &y, &bounds, 0.05).unwrap();
        assert!(p.ci.0 <= p.estimate && p.estimate <= p.ci.1);
        let t = tost_wilcoxon_two_sample(&x, &y, &bounds, 0.05).unwrap();
        assert!(t.ci.0 <= t.estimate && t.estimate <= t.ci.1);
        assert!(t.equivalent);
    }

    #[test]
    fn test_wilcoxon_paired_equivalent() {
        // Small differences
        let before = vec![10.0, 12.0, 11.0, 13.0, 10.5, 11.5, 12.5, 10.0];
        let after = vec![10.1, 11.9, 11.05, 13.05, 10.45, 11.55, 12.45, 10.05];
        let bounds = EquivalenceBounds::Symmetric { delta: 0.5 };

        let result = tost_wilcoxon_paired(&before, &after, &bounds, 0.05).unwrap();

        // Differences are tiny, should likely be equivalent
        assert!(result.estimate.abs() < 0.2);
    }

    #[test]
    fn test_wilcoxon_paired_not_equivalent() {
        // Large differences
        let before = vec![10.0, 12.0, 11.0, 13.0, 10.5];
        let after = vec![12.0, 14.0, 13.0, 15.0, 12.5];
        let bounds = EquivalenceBounds::Symmetric { delta: 0.5 };

        let result = tost_wilcoxon_paired(&before, &after, &bounds, 0.05).unwrap();

        // Differences ~2, should NOT be equivalent within ±0.5
        assert!(!result.equivalent);
    }

    #[test]
    fn test_wilcoxon_two_sample_equivalent() {
        // Similar groups
        let x = vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0];
        let y = vec![10.1, 10.9, 12.1, 12.9, 14.1, 14.9, 16.1, 16.9];
        let bounds = EquivalenceBounds::Symmetric { delta: 0.5 };

        let result = tost_wilcoxon_two_sample(&x, &y, &bounds, 0.05).unwrap();

        // Groups are very similar
        assert!(result.estimate.abs() < 0.3);
    }

    #[test]
    fn test_wilcoxon_two_sample_not_equivalent() {
        // Different groups
        let x = vec![10.0, 11.0, 12.0, 13.0, 14.0];
        let y = vec![15.0, 16.0, 17.0, 18.0, 19.0];
        let bounds = EquivalenceBounds::Symmetric { delta: 0.5 };

        let result = tost_wilcoxon_two_sample(&x, &y, &bounds, 0.05).unwrap();

        // Clear difference
        assert!(!result.equivalent);
    }

    #[test]
    fn test_hodges_lehmann_one_sample() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let hl = hodges_lehmann_one_sample(&data);
        // Should be close to 3 (the median)
        assert!((hl - 3.0).abs() < 0.1);
    }

    #[test]
    fn test_hodges_lehmann_two_sample() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0, 3.0];
        let hl = hodges_lehmann_two_sample(&x, &y);
        // Same groups, location shift should be ~0
        assert!(hl.abs() < 0.1);
    }
}
