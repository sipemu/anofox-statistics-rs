use crate::error::{Result, StatError};
use crate::nonparametric::ranks::rank_with_ties;
use statrs::distribution::{ContinuousCDF, Normal};

/// Result of Mann-Whitney U test
#[derive(Debug, Clone)]
pub struct MannWhitneyResult {
    /// The U statistic (for first sample, matching R's wilcox.test)
    pub statistic: f64,
    /// The p-value (using normal approximation)
    pub p_value: f64,
}

/// Result of Wilcoxon Signed-Rank test
#[derive(Debug, Clone)]
pub struct WilcoxonResult {
    /// The V statistic (sum of positive ranks)
    pub statistic: f64,
    /// The p-value (using normal approximation)
    pub p_value: f64,
}

/// Perform Mann-Whitney U test (Wilcoxon rank-sum test) for two independent samples.
///
/// Uses normal approximation for p-value calculation (no continuity correction).
///
/// # Arguments
/// * `x` - First sample
/// * `y` - Second sample
///
/// # Returns
/// * `MannWhitneyResult` containing U statistic and p-value
pub fn mann_whitney_u(x: &[f64], y: &[f64]) -> Result<MannWhitneyResult> {
    if x.is_empty() {
        return Err(StatError::EmptyData);
    }
    if y.is_empty() {
        return Err(StatError::EmptyData);
    }

    let nx = x.len();
    let ny = y.len();
    let n = nx + ny;

    // Combine samples and rank
    let mut combined: Vec<f64> = Vec::with_capacity(n);
    combined.extend_from_slice(x);
    combined.extend_from_slice(y);

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

    // Variance with tie correction
    // Var(U) = (n1*n2/12) * (n + 1 - sum(t^3 - t)/(n*(n-1)))
    let tie_correction: f64 = tie_sizes
        .iter()
        .map(|&t| {
            let t = t as f64;
            t * t * t - t
        })
        .sum();

    let sigma_sq = (nx_f * ny_f / 12.0) * ((n_f + 1.0) - tie_correction / (n_f * (n_f - 1.0)));
    let sigma = sigma_sq.sqrt();

    // Z-score (no continuity correction)
    let z = (u1 - mu) / sigma;

    // Two-sided p-value
    let normal = Normal::new(0.0, 1.0).unwrap();
    let p_value = 2.0 * (1.0 - normal.cdf(z.abs()));

    Ok(MannWhitneyResult {
        statistic: u1,
        p_value,
    })
}

/// Perform Wilcoxon Signed-Rank test for paired samples.
///
/// Uses normal approximation for p-value calculation (no continuity correction).
///
/// # Arguments
/// * `x` - First sample
/// * `y` - Second sample (must be same length as x)
///
/// # Returns
/// * `WilcoxonResult` containing V statistic and p-value
pub fn wilcoxon_signed_rank(x: &[f64], y: &[f64]) -> Result<WilcoxonResult> {
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

    // Compute differences and filter out zeros
    let diffs: Vec<f64> = x
        .iter()
        .zip(y.iter())
        .map(|(xi, yi)| xi - yi)
        .filter(|&d| d != 0.0)
        .collect();

    let n_nonzero = diffs.len();

    if n_nonzero == 0 {
        // All differences are zero - return 0 statistic with p-value 1
        return Ok(WilcoxonResult {
            statistic: 0.0,
            p_value: 1.0,
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

    // Variance with tie correction
    // Var(V) = n(n+1)(2n+1)/24 - sum(t^3 - t)/48
    let tie_correction: f64 = tie_sizes
        .iter()
        .map(|&t| {
            let t = t as f64;
            t * t * t - t
        })
        .sum();

    let sigma_sq = n_f * (n_f + 1.0) * (2.0 * n_f + 1.0) / 24.0 - tie_correction / 48.0;
    let sigma = sigma_sq.sqrt();

    // Z-score (no continuity correction)
    let z = (v - mu) / sigma;

    // Two-sided p-value
    let normal = Normal::new(0.0, 1.0).unwrap();
    let p_value = 2.0 * (1.0 - normal.cdf(z.abs()));

    Ok(WilcoxonResult {
        statistic: v,
        p_value,
    })
}
