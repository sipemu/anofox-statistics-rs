use crate::error::{Result, StatError};
use crate::nonparametric::ranks::rank_with_ties;
use statrs::distribution::{ChiSquared, ContinuousCDF};

/// Result of Kruskal-Wallis test
#[derive(Debug, Clone)]
pub struct KruskalResult {
    /// The H statistic (chi-squared approximation)
    pub statistic: f64,
    /// Degrees of freedom (k-1)
    pub df: f64,
    /// The p-value
    pub p_value: f64,
}

/// Perform Kruskal-Wallis H test for comparing multiple independent groups.
///
/// This is the nonparametric equivalent of one-way ANOVA.
///
/// # Arguments
/// * `groups` - Slice of slices, each containing one group's data
///
/// # Returns
/// * `KruskalResult` containing H statistic, df, and p-value
pub fn kruskal_wallis(groups: &[&[f64]]) -> Result<KruskalResult> {
    let k = groups.len();

    if k < 2 {
        return Err(StatError::InvalidParameter(
            "Kruskal-Wallis test requires at least 2 groups".to_string(),
        ));
    }

    // Check for empty groups and compute sizes
    let mut group_sizes: Vec<usize> = Vec::with_capacity(k);
    let mut n_total = 0usize;

    for (i, group) in groups.iter().enumerate() {
        if group.is_empty() {
            return Err(StatError::InvalidParameter(format!(
                "Group {} is empty",
                i + 1
            )));
        }
        group_sizes.push(group.len());
        n_total += group.len();
    }

    // Combine all data and rank
    let mut combined: Vec<f64> = Vec::with_capacity(n_total);
    for group in groups {
        combined.extend_from_slice(group);
    }

    let (ranks, tie_sizes) = rank_with_ties(&combined)?;

    // Compute sum of ranks for each group
    let mut rank_sums: Vec<f64> = Vec::with_capacity(k);
    let mut idx = 0;
    for &size in &group_sizes {
        let sum: f64 = ranks[idx..idx + size].iter().sum();
        rank_sums.push(sum);
        idx += size;
    }

    let n_f = n_total as f64;

    // H statistic (uncorrected)
    // H = (12 / (N(N+1))) * sum(R_i^2 / n_i) - 3(N+1)
    let sum_term: f64 = rank_sums
        .iter()
        .zip(group_sizes.iter())
        .map(|(&r_sum, &n_i)| r_sum * r_sum / n_i as f64)
        .sum();

    let h_uncorrected = (12.0 / (n_f * (n_f + 1.0))) * sum_term - 3.0 * (n_f + 1.0);

    // Tie correction factor
    // C = 1 - sum(t^3 - t) / (N^3 - N)
    let tie_correction: f64 = tie_sizes
        .iter()
        .map(|&t| {
            let t = t as f64;
            t * t * t - t
        })
        .sum();

    let c = 1.0 - tie_correction / (n_f * n_f * n_f - n_f);

    // Corrected H statistic
    let h = h_uncorrected / c;

    // Degrees of freedom
    let df = (k - 1) as f64;

    // p-value from chi-squared distribution
    let chi_sq = ChiSquared::new(df).unwrap();
    let p_value = 1.0 - chi_sq.cdf(h);

    Ok(KruskalResult {
        statistic: h,
        df,
        p_value,
    })
}
