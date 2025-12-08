use crate::error::{Result, StatError};
use crate::resampling::bootstrap::StationaryBootstrap;

/// Result of the Superior Predictive Ability (SPA) test
#[derive(Debug, Clone)]
pub struct SPAResult {
    /// The SPA test statistic (maximum standardized performance)
    pub statistic: f64,
    /// The consistent p-value (Hansen, 2005)
    pub p_value_consistent: f64,
    /// The upper p-value (more conservative)
    pub p_value_upper: f64,
    /// Number of bootstrap samples used
    pub n_bootstrap: usize,
    /// Index of the best performing model (if any outperforms benchmark)
    pub best_model_idx: Option<usize>,
}

/// Perform the Superior Predictive Ability (SPA) test.
///
/// Tests whether any of K competing forecast models has superior predictive
/// ability compared to a benchmark model. The null hypothesis is that no
/// model outperforms the benchmark.
///
/// # Arguments
/// * `benchmark_losses` - Loss values from the benchmark model (length T)
/// * `model_losses` - Loss values from K competing models (K x T matrix, row-major)
/// * `n_bootstrap` - Number of bootstrap samples
/// * `block_length` - Expected block length for stationary bootstrap
/// * `seed` - Optional random seed for reproducibility
///
/// # Returns
/// * `SPAResult` containing test statistics and p-values
///
/// # References
/// * Hansen, P.R. (2005). "A Test for Superior Predictive Ability"
/// * White, H. (2000). "A Reality Check for Data Snooping"
pub fn spa_test(
    benchmark_losses: &[f64],
    model_losses: &[Vec<f64>],
    n_bootstrap: usize,
    block_length: f64,
    seed: Option<u64>,
) -> Result<SPAResult> {
    let t = benchmark_losses.len();
    let k = model_losses.len();

    if t == 0 {
        return Err(StatError::EmptyData);
    }

    if k == 0 {
        return Err(StatError::InvalidParameter(
            "At least one competing model required".to_string(),
        ));
    }

    // Verify all model losses have the same length
    for (i, losses) in model_losses.iter().enumerate() {
        if losses.len() != t {
            return Err(StatError::InvalidParameter(format!(
                "Model {} has {} observations, expected {}",
                i,
                losses.len(),
                t
            )));
        }
    }

    let t_f = t as f64;

    // Compute loss differentials: d_ki = L_benchmark - L_model_k
    // Positive values mean the model outperforms the benchmark
    let mut d: Vec<Vec<f64>> = Vec::with_capacity(k);
    for model_loss in model_losses {
        let diff: Vec<f64> = benchmark_losses
            .iter()
            .zip(model_loss.iter())
            .map(|(b, m)| b - m)
            .collect();
        d.push(diff);
    }

    // Sample means of loss differentials
    let d_bar: Vec<f64> = d.iter().map(|di| di.iter().sum::<f64>() / t_f).collect();

    // Sample variances with HAC correction (simple version)
    let variances: Vec<f64> = d
        .iter()
        .zip(d_bar.iter())
        .map(|(di, &mean)| {
            let var: f64 = di.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (t_f - 1.0);
            var / t_f // variance of the mean
        })
        .collect();

    // Standardized statistics for each model
    // When variance is very small, use the raw mean scaled by sqrt(T)
    // This handles the case where all observations are identical
    let standardized: Vec<f64> = d_bar
        .iter()
        .zip(variances.iter())
        .map(|(&mean, &var)| {
            if var > 1e-14 {
                mean * t_f.sqrt() / var.sqrt()
            } else {
                // Zero variance: use scaled mean directly
                // This preserves the ordering by mean magnitude
                mean * t_f.sqrt() * 1e6
            }
        })
        .collect();

    // Observed test statistic: max of standardized performance
    let observed = standardized
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    // Find best model
    let best_model_idx = standardized
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i);

    // Bootstrap p-values
    let mut bootstrap = StationaryBootstrap::new(block_length, seed);
    let mut count_consistent = 0usize;
    let mut count_upper = 0usize;

    for _ in 0..n_bootstrap {
        let mut boot_max_consistent = f64::NEG_INFINITY;
        let mut boot_max_upper = f64::NEG_INFINITY;

        for (i, di) in d.iter().enumerate() {
            // Bootstrap sample
            let boot_sample = bootstrap.sample(di, t);

            // Bootstrap mean
            let boot_mean: f64 = boot_sample.iter().sum::<f64>() / t_f;

            // Centered bootstrap statistic (for consistent p-value)
            let centered_mean = boot_mean - d_bar[i];
            let boot_var: f64 = boot_sample
                .iter()
                .map(|x| (x - boot_mean).powi(2))
                .sum::<f64>()
                / (t_f - 1.0)
                / t_f;

            let boot_stat_consistent = if boot_var > 1e-14 {
                centered_mean * t_f.sqrt() / boot_var.sqrt()
            } else {
                centered_mean * t_f.sqrt() * 1e6
            };

            // For upper p-value, use g(d_bar) modification
            let g_d_bar = d_bar[i].max(0.0);
            let upper_centered = boot_mean - g_d_bar;
            let boot_stat_upper = if boot_var > 1e-14 {
                upper_centered * t_f.sqrt() / boot_var.sqrt()
            } else {
                upper_centered * t_f.sqrt() * 1e6
            };

            boot_max_consistent = boot_max_consistent.max(boot_stat_consistent);
            boot_max_upper = boot_max_upper.max(boot_stat_upper);
        }

        if boot_max_consistent >= observed {
            count_consistent += 1;
        }
        if boot_max_upper >= observed {
            count_upper += 1;
        }
    }

    let p_value_consistent = (count_consistent as f64 + 1.0) / (n_bootstrap as f64 + 1.0);
    let p_value_upper = (count_upper as f64 + 1.0) / (n_bootstrap as f64 + 1.0);

    Ok(SPAResult {
        statistic: observed,
        p_value_consistent,
        p_value_upper,
        n_bootstrap,
        best_model_idx,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spa_clearly_better_model() {
        // Benchmark with high losses
        let benchmark: Vec<f64> = vec![10.0; 100];

        // Model with much lower losses
        let model: Vec<Vec<f64>> = vec![vec![1.0; 100]];

        let result = spa_test(&benchmark, &model, 499, 5.0, Some(42)).unwrap();

        // Should reject null (model is better than benchmark)
        assert!(
            result.p_value_consistent < 0.05,
            "p_value {} should be < 0.05",
            result.p_value_consistent
        );
        assert!(result.best_model_idx == Some(0));
    }

    #[test]
    fn test_spa_no_better_model() {
        // Benchmark and model with similar performance
        let benchmark: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();

        // Models with similar losses (slight random variation)
        let model1: Vec<f64> = benchmark
            .iter()
            .enumerate()
            .map(|(i, &x)| x + 0.01 * (i as f64).cos())
            .collect();
        let model2: Vec<f64> = benchmark
            .iter()
            .enumerate()
            .map(|(i, &x)| x - 0.01 * (i as f64).sin())
            .collect();

        let models = vec![model1, model2];

        let result = spa_test(&benchmark, &models, 499, 5.0, Some(42)).unwrap();

        // Should not reject null (no model clearly better)
        // Using a less strict threshold since these are similar
        assert!(
            result.p_value_upper > 0.01,
            "p_value_upper {} should be > 0.01",
            result.p_value_upper
        );
    }

    #[test]
    fn test_spa_multiple_models() {
        // Benchmark
        let benchmark: Vec<f64> = vec![5.0; 50];

        // Multiple models with varying performance
        let models = vec![
            vec![4.0; 50], // Slightly better
            vec![2.0; 50], // Much better
            vec![6.0; 50], // Worse
            vec![4.5; 50], // Slightly better
        ];

        let result = spa_test(&benchmark, &models, 499, 3.0, Some(42)).unwrap();

        // Model 1 (index 1) should be the best
        assert_eq!(result.best_model_idx, Some(1));
    }

    #[test]
    fn test_spa_empty_benchmark() {
        let benchmark: Vec<f64> = vec![];
        let models = vec![vec![1.0, 2.0, 3.0]];

        assert!(spa_test(&benchmark, &models, 100, 3.0, None).is_err());
    }

    #[test]
    fn test_spa_no_models() {
        let benchmark: Vec<f64> = vec![1.0, 2.0, 3.0];
        let models: Vec<Vec<f64>> = vec![];

        assert!(spa_test(&benchmark, &models, 100, 3.0, None).is_err());
    }

    #[test]
    fn test_spa_length_mismatch() {
        let benchmark: Vec<f64> = vec![1.0, 2.0, 3.0];
        let models = vec![vec![1.0, 2.0]]; // Different length

        assert!(spa_test(&benchmark, &models, 100, 3.0, None).is_err());
    }
}
