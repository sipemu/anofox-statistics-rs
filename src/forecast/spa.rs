use crate::error::Result;
use crate::forecast::spa_common::{
    spa_stationary, validate_finite_data, validate_model_data, validate_spa_parameters,
};

/// Result of the Superior Predictive Ability (SPA) test
#[derive(Debug, Clone)]
pub struct SPAResult {
    /// The SPA test statistic `T_SPA = max(0, max_k sqrt(n) d̄_k / ω̂_k)`
    /// (Hansen, 2005)
    pub statistic: f64,
    /// Hansen's (2005) lower p-value (re-centring `max(d̄_k, 0)`; liberal)
    pub p_value_lower: f64,
    /// Hansen's (2005) consistent p-value (re-centring only models that are
    /// not significantly worse than the benchmark)
    pub p_value_consistent: f64,
    /// Hansen's (2005) upper p-value (re-centring every model at the null
    /// boundary, i.e. White's Reality Check; conservative)
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
/// With `d_{k,t} = L_benchmark,t - L_k,t`, the statistic is
/// `T_SPA = max(0, max_k sqrt(n) d̄_k / ω̂_k)` with Hansen's
/// stationary-bootstrap long-run variance `ω̂_k²`. Each bootstrap
/// replication draws one set of stationary-bootstrap time indices and applies
/// it to all K series jointly (preserving their dependence); the lower,
/// consistent and upper p-values use Hansen's three re-centrings
/// (`p_l <= p_c <= p_u`, `p = #{T* > T_SPA} / B`). Validated against a
/// base-R implementation of Hansen (2005).
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

    // Validate inputs
    validate_model_data(t, model_losses, "competing model")?;
    validate_spa_parameters(t, n_bootstrap, block_length)?;
    validate_finite_data(benchmark_losses, model_losses, "competing model")?;

    // Compute loss differentials: d_ki = L_benchmark - L_model_k
    // Positive values mean the model outperforms the benchmark
    let d: Vec<Vec<f64>> = compute_loss_differentials(benchmark_losses, model_losses);

    let out = spa_stationary(&d, n_bootstrap, block_length, seed);
    Ok(SPAResult {
        statistic: out.statistic,
        p_value_lower: out.p_value_lower,
        p_value_consistent: out.p_value_consistent,
        p_value_upper: out.p_value_upper,
        n_bootstrap,
        best_model_idx: out.best_model_idx,
    })
}

/// Compute loss differentials between benchmark and each model.
fn compute_loss_differentials(benchmark: &[f64], models: &[Vec<f64>]) -> Vec<Vec<f64>> {
    models
        .iter()
        .map(|model_loss| {
            benchmark
                .iter()
                .zip(model_loss.iter())
                .map(|(b, m)| b - m)
                .collect()
        })
        .collect()
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
