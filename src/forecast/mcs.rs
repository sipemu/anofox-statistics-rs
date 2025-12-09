use crate::error::{Result, StatError};
use crate::resampling::bootstrap::StationaryBootstrap;
use std::collections::HashMap;

/// Test statistic type for Model Confidence Set
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MCSStatistic {
    /// T_R statistic: max_{i,j} |t_{ij}|
    /// Most powerful against models with one clearly inferior model
    Range,
    /// T_max statistic: max_i of average t-statistics across all pairs
    /// More balanced when multiple models may be inferior
    Max,
}

/// Information about a single elimination step
#[derive(Debug, Clone)]
pub struct MCSEliminationStep {
    /// The model index that was tested at this step
    pub model_idx: usize,
    /// The p-value for the equal predictive ability test at this step
    pub p_value: f64,
    /// Whether the model was eliminated (p_value < alpha)
    pub eliminated: bool,
}

/// Result of Model Confidence Set procedure
#[derive(Debug, Clone)]
pub struct MCSResult {
    /// Indices of models in the Model Confidence Set (not eliminated)
    pub included_models: Vec<usize>,
    /// Indices of models eliminated from the MCS
    pub eliminated_models: Vec<usize>,
    /// The MCS p-value (p-value when elimination stopped)
    pub mcs_p_value: f64,
    /// Full elimination sequence with p-values at each step
    pub elimination_sequence: Vec<MCSEliminationStep>,
    /// Number of bootstrap samples used
    pub n_bootstrap: usize,
    /// The statistic type used
    pub statistic_type: MCSStatistic,
}

/// Perform the Model Confidence Set procedure.
///
/// Identifies a set of models that contains the best model with a given
/// confidence level. Uses sequential elimination based on bootstrap tests
/// of equal predictive ability.
///
/// # Arguments
/// * `losses` - Loss values for each model (K models, each with T observations)
/// * `alpha` - Significance level for elimination (e.g., 0.10)
/// * `statistic` - Which test statistic to use (Range or Max)
/// * `n_bootstrap` - Number of bootstrap samples
/// * `block_length` - Expected block length for stationary bootstrap
/// * `seed` - Optional random seed for reproducibility
///
/// # Returns
/// * `MCSResult` containing the model confidence set and elimination details
///
/// # References
/// * Hansen, P.R., Lunde, A., and Nason, J.M. (2011) "The Model Confidence Set"
pub fn model_confidence_set(
    losses: &[Vec<f64>],
    alpha: f64,
    statistic: MCSStatistic,
    n_bootstrap: usize,
    block_length: f64,
    seed: Option<u64>,
) -> Result<MCSResult> {
    let k = losses.len();

    // Input validation - check parameters first
    if k == 0 {
        return Err(StatError::InvalidParameter(
            "At least one model required".to_string(),
        ));
    }

    if alpha <= 0.0 || alpha >= 1.0 {
        return Err(StatError::InvalidParameter(
            "alpha must be in (0, 1)".to_string(),
        ));
    }

    if n_bootstrap == 0 {
        return Err(StatError::InvalidParameter(
            "n_bootstrap must be positive".to_string(),
        ));
    }

    // Single model is trivially in the MCS
    if k == 1 {
        return Ok(MCSResult {
            included_models: vec![0],
            eliminated_models: vec![],
            mcs_p_value: 1.0,
            elimination_sequence: vec![],
            n_bootstrap,
            statistic_type: statistic,
        });
    }

    let t = losses[0].len();
    if t == 0 {
        return Err(StatError::EmptyData);
    }

    // Verify all models have same length
    for (i, model_losses) in losses.iter().enumerate() {
        if model_losses.len() != t {
            return Err(StatError::InvalidParameter(format!(
                "Model {} has {} observations, expected {}",
                i,
                model_losses.len(),
                t
            )));
        }
    }

    // Initialize
    let mut remaining_models: Vec<usize> = (0..k).collect();
    let mut eliminated_models: Vec<usize> = vec![];
    let mut elimination_sequence: Vec<MCSEliminationStep> = vec![];
    let mut bootstrap = StationaryBootstrap::new(block_length, seed);
    let mut mcs_p_value = 1.0;

    // Sequential elimination loop
    loop {
        if remaining_models.len() <= 1 {
            break; // Cannot eliminate the last model
        }

        let m = remaining_models.len();

        // Compute pairwise loss differentials for remaining models
        // d_{ij,t} = L_{i,t} - L_{j,t}
        let mut d_ij: HashMap<(usize, usize), Vec<f64>> = HashMap::new();

        for i in 0..m {
            for j in (i + 1)..m {
                let model_i = remaining_models[i];
                let model_j = remaining_models[j];
                let d: Vec<f64> = losses[model_i]
                    .iter()
                    .zip(losses[model_j].iter())
                    .map(|(li, lj)| li - lj)
                    .collect();
                d_ij.insert((i, j), d);
            }
        }

        // Compute statistics for each pair
        let t_f = t as f64;
        let mut t_stats: HashMap<(usize, usize), f64> = HashMap::new();
        let mut d_bars: HashMap<(usize, usize), f64> = HashMap::new();

        for ((i, j), d) in &d_ij {
            let d_bar: f64 = d.iter().sum::<f64>() / t_f;
            d_bars.insert((*i, *j), d_bar);

            // Variance of the mean (simple estimator)
            let var_d: f64 = d.iter().map(|x| (x - d_bar).powi(2)).sum::<f64>() / (t_f - 1.0) / t_f;

            let t_stat = if var_d > 1e-14 {
                d_bar * t_f.sqrt() / var_d.sqrt()
            } else {
                d_bar * t_f.sqrt() * 1e6
            };

            t_stats.insert((*i, *j), t_stat);
        }

        // Compute observed test statistic
        let observed_stat = compute_test_statistic(statistic, &t_stats, m);

        // Identify worst model (highest average loss relative to others)
        let worst_idx = identify_worst_model(&t_stats, m);

        // Bootstrap to get p-value
        let mut count_exceeds = 0usize;

        for _ in 0..n_bootstrap {
            let mut boot_t_stats: HashMap<(usize, usize), f64> = HashMap::new();

            for ((i, j), d) in &d_ij {
                // Bootstrap sample
                let boot_sample = bootstrap.sample(d, t);

                // Bootstrap mean
                let boot_mean: f64 = boot_sample.iter().sum::<f64>() / t_f;
                let original_mean = d_bars[&(*i, *j)];

                // Centered bootstrap statistic
                let centered = boot_mean - original_mean;
                let boot_var: f64 = boot_sample
                    .iter()
                    .map(|x| (x - boot_mean).powi(2))
                    .sum::<f64>()
                    / (t_f - 1.0)
                    / t_f;

                let boot_t = if boot_var > 1e-14 {
                    centered * t_f.sqrt() / boot_var.sqrt()
                } else {
                    centered * t_f.sqrt() * 1e6
                };

                boot_t_stats.insert((*i, *j), boot_t);
            }

            let boot_stat = compute_test_statistic(statistic, &boot_t_stats, m);

            if boot_stat >= observed_stat {
                count_exceeds += 1;
            }
        }

        let p_value = (count_exceeds as f64 + 1.0) / (n_bootstrap as f64 + 1.0);

        // Record step
        let step = MCSEliminationStep {
            model_idx: remaining_models[worst_idx],
            p_value,
            eliminated: p_value < alpha,
        };
        elimination_sequence.push(step);

        if p_value < alpha {
            // Eliminate worst model and continue
            eliminated_models.push(remaining_models[worst_idx]);
            remaining_models.remove(worst_idx);
        } else {
            // Stop: all remaining models are "equivalent"
            mcs_p_value = p_value;
            break;
        }
    }

    // If we eliminated all but one, the last p-value is the MCS p-value
    if remaining_models.len() == 1 && !elimination_sequence.is_empty() {
        mcs_p_value = elimination_sequence.last().unwrap().p_value;
    }

    Ok(MCSResult {
        included_models: remaining_models,
        eliminated_models,
        mcs_p_value,
        elimination_sequence,
        n_bootstrap,
        statistic_type: statistic,
    })
}

/// Compute the test statistic based on pairwise t-statistics
fn compute_test_statistic(
    stat_type: MCSStatistic,
    t_stats: &HashMap<(usize, usize), f64>,
    m: usize,
) -> f64 {
    match stat_type {
        MCSStatistic::Range => {
            // T_R = max_{i,j} |t_{ij}|
            t_stats.values().map(|t| t.abs()).fold(0.0_f64, f64::max)
        }
        MCSStatistic::Max => {
            // T_max = max_i of average t-statistics
            // For each model i, compute average t-stat vs all other models
            let mut model_avgs: Vec<f64> = vec![0.0; m];
            let mut counts: Vec<usize> = vec![0; m];

            for ((i, j), t_val) in t_stats {
                // t_ij > 0 means model i has higher loss (worse)
                model_avgs[*i] += *t_val;
                model_avgs[*j] -= *t_val; // Symmetric: t_ji = -t_ij
                counts[*i] += 1;
                counts[*j] += 1;
            }

            for i in 0..m {
                if counts[i] > 0 {
                    model_avgs[i] /= counts[i] as f64;
                }
            }

            // Return max of averages (worst model has highest average)
            model_avgs.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        }
    }
}

/// Identify the worst model based on average relative performance
fn identify_worst_model(t_stats: &HashMap<(usize, usize), f64>, m: usize) -> usize {
    // Compute average t-statistic for each model
    // Higher average = worse performance (higher relative loss)
    let mut model_scores: Vec<f64> = vec![0.0; m];
    let mut counts: Vec<usize> = vec![0; m];

    for ((i, j), t_val) in t_stats {
        model_scores[*i] += *t_val;
        model_scores[*j] -= *t_val;
        counts[*i] += 1;
        counts[*j] += 1;
    }

    for i in 0..m {
        if counts[i] > 0 {
            model_scores[i] /= counts[i] as f64;
        }
    }

    // Return index of model with highest average (worst performance)
    model_scores
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mcs_single_model() {
        let losses = vec![vec![1.0, 2.0, 3.0]];
        let result =
            model_confidence_set(&losses, 0.10, MCSStatistic::Range, 100, 3.0, Some(42)).unwrap();
        assert_eq!(result.included_models, vec![0]);
        assert!(result.eliminated_models.is_empty());
        assert_eq!(result.mcs_p_value, 1.0);
    }

    #[test]
    fn test_mcs_clearly_inferior_model() {
        // Model 0: Low losses (good), Model 1: High losses (bad)
        let losses = vec![vec![1.0; 100], vec![10.0; 100]];
        let result =
            model_confidence_set(&losses, 0.10, MCSStatistic::Range, 500, 5.0, Some(42)).unwrap();

        // Model 1 should be eliminated
        assert!(result.included_models.contains(&0));
        assert!(result.eliminated_models.contains(&1));
        assert_eq!(result.included_models.len(), 1);
    }

    #[test]
    fn test_mcs_equivalent_models() {
        // All models have identical performance - should all be included
        let base: Vec<f64> = (0..100)
            .map(|i| (i as f64 * 0.1).sin().abs() + 1.0)
            .collect();
        let losses = vec![base.clone(), base.clone(), base.clone()];
        let result =
            model_confidence_set(&losses, 0.10, MCSStatistic::Range, 500, 5.0, Some(42)).unwrap();

        // All models should remain when they're identical
        assert_eq!(result.included_models.len(), 3);
        assert!(result.eliminated_models.is_empty());
    }

    #[test]
    fn test_mcs_multiple_inferior_models() {
        // Model 0: Best, Model 1: Medium, Model 2: Worst
        let losses = vec![
            vec![1.0; 100], // Best
            vec![2.0; 100], // Medium
            vec![5.0; 100], // Worst
        ];
        let result =
            model_confidence_set(&losses, 0.10, MCSStatistic::Range, 500, 5.0, Some(42)).unwrap();

        // Model 0 should definitely be included
        assert!(result.included_models.contains(&0));
        // Worst models should be eliminated
        assert!(result.eliminated_models.contains(&2));
    }

    #[test]
    fn test_mcs_t_max_statistic() {
        // Test with T_max statistic
        let losses = vec![vec![1.0; 100], vec![10.0; 100]];
        let result =
            model_confidence_set(&losses, 0.10, MCSStatistic::Max, 500, 5.0, Some(42)).unwrap();

        assert!(result.included_models.contains(&0));
        assert_eq!(result.statistic_type, MCSStatistic::Max);
    }

    #[test]
    fn test_mcs_elimination_sequence() {
        // Test that elimination sequence is properly recorded
        let losses = vec![
            vec![1.0; 100], // Best
            vec![3.0; 100], // Medium
            vec![5.0; 100], // Worst
        ];
        let result =
            model_confidence_set(&losses, 0.25, MCSStatistic::Range, 500, 5.0, Some(42)).unwrap();

        // Should have some elimination steps
        assert!(!result.elimination_sequence.is_empty());

        // Each step should have proper structure
        for step in &result.elimination_sequence {
            assert!(step.p_value >= 0.0 && step.p_value <= 1.0);
        }
    }

    #[test]
    fn test_mcs_empty_error() {
        let losses: Vec<Vec<f64>> = vec![];
        assert!(model_confidence_set(&losses, 0.10, MCSStatistic::Range, 100, 3.0, None).is_err());
    }

    #[test]
    fn test_mcs_length_mismatch() {
        let losses = vec![vec![1.0, 2.0, 3.0], vec![1.0, 2.0]]; // Different lengths
        assert!(model_confidence_set(&losses, 0.10, MCSStatistic::Range, 100, 3.0, None).is_err());
    }

    #[test]
    fn test_mcs_invalid_alpha() {
        let losses = vec![vec![1.0, 2.0, 3.0]];
        assert!(model_confidence_set(&losses, 0.0, MCSStatistic::Range, 100, 3.0, None).is_err());
        assert!(model_confidence_set(&losses, 1.0, MCSStatistic::Range, 100, 3.0, None).is_err());
        assert!(model_confidence_set(&losses, -0.1, MCSStatistic::Range, 100, 3.0, None).is_err());
    }

    #[test]
    fn test_mcs_zero_bootstrap() {
        let losses = vec![vec![1.0, 2.0, 3.0]];
        assert!(model_confidence_set(&losses, 0.10, MCSStatistic::Range, 0, 3.0, None).is_err());
    }
}
