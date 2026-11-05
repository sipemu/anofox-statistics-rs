mod types;
mod validation;

pub use types::{MCSEliminationStep, MCSResult, MCSStatistic};

use validation::validate_mcs_inputs;

use crate::error::{Result, StatError};
use crate::resampling::bootstrap::{BootstrapMethod, IndexSampler};

/// Perform the Model Confidence Set procedure (Hansen, Lunde & Nason, 2011)
/// with the stationary bootstrap.
///
/// Identifies a set of models that contains the best model with a given
/// confidence level. Uses sequential elimination based on bootstrap tests
/// of equal predictive ability. This is
/// [`model_confidence_set_with_method`] with
/// [`BootstrapMethod::Stationary`] of expected block length `block_length`
/// (values below 1 are treated as 1, the IID bootstrap).
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
    let l = if block_length.is_finite() {
        block_length.max(1.0)
    } else {
        1.0
    };
    model_confidence_set_with_method(
        losses,
        alpha,
        statistic,
        n_bootstrap,
        BootstrapMethod::Stationary {
            expected_block_length: l,
        },
        seed,
    )
}

/// Perform the Model Confidence Set procedure with a chosen bootstrap scheme.
///
/// Follows Hansen, Lunde & Nason (2011) as implemented in R's
/// `MCS::MCSprocedure`: one bootstrap index vector per replication is applied
/// jointly to the losses of all models (preserving their cross-sectional
/// dependence); the variances of the loss contrasts are the bootstrap
/// variances; at each step the worst surviving model is tested
/// (`T_max`: largest `t_{i.}`; `T_R`: largest `max_j t_{ij}`), its step
/// p-value is `#{T*_b >= T} / B`, and the MCS p-values are the running
/// maximum of the step p-values. [`BootstrapMethod::MovingBlock`] with block
/// length `k` is the scheme of `MCSprocedure(k = k)` (`k = 1`: IID).
///
/// Results are deterministic for a given `seed`.
///
/// # Errors
/// No models, unequal lengths, non-finite losses, `alpha` outside (0, 1),
/// `n_bootstrap == 0`, an invalid block length, or a non-constant loss
/// contrast whose bootstrap variance is zero.
pub fn model_confidence_set_with_method(
    losses: &[Vec<f64>],
    alpha: f64,
    statistic: MCSStatistic,
    n_bootstrap: usize,
    method: BootstrapMethod,
    seed: Option<u64>,
) -> Result<MCSResult> {
    validate_mcs_inputs(losses, alpha, n_bootstrap)?;
    method.validate()?;
    let t = losses[0].len();
    let mut sampler = IndexSampler::new(method, seed);
    mcs_core(losses, alpha, statistic, n_bootstrap, || sampler.draw(t))
}

const TOL: f64 = 64.0 * f64::EPSILON;

/// A standardized loss contrast (R `MCS:::StandardizeContrast`).
struct Standardized {
    statistic: f64,
    /// Bootstrap t-ratios (`None`: degenerate contrast, all zero)
    bootstrap: Option<Vec<f64>>,
    se: f64,
}

fn max_abs(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |a, &x| a.max(x.abs()))
}

fn standardize_contrast<C: FnOnce() -> Vec<f64>>(
    observed: f64,
    errors: Vec<f64>,
    contrast: C,
) -> Result<Standardized> {
    let error_scale = max_abs(&errors);
    if error_scale <= TOL * observed.abs().max(1.0) {
        let d = contrast();
        let d_scale = max_abs(&d);
        let constant = d_scale == 0.0 || d.iter().all(|&x| (x - d[0]).abs() <= TOL * d_scale);
        if constant {
            let value = d.iter().sum::<f64>() / d.len() as f64;
            let statistic = if value == 0.0 {
                0.0
            } else if value > 0.0 {
                f64::INFINITY
            } else {
                f64::NEG_INFINITY
            };
            return Ok(Standardized {
                statistic,
                bootstrap: None,
                se: 0.0,
            });
        }
        if error_scale == 0.0 {
            return Err(StatError::InvalidParameter(
                "bootstrap variance is zero for a non-constant loss contrast; \
                 increase n_bootstrap or change the block length"
                    .to_string(),
            ));
        }
    }
    let ms = errors
        .iter()
        .map(|&e| (e / error_scale).powi(2))
        .sum::<f64>()
        / errors.len() as f64;
    let se = error_scale * ms.sqrt();
    let bootstrap = errors.into_iter().map(|e| e / se).collect();
    Ok(Standardized {
        statistic: observed / se,
        bootstrap: Some(bootstrap),
        se,
    })
}

/// Index of the first maximum.
fn first_argmax(v: &[f64]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

fn mean_ge(boot_max: &[f64], observed: f64) -> f64 {
    boot_max.iter().filter(|&&b| b >= observed).count() as f64 / boot_max.len() as f64
}

/// The MCS procedure with bootstrap index vectors supplied by `draw` (one
/// call per replication).
pub(crate) fn mcs_core<D: FnMut() -> Vec<usize>>(
    losses: &[Vec<f64>],
    alpha: f64,
    statistic: MCSStatistic,
    n_bootstrap: usize,
    mut draw: D,
) -> Result<MCSResult> {
    let m = losses.len();
    let t = losses[0].len();
    let t_f = t as f64;

    // Losses relative to model 0 (as MCSprocedure; contrasts are unchanged).
    let work: Vec<Vec<f64>> = losses
        .iter()
        .map(|l| l.iter().zip(&losses[0]).map(|(a, b)| a - b).collect())
        .collect();
    let means: Vec<f64> = work.iter().map(|w| w.iter().sum::<f64>() / t_f).collect();

    let (order, p) = if m == 1 {
        (vec![0], vec![1.0])
    } else {
        // errors[k][b]: bootstrap mean of the centred losses of model k.
        let centered: Vec<Vec<f64>> = work
            .iter()
            .zip(&means)
            .map(|(w, &mu)| w.iter().map(|x| x - mu).collect())
            .collect();
        let mut errors = vec![vec![0.0; n_bootstrap]; m];
        for b in 0..n_bootstrap {
            let idx = draw();
            for (err, c) in errors.iter_mut().zip(&centered) {
                err[b] = idx.iter().map(|&i| c[i]).sum::<f64>() / t_f;
            }
        }
        match statistic {
            MCSStatistic::Max => path_tmax(&work, &means, &errors)?,
            MCSStatistic::Range => path_tr(&work, &means, &errors)?,
        }
    };

    // MCS p-values: running maximum along the elimination path.
    let mut adjusted = Vec::with_capacity(m);
    let mut run = 0.0_f64;
    for &pv in &p {
        run = run.max(pv);
        adjusted.push(run);
    }
    let mut mcs_p_values = vec![1.0; m];
    for (step, &model) in order.iter().enumerate() {
        mcs_p_values[model] = adjusted[step];
    }

    let mut elimination_sequence = Vec::new();
    let mut eliminated_models = Vec::new();
    let mut mcs_p_value = 1.0;
    for step in 0..m.saturating_sub(1) {
        let eliminated = adjusted[step] < alpha;
        elimination_sequence.push(MCSEliminationStep {
            model_idx: order[step],
            p_value: p[step],
            mcs_p_value: adjusted[step],
            eliminated,
        });
        mcs_p_value = adjusted[step];
        if eliminated {
            eliminated_models.push(order[step]);
        } else {
            break;
        }
    }
    let mut included_models: Vec<usize> =
        (0..m).filter(|i| !eliminated_models.contains(i)).collect();
    included_models.sort_unstable();

    Ok(MCSResult {
        included_models,
        eliminated_models,
        mcs_p_value,
        elimination_sequence,
        elimination_order: order,
        mcs_p_values,
        n_bootstrap,
        statistic_type: statistic,
    })
}

/// Elimination path and step p-values with the `T_max` statistic
/// (R `MCS:::MCSPathTmax`).
fn path_tmax(
    work: &[Vec<f64>],
    means: &[f64],
    errors: &[Vec<f64>],
) -> Result<(Vec<usize>, Vec<f64>)> {
    let m = work.len();
    let n_boot = errors[0].len();
    let t = work[0].len();
    let mut active: Vec<usize> = (0..m).collect();
    let mut order = Vec::with_capacity(m);
    let mut p = vec![1.0; m];
    for p_step in p.iter_mut().take(m - 1) {
        let a = active.len() as f64;
        let mean_active = active.iter().map(|&i| means[i]).sum::<f64>() / a;
        let row_mean: Vec<f64> = (0..n_boot)
            .map(|b| active.iter().map(|&i| errors[i][b]).sum::<f64>() / a)
            .collect();
        let mut stats = Vec::with_capacity(active.len());
        let mut boot_max = vec![f64::NEG_INFINITY; n_boot];
        for &i in &active {
            let observed = means[i] - mean_active;
            let centered: Vec<f64> = (0..n_boot).map(|b| errors[i][b] - row_mean[b]).collect();
            let s = standardize_contrast(observed, centered, || {
                (0..t)
                    .map(|j| work[i][j] - active.iter().map(|&h| work[h][j]).sum::<f64>() / a)
                    .collect()
            })?;
            stats.push(s.statistic);
            match &s.bootstrap {
                Some(bs) => {
                    for (bm, &x) in boot_max.iter_mut().zip(bs) {
                        *bm = bm.max(x);
                    }
                }
                None => {
                    for bm in boot_max.iter_mut() {
                        *bm = bm.max(0.0);
                    }
                }
            }
        }
        let e = first_argmax(&stats);
        order.push(active[e]);
        *p_step = mean_ge(&boot_max, stats[e]);
        active.remove(e);
    }
    order.push(active[0]);
    Ok((order, p))
}

/// Elimination path and step p-values with the `T_R` statistic
/// (R `MCS:::MCSPathTR`).
fn path_tr(
    work: &[Vec<f64>],
    means: &[f64],
    errors: &[Vec<f64>],
) -> Result<(Vec<usize>, Vec<f64>)> {
    let m = work.len();
    let n_boot = errors[0].len();
    let mut pair_t = vec![vec![0.0; m]; m];
    let mut pair_se = vec![vec![0.0; m]; m];
    for i in 0..m - 1 {
        for j in i + 1..m {
            let e: Vec<f64> = (0..n_boot).map(|b| errors[i][b] - errors[j][b]).collect();
            let s = standardize_contrast(means[i] - means[j], e, || {
                work[i].iter().zip(&work[j]).map(|(a, b)| a - b).collect()
            })?;
            pair_t[i][j] = s.statistic;
            pair_t[j][i] = -s.statistic;
            pair_se[i][j] = s.se;
            pair_se[j][i] = s.se;
        }
    }
    let mut active: Vec<usize> = (0..m).collect();
    let mut order = Vec::with_capacity(m);
    let mut observed = Vec::with_capacity(m - 1);
    for _ in 0..m - 1 {
        let score: Vec<f64> = active
            .iter()
            .map(|&i| {
                active
                    .iter()
                    .map(|&j| pair_t[i][j])
                    .fold(f64::NEG_INFINITY, f64::max)
            })
            .collect();
        let e = first_argmax(&score);
        order.push(active[e]);
        observed.push(score[e]);
        active.remove(e);
    }
    order.push(active[0]);

    let mut p = vec![1.0; m];
    let mut boot_max = vec![0.0_f64; n_boot];
    for step in (0..m - 1).rev() {
        let i = order[step];
        for &j in &order[step + 1..] {
            let se = pair_se[i][j];
            if se > 0.0 {
                for (b, bm) in boot_max.iter_mut().enumerate() {
                    *bm = bm.max((errors[i][b] - errors[j][b]).abs() / se);
                }
            }
        }
        p[step] = mean_ge(&boot_max, observed[step]);
    }
    Ok((order, p))
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
        let losses = vec![vec![1.0; 100], vec![10.0; 100]];
        let result =
            model_confidence_set(&losses, 0.10, MCSStatistic::Range, 500, 5.0, Some(42)).unwrap();

        assert!(result.included_models.contains(&0));
        assert!(result.eliminated_models.contains(&1));
        assert_eq!(result.included_models.len(), 1);
    }

    #[test]
    fn test_mcs_equivalent_models() {
        let base: Vec<f64> = (0..100)
            .map(|i| (i as f64 * 0.1).sin().abs() + 1.0)
            .collect();
        let losses = vec![base.clone(), base.clone(), base.clone()];
        let result =
            model_confidence_set(&losses, 0.10, MCSStatistic::Range, 500, 5.0, Some(42)).unwrap();

        assert_eq!(result.included_models.len(), 3);
        assert!(result.eliminated_models.is_empty());
    }

    #[test]
    fn test_mcs_multiple_inferior_models() {
        let losses = vec![vec![1.0; 100], vec![2.0; 100], vec![5.0; 100]];
        let result =
            model_confidence_set(&losses, 0.10, MCSStatistic::Range, 500, 5.0, Some(42)).unwrap();

        assert!(result.included_models.contains(&0));
        assert!(result.eliminated_models.contains(&2));
    }

    #[test]
    fn test_mcs_t_max_statistic() {
        let losses = vec![vec![1.0; 100], vec![10.0; 100]];
        let result =
            model_confidence_set(&losses, 0.10, MCSStatistic::Max, 500, 5.0, Some(42)).unwrap();

        assert!(result.included_models.contains(&0));
        assert_eq!(result.statistic_type, MCSStatistic::Max);
    }

    #[test]
    fn test_mcs_elimination_sequence() {
        let losses = vec![vec![1.0; 100], vec![3.0; 100], vec![5.0; 100]];
        let result =
            model_confidence_set(&losses, 0.25, MCSStatistic::Range, 500, 5.0, Some(42)).unwrap();

        assert!(!result.elimination_sequence.is_empty());

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
        let losses = vec![vec![1.0, 2.0, 3.0], vec![1.0, 2.0]];
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
