//! Shared utilities for SPA-style bootstrap tests (Hansen, 2005).
//!
//! Used by both `spa_test` and `mspe_adjusted_spa`.
//!
//! With `d_{k,t}` the performance differential of model `k` against the
//! benchmark (positive = model `k` better), `d̄_k` its mean and `n` the sample
//! size, Hansen's (2005) test of `H0: E[d_k] <= 0 for all k` is
//!
//! ```text
//! T_SPA = max(0, max_k sqrt(n) d̄_k / ω̂_k)
//! ```
//!
//! where `ω̂_k²` is the stationary-bootstrap variance estimator
//! `γ̂_0 + 2 Σ_{i=1}^{n-1} κ(n, i) γ̂_i`,
//! `κ(n, i) = (n-i)/n (1-q)^i + i/n (1-q)^(n-i)`, `q = 1 / block_length`.
//! The null distribution is approximated by one stationary-bootstrap index
//! draw per replication, applied jointly to all `K` series:
//! `T*_b = max(0, max_k sqrt(n) (d̄*_{k,b} - g(d̄_k)) / ω̂_k)` with the
//! re-centrings
//!
//! * lower:      `g(x) = max(x, 0)`,
//! * consistent: `g(x) = x · 1{x >= -sqrt(ω̂_k² / n · 2 ln ln n)}`,
//! * upper:      `g(x) = x` (White's Reality Check centring),
//!
//! and the p-values `p = #{T*_b > T_SPA} / B`, so that `p_l <= p_c <= p_u`.

use crate::error::{Result, StatError};
use crate::resampling::bootstrap::StationaryBootstrap;
use crate::utils::finite::ensure_finite;

/// Validate that model data has consistent dimensions.
pub fn validate_model_data(
    benchmark_len: usize,
    model_data: &[Vec<f64>],
    data_name: &str,
) -> Result<()> {
    if benchmark_len == 0 {
        return Err(StatError::EmptyData);
    }

    if model_data.is_empty() {
        return Err(StatError::InvalidParameter(format!(
            "At least one {} required",
            data_name
        )));
    }

    for (i, data) in model_data.iter().enumerate() {
        if data.len() != benchmark_len {
            return Err(StatError::InvalidParameter(format!(
                "Model {} has {} observations, expected {}",
                i,
                data.len(),
                benchmark_len
            )));
        }
    }

    Ok(())
}

/// Validate that benchmark and model data contain no NaN or ±Inf.
pub fn validate_finite_data(
    benchmark: &[f64],
    model_data: &[Vec<f64>],
    data_name: &str,
) -> Result<()> {
    ensure_finite("benchmark", benchmark)?;
    for (i, data) in model_data.iter().enumerate() {
        ensure_finite(&format!("{} {}", data_name, i), data)?;
    }
    Ok(())
}

/// Validate the bootstrap parameters of an SPA-style test.
pub fn validate_spa_parameters(t: usize, n_bootstrap: usize, block_length: f64) -> Result<()> {
    if t < 2 {
        return Err(StatError::InsufficientData { needed: 2, got: t });
    }
    if n_bootstrap == 0 {
        return Err(StatError::InvalidParameter(
            "n_bootstrap must be positive".to_string(),
        ));
    }
    if !block_length.is_finite() || block_length <= 0.0 {
        return Err(StatError::InvalidParameter(format!(
            "block_length must be positive and finite, got {}",
            block_length
        )));
    }
    Ok(())
}

/// Output of the SPA computation shared by `spa_test` and `mspe_adjusted_spa`.
pub(crate) struct SpaOutput {
    pub statistic: f64,
    pub p_value_lower: f64,
    pub p_value_consistent: f64,
    pub p_value_upper: f64,
    pub best_model_idx: Option<usize>,
}

/// Hansen's (2005) stationary-bootstrap variance estimator `ω̂²` of
/// `sqrt(n) · d̄` (see the module documentation).
pub(crate) fn spa_omega2(d: &[f64], block_length: f64) -> f64 {
    let n = d.len();
    let n_f = n as f64;
    let mean = d.iter().sum::<f64>() / n_f;
    let c: Vec<f64> = d.iter().map(|x| x - mean).collect();
    let q = 1.0 / block_length.max(1.0);
    let gamma = |i: usize| -> f64 { (0..n - i).map(|j| c[j] * c[j + i]).sum::<f64>() / n_f };
    let mut omega2 = gamma(0);
    for i in 1..n {
        let i_f = i as f64;
        let kappa = (n_f - i_f) / n_f * (1.0 - q).powi(i as i32)
            + i_f / n_f * (1.0 - q).powi((n - i) as i32);
        if kappa != 0.0 {
            omega2 += 2.0 * kappa * gamma(i);
        }
    }
    omega2
}

/// The SPA statistic, the three p-values and the best model, with bootstrap
/// index vectors supplied by `draw` (one call per replication).
pub(crate) fn spa_core<D: FnMut() -> Vec<usize>>(
    differentials: &[Vec<f64>],
    block_length: f64,
    n_bootstrap: usize,
    mut draw: D,
) -> SpaOutput {
    let n = differentials[0].len();
    let n_f = n as f64;
    let sqrt_n = n_f.sqrt();
    let loglog = if n_f.ln() > 1.0 {
        2.0 * n_f.ln().ln()
    } else {
        0.0
    };

    let d_bar: Vec<f64> = differentials
        .iter()
        .map(|d| d.iter().sum::<f64>() / n_f)
        .collect();
    let omega: Vec<f64> = differentials
        .iter()
        .map(|d| spa_omega2(d, block_length).max(0.0).sqrt())
        .collect();
    // A model with (numerically) zero variance has a degenerate t-ratio:
    // +inf if it beats the benchmark, otherwise it is ignored.
    let degenerate: Vec<bool> = omega
        .iter()
        .zip(&d_bar)
        .map(|(&w, &m)| w <= 1e-14 * m.abs().max(1.0))
        .collect();

    let t_k: Vec<f64> = (0..d_bar.len())
        .map(|k| {
            if degenerate[k] {
                if d_bar[k] > 0.0 {
                    f64::INFINITY
                } else {
                    f64::NEG_INFINITY
                }
            } else {
                sqrt_n * d_bar[k] / omega[k]
            }
        })
        .collect();
    let statistic = t_k.iter().cloned().fold(0.0_f64, f64::max);

    let mut best_model_idx = None;
    let mut best = f64::NEG_INFINITY;
    let mut best_mean = f64::NEG_INFINITY;
    // Largest t-ratio; ties (e.g. several zero-variance winners at +inf) are
    // broken by the larger mean differential.
    for (k, &v) in t_k.iter().enumerate() {
        if v > best || (v == best && d_bar[k] > best_mean) {
            best = v;
            best_mean = d_bar[k];
            best_model_idx = Some(k);
        }
    }

    // Re-centring terms g(d̄_k) for the lower, consistent and upper p-values.
    let g_l: Vec<f64> = d_bar.iter().map(|&m| m.max(0.0)).collect();
    let g_c: Vec<f64> = d_bar
        .iter()
        .zip(&omega)
        .map(|(&m, &w)| {
            if m >= -(w * w / n_f * loglog).sqrt() {
                m
            } else {
                0.0
            }
        })
        .collect();
    let g_u = d_bar.clone();

    let mut counts = [0usize; 3];
    let mut boot_mean = vec![0.0; d_bar.len()];
    for _ in 0..n_bootstrap {
        let idx = draw();
        for (k, d) in differentials.iter().enumerate() {
            boot_mean[k] = idx.iter().map(|&i| d[i]).sum::<f64>() / n_f;
        }
        for (c, g) in counts.iter_mut().zip([&g_l, &g_c, &g_u]) {
            let mut t_star = 0.0_f64;
            for k in 0..d_bar.len() {
                if !degenerate[k] {
                    t_star = t_star.max(sqrt_n * (boot_mean[k] - g[k]) / omega[k]);
                }
            }
            if t_star > statistic {
                *c += 1;
            }
        }
    }
    let b = n_bootstrap as f64;
    SpaOutput {
        statistic,
        p_value_lower: counts[0] as f64 / b,
        p_value_consistent: counts[1] as f64 / b,
        p_value_upper: counts[2] as f64 / b,
        best_model_idx,
    }
}

/// [`spa_core`] with the stationary bootstrap of expected block length
/// `block_length`.
pub(crate) fn spa_stationary(
    differentials: &[Vec<f64>],
    n_bootstrap: usize,
    block_length: f64,
    seed: Option<u64>,
) -> SpaOutput {
    let n = differentials[0].len();
    let mut bootstrap = StationaryBootstrap::new(block_length, seed);
    spa_core(differentials, block_length, n_bootstrap, || {
        bootstrap.sample_indices(n, n)
    })
}
