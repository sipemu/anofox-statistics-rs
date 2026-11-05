//! Percentile bootstrap confidence intervals.

use crate::error::{Result, StatError};
use crate::resampling::{
    BootstrapMethod, CircularBlockBootstrap, MovingBlockBootstrap, StationaryBootstrap,
};
use crate::utils::rng::index;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Result of a percentile bootstrap confidence interval
#[derive(Debug, Clone)]
pub struct BootstrapCIResult {
    /// The statistic evaluated on the original data
    pub estimate: f64,
    /// Bootstrap standard error (sample SD of the replicates)
    pub se: f64,
    /// Lower bound of the percentile interval
    pub conf_int_lower: f64,
    /// Upper bound of the percentile interval
    pub conf_int_upper: f64,
    /// Confidence level used
    pub conf_level: f64,
    /// Number of bootstrap replicates
    pub n_bootstrap: usize,
}

fn validate(data: &[f64], n_bootstrap: usize, conf_level: f64) -> Result<()> {
    if data.len() < 2 {
        return Err(StatError::InsufficientData {
            needed: 2,
            got: data.len(),
        });
    }
    if n_bootstrap < 2 {
        return Err(StatError::InvalidParameter(format!(
            "n_bootstrap must be >= 2, got {}",
            n_bootstrap
        )));
    }
    if !(conf_level > 0.0 && conf_level < 1.0) {
        return Err(StatError::InvalidParameter(format!(
            "conf_level must be in (0, 1), got {}",
            conf_level
        )));
    }
    Ok(())
}

/// Percentile interval and standard error from bootstrap replicates.
///
/// Sorts the replicates (NaN-safe, NaN last) and returns
/// `(lower, upper, se)` with `lower = R[floor(alpha/2 * B)]` and
/// `upper = R[min(ceil((1 - alpha/2) * B), B - 1)]` (0-based, `alpha = 1 - conf_level`).
fn percentile_interval(mut reps: Vec<f64>, conf_level: f64) -> (f64, f64, f64) {
    if reps.iter().any(|r| r.is_nan()) {
        // An undefined replicate makes the percentiles undefined.
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    reps.sort_by(|a, b| a.total_cmp(b));
    let b = reps.len();
    let alpha = 1.0 - conf_level;
    let lower_idx = ((alpha / 2.0) * b as f64).floor() as usize;
    let upper_idx = ((1.0 - alpha / 2.0) * b as f64).ceil() as usize;
    let m = reps.iter().sum::<f64>() / b as f64;
    let se = (reps.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (b - 1) as f64).sqrt();
    (reps[lower_idx.min(b - 1)], reps[upper_idx.min(b - 1)], se)
}

fn iid_replicates<F: Fn(&[f64]) -> f64>(
    data: &[f64],
    statistic: &F,
    n_bootstrap: usize,
    seed: Option<u64>,
) -> Vec<f64> {
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };
    let n = data.len();
    (0..n_bootstrap)
        .map(|_| {
            let sample: Vec<f64> = (0..n).map(|_| data[index(&mut rng, n)]).collect();
            statistic(&sample)
        })
        .collect()
}

/// IID (nonparametric) percentile bootstrap confidence interval for an
/// arbitrary statistic.
///
/// Draws `n_bootstrap` resamples of size `n` with replacement, evaluates
/// `statistic` on each and reports the percentile interval
/// `[R(floor(alpha/2 * B)), R(min(ceil((1 - alpha/2) * B), B - 1))]` of the sorted
/// replicates (0-based, `alpha = 1 - conf_level`) together with the bootstrap
/// standard error. With a `seed` the result is reproducible
/// (`StdRng::seed_from_u64`); without one the generator is seeded from entropy.
///
/// # Errors
/// Fewer than 2 observations, `n_bootstrap < 2`, or `conf_level` outside (0, 1).
///
/// # Examples
/// ```
/// use anofox_statistics::bootstrap_ci;
///
/// let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
/// let median = |s: &[f64]| {
///     let mut v = s.to_vec();
///     v.sort_by(|a, b| a.total_cmp(b));
///     (v[4] + v[5]) / 2.0
/// };
/// let r = bootstrap_ci(&x, median, 2000, 0.95, Some(42)).unwrap();
/// assert_eq!(r.estimate, 5.5);
/// assert!(r.conf_int_lower <= 5.5 && r.conf_int_upper >= 5.5);
/// ```
pub fn bootstrap_ci<F: Fn(&[f64]) -> f64>(
    data: &[f64],
    statistic: F,
    n_bootstrap: usize,
    conf_level: f64,
    seed: Option<u64>,
) -> Result<BootstrapCIResult> {
    validate(data, n_bootstrap, conf_level)?;
    let reps = iid_replicates(data, &statistic, n_bootstrap, seed);
    let (lo, hi, se) = percentile_interval(reps, conf_level);
    Ok(BootstrapCIResult {
        estimate: statistic(data),
        se,
        conf_int_lower: lo,
        conf_int_upper: hi,
        conf_level,
        n_bootstrap,
    })
}

/// Options of [`bootstrap_ci_with_options`] / [`bootstrap_mean_ci_with_options`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BootstrapCIOptions {
    /// Number of bootstrap replicates (`>= 2`). Default 1000.
    pub n_bootstrap: usize,
    /// Confidence level in (0, 1). Default 0.95.
    pub conf_level: f64,
    /// Resampling scheme and (expected) block length. Default
    /// [`BootstrapMethod::Iid`].
    pub method: BootstrapMethod,
    /// Random seed; `None` seeds from entropy. Default `None`.
    pub seed: Option<u64>,
}

impl Default for BootstrapCIOptions {
    fn default() -> Self {
        Self {
            n_bootstrap: 1000,
            conf_level: 0.95,
            method: BootstrapMethod::Iid,
            seed: None,
        }
    }
}

impl BootstrapMethod {
    /// The scheme [`bootstrap_mean_ci`] uses for a `block_length` argument:
    /// 0 is [`Iid`](Self::Iid), 1 is [`Stationary`](Self::Stationary) with
    /// expected block length 1, larger values are
    /// [`CircularBlock`](Self::CircularBlock) with that block length.
    pub fn from_block_length(block_length: usize) -> Self {
        match block_length {
            0 => BootstrapMethod::Iid,
            1 => BootstrapMethod::Stationary {
                expected_block_length: 1.0,
            },
            l => BootstrapMethod::CircularBlock { block_length: l },
        }
    }
}

fn replicates<F: Fn(&[f64]) -> f64>(
    data: &[f64],
    statistic: &F,
    n_bootstrap: usize,
    method: BootstrapMethod,
    seed: Option<u64>,
) -> Vec<f64> {
    let n = data.len();
    let resample = |idx: Vec<usize>| -> f64 {
        let s: Vec<f64> = idx.into_iter().map(|i| data[i]).collect();
        statistic(&s)
    };
    match method {
        BootstrapMethod::Iid => iid_replicates(data, statistic, n_bootstrap, seed),
        BootstrapMethod::Stationary {
            expected_block_length,
        } => {
            let mut b = StationaryBootstrap::new(expected_block_length, seed);
            (0..n_bootstrap)
                .map(|_| resample(b.sample_indices(n, n)))
                .collect()
        }
        BootstrapMethod::CircularBlock { block_length } => {
            let mut b = CircularBlockBootstrap::new(block_length, seed);
            (0..n_bootstrap)
                .map(|_| resample(b.sample_indices(n, n)))
                .collect()
        }
        BootstrapMethod::MovingBlock { block_length } => {
            let mut b = MovingBlockBootstrap::new(block_length, seed);
            (0..n_bootstrap)
                .map(|_| resample(b.sample_indices(n, n)))
                .collect()
        }
    }
}

/// Percentile (block) bootstrap confidence interval for an arbitrary
/// statistic, with the resampling scheme chosen by `options.method`.
///
/// [`BootstrapMethod::Iid`] reproduces [`bootstrap_ci`] exactly (same seed,
/// same replicates). [`BootstrapMethod::Stationary`] corresponds to R's
/// `boot::tsboot(x, statistic, R, l = expected_block_length, sim = "geom")`,
/// [`BootstrapMethod::CircularBlock`] to `tsboot(..., sim = "fixed")` (both
/// with the default `endcorr = TRUE`, i.e. wrapping around the end of the
/// series). The interval and standard error are computed as in
/// [`bootstrap_ci`].
///
/// # Errors
/// Fewer than 2 observations, `n_bootstrap < 2`, `conf_level` outside (0, 1),
/// or a block length below 1.
pub fn bootstrap_ci_with_options<F: Fn(&[f64]) -> f64>(
    data: &[f64],
    statistic: F,
    options: &BootstrapCIOptions,
) -> Result<BootstrapCIResult> {
    validate(data, options.n_bootstrap, options.conf_level)?;
    options.method.validate()?;
    let reps = replicates(
        data,
        &statistic,
        options.n_bootstrap,
        options.method,
        options.seed,
    );
    let (lo, hi, se) = percentile_interval(reps, options.conf_level);
    Ok(BootstrapCIResult {
        estimate: statistic(data),
        se,
        conf_int_lower: lo,
        conf_int_upper: hi,
        conf_level: options.conf_level,
        n_bootstrap: options.n_bootstrap,
    })
}

/// Percentile (block) bootstrap confidence interval for the mean, with the
/// resampling scheme and (expected) block length chosen by `options.method`.
///
/// See [`bootstrap_ci_with_options`]; validated against R's
/// `boot::tsboot(sim = "geom")` / `tsboot(sim = "fixed")`.
///
/// # Examples
/// ```
/// use anofox_statistics::{bootstrap_mean_ci_with_options, BootstrapCIOptions, BootstrapMethod};
///
/// let x: Vec<f64> = (0..50).map(|i| (i as f64 * 0.3).sin()).collect();
/// let opts = BootstrapCIOptions {
///     n_bootstrap: 2000,
///     method: BootstrapMethod::Stationary { expected_block_length: 5.0 },
///     seed: Some(42),
///     ..Default::default()
/// };
/// let r = bootstrap_mean_ci_with_options(&x, &opts).unwrap();
/// assert!(r.conf_int_lower < r.estimate && r.estimate < r.conf_int_upper);
/// ```
pub fn bootstrap_mean_ci_with_options(
    data: &[f64],
    options: &BootstrapCIOptions,
) -> Result<BootstrapCIResult> {
    bootstrap_ci_with_options(
        data,
        |s: &[f64]| s.iter().sum::<f64>() / s.len() as f64,
        options,
    )
}

/// Percentile bootstrap confidence interval for the mean.
///
/// * `block_length == 0`: IID bootstrap (same as [`bootstrap_ci`] with the mean);
/// * `block_length == 1`: [`StationaryBootstrap`] with expected block length 1;
/// * `block_length > 1`: [`CircularBlockBootstrap`] with that block length
///   (for dependent data).
///
/// This mapping is [`BootstrapMethod::from_block_length`]; use
/// [`bootstrap_mean_ci_with_options`] to choose the scheme and the (expected)
/// block length freely, e.g. a stationary bootstrap with expected block
/// length 5. The interval and standard error are computed as in
/// [`bootstrap_ci`].
///
/// # Errors
/// Fewer than 2 observations, `n_bootstrap < 2`, or `conf_level` outside (0, 1).
///
/// # Examples
/// ```
/// use anofox_statistics::bootstrap_mean_ci;
///
/// let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
/// let r = bootstrap_mean_ci(&x, 1000, 0.95, 0, Some(42)).unwrap();
/// assert!((r.estimate - 5.5).abs() < 1e-12);
/// assert!(r.conf_int_lower < 5.5 && r.conf_int_upper > 5.5);
/// ```
pub fn bootstrap_mean_ci(
    data: &[f64],
    n_bootstrap: usize,
    conf_level: f64,
    block_length: usize,
    seed: Option<u64>,
) -> Result<BootstrapCIResult> {
    bootstrap_mean_ci_with_options(
        data,
        &BootstrapCIOptions {
            n_bootstrap,
            conf_level,
            method: BootstrapMethod::from_block_length(block_length),
            seed,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const X: [f64; 10] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

    #[test]
    fn mean_ci_is_reproducible_and_sensible() {
        for block in [0, 1, 3] {
            let a = bootstrap_mean_ci(&X, 2000, 0.95, block, Some(7)).unwrap();
            let b = bootstrap_mean_ci(&X, 2000, 0.95, block, Some(7)).unwrap();
            assert_eq!(a.conf_int_lower, b.conf_int_lower);
            assert_eq!(a.conf_int_upper, b.conf_int_upper);
            assert!((a.estimate - 5.5).abs() < 1e-12);
            assert!(a.conf_int_lower < 5.5 && a.conf_int_upper > 5.5);
            assert!(a.se > 0.0);
        }
        // IID mean bootstrap: se ~ sd(x) * sqrt((n-1)/n) / sqrt(n) = 0.9083
        let r = bootstrap_mean_ci(&X, 20000, 0.95, 0, Some(1)).unwrap();
        assert!((r.se - 0.9083).abs() < 0.03, "{}", r.se);
        // matches the generic version with the mean
        let g = bootstrap_ci(
            &X,
            |s| s.iter().sum::<f64>() / s.len() as f64,
            2000,
            0.95,
            Some(7),
        )
        .unwrap();
        let m = bootstrap_mean_ci(&X, 2000, 0.95, 0, Some(7)).unwrap();
        assert_eq!(g.conf_int_lower, m.conf_int_lower);
        assert_eq!(g.conf_int_upper, m.conf_int_upper);
    }

    #[test]
    fn rejects_bad_input_and_tolerates_infinities() {
        assert!(bootstrap_mean_ci(&[1.0], 100, 0.95, 0, Some(1)).is_err());
        for nb in [0, 1] {
            assert!(bootstrap_mean_ci(&X, nb, 0.95, 0, Some(1)).is_err());
        }
        for cl in [0.0, 1.0, f64::NAN, 1.5] {
            assert!(bootstrap_mean_ci(&X, 100, cl, 0, Some(1)).is_err());
        }
        let d = [f64::INFINITY, f64::NEG_INFINITY, 1.0, 2.0];
        assert!(bootstrap_mean_ci(&d, 200, 0.95, 0, Some(7)).is_ok());
    }
}
