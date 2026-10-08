//! Percentile bootstrap confidence intervals.

use crate::error::{Result, StatError};
use crate::resampling::{CircularBlockBootstrap, StationaryBootstrap};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

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
            let sample: Vec<f64> = (0..n).map(|_| data[rng.gen_range(0..n)]).collect();
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

/// Percentile bootstrap confidence interval for the mean.
///
/// * `block_length == 0`: IID bootstrap (same as [`bootstrap_ci`] with the mean);
/// * `block_length == 1`: [`StationaryBootstrap`] with expected block length 1;
/// * `block_length > 1`: [`CircularBlockBootstrap`] with that block length
///   (for dependent data).
///
/// The interval and standard error are computed as in [`bootstrap_ci`].
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
    validate(data, n_bootstrap, conf_level)?;
    let mean = |s: &[f64]| s.iter().sum::<f64>() / s.len() as f64;
    let n = data.len();
    let reps: Vec<f64> = match block_length {
        0 => iid_replicates(data, &mean, n_bootstrap, seed),
        1 => StationaryBootstrap::new(1.0, seed)
            .samples(data, n, n_bootstrap)
            .iter()
            .map(|s| mean(s))
            .collect(),
        l => CircularBlockBootstrap::new(l, seed)
            .samples(data, n, n_bootstrap)
            .iter()
            .map(|s| mean(s))
            .collect(),
    };
    let (lo, hi, se) = percentile_interval(reps, conf_level);
    Ok(BootstrapCIResult {
        estimate: mean(data),
        se,
        conf_int_lower: lo,
        conf_int_upper: hi,
        conf_level,
        n_bootstrap,
    })
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
