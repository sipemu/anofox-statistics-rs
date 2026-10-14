use crate::error::{Result, StatError};
use crate::utils::finite::ensure_finite;

/// Sample autocorrelations (ACF) or partial autocorrelations (PACF).
#[derive(Debug, Clone, PartialEq)]
pub struct AcfResult {
    /// Lags: `0..=max_lag` for the ACF, `1..=max_lag` for the PACF (as R).
    pub lags: Vec<usize>,
    /// (Partial) autocorrelation at each lag.
    pub values: Vec<f64>,
    /// Half-width of the white-noise band `qnorm(0.975) / sqrt(n)`
    /// (R's `plot.acf` default `ci = 0.95`). The band is `±conf_bound`.
    pub conf_bound: f64,
    /// Number of observations used.
    pub n: usize,
}

/// 95% white-noise bound `qnorm(0.975) / sqrt(n)`.
fn white_noise_bound(n: usize) -> f64 {
    // qnorm(0.975), to full double precision.
    const Z975: f64 = 1.959_963_984_540_054;
    Z975 / (n as f64).sqrt()
}

fn check_series(x: &[f64], max_lag: usize) -> Result<()> {
    if x.is_empty() {
        return Err(StatError::EmptyData);
    }
    ensure_finite("x", x)?;
    if x.len() < 2 {
        return Err(StatError::InsufficientData {
            needed: 2,
            got: x.len(),
        });
    }
    if max_lag >= x.len() {
        return Err(StatError::InvalidParameter(format!(
            "max_lag must be < n ({} >= {})",
            max_lag,
            x.len()
        )));
    }
    Ok(())
}

/// Autocorrelations at lags `0..=max_lag` of the demeaned series,
/// normalised by the lag-0 autocovariance (biased, divisor `n`).
pub(crate) fn acf_values(x: &[f64], max_lag: usize) -> Vec<f64> {
    let n = x.len();
    let mean = x.iter().sum::<f64>() / n as f64;
    let d: Vec<f64> = x.iter().map(|v| v - mean).collect();
    let c0: f64 = d.iter().map(|v| v * v).sum();
    (0..=max_lag)
        .map(|k| {
            if c0 == 0.0 {
                return f64::NAN;
            }
            let ck: f64 = (0..n - k).map(|t| d[t] * d[t + k]).sum();
            ck / c0
        })
        .collect()
}

/// Sample autocorrelation function, as R
/// `acf(x, lag.max = max_lag, type = "correlation", demean = TRUE)`.
///
/// `r_k = sum_{t=1}^{n-k} (x_t - m)(x_{t+k} - m) / sum_t (x_t - m)^2`.
/// A constant series gives NaN (R gives `NaN` as well).
///
/// # Errors
/// Empty input, `n < 2`, non-finite values, or `max_lag >= n`.
pub fn acf(x: &[f64], max_lag: usize) -> Result<AcfResult> {
    check_series(x, max_lag)?;
    Ok(AcfResult {
        lags: (0..=max_lag).collect(),
        values: acf_values(x, max_lag),
        conf_bound: white_noise_bound(x.len()),
        n: x.len(),
    })
}

/// Sample partial autocorrelation function, as R `pacf(x, lag.max = max_lag)`
/// (Durbin-Levinson recursion on the sample ACF). Lags start at 1.
///
/// # Errors
/// Empty input, `n < 2`, non-finite values, `max_lag == 0` or `max_lag >= n`.
pub fn pacf(x: &[f64], max_lag: usize) -> Result<AcfResult> {
    check_series(x, max_lag)?;
    if max_lag == 0 {
        return Err(StatError::InvalidParameter("max_lag must be >= 1".into()));
    }
    let r = acf_values(x, max_lag);
    let mut out = Vec::with_capacity(max_lag);
    let mut phi: Vec<f64> = Vec::new();
    for k in 1..=max_lag {
        let num = r[k] - (1..k).map(|j| phi[j - 1] * r[k - j]).sum::<f64>();
        let den = 1.0 - (1..k).map(|j| phi[j - 1] * r[j]).sum::<f64>();
        let pk = num / den;
        let mut next: Vec<f64> = (1..k).map(|j| phi[j - 1] - pk * phi[k - j - 1]).collect();
        next.push(pk);
        phi = next;
        out.push(pk);
    }
    Ok(AcfResult {
        lags: (1..=max_lag).collect(),
        values: out,
        conf_bound: white_noise_bound(x.len()),
        n: x.len(),
    })
}

/// Durbin-Watson statistic `sum (e_t - e_{t-1})^2 / sum e_t^2`, the
/// statistic reported by R `lmtest::dwtest`.
///
/// Only the statistic is returned: its p-value depends on the regression's
/// design matrix. Use [`durbin_watson_test`](super::durbin_watson_test) for
/// the p-value of `dwtest`.
///
/// # Errors
/// Fewer than 2 residuals, non-finite values, or all residuals zero.
pub fn durbin_watson(residuals: &[f64]) -> Result<f64> {
    if residuals.is_empty() {
        return Err(StatError::EmptyData);
    }
    if residuals.len() < 2 {
        return Err(StatError::InsufficientData {
            needed: 2,
            got: residuals.len(),
        });
    }
    ensure_finite("residuals", residuals)?;
    let ss: f64 = residuals.iter().map(|e| e * e).sum();
    if ss == 0.0 {
        return Err(StatError::InvalidParameter(
            "Durbin-Watson is undefined when all residuals are zero".into(),
        ));
    }
    let sd: f64 = residuals.windows(2).map(|w| (w[1] - w[0]).powi(2)).sum();
    Ok(sd / ss)
}
