//! Regression metrics, as scikit-learn's `sklearn.metrics` (and
//! `yardstick::smape` for the symmetric MAPE).

use crate::error::{Result, StatError};
use crate::utils::finite::ensure_finite;

/// Regression metrics returned by [`regression_metrics`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RegressionMetrics {
    /// Number of rows.
    pub n: usize,
    /// Mean absolute error.
    pub mae: f64,
    /// Mean squared error.
    pub mse: f64,
    /// Root mean squared error.
    pub rmse: f64,
    /// Mean absolute percentage error as a fraction (not in percent):
    /// `mean(|y - yhat| / max(|y|, eps))` with `eps` the machine epsilon, as
    /// scikit-learn (a zero `y` gives a huge but finite term).
    pub mape: f64,
    /// Symmetric MAPE as a fraction: `mean(|y - yhat| / ((|y| + |yhat|) / 2))`,
    /// a row with `y = yhat = 0` contributing 0 (`yardstick::smape / 100`,
    /// which gives NaN for that row).
    pub smape: f64,
    /// Median absolute error.
    pub median_ae: f64,
    /// Coefficient of determination `1 - SS_res / SS_tot`; for constant `y`
    /// 1 if the predictions are perfect and 0 otherwise (scikit-learn's
    /// `force_finite`); NaN for a single row.
    pub r2: f64,
    /// `1 - var(y - yhat) / var(y)`, constant `y` handled as for `r2`.
    pub explained_variance: f64,
    /// Largest absolute error.
    pub max_error: f64,
}

/// Regression metrics of predictions `y_pred` against `y_true`.
///
/// Errors on empty or mismatched inputs and non-finite values.
///
/// Python equivalent: `mean_absolute_error`, `mean_squared_error`,
/// `root_mean_squared_error`, `mean_absolute_percentage_error`,
/// `median_absolute_error`, `r2_score`, `explained_variance_score`,
/// `max_error`.
///
/// # Example
/// ```
/// use anofox_statistics::metrics::regression_metrics;
/// let m = regression_metrics(&[3.0, -0.5, 2.0, 7.0], &[2.5, 0.0, 2.0, 8.0]).unwrap();
/// assert!((m.mae - 0.5).abs() < 1e-12);
/// assert!((m.r2 - 0.9486081370449679).abs() < 1e-12);
/// ```
pub fn regression_metrics(y_true: &[f64], y_pred: &[f64]) -> Result<RegressionMetrics> {
    if y_true.len() != y_pred.len() {
        return Err(StatError::InvalidParameter(format!(
            "y_true ({}) and y_pred ({}) must have the same length",
            y_true.len(),
            y_pred.len()
        )));
    }
    if y_true.is_empty() {
        return Err(StatError::EmptyData);
    }
    ensure_finite("y_true", y_true)?;
    ensure_finite("y_pred", y_pred)?;
    let n = y_true.len();
    let nf = n as f64;
    let err: Vec<f64> = y_true.iter().zip(y_pred).map(|(t, p)| t - p).collect();
    let mut abs: Vec<f64> = err.iter().map(|e| e.abs()).collect();
    let mae = abs.iter().sum::<f64>() / nf;
    let mse = err.iter().map(|e| e * e).sum::<f64>() / nf;
    let mape = y_true
        .iter()
        .zip(&abs)
        .map(|(t, a)| a / t.abs().max(f64::EPSILON))
        .sum::<f64>()
        / nf;
    let smape = y_true
        .iter()
        .zip(y_pred)
        .zip(&abs)
        .map(|((t, p), a)| {
            let d = (t.abs() + p.abs()) / 2.0;
            if d > 0.0 {
                a / d
            } else {
                0.0
            }
        })
        .sum::<f64>()
        / nf;
    let max_error = abs.iter().copied().fold(0.0, f64::max);
    abs.sort_by(|a, b| a.total_cmp(b));
    let median_ae = if n % 2 == 1 {
        abs[n / 2]
    } else {
        (abs[n / 2 - 1] + abs[n / 2]) / 2.0
    };
    let my = y_true.iter().sum::<f64>() / nf;
    let ss_tot: f64 = y_true.iter().map(|t| (t - my) * (t - my)).sum();
    let ss_res: f64 = err.iter().map(|e| e * e).sum();
    let me = err.iter().sum::<f64>() / nf;
    let ss_err_c: f64 = err.iter().map(|e| (e - me) * (e - me)).sum();
    let finite_ratio = |num: f64, den: f64| {
        if den != 0.0 {
            1.0 - num / den
        } else if num == 0.0 {
            1.0
        } else {
            0.0
        }
    };
    let (r2, explained_variance) = if n < 2 {
        (f64::NAN, f64::NAN)
    } else {
        (finite_ratio(ss_res, ss_tot), finite_ratio(ss_err_c, ss_tot))
    };
    Ok(RegressionMetrics {
        n,
        mae,
        mse,
        rmse: mse.sqrt(),
        mape,
        smape,
        median_ae,
        r2,
        explained_variance,
        max_error,
    })
}
