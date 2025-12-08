use crate::error::{Result, StatError};
use statrs::distribution::{ContinuousCDF, Normal};

/// Loss function for Diebold-Mariano test
#[derive(Debug, Clone, Copy)]
pub enum LossFunction {
    /// Squared error loss: (e)^2
    SquaredError,
    /// Absolute error loss: |e|
    AbsoluteError,
}

/// Result of Diebold-Mariano test
#[derive(Debug, Clone)]
pub struct DMResult {
    /// The DM test statistic
    pub statistic: f64,
    /// The p-value (two-sided)
    pub p_value: f64,
}

/// Perform the Diebold-Mariano test for comparing forecast accuracy.
///
/// Tests the null hypothesis that two forecasts have equal predictive accuracy.
/// Implementation matches R's forecast::dm.test function.
///
/// # Arguments
/// * `e1` - Forecast errors from model 1
/// * `e2` - Forecast errors from model 2
/// * `loss` - Loss function to use (SquaredError or AbsoluteError)
/// * `h` - Forecast horizon (for variance adjustment, h=1 for 1-step ahead)
///
/// # Returns
/// * `DMResult` containing test statistic and p-value
///
/// # References
/// * Diebold, F.X. and Mariano, R.S. (1995) "Comparing Predictive Accuracy"
pub fn diebold_mariano(e1: &[f64], e2: &[f64], loss: LossFunction, h: usize) -> Result<DMResult> {
    let n = e1.len();

    if n == 0 {
        return Err(StatError::EmptyData);
    }

    if e1.len() != e2.len() {
        return Err(StatError::InvalidParameter(
            "Forecast error vectors must have equal length".to_string(),
        ));
    }

    if n < 3 {
        return Err(StatError::InsufficientData { needed: 3, got: n });
    }

    // Compute loss differentials: d_t = g(e1_t) - g(e2_t)
    let d: Vec<f64> = e1
        .iter()
        .zip(e2.iter())
        .map(|(e1_t, e2_t)| {
            let l1 = apply_loss(*e1_t, loss);
            let l2 = apply_loss(*e2_t, loss);
            l1 - l2
        })
        .collect();

    let n_f = n as f64;

    // Mean of loss differentials
    let d_bar: f64 = d.iter().sum::<f64>() / n_f;

    // Compute variance using R's approach:
    // d.cov <- acf(d, lag.max=h-1, type="covariance")$acf
    // d.var <- sum(c(d.cov[1], 2*d.cov[-1])) / n
    let var_d_bar = dm_variance(&d, h);

    // Raw DM statistic
    let dm_raw = if var_d_bar > 1e-30 {
        d_bar / var_d_bar.sqrt()
    } else {
        0.0
    };

    // Apply Harvey, Leybourne, and Newbold (1997) small-sample correction
    // This improves the finite-sample properties of the test
    let h_f = h as f64;
    let correction = ((n_f + 1.0 - 2.0 * h_f + h_f * (h_f - 1.0) / n_f) / n_f).sqrt();
    let dm_stat = dm_raw * correction;

    // Two-sided p-value from standard normal
    let normal = Normal::new(0.0, 1.0).unwrap();
    let p_value = 2.0 * (1.0 - normal.cdf(dm_stat.abs()));

    Ok(DMResult {
        statistic: dm_stat,
        p_value,
    })
}

/// Apply loss function to forecast error
fn apply_loss(e: f64, loss: LossFunction) -> f64 {
    match loss {
        LossFunction::SquaredError => e * e,
        LossFunction::AbsoluteError => e.abs(),
    }
}

/// Compute variance of the mean for DM test using R's approach
/// This matches forecast::dm.test variance calculation
fn dm_variance(d: &[f64], h: usize) -> f64 {
    let n = d.len();
    let n_f = n as f64;

    // Mean of d
    let d_bar: f64 = d.iter().sum::<f64>() / n_f;

    // Centered series
    let d_centered: Vec<f64> = d.iter().map(|x| x - d_bar).collect();

    // Compute autocovariances (matching R's acf with type="covariance")
    // R's acf uses division by n (not n-k)
    let max_lag = if h > 0 { h - 1 } else { 0 };

    // Lag 0 autocovariance (variance * n / n = variance)
    let gamma_0: f64 = d_centered.iter().map(|x| x * x).sum::<f64>() / n_f;

    // Sum autocovariances: gamma_0 + 2 * sum(gamma_k for k=1..h-1)
    let mut acov_sum = gamma_0;

    for k in 1..=max_lag {
        if k >= n {
            break;
        }
        // Autocovariance at lag k (R's style: divide by n)
        let gamma_k: f64 = d_centered
            .iter()
            .skip(k)
            .zip(d_centered.iter())
            .map(|(d_t, d_t_k)| d_t * d_t_k)
            .sum::<f64>()
            / n_f;

        acov_sum += 2.0 * gamma_k;
    }

    // Variance of the mean = sum(autocovariances) / n
    acov_sum / n_f
}
