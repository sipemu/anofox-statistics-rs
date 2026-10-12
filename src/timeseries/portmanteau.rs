use super::autocorrelation::acf;
use crate::error::{Result, StatError};
use crate::utils::dist::chisq_sf;

/// Result of a Ljung-Box or Box-Pierce portmanteau test.
#[derive(Debug, Clone, PartialEq)]
pub struct PortmanteauResult {
    /// `"Box-Ljung test"` or `"Box-Pierce test"` (R's method strings).
    pub method: String,
    /// The Q statistic.
    pub statistic: f64,
    /// Degrees of freedom `lag - fitdf`.
    pub df: f64,
    /// Upper-tail chi-squared p-value.
    pub p_value: f64,
    /// Number of autocorrelations used.
    pub lag: usize,
    /// Degrees of freedom subtracted for fitted parameters.
    pub fitdf: usize,
    /// Number of observations.
    pub n: usize,
}

fn portmanteau(x: &[f64], lag: usize, fitdf: usize, ljung: bool) -> Result<PortmanteauResult> {
    if lag == 0 {
        return Err(StatError::InvalidParameter("lag must be >= 1".into()));
    }
    if fitdf >= lag {
        return Err(StatError::InvalidParameter(format!(
            "fitdf must be < lag ({} >= {})",
            fitdf, lag
        )));
    }
    let r = acf(x, lag)?;
    let n = x.len() as f64;
    let statistic = if ljung {
        n * (n + 2.0)
            * (1..=lag)
                .map(|k| r.values[k].powi(2) / (n - k as f64))
                .sum::<f64>()
    } else {
        n * (1..=lag).map(|k| r.values[k].powi(2)).sum::<f64>()
    };
    let df = (lag - fitdf) as f64;
    Ok(PortmanteauResult {
        method: if ljung {
            "Box-Ljung test"
        } else {
            "Box-Pierce test"
        }
        .to_string(),
        statistic,
        df,
        p_value: chisq_sf(statistic, df),
        lag,
        fitdf,
        n: x.len(),
    })
}

/// Ljung-Box test, as R `Box.test(x, lag, type = "Ljung-Box", fitdf)`:
/// `Q = n (n + 2) sum_{k=1}^{lag} r_k^2 / (n - k)`, chi-squared with
/// `lag - fitdf` degrees of freedom.
///
/// # Errors
/// `lag == 0`, `fitdf >= lag`, `lag >= n`, or invalid input (see [`acf`]).
pub fn ljung_box(x: &[f64], lag: usize, fitdf: usize) -> Result<PortmanteauResult> {
    portmanteau(x, lag, fitdf, true)
}

/// Box-Pierce test, as R `Box.test(x, lag, type = "Box-Pierce", fitdf)`:
/// `Q = n sum_{k=1}^{lag} r_k^2`.
///
/// # Errors
/// As [`ljung_box`].
pub fn box_pierce(x: &[f64], lag: usize, fitdf: usize) -> Result<PortmanteauResult> {
    portmanteau(x, lag, fitdf, false)
}
