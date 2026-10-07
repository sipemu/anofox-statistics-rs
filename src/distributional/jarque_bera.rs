//! Jarque-Bera test for normality.

use crate::error::{Result, StatError};

/// Result of the Jarque-Bera test
#[derive(Debug, Clone)]
pub struct JarqueBeraResult {
    /// JB statistic `n/6 * (S^2 + K^2/4)`
    pub statistic: f64,
    /// p-value from the chi-squared distribution with 2 degrees of freedom
    pub p_value: f64,
    /// Sample skewness `S = m3 / m2^(3/2)` (population moments)
    pub skewness: f64,
    /// Sample excess kurtosis `K = m4 / m2^2 - 3` (population moments)
    pub kurtosis: f64,
    /// Number of observations used
    pub n: usize,
}

/// Jarque-Bera test for normality.
///
/// Uses the biased (population) central moments `m_k = sum((x - mean)^k) / n`,
/// `JB = n/6 * (S^2 + K^2/4)` with `S = m3 / m2^(3/2)` and
/// `K = m4 / m2^2 - 3`, and the asymptotic `chi^2(2)` p-value
/// `P(chi^2_2 > JB) = exp(-JB / 2)`.
///
/// # Errors
/// * fewer than 3 observations;
/// * non-finite values;
/// * zero variance (constant data).
///
/// # Examples
/// ```
/// use anofox_statistics::jarque_bera;
///
/// let x = [2.1, 3.4, 1.9, 5.6, 4.4, 3.3, 2.8, 9.7, 3.0, 4.1];
/// let r = jarque_bera(&x).unwrap();
/// assert!(r.p_value > 0.0 && r.p_value < 1.0);
/// ```
///
/// # R equivalent
/// `tseries::jarque.bera.test(x)` (also `moments::jarque.test(x)`)
pub fn jarque_bera(data: &[f64]) -> Result<JarqueBeraResult> {
    let n = data.len();
    if n < 3 {
        return Err(StatError::InsufficientData { needed: 3, got: n });
    }
    if data.iter().any(|v| !v.is_finite()) {
        return Err(StatError::InvalidParameter(
            "data contains non-finite values".to_string(),
        ));
    }
    let nf = n as f64;
    let mean = data.iter().sum::<f64>() / nf;
    let (mut m2, mut m3, mut m4) = (0.0, 0.0, 0.0);
    for &x in data {
        let d = x - mean;
        let d2 = d * d;
        m2 += d2;
        m3 += d2 * d;
        m4 += d2 * d2;
    }
    m2 /= nf;
    m3 /= nf;
    m4 /= nf;
    if m2 <= 0.0 {
        return Err(StatError::InvalidParameter(
            "data has zero variance".to_string(),
        ));
    }
    let skewness = m3 / m2.powf(1.5);
    let kurtosis = m4 / (m2 * m2) - 3.0;
    let statistic = nf / 6.0 * (skewness * skewness + kurtosis * kurtosis / 4.0);
    // Survival function of chi^2 with 2 df is exactly exp(-x/2).
    let p_value = (-statistic / 2.0).exp();
    Ok(JarqueBeraResult {
        statistic,
        p_value,
        skewness,
        kurtosis,
        n,
    })
}

#[cfg(test)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;

    #[test]
    fn matches_r_tseries() {
        // R: tseries::jarque.bera.test(x)
        let x = [2.1, 3.4, 1.9, 5.6, 4.4, 3.3, 2.8, 9.7, 3.0, 4.1];
        let r = jarque_bera(&x).unwrap();
        assert!((r.statistic - JB_X).abs() < 1e-12, "{}", r.statistic);
        assert!((r.p_value - P_X).abs() < 1e-14, "{}", r.p_value);
        assert_eq!(r.n, 10);
    }

    const JB_X: f64 = 6.1833442919626558;
    const P_X: f64 = 0.045425932072353667;

    #[test]
    fn errors() {
        assert!(jarque_bera(&[1.0, 2.0]).is_err());
        assert!(jarque_bera(&[3.0; 10]).is_err());
        assert!(jarque_bera(&[1.0, 2.0, f64::NAN, 4.0]).is_err());
    }
}
