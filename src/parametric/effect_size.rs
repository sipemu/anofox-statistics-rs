//! Standardised mean-difference effect sizes (Cohen's d) for t-tests.

use crate::error::{Result, StatError};
use crate::parametric::TTestKind;
use crate::utils::math::{mean, variance};

/// Divide by the standardiser, returning NaN when it is zero or not finite.
fn standardise(num: f64, sd: f64) -> f64 {
    if sd > 0.0 && sd.is_finite() {
        num / sd
    } else {
        f64::NAN
    }
}

/// Cohen's d for a two-sample or paired t-test, following the conventions of
/// R's `effectsize::cohens_d`.
///
/// * [`TTestKind::Student`]: `(mean(x) - mean(y) - mu) / s_pooled`, with
///   `s_pooled = sqrt(((n1-1) s1^2 + (n2-1) s2^2) / (n1 + n2 - 2))`
///   (`cohens_d(x, y, pooled_sd = TRUE)`).
/// * [`TTestKind::Welch`]: `(mean(x) - mean(y) - mu) / sqrt((s1^2 + s2^2) / 2)`,
///   the average-variance standardiser that does not assume equal variances
///   (`cohens_d(x, y, pooled_sd = FALSE)`).
/// * [`TTestKind::Paired`]: `d_z = mean(x - y - mu) / sd(x - y)`
///   (`cohens_d(x, y, paired = TRUE)`, or `cohens_d(x - y, mu = mu)`).
///
/// `mu` is the null value of the mean difference, as in [`t_test`](crate::t_test).
/// Returns `Ok(NaN)` when the standardiser is zero (e.g. constant samples).
///
/// # Errors
/// * fewer than 2 observations in a sample;
/// * samples of different length for [`TTestKind::Paired`].
///
/// # Examples
/// ```
/// use anofox_statistics::{cohens_d, TTestKind};
///
/// let x = [5.1, 4.9, 6.2, 5.8, 6.05, 5.5, 5.3, 6.1];
/// let y = [6.5, 7.1, 6.8, 7.4, 6.0, 7.9, 6.6, 7.2, 6.9, 7.05];
/// let d = cohens_d(&x, &y, TTestKind::Student, 0.0).unwrap();
/// assert!((d - -2.606655378290357).abs() < 1e-12);
/// ```
pub fn cohens_d(x: &[f64], y: &[f64], kind: TTestKind, mu: f64) -> Result<f64> {
    if x.len() < 2 {
        return Err(StatError::InsufficientData {
            needed: 2,
            got: x.len(),
        });
    }
    if y.len() < 2 {
        return Err(StatError::InsufficientData {
            needed: 2,
            got: y.len(),
        });
    }
    match kind {
        TTestKind::Paired => {
            if x.len() != y.len() {
                return Err(StatError::InvalidParameter(format!(
                    "paired samples must have equal length, got {} and {}",
                    x.len(),
                    y.len()
                )));
            }
            let d: Vec<f64> = x.iter().zip(y).map(|(a, b)| a - b).collect();
            cohens_d_one_sample(&d, mu)
        }
        TTestKind::Student => {
            let (m1, v1) = (mean(x)?, variance(x)?);
            let (m2, v2) = (mean(y)?, variance(y)?);
            let (n1, n2) = (x.len() as f64, y.len() as f64);
            let pooled = ((n1 - 1.0) * v1 + (n2 - 1.0) * v2) / (n1 + n2 - 2.0);
            Ok(standardise(m1 - m2 - mu, pooled.sqrt()))
        }
        TTestKind::Welch => {
            let (m1, v1) = (mean(x)?, variance(x)?);
            let (m2, v2) = (mean(y)?, variance(y)?);
            Ok(standardise(m1 - m2 - mu, ((v1 + v2) / 2.0).sqrt()))
        }
    }
}

/// Cohen's d for a one-sample t-test: `(mean(x) - mu) / sd(x)`
/// (R `effectsize::cohens_d(x, mu = mu)`).
///
/// Returns `Ok(NaN)` when `sd(x)` is zero.
///
/// # Errors
/// Fewer than 2 observations.
pub fn cohens_d_one_sample(x: &[f64], mu: f64) -> Result<f64> {
    if x.len() < 2 {
        return Err(StatError::InsufficientData {
            needed: 2,
            got: x.len(),
        });
    }
    Ok(standardise(mean(x)? - mu, variance(x)?.sqrt()))
}

#[cfg(test)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;

    const X: [f64; 8] = [5.1, 4.9, 6.2, 5.8, 6.05, 5.5, 5.3, 6.1];
    const Y: [f64; 10] = [6.5, 7.1, 6.8, 7.4, 6.0, 7.9, 6.6, 7.2, 6.9, 7.05];

    #[test]
    fn student_matches_r_effectsize() {
        // R: effectsize::cohens_d(x, y)  (pooled SD)
        let d = cohens_d(&X, &Y, TTestKind::Student, 0.0).unwrap();
        assert!((d - -2.606655378290357).abs() < 1e-12, "{d}");
    }

    #[test]
    fn welch_uses_average_variance() {
        // R: (mean(x) - mean(y)) / sqrt((var(x) + var(y)) / 2)
        //    == effectsize::cohens_d(x, y, pooled_sd = FALSE)
        let d = cohens_d(&X, &Y, TTestKind::Welch, 0.0).unwrap();
        assert!((d - -2.6164932281884563).abs() < 1e-12, "{d}");
    }

    #[test]
    fn paired_is_d_z() {
        // R: d <- x - y[1:8]; mean(d) / sd(d)
        let d = cohens_d(&X, &Y[..8], TTestKind::Paired, 0.0).unwrap();
        assert!((d - -1.6503380487140236).abs() < 1e-12, "{d}");
        // mu shifts the numerator only
        let d = cohens_d(&X, &Y[..8], TTestKind::Paired, -1.0).unwrap();
        assert!((d - -0.3988968743337214).abs() < 1e-12, "{d}");
        assert!(cohens_d(&X, &Y, TTestKind::Paired, 0.0).is_err());
    }

    #[test]
    fn one_sample_and_degenerate() {
        // R: (mean(x) - 5) / sd(x)
        let d = cohens_d_one_sample(&X, 5.0).unwrap();
        assert!((d - 1.2593923076823721).abs() < 1e-12, "{d}");
        assert!(cohens_d_one_sample(&[3.0; 5], 0.0).unwrap().is_nan());
        assert!(cohens_d(&[1.0; 4], &[1.0; 4], TTestKind::Student, 0.0)
            .unwrap()
            .is_nan());
        assert!(cohens_d(&[1.0], &Y, TTestKind::Welch, 0.0).is_err());
    }
}
