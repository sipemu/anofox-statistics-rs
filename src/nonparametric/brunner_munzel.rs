use crate::error::{Result, StatError};
use crate::nonparametric::ranks::rank;
use crate::parametric::Alternative;
use statrs::distribution::{ContinuousCDF, StudentsT};

/// Result of the Brunner-Munzel test
#[derive(Debug, Clone)]
pub struct BrunnerMunzelResult {
    /// The test statistic
    pub statistic: f64,
    /// Degrees of freedom (Welch-Satterthwaite approximation)
    pub df: f64,
    /// The p-value
    pub p_value: f64,
    /// Estimated probability P(X < Y) + 0.5 * P(X = Y)
    pub estimate: f64,
}

/// Perform the Brunner-Munzel test for stochastic equality.
///
/// This is a robust alternative to the Mann-Whitney U test that handles
/// unequal variances and works well with ties. It tests whether
/// P(X < Y) + 0.5 * P(X = Y) = 0.5.
///
/// # Arguments
/// * `x` - First sample
/// * `y` - Second sample
/// * `alternative` - Alternative hypothesis
///
/// # Returns
/// * `BrunnerMunzelResult` containing statistic, df, p-value, and probability estimate
///
/// # References
/// * Brunner, E. and Munzel, U. (2000). "The Nonparametric Behrens-Fisher Problem:
///   Asymptotic Theory and a Small Sample Approximation"
pub fn brunner_munzel(
    x: &[f64],
    y: &[f64],
    alternative: Alternative,
) -> Result<BrunnerMunzelResult> {
    let n1 = x.len();
    let n2 = y.len();

    if n1 == 0 || n2 == 0 {
        return Err(StatError::EmptyData);
    }

    if n1 < 2 {
        return Err(StatError::InsufficientData { needed: 2, got: n1 });
    }

    if n2 < 2 {
        return Err(StatError::InsufficientData { needed: 2, got: n2 });
    }

    let n1_f = n1 as f64;
    let n2_f = n2 as f64;

    // Compute ranks within each sample
    let r1 = rank(x)?;
    let r2 = rank(y)?;

    // Combine samples and compute overall ranks
    let combined: Vec<f64> = x.iter().chain(y.iter()).cloned().collect();
    let r = rank(&combined)?;

    // Split overall ranks back into groups
    let r_x = &r[0..n1];
    let r_y = &r[n1..];

    // Mean ranks in combined sample
    let m1: f64 = r_x.iter().sum::<f64>() / n1_f;
    let m2: f64 = r_y.iter().sum::<f64>() / n2_f;

    // Probability estimate: P(X < Y) + 0.5 * P(X = Y)
    let pst = (m2 - (n2_f + 1.0) / 2.0) / n1_f;

    // Variance estimates
    // v1 = sum((r[1:n1] - r1 - m1 + (n1+1)/2)^2) / (n1-1)
    let v1: f64 = r_x
        .iter()
        .zip(r1.iter())
        .map(|(ri, r1i)| {
            let diff = ri - r1i - m1 + (n1_f + 1.0) / 2.0;
            diff * diff
        })
        .sum::<f64>()
        / (n1_f - 1.0);

    // v2 = sum((r[n1+1:n2] - r2 - m2 + (n2+1)/2)^2) / (n2-1)
    let v2: f64 = r_y
        .iter()
        .zip(r2.iter())
        .map(|(ri, r2i)| {
            let diff = ri - r2i - m2 + (n2_f + 1.0) / 2.0;
            diff * diff
        })
        .sum::<f64>()
        / (n2_f - 1.0);

    // Test statistic
    let n_total = n1_f + n2_f;
    let var_sum = n1_f * v1 + n2_f * v2;

    if var_sum < 1e-14 {
        return Err(StatError::InvalidParameter(
            "Variance is effectively zero".to_string(),
        ));
    }

    let statistic = n1_f * n2_f * (m2 - m1) / n_total / var_sum.sqrt();

    // Degrees of freedom (Welch-Satterthwaite approximation)
    let df =
        var_sum.powi(2) / ((n1_f * v1).powi(2) / (n1_f - 1.0) + (n2_f * v2).powi(2) / (n2_f - 1.0));

    // P-value from t-distribution
    let t_dist = StudentsT::new(0.0, 1.0, df).unwrap();

    let p_value = match alternative {
        Alternative::TwoSided => {
            2.0 * (1.0 - t_dist.cdf(statistic.abs())).min(t_dist.cdf(statistic.abs()))
        }
        Alternative::Greater => t_dist.cdf(statistic),
        Alternative::Less => 1.0 - t_dist.cdf(statistic),
    };

    Ok(BrunnerMunzelResult {
        statistic,
        df,
        p_value,
        estimate: pst,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_brunner_munzel_basic() {
        // Test data from R: lawstat::brunner.munzel.test
        let x = vec![
            1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 4.0, 1.0, 1.0,
        ];
        let y = vec![3.0, 3.0, 4.0, 3.0, 1.0, 2.0, 3.0, 1.0, 1.0, 5.0, 4.0];

        let result = brunner_munzel(&x, &y, Alternative::TwoSided).unwrap();

        // R reference values:
        // statistic = 3.1375, df = 17.683, p-value = 0.005786, estimate = 0.788961
        assert_relative_eq!(result.statistic, 3.1375, epsilon = 0.001);
        assert_relative_eq!(result.df, 17.683, epsilon = 0.01);
        assert_relative_eq!(result.p_value, 0.005786, epsilon = 0.0001);
        assert_relative_eq!(result.estimate, 0.788961, epsilon = 0.0001);
    }

    #[test]
    fn test_brunner_munzel_equal_samples() {
        // Equal samples should give statistic close to 0
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let result = brunner_munzel(&x, &y, Alternative::TwoSided).unwrap();

        // Statistic should be close to 0, estimate close to 0.5
        assert!(result.statistic.abs() < 0.5);
        assert_relative_eq!(result.estimate, 0.5, epsilon = 0.1);
    }

    #[test]
    fn test_brunner_munzel_empty() {
        let x: Vec<f64> = vec![];
        let y = vec![1.0, 2.0, 3.0];

        assert!(brunner_munzel(&x, &y, Alternative::TwoSided).is_err());
    }

    #[test]
    fn test_brunner_munzel_insufficient() {
        let x = vec![1.0];
        let y = vec![1.0, 2.0, 3.0];

        assert!(brunner_munzel(&x, &y, Alternative::TwoSided).is_err());
    }
}
