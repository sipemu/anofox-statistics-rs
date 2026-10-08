//! McNemar's test for paired nominal data.

use crate::categorical::validate_2x2_table;
use crate::error::Result;
use statrs::distribution::{Binomial, ChiSquared, ContinuousCDF, DiscreteCDF};

/// Result of McNemar's test
#[derive(Debug, Clone)]
pub struct McNemarkResult {
    /// Chi-square test statistic
    pub statistic: f64,
    /// Degrees of freedom (always 1)
    pub df: f64,
    /// p-value
    pub p_value: f64,
    /// Whether continuity correction was applied
    pub corrected: bool,
    /// Name of the method
    pub method: String,
}

/// McNemar's test for paired nominal data.
///
/// Tests the null hypothesis that marginal homogeneity holds for
/// a 2x2 contingency table of paired observations.
///
/// # Table structure
/// ```text
///              | Time 2: +  | Time 2: -  |
/// -------------|------------|------------|
/// Time 1: +    |    a       |    b       |
/// Time 1: -    |    c       |    d       |
/// ```
///
/// The test examines whether b and c differ significantly.
///
/// # Arguments
/// * `table` - 2x2 contingency table [[a, b], [c, d]]
/// * `correction` - Apply Edwards' continuity correction
///
/// # Returns
/// * `McNemarkResult` containing test statistic, df, and p-value
///
/// # Examples
/// ```
/// use anofox_statistics::categorical::mcnemar_test;
///
/// // Before/after treatment: 10 stayed positive, 20 went from + to -,
/// // 5 went from - to +, 65 stayed negative
/// let table = [[10, 20], [5, 65]];
///
/// let result = mcnemar_test(&table, false).unwrap();
/// println!("Chi-square = {:.4}", result.statistic);
/// println!("p-value = {:.4}", result.p_value);
/// ```
///
/// # R equivalent
/// `mcnemar.test(matrix, correct = FALSE)`
pub fn mcnemar_test(table: &[[usize; 2]; 2], correction: bool) -> Result<McNemarkResult> {
    validate_2x2_table(table)?;

    let b = table[0][1] as f64;
    let c = table[1][0] as f64;

    let bc_sum = b + c;

    let statistic = if bc_sum == 0.0 {
        0.0
    } else if correction {
        // Edwards' continuity correction
        let diff = (b - c).abs() - 1.0;
        if diff <= 0.0 {
            0.0
        } else {
            diff * diff / bc_sum
        }
    } else {
        (b - c).powi(2) / bc_sum
    };

    // p-value from chi-square distribution with 1 df
    let df = 1.0;
    let p_value = if statistic > 0.0 {
        let chi_dist = ChiSquared::new(df).unwrap();
        chi_dist.sf(statistic)
    } else {
        1.0
    };

    let method = if correction {
        "McNemar's Chi-squared test with continuity correction"
    } else {
        "McNemar's Chi-squared test"
    };

    Ok(McNemarkResult {
        statistic,
        df,
        p_value,
        corrected: correction,
        method: method.to_string(),
    })
}

/// Result of an exact binomial test (McNemar's exact test)
#[derive(Debug, Clone)]
pub struct McNemarkExactResult {
    /// p-value
    pub p_value: f64,
    /// Number of discordant pairs with b > c
    pub b: usize,
    /// Number of discordant pairs with c > b
    pub c: usize,
    /// Name of the method
    pub method: String,
}

/// McNemar's exact test using binomial distribution.
///
/// More accurate than the chi-square approximation for small samples.
///
/// # Arguments
/// * `table` - 2x2 contingency table [[a, b], [c, d]]
///
/// # Returns
/// * `McNemarkExactResult` containing the exact p-value
///
/// # R equivalent
/// `binom.test(b, b + c, 0.5)` (equivalently `exact2x2::mcnemar.exact`)
pub fn mcnemar_exact(table: &[[usize; 2]; 2]) -> Result<McNemarkExactResult> {
    validate_2x2_table(table)?;

    let b = table[0][1];
    let c = table[1][0];
    let n = b + c;

    // Two-sided exact p-value, R's `binom.test(b, b + c, 0.5)`: the null
    // distribution is symmetric, so p = min(1, 2 * P(X <= min(b, c))) with
    // X ~ Binomial(n, 0.5). The CDF is the regularized incomplete beta
    // function, which neither overflows nor underflows for large n (the
    // former `exp(log C(n, k)) * 0.5^n` summation returned NaN for n > ~1074).
    let p_value = if n == 0 || b == c {
        1.0
    } else {
        let k = b.min(c) as u64;
        // p = 0.5 is always a valid success probability.
        let binom = Binomial::new(0.5, n as u64).expect("p = 0.5 is valid");
        (2.0 * binom.cdf(k)).clamp(0.0, 1.0)
    };

    Ok(McNemarkExactResult {
        p_value,
        b,
        c,
        method: "McNemar's Chi-squared test (exact)".to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mcnemar_basic() {
        // Example from many textbooks
        let table = [[59, 6], [16, 80]];

        let result = mcnemar_test(&table, false).unwrap();

        // Chi-square = (6-16)^2 / (6+16) = 100/22 ≈ 4.545
        assert!((result.statistic - 4.545454545).abs() < 0.001);
        assert!((result.df - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mcnemar_with_correction() {
        let table = [[59, 6], [16, 80]];

        let without = mcnemar_test(&table, false).unwrap();
        let with_correction = mcnemar_test(&table, true).unwrap();

        // Correction should reduce the statistic
        assert!(with_correction.statistic < without.statistic);
    }

    #[test]
    fn test_mcnemar_symmetric() {
        // When b = c, no significant difference
        let table = [[50, 10], [10, 30]];

        let result = mcnemar_test(&table, false).unwrap();

        assert!((result.statistic - 0.0).abs() < 1e-10);
        assert!((result.p_value - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mcnemar_exact() {
        let table = [[10, 3], [7, 80]];

        let result = mcnemar_exact(&table).unwrap();

        // With b=3, c=7, n=10, exact test
        assert!(result.p_value > 0.0 && result.p_value < 1.0);
    }

    #[test]
    fn test_mcnemar_exact_equal() {
        let table = [[50, 5], [5, 40]];

        let result = mcnemar_exact(&table).unwrap();

        // b = c = 5, should give p-value = 1.0
        assert!((result.p_value - 1.0).abs() < 1e-10);
    }
}
