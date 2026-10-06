//! Fisher's exact test for 2x2 contingency tables.

use crate::categorical::{validate_2x2_table, Alternative};
use crate::error::Result;
use statrs::distribution::{ContinuousCDF, Normal};

/// Result of Fisher's exact test
#[derive(Debug, Clone)]
pub struct FisherResult {
    /// p-value
    pub p_value: f64,
    /// Odds ratio estimate
    pub odds_ratio: f64,
    /// 95% confidence interval for odds ratio (lower bound)
    pub conf_int_lower: f64,
    /// 95% confidence interval for odds ratio (upper bound)
    pub conf_int_upper: f64,
    /// Alternative hypothesis used
    pub alternative: Alternative,
    /// Name of the method
    pub method: String,
}

/// Fisher's exact test for 2x2 contingency tables.
///
/// Computes exact p-values using the hypergeometric distribution.
/// This test is exact and does not require large sample approximations.
///
/// # Arguments
/// * `table` - 2x2 contingency table [[a, b], [c, d]]
/// * `alternative` - Alternative hypothesis (two-sided, greater, or less)
///
/// # Returns
/// * `FisherResult` containing the p-value and odds ratio
///
/// # Table structure
/// ```text
///              | Success | Failure |
/// -------------|---------|---------|
/// Group 1      |    a    |    b    |
/// Group 2      |    c    |    d    |
/// ```
///
/// # Examples
/// ```
/// use anofox_statistics::categorical::{fisher_exact, Alternative};
///
/// let table = [[3, 1], [1, 3]];
///
/// let result = fisher_exact(&table, Alternative::TwoSided).unwrap();
/// println!("p-value = {:.4}", result.p_value);
/// println!("Odds ratio = {:.4}", result.odds_ratio);
/// ```
///
/// # R equivalent
/// `fisher.test(matrix(c(a, c, b, d), nrow=2))`
pub fn fisher_exact(table: &[[usize; 2]; 2], alternative: Alternative) -> Result<FisherResult> {
    fisher_exact_with_conf_level(table, alternative, 0.95)
}

/// Fisher's exact test with a configurable confidence level for the
/// (Woolf / logit) odds-ratio interval: `exp(log(OR) -/+ qnorm((1 + conf_level) / 2) * se)`.
pub fn fisher_exact_with_conf_level(
    table: &[[usize; 2]; 2],
    alternative: Alternative,
    conf_level: f64,
) -> Result<FisherResult> {
    validate_2x2_table(table)?;
    if !(conf_level.is_finite() && conf_level > 0.0 && conf_level < 1.0) {
        return Err(crate::error::StatError::InvalidParameter(format!(
            "conf_level must be in (0, 1), got {}",
            conf_level
        )));
    }

    let a = table[0][0];
    let b = table[0][1];
    let c = table[1][0];
    let d = table[1][1];

    // Marginals
    let row1 = a + b;
    let _row2 = c + d;
    let col1 = a + c;
    let col2 = b + d;
    let n = a + b + c + d;

    // Compute odds ratio
    let odds_ratio = if b == 0 || c == 0 {
        if a == 0 || d == 0 {
            f64::NAN
        } else {
            f64::INFINITY
        }
    } else if a == 0 || d == 0 {
        0.0
    } else {
        (a as f64 * d as f64) / (b as f64 * c as f64)
    };

    // Compute p-value using hypergeometric distribution
    // P(X = k) = C(K, k) * C(N-K, n-k) / C(N, n)
    // where K = col1, N = n, n = row1
    let observed_prob = hypergeometric_pmf(a, col1, n - col1, row1);

    let p_value = match alternative {
        Alternative::TwoSided => {
            // Sum probabilities of all tables no more likely than the observed
            // one, with R's relative tolerance (an absolute tolerance floors the
            // p-value for large tables).
            let rel_err = 1.0 + 1e-7;
            let min_a = row1.saturating_sub(col2);
            let max_a = row1.min(col1);

            let mut p = 0.0;
            for k in min_a..=max_a {
                let prob = hypergeometric_pmf(k, col1, n - col1, row1);
                if prob <= observed_prob * rel_err {
                    p += prob;
                }
            }
            p.min(1.0)
        }
        Alternative::Greater => {
            // P(X >= a)
            let max_a = row1.min(col1);
            let mut p = 0.0;
            for k in a..=max_a {
                p += hypergeometric_pmf(k, col1, n - col1, row1);
            }
            p.min(1.0)
        }
        Alternative::Less => {
            // P(X <= a)
            let min_a = row1.saturating_sub(col2);
            let mut p = 0.0;
            for k in min_a..=a {
                p += hypergeometric_pmf(k, col1, n - col1, row1);
            }
            p.min(1.0)
        }
    };

    // Compute confidence interval for odds ratio
    // Using the mid-p exact method approximation
    let (conf_int_lower, conf_int_upper) = odds_ratio_ci(a, b, c, d, conf_level);

    Ok(FisherResult {
        p_value,
        odds_ratio,
        conf_int_lower,
        conf_int_upper,
        alternative,
        method: "Fisher's Exact Test for Count Data".to_string(),
    })
}

/// Compute hypergeometric PMF: P(X = k)
/// where X ~ Hypergeom(N, K, n)
/// N = population size, K = success states in population, n = draws
fn hypergeometric_pmf(k: usize, big_k: usize, big_n_minus_k: usize, n: usize) -> f64 {
    // P(X = k) = C(K, k) * C(N-K, n-k) / C(N, n)
    // Use log-space to avoid overflow

    let n_minus_k = if n >= k { n - k } else { return 0.0 };

    // Check bounds
    if k > big_k || n_minus_k > big_n_minus_k {
        return 0.0;
    }

    let big_n = big_k + big_n_minus_k;

    let log_prob = log_binomial_coeff(big_k, k) + log_binomial_coeff(big_n_minus_k, n_minus_k)
        - log_binomial_coeff(big_n, n);

    log_prob.exp()
}

/// Compute log of binomial coefficient C(n, k)
fn log_binomial_coeff(n: usize, k: usize) -> f64 {
    if k > n {
        return f64::NEG_INFINITY;
    }
    if k == 0 || k == n {
        return 0.0;
    }

    // Use log-gamma: log(C(n,k)) = log(n!) - log(k!) - log((n-k)!)
    // = lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1)
    log_factorial(n) - log_factorial(k) - log_factorial(n - k)
}

/// Compute log(n!) (statrs: exact table for n < 255, accurate ln-gamma above)
fn log_factorial(n: usize) -> f64 {
    statrs::function::factorial::ln_factorial(n as u64)
}

/// Compute confidence interval for odds ratio using Woolf's method (log method)
fn odds_ratio_ci(a: usize, b: usize, c: usize, d: usize, conf_level: f64) -> (f64, f64) {
    let z = Normal::new(0.0, 1.0)
        .unwrap()
        .inverse_cdf((1.0 + conf_level) / 2.0);
    // Handle zero cells
    if a == 0 || b == 0 || c == 0 || d == 0 {
        // Add 0.5 to each cell (Haldane-Anscombe correction)
        let a_adj = a as f64 + 0.5;
        let b_adj = b as f64 + 0.5;
        let c_adj = c as f64 + 0.5;
        let d_adj = d as f64 + 0.5;

        let log_or = (a_adj * d_adj).ln() - (b_adj * c_adj).ln();
        let se_log_or = (1.0 / a_adj + 1.0 / b_adj + 1.0 / c_adj + 1.0 / d_adj).sqrt();

        let lower = (log_or - z * se_log_or).exp();
        let upper = (log_or + z * se_log_or).exp();

        return (lower, upper);
    }

    let a_f = a as f64;
    let b_f = b as f64;
    let c_f = c as f64;
    let d_f = d as f64;

    let log_or = (a_f * d_f).ln() - (b_f * c_f).ln();
    let se_log_or = (1.0 / a_f + 1.0 / b_f + 1.0 / c_f + 1.0 / d_f).sqrt();

    let lower = (log_or - z * se_log_or).exp();
    let upper = (log_or + z * se_log_or).exp();

    (lower, upper)
}

#[cfg(test)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;

    #[test]
    fn test_fisher_exact_two_sided() {
        // Classic tea-tasting example
        let table = [[3, 1], [1, 3]];

        let result = fisher_exact(&table, Alternative::TwoSided).unwrap();

        // Known p-value for this table is approximately 0.486
        assert!(result.p_value > 0.4 && result.p_value < 0.6);
        assert!((result.odds_ratio - 9.0).abs() < 1e-10); // (3*3)/(1*1) = 9
    }

    #[test]
    fn test_fisher_exact_one_sided_greater() {
        let table = [[4, 0], [0, 4]];

        let result = fisher_exact(&table, Alternative::Greater).unwrap();

        // Very strong association
        assert!(result.p_value < 0.05);
    }

    #[test]
    fn test_fisher_exact_one_sided_less() {
        let table = [[0, 4], [4, 0]];

        let result = fisher_exact(&table, Alternative::Less).unwrap();

        // Very strong negative association
        assert!(result.p_value < 0.05);
    }

    #[test]
    fn test_fisher_exact_no_association() {
        // Table with perfect independence
        let table = [[10, 10], [10, 10]];

        let result = fisher_exact(&table, Alternative::TwoSided).unwrap();

        assert!(result.p_value > 0.9);
        assert!((result.odds_ratio - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_hypergeometric_pmf() {
        // Simple test: drawing from an urn
        // N=10, K=5, n=4
        let prob = hypergeometric_pmf(2, 5, 5, 4);
        // P(X=2) when drawing 4 from 10 with 5 successes
        // = C(5,2) * C(5,2) / C(10,4) = 10*10/210 ≈ 0.476
        assert!((prob - 0.476).abs() < 0.01);
    }

    #[test]
    fn test_log_binomial_coeff() {
        // C(10, 4) = 210
        let log_c = log_binomial_coeff(10, 4);
        assert!((log_c.exp() - 210.0).abs() < 0.001);

        // C(5, 0) = 1
        let log_c0 = log_binomial_coeff(5, 0);
        assert!((log_c0.exp() - 1.0).abs() < 1e-10);

        // C(5, 5) = 1
        let log_c5 = log_binomial_coeff(5, 5);
        assert!((log_c5.exp() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_fisher_conf_level_and_large_table_match_r() {
        // Woolf interval at 90%: exp(log(OR) -/+ qnorm(0.95) * se)
        let r =
            fisher_exact_with_conf_level(&[[18, 7], [9, 16]], Alternative::TwoSided, 0.9).unwrap();
        assert!((r.conf_int_lower - 1.6762644393608406).abs() < 1e-10);
        assert!((r.conf_int_upper - 12.466982352523004).abs() < 1e-9);
        // R: fisher.test(matrix(c(300, 200, 250, 260), 2))$p.value
        let r = fisher_exact(&[[300, 250], [200, 260]], Alternative::TwoSided).unwrap();
        assert!(
            (r.p_value / 0.00050817575098994735 - 1.0).abs() < 1e-8,
            "{}",
            r.p_value
        );
    }
}
