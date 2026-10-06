//! Proportion tests and binomial tests.

use crate::error::{Result, StatError};
use crate::parametric::Alternative;
use statrs::distribution::{Beta, Binomial, ContinuousCDF, Discrete, DiscreteCDF, Normal};

/// Result of a proportion test
#[derive(Debug, Clone)]
pub struct PropTestResult {
    /// Estimated proportion(s)
    pub estimate: Vec<f64>,
    /// Test statistic (chi-square or z)
    pub statistic: f64,
    /// Degrees of freedom (for chi-square test)
    pub df: Option<f64>,
    /// p-value
    pub p_value: f64,
    /// 95% confidence interval lower bound
    pub conf_int_lower: f64,
    /// 95% confidence interval upper bound
    pub conf_int_upper: f64,
    /// Null hypothesis proportion
    pub null_value: f64,
    /// Alternative hypothesis
    pub alternative: Alternative,
    /// Name of the method
    pub method: String,
}

/// Result of an exact binomial test
#[derive(Debug, Clone)]
pub struct BinomTestResult {
    /// Estimated proportion
    pub estimate: f64,
    /// Number of successes
    pub successes: usize,
    /// Number of trials
    pub n: usize,
    /// p-value
    pub p_value: f64,
    /// 95% confidence interval lower bound (Clopper-Pearson)
    pub conf_int_lower: f64,
    /// 95% confidence interval upper bound (Clopper-Pearson)
    pub conf_int_upper: f64,
    /// Null hypothesis proportion
    pub null_value: f64,
    /// Alternative hypothesis
    pub alternative: Alternative,
    /// Name of the method
    pub method: String,
}

/// One-sample proportion test (z-test approximation).
///
/// Tests the null hypothesis that the true proportion equals p0.
///
/// # Arguments
/// * `successes` - Number of successes
/// * `n` - Total number of trials
/// * `p0` - Null hypothesis proportion
/// * `alternative` - Alternative hypothesis
///
/// # Returns
/// * `PropTestResult` containing test statistic and p-value
///
/// # Examples
/// ```
/// use anofox_statistics::categorical::{prop_test_one, Alternative};
///
/// // Test if coin is fair: 60 heads out of 100 flips
/// let result = prop_test_one(60, 100, 0.5, Alternative::TwoSided).unwrap();
/// println!("z = {:.4}", result.statistic);
/// println!("p-value = {:.4}", result.p_value);
/// ```
///
/// # R equivalent
/// `prop.test(x, n, p = p0, alternative = "...", correct = FALSE)`
pub fn prop_test_one(
    successes: usize,
    n: usize,
    p0: f64,
    alternative: Alternative,
) -> Result<PropTestResult> {
    prop_test_one_with_conf_level(successes, n, p0, alternative, 0.95)
}

/// One-sample proportion test with a configurable confidence level.
///
/// Same as [`prop_test_one`], but the Wilson score interval is computed at
/// `conf_level`. As in R's `prop.test(correct = FALSE)`, the interval is
/// two-sided for `TwoSided` and one-sided (`[0, U]` / `[L, 1]`, using
/// `qnorm(conf_level)`) for `Less` / `Greater`.
pub fn prop_test_one_with_conf_level(
    successes: usize,
    n: usize,
    p0: f64,
    alternative: Alternative,
    conf_level: f64,
) -> Result<PropTestResult> {
    validate_conf_level(conf_level)?;
    if n == 0 {
        return Err(StatError::EmptyData);
    }
    if !(0.0..=1.0).contains(&p0) {
        return Err(StatError::InvalidParameter(format!(
            "Null proportion must be between 0 and 1, got {}",
            p0
        )));
    }
    if successes > n {
        return Err(StatError::InvalidParameter(format!(
            "Successes ({}) cannot exceed n ({})",
            successes, n
        )));
    }

    let p_hat = successes as f64 / n as f64;
    let n_f = n as f64;

    // Standard error under null hypothesis
    let se = (p0 * (1.0 - p0) / n_f).sqrt();

    // Z-statistic
    let z = if se > 0.0 { (p_hat - p0) / se } else { 0.0 };

    // p-value
    let normal = Normal::new(0.0, 1.0).unwrap();
    let p_value = match alternative {
        Alternative::TwoSided => 2.0 * normal.sf(z.abs()),
        Alternative::Greater => normal.sf(z),
        Alternative::Less => normal.cdf(z),
    };

    // Wilson score confidence interval (R prop.test, correct = FALSE)
    let (conf_int_lower, conf_int_upper) = wilson_ci(successes, n, conf_level, alternative);

    Ok(PropTestResult {
        estimate: vec![p_hat],
        statistic: z,
        df: None,
        p_value,
        conf_int_lower,
        conf_int_upper,
        null_value: p0,
        alternative,
        method: "1-sample proportions test without continuity correction".to_string(),
    })
}

/// Two-sample proportion test.
///
/// Tests the null hypothesis that two population proportions are equal.
///
/// # Arguments
/// * `successes` - [x1, x2] number of successes in each group
/// * `totals` - [n1, n2] total number of trials in each group
/// * `alternative` - Alternative hypothesis
/// * `correction` - Apply Yates' continuity correction
///
/// # Returns
/// * `PropTestResult` containing test statistic and p-value
///
/// # Examples
/// ```
/// use anofox_statistics::categorical::{prop_test_two, Alternative};
///
/// // Compare conversion rates: 30/100 vs 45/120
/// let result = prop_test_two([30, 45], [100, 120], Alternative::TwoSided, false).unwrap();
/// println!("Chi-square = {:.4}", result.statistic);
/// println!("p-value = {:.4}", result.p_value);
/// ```
///
/// # R equivalent
/// `prop.test(c(x1, x2), c(n1, n2), alternative = "...")`
pub fn prop_test_two(
    successes: [usize; 2],
    totals: [usize; 2],
    alternative: Alternative,
    correction: bool,
) -> Result<PropTestResult> {
    prop_test_two_with_conf_level(successes, totals, alternative, correction, 0.95)
}

/// Two-sample proportion test with a configurable confidence level.
///
/// Same as [`prop_test_two`], but the confidence interval for `p1 - p2` is
/// computed at `conf_level`, exactly as R's `prop.test`: a Wald interval,
/// widened by the continuity correction `min(0.5, |p1 - p2| / (1/n1 + 1/n2)) *
/// (1/n1 + 1/n2)` when `correction` is true, two-sided for `TwoSided` and
/// one-sided (`[-1, U]` / `[L, 1]`) otherwise, clipped to `[-1, 1]`.
pub fn prop_test_two_with_conf_level(
    successes: [usize; 2],
    totals: [usize; 2],
    alternative: Alternative,
    correction: bool,
    conf_level: f64,
) -> Result<PropTestResult> {
    validate_conf_level(conf_level)?;
    if totals[0] == 0 || totals[1] == 0 {
        return Err(StatError::EmptyData);
    }
    if successes[0] > totals[0] || successes[1] > totals[1] {
        return Err(StatError::InvalidParameter(
            "Successes cannot exceed total trials".to_string(),
        ));
    }

    let p1 = successes[0] as f64 / totals[0] as f64;
    let p2 = successes[1] as f64 / totals[1] as f64;

    let n1 = totals[0] as f64;
    let n2 = totals[1] as f64;

    // Pooled proportion
    let p_pooled = (successes[0] + successes[1]) as f64 / (totals[0] + totals[1]) as f64;

    // Standard error under null (equal proportions)
    let se = (p_pooled * (1.0 - p_pooled) * (1.0 / n1 + 1.0 / n2)).sqrt();

    // Z-statistic (or chi-square)
    let diff = p1 - p2;
    let z = if se > 0.0 {
        if correction {
            // Yates' correction
            let correction_term = 0.5 * (1.0 / n1 + 1.0 / n2);
            let adj_diff = if diff.abs() > correction_term {
                diff.abs() - correction_term
            } else {
                0.0
            };
            adj_diff / se * diff.signum()
        } else {
            diff / se
        }
    } else {
        0.0
    };

    // Chi-square statistic (z^2)
    let chi_sq = z * z;

    // p-value
    let normal = Normal::new(0.0, 1.0).unwrap();
    let p_value = match alternative {
        Alternative::TwoSided => 2.0 * normal.sf(z.abs()),
        Alternative::Greater => normal.sf(z),
        Alternative::Less => normal.cdf(z),
    };

    // Confidence interval for difference in proportions (R prop.test)
    let se_diff = (p1 * (1.0 - p1) / n1 + p2 * (1.0 - p2) / n2).sqrt();
    let z_crit = normal_quantile_for(conf_level, alternative);
    let inv_sum = 1.0 / n1 + 1.0 / n2;
    let yates = if correction {
        0.5_f64.min(diff.abs() / inv_sum)
    } else {
        0.0
    };
    let width = z_crit * se_diff + yates * inv_sum;
    let (conf_int_lower, conf_int_upper) = match alternative {
        Alternative::TwoSided => ((diff - width).max(-1.0), (diff + width).min(1.0)),
        Alternative::Less => (-1.0, (diff + width).min(1.0)),
        Alternative::Greater => ((diff - width).max(-1.0), 1.0),
    };

    Ok(PropTestResult {
        estimate: vec![p1, p2],
        statistic: chi_sq,
        df: Some(1.0),
        p_value,
        conf_int_lower,
        conf_int_upper,
        null_value: 0.0, // Difference = 0 under null
        alternative,
        method: if correction {
            "2-sample test for equality of proportions with continuity correction"
        } else {
            "2-sample test for equality of proportions without continuity correction"
        }
        .to_string(),
    })
}

/// Exact binomial test.
///
/// Tests the null hypothesis that the probability of success equals p0.
/// Uses the exact binomial distribution rather than normal approximation.
///
/// # Arguments
/// * `successes` - Number of successes
/// * `n` - Total number of trials
/// * `p0` - Null hypothesis probability
/// * `alternative` - Alternative hypothesis
///
/// # Returns
/// * `BinomTestResult` containing the exact p-value and Clopper-Pearson CI
///
/// # Examples
/// ```
/// use anofox_statistics::categorical::{binom_test, Alternative};
///
/// // Test if proportion of heads is 0.5: 7 heads out of 10 flips
/// let result = binom_test(7, 10, 0.5, Alternative::TwoSided).unwrap();
/// println!("p-value = {:.4}", result.p_value);
/// ```
///
/// # R equivalent
/// `binom.test(x, n, p = p0, alternative = "...")`
pub fn binom_test(
    successes: usize,
    n: usize,
    p0: f64,
    alternative: Alternative,
) -> Result<BinomTestResult> {
    binom_test_with_conf_level(successes, n, p0, alternative, 0.95)
}

/// Exact binomial test with a configurable confidence level.
///
/// Same as [`binom_test`], but the Clopper-Pearson interval is computed at
/// `conf_level`. As in R's `binom.test`, the interval is two-sided for
/// `TwoSided` and one-sided (`[0, U]` / `[L, 1]`) for `Less` / `Greater`.
pub fn binom_test_with_conf_level(
    successes: usize,
    n: usize,
    p0: f64,
    alternative: Alternative,
    conf_level: f64,
) -> Result<BinomTestResult> {
    validate_conf_level(conf_level)?;
    if n == 0 {
        return Err(StatError::EmptyData);
    }
    if !(0.0..=1.0).contains(&p0) {
        return Err(StatError::InvalidParameter(format!(
            "Null probability must be between 0 and 1, got {}",
            p0
        )));
    }
    if successes > n {
        return Err(StatError::InvalidParameter(format!(
            "Successes ({}) cannot exceed n ({})",
            successes, n
        )));
    }

    let p_hat = successes as f64 / n as f64;

    // Exact p-value, following R's binom.test
    let p_value = binom_p_value(successes, n, p0, alternative);

    // Clopper-Pearson exact confidence interval
    let (conf_int_lower, conf_int_upper) =
        clopper_pearson_ci(successes, n, conf_level, alternative);

    Ok(BinomTestResult {
        estimate: p_hat,
        successes,
        n,
        p_value,
        conf_int_lower,
        conf_int_upper,
        null_value: p0,
        alternative,
        method: "Exact binomial test".to_string(),
    })
}

fn validate_conf_level(conf_level: f64) -> Result<()> {
    if conf_level.is_finite() && conf_level > 0.0 && conf_level < 1.0 {
        Ok(())
    } else {
        Err(StatError::InvalidParameter(format!(
            "conf_level must be in (0, 1), got {}",
            conf_level
        )))
    }
}

/// Normal quantile used by R's prop.test intervals:
/// `qnorm((1 + conf_level) / 2)` two-sided, `qnorm(conf_level)` one-sided.
fn normal_quantile_for(conf_level: f64, alternative: Alternative) -> f64 {
    let normal = Normal::new(0.0, 1.0).unwrap();
    match alternative {
        Alternative::TwoSided => normal.inverse_cdf((1.0 + conf_level) / 2.0),
        _ => normal.inverse_cdf(conf_level),
    }
}

/// Wilson score confidence interval for a proportion (R `prop.test(correct = FALSE)`).
fn wilson_ci(successes: usize, n: usize, conf_level: f64, alternative: Alternative) -> (f64, f64) {
    let p_hat = successes as f64 / n as f64;
    let n_f = n as f64;

    let z = normal_quantile_for(conf_level, alternative);
    let z22n = z * z / (2.0 * n_f);
    let half = z * (p_hat * (1.0 - p_hat) / n_f + z22n / (2.0 * n_f)).sqrt();

    let upper = if p_hat >= 1.0 {
        1.0
    } else {
        ((p_hat + z22n + half) / (1.0 + 2.0 * z22n)).min(1.0)
    };
    let lower = if p_hat <= 0.0 {
        0.0
    } else {
        ((p_hat + z22n - half) / (1.0 + 2.0 * z22n)).max(0.0)
    };

    match alternative {
        Alternative::TwoSided => (lower, upper),
        Alternative::Less => (0.0, upper),
        Alternative::Greater => (lower, 1.0),
    }
}

/// Exact binomial p-value, a port of R's `binom.test`.
///
/// The two-sided p-value sums the probabilities of all outcomes no more
/// likely than the observed one, using R's relative tolerance `1 + 1e-7`
/// (an absolute tolerance would include every outcome with probability below
/// it and floor the p-value for large `n`). Tails use the binomial CDF/SF.
fn binom_p_value(x: usize, n: usize, p0: f64, alternative: Alternative) -> f64 {
    // P(X <= k)
    let cdf = |binom: &Binomial, k: i64| -> f64 {
        if k < 0 {
            0.0
        } else {
            binom.cdf(k as u64)
        }
    };
    // P(X > k)
    let sf = |binom: &Binomial, k: i64| -> f64 {
        if k < 0 {
            1.0
        } else {
            binom.sf(k as u64)
        }
    };

    let binom = Binomial::new(p0, n as u64).unwrap();
    let xi = x as i64;
    let ni = n as i64;
    let p = match alternative {
        Alternative::Less => cdf(&binom, xi),
        Alternative::Greater => sf(&binom, xi - 1),
        Alternative::TwoSided => {
            if p0 == 0.0 {
                if x == 0 {
                    1.0
                } else {
                    0.0
                }
            } else if p0 == 1.0 {
                if x == n {
                    1.0
                } else {
                    0.0
                }
            } else {
                let rel_err = 1.0 + 1e-7;
                let d = binom.pmf(x as u64);
                let m = n as f64 * p0;
                let xf = x as f64;
                if xf == m {
                    1.0
                } else if xf < m {
                    let start = m.ceil() as i64;
                    let y = (start..=ni)
                        .filter(|&i| binom.pmf(i as u64) <= d * rel_err)
                        .count() as i64;
                    cdf(&binom, xi) + sf(&binom, ni - y)
                } else {
                    let end = m.floor() as i64;
                    let y = (0..=end)
                        .filter(|&i| binom.pmf(i as u64) <= d * rel_err)
                        .count() as i64;
                    cdf(&binom, y - 1) + sf(&binom, xi - 1)
                }
            }
        }
    };
    p.clamp(0.0, 1.0)
}

/// Clopper-Pearson exact confidence interval for a proportion (R `binom.test`).
fn clopper_pearson_ci(
    successes: usize,
    n: usize,
    conf_level: f64,
    alternative: Alternative,
) -> (f64, f64) {
    // p.L(alpha) = qbeta(alpha, x, n - x + 1), 0 when x == 0
    let p_lower = |alpha: f64| -> f64 {
        if successes == 0 {
            0.0
        } else {
            Beta::new(successes as f64, (n - successes + 1) as f64)
                .unwrap()
                .inverse_cdf(alpha)
        }
    };
    // p.U(alpha) = qbeta(1 - alpha, x + 1, n - x), 1 when x == n
    let p_upper = |alpha: f64| -> f64 {
        if successes == n {
            1.0
        } else {
            Beta::new((successes + 1) as f64, (n - successes) as f64)
                .unwrap()
                .inverse_cdf(1.0 - alpha)
        }
    };

    match alternative {
        Alternative::TwoSided => {
            let alpha = (1.0 - conf_level) / 2.0;
            (p_lower(alpha), p_upper(alpha))
        }
        Alternative::Less => (0.0, p_upper(1.0 - conf_level)),
        Alternative::Greater => (p_lower(1.0 - conf_level), 1.0),
    }
}

#[cfg(test)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;

    #[test]
    fn test_prop_test_one() {
        let result = prop_test_one(60, 100, 0.5, Alternative::TwoSided).unwrap();

        // z = (0.6 - 0.5) / sqrt(0.5*0.5/100) = 0.1 / 0.05 = 2.0
        assert!((result.statistic - 2.0).abs() < 0.01);
        assert!(result.p_value < 0.05);
    }

    #[test]
    fn test_prop_test_one_fair_coin() {
        let result = prop_test_one(50, 100, 0.5, Alternative::TwoSided).unwrap();

        assert!((result.statistic - 0.0).abs() < 1e-10);
        assert!((result.p_value - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_prop_test_two() {
        let result = prop_test_two([30, 50], [100, 100], Alternative::TwoSided, false).unwrap();

        // Pooled proportion = 80/200 = 0.4
        // SE = sqrt(0.4*0.6*(1/100 + 1/100)) = sqrt(0.0048) ≈ 0.069
        // z = (0.3-0.5)/0.069 ≈ -2.9
        assert!(result.statistic > 0.0);
        assert!(result.p_value < 0.05);
    }

    #[test]
    fn test_prop_test_two_equal() {
        let result = prop_test_two([50, 50], [100, 100], Alternative::TwoSided, false).unwrap();

        assert!((result.statistic - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_binom_test() {
        let result = binom_test(7, 10, 0.5, Alternative::TwoSided).unwrap();

        // 7 heads out of 10 with fair coin
        // p-value should be significant but not extremely small
        assert!(result.p_value > 0.1);
        assert!(result.p_value < 1.0);
    }

    #[test]
    fn test_binom_test_extreme() {
        let result = binom_test(10, 10, 0.5, Alternative::Greater).unwrap();

        // 10 out of 10 successes, one-sided test
        // P(X >= 10) = 0.5^10 ≈ 0.001
        assert!(result.p_value < 0.01);
    }

    #[test]
    fn test_binom_test_ci() {
        let result = binom_test(30, 100, 0.5, Alternative::TwoSided).unwrap();

        // CI should contain the estimate
        assert!(result.conf_int_lower < result.estimate);
        assert!(result.conf_int_upper > result.estimate);

        // CI should be reasonable (0.2-0.4ish for 30/100)
        assert!(result.conf_int_lower > 0.15);
        assert!(result.conf_int_upper < 0.45);
    }

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol, "got {a}, expected {b}");
    }

    /// Reference values: R 4.x binom.test()
    #[test]
    fn test_binom_test_matches_r() {
        let r = binom_test(13, 20, 0.5, Alternative::TwoSided).unwrap();
        close(r.p_value, 0.26317596435546875, 1e-14);
        close(r.conf_int_lower, 0.4078114654671719, 1e-12);
        close(r.conf_int_upper, 0.84609079521545882, 1e-12);

        let r = binom_test_with_conf_level(13, 20, 0.5, Alternative::Less, 0.9).unwrap();
        close(r.p_value, 0.94234085083007812, 1e-14);
        close(r.conf_int_lower, 0.0, 0.0);
        close(r.conf_int_upper, 0.79333596671715334, 1e-12);

        let r = binom_test_with_conf_level(13, 20, 0.5, Alternative::Greater, 0.8).unwrap();
        close(r.p_value, 0.13158798217773438, 1e-14);
        close(r.conf_int_lower, 0.53078479585500027, 1e-12);
        close(r.conf_int_upper, 1.0, 0.0);

        let r = binom_test(0, 10, 0.5, Alternative::TwoSided).unwrap();
        close(r.p_value, 0.0019531250000000004, 1e-15);
        close(r.conf_int_upper, 0.30849710781876083, 1e-12);
        let r = binom_test(10, 10, 0.3, Alternative::TwoSided).unwrap();
        close(r.p_value, 5.9048999999999915e-06, 1e-15);
        close(r.conf_int_lower, 0.69150289218123917, 1e-12);
    }

    /// Large n: the two-sided p-value must not be floored by an absolute tolerance.
    #[test]
    fn test_binom_test_large_n_matches_r() {
        let r = binom_test(5200, 10000, 0.5, Alternative::TwoSided).unwrap();
        assert!((r.p_value / 6.593515598672462e-05 - 1.0).abs() < 1e-8);
        close(r.conf_int_lower, 0.51015339471986465, 1e-10);
        close(r.conf_int_upper, 0.52983495148605675, 1e-10);
        let r = binom_test(4500, 10000, 0.5, Alternative::TwoSided).unwrap();
        assert!(
            (r.p_value / 1.5510640568246068e-23 - 1.0).abs() < 1e-6,
            "{}",
            r.p_value
        );
        let r = binom_test(30, 1000, 0.05, Alternative::TwoSided).unwrap();
        assert!(
            (r.p_value / 0.00281865528447252 - 1.0).abs() < 1e-8,
            "{}",
            r.p_value
        );
    }

    /// Reference values: R prop.test()
    #[test]
    fn test_prop_test_ci_matches_r() {
        let r = prop_test_one_with_conf_level(13, 20, 0.5, Alternative::TwoSided, 0.9).unwrap();
        close(r.conf_int_lower, 0.46651268884847175, 1e-12);
        close(r.conf_int_upper, 0.79773995994323899, 1e-12);
        let r = prop_test_one(13, 20, 0.5, Alternative::Greater).unwrap();
        close(r.conf_int_lower, 0.46651268884847175, 1e-12);
        close(r.conf_int_upper, 1.0, 0.0);

        // correct = TRUE (default in R)
        let r = prop_test_two([18, 11], [30, 28], Alternative::TwoSided, true).unwrap();
        close(r.conf_int_lower, -0.079284640161607245, 1e-12);
        close(r.conf_int_upper, 0.4935703544473215, 1e-12);
        let r =
            prop_test_two_with_conf_level([18, 11], [30, 28], Alternative::TwoSided, false, 0.99)
                .unwrap();
        close(r.conf_int_lower, -0.12391470604482185, 1e-12);
        close(r.conf_int_upper, 0.53820042033053617, 1e-12);
        let r = prop_test_two([18, 11], [30, 28], Alternative::Less, true).unwrap();
        close(r.conf_int_lower, -1.0, 0.0);
        close(r.conf_int_upper, 0.45307090560001412, 1e-12);
    }
}
