//! Fisher's exact test for 2x2 contingency tables.

use crate::categorical::{validate_2x2_table, Alternative};
use crate::error::Result;
use statrs::distribution::{ContinuousCDF, Normal};

/// Result of Fisher's exact test
#[derive(Debug, Clone)]
pub struct FisherResult {
    /// p-value
    pub p_value: f64,
    /// Odds ratio estimate: the sample odds ratio `ad/bc` from
    /// [`fisher_exact`] / [`fisher_exact_with_conf_level`], the conditional MLE
    /// from [`fisher_exact_conditional`]
    pub odds_ratio: f64,
    /// Lower bound of the confidence interval for the odds ratio (Woolf from
    /// [`fisher_exact`], exact conditional from [`fisher_exact_conditional`])
    pub conf_int_lower: f64,
    /// Upper bound of the confidence interval for the odds ratio
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
/// # Odds ratio and interval
/// The p-value matches R's `fisher.test`, but `odds_ratio` is the **sample**
/// odds ratio `ad/bc` and the interval is a two-sided 95% Woolf (log-odds Wald)
/// interval (Haldane-Anscombe +0.5 correction when a cell is zero). For R's
/// conditional maximum-likelihood estimate and exact conditional interval use
/// [`fisher_exact_conditional`].
///
/// # R equivalent
/// `fisher.test(matrix(c(a, c, b, d), nrow=2))$p.value`
pub fn fisher_exact(table: &[[usize; 2]; 2], alternative: Alternative) -> Result<FisherResult> {
    fisher_exact_with_conf_level(table, alternative, 0.95)
}

/// Fisher's exact test with a configurable confidence level for the
/// (Woolf / logit) odds-ratio interval: `exp(log(OR) -/+ qnorm((1 + conf_level) / 2) * se)`.
///
/// The interval is always two-sided. See [`fisher_exact_conditional`] for R's
/// exact conditional interval (one-sided for one-sided alternatives).
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

    // Woolf (log-odds Wald) confidence interval for the sample odds ratio
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

/// Fisher's exact test for 2x2 tables with the full semantics of R's
/// `fisher.test`.
///
/// Unlike [`fisher_exact`] (which reports the *sample* odds ratio `ad/bc` and a
/// Woolf interval), this function reports:
///
/// * `odds_ratio`: the **conditional maximum-likelihood estimate** of the odds
///   ratio, i.e. the noncentrality parameter of the noncentral hypergeometric
///   distribution whose mean equals the observed `a` (0 or `inf` when `a` is at
///   the edge of its support);
/// * `conf_int_lower` / `conf_int_upper`: the **exact conditional** confidence
///   interval at `conf_level`: two-sided for [`Alternative::TwoSided`], `[0, U]`
///   for [`Alternative::Less`] and `[L, inf)` for [`Alternative::Greater`];
/// * `p_value`: identical to [`fisher_exact`] (exact hypergeometric, relative
///   tolerance `1 + 1e-7` for the two-sided test).
///
/// R solves the root-finding problems with `uniroot` (default tolerance about
/// `1e-4`), this implementation bisects to machine precision, so estimates and
/// bounds agree with R to about 1e-4 relative (scipy's
/// `contingency.odds_ratio(kind = "conditional")` agrees similarly).
///
/// # Errors
/// An all-zero table, or `conf_level` outside (0, 1).
///
/// # Examples
/// ```
/// use anofox_statistics::{fisher_exact_conditional, Alternative};
///
/// // R: fisher.test(matrix(c(18, 9, 7, 16), 2))
/// let r = fisher_exact_conditional(&[[18, 7], [9, 16]], Alternative::TwoSided, 0.95).unwrap();
/// assert!((r.odds_ratio - 4.4215633).abs() < 1e-3);
/// assert!((r.conf_int_lower - 1.2001231).abs() < 1e-3);
/// assert!((r.conf_int_upper - 18.0348).abs() < 2e-2);
/// ```
///
/// # R equivalent
/// `fisher.test(matrix(c(a, c, b, d), nrow = 2), alternative = ..., conf.level = ...)`
pub fn fisher_exact_conditional(
    table: &[[usize; 2]; 2],
    alternative: Alternative,
    conf_level: f64,
) -> Result<FisherResult> {
    // validates the table and conf_level, and computes the exact p-value
    let base = fisher_exact_with_conf_level(table, alternative, conf_level)?;
    let [[a, b], [c, d]] = *table;
    let x = a as f64;
    let h = NcHyper::new(a + c, b + d, a + b);
    let eps = f64::EPSILON;

    // R tests `x == lo` first, so a degenerate support (lo == hi) gives 0.
    let odds_ratio = if a == h.lo {
        0.0
    } else if a == h.hi {
        f64::INFINITY
    } else {
        let mu = h.mean(1.0);
        if mu > x {
            bisect(|t| h.mean(t) - x, 0.0, 1.0)
        } else if mu < x {
            1.0 / bisect(|t| h.mean(1.0 / t) - x, eps, 1.0)
        } else {
            1.0
        }
    };

    let ncp_upper = |alpha: f64| -> f64 {
        if a == h.hi {
            return f64::INFINITY;
        }
        let p = h.cdf(x, 1.0, false);
        if p < alpha {
            bisect(|t| h.cdf(x, t, false) - alpha, 0.0, 1.0)
        } else if p > alpha {
            1.0 / bisect(|t| h.cdf(x, 1.0 / t, false) - alpha, eps, 1.0)
        } else {
            1.0
        }
    };
    let ncp_lower = |alpha: f64| -> f64 {
        if a == h.lo {
            return 0.0;
        }
        let p = h.cdf(x, 1.0, true);
        if p > alpha {
            bisect(|t| h.cdf(x, t, true) - alpha, 0.0, 1.0)
        } else if p < alpha {
            1.0 / bisect(|t| h.cdf(x, 1.0 / t, true) - alpha, eps, 1.0)
        } else {
            1.0
        }
    };

    let (conf_int_lower, conf_int_upper) = match alternative {
        Alternative::Less => (0.0, ncp_upper(1.0 - conf_level)),
        Alternative::Greater => (ncp_lower(1.0 - conf_level), f64::INFINITY),
        Alternative::TwoSided => {
            let alpha = (1.0 - conf_level) / 2.0;
            (ncp_lower(alpha), ncp_upper(alpha))
        }
    };

    Ok(FisherResult {
        p_value: base.p_value,
        odds_ratio,
        conf_int_lower,
        conf_int_upper,
        alternative,
        method: "Fisher's Exact Test for Count Data (conditional MLE)".to_string(),
    })
}

/// Noncentral hypergeometric distribution of the (1,1) cell given the margins
/// (a port of the closures inside R's `fisher.test` for 2x2 tables).
struct NcHyper {
    lo: usize,
    hi: usize,
    support: Vec<f64>,
    logdc: Vec<f64>,
}

impl NcHyper {
    /// `m` = first column total, `n` = second column total, `k` = first row total.
    fn new(m: usize, n: usize, k: usize) -> Self {
        let lo = k.saturating_sub(n);
        let hi = k.min(m);
        let support: Vec<f64> = (lo..=hi).map(|s| s as f64).collect();
        let ln_total = log_binomial_coeff(m + n, k);
        let logdc = (lo..=hi)
            .map(|s| log_binomial_coeff(m, s) + log_binomial_coeff(n, k - s) - ln_total)
            .collect();
        Self {
            lo,
            hi,
            support,
            logdc,
        }
    }

    /// Density over the support for odds ratio `ncp` (0 < ncp < inf).
    fn density(&self, ncp: f64) -> Vec<f64> {
        let ln_ncp = ncp.ln();
        let d: Vec<f64> = self
            .logdc
            .iter()
            .zip(&self.support)
            .map(|(l, s)| l + ln_ncp * s)
            .collect();
        let max = d.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let e: Vec<f64> = d.iter().map(|v| (v - max).exp()).collect();
        let sum: f64 = e.iter().sum();
        e.into_iter().map(|v| v / sum).collect()
    }

    /// Mean of the distribution for odds ratio `ncp`.
    fn mean(&self, ncp: f64) -> f64 {
        if ncp == 0.0 {
            return self.lo as f64;
        }
        if ncp.is_infinite() {
            return self.hi as f64;
        }
        self.density(ncp)
            .iter()
            .zip(&self.support)
            .map(|(d, s)| d * s)
            .sum()
    }

    /// `P(X <= q)` (or `P(X >= q)` when `upper`) for odds ratio `ncp`.
    fn cdf(&self, q: f64, ncp: f64, upper: bool) -> f64 {
        let edge = |e: usize| {
            let e = e as f64;
            if (upper && q <= e) || (!upper && q >= e) {
                1.0
            } else {
                0.0
            }
        };
        if ncp == 0.0 {
            return edge(self.lo);
        }
        if ncp.is_infinite() {
            return edge(self.hi);
        }
        self.density(ncp)
            .iter()
            .zip(&self.support)
            .filter(|(_, &s)| if upper { s >= q } else { s <= q })
            .map(|(d, _)| d)
            .sum()
    }
}

/// Root of a monotone function on `(a, b)` by bisection (to machine precision).
fn bisect<F: Fn(f64) -> f64>(f: F, mut a: f64, mut b: f64) -> f64 {
    let fa_neg = f(a) < 0.0;
    for _ in 0..200 {
        let mid = 0.5 * (a + b);
        if mid <= a || mid >= b {
            break;
        }
        if (f(mid) < 0.0) == fa_neg {
            a = mid;
        } else {
            b = mid;
        }
    }
    0.5 * (a + b)
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
    /// Reference values: R's own `fisher.test` closures re-solved with
    /// `uniroot(tol = 1e-14)` (scratch R script in the PR); `fisher.test()` with
    /// its default tolerance agrees to ~1e-4 relative.
    #[test]
    #[allow(clippy::type_complexity)]
    fn test_fisher_conditional_matches_r() {
        let rel = |a: f64, b: f64| ((a - b) / b).abs();
        let cases: [([[usize; 2]; 2], Alternative, f64, f64, f64, f64, f64); 8] = [
            (
                [[18, 7], [9, 16]],
                Alternative::TwoSided,
                0.95,
                4.4215632150995905,
                1.2001279547251849,
                18.026777985373542,
                0.022241295691722542,
            ),
            (
                [[18, 7], [9, 16]],
                Alternative::Less,
                0.95,
                4.4215632150995905,
                0.0,
                14.655486900873857,
                0.99796884226026217,
            ),
            (
                [[18, 7], [9, 16]],
                Alternative::Greater,
                0.95,
                4.4215632150995905,
                1.4321845409277605,
                f64::INFINITY,
                0.011120647845861288,
            ),
            (
                [[18, 7], [9, 16]],
                Alternative::TwoSided,
                0.90,
                4.4215632150995905,
                1.43218454092776,
                14.655486900873875,
                0.022241295691722542,
            ),
            (
                [[0, 5], [3, 2]],
                Alternative::TwoSided,
                0.95,
                0.0,
                0.0,
                2.0268127716666622,
                0.16666666666666657,
            ),
            (
                [[1, 0], [0, 1]],
                Alternative::TwoSided,
                0.95,
                f64::INFINITY,
                0.025641025641025727,
                f64::INFINITY,
                1.0,
            ),
            (
                [[3, 1], [1, 3]],
                Alternative::TwoSided,
                0.95,
                6.4083196581996678,
                0.21173559544657861,
                626.24353058881309,
                0.4857142857142856,
            ),
            (
                [[2, 7], [8, 2]],
                Alternative::Less,
                0.99,
                0.085852399951238875,
                0.0,
                1.2352761398565795,
                0.018521725952066519,
            ),
        ];
        for (t, alt, cl, est, lo, hi, p) in cases {
            let r = fisher_exact_conditional(&t, alt, cl).unwrap();
            for (got, want) in [
                (r.odds_ratio, est),
                (r.conf_int_lower, lo),
                (r.conf_int_upper, hi),
            ] {
                if want == 0.0 || want.is_infinite() {
                    assert_eq!(got, want, "{t:?} {alt:?}");
                } else {
                    assert!(rel(got, want) < 1e-7, "{t:?} {alt:?}: {got} vs {want}");
                }
            }
            assert!(
                rel(r.p_value, p) < 1e-9,
                "{t:?} {alt:?}: p {} vs {p}",
                r.p_value
            );
        }
        // fisher.test() defaults (uniroot tol ~1e-4): estimate 4.4215633454100463,
        // CI [1.2001230942188943, 18.034801377448705]
        let r = fisher_exact_conditional(&[[18, 7], [9, 16]], Alternative::TwoSided, 0.95).unwrap();
        assert!(rel(r.odds_ratio, 4.4215633454100463) < 1e-4);
        assert!(rel(r.conf_int_lower, 1.2001230942188943) < 1e-4);
        assert!(rel(r.conf_int_upper, 18.034801377448705) < 1e-3);
        assert!(fisher_exact_conditional(&[[0, 0], [0, 0]], Alternative::TwoSided, 0.95).is_err());
        assert!(fisher_exact_conditional(&[[1, 2], [3, 4]], Alternative::TwoSided, 1.0).is_err());
    }
}
