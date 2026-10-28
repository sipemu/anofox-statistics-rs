//! Two-sample Cramér-von Mises test.

use super::ranks::rank;
use crate::error::{Result, StatError};
use crate::utils::finite::ensure_no_nan;

/// How [`cvm_test_2sample`] computes the p-value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CvmMethod {
    /// Exact when `max(n_x, n_y) <= 20`, asymptotic otherwise (scipy's rule).
    Auto,
    /// Exact distribution (Xiao, Gordon and Yakovlev, 2006).
    Exact,
    /// Limiting distribution of the normalized statistic (Anderson, 1962).
    Asymptotic,
}

/// Result of [`cvm_test_2sample`].
#[derive(Debug, Clone, PartialEq)]
pub struct CvmTestResult {
    /// The statistic `T` of Anderson (1962).
    pub statistic: f64,
    /// P-value.
    pub p_value: f64,
    /// Whether the exact p-value was computed (otherwise asymptotic).
    pub exact: bool,
    /// Size of `x`.
    pub n_x: usize,
    /// Size of `y`.
    pub n_y: usize,
}

/// Two-sample Cramér-von Mises test of whether `x` and `y` come from the
/// same distribution (the statistic of `scipy.stats.cramervonmises_2samp`).
///
/// With the midranks `r_i` of the sorted `x` and `s_j` of the sorted `y` in
/// the pooled sample (`m = n_x`, `n = n_y`, `N = m + n`),
/// `U = m sum (r_i - i)^2 + n sum (s_j - j)^2` and
/// `T = U / (m n N) - (4 m n - 1) / (6 N)` (Anderson, 1962).
///
/// P-value:
/// - Exact: the permutation distribution of `U` over all `choose(N, m)`
///   splits of the pooled sample, by dynamic programming over the lattice
///   paths (Xiao, Gordon and Yakovlev, 2006) in exact integer arithmetic.
///   With ties it is the exact distribution of the midrank `U` given the tie
///   pattern, i.e. full enumeration of the splits in R; scipy instead
///   compares the midrank `U` with the no-ties distribution, so its exact
///   p-value differs with ties. Cost grows quickly with the sample sizes, so
///   `Exact` is meant for small samples.
/// - Asymptotic: `T_n = 1/6 + (T - E[T]) / sqrt(45 Var[T])` against the
///   limiting Cramér-von Mises distribution (Csörgő and Faraway, 1996),
///   computed as `goftest::pCvM`.
/// - `Auto` picks exact when `max(n_x, n_y) <= 20` (scipy's rule).
///
/// This is evidently's `"cramer_von_mises"` drift test.
///
/// `±Inf` ranks as an extreme value; `NaN` or an empty sample is an error.
///
/// R equivalent: `CvM2SL2Test` has been archived on CRAN; the asymptotic
/// p-value is `1 - goftest::pCvM(Tn)` with `Tn` as above, the exact one the
/// share of splits `combn(N, m)` with `U >= U_obs`.
///
/// # Example
/// ```
/// use anofox_statistics::{cvm_test_2sample, CvmMethod};
/// let x: Vec<f64> = (0..50).map(|i| 2.0 * (0.37 * i as f64).sin()).collect();
/// let y: Vec<f64> = (0..40).map(|i| 2.0 * (0.53 * i as f64).sin() + 0.5).collect();
/// let r = cvm_test_2sample(&x, &y, CvmMethod::Auto).unwrap();
/// assert!((r.statistic - 0.4722592592593).abs() < 1e-9);
/// assert!((r.p_value - 0.0468933196396).abs() < 1e-9);
/// assert!(!r.exact);
/// ```
pub fn cvm_test_2sample(x: &[f64], y: &[f64], method: CvmMethod) -> Result<CvmTestResult> {
    if x.is_empty() || y.is_empty() {
        return Err(StatError::EmptyData);
    }
    ensure_no_nan("x", x)?;
    ensure_no_nan("y", y)?;
    let (m, n) = (x.len(), y.len());
    let exact = match method {
        CvmMethod::Auto => m.max(n) <= 20,
        CvmMethod::Exact => true,
        CvmMethod::Asymptotic => false,
    };

    // Twice the midranks are integers, so 4 U is an integer.
    let ranks = rank(&[x, y].concat())?;
    let twice_sorted = |r: &[f64]| {
        let mut v: Vec<i128> = r.iter().map(|v| (2.0 * v).round() as i128).collect();
        v.sort_unstable();
        v
    };
    let sum_sq = |r2: &[i128]| -> i128 {
        r2.iter()
            .enumerate()
            .map(|(i, &v)| (v - 2 * (i as i128 + 1)).pow(2))
            .sum()
    };
    let (mi, ni) = (m as i128, n as i128);
    let u4 = mi * sum_sq(&twice_sorted(&ranks[..m])) + ni * sum_sq(&twice_sorted(&ranks[m..]));

    let (mf, nf) = (m as f64, n as f64);
    let (k, big_n) = (mf * nf, mf + nf);
    let u = u4 as f64 / 4.0;
    let statistic = u / (k * big_n) - (4.0 * k - 1.0) / (6.0 * big_n);

    let p_value = if exact {
        let mut pooled = [x, y].concat();
        pooled.sort_by(|a, b| a.total_cmp(b));
        let mut ties = Vec::new();
        let mut k = 0;
        while k < pooled.len() {
            let l = pooled[k..].iter().take_while(|&&v| v == pooled[k]).count();
            ties.push(l);
            k += l;
        }
        cvm_exact_pvalue(u4, m, n, &ties)
    } else {
        let et = (1.0 + 1.0 / big_n) / 6.0;
        let vt = (big_n + 1.0) * (4.0 * k * big_n - 3.0 * (mf * mf + nf * nf) - 2.0 * k)
            / (45.0 * big_n * big_n * 4.0 * k);
        let tn = 1.0 / 6.0 + (statistic - et) / (45.0 * vt).sqrt();
        if tn.is_nan() {
            // Var[T] = 0 (m = n = 1)
            1.0
        } else {
            1.0 - cvm_limit_cdf(tn)
        }
    };

    Ok(CvmTestResult {
        statistic,
        p_value: p_value.clamp(0.0, 1.0),
        exact,
        n_x: m,
        n_y: n,
    })
}

/// Add the counts of `b`, shifted by `shift` and multiplied by `weight`, to
/// the frequency table `a`; both are sorted by value.
fn merge_into(a: &[(u64, f64)], b: &[(u64, f64)], shift: u64, weight: f64) -> Vec<(u64, f64)> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() || j < b.len() {
        let bj = b.get(j).map(|e| (e.0 + shift, e.1 * weight));
        match (a.get(i), bj) {
            (Some(&ai), Some(bj)) if ai.0 == bj.0 => {
                out.push((ai.0, ai.1 + bj.1));
                i += 1;
                j += 1;
            }
            (Some(&ai), Some(bj)) if ai.0 < bj.0 => {
                out.push(ai);
                i += 1;
            }
            (Some(&ai), None) => {
                out.push(ai);
                i += 1;
            }
            (_, Some(bj)) => {
                out.push(bj);
                j += 1;
            }
            (None, None) => unreachable!(),
        }
    }
    out
}

/// Exact `P(U >= U_obs)` from `u4 = 4 U_obs`: the permutation distribution
/// of the midrank statistic `U` over all `choose(N, m)` splits of the pooled
/// sample, given its tie groups `ties` (sizes, in ascending order of value).
///
/// Dynamic programming over the tie groups: a split is a lattice path, and a
/// tie group of size `l` with doubled midrank `R` that sends `a` values to
/// `x` (`choose(l, a)` ways), after `i` values of `x` and `j` of `y`, adds
/// `m sum_{t=1}^{a} (R - 2(i + t))^2 + n sum_{t=1}^{l-a} (R - 2(j + t))^2`
/// to `4 U`. Without ties this is the distribution of Xiao, Gordon and
/// Yakovlev (2006); with ties it equals full enumeration of the splits.
fn cvm_exact_pvalue(u4: i128, m: usize, n: usize, ties: &[usize]) -> f64 {
    let (mi, ni) = (m as i128, n as i128);
    // tables[i]: frequency table of 4 U over the paths with i values of x
    let mut tables: Vec<Vec<(u64, f64)>> = vec![Vec::new(); m + 1];
    tables[0] = vec![(0, 1.0)];
    let mut done = 0usize;
    for &l in ties {
        let r2 = (2 * done + l + 1) as i128;
        let mut next: Vec<Vec<(u64, f64)>> = vec![Vec::new(); m + 1];
        for (i, table) in tables.iter().enumerate() {
            if table.is_empty() {
                continue;
            }
            let j = done - i;
            let mut binom = 1.0f64;
            for a in 0..=l {
                if a > 0 {
                    binom = binom * (l - a + 1) as f64 / a as f64;
                }
                if i + a > m || j + (l - a) > n {
                    continue;
                }
                let sx: i128 = (1..=a as i128)
                    .map(|t| (r2 - 2 * (i as i128 + t)).pow(2))
                    .sum();
                let sy: i128 = (1..=(l - a) as i128)
                    .map(|t| (r2 - 2 * (j as i128 + t)).pow(2))
                    .sum();
                let add = (mi * sx + ni * sy) as u64;
                next[i + a] = merge_into(&next[i + a], table, add, binom);
            }
        }
        tables = next;
        done += l;
    }
    let total: f64 = tables[m].iter().map(|e| e.1).sum();
    let tail: f64 = tables[m]
        .iter()
        .filter(|e| e.0 as i128 >= u4)
        .map(|e| e.1)
        .sum();
    tail / total
}

/// `exp(-q) K_{1/4}(q)` for `q > 0`, with `K` the modified Bessel function
/// of the second kind, from `K_nu(q) = int_0^inf exp(-q cosh t) cosh(nu t) dt`
/// by the trapezoidal rule (spectrally accurate for this integrand).
fn exp_neg_bessel_k_quarter(q: f64) -> f64 {
    const H: f64 = 0.05;
    let f = |t: f64| (-q * (t.cosh() + 1.0) + 0.25 * t).exp() * (1.0 + (-0.5 * t).exp()) / 2.0;
    let mut sum = 0.5 * f(0.0);
    let mut i = 1.0;
    loop {
        let t = i * H;
        let v = f(t);
        sum += v;
        // past the peak and negligible
        if v < 1e-300 || (v < sum * 1e-18 && q * t.sinh() > 0.25) {
            break;
        }
        i += 1.0;
    }
    sum * H
}

/// Limiting CDF of the Cramér-von Mises statistic, as `goftest::pCvM(q)`
/// (`n = Inf`): `0` for `q <= 1/1200`, `1` for `q >= 100/3`, otherwise the
/// series of Csörgő and Faraway (1996)
/// `(1/pi) sum_k (-1)^k choose(-1/2, k) sqrt(4k + 1) exp(-q_k) K_{1/4}(q_k) / sqrt(q)`,
/// `q_k = (4k + 1)^2 / (16 q)`, summed until a term is below `1e-9`; values
/// within `2e-10` of 0 or 1 are rounded to 0 or 1.
pub(crate) fn cvm_limit_cdf(q: f64) -> f64 {
    if q <= 1.0 / 1200.0 {
        return 0.0;
    }
    if q >= 100.0 / 3.0 {
        return 1.0;
    }
    let mut tot = 0.0;
    // (-1)^k choose(-1/2, k) = Gamma(k + 1/2) / (sqrt(pi) k!)
    let mut c = 1.0;
    for k in 0..=200 {
        if k > 0 {
            c *= (k as f64 - 0.5) / k as f64;
        }
        let kf = 4.0 * k as f64 + 1.0;
        let qk = kf * kf / (16.0 * q);
        let z = c * kf.sqrt() * exp_neg_bessel_k_quarter(qk) / q.sqrt();
        tot += z;
        if k > 10 && z.abs() < 1e-9 {
            break;
        }
    }
    let p = tot / std::f64::consts::PI;
    if p < 2e-10 {
        0.0
    } else if 1.0 - p < 2e-10 {
        1.0
    } else {
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bessel_k_quarter() {
        // R: exp(-q) * besselK(q, 1/4)
        for (q, e) in [
            (0.01, 6.1043911135014399),
            (0.5, 0.58246129409374836),
            (3.0, 0.0017453880482996523),
            (50.0, 6.5814323856205467e-45),
        ] {
            let v = exp_neg_bessel_k_quarter(q);
            assert!(((v - e) / e).abs() < 1e-13, "{} {} {}", q, v, e);
        }
    }

    #[test]
    fn limit_cdf_matches_goftest() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/R/data/drift_reference.csv");
        let mut rdr = csv::Reader::from_path(path).unwrap();
        let mut get = |key: &str| -> Vec<f64> {
            for r in rdr.records() {
                let r = r.unwrap();
                if &r[0] == "pcvm" && &r[1] == key {
                    return r[2].split(' ').map(|v| v.parse().unwrap()).collect();
                }
            }
            panic!("missing {}", key)
        };
        let (q, p) = (get("in_q"), get("p"));
        for (q, p) in q.iter().zip(&p) {
            let v = cvm_limit_cdf(*q);
            assert!((v - p).abs() < 1e-13, "{} {} {}", q, v, p);
        }
    }

    #[test]
    fn exact_tiny() {
        // m = n = 1: U = 1/2 for both splits; T = 0, p = 1.
        assert_eq!(cvm_exact_pvalue(2, 1, 1, &[1, 1]), 1.0);
    }
}
