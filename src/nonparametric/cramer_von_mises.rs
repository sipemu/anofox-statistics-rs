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
/// same continuous distribution, as `scipy.stats.cramervonmises_2samp`.
///
/// With the midranks `r_i` of the sorted `x` and `s_j` of the sorted `y` in
/// the pooled sample (`m = n_x`, `n = n_y`, `N = m + n`),
/// `U = m sum (r_i - i)^2 + n sum (s_j - j)^2` and
/// `T = U / (m n N) - (4 m n - 1) / (6 N)` (Anderson, 1962).
///
/// P-value:
/// - Exact: the distribution of `U` over all `choose(N, m)` splits, by the
///   recursion of Xiao, Gordon and Yakovlev (2006), in exact integer
///   arithmetic. It assumes no ties; with ties the midrank `U` is compared
///   with the no-ties distribution, as in scipy. Cost grows quickly with the
///   sample sizes, so `Exact` is meant for small samples.
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
/// p-value is `1 - goftest::pCvM(Tn)` with `Tn` as above.
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
        cvm_exact_pvalue(u4, m, n)
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

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// Add the counts of two frequency tables sorted by value.
fn merge(a: &[(u64, f64)], b: &[(u64, f64)]) -> Vec<(u64, f64)> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].0.cmp(&b[j].0) {
            std::cmp::Ordering::Less => {
                out.push(a[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                out.push(b[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                out.push((a[i].0, a[i].1 + b[j].1));
                i += 1;
                j += 1;
            }
        }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
    out
}

/// Exact `P(U >= U_obs)` from `u4 = 4 U_obs` (Xiao, Gordon and Yakovlev,
/// 2006, as scipy's `_pval_cvm_2samp_exact`).
///
/// With `L = lcm(m, n)`, `a = L / m`, `b = L / n`, the statistic of a split is
/// an affine function of `zeta = sum over the lattice path of (a v - b u)^2`.
/// The frequency table of `zeta` over the paths to `(u, v)` is built by
/// dynamic programming over the lattice.
fn cvm_exact_pvalue(u4: i128, m: usize, n: usize) -> f64 {
    let (mu, nu) = (m as u64, n as u64);
    let lcm = mu / gcd(mu, nu) * nu;
    let (a, b) = ((lcm / mu) as i128, (lcm / nu) as i128);
    let (mi, ni, li) = (m as i128, n as i128, lcm as i128);
    let mn = mi * ni;
    // zeta_obs = floor(L^2 N (6 U - mn (4 mn - 1)) / (6 m^2 n^2)), U = u4 / 4
    let num = li * li * (mi + ni) * (6 * u4 - 4 * mn * (4 * mn - 1));
    let zeta = num.div_euclid(24 * mn * mn);

    // gs[v]: frequency table of zeta over the paths to (u, v).
    let mut gs: Vec<Vec<(u64, f64)>> = vec![Vec::new(); m + 1];
    gs[0] = vec![(0, 1.0)];
    for u in 0..=n {
        let mut tmp: Vec<(u64, f64)> = Vec::new();
        let mut next = Vec::with_capacity(m + 1);
        for (v, g) in gs.iter().enumerate() {
            tmp = merge(&tmp, g);
            let res = (a * v as i128 - b * u as i128).pow(2) as u64;
            for e in tmp.iter_mut() {
                e.0 += res;
            }
            next.push(tmp.clone());
        }
        gs = next;
    }
    let total: f64 = gs[m].iter().map(|e| e.1).sum();
    let tail: f64 = gs[m]
        .iter()
        .filter(|e| e.0 as i128 >= zeta)
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
        assert_eq!(cvm_exact_pvalue(2, 1, 1), 1.0);
    }
}
