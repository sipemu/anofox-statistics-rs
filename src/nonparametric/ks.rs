//! Two-sample Kolmogorov-Smirnov test.

use crate::error::{Result, StatError};
use crate::parametric::Alternative;
use crate::utils::finite::ensure_no_nan;

/// Result of [`ks_test_2sample`].
#[derive(Debug, Clone, PartialEq)]
pub struct KsTestResult {
    /// `D = sup |F_x - F_y|` (two-sided), `D+ = sup (F_x - F_y)` (greater) or
    /// `D- = sup (F_y - F_x)` (less), with `F_x`, `F_y` the empirical CDFs.
    pub statistic: f64,
    /// P-value.
    pub p_value: f64,
    /// Whether the exact p-value was computed (otherwise asymptotic).
    pub exact: bool,
    /// Alternative hypothesis.
    pub alternative: Alternative,
    /// Size of `x`.
    pub n_x: usize,
    /// Size of `y`.
    pub n_y: usize,
}

/// Two-sample Kolmogorov-Smirnov test of whether `x` and `y` come from the
/// same continuous distribution.
///
/// The statistic compares the empirical CDFs at the distinct pooled values:
/// `D = max |F_x - F_y|` for `Alternative::TwoSided`, `D+ = max (F_x - F_y)`
/// for `Greater` (the CDF of `x` lies above that of `y`) and
/// `D- = max (F_y - F_x)` for `Less`.
///
/// P-value, as R `ks.test`:
/// - `exact = None` uses the exact p-value when `n_x * n_y < 10000` and the
///   asymptotic one otherwise (R's default).
/// - Exact: R's `psmirnov` lattice-path recursion (Schröer and Trenkler,
///   1995; Viehmann, 2021). With ties it is the exact conditional
///   distribution given the tie pattern, as in R >= 4.2.
/// - Asymptotic: the limiting Kolmogorov distribution of `sqrt(m n / (m + n)) D`,
///   summed to R's tolerance of 1e-6 (two-sided), and `exp(-2 m n / (m + n) D^2)`
///   (one-sided). This is R's `exact = FALSE` value; it differs from scipy's
///   `ks_2samp(method = "asymp")`, which uses a finite-sample approximation.
///
/// evidently's default drift test for numeric columns is this test,
/// two-sided, flagging drift at `p <= 0.05` (via `scipy.stats.ks_2samp`,
/// whose default exact p-value equals R's).
///
/// `±Inf` is ranked as an extreme value; `NaN` is an error, as is an empty
/// sample.
///
/// R equivalent: `ks.test(x, y, alternative = , exact = )`.
///
/// # Example
/// ```
/// use anofox_statistics::{ks_test_2sample, Alternative};
/// let x: Vec<f64> = (0..50).map(|i| 2.0 * (0.37 * i as f64).sin()).collect();
/// let y: Vec<f64> = (0..40).map(|i| 2.0 * (0.53 * i as f64).sin() + 0.5).collect();
/// let r = ks_test_2sample(&x, &y, Alternative::TwoSided, None).unwrap();
/// assert!((r.statistic - 0.275).abs() < 1e-12);
/// assert!((r.p_value - 0.0574636491581).abs() < 1e-9);
/// assert!(r.exact);
/// ```
pub fn ks_test_2sample(
    x: &[f64],
    y: &[f64],
    alternative: Alternative,
    exact: Option<bool>,
) -> Result<KsTestResult> {
    if x.is_empty() || y.is_empty() {
        return Err(StatError::EmptyData);
    }
    ensure_no_nan("x", x)?;
    ensure_no_nan("y", y)?;
    let (m, n) = (x.len(), y.len());
    let exact = exact.unwrap_or((m as f64) * (n as f64) < 10000.0);

    // Pooled sample in ascending order, tagged with its group.
    let mut pooled: Vec<(f64, bool)> = x
        .iter()
        .map(|&v| (v, true))
        .chain(y.iter().map(|&v| (v, false)))
        .collect();
    pooled.sort_by(|a, b| a.0.total_cmp(&b.0));

    // ECDF differences at the end of every run of tied values, from integer
    // counts (R accumulates +1/m, -1/n, which leaves rounding noise).
    let (md, nd) = (m as f64, n as f64);
    let (mut cx, mut cy) = (0usize, 0usize);
    let (mut d_plus, mut d_minus) = (0.0f64, 0.0f64);
    for (k, &(v, is_x)) in pooled.iter().enumerate() {
        if is_x {
            cx += 1;
        } else {
            cy += 1;
        }
        if k + 1 == pooled.len() || pooled[k + 1].0 != v {
            let diff = cx as f64 / md - cy as f64 / nd;
            d_plus = d_plus.max(diff);
            d_minus = d_minus.max(-diff);
        }
    }
    let statistic = match alternative {
        Alternative::TwoSided => d_plus.max(d_minus),
        Alternative::Greater => d_plus,
        Alternative::Less => d_minus,
    };

    let p_value = if statistic <= 0.0 {
        1.0
    } else if statistic > 1.0 {
        0.0
    } else if exact {
        // Tie mask of R's psmirnov: tied[k] = the k-th and (k+1)-th pooled
        // values differ (k = 1..N-1), with z[0] = 0 and z[N] = 1.
        let has_ties = pooled.windows(2).any(|w| w[0].0 == w[1].0);
        let z: Option<Vec<bool>> = if has_ties {
            let mut z = Vec::with_capacity(m + n + 1);
            z.push(false);
            z.extend(pooled.windows(2).map(|w| w[0].0 != w[1].0));
            z.push(true);
            // R drops the mask when every pooled value is equal.
            z.iter().any(|&b| b).then_some(z)
        } else {
            None
        };
        let (a, b) = match alternative {
            Alternative::Less => (n, m),
            _ => (m, n),
        };
        psmirnov_exact_upper(
            statistic,
            a,
            b,
            z.as_deref(),
            alternative == Alternative::TwoSided,
        )
    } else {
        let ne = md * nd / (md + nd);
        match alternative {
            Alternative::TwoSided => kolmogorov_limit_upper(ne.sqrt() * statistic, 1e-6),
            _ => (-2.0 * ne * statistic * statistic).exp(),
        }
    };

    Ok(KsTestResult {
        statistic,
        p_value: p_value.clamp(0.0, 1.0),
        exact,
        alternative,
        n_x: m,
        n_y: n,
    })
}

/// `P(D >= q)` for samples of size `m` and `n` (R `psmirnov_exact`, upper
/// tail), by Viehmann's (2021) recursion over the lattice paths. `z` is the
/// tie mask (`z[i + j]` true when a path may stop between pooled values
/// `i + j` and `i + j + 1`); `None` means no ties.
fn psmirnov_exact_upper(q: f64, m: usize, n: usize, z: Option<&[bool]>, two: bool) -> f64 {
    let (md, nd) = (m as f64, n as f64);
    // Shift q down by 0.5 / (m n) so that rounding cannot turn an equality
    // into an inequality.
    let q = (0.5 + (q * md * nd - 1e-7).floor()) / (md * nd);
    let reached = |i: usize, j: usize| {
        let d = i as f64 / md - j as f64 / nd;
        let hit = if two { d.abs() >= q } else { d >= q };
        hit && z.is_none_or(|z| z[i + j])
    };
    let mut u = vec![0.0f64; n + 1];
    for j in 1..=n {
        u[j] = if reached(0, j) { 1.0 } else { u[j - 1] };
    }
    for i in 1..=m {
        if reached(i, 0) {
            u[0] = 1.0;
        }
        for j in 1..=n {
            u[j] = if reached(i, j) {
                1.0
            } else {
                let v = i as f64 / (i + j) as f64;
                let w = j as f64 / (i + j) as f64;
                v * u[j] + w * u[j - 1]
            };
        }
    }
    u[n]
}

/// Upper tail `1 - K(x)` of the limiting Kolmogorov distribution
/// `K(x) = 1 + 2 sum_k (-1)^k exp(-2 k^2 x^2)`, as R's `K2l(x, lower = FALSE,
/// tol)`: for `x < 1` the alternative series in `exp(-(2k-1)^2 pi^2 / (8 x^2))`
/// with R's fixed number of terms, otherwise the alternating series until
/// successive partial sums differ by at most `tol`.
fn kolmogorov_limit_upper(x: f64, tol: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    if x < 1.0 {
        let k_max = (2.0 - tol.ln()).sqrt() as usize;
        let w = x.ln();
        let z = -(std::f64::consts::FRAC_PI_2 * std::f64::consts::FRAC_PI_4) / (x * x);
        let s: f64 = (1..k_max)
            .step_by(2)
            .map(|k| ((k * k) as f64 * z - w).exp())
            .sum();
        // s / (1 / sqrt(2 pi))
        1.0 - s * (2.0 * std::f64::consts::PI).sqrt()
    } else {
        let z = -2.0 * x * x;
        let mut s = -1.0;
        let mut k = 2.0f64;
        let mut old = 0.0;
        let mut new = 2.0 * z.exp();
        while (old - new).abs() > tol {
            old = new;
            new += 2.0 * s * (z * k * k).exp();
            s = -s;
            k += 1.0;
        }
        new
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_upper_tail_small() {
        // m = n = 1: D = 1 always.
        assert_eq!(psmirnov_exact_upper(1.0, 1, 1, None, true), 1.0);
        // m = n = 2, two-sided: P(D >= 1) = 2 / choose(4, 2) = 1/3.
        assert!((psmirnov_exact_upper(1.0, 2, 2, None, true) - 1.0 / 3.0).abs() < 1e-15);
        // one-sided: P(D+ >= 1) = 1/6.
        assert!((psmirnov_exact_upper(1.0, 2, 2, None, false) - 1.0 / 6.0).abs() < 1e-15);
    }

    #[test]
    fn kolmogorov_limit_values() {
        // 1 - K(x) for x >= 1 (R: .Call(stats:::C_pkolmogorov_two_limit, 1.2, FALSE, 1e-6)).
        assert!((kolmogorov_limit_upper(1.2, 1e-6) - 0.11224966667072496).abs() < 1e-15);
        assert_eq!(kolmogorov_limit_upper(0.0, 1e-6), 1.0);
    }
}
