//! k-sample Anderson-Darling test (Scholz and Stephens, 1987).

use super::anderson_darling_table::{LOWER_P, M, TABLE_V1, TABLE_V2};
use crate::error::{Result, StatError};
use crate::utils::finite::ensure_no_nan;
use crate::utils::smooth_spline::SmoothSpline;

/// Result of [`ad_test_ksamp`].
#[derive(Debug, Clone, PartialEq)]
pub struct AndersonDarlingKResult {
    /// The Anderson-Darling criterion `A^2_kN` (version 1) or `A^2_akN`
    /// (version 2, midranks); kSamples' `AD` column.
    pub ad: f64,
    /// Standardized statistic `T = (AD - (k - 1)) / sigma_N`; kSamples'
    /// `T.AD` and scipy's `anderson_ksamp` statistic.
    pub statistic: f64,
    /// Asymptotic p-value (kSamples).
    pub p_value: f64,
    /// Whether the midrank version (version 2) was computed.
    pub midrank: bool,
    /// Number of samples.
    pub k: usize,
    /// Total number of observations.
    pub n: usize,
}

/// k-sample Anderson-Darling test of whether `k >= 2` samples come from the
/// same continuous distribution (Scholz and Stephens, 1987).
///
/// `midrank = true` computes version 2 of kSamples (`A^2_akN`, based on
/// midranks; the version for data with ties and scipy's default),
/// `midrank = false` version 1 (`A^2_kN`). Both are standardized to
/// `T = (AD - (k - 1)) / sigma_N` with the exact variance of the statistic
/// under the null hypothesis of a continuous distribution.
///
/// The p-value is kSamples' asymptotic p-value: Scholz and Stephens'
/// critical values (extended by kSamples to 35 levels and `m = k - 1` up to
/// 10 and infinity) are interpolated in `1 / sqrt(m)` by smoothing splines
/// (`spar = 0.4`), and the log-odds of the levels are interpolated in `T` by
/// another smoothing spline (`spar = 0.25`), extrapolated linearly beyond
/// the table, through a port of R's `smooth.spline`. Unlike scipy's
/// `anderson_ksamp`, the p-value is not capped to `[0.001, 0.25]`, so it
/// differs from scipy's; the statistic agrees.
///
/// For `k = 2` this is evidently's `"anderson"` drift test (default
/// threshold 0.1). It weighs the tails more than the Kolmogorov-Smirnov test.
///
/// As kSamples: with `N = 2` observations, or `N = 3` in three samples, `T`
/// is undefined (`NaN`) and the p-value is 1. `±Inf` ranks as an extreme
/// value; `NaN`, fewer than two samples or an empty sample are errors.
///
/// R equivalent: `kSamples::ad.test(x, y, ..., method = "asymptotic")`.
///
/// # Example
/// ```
/// use anofox_statistics::ad_test_ksamp;
/// let x: Vec<f64> = (0..50).map(|i| 2.0 * (0.37 * i as f64).sin()).collect();
/// let y: Vec<f64> = (0..40).map(|i| 2.0 * (0.53 * i as f64).sin() + 0.5).collect();
/// let r = ad_test_ksamp(&[&x, &y], true).unwrap();
/// assert!((r.statistic - 3.61044012836).abs() < 1e-9);
/// assert!((r.p_value - 0.011626).abs() < 1e-6);
/// ```
pub fn ad_test_ksamp(groups: &[&[f64]], midrank: bool) -> Result<AndersonDarlingKResult> {
    let k = groups.len();
    if k < 2 {
        return Err(StatError::InsufficientData { needed: 2, got: k });
    }
    for (i, g) in groups.iter().enumerate() {
        if g.is_empty() {
            return Err(StatError::EmptyData);
        }
        ensure_no_nan(&format!("groups[{}]", i), g)?;
    }
    let ns: Vec<usize> = groups.iter().map(|g| g.len()).collect();
    let n: usize = ns.iter().sum();
    let ad = ad_criterion(groups, midrank);

    let kf = k as f64;
    let (statistic, p_value) = if n == 2 || (n == 3 && k == 3) {
        (f64::NAN, 1.0)
    } else {
        let sigma = if n == 3 { 0.3535534 } else { ad_sigma(&ns) };
        let t = (ad - (kf - 1.0)) / sigma;
        (t, ad_pvalue(t, kf - 1.0, midrank))
    };

    Ok(AndersonDarlingKResult {
        ad,
        statistic,
        p_value,
        midrank,
        k,
        n,
    })
}

/// `A^2_kN` (version 1, eq. 7 of Scholz and Stephens) or `A^2_akN`
/// (version 2, eq. 6) over the distinct pooled values `z*_1 < ... < z*_L`.
fn ad_criterion(groups: &[&[f64]], midrank: bool) -> f64 {
    let n: usize = groups.iter().map(|g| g.len()).sum();
    let nf = n as f64;
    let mut pooled: Vec<(f64, usize)> = groups
        .iter()
        .enumerate()
        .flat_map(|(i, g)| g.iter().map(move |&v| (v, i)))
        .collect();
    pooled.sort_by(|a, b| a.0.total_cmp(&b.0));

    let k = groups.len();
    let mut m = vec![0.0f64; k]; // M_ij: observations of sample i <= z*_j
    let mut f = vec![0.0f64; k]; // f_ij: observations of sample i == z*_j
    let mut b = 0.0f64; // B_j
    let mut sums = vec![0.0f64; k];
    let mut start = 0;
    while start < pooled.len() {
        let v = pooled[start].0;
        let end = start + pooled[start..].iter().take_while(|p| p.0 == v).count();
        f.iter_mut().for_each(|x| *x = 0.0);
        for p in &pooled[start..end] {
            f[p.1] += 1.0;
        }
        let l = (end - start) as f64;
        for i in 0..k {
            m[i] += f[i];
        }
        b += l;
        let last = end == pooled.len();
        for i in 0..k {
            let ni = groups[i].len() as f64;
            if midrank {
                let (ma, ba) = (m[i] - f[i] / 2.0, b - l / 2.0);
                let num = nf * ma - ni * ba;
                sums[i] += l * num * num / (ba * (nf - ba) - nf * l / 4.0);
            } else if !last {
                let num = nf * m[i] - ni * b;
                sums[i] += l * num * num / (b * (nf - b));
            }
        }
        start = end;
    }
    let total: f64 = sums
        .iter()
        .zip(groups)
        .map(|(s, g)| s / g.len() as f64)
        .sum();
    if midrank {
        (nf - 1.0) / (nf * nf) * total
    } else {
        total / nf
    }
}

/// Standard deviation of `A^2_kN` under the null (Scholz and Stephens,
/// eq. 4), for `N > 3`.
fn ad_sigma(ns: &[usize]) -> f64 {
    let k = ns.len() as f64;
    let n: usize = ns.iter().sum();
    let nf = n as f64;
    let h_big: f64 = ns.iter().map(|&v| 1.0 / v as f64).sum();
    let h: f64 = (1..n).map(|i| 1.0 / i as f64).sum();
    let mut g = 0.0;
    for i in 1..=(n - 2) {
        let inner: f64 = ((i + 1)..n).map(|j| 1.0 / j as f64).sum();
        g += inner / (n - i) as f64;
    }
    let a = (4.0 * g - 6.0) * (k - 1.0) + (10.0 - 6.0 * g) * h_big;
    let b = (2.0 * g - 4.0) * k * k + 8.0 * h * k + (2.0 * g - 14.0 * h - 4.0) * h_big - 8.0 * h
        + 4.0 * g
        - 6.0;
    let c = (6.0 * h + 2.0 * g - 2.0) * k * k
        + (4.0 * h - 4.0 * g + 6.0) * k
        + (2.0 * h - 6.0) * h_big
        + 4.0 * h;
    let d = (2.0 * h + 6.0) * k * k - 4.0 * h * k;
    let var = (a * nf.powi(3) + b * nf * nf + c * nf + d) / ((nf - 1.0) * (nf - 2.0) * (nf - 3.0));
    var.sqrt()
}

/// Asymptotic p-value of `T` for `m = k - 1`, as `kSamples::ad.pval`.
fn ad_pvalue(t: f64, m: f64, midrank: bool) -> f64 {
    if t.is_nan() {
        return f64::NAN;
    }
    if t.is_infinite() {
        return if t > 0.0 { 0.0 } else { 1.0 };
    }
    let table = if midrank { &TABLE_V2 } else { &TABLE_V1 };
    let sqm1: Vec<f64> = M.iter().map(|v| 1.0 / v.sqrt()).collect();
    let sqm0 = 1.0 / m.sqrt();
    let mut tm = Vec::with_capacity(LOWER_P.len());
    for col in 0..LOWER_P.len() {
        let ys: Vec<f64> = table.iter().map(|row| row[col]).collect();
        match SmoothSpline::fit(&sqm1, &ys, 0.4) {
            Some(s) => tm.push(s.predict(sqm0)),
            None => return f64::NAN,
        }
    }
    let lp: Vec<f64> = LOWER_P
        .iter()
        .map(|&c| {
            let p = 1.0 - c;
            (p / (1.0 - p)).ln()
        })
        .collect();
    let lp0 = match SmoothSpline::fit(&tm, &lp, 0.25) {
        Some(s) => s.predict(t),
        None => return f64::NAN,
    };
    // exp(lp0) / (1 + exp(lp0)), without overflow
    1.0 / (1.0 + (-lp0).exp())
}
