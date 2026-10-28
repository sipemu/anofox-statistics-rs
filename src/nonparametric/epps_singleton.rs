//! Epps-Singleton two-sample test.

use crate::error::{Result, StatError};
use crate::utils::dist::chisq_sf;
use crate::utils::finite::ensure_finite;
use crate::utils::math::symmetric_eigen;

/// Result of [`epps_singleton_test`].
#[derive(Debug, Clone, PartialEq)]
pub struct EppsSingletonResult {
    /// The statistic `W` (with the small-sample correction when
    /// `max(n_x, n_y) < 25`).
    pub statistic: f64,
    /// P-value from the chi-squared distribution with `df` degrees of
    /// freedom.
    pub p_value: f64,
    /// Rank of the estimated covariance matrix (`2 * len(t)` when it has full
    /// rank).
    pub df: usize,
    /// Size of `x`.
    pub n_x: usize,
    /// Size of `y`.
    pub n_y: usize,
}

/// R `quantile(v, c(0.25, 0.75), type = 7)` difference (numpy's linear IQR).
fn iqr(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let q = |p: f64| {
        let h = (s.len() - 1) as f64 * p;
        let lo = h.floor() as usize;
        s[lo] + (h - lo as f64) * (s[h.ceil() as usize] - s[lo])
    };
    q(0.75) - q(0.25)
}

/// Epps-Singleton two-sample test (Epps and Singleton, 1986) of whether `x`
/// and `y` come from the same distribution, based on the empirical
/// characteristic function at the points `t` (default `[0.4, 0.8]`).
///
/// The points are scaled by the semi-interquartile range of the pooled
/// sample, `t / (IQR / 2)`. For each observation `v`,
/// `g(v) = (cos(t_1 v), ..., cos(t_J v), sin(t_1 v), ..., sin(t_J v))`;
/// with the mean difference `d` of `g` between the samples and
/// `S = (n / n_x) Cov(g_x) + (n / n_y) Cov(g_y)` (biased covariances),
/// `W = n d' S^+ d`, which is asymptotically chi-squared with
/// `df = rank(S)` degrees of freedom. When `max(n_x, n_y) < 25`, `W` is
/// multiplied by the small-sample correction
/// `1 / (1 + n^-0.45 + 10.1 (n_x^-1.7 + n_y^-1.7))`.
///
/// The pseudo-inverse and the rank use the eigenvalues of `S` above
/// `1e-15` times the largest (numpy's `pinv` default; scipy takes the rank of
/// the pseudo-inverse). A rank below `2 * len(t)` means `t` is a poor choice
/// for the data, e.g. data with very few distinct values.
///
/// Unlike the Kolmogorov-Smirnov test it also suits discrete distributions.
/// This is evidently's `"es"` drift test (threshold 0.05), which calls
/// `scipy.stats.epps_singleton_2samp`.
///
/// Errors when `n_x < 5` or `n_y < 5`, on non-finite data, when `t` is empty
/// or not all positive and finite, when the pooled IQR is 0, or when the
/// estimated covariance matrix is zero.
///
/// R equivalent: none on CRAN; `R/generate_drift_refs.R` has a direct port
/// (`es2`) of scipy's implementation.
///
/// # Example
/// ```
/// use anofox_statistics::epps_singleton_test;
/// let x: Vec<f64> = (0..50).map(|i| 2.0 * (0.37 * i as f64).sin()).collect();
/// let y: Vec<f64> = (0..40).map(|i| 2.0 * (0.53 * i as f64).sin() + 0.5).collect();
/// let r = epps_singleton_test(&x, &y, None).unwrap();
/// assert!((r.statistic - 19.25205304182).abs() < 1e-9);
/// assert!((r.p_value - 0.00070119804217).abs() < 1e-12);
/// assert_eq!(r.df, 4);
/// ```
pub fn epps_singleton_test(x: &[f64], y: &[f64], t: Option<&[f64]>) -> Result<EppsSingletonResult> {
    let (nx, ny) = (x.len(), y.len());
    if nx < 5 || ny < 5 {
        return Err(StatError::InsufficientData {
            needed: 5,
            got: nx.min(ny),
        });
    }
    ensure_finite("x", x)?;
    ensure_finite("y", y)?;
    let t = t.unwrap_or(&[0.4, 0.8]);
    if t.is_empty() || t.iter().any(|&v| !(v.is_finite() && v > 0.0)) {
        return Err(StatError::InvalidParameter(
            "t must contain positive, finite values only".to_string(),
        ));
    }
    let sigma = iqr(&[x, y].concat()) / 2.0;
    if sigma <= 0.0 || !sigma.is_finite() {
        return Err(StatError::InvalidParameter(
            "the interquartile range of the pooled sample must be positive".to_string(),
        ));
    }
    let ts: Vec<f64> = t.iter().map(|v| v / sigma).collect();
    let dim = 2 * ts.len();
    let g = |v: f64| -> Vec<f64> {
        let mut out: Vec<f64> = ts.iter().map(|t| (t * v).cos()).collect();
        out.extend(ts.iter().map(|t| (t * v).sin()));
        out
    };
    // Means and biased covariances of g over a sample.
    let moments = |data: &[f64]| {
        let gs: Vec<Vec<f64>> = data.iter().map(|&v| g(v)).collect();
        let nf = data.len() as f64;
        let mean: Vec<f64> = (0..dim)
            .map(|j| gs.iter().map(|r| r[j]).sum::<f64>() / nf)
            .collect();
        let cov: Vec<Vec<f64>> = (0..dim)
            .map(|i| {
                (0..dim)
                    .map(|j| {
                        gs.iter()
                            .map(|r| (r[i] - mean[i]) * (r[j] - mean[j]))
                            .sum::<f64>()
                            / nf
                    })
                    .collect()
            })
            .collect();
        (mean, cov)
    };
    let (mx, cx) = moments(x);
    let (my, cy) = moments(y);
    let n = (nx + ny) as f64;
    let (wx, wy) = (n / nx as f64, n / ny as f64);
    let s: Vec<Vec<f64>> = (0..dim)
        .map(|i| (0..dim).map(|j| wx * cx[i][j] + wy * cy[i][j]).collect())
        .collect();
    let d: Vec<f64> = mx.iter().zip(&my).map(|(a, b)| a - b).collect();

    let (eval, evec) = symmetric_eigen(&s);
    let max_abs = eval.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    if max_abs == 0.0 {
        return Err(StatError::InvalidParameter(
            "estimated covariance matrix is zero".to_string(),
        ));
    }
    let cutoff = 1e-15 * max_abs;
    let mut df = 0usize;
    let mut quad = 0.0;
    for (k, &l) in eval.iter().enumerate() {
        if l.abs() > cutoff {
            df += 1;
            let proj: f64 = (0..dim).map(|i| evec[i][k] * d[i]).sum();
            quad += proj * proj / l;
        }
    }
    let mut w = n * quad;
    if nx.max(ny) < 25 {
        let corr =
            1.0 / (1.0 + n.powf(-0.45) + 10.1 * ((nx as f64).powf(-1.7) + (ny as f64).powf(-1.7)));
        w *= corr;
    }
    Ok(EppsSingletonResult {
        statistic: w,
        p_value: chisq_sf(w, df as f64),
        df,
        n_x: nx,
        n_y: ny,
    })
}
