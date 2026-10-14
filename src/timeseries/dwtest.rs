// Dense matrix code: index loops mirror the linear algebra.
#![allow(clippy::needless_range_loop)]

use super::autocorrelation::durbin_watson;
use crate::error::{Result, StatError};
use crate::parametric::Alternative;
use crate::test_result::TestResult;
use crate::utils::finite::ensure_finite;
use statrs::distribution::{ContinuousCDF, Normal};

/// Eigenvalues at or below this are dropped (`dwtest`'s `tol`).
const EV_TOL: f64 = 1e-10;
/// Absolute error target of the exact p-value (Imhof integral).
const IMHOF_TOL: f64 = 1e-11;

/// Durbin-Watson test for first-order autocorrelation of regression
/// residuals, as R `lmtest::dwtest(..., alternative, exact)`.
///
/// `residuals` are the OLS residuals of a regression on the design matrix
/// `x`, given as its columns (each of length `n`), **including** the
/// intercept column when the model has one. The statistic is
/// [`durbin_watson`]; its null distribution depends on `x`.
///
/// `alternative` refers to the autocorrelation: `Greater` (R's default) tests
/// positive autocorrelation, `p = P(DW <= d)`; `Less` tests negative
/// autocorrelation; `TwoSided` doubles the smaller tail.
///
/// `exact = None` uses R's default `exact = n < 100`:
/// - exact: under the null, `P(DW < d) = P(sum_j (lambda_j - d) z_j^2 < 0)`
///   with `lambda_j` the non-zero eigenvalues of `M A`
///   (`M = I - X (X'X)^-1 X'`, `A` the first-difference matrix; Durbin and
///   Watson, 1971), evaluated by numerical integration of Imhof's (1961)
///   formula to an absolute error of about `1e-11`. `dwtest` evaluates the
///   same probability with a 15-term series (Pan's procedure), so the two
///   agree to about `1e-6` or better. Where that series breaks down, R reports
///   a value outside `[0, 1]` and falls back to the normal approximation;
///   this function returns the correct exact p-value instead. It falls back
///   to the approximation only if the exact value cannot be computed (no
///   non-zero eigenvalue);
/// - otherwise the normal approximation with the exact mean and variance of
///   DW under the null.
///
/// For `n < 3` (or `n < max(5, k)` on the approximate path) the p-value is 1,
/// as R. The result has method `"Durbin-Watson test"`, the statistic, the
/// p-value, the alternative and `n`.
///
/// # References
/// - Durbin, J. and Watson, G. S. (1971). Testing for serial correlation in
///   least squares regression III. *Biometrika* 58, 1-19.
/// - Imhof, J. P. (1961). Computing the distribution of quadratic forms in
///   normal variables. *Biometrika* 48, 419-426.
///
/// # Errors
/// Empty or non-finite input, a column of the wrong length, no columns,
/// `k >= n`, a rank-deficient design, or all residuals zero.
///
/// # Example
/// ```
/// use anofox_statistics::{durbin_watson_test, Alternative};
/// let t: Vec<f64> = (1..=8).map(f64::from).collect();
/// let ones = vec![1.0; 8];
/// let e = [0.4, 0.6, 0.1, -0.3, -0.7, -0.4, 0.1, 0.2];
/// let r = durbin_watson_test(&e, &[&ones, &t], Alternative::Greater, None).unwrap();
/// assert_eq!(r.method, "Durbin-Watson test");
/// assert!(r.p_value.unwrap() < 0.05);
/// ```
pub fn durbin_watson_test(
    residuals: &[f64],
    x: &[&[f64]],
    alternative: Alternative,
    exact: Option<bool>,
) -> Result<TestResult> {
    let n = residuals.len();
    if x.is_empty() {
        return Err(StatError::InvalidParameter(
            "x must have at least one column".into(),
        ));
    }
    for (j, col) in x.iter().enumerate() {
        if col.len() != n {
            return Err(StatError::InvalidParameter(format!(
                "column {} of x has length {}, expected {}",
                j,
                col.len(),
                n
            )));
        }
        ensure_finite("x", col)?;
    }
    let dw = durbin_watson(residuals)?;
    let k = x.len();
    if k >= n {
        return Err(StatError::InvalidParameter(format!(
            "need more observations than columns of x ({} <= {})",
            n, k
        )));
    }
    let xtx_inv = xtx_inverse(x)?;

    let p_value = if n < 3 {
        1.0
    } else {
        let exact_p = if exact.unwrap_or(n < 100) {
            // P(DW < d) = P(sum_j (lambda_j - d) z_j^2 < 0).
            let shifted: Vec<f64> = ma_eigenvalues(x, &xtx_inv).iter().map(|l| l - dw).collect();
            let pdw = imhof_prob_below_zero(&shifted).clamp(0.0, 1.0);
            let p = match alternative {
                Alternative::TwoSided => 2.0 * pdw.min(1.0 - pdw),
                Alternative::Less => 1.0 - pdw,
                Alternative::Greater => pdw,
            };
            (0.0..=1.0).contains(&p).then_some(p)
        } else {
            None
        };
        match exact_p {
            Some(p) => p,
            None => normal_approx_p(dw, x, &xtx_inv, alternative),
        }
    };

    Ok(TestResult {
        method: "Durbin-Watson test".into(),
        alternative: Some(alternative),
        statistic: Some(dw),
        p_value: Some(p_value),
        n: Some(n),
        ..Default::default()
    })
}

/// `(X'X)^-1` by Cholesky; errors when `X` is (numerically) rank deficient.
fn xtx_inverse(x: &[&[f64]]) -> Result<Vec<Vec<f64>>> {
    let k = x.len();
    let mut l = vec![vec![0.0; k]; k];
    for i in 0..k {
        for j in 0..=i {
            let xtx: f64 = x[i].iter().zip(x[j]).map(|(a, b)| a * b).sum();
            let s = xtx - (0..j).map(|m| l[i][m] * l[j][m]).sum::<f64>();
            if i == j {
                let scale: f64 = x[i].iter().map(|v| v * v).sum();
                if s <= 1e-12 * scale.max(f64::MIN_POSITIVE) {
                    return Err(StatError::InvalidParameter(
                        "design matrix x is rank deficient".into(),
                    ));
                }
                l[i][i] = s.sqrt();
            } else {
                l[i][j] = s / l[j][j];
            }
        }
    }
    // L^-1 (lower triangular), then (X'X)^-1 = L^-T L^-1.
    let mut li = vec![vec![0.0; k]; k];
    for i in 0..k {
        li[i][i] = 1.0 / l[i][i];
        for j in 0..i {
            let s: f64 = (j..i).map(|m| l[i][m] * li[m][j]).sum();
            li[i][j] = -s / l[i][i];
        }
    }
    let mut inv = vec![vec![0.0; k]; k];
    for i in 0..k {
        for j in 0..k {
            inv[i][j] = (i.max(j)..k).map(|m| li[m][i] * li[m][j]).sum();
        }
    }
    Ok(inv)
}

/// `A v` for the Durbin-Watson difference matrix `A`
/// (`diag(1, 2, ..., 2, 1)`, `-1` on the off-diagonals).
fn diff_matrix_times(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    (0..n)
        .map(|t| {
            let mut s = 0.0;
            if t > 0 {
                s += v[t] - v[t - 1];
            }
            if t + 1 < n {
                s += v[t] - v[t + 1];
            }
            s
        })
        .collect()
}

/// The `n - k` largest eigenvalues of `M A` that exceed [`EV_TOL`], in
/// decreasing order (as `dwtest`). `M A` has the eigenvalues of the
/// symmetric `M A M` because `M` is idempotent.
fn ma_eigenvalues(x: &[&[f64]], xtx_inv: &[Vec<f64>]) -> Vec<f64> {
    let n = x[0].len();
    let k = x.len();
    // X (X'X)^-1, row-major n x k.
    let xq: Vec<Vec<f64>> = (0..n)
        .map(|t| {
            (0..k)
                .map(|j| (0..k).map(|m| x[m][t] * xtx_inv[m][j]).sum())
                .collect()
        })
        .collect();
    let mut m = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let h: f64 = (0..k).map(|c| xq[i][c] * x[c][j]).sum();
            let v = if i == j { 1.0 - h } else { -h };
            m[i][j] = v;
            m[j][i] = v;
        }
    }
    // M A M = M (A M); M and A are symmetric, so A M's columns are A m_j.
    let am: Vec<Vec<f64>> = m.iter().map(|col| diff_matrix_times(col)).collect();
    let mut b = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let v: f64 = m[i].iter().zip(&am[j]).map(|(p, q)| p * q).sum();
            b[i][j] = v;
            b[j][i] = v;
        }
    }
    let mut ev = symmetric_eigenvalues(b);
    ev.sort_by(|a, b| b.total_cmp(a));
    ev.truncate(n - k);
    ev.retain(|&v| v > EV_TOL);
    ev
}

/// Eigenvalues of a symmetric matrix: Householder reduction to tridiagonal
/// form followed by the implicit QL algorithm with Wilkinson shifts.
fn symmetric_eigenvalues(mut a: Vec<Vec<f64>>) -> Vec<f64> {
    let n = a.len();
    // Householder: zero column k below the subdiagonal.
    for k in 0..n.saturating_sub(2) {
        let norm: f64 = (k + 1..n).map(|i| a[i][k] * a[i][k]).sum::<f64>().sqrt();
        if norm == 0.0 {
            continue;
        }
        let alpha = if a[k + 1][k] > 0.0 { -norm } else { norm };
        let mut v: Vec<f64> = (k + 1..n).map(|i| a[i][k]).collect();
        v[0] -= alpha;
        let vtv: f64 = v.iter().map(|x| x * x).sum();
        if vtv == 0.0 {
            continue;
        }
        // S <- H S H with H = I - 2 v v' / v'v on the trailing block.
        let p: Vec<f64> = (k + 1..n)
            .map(|i| 2.0 * (k + 1..n).map(|j| a[i][j] * v[j - k - 1]).sum::<f64>() / vtv)
            .collect();
        let kk = v.iter().zip(&p).map(|(a, b)| a * b).sum::<f64>() / vtv;
        let q: Vec<f64> = p.iter().zip(&v).map(|(p, v)| p - kk * v).collect();
        for i in k + 1..n {
            for j in k + 1..n {
                let (ii, jj) = (i - k - 1, j - k - 1);
                a[i][j] -= v[ii] * q[jj] + q[ii] * v[jj];
            }
        }
        a[k + 1][k] = alpha;
        a[k][k + 1] = alpha;
        for i in k + 2..n {
            a[i][k] = 0.0;
            a[k][i] = 0.0;
        }
    }
    let mut d: Vec<f64> = (0..n).map(|i| a[i][i]).collect();
    let mut e: Vec<f64> = (0..n)
        .map(|i| if i + 1 < n { a[i + 1][i] } else { 0.0 })
        .collect();

    // Implicit QL on the tridiagonal (d, e); e[i] couples d[i] and d[i + 1].
    for l in 0..n {
        let mut iter = 0;
        loop {
            let mut m = l;
            while m + 1 < n {
                let dd = d[m].abs() + d[m + 1].abs();
                if e[m].abs() <= f64::EPSILON * dd {
                    break;
                }
                m += 1;
            }
            if m == l {
                break;
            }
            iter += 1;
            if iter > 60 {
                break;
            }
            let mut g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            let mut r = g.hypot(1.0);
            g = d[m] - d[l] + e[l] / (g + r.copysign(g));
            let (mut s, mut c, mut p) = (1.0, 1.0, 0.0);
            let mut deflated = false;
            for i in (l..m).rev() {
                let f = s * e[i];
                let b = c * e[i];
                r = f.hypot(g);
                e[i + 1] = r;
                if r == 0.0 {
                    d[i + 1] -= p;
                    e[m] = 0.0;
                    deflated = true;
                    break;
                }
                s = f / r;
                c = g / r;
                g = d[i + 1] - p;
                r = (d[i] - g) * s + 2.0 * c * b;
                p = s * r;
                d[i + 1] = g + p;
                g = c * r - b;
            }
            if deflated {
                continue;
            }
            d[l] -= p;
            e[l] = g;
            e[m] = 0.0;
        }
    }
    d
}

/// `P(sum_j w_j z_j^2 < 0)` for independent standard normal `z_j`, by
/// Imhof's (1961) inversion of the characteristic function:
///
/// `P = 1/2 - (1/pi) int_0^inf sin(theta(u)) / (u rho(u)) du`,
/// `theta(u) = 1/2 sum_j atan(w_j u)`, `rho(u) = prod_j (1 + w_j^2 u^2)^(1/4)`.
///
/// With `u = e^s` the integrand becomes `sin(theta) / rho`, which decays
/// exponentially at both ends. The range of `s` is cut where the neglected
/// parts are provably below `IMHOF_TOL` (`|sin theta| <= |theta|` near 0,
/// `rho(u) >= prod_j (|w_j| u)^(1/2)` in the tail) and integrated by
/// adaptive Gauss-Kronrod quadrature. Zero weights are ignored.
fn imhof_prob_below_zero(weights: &[f64]) -> f64 {
    let w: Vec<f64> = weights.iter().copied().filter(|&v| v != 0.0).collect();
    if w.is_empty() {
        return f64::NAN;
    }
    if w.iter().all(|&v| v > 0.0) {
        return 0.0;
    }
    if w.iter().all(|&v| v < 0.0) {
        return 1.0;
    }
    let m = w.len() as f64;
    let half_abs_sum: f64 = 0.5 * w.iter().map(|v| v.abs()).sum::<f64>();
    let half_log_prod: f64 = 0.5 * w.iter().map(|v| v.abs().ln()).sum::<f64>();
    // Each cut contributes at most IMHOF_TOL / 4 to the integral.
    let cut = 0.25 * IMHOF_TOL;
    // int_0^u0 |integrand| du <= u0 * half_abs_sum.
    let s_lo = (cut / half_abs_sum).ln();
    // int_U^inf |integrand| du <= (2 / m) U^(-m/2) / prod |w_j|^(1/2).
    let s_hi = (2.0 / m) * ((2.0 / m).ln() - cut.ln() - half_log_prod);
    let g = |s: f64| {
        let u = s.exp();
        let theta: f64 = 0.5 * w.iter().map(|v| (v * u).atan()).sum::<f64>();
        let log_rho: f64 = 0.25 * w.iter().map(|v| (v * u).powi(2).ln_1p()).sum::<f64>();
        theta.sin() * (-log_rho).exp()
    };
    let integral = gauss_kronrod_adaptive(&g, s_lo, s_hi, 0.5 * IMHOF_TOL);
    0.5 - integral / std::f64::consts::PI
}

/// Gauss-Kronrod 7-15 nodes (non-negative half) and weights.
const GK_X: [f64; 8] = [
    0.991_455_371_120_812_6,
    0.949_107_912_342_758_5,
    0.864_864_423_359_769_1,
    0.741_531_185_599_394_4,
    0.586_087_235_467_691_1,
    0.405_845_151_377_397_2,
    0.207_784_955_007_898_5,
    0.0,
];
const GK_WK: [f64; 8] = [
    0.022_935_322_010_529_22,
    0.063_092_092_629_978_55,
    0.104_790_010_322_250_2,
    0.140_653_259_715_525_9,
    0.169_004_726_639_267_9,
    0.190_350_578_064_785_4,
    0.204_432_940_075_298_9,
    0.209_482_141_084_727_8,
];
/// Gauss weights for the nodes `GK_X[1]`, `GK_X[3]`, `GK_X[5]`, `GK_X[7]`.
const GK_WG: [f64; 4] = [
    0.129_484_966_168_869_7,
    0.279_705_391_489_276_7,
    0.381_830_050_505_118_9,
    0.417_959_183_673_469_4,
];

/// Kronrod estimate and `|Kronrod - Gauss|` on `[a, b]`.
fn gk15<F: Fn(f64) -> f64>(f: &F, a: f64, b: f64) -> (f64, f64) {
    let c = 0.5 * (a + b);
    let h = 0.5 * (b - a);
    let fc = f(c);
    let mut k = GK_WK[7] * fc;
    let mut gs = GK_WG[3] * fc;
    for i in 0..7 {
        let pair = f(c - h * GK_X[i]) + f(c + h * GK_X[i]);
        k += GK_WK[i] * pair;
        if i % 2 == 1 {
            gs += GK_WG[i / 2] * pair;
        }
    }
    (k * h, ((k - gs) * h).abs())
}

/// Adaptive Gauss-Kronrod: bisect the interval with the largest error
/// estimate until the summed estimate is below `tol`.
fn gauss_kronrod_adaptive<F: Fn(f64) -> f64>(f: &F, a: f64, b: f64, tol: f64) -> f64 {
    let (v, e) = gk15(f, a, b);
    let mut parts = vec![(a, b, v, e)];
    for _ in 0..2000 {
        let err: f64 = parts.iter().map(|p| p.3).sum();
        if err <= tol {
            break;
        }
        let worst = (0..parts.len())
            .max_by(|&i, &j| parts[i].3.total_cmp(&parts[j].3))
            .unwrap();
        let (lo, hi, _, _) = parts.swap_remove(worst);
        let mid = 0.5 * (lo + hi);
        let (v1, e1) = gk15(f, lo, mid);
        let (v2, e2) = gk15(f, mid, hi);
        parts.push((lo, mid, v1, e1));
        parts.push((mid, hi, v2, e2));
    }
    parts.iter().map(|p| p.2).sum()
}

/// `dwtest`'s normal approximation: the exact null mean and variance of DW
/// from traces of `X'AX (X'X)^-1` and `(AX)'(AX) (X'X)^-1`.
fn normal_approx_p(dw: f64, x: &[&[f64]], xtx_inv: &[Vec<f64>], alt: Alternative) -> f64 {
    let n = x[0].len();
    let k = x.len();
    if n < 5.max(k) {
        return 1.0;
    }
    let ax: Vec<Vec<f64>> = x.iter().map(|col| diff_matrix_times(col)).collect();
    let cross = |u: &[&[f64]], v: &[Vec<f64>]| -> Vec<Vec<f64>> {
        (0..k)
            .map(|i| {
                (0..k)
                    .map(|j| u[i].iter().zip(&v[j]).map(|(a, b)| a * b).sum())
                    .collect()
            })
            .collect()
    };
    let matmul = |p: &[Vec<f64>], q: &[Vec<f64>]| -> Vec<Vec<f64>> {
        (0..k)
            .map(|i| {
                (0..k)
                    .map(|j| (0..k).map(|m| p[i][m] * q[m][j]).sum())
                    .collect()
            })
            .collect()
    };
    let trace = |p: &[Vec<f64>]| (0..k).map(|i| p[i][i]).sum::<f64>();

    let xaxq = matmul(&cross(x, &ax), xtx_inv);
    let ax_refs: Vec<&[f64]> = ax.iter().map(|c| c.as_slice()).collect();
    let axax = cross(&ax_refs, &ax);
    let nf = n as f64;
    let kf = k as f64;
    let p = 2.0 * (nf - 1.0) - trace(&xaxq);
    let q = 2.0 * (3.0 * nf - 4.0) - 2.0 * trace(&matmul(&axax, xtx_inv))
        + trace(&matmul(&xaxq, &xaxq));
    let dmean = p / (nf - kf);
    let dvar = 2.0 / ((nf - kf) * (nf - kf + 2.0)) * (q - p * dmean);
    let Ok(norm) = Normal::new(dmean, dvar.sqrt()) else {
        return f64::NAN;
    };
    match alt {
        Alternative::TwoSided => 2.0 * norm.sf(dmean + (dw - dmean).abs()),
        Alternative::Less => norm.sf(dw),
        Alternative::Greater => norm.cdf(dw),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn imhof_matches_closed_forms() {
        use statrs::distribution::FisherSnedecor;
        // P(z1^2 < a z2^2) = P(|Cauchy| < sqrt(a)) = (2 / pi) atan(sqrt(a)).
        for a in [0.01, 1.0, 3.0, 250.0] {
            let got = imhof_prob_below_zero(&[1.0, -a]);
            let want = 2.0 / std::f64::consts::PI * a.sqrt().atan();
            assert!((got - want).abs() < 1e-10, "a={}: {} vs {}", a, got, want);
        }
        // P(chi2_p / p < c chi2_q / q) = pf(c, p, q).
        for (p, q, c) in [(1, 3, 0.4), (3, 5, 1.7), (4, 30, 2.5), (12, 2, 0.8)] {
            let mut w = vec![1.0 / p as f64; p];
            w.extend(vec![-c / q as f64; q]);
            let got = imhof_prob_below_zero(&w);
            let want = FisherSnedecor::new(p as f64, q as f64).unwrap().cdf(c);
            assert!(
                (got - want).abs() < 1e-10,
                "F({}, {}) at {}: {} vs {}",
                p,
                q,
                c,
                got,
                want
            );
        }
        assert_eq!(imhof_prob_below_zero(&[1.0, 2.0]), 0.0);
        assert_eq!(imhof_prob_below_zero(&[-1.0, 0.0]), 1.0);
        assert!(imhof_prob_below_zero(&[0.0]).is_nan());
    }

    #[test]
    fn gauss_kronrod_integrates_smooth_functions() {
        let v = gauss_kronrod_adaptive(&|x: f64| x.cos(), 0.0, 1.0, 1e-14);
        assert!((v - 1f64.sin()).abs() < 1e-14);
        let v = gauss_kronrod_adaptive(&|x: f64| (-x * x).exp(), -30.0, 30.0, 1e-13);
        assert!((v - std::f64::consts::PI.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn eigenvalues_of_known_matrix() {
        // Second-difference matrix: eigenvalues 2 - 2 cos(j pi / (n + 1)).
        let n = 7;
        let mut a = vec![vec![0.0; n]; n];
        for i in 0..n {
            a[i][i] = 2.0;
            if i + 1 < n {
                a[i][i + 1] = -1.0;
                a[i + 1][i] = -1.0;
            }
        }
        let mut ev = symmetric_eigenvalues(a);
        ev.sort_by(|a, b| a.total_cmp(b));
        for (j, v) in ev.iter().enumerate() {
            let want = 2.0 - 2.0 * ((j + 1) as f64 * std::f64::consts::PI / (n + 1) as f64).cos();
            assert!((v - want).abs() < 1e-13, "{} vs {}", v, want);
        }
    }

    #[test]
    fn xtx_inverse_is_inverse() {
        let c0 = [1.0, 1.0, 1.0, 1.0];
        let c1 = [1.0, 2.0, 4.0, 7.0];
        let x: [&[f64]; 2] = [&c0, &c1];
        let inv = xtx_inverse(&x).unwrap();
        // X'X = [[4, 14], [14, 70]], det = 84.
        assert!((inv[0][0] - 70.0 / 84.0).abs() < 1e-14);
        assert!((inv[0][1] + 14.0 / 84.0).abs() < 1e-14);
        assert!((inv[1][1] - 4.0 / 84.0).abs() < 1e-14);
        let dup: [&[f64]; 2] = [&c1, &c1];
        assert!(xtx_inverse(&dup).is_err());
    }
}
