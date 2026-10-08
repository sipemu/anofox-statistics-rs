//! Hodges-Lehmann estimates and confidence intervals for the Wilcoxon
//! rank-sum (Mann-Whitney) and signed-rank tests, following R's
//! `wilcox.test(..., conf.int = TRUE)`.
//!
//! * **Asymptotic** (`exact = FALSE` in R, or ties/zeros, or a sample too
//!   large for the exact distribution): R root-finds the shift `d` at which
//!   the standardised rank statistic of the shifted sample crosses the normal
//!   quantile (`uniroot`, tolerance `1e-4`). This module ports that algorithm
//!   including R's Brent `zeroin`. The samples are sorted once; each statistic
//!   evaluation is then an `O(n)` merge, so the interval costs
//!   `O(n log n)` time and `O(n)` memory — the `n1 * n2` pairwise differences
//!   (or `n(n+1)/2` Walsh averages) are never materialised.
//! * **Exact** (small samples without ties, see
//!   [`super::wilcoxon_dist`]): the classical order statistics of the pairwise
//!   differences / Walsh averages. The number of differences is bounded by
//!   the exact-path size limits, so the memory stays bounded.

use super::wilcoxon::ConfidenceInterval;
use super::wilcoxon_dist::{cdf_le, mann_whitney_pmf, quantile, wilcoxon_pmf};
use crate::parametric::Alternative;
use statrs::distribution::{ContinuousCDF, Normal};

/// R's default `tol.root` for `wilcox.test`.
const TOL_ROOT: f64 = 1e-4;
/// R's default `maxiter` for `uniroot`.
const MAX_ITER: usize = 1000;
/// R's `toler` in the exact interval code.
const TOLER: f64 = 10.0 * f64::EPSILON;

/// Result of an estimate/interval computation: (estimate, interval).
pub(crate) type EstimateCi = (f64, ConfidenceInterval);

fn qnorm(p: f64) -> f64 {
    let normal = Normal::new(0.0, 1.0).unwrap();
    if p <= 0.0 {
        f64::NEG_INFINITY
    } else if p >= 1.0 {
        f64::INFINITY
    } else {
        normal.inverse_cdf(p)
    }
}

/// `qnorm(p, lower.tail = FALSE)`.
fn qnorm_upper(p: f64) -> f64 {
    -qnorm(p)
}

fn nan_ci(conf_level: f64) -> EstimateCi {
    (
        f64::NAN,
        ConfidenceInterval {
            lower: f64::NAN,
            upper: f64::NAN,
            conf_level,
        },
    )
}

fn median_sorted(s: &[f64]) -> f64 {
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    }
}

fn sorted(v: &[f64]) -> Vec<f64> {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    s
}

/// R's `sign(dz) * 0.5` / `0.5` / `-0.5` continuity correction.
fn correction(dz: f64, correct: bool, alternative: Alternative) -> f64 {
    if !correct {
        return 0.0;
    }
    match alternative {
        Alternative::TwoSided => {
            if dz > 0.0 {
                0.5
            } else if dz < 0.0 {
                -0.5
            } else {
                0.0
            }
        }
        Alternative::Greater => 0.5,
        Alternative::Less => -0.5,
    }
}

/// Port of R's `R_zeroin2` (Brent's method, `stats/src/zeroin.c`) as called
/// by `uniroot` (function values clamped to `±DBL_MAX`). Returns `None` when
/// the function returns NaN.
fn zeroin<F: FnMut(f64) -> f64>(
    ax: f64,
    bx: f64,
    fa: f64,
    fb: f64,
    mut f: F,
    tol: f64,
    max_iter: usize,
) -> Option<f64> {
    let clamp = |v: f64| v.clamp(-f64::MAX, f64::MAX);
    let (mut a, mut b) = (ax, bx);
    let (mut fa, mut fb) = (clamp(fa), clamp(fb));
    let mut c = a;
    let mut fc = fa;
    if fa == 0.0 {
        return Some(a);
    }
    if fb == 0.0 {
        return Some(b);
    }
    for _ in 0..=max_iter {
        let prev_step = b - a;
        if fc.abs() < fb.abs() {
            a = b;
            b = c;
            c = a;
            fa = fb;
            fb = fc;
            fc = fa;
        }
        let tol_act = 2.0 * f64::EPSILON * b.abs() + tol / 2.0;
        let mut new_step = (c - b) / 2.0;
        if new_step.abs() <= tol_act || fb == 0.0 {
            return Some(b);
        }
        if prev_step.abs() >= tol_act && fa.abs() > fb.abs() {
            let cb = c - b;
            let (mut p, mut q);
            if a == c {
                let t1 = fb / fa;
                p = cb * t1;
                q = 1.0 - t1;
            } else {
                let qq = fa / fc;
                let t1 = fb / fc;
                let t2 = fb / fa;
                p = t2 * (cb * qq * (qq - t1) - (b - a) * (t1 - 1.0));
                q = (qq - 1.0) * (t1 - 1.0) * (t2 - 1.0);
            }
            if p > 0.0 {
                q = -q;
            } else {
                p = -p;
            }
            if p < (0.75 * cb * q - (tol_act * q).abs() / 2.0) && p < (prev_step * q / 2.0).abs() {
                new_step = p / q;
            }
        }
        if new_step.abs() < tol_act {
            new_step = if new_step > 0.0 { tol_act } else { -tol_act };
        }
        a = b;
        fa = fb;
        b += new_step;
        let v = f(b);
        if v.is_nan() {
            return None;
        }
        fb = clamp(v);
        if (fb > 0.0 && fc > 0.0) || (fb < 0.0 && fc < 0.0) {
            c = a;
            fc = fa;
        }
    }
    Some(b)
}

/// R's `uniroot(f, lower, upper, f.lower, f.upper, tol)`; `None` where R
/// would stop with an error.
fn uniroot<F: FnMut(f64) -> f64>(
    lower: f64,
    upper: f64,
    f_lower: f64,
    f_upper: f64,
    f: F,
) -> Option<f64> {
    if lower.partial_cmp(&upper) != Some(std::cmp::Ordering::Less)
        || !lower.is_finite()
        || !upper.is_finite()
    {
        return None;
    }
    if f_lower.is_nan() || f_upper.is_nan() {
        return None;
    }
    let fl = f_lower.clamp(-f64::MAX, f64::MAX);
    let fu = f_upper.clamp(-f64::MAX, f64::MAX);
    if fl * fu > 0.0 {
        return None;
    }
    zeroin(lower, upper, fl, fu, f, TOL_ROOT, MAX_ITER)
}

// ============================================================
// Two-sample (Mann-Whitney)
// ============================================================

/// Standardised rank-sum statistic of `c(x - d, y)` as in R's
/// `.wilcox_test_two_cint_asymp` `W(d)`; `xs`, `ys` sorted ascending.
/// `O(n1 + n2)` time, `O(1)` extra memory.
fn two_sample_w(xs: &[f64], ys: &[f64], d: f64, correct: bool, alternative: Alternative) -> f64 {
    let nx = xs.len();
    let ny = ys.len();
    let (mut i, mut j) = (0usize, 0usize);
    let mut pos = 0usize;
    let mut rank_sum_x = 0.0f64;
    let mut ties = 0.0f64;
    while i < nx || j < ny {
        let xv = if i < nx { xs[i] - d } else { f64::INFINITY };
        let yv = if j < ny { ys[j] } else { f64::INFINITY };
        // Smallest remaining value (handles +Inf data via the index checks).
        let v = if j >= ny || (i < nx && xv <= yv) {
            xv
        } else {
            yv
        };
        let mut cx = 0usize;
        while i < nx && xs[i] - d == v {
            cx += 1;
            i += 1;
        }
        let mut cy = 0usize;
        while j < ny && ys[j] == v {
            cy += 1;
            j += 1;
        }
        let t = cx + cy;
        // average of ranks pos+1 ..= pos+t
        let avg = pos as f64 + (t as f64 + 1.0) / 2.0;
        rank_sum_x += cx as f64 * avg;
        let tf = t as f64;
        ties += tf * tf * tf - tf;
        pos += t;
    }
    let nxf = nx as f64;
    let nyf = ny as f64;
    let dz = rank_sum_x - nxf * (nxf + 1.0) / 2.0 - nxf * nyf / 2.0;
    let corr = correction(dz, correct, alternative);
    let sigma = ((nxf * nyf / 12.0)
        * ((nxf + nyf + 1.0) - ties / ((nxf + nyf) * (nxf + nyf - 1.0))))
        .sqrt();
    (dz - corr) / sigma
}

/// Hodges-Lehmann shift estimate and confidence interval for `x - y`
/// (R: `wilcox.test(x, y, conf.int = TRUE)`).
///
/// With `exact` (caller guarantees no ties and a feasible size) the classical
/// order statistics of the `n1 * n2` pairwise differences are used; otherwise
/// R's root-finding on the normal approximation (`O((n1+n2) log(n1+n2))`
/// time, `O(n1 + n2)` memory).
pub(crate) fn mann_whitney_estimate_ci(
    x: &[f64],
    y: &[f64],
    conf_level: f64,
    alternative: Alternative,
    correct: bool,
    exact: bool,
) -> EstimateCi {
    if exact {
        mann_whitney_ci_exact(x, y, conf_level, alternative)
    } else {
        mann_whitney_ci_asymp(x, y, conf_level, alternative, correct)
    }
}

fn mann_whitney_ci_asymp(
    x: &[f64],
    y: &[f64],
    conf_level: f64,
    alternative: Alternative,
    correct: bool,
) -> EstimateCi {
    let xs = sorted(x);
    let ys = sorted(y);
    let mumin = xs[0] - ys[ys.len() - 1];
    let mumax = xs[xs.len() - 1] - ys[0];
    if !mumin.is_finite() || !mumax.is_finite() {
        // R's uniroot cannot search an infinite bracket.
        return nan_ci(conf_level);
    }
    let alpha = 1.0 - conf_level;
    let w = |d: f64| two_sample_w(&xs, &ys, d, correct, alternative);
    let w_min = w(mumin);
    let w_max = w(mumax);
    if !w_min.is_finite() || !w_max.is_finite() {
        // Every observation tied (sigma = 0).
        return nan_ci(conf_level);
    }
    let root = |zq: f64| -> f64 {
        let f_lower = w_min - zq;
        if f_lower <= 0.0 {
            return mumin;
        }
        let f_upper = w_max - zq;
        if f_upper >= 0.0 {
            return mumax;
        }
        uniroot(mumin, mumax, f_lower, f_upper, |d| w(d) - zq).unwrap_or(f64::NAN)
    };
    let (lower, upper) = match alternative {
        Alternative::TwoSided => (root(qnorm_upper(alpha / 2.0)), root(qnorm(alpha / 2.0))),
        Alternative::Greater => (root(qnorm_upper(alpha)), f64::INFINITY),
        Alternative::Less => (f64::NEG_INFINITY, root(qnorm(alpha))),
    };
    // R sets `correct <- FALSE` before computing the estimate.
    let w0 = |d: f64| two_sample_w(&xs, &ys, d, false, alternative);
    let estimate = uniroot(mumin, mumax, w0(mumin), w0(mumax), w0).unwrap_or(f64::NAN);
    (
        estimate,
        ConfidenceInterval {
            lower,
            upper,
            conf_level,
        },
    )
}

/// Exact interval from the order statistics of the pairwise differences
/// (R's `.wilcox_test_two_cint_exact` without ties). Memory `O(n1 * n2)`,
/// bounded by `MW_EXACT_MAX_PAIRS`.
fn mann_whitney_ci_exact(
    x: &[f64],
    y: &[f64],
    conf_level: f64,
    alternative: Alternative,
) -> EstimateCi {
    let (n1, n2) = (x.len(), y.len());
    let mut diffs: Vec<f64> = Vec::with_capacity(n1 * n2);
    for &xi in x {
        for &yj in y {
            diffs.push(xi - yj);
        }
    }
    if diffs.iter().any(|d| d.is_nan()) {
        return nan_ci(conf_level);
    }
    diffs.sort_by(|a, b| a.total_cmp(b));
    let pmf = mann_whitney_pmf(n1, n2);
    let (lower, upper) = exact_order_ci(&diffs, &pmf, 1.0 - conf_level, alternative);
    (
        median_sorted(&diffs),
        ConfidenceInterval {
            lower,
            upper,
            conf_level,
        },
    )
}

/// R's exact interval rule shared by the rank-sum and signed-rank tests:
/// `qu <- q(alpha/2); if (p(qu) <= alpha/2 + toler) qu <- qu + 1`, interval
/// `(d[qu], d[N - qu + 1])` (1-based), infinite when `qu == 0`.
fn exact_order_ci(
    sorted_d: &[f64],
    pmf: &[f64],
    alpha: f64,
    alternative: Alternative,
) -> (f64, f64) {
    let n_d = sorted_d.len();
    let tail = match alternative {
        Alternative::TwoSided => alpha / 2.0,
        _ => alpha,
    };
    let mut qu = quantile(pmf, tail);
    if cdf_le(pmf, qu as f64) <= tail + TOLER {
        qu += 1;
    }
    if qu == 0 {
        return (f64::NEG_INFINITY, f64::INFINITY);
    }
    let qu = qu.min(n_d);
    let ql = n_d - qu; // 1-based index ql + 1 -> 0-based ql
    match alternative {
        Alternative::TwoSided => (sorted_d[qu - 1], sorted_d[ql.min(n_d - 1)]),
        Alternative::Greater => (sorted_d[qu - 1], f64::INFINITY),
        Alternative::Less => (f64::NEG_INFINITY, sorted_d[ql.min(n_d - 1)]),
    }
}

// ============================================================
// One-sample / paired (signed-rank)
// ============================================================

/// Standardised signed-rank statistic of `x - d` as in R's
/// `.wilcox_test_one_cint_asymp` `W(d)` (zeros dropped); `xs` sorted
/// ascending. The negative part, read backwards, and the positive part are
/// both ascending in `|x - d|`, so ranking is an `O(n)` merge.
fn one_sample_w(xs: &[f64], d: f64, correct: bool, alternative: Alternative) -> f64 {
    let n = xs.len();
    // First index with xs[i] - d >= 0 and first with xs[i] - d > 0.
    let neg_end = xs.partition_point(|&v| v - d < 0.0);
    let pos_start = xs.partition_point(|&v| v - d <= 0.0);
    let n_nonzero = neg_end + (n - pos_start);
    // Negatives: indices neg_end-1 down to 0 (|.| ascending).
    let mut ni = neg_end; // next negative is xs[ni - 1]
    let mut pi = pos_start;
    let mut pos = 0usize;
    let mut rank_sum_pos = 0.0f64;
    let mut ties = 0.0f64;
    while ni > 0 || pi < n {
        let nv = if ni > 0 {
            (xs[ni - 1] - d).abs()
        } else {
            f64::INFINITY
        };
        let pv = if pi < n {
            (xs[pi] - d).abs()
        } else {
            f64::INFINITY
        };
        let v = if pi >= n || (ni > 0 && nv <= pv) {
            nv
        } else {
            pv
        };
        let mut cn = 0usize;
        while ni > 0 && (xs[ni - 1] - d).abs() == v {
            cn += 1;
            ni -= 1;
        }
        let mut cp = 0usize;
        while pi < n && (xs[pi] - d).abs() == v {
            cp += 1;
            pi += 1;
        }
        let t = cn + cp;
        let avg = pos as f64 + (t as f64 + 1.0) / 2.0;
        rank_sum_pos += cp as f64 * avg;
        let tf = t as f64;
        ties += tf * tf * tf - tf;
        pos += t;
    }
    let nx = n_nonzero as f64;
    let zd = rank_sum_pos - nx * (nx + 1.0) / 4.0;
    let sigma = (nx * (nx + 1.0) * (2.0 * nx + 1.0) / 24.0 - ties / 48.0).sqrt();
    let corr = correction(zd, correct, alternative);
    (zd - corr) / sigma
}

/// Hodges-Lehmann (pseudo)median and confidence interval of the paired
/// differences `d` (R: `wilcox.test(x, y, paired = TRUE, conf.int = TRUE)`;
/// `d = x - y` *without* subtracting `mu` and *including* zeros, as in R).
///
/// With `exact` (caller guarantees no ties/zeros and a feasible size) the
/// order statistics of the `n(n+1)/2` Walsh averages are used; otherwise R's
/// root-finding on the normal approximation (`O(n log n)` time, `O(n)`
/// memory).
pub(crate) fn wilcoxon_estimate_ci(
    d: &[f64],
    conf_level: f64,
    alternative: Alternative,
    correct: bool,
    exact: bool,
) -> EstimateCi {
    if exact {
        wilcoxon_ci_exact(d, conf_level, alternative)
    } else {
        wilcoxon_ci_asymp(d, conf_level, alternative, correct)
    }
}

fn wilcoxon_ci_asymp(
    d: &[f64],
    conf_level: f64,
    alternative: Alternative,
    correct: bool,
) -> EstimateCi {
    let xs = sorted(d);
    let n = xs.len();
    let mut level = conf_level;
    if n == 0 {
        return nan_ci(conf_level);
    }
    let mumin = xs[0];
    let mumax = xs[n - 1];
    if !mumin.is_finite() || !mumax.is_finite() {
        return nan_ci(conf_level);
    }
    let w = |dd: f64| one_sample_w(&xs, dd, correct, alternative);
    let w_min = w(mumin);
    let w_max = if w_min.is_finite() {
        w(mumax)
    } else {
        f64::NAN
    };
    if !w_max.is_finite() {
        let lower = if alternative == Alternative::Less {
            f64::NEG_INFINITY
        } else {
            f64::NAN
        };
        let upper = if alternative == Alternative::Greater {
            f64::INFINITY
        } else {
            f64::NAN
        };
        return (
            (mumin + mumax) / 2.0,
            ConfidenceInterval {
                lower,
                upper,
                conf_level: 0.0,
            },
        );
    }
    let root = |zq: f64| {
        uniroot(mumin, mumax, w_min - zq, w_max - zq, |dd| w(dd) - zq).unwrap_or(f64::NAN)
    };
    let med = median_sorted(&xs);
    let mut alpha = 1.0 - conf_level;
    let adjust = |alpha: f64, level: &mut f64| {
        if alpha >= 1.0 || 1.0 - conf_level < alpha * 0.75 {
            *level = 1.0 - alpha.min(1.0);
        }
    };
    let (lower, upper) = match alternative {
        Alternative::TwoSided => {
            while alpha < 1.0 {
                let mindiff = w_min - qnorm_upper(alpha / 2.0);
                let maxdiff = w_max - qnorm(alpha / 2.0);
                if mindiff < 0.0 || maxdiff > 0.0 {
                    alpha *= 2.0;
                } else {
                    break;
                }
            }
            adjust(alpha, &mut level);
            if alpha < 1.0 {
                (root(qnorm_upper(alpha / 2.0)), root(qnorm(alpha / 2.0)))
            } else {
                (med, med)
            }
        }
        Alternative::Greater => {
            while alpha < 1.0 && w_min - qnorm_upper(alpha) < 0.0 {
                alpha *= 2.0;
            }
            adjust(alpha, &mut level);
            let l = if alpha < 1.0 {
                root(qnorm_upper(alpha))
            } else {
                med
            };
            (l, f64::INFINITY)
        }
        Alternative::Less => {
            // R checks `qnorm(alpha/2)` here (sic) and then uses qnorm(alpha).
            while alpha < 1.0 && w_max - qnorm(alpha / 2.0) > 0.0 {
                alpha *= 2.0;
            }
            adjust(alpha, &mut level);
            let u = if alpha < 1.0 { root(qnorm(alpha)) } else { med };
            (f64::NEG_INFINITY, u)
        }
    };
    let w0 = |dd: f64| one_sample_w(&xs, dd, false, alternative);
    let estimate = uniroot(mumin, mumax, w0(mumin), w0(mumax), w0).unwrap_or(f64::NAN);
    (
        estimate,
        ConfidenceInterval {
            lower,
            upper,
            conf_level: level,
        },
    )
}

/// Exact interval from the order statistics of the Walsh averages (R's
/// `.wilcox_test_one_cint_exact` without ties/zeros). Memory `O(n^2)`,
/// bounded by `WSR_EXACT_MAX_N`.
fn wilcoxon_ci_exact(d: &[f64], conf_level: f64, alternative: Alternative) -> EstimateCi {
    let n = d.len();
    let mut walsh: Vec<f64> = Vec::with_capacity(n * (n + 1) / 2);
    for i in 0..n {
        for j in i..n {
            walsh.push(d[i] + d[j]);
        }
    }
    if walsh.iter().any(|w| w.is_nan()) {
        return nan_ci(conf_level);
    }
    walsh.sort_by(|a, b| a.total_cmp(b));
    for w in walsh.iter_mut() {
        *w /= 2.0;
    }
    let pmf = wilcoxon_pmf(n);
    let (lower, upper) = exact_order_ci(&walsh, &pmf, 1.0 - conf_level, alternative);
    (
        median_sorted(&walsh),
        ConfidenceInterval {
            lower,
            upper,
            conf_level,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Naive W(d) via full ranking, for checking the merge-based version.
    fn naive_two(x: &[f64], y: &[f64], d: f64) -> f64 {
        let mut all: Vec<(f64, bool)> = x.iter().map(|&v| (v - d, true)).collect();
        all.extend(y.iter().map(|&v| (v, false)));
        let vals: Vec<f64> = all.iter().map(|p| p.0).collect();
        let (ranks, tie_sizes) = crate::nonparametric::ranks::rank_with_ties(&vals).unwrap();
        let rs: f64 = ranks
            .iter()
            .zip(all.iter())
            .filter(|(_, p)| p.1)
            .map(|(r, _)| r)
            .sum();
        let (nx, ny) = (x.len() as f64, y.len() as f64);
        let tc: f64 = tie_sizes.iter().map(|&t| (t * t * t - t) as f64).sum();
        let dz = rs - nx * (nx + 1.0) / 2.0 - nx * ny / 2.0;
        let sigma =
            ((nx * ny / 12.0) * ((nx + ny + 1.0) - tc / ((nx + ny) * (nx + ny - 1.0)))).sqrt();
        dz / sigma
    }

    #[test]
    fn merge_statistic_matches_full_ranking() {
        let x: Vec<f64> = (0..37).map(|i| ((i * 7 % 11) as f64) / 2.0).collect();
        let y: Vec<f64> = (0..23).map(|i| ((i * 5 % 13) as f64) / 2.0 - 1.0).collect();
        let mut xs = x.clone();
        xs.sort_by(f64::total_cmp);
        let mut ys = y.clone();
        ys.sort_by(f64::total_cmp);
        for d in [-3.0, -0.5, 0.0, 0.25, 1.0, 2.5] {
            let a = two_sample_w(&xs, &ys, d, false, Alternative::TwoSided);
            let b = naive_two(&x, &y, d);
            assert!((a - b).abs() < 1e-12, "d={d}: {a} vs {b}");
        }
    }
}
