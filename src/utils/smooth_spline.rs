//! Port of R's `smooth.spline(x, y, spar = )` and `predict()` for a fixed
//! `spar`, unit weights and all knots (R's choice for fewer than 50 distinct
//! `x`). Used to reproduce interpolations that R packages do with
//! `smooth.spline` (`kSamples::ad.pval`).
//!
//! As R: `x` is scaled to `[0, 1]`, the fit is a cubic B-spline with a knot at
//! every distinct `x` minimizing `sum w (y - f)^2 + lambda c' SIGMA c` with
//! `lambda = r * 256^(3 spar - 1)`, `r = tr(X'WX) / tr(SIGMA)` over the
//! interior basis functions, and `SIGMA` the Gram matrix of the second
//! derivatives as computed by R's `sgram.f` (which integrates the square of
//! the linear term with the factor `0.333` rather than `1/3`). Prediction
//! outside the range of `x` is linear, from the value and slope at the end.

/// A fitted smoothing spline.
#[derive(Debug, Clone)]
pub(crate) struct SmoothSpline {
    knots: Vec<f64>,
    coef: Vec<f64>,
    min: f64,
    range: f64,
}

/// R `quantile(x, c(0.25, 0.75), type = 7)` difference.
fn iqr(x: &[f64]) -> f64 {
    let mut s = x.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let q = |p: f64| {
        let h = (s.len() - 1) as f64 * p;
        let lo = h.floor() as usize;
        let hi = h.ceil() as usize;
        s[lo] + (h - lo as f64) * (s[hi] - s[lo])
    };
    q(0.75) - q(0.25)
}

/// Values (`deriv = 0`) or derivatives of the four cubic B-splines
/// `B_{l-3}, ..., B_l` that are non-zero on the knot interval
/// `[t_l, t_{l+1})`, using the polynomial piece of that interval at `x`
/// (de Boor; R's `bsplvd`).
fn bspline_piece(t: &[f64], l: usize, x: f64, deriv: usize) -> [f64; 4] {
    // b[i] holds B_{l-3+i, r}; start with order 1: B_{l,1} = 1.
    let mut b = [0.0, 0.0, 0.0, 1.0];
    let get = |b: &[f64; 4], j: usize| if j > l { 0.0 } else { b[j + 3 - l] };
    for r in 1..(4 - deriv) {
        let w = |j: usize| {
            let d = t[j + r] - t[j];
            if d > 0.0 {
                (x - t[j]) / d
            } else {
                0.0
            }
        };
        let old = b;
        for (i, v) in b.iter_mut().enumerate() {
            let j = l - 3 + i;
            *v = w(j) * get(&old, j) + (1.0 - w(j + 1)) * get(&old, j + 1);
        }
    }
    for r in (4 - deriv)..4 {
        let old = b;
        let term = |j: usize| {
            let d = t[j + r] - t[j];
            if d > 0.0 {
                get(&old, j) / d
            } else {
                0.0
            }
        };
        for (i, v) in b.iter_mut().enumerate() {
            let j = l - 3 + i;
            *v = r as f64 * (term(j) - term(j + 1));
        }
    }
    b
}

/// Solve the symmetric positive definite system `a x = b` (Cholesky).
#[allow(clippy::needless_range_loop)]
fn cholesky_solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for j in 0..n {
        let mut d = a[j][j];
        for k in 0..j {
            d -= a[j][k] * a[j][k];
        }
        if d <= 0.0 || !d.is_finite() {
            return None;
        }
        let d = d.sqrt();
        a[j][j] = d;
        for i in (j + 1)..n {
            let mut s = a[i][j];
            for k in 0..j {
                s -= a[i][k] * a[j][k];
            }
            a[i][j] = s / d;
        }
    }
    for i in 0..n {
        for k in 0..i {
            b[i] -= a[i][k] * b[k];
        }
        b[i] /= a[i][i];
    }
    for i in (0..n).rev() {
        for k in (i + 1)..n {
            b[i] -= a[k][i] * b[k];
        }
        b[i] /= a[i][i];
    }
    Some(b)
}

impl SmoothSpline {
    /// Index `l` of the knot interval `[t_l, t_{l+1})` holding the scaled
    /// value `s` in `[0, 1]` (the last interval is closed).
    fn interval(&self, s: f64) -> usize {
        let nk = self.coef.len();
        let interior = &self.knots[4..nk];
        3 + interior.partition_point(|&k| k <= s)
    }

    /// `smooth.spline(x, y, spar = spar)` with unit weights. `x` values
    /// closer than `1e-6 * IQR(x)` are merged (weights add, `y` averaged), as
    /// in R. `None` with fewer than four distinct `x` or a singular system.
    pub(crate) fn fit(x: &[f64], y: &[f64], spar: f64) -> Option<SmoothSpline> {
        let tol = 1e-6 * iqr(x);
        if tol.is_nan() || tol <= 0.0 {
            return None;
        }
        let mean = x.iter().sum::<f64>() / x.len() as f64;
        // Group by round((x - mean) / tol), keeping the first x of a group.
        let mut idx: Vec<usize> = (0..x.len()).collect();
        let key = |i: usize| ((x[i] - mean) / tol).round();
        idx.sort_by(|&a, &b| key(a).total_cmp(&key(b)).then(a.cmp(&b)));
        let (mut ux, mut wbar, mut ybar) = (Vec::new(), Vec::new(), Vec::new());
        let mut last_key = f64::NAN;
        for &i in &idx {
            if key(i) == last_key {
                let k = ux.len() - 1;
                wbar[k] += 1.0;
                ybar[k] += y[i];
            } else {
                last_key = key(i);
                ux.push(x[i]);
                wbar.push(1.0);
                ybar.push(y[i]);
            }
        }
        for (s, w) in ybar.iter_mut().zip(&wbar) {
            *s /= w;
        }
        let nx = ux.len();
        if nx < 4 {
            return None;
        }
        let (min, range) = (ux[0], ux[nx - 1] - ux[0]);
        let xbar: Vec<f64> = ux.iter().map(|v| (v - min) / range).collect();
        let mut knots = vec![xbar[0]; 3];
        knots.extend_from_slice(&xbar);
        knots.extend(std::iter::repeat_n(xbar[nx - 1], 3));
        let nk = nx + 2;
        let mut spline = SmoothSpline {
            knots,
            coef: vec![0.0; nk],
            min,
            range,
        };
        let t = &spline.knots;

        // X'WX and X'Wy.
        let mut xtwx = vec![vec![0.0; nk]; nk];
        let mut xtwy = vec![0.0; nk];
        for ((&s, &w), &yv) in xbar.iter().zip(&wbar).zip(&ybar) {
            let l = spline.interval(s);
            let b = bspline_piece(t, l, s, 0);
            for p in 0..4 {
                xtwy[l - 3 + p] += w * b[p] * yv;
                for q in 0..4 {
                    xtwx[l - 3 + p][l - 3 + q] += w * b[p] * b[q];
                }
            }
        }
        // SIGMA as R's sgram: B'' is linear on each interval.
        let mut sigma = vec![vec![0.0; nk]; nk];
        for l in 3..nk {
            let wpt = t[l + 1] - t[l];
            if wpt <= 0.0 {
                continue;
            }
            let y1 = bspline_piece(t, l, t[l], 2);
            let y2e = bspline_piece(t, l, t[l + 1], 2);
            for p in 0..4 {
                for q in 0..4 {
                    let (a1, a2) = (y1[p], y1[q]);
                    let (b1, b2) = (y2e[p] - a1, y2e[q] - a2);
                    sigma[l - 3 + p][l - 3 + q] +=
                        wpt * (a1 * a2 + (b1 * a2 + b2 * a1) * 0.5 + b1 * b2 * 0.3330);
                }
            }
        }
        let (t1, t2) = (2..nk - 3).fold((0.0, 0.0), |(a, b), i| (a + xtwx[i][i], b + sigma[i][i]));
        let lambda = t1 / t2 * 16f64.powf(spar * 6.0 - 2.0);
        let a: Vec<Vec<f64>> = (0..nk)
            .map(|i| (0..nk).map(|j| xtwx[i][j] + lambda * sigma[i][j]).collect())
            .collect();
        spline.coef = cholesky_solve(a, xtwy)?;
        Some(spline)
    }

    /// Value (`deriv = 0`) or first derivative with respect to the scaled
    /// `s` in `[0, 1]`.
    fn eval_scaled(&self, s: f64, deriv: usize) -> f64 {
        let l = self.interval(s);
        let b = bspline_piece(&self.knots, l, s, deriv);
        (0..4).map(|p| b[p] * self.coef[l - 3 + p]).sum()
    }

    /// `predict(fit, x)$y`.
    pub(crate) fn predict(&self, x: f64) -> f64 {
        let s = (x - self.min) / self.range;
        if s < 0.0 {
            self.eval_scaled(0.0, 0) + self.eval_scaled(0.0, 1) * s
        } else if s > 1.0 {
            self.eval_scaled(1.0, 0) + self.eval_scaled(1.0, 1) * (s - 1.0)
        } else {
            self.eval_scaled(s, 0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bsplines_partition_unity() {
        let t = [0.0, 0.0, 0.0, 0.0, 0.3, 0.6, 1.0, 1.0, 1.0, 1.0];
        for (l, x) in [(3, 0.1), (4, 0.4), (5, 0.99)] {
            let b = bspline_piece(&t, l, x, 0);
            assert!((b.iter().sum::<f64>() - 1.0).abs() < 1e-15);
            let d = bspline_piece(&t, l, x, 1);
            assert!(d.iter().sum::<f64>().abs() < 1e-12);
        }
    }

    /// R reference values from `R/generate_drift_refs.R`.
    fn reference(key: &str) -> Vec<f64> {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/R/data/drift_reference.csv");
        let mut rdr = csv::Reader::from_path(path).unwrap();
        for r in rdr.records() {
            let r = r.unwrap();
            if &r[0] == "smooth_spline" && &r[1] == key {
                return r[2].split(' ').map(|v| v.parse().unwrap()).collect();
            }
        }
        panic!("missing {}", key)
    }

    #[test]
    fn matches_r_smooth_spline() {
        let (x, y, xp) = (reference("in_x"), reference("in_y"), reference("in_xp"));
        for spar in ["0.25", "0.4"] {
            let f = SmoothSpline::fit(&x, &y, spar.parse().unwrap()).unwrap();
            for (v, e) in xp.iter().zip(reference(&format!("pred_{}", spar))) {
                assert!((f.predict(*v) - e).abs() < 1e-9, "{} {}", f.predict(*v), e);
            }
        }
    }

    #[test]
    fn fits_a_line_exactly() {
        // A straight line has zero penalty: the fit reproduces it.
        let x: Vec<f64> = (0..10).map(|i| i as f64 * 0.37).collect();
        let y: Vec<f64> = x.iter().map(|v| 2.0 * v - 1.0).collect();
        let f = SmoothSpline::fit(&x, &y, 0.5).unwrap();
        for v in [-1.0, 0.5, 2.0, 5.0] {
            assert!((f.predict(v) - (2.0 * v - 1.0)).abs() < 1e-10);
        }
    }
}
