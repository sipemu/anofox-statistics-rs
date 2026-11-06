//! Probability calibrators: isotonic regression (scikit-learn's
//! `IsotonicRegression(out_of_bounds = "clip")`, NannyML's CBPE default) and
//! Platt's sigmoid (scikit-learn's `CalibratedClassifierCV(method =
//! "sigmoid")`).

use crate::error::{Result, StatError};
use crate::utils::finite::ensure_finite;

fn check_inputs(scores: &[f64], labels: &[bool]) -> Result<()> {
    if scores.len() != labels.len() {
        return Err(StatError::InvalidParameter(format!(
            "scores ({}) and labels ({}) must have the same length",
            scores.len(),
            labels.len()
        )));
    }
    if scores.is_empty() {
        return Err(StatError::EmptyData);
    }
    ensure_finite("scores", scores)
}

/// An isotonic (non-decreasing) calibration map fitted by
/// [`IsotonicCalibrator::fit`].
///
/// The map is piecewise linear between the knots `(x[i], y[i])` and constant
/// outside `[x[0], x[last]]`. The fields are public so that a fitted map can
/// be stored and rebuilt with [`IsotonicCalibrator::from_parts`].
#[derive(Debug, Clone, PartialEq)]
pub struct IsotonicCalibrator {
    /// Knots: the distinct fitted scores, increasing.
    pub x: Vec<f64>,
    /// Calibrated probability at each knot, non-decreasing.
    pub y: Vec<f64>,
}

impl IsotonicCalibrator {
    /// Fits the isotonic regression of `labels` on `scores` by the
    /// pool-adjacent-violators algorithm: tied scores are first merged into
    /// one point (mean label, weight = count), then adjacent blocks that
    /// violate monotonicity are pooled (weighted means).
    ///
    /// Errors on empty or mismatched inputs and non-finite scores.
    ///
    /// Python equivalent: `IsotonicRegression(out_of_bounds="clip",
    /// increasing=True).fit(scores, labels)`.
    pub fn fit(scores: &[f64], labels: &[bool]) -> Result<Self> {
        check_inputs(scores, labels)?;
        let mut idx: Vec<usize> = (0..scores.len()).collect();
        idx.sort_by(|&a, &b| scores[a].total_cmp(&scores[b]));
        // Merge tied scores.
        let mut xs: Vec<f64> = Vec::new();
        let mut sums: Vec<f64> = Vec::new();
        let mut weights: Vec<f64> = Vec::new();
        for &i in &idx {
            let y = if labels[i] { 1.0 } else { 0.0 };
            if xs.last() == Some(&scores[i]) {
                *sums.last_mut().unwrap() += y;
                *weights.last_mut().unwrap() += 1.0;
            } else {
                xs.push(scores[i]);
                sums.push(y);
                weights.push(1.0);
            }
        }
        // PAVA over blocks (mean, weight, number of points).
        let mut blocks: Vec<(f64, f64, usize)> = Vec::with_capacity(xs.len());
        for (s, w) in sums.iter().zip(&weights) {
            let mut cur = (s / w, *w, 1usize);
            while let Some(&(m, bw, c)) = blocks.last() {
                if m > cur.0 {
                    blocks.pop();
                    let tw = bw + cur.1;
                    cur = ((m * bw + cur.0 * cur.1) / tw, tw, c + cur.2);
                } else {
                    break;
                }
            }
            blocks.push(cur);
        }
        let mut y = Vec::with_capacity(xs.len());
        for (m, _, c) in blocks {
            y.extend(std::iter::repeat_n(m, c));
        }
        Ok(Self { x: xs, y })
    }

    /// Rebuilds a fitted map from its knots. Errors when the lengths differ
    /// or are zero, `x` is not strictly increasing, `y` is decreasing, or a
    /// value is not finite.
    pub fn from_parts(x: Vec<f64>, y: Vec<f64>) -> Result<Self> {
        if x.is_empty() || x.len() != y.len() {
            return Err(StatError::InvalidParameter(format!(
                "isotonic knots need equal, non-zero lengths, got {} and {}",
                x.len(),
                y.len()
            )));
        }
        ensure_finite("x", &x)?;
        ensure_finite("y", &y)?;
        if x.windows(2).any(|w| w[1] <= w[0]) || y.windows(2).any(|w| w[1] < w[0]) {
            return Err(StatError::InvalidParameter(
                "isotonic knots must have increasing x and non-decreasing y".into(),
            ));
        }
        Ok(Self { x, y })
    }

    /// Calibrated probability of one score: linear interpolation between the
    /// knots, clipped to the end knots outside their range (`np.interp`).
    /// NaN maps to NaN.
    pub fn calibrate_one(&self, score: f64) -> f64 {
        if score.is_nan() {
            return f64::NAN;
        }
        let n = self.x.len();
        if score <= self.x[0] {
            return self.y[0];
        }
        if score >= self.x[n - 1] {
            return self.y[n - 1];
        }
        // First knot > score; 1 <= j <= n - 1.
        let j = self.x.partition_point(|&v| v <= score);
        let (x0, x1, y0, y1) = (self.x[j - 1], self.x[j], self.y[j - 1], self.y[j]);
        y0 + (score - x0) * (y1 - y0) / (x1 - x0)
    }

    /// [`calibrate_one`](Self::calibrate_one) of each score.
    pub fn calibrate(&self, scores: &[f64]) -> Vec<f64> {
        scores.iter().map(|&s| self.calibrate_one(s)).collect()
    }
}

/// Platt's sigmoid calibration map `p = 1 / (1 + exp(a * score + b))`,
/// fitted by [`PlattCalibrator::fit`] (scikit-learn's sign convention: `a`
/// is negative for an informative score).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlattCalibrator {
    pub a: f64,
    pub b: f64,
}

impl PlattCalibrator {
    /// Fits Platt's sigmoid by maximum likelihood on the smoothed targets
    /// `t+ = (N+ + 1) / (N+ + 2)` for positives and `t- = 1 / (N- + 2)` for
    /// negatives (Platt 1999), with Newton's method and a backtracking line
    /// search (Lin, Lin & Weng 2007), to convergence.
    ///
    /// Errors on empty or mismatched inputs, non-finite scores, and when the
    /// optimisation does not converge.
    ///
    /// Python equivalent: `sklearn.calibration._sigmoid_calibration(scores,
    /// labels)` (L-BFGS-B, `gtol = 1e-6`, so it agrees to about 1e-6). R
    /// equivalent: `glm(t ~ score, family = quasibinomial)` on the smoothed
    /// targets, with `a = -slope`, `b = -intercept`.
    pub fn fit(scores: &[f64], labels: &[bool]) -> Result<Self> {
        check_inputs(scores, labels)?;
        let n1 = labels.iter().filter(|&&l| l).count() as f64;
        let n0 = labels.len() as f64 - n1;
        let hi = (n1 + 1.0) / (n1 + 2.0);
        let lo = 1.0 / (n0 + 2.0);
        let t: Vec<f64> = labels.iter().map(|&l| if l { hi } else { lo }).collect();
        // Negative log-likelihood of (a, b), computed stably.
        let nll = |a: f64, b: f64| -> f64 {
            scores
                .iter()
                .zip(&t)
                .map(|(&f, &ti)| {
                    let z = a * f + b; // p = 1 / (1 + e^z)
                                       // -[t log p + (1 - t) log(1 - p)] = t log(1 + e^z) + (1 - t) log(1 + e^-z)
                    let l1 = softplus(z);
                    let l0 = softplus(-z);
                    ti * l1 + (1.0 - ti) * l0
                })
                .sum()
        };
        let mut a = 0.0;
        let mut b = ((n0 + 1.0) / (n1 + 1.0)).ln();
        let mut f = nll(a, b);
        for _ in 0..200 {
            // gradient and Hessian (with a tiny ridge, Lin et al.)
            let (mut ga, mut gb, mut h11, mut h22, mut h21) = (0.0, 0.0, 1e-12, 1e-12, 0.0);
            for (&fi, &ti) in scores.iter().zip(&t) {
                let z = a * fi + b;
                let p = expit(-z);
                let q = 1.0 - p;
                let d1 = ti - p; // dL/dz
                let d2 = p * q;
                ga += fi * d1;
                gb += d1;
                h11 += fi * fi * d2;
                h22 += d2;
                h21 += fi * d2;
            }
            if ga.abs() < 1e-11 * (1.0 + f) && gb.abs() < 1e-11 * (1.0 + f) {
                return Ok(Self { a, b });
            }
            let det = h11 * h22 - h21 * h21;
            let da = -(h22 * ga - h21 * gb) / det;
            let db = -(-h21 * ga + h11 * gb) / det;
            let gd = ga * da + gb * db;
            let mut step = 1.0;
            loop {
                let (na, nb) = (a + step * da, b + step * db);
                let nf = nll(na, nb);
                if nf < f + 1e-4 * step * gd || step < 1e-10 {
                    if (f - nf).abs() <= 1e-15 * f.abs().max(1.0) && step >= 1.0 {
                        return Ok(Self { a: na, b: nb });
                    }
                    a = na;
                    b = nb;
                    f = nf;
                    break;
                }
                step *= 0.5;
            }
            if step < 1e-10 {
                return Ok(Self { a, b });
            }
        }
        Err(StatError::InvalidParameter(
            "Platt calibration did not converge".into(),
        ))
    }

    /// Calibrated probability `1 / (1 + exp(a * score + b))`; NaN maps to NaN.
    pub fn calibrate_one(&self, score: f64) -> f64 {
        expit(-(self.a * score + self.b))
    }

    /// [`calibrate_one`](Self::calibrate_one) of each score.
    pub fn calibrate(&self, scores: &[f64]) -> Vec<f64> {
        scores.iter().map(|&s| self.calibrate_one(s)).collect()
    }
}

/// `log(1 + e^z)` without overflow.
fn softplus(z: f64) -> f64 {
    if z > 0.0 {
        z + (-z).exp().ln_1p()
    } else {
        z.exp().ln_1p()
    }
}

/// Logistic function `1 / (1 + e^-z)`.
fn expit(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}
