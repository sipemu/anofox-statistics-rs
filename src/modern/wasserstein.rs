//! One-dimensional Wasserstein (earth mover's) distance.

use crate::error::{Result, StatError};
use crate::utils::finite::{ensure_finite, ensure_finite_param};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Result of [`wasserstein_test_1d`].
#[derive(Debug, Clone, PartialEq)]
pub struct WassersteinResult {
    /// Wasserstein-1 distance `W_1` between the two samples.
    pub distance: f64,
    /// `W_1 / max(sd(reference), 0.001)` with the population standard
    /// deviation (`ddof = 0`), evidently's drift score.
    pub distance_normed: f64,
    /// Permutation p-value of `W_1`; `None` when `n_permutations = 0`.
    pub p_value: Option<f64>,
    /// Number of permutations used.
    pub n_permutations: usize,
}

fn sorted(v: &[f64]) -> Vec<f64> {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    s
}

/// Wasserstein-`p` distance between the empirical distributions of `x` and
/// `y`: `(int_0^1 |F^-1(u) - G^-1(u)|^p du)^(1/p)` over the empirical
/// quantile functions. For `p = 1` this is the earth mover's distance
/// `int |F(t) - G(t)| dt`.
///
/// O((m + n) log(m + n)). Errors on empty or non-finite input and unless
/// `p >= 1` is finite.
///
/// R equivalent: `transport::wasserstein1d(x, y, p)`; scipy:
/// `scipy.stats.wasserstein_distance(x, y)` for `p = 1`.
///
/// # Example
/// ```
/// use anofox_statistics::wasserstein_distance_1d;
/// let d = wasserstein_distance_1d(&[0.0, 1.0, 3.0], &[5.0, 6.0, 8.0], 1.0).unwrap();
/// assert!((d - 5.0).abs() < 1e-12);
/// ```
pub fn wasserstein_distance_1d(x: &[f64], y: &[f64], p: f64) -> Result<f64> {
    if x.is_empty() || y.is_empty() {
        return Err(StatError::EmptyData);
    }
    ensure_finite("x", x)?;
    ensure_finite("y", y)?;
    ensure_finite_param("p", p)?;
    if p < 1.0 {
        return Err(StatError::InvalidParameter(format!(
            "p must be at least 1, got {}",
            p
        )));
    }
    let (a, b) = (sorted(x), sorted(y));
    let (m, n) = (a.len(), b.len());
    // Merge the quantile breakpoints i/m and j/n; the quantile functions are
    // constant in between. Breakpoints are compared exactly as i n vs j m.
    let (mut i, mut j) = (0usize, 0usize);
    let mut u = 0.0f64;
    let mut total = 0.0f64;
    while i < m && j < n {
        let (ni, nj) = ((i + 1) * n, (j + 1) * m);
        let next = if ni <= nj {
            (i + 1) as f64 / m as f64
        } else {
            (j + 1) as f64 / n as f64
        };
        total += (next - u) * (a[i] - b[j]).abs().powf(p);
        u = next;
        if ni <= nj {
            i += 1;
        }
        if nj <= ni {
            j += 1;
        }
    }
    Ok(total.powf(1.0 / p))
}

/// `W_1` of every split of a pooled sample sorted once: O(N) per split.
struct Pooled {
    z: Vec<f64>,
    idx: Vec<usize>,
    m: usize,
    n: usize,
}

impl Pooled {
    fn new(x: &[f64], y: &[f64]) -> Self {
        let pooled: Vec<f64> = x.iter().chain(y).copied().collect();
        let mut idx: Vec<usize> = (0..pooled.len()).collect();
        idx.sort_by(|&a, &b| pooled[a].total_cmp(&pooled[b]));
        let z = idx.iter().map(|&i| pooled[i]).collect();
        Pooled {
            z,
            idx,
            m: x.len(),
            n: y.len(),
        }
    }

    /// `int |F_x - F_y|` for the split where pooled index `i` belongs to the
    /// first sample iff `in_x[i]`.
    fn w1(&self, in_x: &[bool]) -> f64 {
        let (mut cx, mut cy) = (0usize, 0usize);
        let (mf, nf) = (self.m as f64, self.n as f64);
        let mut total = 0.0;
        for k in 0..self.z.len() - 1 {
            if in_x[self.idx[k]] {
                cx += 1;
            } else {
                cy += 1;
            }
            total += (self.z[k + 1] - self.z[k]) * (cx as f64 / mf - cy as f64 / nf).abs();
        }
        total
    }
}

/// Wasserstein-1 distance between a reference and a current sample, with
/// evidently's normed score and an optional permutation p-value.
///
/// `distance` is `W_1` ([`wasserstein_distance_1d`] with `p = 1`);
/// `distance_normed = W_1 / max(sd(reference), 0.001)` with the population
/// standard deviation (`ddof = 0`, numpy's `std`). evidently's
/// `"wasserstein"` drift test reports `distance_normed` and flags drift at
/// `>= 0.1`; it is evidently's default for large numeric columns.
///
/// With `n_permutations > 0`, `p_value` is the permutation p-value of
/// `W_1`, `(1 + #{W_1* >= W_1}) / (1 + B)` over `B` random splits of the
/// pooled sample (never 0), as the energy test; the pooled sample is sorted
/// once, so each permutation costs O(N). `seed` makes it reproducible
/// (`ChaCha8Rng`). `n_permutations = 0` skips it (`p_value = None`).
///
/// Errors on empty or non-finite input.
///
/// R equivalent: `transport::wasserstein1d(reference, current, p = 1)`;
/// permutation p-value: `twosamples::wass_test(reference, current)` (which
/// reports `#{W_1* >= W_1} / B`).
///
/// # Example
/// ```
/// use anofox_statistics::wasserstein_test_1d;
/// let reference: Vec<f64> = (0..50).map(|i| 2.0 * (0.37 * i as f64).sin()).collect();
/// let current: Vec<f64> = (0..40).map(|i| 2.0 * (0.53 * i as f64).sin() + 0.5).collect();
/// let r = wasserstein_test_1d(&reference, &current, 0, None).unwrap();
/// assert!((r.distance - 0.625792792352).abs() < 1e-11);
/// assert!((r.distance_normed - 0.43940285495).abs() < 1e-10);
/// assert_eq!(r.p_value, None);
/// ```
pub fn wasserstein_test_1d(
    reference: &[f64],
    current: &[f64],
    n_permutations: usize,
    seed: Option<u64>,
) -> Result<WassersteinResult> {
    let distance = wasserstein_distance_1d(reference, current, 1.0)?;
    let nr = reference.len() as f64;
    let mean = reference.iter().sum::<f64>() / nr;
    let sd = (reference.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / nr).sqrt();
    let distance_normed = distance / sd.max(0.001);

    let p_value = (n_permutations > 0).then(|| {
        let pooled = Pooled::new(reference, current);
        let n = reference.len() + current.len();
        let m = reference.len();
        let mut in_x: Vec<bool> = (0..n).map(|i| i < m).collect();
        let observed = pooled.w1(&in_x);
        let range = pooled.z[n - 1] - pooled.z[0];
        let threshold = observed - 64.0 * f64::EPSILON * n as f64 * range;
        let mut rng = match seed {
            Some(s) => ChaCha8Rng::seed_from_u64(s),
            None => ChaCha8Rng::from_entropy(),
        };
        let mut indices: Vec<usize> = (0..n).collect();
        let mut count = 0usize;
        for _ in 0..n_permutations {
            indices.shuffle(&mut rng);
            for (pos, &i) in indices.iter().enumerate() {
                in_x[i] = pos < m;
            }
            if pooled.w1(&in_x) >= threshold {
                count += 1;
            }
        }
        (count as f64 + 1.0) / (n_permutations as f64 + 1.0)
    });

    Ok(WassersteinResult {
        distance,
        distance_normed,
        p_value,
        n_permutations,
    })
}
