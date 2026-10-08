use crate::error::{Result, StatError};
use crate::utils::finite::ensure_finite;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Result of the Energy Distance test
#[derive(Debug, Clone)]
pub struct EnergyDistanceResult {
    /// The energy distance statistic
    pub statistic: f64,
    /// The p-value (from permutation test)
    pub p_value: f64,
    /// Number of permutations used
    pub n_permutations: usize,
}

/// Compute the Euclidean distance between two points.
#[inline]
fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Compute mean pairwise distance between two samples.
fn mean_pairwise_distance(x: &[&[f64]], y: &[&[f64]]) -> f64 {
    let mut sum = 0.0;
    for xi in x.iter() {
        for yj in y.iter() {
            sum += euclidean_distance(xi, yj);
        }
    }
    sum / (x.len() as f64 * y.len() as f64)
}

/// Compute mean within-sample distance (excluding diagonal).
fn mean_within_distance(samples: &[&[f64]]) -> f64 {
    let n = samples.len();
    if n < 2 {
        return 0.0;
    }

    let mut sum = 0.0;
    for i in 0..n {
        for j in 0..n {
            if i != j {
                sum += euclidean_distance(samples[i], samples[j]);
            }
        }
    }
    let n_f = n as f64;
    sum / (n_f * (n_f - 1.0))
}

/// Compute the energy distance statistic between two samples.
///
/// E(X,Y) = 2*E|X-Y| - E|X-X'| - E|Y-Y'|
/// where X,X' are iid from first distribution, Y,Y' from second.
fn energy_distance_statistic(x: &[&[f64]], y: &[&[f64]]) -> f64 {
    if x.is_empty() || y.is_empty() {
        return 0.0;
    }

    let mean_xy = mean_pairwise_distance(x, y);
    let mean_xx = mean_within_distance(x);
    let mean_yy = mean_within_distance(y);

    2.0 * mean_xy - mean_xx - mean_yy
}

/// Validate energy distance inputs and return the dimension.
fn validate_energy_inputs(x: &[Vec<f64>], y: &[Vec<f64>]) -> Result<()> {
    if x.is_empty() || y.is_empty() {
        return Err(StatError::EmptyData);
    }

    let dim = x[0].len();
    if dim == 0 {
        return Err(StatError::InvalidParameter(
            "Data points must have at least one dimension".to_string(),
        ));
    }

    for (i, v) in x.iter().enumerate() {
        ensure_finite(&format!("x[{}]", i), v)?;
    }
    for (i, v) in y.iter().enumerate() {
        ensure_finite(&format!("y[{}]", i), v)?;
    }

    let all_same_dim = x.iter().chain(y.iter()).all(|v| v.len() == dim);
    if !all_same_dim {
        return Err(StatError::InvalidParameter(
            "All data points must have the same dimension".to_string(),
        ));
    }

    Ok(())
}

/// Run permutation test for energy distance.
fn run_energy_permutation_test(
    x: &[Vec<f64>],
    y: &[Vec<f64>],
    observed: f64,
    n_permutations: usize,
    seed: Option<u64>,
) -> f64 {
    let combined: Vec<Vec<f64>> = x.iter().chain(y.iter()).cloned().collect();
    let n1 = x.len();

    let mut rng = match seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => ChaCha8Rng::from_entropy(),
    };

    let mut indices: Vec<usize> = (0..combined.len()).collect();
    let mut count_extreme = 0usize;

    for _ in 0..n_permutations {
        indices.shuffle(&mut rng);

        let perm_x: Vec<&[f64]> = indices[0..n1]
            .iter()
            .map(|&i| combined[i].as_slice())
            .collect();
        let perm_y: Vec<&[f64]> = indices[n1..]
            .iter()
            .map(|&i| combined[i].as_slice())
            .collect();

        let perm_stat = energy_distance_statistic(&perm_x, &perm_y);

        if perm_stat >= observed {
            count_extreme += 1;
        }
    }

    (count_extreme as f64 + 1.0) / (n_permutations as f64 + 1.0)
}

/// Pooled univariate sample sorted once, for O(N) evaluation of the energy
/// statistic under any split into two groups.
struct Pooled1d {
    /// Pooled values (centred at the pooled mean) in ascending order.
    z: Vec<f64>,
    /// Pooled index (0..n1 = x, n1.. = y) of each sorted value.
    idx: Vec<usize>,
    n1: usize,
    n2: usize,
}

impl Pooled1d {
    /// O(N log N) time, O(N) memory.
    fn new(x: &[f64], y: &[f64]) -> Self {
        let n = x.len() + y.len();
        let mean = (x.iter().sum::<f64>() + y.iter().sum::<f64>()) / n as f64;
        let pooled: Vec<f64> = x.iter().chain(y.iter()).map(|v| v - mean).collect();
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_unstable_by(|&a, &b| pooled[a].total_cmp(&pooled[b]));
        let z = idx.iter().map(|&i| pooled[i]).collect();
        Self {
            z,
            idx,
            n1: x.len(),
            n2: y.len(),
        }
    }

    /// Energy statistic `2 E|X-Y| - E|X-X'| - E|Y-Y'|` (within-sample means
    /// exclude the diagonal) for the split where pooled index `i` belongs to
    /// the first sample iff `in_x[i]`.
    ///
    /// One sweep over the sorted values: for each value, the summed distance
    /// to all smaller values of a group is `count * z - sum`. O(N) time.
    fn statistic(&self, in_x: &[bool]) -> f64 {
        let (mut cx, mut sx, mut cy, mut sy) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        let (mut sxx, mut syy, mut sxy) = (0.0f64, 0.0f64, 0.0f64);
        for (&v, &i) in self.z.iter().zip(&self.idx) {
            if in_x[i] {
                sxx += cx * v - sx;
                sxy += cy * v - sy;
                cx += 1.0;
                sx += v;
            } else {
                syy += cy * v - sy;
                sxy += cx * v - sx;
                cy += 1.0;
                sy += v;
            }
        }
        let (n1, n2) = (self.n1 as f64, self.n2 as f64);
        let mean_xy = sxy / (n1 * n2);
        let mean_xx = if self.n1 < 2 {
            0.0
        } else {
            2.0 * sxx / (n1 * (n1 - 1.0))
        };
        let mean_yy = if self.n2 < 2 {
            0.0
        } else {
            2.0 * syy / (n2 * (n2 - 1.0))
        };
        2.0 * mean_xy - mean_xx - mean_yy
    }

    /// Absolute tolerance for treating a permutation statistic as tied with
    /// the observed one (rounding level of the sweep).
    fn tie_tolerance(&self) -> f64 {
        let range = self.z.last().unwrap_or(&0.0) - self.z.first().unwrap_or(&0.0);
        64.0 * f64::EPSILON * (self.n1 + self.n2) as f64 * range
    }
}

/// Univariate energy test: O(N log N + B N) time, O(N) memory.
fn energy_test_univariate(
    x: &[f64],
    y: &[f64],
    n_permutations: usize,
    seed: Option<u64>,
) -> EnergyDistanceResult {
    let pooled = Pooled1d::new(x, y);
    let n = x.len() + y.len();
    let n1 = x.len();
    let mut in_x: Vec<bool> = (0..n).map(|i| i < n1).collect();
    let observed = pooled.statistic(&in_x);
    let threshold = observed - pooled.tie_tolerance();

    // Same RNG stream and shuffles as the multivariate path.
    let mut rng = match seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => ChaCha8Rng::from_entropy(),
    };
    let mut indices: Vec<usize> = (0..n).collect();
    let mut count_extreme = 0usize;
    for _ in 0..n_permutations {
        indices.shuffle(&mut rng);
        for (pos, &i) in indices.iter().enumerate() {
            in_x[i] = pos < n1;
        }
        if pooled.statistic(&in_x) >= threshold {
            count_extreme += 1;
        }
    }

    EnergyDistanceResult {
        statistic: observed,
        p_value: (count_extreme as f64 + 1.0) / (n_permutations as f64 + 1.0),
        n_permutations,
    }
}

/// Perform the Energy Distance test for equality of distributions.
///
/// This is a powerful non-parametric two-sample test that can detect
/// differences in both location and shape of distributions.
///
/// For univariate data, pass single-element slices.
/// For multivariate data, pass vectors of the same dimension.
///
/// # Arguments
/// * `x` - First sample (each element is a d-dimensional observation)
/// * `y` - Second sample (each element is a d-dimensional observation)
/// * `n_permutations` - Number of permutations for p-value estimation
/// * `seed` - Optional random seed for reproducibility
///
/// # Returns
/// * `EnergyDistanceResult` containing energy distance and p-value
///
/// # References
/// * Székely, G.J. and Rizzo, M.L. (2004). "Testing for equal distributions in high dimension"
/// * Székely, G.J. and Rizzo, M.L. (2013). "Energy statistics: A class of statistics based on distances"
///
/// # Complexity
/// For `N = m + n` points and `B = n_permutations`:
/// * univariate (`d = 1`): O(N log N + B N) time, O(N) memory — the pooled
///   sample is sorted once and each statistic is a single prefix-sum sweep;
/// * multivariate: inherently O((B + 1) N^2 d) time; distances are streamed,
///   so memory is O(N d) (no distance matrix is stored).
///
/// In the univariate path, permutation statistics within rounding error
/// (`64 eps N range`) of the observed one count as at least as extreme.
pub fn energy_distance_test(
    x: &[Vec<f64>],
    y: &[Vec<f64>],
    n_permutations: usize,
    seed: Option<u64>,
) -> Result<EnergyDistanceResult> {
    validate_energy_inputs(x, y)?;

    if x[0].len() == 1 {
        let x1: Vec<f64> = x.iter().map(|v| v[0]).collect();
        let y1: Vec<f64> = y.iter().map(|v| v[0]).collect();
        return Ok(energy_test_univariate(&x1, &y1, n_permutations, seed));
    }

    // Convert to slices and compute observed statistic
    let x_slices: Vec<&[f64]> = x.iter().map(|v| v.as_slice()).collect();
    let y_slices: Vec<&[f64]> = y.iter().map(|v| v.as_slice()).collect();
    let observed = energy_distance_statistic(&x_slices, &y_slices);

    // Run permutation test
    let p_value = run_energy_permutation_test(x, y, observed, n_permutations, seed);

    Ok(EnergyDistanceResult {
        statistic: observed,
        p_value,
        n_permutations,
    })
}

/// Convenience function for univariate energy distance test.
///
/// # Arguments
/// * `x` - First sample (univariate)
/// * `y` - Second sample (univariate)
/// * `n_permutations` - Number of permutations
/// * `seed` - Optional random seed
///
/// # Complexity
/// O(N log N + B N) time and O(N) memory for `N = m + n`,
/// `B = n_permutations`.
pub fn energy_distance_test_1d(
    x: &[f64],
    y: &[f64],
    n_permutations: usize,
    seed: Option<u64>,
) -> Result<EnergyDistanceResult> {
    if x.is_empty() || y.is_empty() {
        return Err(StatError::EmptyData);
    }
    ensure_finite("x", x)?;
    ensure_finite("y", y)?;
    Ok(energy_test_univariate(x, y, n_permutations, seed))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Naive O(N^2)-per-statistic reference (the previous implementation).
    fn naive(x: &[f64], y: &[f64], b: usize, seed: u64) -> (f64, f64) {
        let xv: Vec<Vec<f64>> = x.iter().map(|&v| vec![v]).collect();
        let yv: Vec<Vec<f64>> = y.iter().map(|&v| vec![v]).collect();
        let xs: Vec<&[f64]> = xv.iter().map(|v| v.as_slice()).collect();
        let ys: Vec<&[f64]> = yv.iter().map(|v| v.as_slice()).collect();
        let obs = energy_distance_statistic(&xs, &ys);
        (
            obs,
            run_energy_permutation_test(&xv, &yv, obs, b, Some(seed)),
        )
    }

    #[test]
    fn test_univariate_matches_naive() {
        let mut s = 41u64;
        for &(n1, n2) in &[(1usize, 2usize), (2, 2), (5, 9), (30, 21), (120, 80)] {
            for &(levels, shift) in &[(0usize, 0.0), (0, 0.7), (4, 0.0), (6, 1.0)] {
                let mut draw = |sh: f64| {
                    let u = lcg(&mut s);
                    sh + if levels == 0 {
                        u * 3.0 + 100.0
                    } else {
                        (u * levels as f64).floor()
                    }
                };
                let x: Vec<f64> = (0..n1).map(|_| draw(0.0)).collect();
                let y: Vec<f64> = (0..n2).map(|_| draw(shift)).collect();
                let r = energy_distance_test_1d(&x, &y, 59, Some(3)).unwrap();
                let (obs, p) = naive(&x, &y, 59, 3);
                let scale = obs.abs().max(1.0);
                assert!(
                    (r.statistic - obs).abs() <= 1e-10 * scale,
                    "{} vs {obs}",
                    r.statistic
                );
                if levels == 0 {
                    assert_eq!(r.p_value, p, "n1={n1} n2={n2}");
                }
            }
        }
    }

    /// n = 100_000 per group with 99 permutations (previously ~4e12 distance
    /// evaluations).
    #[test]
    #[ignore]
    fn test_energy_1d_large_n() {
        let mut s = 8u64;
        let x: Vec<f64> = (0..100_000).map(|_| lcg(&mut s)).collect();
        let y: Vec<f64> = (0..100_000).map(|_| lcg(&mut s) + 0.05).collect();
        let r = energy_distance_test_1d(&x, &y, 99, Some(1)).unwrap();
        assert!(r.statistic > 0.0 && r.p_value <= 0.01 + 1e-12);
    }

    #[test]
    fn test_energy_distance_different_distributions() {
        // Clearly different samples
        let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5, 5.5];
        let y: Vec<f64> = vec![10.0, 11.0, 12.0, 13.0, 14.0, 10.5, 11.5, 12.5, 13.5, 14.5];

        let result = energy_distance_test_1d(&x, &y, 999, Some(42)).unwrap();

        assert!(result.statistic > 0.0);
        assert!(
            result.p_value < 0.05,
            "p_value {} should be < 0.05",
            result.p_value
        );
    }

    #[test]
    fn test_energy_distance_similar_distributions() {
        // Similar samples
        let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y: Vec<f64> = vec![1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1];

        let result = energy_distance_test_1d(&x, &y, 999, Some(42)).unwrap();

        // Should not detect significant difference
        assert!(
            result.p_value > 0.1,
            "p_value {} should be > 0.1",
            result.p_value
        );
    }

    #[test]
    fn test_energy_distance_multivariate() {
        // 2D data with clearly different distributions
        let x = vec![
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![3.0, 3.0],
            vec![1.5, 1.5],
            vec![2.5, 2.5],
        ];
        let y = vec![
            vec![10.0, 10.0],
            vec![11.0, 11.0],
            vec![12.0, 12.0],
            vec![10.5, 10.5],
            vec![11.5, 11.5],
        ];

        let result = energy_distance_test(&x, &y, 499, Some(42)).unwrap();

        assert!(
            result.p_value < 0.05,
            "p_value {} should be < 0.05",
            result.p_value
        );
    }

    #[test]
    fn test_energy_distance_empty() {
        let x: Vec<f64> = vec![];
        let y = vec![1.0, 2.0, 3.0];

        assert!(energy_distance_test_1d(&x, &y, 100, None).is_err());
    }
}
