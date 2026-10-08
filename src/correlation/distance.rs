//! Distance correlation and distance covariance.
//!
//! Distance correlation is a measure of dependence between random vectors
//! that is zero if and only if the vectors are independent.

use crate::error::{Result, StatError};

/// Result of a distance correlation test
#[derive(Debug, Clone)]
pub struct DistanceCorResult {
    /// Distance correlation coefficient (0 to 1)
    pub dcor: f64,
    /// Distance covariance
    pub dcov: f64,
    /// Distance variance of X
    pub dvar_x: f64,
    /// Distance variance of Y
    pub dvar_y: f64,
    /// Test statistic (for permutation test)
    pub statistic: f64,
    /// p-value (from permutation test, if computed)
    pub p_value: Option<f64>,
    /// Number of observations
    pub n: usize,
    /// Method name
    pub method: String,
}

/// Compute distance correlation between two vectors.
///
/// Distance correlation is a measure of dependence between random variables
/// that is zero if and only if the variables are independent. Unlike Pearson
/// correlation, it can detect non-linear dependencies.
///
/// # Arguments
/// * `x` - First variable
/// * `y` - Second variable
///
/// # Returns
/// * `DistanceCorResult` containing dCor, dCov, and dVar values
///
/// # Examples
/// ```
/// use anofox_statistics::correlation::distance_cor;
///
/// let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
/// let y = vec![1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0, 100.0]; // y = x^2
///
/// let result = distance_cor(&x, &y).unwrap();
/// println!("Distance correlation = {:.4}", result.dcor);
/// ```
///
/// # R equivalent
/// `energy::dcor(x, y)`
///
/// # Complexity
/// O(n log n) time and O(n) memory (Huo & Székely 2016): no n x n distance
/// matrix is formed. The double-centred V-statistic is expanded as
/// `n^2 dCov^2 = S_ab - (2/n) sum_i a_i. b_i. + a.. b.. / n^2`, where the
/// row sums `a_i.` come from sorted prefix sums and the cross term
/// `S_ab = sum_ij |x_i - x_j| |y_i - y_j|` from a Fenwick tree over y ranks
/// while sweeping x in sorted order.
pub fn distance_cor(x: &[f64], y: &[f64]) -> Result<DistanceCorResult> {
    validate_input(x, y)?;

    let n = x.len();
    let mx = Marginal::new(x);
    let my = Marginal::new(y);
    let identity: Vec<usize> = (0..n).collect();
    let dcov_sq = dcov_sq_permuted(&mx, &my, &identity);

    Ok(build_result(n, dcov_sq, mx.dvar_sq, my.dvar_sq))
}

/// Assemble a [`DistanceCorResult`] from the squared V-statistics.
fn build_result(n: usize, dcov_sq: f64, dvar_x_sq: f64, dvar_y_sq: f64) -> DistanceCorResult {
    let dcov = dcov_sq.max(0.0).sqrt();
    let dvar_x = dvar_x_sq.max(0.0).sqrt();
    let dvar_y = dvar_y_sq.max(0.0).sqrt();

    // Distance correlation
    let dcor = if dvar_x > 0.0 && dvar_y > 0.0 {
        (dcov_sq / (dvar_x_sq * dvar_y_sq).sqrt()).max(0.0).sqrt()
    } else {
        0.0
    };

    // Clamp to [0, 1]
    let dcor = dcor.clamp(0.0, 1.0);

    // Test statistic (n * dCov^2 is asymptotically chi-squared under independence)
    let statistic = n as f64 * dcov_sq;

    DistanceCorResult {
        dcor,
        dcov,
        dvar_x,
        dvar_y,
        statistic,
        p_value: None,
        n,
        method: "Distance correlation".to_string(),
    }
}

/// Compute distance correlation with permutation test for significance.
///
/// # Arguments
/// * `x` - First variable
/// * `y` - Second variable
/// * `n_permutations` - Number of permutations for the test
/// * `seed` - Optional seed for reproducibility
///
/// # Returns
/// * `DistanceCorResult` with p-value from permutation test
///
/// # R equivalent
/// `energy::dcor.test(x, y, R = n_permutations)`
///
/// # Complexity
/// O((B + 1) n log n) time for `B = n_permutations` and O(n) memory; the
/// sorted order, row sums and ranks of each margin are computed once and only
/// the cross terms are recomputed per permutation.
///
/// Permutation statistics within `64 eps n dVar_x dVar_y` of the observed
/// statistic (rounding-level ties) count as at least as extreme.
pub fn distance_cor_test(
    x: &[f64],
    y: &[f64],
    n_permutations: usize,
    seed: Option<u64>,
) -> Result<DistanceCorResult> {
    validate_input(x, y)?;

    let n = x.len();

    // Marginal quantities do not change under permutation of y.
    let mx = Marginal::new(x);
    let my = Marginal::new(y);
    let mut perm: Vec<usize> = (0..n).collect();
    let observed = build_result(n, dcov_sq_permuted(&mx, &my, &perm), mx.dvar_sq, my.dvar_sq);
    let observed_stat = observed.statistic;
    // Permutations that are mathematically tied with the observed statistic
    // (e.g. swapping y between tied x values) can differ by rounding only;
    // count them as at least as extreme.
    let tol = tie_tolerance(n, mx.dvar_sq, my.dvar_sq);

    // Permutation test: permuting the index vector with the same Fisher-Yates
    // draws reproduces exactly the y permutations of the previous
    // implementation (which shuffled a copy of y).
    let mut rng = SimpleRng::new(seed.unwrap_or(12345));
    let mut count_greater = 0usize;

    for _ in 0..n_permutations {
        fisher_yates_shuffle(&mut perm, &mut rng);

        let perm_stat = n as f64 * dcov_sq_permuted(&mx, &my, &perm);
        if perm_stat >= observed_stat - tol {
            count_greater += 1;
        }
    }

    let p_value = (count_greater as f64 + 1.0) / (n_permutations as f64 + 1.0);

    Ok(DistanceCorResult {
        dcor: observed.dcor,
        dcov: observed.dcov,
        dvar_x: observed.dvar_x,
        dvar_y: observed.dvar_y,
        statistic: observed_stat,
        p_value: Some(p_value),
        n,
        method: format!(
            "Distance correlation test ({} permutations)",
            n_permutations
        ),
    })
}

/// Absolute tolerance below which a permutation statistic `n dCov^2` is
/// considered tied with the observed one: `64 eps n sqrt(dVar_x^2 dVar_y^2)`,
/// the order of the rounding error of the O(n log n) expansion.
fn tie_tolerance(n: usize, dvar_x_sq: f64, dvar_y_sq: f64) -> f64 {
    64.0 * f64::EPSILON * n as f64 * (dvar_x_sq.max(0.0) * dvar_y_sq.max(0.0)).sqrt()
}

/// Validate input vectors for distance correlation.
fn validate_input(x: &[f64], y: &[f64]) -> Result<()> {
    if x.is_empty() || y.is_empty() {
        return Err(StatError::EmptyData);
    }

    if x.len() != y.len() {
        return Err(StatError::InvalidParameter(format!(
            "x and y must have same length: {} vs {}",
            x.len(),
            y.len()
        )));
    }

    if x.len() < 3 {
        return Err(StatError::InsufficientData {
            needed: 3,
            got: x.len(),
        });
    }

    // Check for non-finite values
    for (i, (&xi, &yi)) in x.iter().zip(y.iter()).enumerate() {
        if !xi.is_finite() || !yi.is_finite() {
            return Err(StatError::InvalidParameter(format!(
                "Non-finite value at index {}: x={}, y={}",
                i, xi, yi
            )));
        }
    }

    Ok(())
}

/// Per-variable quantities of the distance matrix `a_ij = |v_i - v_j|`,
/// computed without materialising it.
struct Marginal {
    /// Values centred at their mean (distances are shift invariant; centring
    /// reduces cancellation in the cross-term expansion).
    centered: Vec<f64>,
    /// Row sums `a_i. = sum_j |v_i - v_j|`.
    row_sums: Vec<f64>,
    /// Grand sum `a.. = sum_ij |v_i - v_j|`.
    total: f64,
    /// Indices sorting the values ascending.
    order: Vec<usize>,
    /// Dense rank (0-based) of each value among the distinct values.
    rank: Vec<usize>,
    /// Number of distinct values.
    n_ranks: usize,
    /// Squared distance variance (V-statistic) `dVar^2`.
    dvar_sq: f64,
}

impl Marginal {
    /// O(n log n) time, O(n) memory.
    fn new(v: &[f64]) -> Self {
        let n = v.len();
        let n_f = n as f64;
        let mean = v.iter().sum::<f64>() / n_f;
        let centered: Vec<f64> = v.iter().map(|&vi| vi - mean).collect();

        let mut order: Vec<usize> = (0..n).collect();
        order.sort_unstable_by(|&i, &j| centered[i].total_cmp(&centered[j]));

        // Row sums from prefix sums over the sorted values:
        // a_(k). = v_(k) k - P_k + (T - P_k - v_(k)) - v_(k) (n - k - 1)
        let grand: f64 = centered.iter().sum();
        let mut row_sums = vec![0.0; n];
        let mut rank = vec![0usize; n];
        let mut prefix = 0.0;
        let mut r = 0usize;
        for (k, &i) in order.iter().enumerate() {
            let vk = centered[i];
            if k > 0 && vk != centered[order[k - 1]] {
                r += 1;
            }
            rank[i] = r;
            let below = vk * k as f64 - prefix;
            let above = (grand - prefix - vk) - vk * (n - k - 1) as f64;
            row_sums[i] = below + above;
            prefix += vk;
        }
        let total: f64 = row_sums.iter().sum();

        // sum_ij a_ij^2 = 2 n sum v_i^2 - 2 (sum v_i)^2
        let sum_sq: f64 = centered.iter().map(|c| c * c).sum();
        let sum_a2 = 2.0 * n_f * sum_sq - 2.0 * grand * grand;
        let sum_row2: f64 = row_sums.iter().map(|a| a * a).sum();
        let dvar_sq = (sum_a2 - 2.0 / n_f * sum_row2 + total * total / (n_f * n_f)) / (n_f * n_f);

        Self {
            centered,
            row_sums,
            total,
            order,
            rank,
            n_ranks: r + 1,
            dvar_sq,
        }
    }
}

/// Fenwick (binary indexed) tree over y ranks holding, per rank, the count
/// and the sums of y, x and x*y of the points inserted so far.
struct Fenwick {
    tree: Vec<[f64; 4]>,
}

// `isolate_lowest_one` is too recent for the supported toolchains.
#[allow(unknown_lints, clippy::manual_isolate_lowest_one)]
impl Fenwick {
    fn new(n: usize) -> Self {
        Self {
            tree: vec![[0.0; 4]; n + 1],
        }
    }

    fn add(&mut self, rank: usize, val: [f64; 4]) {
        let mut i = rank + 1;
        while i < self.tree.len() {
            for (t, v) in self.tree[i].iter_mut().zip(val.iter()) {
                *t += v;
            }
            i += i & i.wrapping_neg();
        }
    }

    /// Sums over ranks `< rank`.
    fn prefix(&self, rank: usize) -> [f64; 4] {
        let mut acc = [0.0; 4];
        let mut i = rank;
        while i > 0 {
            for (a, t) in acc.iter_mut().zip(self.tree[i].iter()) {
                *a += t;
            }
            i -= i & i.wrapping_neg();
        }
        acc
    }
}

/// Squared distance covariance (V-statistic) between `x` and `y[perm]`.
///
/// `n^2 dCov^2 = S_ab - (2/n) sum_i a_i. b_i. + a.. b.. / n^2` with
/// `S_ab = sum_ij |x_i - x_j| |y_i - y_j|` computed by sweeping the points in
/// increasing x and querying a Fenwick tree over y ranks (Huo & Székely 2016).
///
/// O(n log n) time, O(n) memory.
fn dcov_sq_permuted(mx: &Marginal, my: &Marginal, perm: &[usize]) -> f64 {
    let n = mx.centered.len();
    let n_f = n as f64;

    let mut fen = Fenwick::new(my.n_ranks);
    let mut tot = [0.0f64; 4];
    let mut half = 0.0;
    for &i in &mx.order {
        let xi = mx.centered[i];
        let pi = perm[i];
        let yi = my.centered[pi];
        let r = my.rank[pi];
        // Earlier points have x_j <= x_i, so |x_i - x_j| = x_i - x_j; split
        // them by y_j < y_i (lo) and y_j >= y_i (hi; equal y contributes 0).
        let lo = fen.prefix(r);
        let hi = [
            tot[0] - lo[0],
            tot[1] - lo[1],
            tot[2] - lo[2],
            tot[3] - lo[3],
        ];
        let expand = |s: &[f64; 4]| s[0] * xi * yi - xi * s[1] - yi * s[2] + s[3];
        half += expand(&lo) - expand(&hi);
        let val = [1.0, yi, xi, xi * yi];
        fen.add(r, val);
        for (t, v) in tot.iter_mut().zip(val.iter()) {
            *t += v;
        }
    }
    let s_ab = 2.0 * half;

    let cross_rows: f64 = mx
        .row_sums
        .iter()
        .zip(perm.iter())
        .map(|(a, &p)| a * my.row_sums[p])
        .sum();

    (s_ab - 2.0 / n_f * cross_rows + mx.total * my.total / (n_f * n_f)) / (n_f * n_f)
}

/// Simple random number generator (xorshift64).
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_usize(&mut self, max: usize) -> usize {
        (self.next_u64() as usize) % max
    }
}

/// Fisher-Yates shuffle.
fn fisher_yates_shuffle(arr: &mut [usize], rng: &mut SimpleRng) {
    let n = arr.len();
    for i in (1..n).rev() {
        let j = rng.next_usize(i + 1);
        arr.swap(i, j);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Naive O(n^2)-memory reference (the previous implementation).
    fn naive(x: &[f64], y: &[f64]) -> (f64, f64, f64) {
        let n = x.len();
        let dm = |v: &[f64]| -> Vec<Vec<f64>> {
            (0..n)
                .map(|i| (0..n).map(|j| (v[i] - v[j]).abs()).collect())
                .collect()
        };
        let center = |a: &[Vec<f64>]| -> Vec<Vec<f64>> {
            let rm: Vec<f64> = a.iter().map(|r| r.iter().sum::<f64>() / n as f64).collect();
            let gm = rm.iter().sum::<f64>() / n as f64;
            (0..n)
                .map(|i| (0..n).map(|j| a[i][j] - rm[i] - rm[j] + gm).collect())
                .collect()
        };
        let dot = |a: &[Vec<f64>], b: &[Vec<f64>]| -> f64 {
            let mut s = 0.0;
            for i in 0..n {
                for j in 0..n {
                    s += a[i][j] * b[i][j];
                }
            }
            s / (n * n) as f64
        };
        let a = center(&dm(x));
        let b = center(&dm(y));
        (dot(&a, &b), dot(&a, &a), dot(&b, &b))
    }

    fn naive_test_pvalue(x: &[f64], y: &[f64], b: usize, seed: u64) -> f64 {
        let n = x.len() as f64;
        let (c, vx, vy) = naive(x, y);
        let obs = n * c - tie_tolerance(x.len(), vx, vy);
        let mut rng = SimpleRng::new(seed);
        let mut perm: Vec<usize> = (0..x.len()).collect();
        let mut c = 0;
        for _ in 0..b {
            fisher_yates_shuffle(&mut perm, &mut rng);
            let yp: Vec<f64> = perm.iter().map(|&p| y[p]).collect();
            if n * naive(x, &yp).0 >= obs {
                c += 1;
            }
        }
        (c as f64 + 1.0) / (b as f64 + 1.0)
    }

    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 11) as f64 / (1u64 << 53) as f64
    }

    fn close(a: f64, b: f64, scale: f64) -> bool {
        (a - b).abs() <= 1e-10 * scale.max(1e-300)
    }

    #[test]
    fn test_matches_naive_reference() {
        let mut s = 3u64;
        for &n in &[3usize, 4, 7, 25, 100, 301] {
            for &(levels, offset) in &[(0usize, 0.0), (3, 0.0), (10, 1e4), (0, -250.0)] {
                let mut draw = || {
                    let u = lcg(&mut s);
                    offset
                        + if levels == 0 {
                            u * 4.0
                        } else {
                            (u * levels as f64).floor()
                        }
                };
                let x: Vec<f64> = (0..n).map(|_| draw()).collect();
                let noise: Vec<f64> = (0..n).map(|_| draw()).collect();
                for dep in [0.0, 0.3, 1.0] {
                    let y: Vec<f64> = x
                        .iter()
                        .zip(&noise)
                        .map(|(&a, &e)| dep * (a - offset).powi(2) + e)
                        .collect();
                    let (c, vx, vy) = naive(&x, &y);
                    let r = distance_cor(&x, &y).unwrap();
                    let scale = (vx * vy).sqrt();
                    assert!(
                        close(r.dcov.powi(2), c.max(0.0), scale),
                        "n={n} {c} {}",
                        r.dcov
                    );
                    assert!(close(r.dvar_x.powi(2), vx, vx), "n={n}");
                    assert!(close(r.dvar_y.powi(2), vy, vy), "n={n}");
                    let dc = if vx > 0.0 && vy > 0.0 {
                        (c / scale).max(0.0).sqrt().clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    assert!((r.dcor - dc).abs() < 1e-8, "n={n} {} vs {dc}", r.dcor);
                }
            }
        }
    }

    #[test]
    fn test_permutation_pvalue_matches_naive() {
        let mut s = 5u64;
        for &n in &[6usize, 20, 40] {
            let x: Vec<f64> = (0..n).map(|_| (lcg(&mut s) * 5.0).floor()).collect();
            let y: Vec<f64> = x.iter().map(|&a| a + lcg(&mut s) * 3.0).collect();
            let r = distance_cor_test(&x, &y, 99, Some(17)).unwrap();
            assert_eq!(r.p_value.unwrap(), naive_test_pvalue(&x, &y, 99, 17));
        }
    }

    /// n = 100_000: the old code needed four 80 GB n x n matrices.
    #[test]
    #[ignore]
    fn test_distance_cor_large_n() {
        let mut s = 9u64;
        let n = 100_000;
        let x: Vec<f64> = (0..n).map(|_| lcg(&mut s)).collect();
        let y: Vec<f64> = x.iter().map(|&a| a * a + 0.1 * lcg(&mut s)).collect();
        let r = distance_cor(&x, &y).unwrap();
        assert!(r.dcor > 0.5 && r.dcor <= 1.0);
        let t = distance_cor_test(&x, &y, 9, Some(1)).unwrap();
        assert!(t.p_value.unwrap() <= 0.1 + 1e-12);
    }

    #[test]
    fn test_distance_cor_linear() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();

        let result = distance_cor(&x, &y).unwrap();

        // Perfect linear relationship should give dCor close to 1
        assert!(result.dcor > 0.99);
    }

    #[test]
    fn test_distance_cor_nonlinear() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y: Vec<f64> = x.iter().map(|&xi| xi * xi).collect();

        let result = distance_cor(&x, &y).unwrap();

        // Quadratic relationship should still give high dCor
        assert!(result.dcor > 0.9);
    }

    #[test]
    fn test_distance_cor_weak() {
        // Data with weak relationship
        let x = vec![1.0, 5.0, 2.0, 8.0, 3.0, 9.0, 4.0, 7.0, 6.0, 10.0];
        let y = vec![3.0, 7.0, 1.0, 6.0, 9.0, 2.0, 8.0, 4.0, 10.0, 5.0];

        let result = distance_cor(&x, &y).unwrap();

        // Distance correlation is always between 0 and 1
        assert!(result.dcor >= 0.0 && result.dcor <= 1.0);
        // With small samples, even "independent" data can show moderate dCor
        // Just verify it's less than perfect correlation
        assert!(result.dcor < 0.99);
    }

    #[test]
    fn test_distance_cor_symmetric() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![5.0, 4.0, 3.0, 2.0, 1.0];

        let result_xy = distance_cor(&x, &y).unwrap();
        let result_yx = distance_cor(&y, &x).unwrap();

        // Distance correlation is symmetric
        assert!((result_xy.dcor - result_yx.dcor).abs() < 1e-10);
    }

    #[test]
    fn test_distance_cor_test() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y: Vec<f64> = x.iter().map(|&xi| xi * 2.0).collect();

        let result = distance_cor_test(&x, &y, 99, Some(42)).unwrap();

        // Strong relationship should give small p-value
        assert!(result.p_value.unwrap() < 0.05);
        assert!(result.dcor > 0.9);
    }

    #[test]
    fn test_distance_cor_bounds() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 1.0, 4.0, 3.0, 5.0];

        let result = distance_cor(&x, &y).unwrap();

        // Distance correlation is always between 0 and 1
        assert!(result.dcor >= 0.0);
        assert!(result.dcor <= 1.0);
    }
}
