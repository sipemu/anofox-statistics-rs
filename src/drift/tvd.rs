//! Total variation distance with a permutation p-value.

use crate::error::{Result, StatError};
use crate::utils::finite::ensure_finite;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::BTreeMap;

/// Result of [`tvd_test`].
#[derive(Debug, Clone, PartialEq)]
pub struct TvdResult {
    /// Total variation distance `(1/2) sum_k |p_k - q_k|`, in `[0, 1]`.
    pub statistic: f64,
    /// Permutation p-value.
    pub p_value: f64,
    /// Number of categories seen in either sample.
    pub n_categories: usize,
    /// Number of permutations used.
    pub n_permutations: usize,
    /// Size of the reference sample.
    pub n_reference: usize,
    /// Size of the current sample.
    pub n_current: usize,
}

/// `(1/2) sum_k |r_k / m - c_k / n|` from category counts.
fn tvd_from_counts(r: &[usize], c: &[usize], m: f64, n: f64) -> f64 {
    0.5 * r
        .iter()
        .zip(c)
        .map(|(&a, &b)| (a as f64 / m - b as f64 / n).abs())
        .sum::<f64>()
}

/// Total variation distance between the category distributions of a
/// reference and a current sample, with a permutation p-value.
///
/// `TVD = (1/2) sum_k |p_k - q_k|` over the categories seen in either
/// sample, with `p` and `q` the category proportions of the reference and
/// the current sample.
///
/// The p-value is `(1 + #{TVD* >= TVD}) / (1 + B)` over `B = n_permutations`
/// random splits of the pooled sample into groups of the original sizes
/// (never 0, as the other permutation tests of the crate). Each permutation
/// only recounts the categories, O(N). `seed` makes it reproducible
/// (`ChaCha8Rng`).
///
/// This is evidently's `"TVD"` drift test for categorical columns (1000
/// permutations, drift at `p < 0.1`). TVD can be much less powerful than a
/// chi-squared test of the same table: for the evidently validation data
/// (`xc = (0:49) %% 4`, `yc = ((0:39) * 3) %% 5`) TVD is 0.2 with a
/// permutation p-value of about 0.29, while `chisq.test(..., simulate.p.value
/// = TRUE)` gives about 0.025 (see [`chisq_test`](crate::chisq_test)).
///
/// Errors on an empty sample or `n_permutations = 0`.
///
/// R equivalent (statistic):
/// `0.5 * sum(abs(table(factor(a, lv)) / length(a) - table(factor(b, lv)) / length(b)))`,
/// or `philentropy::distance(rbind(p, q), method = "manhattan") / 2`.
///
/// # Example
/// ```
/// use anofox_statistics::drift::tvd_test;
/// let r = tvd_test(&["a", "a", "b", "b"], &["a", "b", "b", "b"], 999, Some(1)).unwrap();
/// assert_eq!(r.statistic, 0.25);
/// assert_eq!(r.n_categories, 2);
/// ```
pub fn tvd_test<T: Ord>(
    reference: &[T],
    current: &[T],
    n_permutations: usize,
    seed: Option<u64>,
) -> Result<TvdResult> {
    if reference.is_empty() || current.is_empty() {
        return Err(StatError::EmptyData);
    }
    if n_permutations == 0 {
        return Err(StatError::InvalidParameter(
            "n_permutations must be at least 1".to_string(),
        ));
    }
    // Category codes of the pooled sample (reference first).
    let mut codes: BTreeMap<&T, usize> = BTreeMap::new();
    for v in reference.iter().chain(current) {
        let next = codes.len();
        codes.entry(v).or_insert(next);
    }
    let pooled: Vec<usize> = reference.iter().chain(current).map(|v| codes[v]).collect();
    let k = codes.len();
    let (m, n) = (reference.len(), current.len());
    let (mf, nf) = (m as f64, n as f64);
    let mut total = vec![0usize; k];
    for &c in &pooled {
        total[c] += 1;
    }
    let counts_of = |first: &[usize]| {
        let mut r = vec![0usize; k];
        for &c in first {
            r[c] += 1;
        }
        let c: Vec<usize> = total.iter().zip(&r).map(|(t, a)| t - a).collect();
        (r, c)
    };
    let (r, c) = counts_of(&pooled[..m]);
    let statistic = tvd_from_counts(&r, &c, mf, nf);

    let mut rng = match seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => ChaCha8Rng::from_entropy(),
    };
    let threshold = statistic - 1e-12;
    let mut perm = pooled;
    let mut count = 0usize;
    for _ in 0..n_permutations {
        perm.shuffle(&mut rng);
        let (r, c) = counts_of(&perm[..m]);
        if tvd_from_counts(&r, &c, mf, nf) >= threshold {
            count += 1;
        }
    }

    Ok(TvdResult {
        statistic,
        p_value: (count as f64 + 1.0) / (n_permutations as f64 + 1.0),
        n_categories: k,
        n_permutations,
        n_reference: m,
        n_current: n,
    })
}

/// [`tvd_test`] for numeric category codes: one category per distinct
/// value. Errors additionally on non-finite values.
pub fn tvd_test_numeric(
    reference: &[f64],
    current: &[f64],
    n_permutations: usize,
    seed: Option<u64>,
) -> Result<TvdResult> {
    ensure_finite("reference", reference)?;
    ensure_finite("current", current)?;
    // Finite values are totally ordered by their bits' total order; map
    // -0.0 to 0.0 so that both are one category.
    let key = |v: &f64| {
        let v = if *v == 0.0 { 0.0 } else { *v };
        OrdF64(v)
    };
    let r: Vec<OrdF64> = reference.iter().map(key).collect();
    let c: Vec<OrdF64> = current.iter().map(key).collect();
    tvd_test(&r, &c, n_permutations, seed)
}

/// `f64` with the total order (finite values only).
#[derive(Debug, Clone, Copy, PartialEq)]
struct OrdF64(f64);

impl Eq for OrdF64 {}

impl PartialOrd for OrdF64 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdF64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}
