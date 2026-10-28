//! Binning of a reference and a current sample into matching proportions,
//! shared by the divergence measures of the [`drift`](crate::drift) module.
//!
//! [`Binning::Auto`] reproduces evidently's `get_binned_data`
//! (`legacy/calculations/stattests/utils.py`): equal-width Sturges bins over
//! the pooled sample when the reference has more than 20 distinct values, one
//! bin per distinct pooled value otherwise.

use crate::error::{Result, StatError};
use crate::utils::finite::ensure_finite;
use std::collections::BTreeMap;

/// How [`binned_proportions`] bins numeric data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Binning {
    /// evidently's rule: [`Binning::Sturges`] when the reference has more than
    /// 20 distinct values, [`Binning::Categorical`] otherwise.
    Auto,
    /// Equal-width bins over the range of the pooled sample, with
    /// `ceil(log2(n) + 1)` bins for `n` pooled values (numpy
    /// `histogram_bin_edges(pooled, "sturges")`). Bins are `[a, b)` except the
    /// last, which is closed. Note: R's `hist(breaks = "Sturges")` uses
    /// `pretty()` breaks instead; the equivalent R breaks are
    /// `seq(min(z), max(z), length.out = nclass.Sturges(z) + 1)`.
    Sturges,
    /// `n` bins at the reference quantiles `0, 1/n, ..., 1` (R `quantile`
    /// type 7), the classic credit-scoring PSI convention. The outer edges are
    /// open (`-Inf`, `+Inf`), so current values outside the reference range
    /// fall into the first or last bin; duplicate breaks (ties in the
    /// reference) are merged, so there can be fewer than `n` bins.
    Quantile(usize),
    /// One bin per distinct pooled value.
    Categorical,
}

/// Number of distinct values of a sorted slice.
fn n_distinct_sorted(sorted: &[f64]) -> usize {
    if sorted.is_empty() {
        return 0;
    }
    1 + sorted.windows(2).filter(|w| w[0] != w[1]).count()
}

fn sorted(data: &[f64]) -> Vec<f64> {
    let mut v = data.to_vec();
    v.sort_unstable_by(|a, b| a.total_cmp(b));
    v
}

/// `n_bins + 1` equally spaced edges from `lo` to `hi` (numpy `linspace`, R
/// `seq(length.out =)`): `lo + i * step`, with the last edge exactly `hi`.
fn linspace(lo: f64, hi: f64, n_bins: usize) -> Vec<f64> {
    let step = (hi - lo) / n_bins as f64;
    let mut edges: Vec<f64> = (0..n_bins).map(|i| lo + i as f64 * step).collect();
    edges.push(hi);
    edges
}

/// Equal-width Sturges edges over `pool`, as numpy
/// `histogram_bin_edges(pool, bins = "sturges")`. The bin count is computed
/// from `n_for_count` values. Errors when the range overflows.
pub(crate) fn sturges_edges(pool: &[f64], n_for_count: usize) -> Result<Vec<f64>> {
    let lo = pool.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = pool.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !(hi - lo).is_finite() {
        return Err(StatError::InvalidParameter(
            "range of the pooled sample is not finite".to_string(),
        ));
    }
    if hi - lo == 0.0 {
        // numpy widens a degenerate range by 0.5 on each side, one bin.
        return Ok(vec![lo - 0.5, hi + 0.5]);
    }
    // numpy: width = ptp / (log2(n) + 1), n_bins = ceil(ptp / width).
    let width = (hi - lo) / ((n_for_count as f64).log2() + 1.0);
    let n_bins = ((hi - lo) / width).ceil().max(1.0) as usize;
    Ok(linspace(lo, hi, n_bins))
}

/// R `quantile(sorted, probs, type = 7)` for sorted data.
fn quantile_type7(sorted: &[f64], prob: f64) -> f64 {
    let n = sorted.len();
    let index = 1.0 + (n - 1) as f64 * prob;
    let lo = index.floor() as usize;
    let hi = index.ceil() as usize;
    let q = sorted[lo - 1];
    if index > lo as f64 && sorted[hi - 1] != q {
        let h = index - lo as f64;
        (1.0 - h) * q + h * sorted[hi - 1]
    } else {
        q
    }
}

/// Reference-quantile edges with open outer edges and duplicates removed.
fn quantile_edges(reference_sorted: &[f64], n_bins: usize) -> Vec<f64> {
    let mut edges: Vec<f64> = (0..=n_bins)
        .map(|i| quantile_type7(reference_sorted, i as f64 / n_bins as f64))
        .collect();
    edges.dedup();
    let last = edges.len() - 1;
    edges[0] = f64::NEG_INFINITY;
    if last == 0 {
        edges.push(f64::INFINITY);
    } else {
        edges[last] = f64::INFINITY;
    }
    edges
}

/// Proportions of `data` in the bins `[e_i, e_{i+1})`, the last bin closed
/// (R `findInterval(..., rightmost.closed = TRUE)`, numpy `histogram`).
/// Values outside `[e_0, e_k]` are not counted.
pub(crate) fn histogram_proportions(data: &[f64], edges: &[f64]) -> Vec<f64> {
    let n_bins = edges.len() - 1;
    let mut counts = vec![0usize; n_bins];
    for &v in data {
        if v < edges[0] || v > edges[n_bins] {
            continue;
        }
        // number of edges <= v, minus one
        let k = edges.partition_point(|&e| e <= v);
        counts[(k - 1).min(n_bins - 1)] += 1;
    }
    let n = data.len() as f64;
    counts.into_iter().map(|c| c as f64 / n).collect()
}

/// Proportions per distinct value, over the sorted distinct pooled values.
fn value_proportions(reference: &[f64], current: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut levels = sorted(&[reference, current].concat());
    levels.dedup();
    let props = |data: &[f64]| {
        let mut counts = vec![0usize; levels.len()];
        for v in data {
            let k = levels.partition_point(|l| l < v);
            counts[k] += 1;
        }
        let n = data.len() as f64;
        counts.into_iter().map(|c| c as f64 / n).collect::<Vec<_>>()
    };
    (props(reference), props(current))
}

fn validate(reference: &[f64], current: &[f64]) -> Result<()> {
    if reference.is_empty() || current.is_empty() {
        return Err(StatError::EmptyData);
    }
    ensure_finite("reference", reference)?;
    ensure_finite("current", current)
}

/// Bin a numeric reference and current sample on common bins and return the
/// proportions `(p, q)` of the reference and the current sample per bin.
///
/// Empty bins have proportion 0; no zero fill is applied (see
/// [`zero_fill`]). Errors on empty or non-finite input and on
/// `Binning::Quantile(0)`.
///
/// # Example
/// ```
/// use anofox_statistics::drift::{binned_proportions, Binning};
/// let (p, q) = binned_proportions(&[1.0, 1.0, 2.0], &[2.0, 3.0], Binning::Categorical).unwrap();
/// assert_eq!(p, vec![2.0 / 3.0, 1.0 / 3.0, 0.0]);
/// assert_eq!(q, vec![0.0, 0.5, 0.5]);
/// ```
pub fn binned_proportions(
    reference: &[f64],
    current: &[f64],
    binning: Binning,
) -> Result<(Vec<f64>, Vec<f64>)> {
    validate(reference, current)?;
    match binning {
        Binning::Auto => {
            let binning = if n_distinct_sorted(&sorted(reference)) > 20 {
                Binning::Sturges
            } else {
                Binning::Categorical
            };
            binned_proportions(reference, current, binning)
        }
        Binning::Sturges => {
            let pool = [reference, current].concat();
            let edges = sturges_edges(&pool, pool.len())?;
            Ok((
                histogram_proportions(reference, &edges),
                histogram_proportions(current, &edges),
            ))
        }
        Binning::Quantile(n_bins) => {
            if n_bins == 0 {
                return Err(StatError::InvalidParameter(
                    "number of quantile bins must be at least 1".to_string(),
                ));
            }
            let edges = quantile_edges(&sorted(reference), n_bins);
            Ok((
                histogram_proportions(reference, &edges),
                histogram_proportions(current, &edges),
            ))
        }
        Binning::Categorical => Ok(value_proportions(reference, current)),
    }
}

/// Proportions on equal-width Sturges bins over the pooled range whose count
/// comes from the number of *distinct* pooled values: evidently's Hellinger
/// binning (`histogram_bin_edges(unique(ref, cur), "sturges")`).
pub(crate) fn sturges_distinct_proportions(
    reference: &[f64],
    current: &[f64],
) -> Result<(Vec<f64>, Vec<f64>)> {
    validate(reference, current)?;
    let mut distinct = sorted(&[reference, current].concat());
    distinct.dedup();
    let edges = sturges_edges(&distinct, distinct.len())?;
    Ok((
        histogram_proportions(reference, &edges),
        histogram_proportions(current, &edges),
    ))
}

/// Proportions `(p, q)` of the reference and the current sample per category,
/// over the sorted distinct categories of both samples. Errors on empty
/// input.
///
/// # Example
/// ```
/// use anofox_statistics::drift::category_proportions;
/// let (p, q) = category_proportions(&["a", "b", "b", "a"], &["b", "c"]).unwrap();
/// assert_eq!(p, vec![0.5, 0.5, 0.0]);
/// assert_eq!(q, vec![0.0, 0.5, 0.5]);
/// ```
pub fn category_proportions<T: Ord>(
    reference: &[T],
    current: &[T],
) -> Result<(Vec<f64>, Vec<f64>)> {
    if reference.is_empty() || current.is_empty() {
        return Err(StatError::EmptyData);
    }
    let mut counts: BTreeMap<&T, (usize, usize)> = BTreeMap::new();
    for v in reference {
        counts.entry(v).or_default().0 += 1;
    }
    for v in current {
        counts.entry(v).or_default().1 += 1;
    }
    let (nr, nc) = (reference.len() as f64, current.len() as f64);
    Ok(counts
        .values()
        .map(|&(r, c)| (r as f64 / nr, c as f64 / nc))
        .unzip())
}

/// evidently's zero fill: every zero proportion is replaced by `1e-4`, or by
/// `min_nonzero / 1e6` when the smallest non-zero proportion is `<= 1e-4`.
/// Used by PSI and the Kullback-Leibler divergence. A vector without
/// non-zero entries is returned unchanged.
pub fn zero_fill(props: &[f64]) -> Vec<f64> {
    let min_nonzero = props
        .iter()
        .copied()
        .filter(|&v| v != 0.0)
        .fold(f64::INFINITY, f64::min);
    if !min_nonzero.is_finite() {
        return props.to_vec();
    }
    let fill = if min_nonzero <= 1e-4 {
        min_nonzero / 1e6
    } else {
        1e-4
    };
    props
        .iter()
        .map(|&v| if v == 0.0 { fill } else { v })
        .collect()
}

/// Validate a pair of proportion vectors: same non-zero length, finite,
/// non-negative, and each with a positive sum.
pub(crate) fn validate_proportions(p: &[f64], q: &[f64]) -> Result<()> {
    if p.is_empty() || q.is_empty() {
        return Err(StatError::EmptyData);
    }
    if p.len() != q.len() {
        return Err(StatError::InvalidParameter(format!(
            "proportion vectors must have the same length, got {} and {}",
            p.len(),
            q.len()
        )));
    }
    ensure_finite("p", p)?;
    ensure_finite("q", q)?;
    for (name, v) in [("p", p), ("q", q)] {
        if v.iter().any(|&x| x < 0.0) {
            return Err(StatError::InvalidParameter(format!(
                "{} must not contain negative values",
                name
            )));
        }
        if v.iter().sum::<f64>() <= 0.0 {
            return Err(StatError::InvalidParameter(format!(
                "{} must have a positive sum",
                name
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sturges_matches_numpy() {
        // numpy.histogram_bin_edges([0, 1, ..., 9], "sturges"): 5 bins.
        let pool: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let e = sturges_edges(&pool, pool.len()).unwrap();
        assert_eq!(e.len(), 6);
        assert_eq!((e[0], e[5]), (0.0, 9.0));
        // constant data: one bin of width 1
        assert_eq!(sturges_edges(&[2.0, 2.0], 2).unwrap(), vec![1.5, 2.5]);
        assert!(sturges_edges(&[-1e308, 1e308], 2).is_err());
    }

    #[test]
    fn histogram_last_bin_closed() {
        let p = histogram_proportions(&[0.0, 0.5, 1.0, 2.0, 3.0], &[0.0, 1.0, 2.0]);
        assert_eq!(p, vec![0.4, 0.4]);
    }

    #[test]
    fn quantile_type7_matches_r() {
        // quantile(1:5 + 0.0, c(0, 0.3, 1)) = 1, 2.2, 5
        let s = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(quantile_type7(&s, 0.0), 1.0);
        assert!((quantile_type7(&s, 0.3) - 2.2).abs() < 1e-15);
        assert_eq!(quantile_type7(&s, 1.0), 5.0);
        // all-tied reference: a single bin covering everything
        assert_eq!(
            quantile_edges(&[1.0, 1.0], 4),
            vec![f64::NEG_INFINITY, f64::INFINITY]
        );
    }

    #[test]
    fn zero_fill_rules() {
        assert_eq!(zero_fill(&[0.5, 0.5, 0.0]), vec![0.5, 0.5, 1e-4]);
        assert_eq!(zero_fill(&[1e-4, 1.0, 0.0]), vec![1e-4, 1.0, 1e-4 / 1e6]);
    }
}
