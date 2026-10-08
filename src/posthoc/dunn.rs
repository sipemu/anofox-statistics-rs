//! Dunn's (1964) test of pairwise mean-rank differences after Kruskal-Wallis.

use super::{group_index, p_adjust, pairs, PAdjustMethod, PairwiseComparison, PairwiseResult};
use crate::error::Result;
use crate::nonparametric::ranks::rank_with_ties;
use crate::parametric::Alternative;
use crate::utils::finite::ensure_no_nan;
use statrs::distribution::{ContinuousCDF, Normal};

/// Dunn's test for all pairwise differences in mean ranks, with tie
/// correction and p-value adjustment, matching `FSA::dunnTest(values ~
/// groups, method = ...)` and `dunn.test::dunn.test(values, groups,
/// method = ..., altp = TRUE)` (two-sided p-values).
///
/// Ranks are taken over all observations (midranks for ties). For each pair,
/// `estimate = mean_rank(group2) - mean_rank(group1)`,
/// `statistic = z = estimate / sqrt((N(N+1)/12 - T/(12(N-1))) * (1/n1 + 1/n2))`
/// with `T = Σ(t³ - t)` over tie groups, `p_value = 2 * P(Z > |z|)`, and
/// `p_adj = p_adjust(p_value, p_adjust_method)` over all k(k-1)/2 pairs.
/// Note the sign: `dunn.test` / `FSA` report `z` for `group1 - group2`.
/// `df`, `conf_low` and `conf_high` are `None`.
///
/// ±Inf ranks as an extreme value; NaN is rejected. Memory is O(n + k²).
///
/// # Example
/// ```
/// use anofox_statistics::{dunn_test, PAdjustMethod};
/// let values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
/// let groups = ["a", "a", "a", "b", "b", "b", "c", "c", "c"];
/// let res = dunn_test(&values, &groups, PAdjustMethod::Holm).unwrap();
/// assert_eq!(res.comparisons[0].estimate, 3.0); // mean rank b - a
/// ```
pub fn dunn_test<G: Ord + Clone>(
    values: &[f64],
    groups: &[G],
    p_adjust_method: PAdjustMethod,
) -> Result<PairwiseResult<G>> {
    ensure_no_nan("values", values)?;
    let g = group_index(values, groups)?;
    let k = g.levels.len();
    let (ranks, tie_sizes) = rank_with_ties(values)?;
    let mut rank_sum = vec![0.0; k];
    for (&r, &i) in ranks.iter().zip(&g.index) {
        rank_sum[i] += r;
    }
    let mean_rank: Vec<f64> = rank_sum
        .iter()
        .zip(&g.sizes)
        .map(|(s, &n)| s / n as f64)
        .collect();
    let n = values.len() as f64;
    let ties: f64 = tie_sizes
        .iter()
        .map(|&t| {
            let t = t as f64;
            t * t * t - t
        })
        .sum();
    let var_base = n * (n + 1.0) / 12.0 - ties / (12.0 * (n - 1.0));
    let normal = Normal::standard();

    let mut comparisons: Vec<PairwiseComparison<G>> = pairs(k)
        .map(|(j, i)| {
            let est = mean_rank[i] - mean_rank[j];
            let se = (var_base * (1.0 / g.sizes[i] as f64 + 1.0 / g.sizes[j] as f64)).sqrt();
            let z = est / se;
            let p = if z.is_nan() {
                f64::NAN
            } else {
                2.0 * normal.sf(z.abs())
            };
            PairwiseComparison {
                group1: g.levels[j].clone(),
                group2: g.levels[i].clone(),
                estimate: est,
                statistic: z,
                df: None,
                p_value: p,
                p_adj: f64::NAN,
                conf_low: None,
                conf_high: None,
            }
        })
        .collect();
    let raw: Vec<f64> = comparisons.iter().map(|c| c.p_value).collect();
    for (c, p) in comparisons.iter_mut().zip(p_adjust(&raw, p_adjust_method)) {
        c.p_adj = p;
    }

    Ok(PairwiseResult {
        method: "Dunn (1964) Kruskal-Wallis multiple comparison".to_string(),
        p_adjust_method: Some(p_adjust_method),
        alternative: Alternative::TwoSided,
        conf_level: None,
        groups: g.levels,
        group_sizes: g.sizes,
        comparisons,
    })
}
