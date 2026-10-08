//! Pairwise t-tests, a port of R's `pairwise.t.test` (unpaired).

use super::{
    group_index, group_moments, p_adjust, pairs, PAdjustMethod, PairwiseComparison, PairwiseResult,
};
use crate::error::Result;
use crate::parametric::Alternative;
use crate::utils::dist::{t_cdf, t_sf};
use crate::utils::finite::ensure_finite;

fn t_p_value(t: f64, df: f64, alternative: Alternative) -> f64 {
    match alternative {
        Alternative::TwoSided => 2.0 * t_cdf(-t.abs(), df),
        Alternative::Greater => t_sf(t, df),
        Alternative::Less => t_cdf(t, df),
    }
}

/// Pairwise comparisons of group means with t-tests and p-value adjustment,
/// matching R's `pairwise.t.test(values, groups, p.adjust.method,
/// pool.sd, alternative = ...)` (unpaired).
///
/// - `pooled_sd = true`: every comparison uses the standard deviation pooled
///   over **all** groups, with `N - k` degrees of freedom (R's default).
///   A group with a single observation makes the pooled SD undefined, and
///   all results are NaN (R returns `NA`).
/// - `pooled_sd = false`: a Welch two-sample t-test per pair. A pair with a
///   group of fewer than 2 observations, or with zero variance in both groups,
///   gives NaN (R stops with an error).
///
/// `alternative` refers to `group2 - group1` (R's row level minus column
/// level). `estimate` is the difference in means, `statistic` the t statistic
/// and `df` its degrees of freedom; `conf_low` / `conf_high` are `None`
/// (R computes no intervals). `p_adj` applies `p_adjust_method` over all
/// k(k-1)/2 p-values.
///
/// Values must be finite. Memory is O(n + k²) for n values in k groups.
///
/// # Example
/// ```
/// use anofox_statistics::{pairwise_t_test, Alternative, PAdjustMethod};
/// let values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
/// let groups = ["a", "a", "a", "b", "b", "b", "c", "c", "c"];
/// let res = pairwise_t_test(&values, &groups, true, PAdjustMethod::Holm,
///                           Alternative::TwoSided).unwrap();
/// assert_eq!(res.comparisons.len(), 3);
/// assert_eq!((res.comparisons[0].group1, res.comparisons[0].group2), ("a", "b"));
/// ```
pub fn pairwise_t_test<G: Ord + Clone>(
    values: &[f64],
    groups: &[G],
    pooled_sd: bool,
    p_adjust_method: PAdjustMethod,
    alternative: Alternative,
) -> Result<PairwiseResult<G>> {
    ensure_finite("values", values)?;
    let g = group_index(values, groups)?;
    let (means, ss) = group_moments(values, &g);
    let n: Vec<f64> = g.sizes.iter().map(|&s| s as f64).collect();
    let k = g.levels.len();

    // R: pooled.sd <- sqrt(sum(s^2 * degf) / total.degf); NA if any group has n = 1.
    let total_degf: f64 = n.iter().map(|ni| ni - 1.0).sum();
    let pooled = if n.iter().any(|&ni| ni < 2.0) {
        f64::NAN
    } else {
        (ss.iter().sum::<f64>() / total_degf).sqrt()
    };

    let mut comparisons = Vec::with_capacity(k * (k - 1) / 2);
    for (j, i) in pairs(k) {
        let dif = means[i] - means[j];
        let (t, df) = if pooled_sd {
            let se = pooled * (1.0 / n[i] + 1.0 / n[j]).sqrt();
            (dif / se, total_degf)
        } else if n[i] < 2.0 || n[j] < 2.0 {
            (f64::NAN, f64::NAN)
        } else {
            let vi = ss[i] / (n[i] - 1.0) / n[i];
            let vj = ss[j] / (n[j] - 1.0) / n[j];
            let se2 = vi + vj;
            let df = se2 * se2 / (vi * vi / (n[i] - 1.0) + vj * vj / (n[j] - 1.0));
            (dif / se2.sqrt(), df)
        };
        comparisons.push(PairwiseComparison {
            group1: g.levels[j].clone(),
            group2: g.levels[i].clone(),
            estimate: dif,
            statistic: t,
            df: Some(df),
            p_value: t_p_value(t, df, alternative),
            p_adj: f64::NAN,
            conf_low: None,
            conf_high: None,
        });
    }
    let raw: Vec<f64> = comparisons.iter().map(|c| c.p_value).collect();
    for (c, p) in comparisons.iter_mut().zip(p_adjust(&raw, p_adjust_method)) {
        c.p_adj = p;
    }

    Ok(PairwiseResult {
        method: if pooled_sd {
            "t tests with pooled SD".to_string()
        } else {
            "t tests with non-pooled SD".to_string()
        },
        p_adjust_method: Some(p_adjust_method),
        alternative,
        conf_level: None,
        groups: g.levels,
        group_sizes: g.sizes,
        comparisons,
    })
}
