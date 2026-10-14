//! Pairwise t-tests, a port of R's `pairwise.t.test` (unpaired).

use super::{
    group_index, group_moments, p_adjust, pairs, PAdjustMethod, PairwiseComparison, PairwiseResult,
};
use crate::error::{Result, StatError};
use crate::parametric::Alternative;
use crate::utils::dist::{t_cdf, t_inv, t_sf};
use crate::utils::finite::ensure_finite;

fn t_p_value(t: f64, df: f64, alternative: Alternative) -> f64 {
    match alternative {
        Alternative::TwoSided => 2.0 * t_cdf(-t.abs(), df),
        Alternative::Greater => t_sf(t, df),
        Alternative::Less => t_cdf(t, df),
    }
}

/// Confidence interval for `estimate` with standard error `se` and `df`
/// degrees of freedom at level `level`; one-sided for a one-sided alternative.
/// NaN when the standard error or the degrees of freedom are undefined.
fn t_interval(estimate: f64, se: f64, df: f64, level: f64, alternative: Alternative) -> (f64, f64) {
    if !(se.is_finite() && df > 0.0) {
        return (f64::NAN, f64::NAN);
    }
    match alternative {
        Alternative::TwoSided => {
            let half = t_inv(1.0 - (1.0 - level) / 2.0, df) * se;
            (estimate - half, estimate + half)
        }
        Alternative::Greater => (estimate - t_inv(level, df) * se, f64::INFINITY),
        Alternative::Less => (f64::NEG_INFINITY, estimate + t_inv(level, df) * se),
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
/// and `df` its degrees of freedom. `conf_low` / `conf_high` are the 95%
/// confidence interval for `estimate` (see [`pairwise_t_test_with_conf_level`]).
/// `p_adj` applies `p_adjust_method` over all k(k-1)/2 p-values.
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
    pairwise_t_test_with_conf_level(
        values,
        groups,
        pooled_sd,
        p_adjust_method,
        alternative,
        0.95,
    )
}

/// [`pairwise_t_test`] with the confidence level of the intervals.
///
/// Each interval uses the comparison's own standard error and degrees of
/// freedom (pooled SD with `N - k` df, or Welch), and is one-sided for a
/// one-sided `alternative`. With `PAdjustMethod::None` it is the
/// per-comparison interval at `conf_level`. With any other adjustment it is
/// the Bonferroni simultaneous interval at `1 - (1 - conf_level) / m` for the
/// m = k(k-1)/2 comparisons, as `emmeans::confint(pairs(...), adjust = )` does
/// (step-wise and FDR adjustments have no interval counterpart, and emmeans
/// substitutes Bonferroni for them). `conf_level` of the result is the
/// family-wise level `conf_level`. Validated against
/// `confint(pairs(emmeans(lm(values ~ groups), ~ groups), reverse = TRUE), adjust = )`
/// (pooled SD) and `t.test` (Welch).
///
/// `conf_level` must be in (0, 1).
pub fn pairwise_t_test_with_conf_level<G: Ord + Clone>(
    values: &[f64],
    groups: &[G],
    pooled_sd: bool,
    p_adjust_method: PAdjustMethod,
    alternative: Alternative,
    conf_level: f64,
) -> Result<PairwiseResult<G>> {
    if !(conf_level > 0.0 && conf_level < 1.0) {
        return Err(StatError::InvalidParameter(
            "conf_level must be between 0 and 1".to_string(),
        ));
    }
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

    let m = k * (k - 1) / 2;
    let level = if p_adjust_method == PAdjustMethod::None {
        conf_level
    } else {
        1.0 - (1.0 - conf_level) / m as f64
    };

    let mut comparisons = Vec::with_capacity(m);
    for (j, i) in pairs(k) {
        let dif = means[i] - means[j];
        let (se, df) = if pooled_sd {
            (pooled * (1.0 / n[i] + 1.0 / n[j]).sqrt(), total_degf)
        } else if n[i] < 2.0 || n[j] < 2.0 {
            (f64::NAN, f64::NAN)
        } else {
            let vi = ss[i] / (n[i] - 1.0) / n[i];
            let vj = ss[j] / (n[j] - 1.0) / n[j];
            let se2 = vi + vj;
            let df = se2 * se2 / (vi * vi / (n[i] - 1.0) + vj * vj / (n[j] - 1.0));
            (se2.sqrt(), df)
        };
        let t = dif / se;
        let (lo, hi) = t_interval(dif, se, df, level, alternative);
        comparisons.push(PairwiseComparison {
            group1: g.levels[j].clone(),
            group2: g.levels[i].clone(),
            estimate: dif,
            statistic: t,
            df: Some(df),
            p_value: t_p_value(t, df, alternative),
            p_adj: f64::NAN,
            conf_low: Some(lo),
            conf_high: Some(hi),
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
        conf_level: Some(conf_level),
        groups: g.levels,
        group_sizes: g.sizes,
        comparisons,
    })
}
