//! Tukey's honest significant difference, a port of R's `TukeyHSD(aov(...))`
//! for a one-way layout.

use super::{
    group_index, group_moments, pairs, ptukey, qtukey, PairwiseComparison, PairwiseResult,
};
use crate::error::{Result, StatError};
use crate::parametric::Alternative;
use crate::utils::finite::ensure_finite;

/// Tukey HSD (Tukey-Kramer for unequal group sizes) for all pairwise
/// differences of group means, matching R's
/// `TukeyHSD(aov(values ~ factor(groups)), conf.level = conf_level)`.
///
/// For each pair, `estimate = mean(group2) - mean(group1)` (R's `diff`
/// column for row `"group2-group1"`), `se = sqrt(MSE / 2 * (1/n1 + 1/n2))`,
/// `statistic = estimate / se` (the studentized range statistic, signed),
/// `conf_low/high = estimate ∓ qtukey(conf_level, k, N - k) * se`, and
/// `p_value = p_adj = ptukey(|statistic|, k, N - k, upper tail)` (R's
/// `p adj`; the Tukey p-value is already simultaneous). `df` is `N - k`.
///
/// Values must be finite. With fewer than 2 residual degrees of freedom the
/// p-values and limits are NaN (as `ptukey` / `qtukey` in R). Memory is
/// O(n + k²) for n values in k groups.
///
/// # Example
/// ```
/// use anofox_statistics::tukey_hsd;
/// let values = [4.2, 5.1, 4.8, 6.3, 6.9, 7.1, 5.5, 5.0, 5.9];
/// let groups = [1, 1, 1, 2, 2, 2, 3, 3, 3];
/// let res = tukey_hsd(&values, &groups, 0.95).unwrap();
/// let c = &res.comparisons[0]; // 2 - 1
/// assert!(c.conf_low.unwrap() < c.estimate && c.estimate < c.conf_high.unwrap());
/// ```
pub fn tukey_hsd<G: Ord + Clone>(
    values: &[f64],
    groups: &[G],
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
    let k = g.levels.len();
    let n_total: usize = g.sizes.iter().sum();
    let df = (n_total - k) as f64;
    let mse = ss.iter().sum::<f64>() / df;
    let kf = k as f64;
    let crit = qtukey(conf_level, kf, df, 1.0, true);

    let comparisons = pairs(k)
        .map(|(j, i)| {
            let est = means[i] - means[j];
            let se = ((mse / 2.0) * (1.0 / g.sizes[i] as f64 + 1.0 / g.sizes[j] as f64)).sqrt();
            let q = est / se;
            let p = ptukey(q.abs(), kf, df, 1.0, false);
            let width = crit * se;
            PairwiseComparison {
                group1: g.levels[j].clone(),
                group2: g.levels[i].clone(),
                estimate: est,
                statistic: q,
                df: Some(df),
                p_value: p,
                p_adj: p,
                conf_low: Some(est - width),
                conf_high: Some(est + width),
            }
        })
        .collect();

    Ok(PairwiseResult {
        method: "Tukey multiple comparisons of means".to_string(),
        p_adjust_method: None,
        alternative: Alternative::TwoSided,
        conf_level: Some(conf_level),
        groups: g.levels,
        group_sizes: g.sizes,
        comparisons,
    })
}
