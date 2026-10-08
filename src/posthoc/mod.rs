//! Multiple-comparison adjustment and pairwise post-hoc tests.
//!
//! - [`p_adjust`]: R's `p.adjust` (holm, hochberg, hommel, bonferroni, BH, BY).
//! - [`pairwise_t_test`]: R's `pairwise.t.test` (pooled or Welch).
//! - [`tukey_hsd`]: R's `TukeyHSD(aov(values ~ groups))`, with [`ptukey`] /
//!   [`qtukey`] ports of R's studentized range distribution.
//! - [`dunn_test`]: Dunn's (1964) rank test, as `dunn.test(altp = TRUE)` /
//!   `FSA::dunnTest`.
//!
//! Each post-hoc test returns a [`PairwiseResult`] with one
//! [`PairwiseComparison`] per pair of groups. Groups are the sorted distinct
//! labels (R's factor levels for labels that sort the same way); comparisons
//! run over every pair `(group1, group2)` with `group1` before `group2` in
//! that order, ordered as R's `TukeyHSD` rows (`B-A, C-A, C-B, ...`), and
//! `estimate` is always `group2 - group1`.
//!
//! **Memory:** O(n) for the group index of the observations plus O(k²) for the
//! k(k-1)/2 output rows of k groups; no O(n²) structure is built.

mod dunn;
mod p_adjust;
mod pairwise_t;
mod tukey_dist;
mod tukey_hsd;

pub use dunn::dunn_test;
pub use p_adjust::{p_adjust, PAdjustMethod};
pub use pairwise_t::pairwise_t_test;
pub use tukey_dist::{ptukey, qtukey};
pub use tukey_hsd::tukey_hsd;

use crate::error::{Result, StatError};
use crate::parametric::Alternative;
use std::collections::BTreeMap;

/// One pairwise comparison of a post-hoc test.
#[derive(Debug, Clone, PartialEq)]
pub struct PairwiseComparison<G> {
    /// First group (earlier in sorted label order).
    pub group1: G,
    /// Second group (later in sorted label order).
    pub group2: G,
    /// Estimated difference `group2 - group1`: difference in means (t-test,
    /// Tukey) or in mean ranks (Dunn).
    pub estimate: f64,
    /// Test statistic for `group2 - group1`: t (pairwise t-test), the
    /// studentized range statistic q signed by the difference (Tukey), or z (Dunn).
    pub statistic: f64,
    /// Degrees of freedom of the statistic, if any (`None` for Dunn's z).
    pub df: Option<f64>,
    /// Unadjusted p-value. For Tukey HSD this is already the
    /// family-wise p-value (equal to `p_adj`), as in R.
    pub p_value: f64,
    /// Multiplicity-adjusted p-value.
    pub p_adj: f64,
    /// Lower confidence limit for `estimate` (Tukey HSD only).
    pub conf_low: Option<f64>,
    /// Upper confidence limit for `estimate` (Tukey HSD only).
    pub conf_high: Option<f64>,
}

/// Result of a pairwise post-hoc test.
#[derive(Debug, Clone, PartialEq)]
pub struct PairwiseResult<G> {
    /// Description of the method.
    pub method: String,
    /// Adjustment applied to `p_value` to give `p_adj`; `None` for Tukey HSD
    /// (whose p-values are simultaneous by construction).
    pub p_adjust_method: Option<PAdjustMethod>,
    /// Alternative hypothesis for `group2 - group1`.
    pub alternative: Alternative,
    /// Family-wise confidence level of `conf_low` / `conf_high`, if computed.
    pub conf_level: Option<f64>,
    /// Sorted distinct group labels.
    pub groups: Vec<G>,
    /// Per-group number of observations, aligned with `groups`.
    pub group_sizes: Vec<usize>,
    /// The k(k-1)/2 comparisons.
    pub comparisons: Vec<PairwiseComparison<G>>,
}

/// Observations split by group: sorted labels plus each observation's group index.
pub(crate) struct Grouping<G> {
    pub levels: Vec<G>,
    pub index: Vec<usize>,
    pub sizes: Vec<usize>,
}

pub(crate) fn group_index<G: Ord + Clone>(values: &[f64], groups: &[G]) -> Result<Grouping<G>> {
    if values.len() != groups.len() {
        return Err(StatError::InvalidParameter(format!(
            "values and groups must have the same length ({} vs {})",
            values.len(),
            groups.len()
        )));
    }
    if values.is_empty() {
        return Err(StatError::EmptyData);
    }
    let mut map: BTreeMap<&G, usize> = BTreeMap::new();
    for g in groups {
        map.entry(g).or_insert(0);
    }
    if map.len() < 2 {
        return Err(StatError::InvalidParameter(
            "post-hoc comparisons need at least 2 groups".to_string(),
        ));
    }
    let levels: Vec<G> = map.keys().map(|g| (*g).clone()).collect();
    for (i, v) in map.values_mut().enumerate() {
        *v = i;
    }
    let index: Vec<usize> = groups.iter().map(|g| map[g]).collect();
    let mut sizes = vec![0usize; levels.len()];
    for &i in &index {
        sizes[i] += 1;
    }
    Ok(Grouping {
        levels,
        index,
        sizes,
    })
}

/// Per-group means and within-group sums of squared deviations (two-pass).
pub(crate) fn group_moments(values: &[f64], g: &Grouping<impl Clone>) -> (Vec<f64>, Vec<f64>) {
    let k = g.levels.len();
    let mut sum = vec![0.0; k];
    for (&v, &i) in values.iter().zip(&g.index) {
        sum[i] += v;
    }
    let means: Vec<f64> = sum
        .iter()
        .zip(&g.sizes)
        .map(|(s, &n)| s / n as f64)
        .collect();
    let mut ss = vec![0.0; k];
    for (&v, &i) in values.iter().zip(&g.index) {
        let d = v - means[i];
        ss[i] += d * d;
    }
    (means, ss)
}

/// Pairs `(j, i)` with `j < i`, in R's lower-triangle column-major order.
pub(crate) fn pairs(k: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..k).flat_map(move |j| (j + 1..k).map(move |i| (j, i)))
}
