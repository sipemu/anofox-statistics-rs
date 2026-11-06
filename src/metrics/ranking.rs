//! Ranking (recommender) metrics over queries: NDCG@k (scikit-learn's
//! `ndcg_score`), and evidently's MAP@k, MRR@k, precision@k, recall@k and hit
//! rate@k.

use crate::error::{Result, StatError};
use crate::utils::finite::ensure_finite;
use std::collections::BTreeMap;

/// Options of [`ranking_metrics`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RankingOptions {
    /// Cut-off `k >= 1`. Default 10.
    pub k: usize,
    /// `None` (default): an item is relevant when its relevance is `> 0`, and
    /// NDCG uses the graded relevance as gain. `Some(m)`: an item is relevant
    /// when its relevance is `>= m`, and every metric, NDCG included, uses
    /// the binary relevance (evidently's `min_rel_score`).
    pub min_relevance: Option<f64>,
    /// Average over all queries (`true`, scikit-learn's `ndcg_score`), or only
    /// over queries with at least one relevant item (`false`, default,
    /// evidently's `no_feedback_users = False`). A query without relevant
    /// items scores 0 on every metric.
    pub include_no_feedback: bool,
}

impl Default for RankingOptions {
    fn default() -> Self {
        Self {
            k: 10,
            min_relevance: None,
            include_no_feedback: false,
        }
    }
}

/// Ranking metrics averaged over queries, returned by [`ranking_metrics`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RankingMetrics {
    /// Queries averaged over.
    pub n_queries: usize,
    /// Normalised discounted cumulative gain at k.
    pub ndcg: f64,
    /// Mean average precision at k.
    pub map: f64,
    /// Mean reciprocal rank of the first relevant item within the top k.
    pub mrr: f64,
    /// Relevant items in the top k over `min(k, items of the query)`.
    pub precision: f64,
    /// Relevant items in the top k over the relevant items of the query.
    pub recall: f64,
    /// Share of queries with a relevant item in the top k.
    pub hit_rate: f64,
}

/// NDCG@k of one query, as scikit-learn's `ndcg_score([relevance],
/// [score], k = k)`: gain = relevance, discount `1 / log2(rank + 1)`, tied
/// scores share the mean gain of their group over the discounts of the
/// positions they occupy (`ignore_ties = False`). 0 when the ideal DCG is 0.
///
/// Errors on empty or mismatched inputs, `k = 0`, non-finite values and
/// negative relevance (as scikit-learn).
pub fn ndcg_at_k(relevance: &[f64], score: &[f64], k: usize) -> Result<f64> {
    check(relevance, score, k)?;
    if relevance.iter().any(|&r| r < 0.0) {
        return Err(StatError::InvalidParameter(
            "relevance must be non-negative for NDCG".into(),
        ));
    }
    Ok(ndcg_unchecked(relevance, score, k))
}

fn check(relevance: &[f64], score: &[f64], k: usize) -> Result<()> {
    if relevance.len() != score.len() {
        return Err(StatError::InvalidParameter(format!(
            "relevance ({}) and score ({}) must have the same length",
            relevance.len(),
            score.len()
        )));
    }
    if relevance.is_empty() {
        return Err(StatError::EmptyData);
    }
    if k == 0 {
        return Err(StatError::InvalidParameter("k must be at least 1".into()));
    }
    ensure_finite("relevance", relevance)?;
    ensure_finite("score", score)
}

fn discount(pos: usize, k: usize) -> f64 {
    if pos < k {
        1.0 / ((pos + 2) as f64).log2()
    } else {
        0.0
    }
}

fn ndcg_unchecked(gain: &[f64], score: &[f64], k: usize) -> f64 {
    let n = gain.len();
    let mut ideal = gain.to_vec();
    ideal.sort_by(|a, b| b.total_cmp(a));
    let idcg: f64 = ideal
        .iter()
        .enumerate()
        .map(|(i, g)| g * discount(i, k))
        .sum();
    if idcg == 0.0 {
        return 0.0;
    }
    let order = descending(score);
    let mut dcg = 0.0;
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && score[order[j]] == score[order[i]] {
            j += 1;
        }
        let mean_gain = order[i..j].iter().map(|&o| gain[o]).sum::<f64>() / (j - i) as f64;
        let disc: f64 = (i..j).map(|p| discount(p, k)).sum();
        dcg += mean_gain * disc;
        i = j;
    }
    dcg / idcg
}

/// Item indices by decreasing score, ties in input order.
fn descending(score: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..score.len()).collect();
    idx.sort_by(|&a, &b| score[b].total_cmp(&score[a]));
    idx
}

/// Ranking metrics at `options.k` over the queries in `query` (one entry
/// per item; any ordered query id), with the items of a query ranked by
/// decreasing `score`.
///
/// * NDCG@k: [`ndcg_at_k`] per query (scikit-learn; tied scores averaged).
/// * The other metrics rank tied scores in input order and use evidently's
///   definitions (`legacy/metrics/recsys`), with `rel_i` the binary relevance
///   at rank `i`, `R` the relevant items of the query and `m` its item count:
///   precision@k `= hits@k / min(k, m)`, recall@k `= hits@k / R`,
///   MAP@k `= sum_{i <= min(k, m)} precision@i * rel_i / R` (evidently divides
///   by `R`, not `min(k, R)`; for `k > m` evidently also re-counts the last
///   item, which this does not), MRR@k `= 1 / rank` of the first relevant
///   item within the top k (0 if none) and hit rate@k `= 1` if any.
///
/// Errors on empty or mismatched inputs, `k = 0`, non-finite values,
/// negative relevance with graded NDCG, and when no query is left to
/// average (every query lacks relevant items and `include_no_feedback` is
/// off).
///
/// # Example
/// ```
/// use anofox_statistics::metrics::{ranking_metrics, RankingOptions};
/// let query = [1, 1, 1, 2, 2, 2];
/// let relevance = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0];
/// let score = [0.9, 0.8, 0.1, 0.7, 0.5, 0.2];
/// let m = ranking_metrics(&query, &relevance, &score,
///                         &RankingOptions { k: 2, ..Default::default() }).unwrap();
/// assert_eq!(m.n_queries, 2);
/// assert!((m.mrr - 0.5).abs() < 1e-12); // (1 + 0) / 2
/// assert!((m.hit_rate - 0.5).abs() < 1e-12);
/// ```
pub fn ranking_metrics<Q: Ord + Clone>(
    query: &[Q],
    relevance: &[f64],
    score: &[f64],
    options: &RankingOptions,
) -> Result<RankingMetrics> {
    check(relevance, score, options.k)?;
    if query.len() != relevance.len() {
        return Err(StatError::InvalidParameter(format!(
            "query ({}) and relevance ({}) must have the same length",
            query.len(),
            relevance.len()
        )));
    }
    if let Some(m) = options.min_relevance {
        if !m.is_finite() {
            return Err(StatError::InvalidParameter(format!(
                "min_relevance must be finite, got {}",
                m
            )));
        }
    } else if relevance.iter().any(|&r| r < 0.0) {
        return Err(StatError::InvalidParameter(
            "relevance must be non-negative for graded NDCG".into(),
        ));
    }
    let mut groups: BTreeMap<Q, Vec<usize>> = BTreeMap::new();
    for (i, q) in query.iter().enumerate() {
        groups.entry(q.clone()).or_default().push(i);
    }
    let k = options.k;
    let mut sums = [0.0f64; 6];
    let mut used = 0usize;
    for items in groups.values() {
        let rel_bin: Vec<f64> = items
            .iter()
            .map(|&i| {
                let r = relevance[i];
                let hit = match options.min_relevance {
                    Some(m) => r >= m,
                    None => r > 0.0,
                };
                if hit {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        let n_rel: f64 = rel_bin.iter().sum();
        if n_rel == 0.0 {
            if options.include_no_feedback {
                used += 1;
            }
            continue;
        }
        used += 1;
        let s: Vec<f64> = items.iter().map(|&i| score[i]).collect();
        let gain: Vec<f64> = match options.min_relevance {
            Some(_) => rel_bin.clone(),
            None => items.iter().map(|&i| relevance[i]).collect(),
        };
        let order = descending(&s);
        let m = order.len();
        let top = k.min(m);
        let mut hits = 0.0;
        let mut ap = 0.0;
        let mut rr = 0.0;
        for (pos, &o) in order.iter().take(top).enumerate() {
            if rel_bin[o] > 0.0 {
                hits += 1.0;
                ap += hits / (pos + 1) as f64;
                if rr == 0.0 {
                    rr = 1.0 / (pos + 1) as f64;
                }
            }
        }
        sums[0] += ndcg_unchecked(&gain, &s, k);
        sums[1] += ap / n_rel;
        sums[2] += rr;
        sums[3] += hits / top as f64;
        sums[4] += hits / n_rel;
        sums[5] += if hits > 0.0 { 1.0 } else { 0.0 };
    }
    if used == 0 {
        return Err(StatError::InvalidParameter(
            "no query has a relevant item".into(),
        ));
    }
    let u = used as f64;
    Ok(RankingMetrics {
        n_queries: used,
        ndcg: sums[0] / u,
        map: sums[1] / u,
        mrr: sums[2] / u,
        precision: sums[3] / u,
        recall: sums[4] / u,
        hit_rate: sums[5] / u,
    })
}
