//! Drift between two samples of embeddings (evidently's
//! `embedding_drift_methods`: `distance`, `ratio` and `mmd`).
//!
//! Embeddings are rows (`&[Vec<f64>]`, one vector per observation). evidently
//! optionally reduces both samples with PCA first; do that with
//! [`multivariate::pca`](crate::multivariate::pca) on the reference and
//! [`Pca::transform`](crate::multivariate::Pca::transform) of both samples.
//! evidently's classifier-based `model` method is out of scope; its
//! bootstrap threshold for random scores is
//! [`auc_null_quantile`](crate::classification::auc_null_quantile).

use super::{column_drift_numeric, DriftOptions, DriftTest};
use crate::error::{Result, StatError};
use crate::modern::mmd::mmd_squared;
use crate::modern::Kernel;
use crate::utils::finite::ensure_finite;
use crate::utils::rng::index;
use rand::seq::index::sample;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Distance between the mean embeddings for [`embedding_distance_drift`]
/// (scipy's `euclidean`, `cosine`, `cityblock`, `chebyshev`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EmbeddingDistance {
    /// `sqrt(sum (a - b)^2)`.
    #[default]
    Euclidean,
    /// `1 - a.b / (|a| |b|)`.
    Cosine,
    /// `sum |a - b|`.
    Cityblock,
    /// `max |a - b|`.
    Chebyshev,
}

impl EmbeddingDistance {
    fn eval(self, a: &[f64], b: &[f64]) -> f64 {
        let diffs = a.iter().zip(b).map(|(x, y)| (x - y).abs());
        match self {
            EmbeddingDistance::Euclidean => diffs.map(|v| v * v).sum::<f64>().sqrt(),
            EmbeddingDistance::Cityblock => diffs.sum(),
            EmbeddingDistance::Chebyshev => diffs.fold(0.0, f64::max),
            EmbeddingDistance::Cosine => {
                let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
                let na = a.iter().map(|x| x * x).sum::<f64>().sqrt();
                let nb = b.iter().map(|x| x * x).sum::<f64>().sqrt();
                1.0 - dot / (na * nb)
            }
        }
    }
}

/// How an embedding drift score becomes a drift decision.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EmbeddingThreshold {
    /// Drift when the score exceeds this value (evidently: 0.2 for the
    /// distance, 0.015 for MMD).
    Fixed(f64),
    /// Drift when the score exceeds the `quantile` (R type 7, numpy's
    /// default) of `n_bootstrap` scores between two with-replacement
    /// resamples of the reference, of sizes `floor(m^2 / (m + n))` and
    /// `floor(m n / (m + n))` (evidently: 100 resamples, quantile 0.95 for
    /// the distance and 0.05 for MMD). `seed` makes it reproducible
    /// (`ChaCha8Rng`); evidently's numpy draws are not reproduced.
    Bootstrap {
        /// Number of resamples.
        n_bootstrap: usize,
        /// Quantile of the resampled scores, in `[0, 1]`.
        quantile: f64,
        /// Seed of the resampling (`None`: from entropy).
        seed: Option<u64>,
    },
}

/// Result of an embedding drift measure.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingDriftResult {
    /// evidently's name of the method: `"distance"`, `"ratio"` or `"mmd"`.
    pub method: &'static str,
    /// The drift score.
    pub score: f64,
    /// The threshold the score was compared with (the bootstrap quantile
    /// when bootstrapped).
    pub threshold: f64,
    /// `score > threshold`.
    pub drifted: bool,
}

/// Validate two embedding samples; returns the dimension.
fn validate(reference: &[Vec<f64>], current: &[Vec<f64>]) -> Result<usize> {
    if reference.is_empty() || current.is_empty() {
        return Err(StatError::EmptyData);
    }
    let d = reference[0].len();
    if d == 0 {
        return Err(StatError::InvalidParameter(
            "embeddings must have at least one dimension".into(),
        ));
    }
    for (name, rows) in [("reference", reference), ("current", current)] {
        for (i, row) in rows.iter().enumerate() {
            if row.len() != d {
                return Err(StatError::InvalidParameter(format!(
                    "{}[{}] has {} dimensions, expected {}",
                    name,
                    i,
                    row.len(),
                    d
                )));
            }
            ensure_finite(&format!("{}[{}]", name, i), row)?;
        }
    }
    Ok(d)
}

/// Column means of the rows.
fn mean_of<'a>(rows: impl Iterator<Item = &'a Vec<f64>>, d: usize) -> Vec<f64> {
    let mut sum = vec![0.0; d];
    let mut count = 0usize;
    for row in rows {
        for (s, v) in sum.iter_mut().zip(row) {
            *s += v;
        }
        count += 1;
    }
    sum.iter().map(|s| s / count as f64).collect()
}

/// Quantile type 7 (R's default, numpy's `percentile`).
fn quantile7(mut v: Vec<f64>, prob: f64) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    let h = (v.len() - 1) as f64 * prob;
    let lo = h.floor() as usize;
    let hi = (lo + 1).min(v.len() - 1);
    v[lo] + (h - lo as f64) * (v[hi] - v[lo])
}

/// Sizes `(max(floor(m^2 / (m + n)), 1), max(floor(m n / (m + n)), 1))` of
/// the two reference resamples.
fn resample_sizes(m: usize, n: usize) -> (usize, usize) {
    ((m * m / (m + n)).max(1), (m * n / (m + n)).max(1))
}

/// Validate the parameters of a threshold.
fn check_threshold(threshold: &EmbeddingThreshold) -> Result<()> {
    match *threshold {
        EmbeddingThreshold::Fixed(t) if !t.is_finite() => Err(StatError::InvalidParameter(
            format!("threshold must be finite, got {}", t),
        )),
        EmbeddingThreshold::Bootstrap { n_bootstrap: 0, .. } => Err(StatError::InvalidParameter(
            "n_bootstrap must be at least 1".into(),
        )),
        EmbeddingThreshold::Bootstrap { quantile, .. } if !(0.0..=1.0).contains(&quantile) => Err(
            StatError::InvalidParameter(format!("quantile must be in [0, 1], got {}", quantile)),
        ),
        _ => Ok(()),
    }
}

fn rng(seed: Option<u64>) -> ChaCha8Rng {
    match seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => ChaCha8Rng::from_entropy(),
    }
}

/// Embedding drift as the distance between the mean embeddings of the
/// reference and the current sample (evidently's `distance` method).
///
/// The score is `metric(mean(reference), mean(current))`; drift when it
/// exceeds the `threshold` (evidently's default `Fixed(0.2)`, or a
/// bootstrap quantile of the distances between the means of two reference
/// resamples, see [`EmbeddingThreshold::Bootstrap`]).
///
/// Errors on empty samples, rows of unequal or zero length, non-finite
/// values and invalid threshold parameters.
///
/// R equivalent: `dist(rbind(colMeans(ref), colMeans(cur)), method =
/// "euclidean" | "manhattan" | "maximum")`; cosine
/// `1 - sum(a * b) / sqrt(sum(a^2) * sum(b^2))`.
///
/// # Example
/// ```
/// use anofox_statistics::drift::{embedding_distance_drift, EmbeddingDistance, EmbeddingThreshold};
/// let r = vec![vec![0.0, 0.0], vec![2.0, 0.0]];
/// let c = vec![vec![1.0, 3.0], vec![1.0, 1.0]];
/// let res = embedding_distance_drift(&r, &c, EmbeddingDistance::Euclidean,
///     EmbeddingThreshold::Fixed(0.2)).unwrap();
/// assert_eq!(res.score, 2.0);
/// assert!(res.drifted);
/// ```
pub fn embedding_distance_drift(
    reference: &[Vec<f64>],
    current: &[Vec<f64>],
    metric: EmbeddingDistance,
    threshold: EmbeddingThreshold,
) -> Result<EmbeddingDriftResult> {
    let d = validate(reference, current)?;
    check_threshold(&threshold)?;
    let score = metric.eval(&mean_of(reference.iter(), d), &mean_of(current.iter(), d));
    let threshold = match threshold {
        EmbeddingThreshold::Fixed(t) => t,
        EmbeddingThreshold::Bootstrap {
            n_bootstrap,
            quantile,
            seed,
        } => {
            let m = reference.len();
            let (a, b) = resample_sizes(m, current.len());
            let mut rng = rng(seed);
            let null: Vec<f64> = (0..n_bootstrap)
                .map(|_| {
                    let ia: Vec<usize> = (0..a).map(|_| index(&mut rng, m)).collect();
                    let ib: Vec<usize> = (0..b).map(|_| index(&mut rng, m)).collect();
                    metric.eval(
                        &mean_of(ia.iter().map(|&i| &reference[i]), d),
                        &mean_of(ib.iter().map(|&i| &reference[i]), d),
                    )
                })
                .collect();
            quantile7(null, quantile)
        }
    };
    Ok(EmbeddingDriftResult {
        method: "distance",
        score,
        threshold,
        drifted: score > threshold,
    })
}

/// Embedding drift as the share of drifted components (evidently's `ratio`
/// method).
///
/// Every embedding dimension is tested as a numeric column with
/// `component_test` at `component_threshold` (the test's default when
/// `None`) by [`column_drift_numeric`]; the score is the share of drifted
/// dimensions, with drift when it exceeds `threshold`. evidently's defaults
/// are [`DriftTest::Wasserstein`] at 0.1 and a share of 0.2. `seed` seeds the
/// permutation tests (`TVD`, `empirical_mmd`).
///
/// Errors as [`embedding_distance_drift`], on a non-finite `threshold` and
/// whenever a component test fails.
///
/// # Example
/// ```
/// use anofox_statistics::drift::{embedding_ratio_drift, DriftTest};
/// let r: Vec<Vec<f64>> = (0..30).map(|i| vec![i as f64, (i % 7) as f64]).collect();
/// let c: Vec<Vec<f64>> = (0..30).map(|i| vec![i as f64 + 10.0, (i % 7) as f64]).collect();
/// let res = embedding_ratio_drift(&r, &c, DriftTest::Wasserstein, None, 0.2, None).unwrap();
/// assert_eq!(res.score, 0.5);
/// assert!(res.drifted);
/// ```
pub fn embedding_ratio_drift(
    reference: &[Vec<f64>],
    current: &[Vec<f64>],
    component_test: DriftTest,
    component_threshold: Option<f64>,
    threshold: f64,
    seed: Option<u64>,
) -> Result<EmbeddingDriftResult> {
    let d = validate(reference, current)?;
    check_threshold(&EmbeddingThreshold::Fixed(threshold))?;
    let options = DriftOptions {
        test: Some(component_test),
        threshold: component_threshold,
        seed,
        ..DriftOptions::default()
    };
    let mut drifted = 0usize;
    for j in 0..d {
        let a: Vec<f64> = reference.iter().map(|row| row[j]).collect();
        let b: Vec<f64> = current.iter().map(|row| row[j]).collect();
        if column_drift_numeric(&a, &b, &options)?.drifted {
            drifted += 1;
        }
    }
    let score = drifted as f64 / d as f64;
    Ok(EmbeddingDriftResult {
        method: "ratio",
        score,
        threshold,
        drifted: score > threshold,
    })
}

/// Rows evidently uses for the median distance: all rows up to 1000, else
/// 1000 rows drawn without replacement with a fixed seed.
fn median_rows(rows: &[Vec<f64>], seed: u64) -> Vec<&Vec<f64>> {
    const MAX_ROWS: usize = super::defaults::EMBEDDING_MEDIAN_MAX_ROWS;
    if rows.len() <= MAX_ROWS {
        return rows.iter().collect();
    }
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    sample(&mut rng, rows.len(), MAX_ROWS)
        .into_iter()
        .map(|i| &rows[i])
        .collect()
}

fn euclidean(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Median (mean of the middle two for an even count) of all distances
/// between the rows of `a` and `b`.
fn median_distance(a: &[&Vec<f64>], b: &[&Vec<f64>]) -> f64 {
    let mut dist: Vec<f64> = a
        .iter()
        .flat_map(|x| b.iter().map(move |y| euclidean(x, y)))
        .collect();
    let k = dist.len();
    dist.sort_unstable_by(|p, q| p.total_cmp(q));
    if k % 2 == 1 {
        dist[k / 2]
    } else {
        (dist[k / 2 - 1] + dist[k / 2]) / 2.0
    }
}

/// Gaussian kernel `exp(-|a - b|^2 / sigma2)` as the crate's [`Kernel`].
fn gaussian(sigma2: f64) -> Kernel {
    Kernel::Gaussian {
        bandwidth: (sigma2 / 2.0).sqrt(),
    }
}

/// Bandwidth of the Gaussian kernel of [`embedding_mmd_drift`], as the
/// crate's [`Kernel::Gaussian`] `bandwidth` (`k = exp(-d^2 / (2 bandwidth^2))`):
/// `median / sqrt(2)`, with `median` the median euclidean distance between
/// reference and current rows (at most 1000 rows per sample, larger samples
/// subsampled with a fixed seed; `1 / sqrt(2)` if the median is 0), i.e.
/// evidently's kernel `exp(-d^2 / median^2)`.
///
/// Errors as [`embedding_distance_drift`].
///
/// R equivalent: `median(as.matrix(dist(rbind(ref, cur)))[1:m, m + 1:n]) / sqrt(2)`.
pub fn embedding_mmd_bandwidth(reference: &[Vec<f64>], current: &[Vec<f64>]) -> Result<f64> {
    validate(reference, current)?;
    let med = median_distance(&median_rows(reference, 0), &median_rows(current, 0));
    let sigma2 = if med > 0.0 { med * med } else { 1.0 };
    Ok((sigma2 / 2.0).sqrt())
}

/// Embedding drift by the maximum mean discrepancy (evidently's `mmd`
/// method).
///
/// With `sigma^2` the squared median euclidean distance between reference
/// and current rows (`1` if that median is 0), the score is the unbiased
/// MMD² with the Gaussian kernel `exp(-|a - b|^2 / sigma^2)`, floored at 0.
/// Drift when the unfloored MMD² exceeds the `threshold` (evidently's
/// default `Fixed(0.015)`), or a bootstrap quantile (evidently's default
/// quantile 0.05) of the MMD² between two reference resamples, see
/// [`EmbeddingThreshold::Bootstrap`]. The bootstrap uses the bandwidth of the
/// reference alone (median distance between reference rows, the zero
/// self-distances included), and, as evidently, leaves out only the kernel
/// diagonal of each resample, so repeated rows count as pairs.
///
/// evidently computes the median distances on at most 1000 rows per
/// sample, drawn without replacement; here larger samples are subsampled
/// with a fixed seed. The kernel sums use all rows: O((m + n)^2 d) time;
/// the bootstrap stores the `m x m` reference kernel matrix.
///
/// Errors as [`embedding_distance_drift`], and on samples of fewer than two
/// rows.
///
/// R equivalent: `s2 <- median(as.matrix(dist(rbind(ref, cur)))[1:m, m + 1:n])^2`,
/// then the unbiased MMD² of `exp(-D^2 / s2)`.
pub fn embedding_mmd_drift(
    reference: &[Vec<f64>],
    current: &[Vec<f64>],
    threshold: EmbeddingThreshold,
) -> Result<EmbeddingDriftResult> {
    validate(reference, current)?;
    check_threshold(&threshold)?;
    let (m, n) = (reference.len(), current.len());
    if m < 2 || n < 2 {
        return Err(StatError::InsufficientData {
            needed: 2,
            got: m.min(n),
        });
    }
    let positive = |s: f64| if s > 0.0 { s * s } else { 1.0 };
    let sigma2 = positive(median_distance(
        &median_rows(reference, 0),
        &median_rows(current, 0),
    ));
    let xs: Vec<&[f64]> = reference.iter().map(|v| v.as_slice()).collect();
    let ys: Vec<&[f64]> = current.iter().map(|v| v.as_slice()).collect();
    let mmd2 = mmd_squared(&xs, &ys, gaussian(sigma2));
    let threshold = match threshold {
        EmbeddingThreshold::Fixed(t) => t,
        EmbeddingThreshold::Bootstrap {
            n_bootstrap,
            quantile,
            seed,
        } => {
            let rows = median_rows(reference, 0);
            let gamma = 1.0 / positive(median_distance(&rows, &rows));
            let k: Vec<Vec<f64>> = reference
                .iter()
                .map(|a| {
                    reference
                        .iter()
                        .map(|b| (-gamma * euclidean(a, b).powi(2)).exp())
                        .collect()
                })
                .collect();
            // MMD² needs at least two rows per resample.
            let (a, b) = resample_sizes(m, n);
            let (a, b) = (a.max(2), b.max(2));
            let mut rng = rng(seed);
            let within = |idx: &[usize]| {
                let s: f64 = idx
                    .iter()
                    .map(|&i| idx.iter().map(|&j| k[i][j]).sum::<f64>())
                    .sum();
                let diag: f64 = idx.iter().map(|&i| k[i][i]).sum();
                let l = idx.len() as f64;
                (s - diag) / (l * (l - 1.0))
            };
            let null: Vec<f64> = (0..n_bootstrap)
                .map(|_| {
                    let ia: Vec<usize> = (0..a).map(|_| index(&mut rng, m)).collect();
                    let ib: Vec<usize> = (0..b).map(|_| index(&mut rng, m)).collect();
                    let cross: f64 = ia
                        .iter()
                        .map(|&i| ib.iter().map(|&j| k[i][j]).sum::<f64>())
                        .sum();
                    within(&ia) + within(&ib) - 2.0 * cross / (a * b) as f64
                })
                .collect();
            quantile7(null, quantile)
        }
    };
    Ok(EmbeddingDriftResult {
        method: "mmd",
        score: mmd2.max(0.0),
        threshold,
        drifted: mmd2 > threshold,
    })
}
