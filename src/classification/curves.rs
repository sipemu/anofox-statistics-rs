use crate::error::{Result, StatError};
use crate::utils::dist::inv_or_nan;
use crate::utils::finite::{ensure_finite, ensure_no_nan};
use statrs::distribution::Normal;

/// ROC curve, as scikit-learn's `roc_curve`.
///
/// Points are ordered by decreasing threshold. The first point is
/// `(fpr, tpr) = (0, 0)` with threshold `+inf`; each later point is one
/// distinct score `t`, classifying `score >= t` as positive. The last point
/// is `(1, 1)` at the smallest score.
#[derive(Debug, Clone, PartialEq)]
pub struct RocCurve {
    /// False positive rate per point (NaN if there are no negatives).
    pub fpr: Vec<f64>,
    /// True positive rate per point (NaN if there are no positives).
    pub tpr: Vec<f64>,
    /// Decision threshold per point (`thresholds[0] = +inf`).
    pub thresholds: Vec<f64>,
}

/// Precision-recall curve, as scikit-learn's `precision_recall_curve`.
///
/// `precision` and `recall` have one more element than `thresholds`: the
/// points are ordered by increasing threshold and the final point
/// `(precision, recall) = (1, 0)` has no threshold, as in sklearn.
#[derive(Debug, Clone, PartialEq)]
pub struct PrCurve {
    /// Precision per point (0 where no sample is predicted positive).
    pub precision: Vec<f64>,
    /// Recall per point (1 everywhere if there are no positives, as sklearn).
    pub recall: Vec<f64>,
    /// Increasing thresholds; `precision[i]`/`recall[i]` use `score >= thresholds[i]`.
    pub thresholds: Vec<f64>,
}

/// One non-empty bin of a calibration curve.
#[derive(Debug, Clone, PartialEq)]
pub struct CalibrationBin {
    /// Zero-based bin index in `0..n_bins`.
    pub bin: usize,
    /// Mean predicted probability in the bin (sklearn's `prob_pred`).
    pub mean_pred: f64,
    /// Fraction of positives in the bin (sklearn's `prob_true`).
    pub frac_pos: f64,
    /// Number of samples in the bin.
    pub n: usize,
    /// Lower Wilson score bound for `frac_pos` (R `prop.test(correct = FALSE)`).
    pub conf_low: f64,
    /// Upper Wilson score bound for `frac_pos`.
    pub conf_high: f64,
}

/// Calibration curve: the non-empty bins in increasing order.
#[derive(Debug, Clone, PartialEq)]
pub struct CalibrationCurve {
    /// Non-empty bins (empty bins are dropped, as in sklearn).
    pub bins: Vec<CalibrationBin>,
    /// Number of equal-width bins on `[0, 1]`.
    pub n_bins: usize,
    /// Confidence level of the Wilson intervals.
    pub conf_level: f64,
}

fn check_inputs(scores: &[f64], labels: &[bool]) -> Result<()> {
    if scores.is_empty() {
        return Err(StatError::EmptyData);
    }
    if scores.len() != labels.len() {
        return Err(StatError::InvalidParameter(format!(
            "scores and labels must have equal length ({} vs {})",
            scores.len(),
            labels.len()
        )));
    }
    Ok(())
}

/// Cumulative false/true positive counts per distinct threshold, in
/// decreasing threshold order (sklearn `_binary_clf_curve`).
pub(crate) fn binary_clf_curve(
    scores: &[f64],
    labels: &[bool],
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    check_inputs(scores, labels)?;
    ensure_no_nan("scores", scores)?;
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    // Stable sort by decreasing score.
    idx.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
    let mut fps = Vec::new();
    let mut tps = Vec::new();
    let mut thr = Vec::new();
    let (mut tp, mut fp) = (0.0, 0.0);
    for (k, &i) in idx.iter().enumerate() {
        if labels[i] {
            tp += 1.0;
        } else {
            fp += 1.0;
        }
        let last = k + 1 == idx.len() || scores[idx[k + 1]] != scores[i];
        if last {
            fps.push(fp);
            tps.push(tp);
            thr.push(scores[i]);
        }
    }
    Ok((fps, tps, thr))
}

/// Receiver operating characteristic curve (scikit-learn `roc_curve`).
///
/// Tied scores form one point. With `drop_intermediate = true` (sklearn's
/// default) collinear points that do not change the curve's shape are
/// dropped; with `false` every distinct score yields a point, matching
/// `pROC::coords(roc, "all")`. If there are no negatives (positives) `fpr`
/// (`tpr`) is NaN, as sklearn (which warns).
///
/// # Errors
/// Empty input, unequal lengths, or NaN scores (±inf is allowed).
pub fn roc_curve(scores: &[f64], labels: &[bool], drop_intermediate: bool) -> Result<RocCurve> {
    let (mut fps, mut tps, mut thr) = binary_clf_curve(scores, labels)?;
    if drop_intermediate && fps.len() > 2 {
        let n = fps.len();
        let keep: Vec<usize> = (0..n)
            .filter(|&i| {
                i == 0
                    || i == n - 1
                    || (fps[i - 1] - 2.0 * fps[i] + fps[i + 1]) != 0.0
                    || (tps[i - 1] - 2.0 * tps[i] + tps[i + 1]) != 0.0
            })
            .collect();
        fps = keep.iter().map(|&i| fps[i]).collect();
        tps = keep.iter().map(|&i| tps[i]).collect();
        thr = keep.iter().map(|&i| thr[i]).collect();
    }
    fps.insert(0, 0.0);
    tps.insert(0, 0.0);
    thr.insert(0, f64::INFINITY);
    let fp_tot = *fps.last().unwrap();
    let tp_tot = *tps.last().unwrap();
    let rate = |v: &[f64], tot: f64| -> Vec<f64> {
        v.iter()
            .map(|&x| if tot > 0.0 { x / tot } else { f64::NAN })
            .collect()
    };
    Ok(RocCurve {
        fpr: rate(&fps, fp_tot),
        tpr: rate(&tps, tp_tot),
        thresholds: thr,
    })
}

/// Precision-recall curve (scikit-learn `precision_recall_curve`,
/// `drop_intermediate = False`).
///
/// One point per distinct score (ties form one point), ordered by increasing
/// threshold, followed by the point `(precision = 1, recall = 0)` which has
/// no threshold. Precision is `tp / (tp + fp)` for `score >= threshold`.
///
/// # Errors
/// Empty input, unequal lengths, or NaN scores.
pub fn pr_curve(scores: &[f64], labels: &[bool]) -> Result<PrCurve> {
    let (fps, tps, thr) = binary_clf_curve(scores, labels)?;
    let tp_tot = *tps.last().unwrap();
    let mut precision: Vec<f64> = fps
        .iter()
        .zip(&tps)
        .map(|(&f, &t)| if t + f > 0.0 { t / (t + f) } else { 0.0 })
        .collect();
    let mut recall: Vec<f64> = tps
        .iter()
        .map(|&t| if tp_tot > 0.0 { t / tp_tot } else { 1.0 })
        .collect();
    let mut thresholds = thr;
    precision.reverse();
    recall.reverse();
    thresholds.reverse();
    precision.push(1.0);
    recall.push(0.0);
    Ok(PrCurve {
        precision,
        recall,
        thresholds,
    })
}

/// Wilson score interval for `x` successes out of `n` (no continuity
/// correction), as R `prop.test(x, n, correct = FALSE)$conf.int`.
pub(crate) fn wilson_interval(x: f64, n: f64, conf_level: f64) -> (f64, f64) {
    let z = inv_or_nan(&Normal::standard(), (1.0 + conf_level) / 2.0);
    let p = x / n;
    let z2 = z * z;
    let center = p + z2 / (2.0 * n);
    let half = z * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt();
    let denom = 1.0 + z2 / n;
    let lo = if x == 0.0 {
        0.0
    } else {
        (center - half) / denom
    };
    let hi = if x == n { 1.0 } else { (center + half) / denom };
    (lo.max(0.0), hi.min(1.0))
}

/// Calibration (reliability) curve with 95% Wilson intervals.
///
/// See [`calibration_curve_with_conf_level`].
pub fn calibration_curve(
    probs: &[f64],
    labels: &[bool],
    n_bins: usize,
) -> Result<CalibrationCurve> {
    calibration_curve_with_conf_level(probs, labels, n_bins, 0.95)
}

/// Calibration (reliability) curve, as scikit-learn
/// `calibration_curve(strategy = "uniform")`, plus a Wilson score interval
/// for the fraction of positives in each bin.
///
/// `[0, 1]` is split into `n_bins` equal-width bins. Bin `0` is
/// `[0, 1/n_bins]`, bin `i > 0` is `(i/n_bins, (i+1)/n_bins]` (sklearn's
/// `searchsorted` assignment). Empty bins are omitted.
///
/// # Errors
/// Empty input, unequal lengths, `n_bins == 0`, probabilities outside
/// `[0, 1]` or non-finite, or `conf_level` outside `(0, 1)`.
pub fn calibration_curve_with_conf_level(
    probs: &[f64],
    labels: &[bool],
    n_bins: usize,
    conf_level: f64,
) -> Result<CalibrationCurve> {
    check_inputs(probs, labels)?;
    ensure_finite("probs", probs)?;
    if n_bins == 0 {
        return Err(StatError::InvalidParameter("n_bins must be >= 1".into()));
    }
    if !(conf_level > 0.0 && conf_level < 1.0) {
        return Err(StatError::InvalidParameter(format!(
            "conf_level must be in (0, 1), got {}",
            conf_level
        )));
    }
    if let Some(p) = probs.iter().find(|p| !(0.0..=1.0).contains(*p)) {
        return Err(StatError::InvalidParameter(format!(
            "probabilities must be in [0, 1], got {}",
            p
        )));
    }
    // Inner bin edges, computed as numpy.linspace(0, 1, n_bins + 1)[1:-1].
    let step = 1.0 / n_bins as f64;
    let edges: Vec<f64> = (1..n_bins).map(|i| i as f64 * step).collect();
    let mut sum_p = vec![0.0; n_bins];
    let mut sum_y = vec![0.0; n_bins];
    let mut cnt = vec![0usize; n_bins];
    for (&p, &y) in probs.iter().zip(labels) {
        // searchsorted(side = "left"): number of edges strictly below p.
        let b = edges.partition_point(|&e| e < p);
        sum_p[b] += p;
        sum_y[b] += if y { 1.0 } else { 0.0 };
        cnt[b] += 1;
    }
    let bins = (0..n_bins)
        .filter(|&b| cnt[b] > 0)
        .map(|b| {
            let n = cnt[b] as f64;
            let (lo, hi) = wilson_interval(sum_y[b], n, conf_level);
            CalibrationBin {
                bin: b,
                mean_pred: sum_p[b] / n,
                frac_pos: sum_y[b] / n,
                n: cnt[b],
                conf_low: lo,
                conf_high: hi,
            }
        })
        .collect();
    Ok(CalibrationCurve {
        bins,
        n_bins,
        conf_level,
    })
}
