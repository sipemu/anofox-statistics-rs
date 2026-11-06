//! Confidence-based performance estimation (CBPE) of a binary classifier,
//! as NannyML's `CBPE(problem_type = "classification_binary")`.
//!
//! On a labelled reference period the predicted scores are calibrated
//! ([`CbpeCalibration`]) and the sampling-error components of each metric are
//! stored. On a monitored chunk without labels, each row with calibrated
//! probability `p` contributes `p` to the expected true positives (if
//! predicted positive) or false negatives (if predicted negative), and `1 - p`
//! to the expected false positives or true negatives. Accuracy, precision,
//! recall, F1 and specificity follow from the expected confusion matrix, and
//! ROC AUC from the calibrated probabilities ordered by the raw scores.
//!
//! The sampling errors are NannyML's (`nannyml.sampling_error.
//! binary_classification`): the standard deviation of an observation-level
//! contribution on the reference, divided by `sqrt(n * fraction)` for a chunk
//! of `n` rows.
//!
//! Validated against NannyML 0.13.1 on its synthetic binary-classification
//! dataset and on simulated continuous scores
//! (`R/generate_monitoring_refs.py`, `tests/monitoring_reference.rs`).

use crate::classification::{IsotonicCalibrator, PlattCalibrator};
use crate::error::{Result, StatError};
use crate::utils::finite::ensure_finite;

/// NannyML's `SAMPLING_ERROR_RANGE`: the confidence band is `estimate ± 3 *
/// sampling_error`.
pub const SAMPLING_ERROR_RANGE: f64 = 3.0;

/// How the reference scores are calibrated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CbpeCalibration {
    /// Isotonic regression (NannyML's `calibration = "isotonic"`, its default).
    #[default]
    Isotonic,
    /// Platt's sigmoid.
    Platt,
    /// Use the scores as probabilities (NannyML's `NoopCalibrator`, used when
    /// its check finds that calibration does not lower the expected
    /// calibration error).
    None,
}

/// Options of [`fit_reference`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CbpeOptions {
    /// Calibration of the scores. Default [`CbpeCalibration::Isotonic`].
    pub calibration: CbpeCalibration,
    /// A row is predicted positive when `score >= threshold`. Default 0.5.
    pub threshold: f64,
}

impl Default for CbpeOptions {
    fn default() -> Self {
        Self {
            calibration: CbpeCalibration::Isotonic,
            threshold: 0.5,
        }
    }
}

/// A fitted calibration map.
#[derive(Debug, Clone, PartialEq)]
pub enum Calibrator {
    /// Identity: the scores are used as probabilities.
    Identity,
    Isotonic(IsotonicCalibrator),
    Platt(PlattCalibrator),
}

impl Calibrator {
    /// Calibrated probability of one score.
    pub fn calibrate_one(&self, score: f64) -> f64 {
        match self {
            Calibrator::Identity => score,
            Calibrator::Isotonic(c) => c.calibrate_one(score),
            Calibrator::Platt(c) => c.calibrate_one(score),
        }
    }

    /// Calibrated probabilities of the scores.
    pub fn calibrate(&self, scores: &[f64]) -> Vec<f64> {
        scores.iter().map(|&s| self.calibrate_one(s)).collect()
    }
}

/// Sampling-error components computed on the reference period: for each
/// metric the population standard deviation of its observation-level
/// contributions (`*_std`) and the share of reference rows they come from
/// (`*_fraction`). A chunk of `n` rows has sampling error
/// `std / sqrt(n * fraction)` (`std / sqrt(n)` for accuracy, `std * sqrt(n)`
/// for the confusion-matrix counts).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CbpeSamplingComponents {
    pub accuracy_std: f64,
    pub precision_std: f64,
    pub precision_fraction: f64,
    pub recall_std: f64,
    pub recall_fraction: f64,
    pub specificity_std: f64,
    pub specificity_fraction: f64,
    pub f1_std: f64,
    pub f1_fraction: f64,
    pub roc_auc_std: f64,
    pub roc_auc_fraction: f64,
    pub true_positive_std: f64,
    pub false_positive_std: f64,
    pub false_negative_std: f64,
    pub true_negative_std: f64,
}

/// A CBPE model fitted on a labelled reference period by [`fit_reference`].
/// All fields are public, so a model can be stored and rebuilt from its
/// parts.
#[derive(Debug, Clone, PartialEq)]
pub struct CbpeModel {
    /// The calibration map fitted on the reference.
    pub calibrator: Calibrator,
    /// Decision threshold of [`estimate`](CbpeModel::estimate).
    pub threshold: f64,
    /// Sampling-error components from the reference.
    pub sampling: CbpeSamplingComponents,
}

/// An estimated metric with its sampling error and NannyML's confidence band
/// `value ∓ 3 * sampling_error`, clipped to the metric's range.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MetricEstimate {
    pub value: f64,
    pub sampling_error: f64,
    pub lower_confidence: f64,
    pub upper_confidence: f64,
}

impl MetricEstimate {
    fn new(value: f64, sampling_error: f64, lo: f64, hi: f64) -> Self {
        Self {
            value,
            sampling_error,
            lower_confidence: (value - SAMPLING_ERROR_RANGE * sampling_error).max(lo),
            upper_confidence: (value + SAMPLING_ERROR_RANGE * sampling_error).min(hi),
        }
    }
}

/// CBPE estimates for one chunk. The rates lie in `[0, 1]`; the
/// confusion-matrix entries are expected counts (NannyML's
/// `normalize_confusion_matrix = None`), bounded below by 0.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CbpeEstimate {
    /// Rows in the chunk.
    pub n: usize,
    pub roc_auc: MetricEstimate,
    pub f1: MetricEstimate,
    pub precision: MetricEstimate,
    pub recall: MetricEstimate,
    pub specificity: MetricEstimate,
    pub accuracy: MetricEstimate,
    pub true_positive: MetricEstimate,
    pub false_positive: MetricEstimate,
    pub false_negative: MetricEstimate,
    pub true_negative: MetricEstimate,
}

/// Realized metrics of a labelled chunk, with NannyML's rules for undefined
/// values: ROC AUC is NaN when the labels have one class; F1, precision and
/// recall are NaN when the labels or the predictions have one class;
/// specificity is NaN without actual negatives.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RealizedPerformance {
    pub n: usize,
    pub roc_auc: f64,
    pub f1: f64,
    pub precision: f64,
    pub recall: f64,
    pub specificity: f64,
    pub accuracy: f64,
    pub true_positive: f64,
    pub false_positive: f64,
    pub false_negative: f64,
    pub true_negative: f64,
}

fn check_scores(scores: &[f64]) -> Result<()> {
    if scores.is_empty() {
        return Err(StatError::EmptyData);
    }
    ensure_finite("scores", scores)
}

fn check_len(name: &str, len: usize, n: usize) -> Result<()> {
    if len != n {
        return Err(StatError::InvalidParameter(format!(
            "{} has {} values, expected {}",
            name, len, n
        )));
    }
    Ok(())
}

fn predictions(scores: &[f64], threshold: f64) -> Vec<bool> {
    scores.iter().map(|&s| s >= threshold).collect()
}

/// Population standard deviation (`np.std`).
fn pstd(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let n = v.len() as f64;
    let m = v.iter().sum::<f64>() / n;
    (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / n).sqrt()
}

/// Std of a 0/1 vector with `ones` ones out of `total`.
fn bernoulli_std(ones: usize, total: usize) -> f64 {
    if total == 0 {
        return f64::NAN;
    }
    let p = ones as f64 / total as f64;
    (p * (1.0 - p)).sqrt()
}

/// NannyML's `auroc_sampling_error_components`: with the minority class
/// relabelled as positive, the standard deviation over the positives of the
/// share of negatives scored below them. Tied scores count half
/// (NannyML's order of tied scores is that of an unstable sort); NannyML
/// also drops a positive that has the lowest score, which this does not.
fn auroc_components(scores: &[f64], labels: &[bool]) -> (f64, f64) {
    let n = scores.len();
    let n_pos_raw = labels.iter().filter(|&&l| l).count();
    let flip = n_pos_raw as f64 / n as f64 > 0.5;
    let (s, y): (Vec<f64>, Vec<bool>) = if flip {
        (
            scores.iter().map(|v| 1.0 - v).collect(),
            labels.iter().map(|l| !l).collect(),
        )
    } else {
        (scores.to_vec(), labels.to_vec())
    };
    let n_pos = y.iter().filter(|&&l| l).count();
    let n_neg = n - n_pos;
    if n_pos == 0 || n_neg == 0 {
        return (f64::NAN, n_pos as f64 / n as f64);
    }
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| s[a].total_cmp(&s[b]));
    let mut ser = Vec::with_capacity(n_pos);
    let mut neg_below = 0usize;
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && s[idx[j]] == s[idx[i]] {
            j += 1;
        }
        let tied_neg = idx[i..j].iter().filter(|&&k| !y[k]).count();
        let share = (neg_below as f64 + 0.5 * tied_neg as f64) / n_neg as f64;
        for &k in &idx[i..j] {
            if y[k] {
                ser.push(share);
            }
        }
        neg_below += tied_neg;
        i = j;
    }
    (pstd(&ser), n_pos as f64 / n as f64)
}

fn sampling_components(
    scores: &[f64],
    preds: &[bool],
    labels: &[bool],
) -> Result<CbpeSamplingComponents> {
    let n = labels.len();
    let (mut tp, mut fp, mut fn_, mut tn) = (0usize, 0usize, 0usize, 0usize);
    for (&p, &y) in preds.iter().zip(labels) {
        match (p, y) {
            (true, true) => tp += 1,
            (true, false) => fp += 1,
            (false, true) => fn_ += 1,
            (false, false) => tn += 1,
        }
    }
    let nf = n as f64;
    // F1: observation-level values (1 for TP, 0 for FP / FN) times the
    // correcting factor (TP + FP + FN) / (TP + (FP + FN) / 2).
    let rel = tp + fp + fn_;
    let (f1_std, f1_fraction) = if rel == 0 {
        (f64::NAN, 0.0)
    } else {
        let factor = rel as f64 / (tp as f64 + 0.5 * (fp + fn_) as f64);
        (factor * bernoulli_std(tp, rel), rel as f64 / nf)
    };
    let (roc_auc_std, roc_auc_fraction) = auroc_components(scores, labels);
    Ok(CbpeSamplingComponents {
        accuracy_std: bernoulli_std(tp + tn, n),
        precision_std: bernoulli_std(tp, tp + fp),
        precision_fraction: (tp + fp) as f64 / nf,
        // NannyML: observations TP (1) and FN (0); fraction = actual positives.
        recall_std: bernoulli_std(tp, tp + fn_),
        recall_fraction: (tp + fn_) as f64 / nf,
        specificity_std: bernoulli_std(tn, tn + fp),
        specificity_fraction: (tn + fp) as f64 / nf,
        f1_std,
        f1_fraction,
        roc_auc_std,
        roc_auc_fraction,
        true_positive_std: bernoulli_std(tp, n),
        false_positive_std: bernoulli_std(fp, n),
        false_negative_std: bernoulli_std(fn_, n),
        true_negative_std: bernoulli_std(tn, n),
    })
}

/// Fits CBPE on a labelled reference period: the calibration map of
/// `options.calibration` and the sampling-error components, with predictions
/// `score >= options.threshold`.
///
/// Errors on empty or mismatched inputs, non-finite scores or threshold, and
/// labels with a single class.
///
/// Python equivalent: `nannyml.CBPE(..., problem_type =
/// "classification_binary", calibration = "isotonic").fit(reference)` with
/// `y_pred = y_pred_proba >= threshold`. NannyML decides by a randomised
/// cross-validation whether to calibrate at all; here the caller chooses
/// ([`CbpeCalibration::None`] for no calibration).
pub fn fit_reference(scores: &[f64], labels: &[bool], options: &CbpeOptions) -> Result<CbpeModel> {
    check_scores(scores)?;
    if !options.threshold.is_finite() {
        return Err(StatError::InvalidParameter(format!(
            "threshold must be finite, got {}",
            options.threshold
        )));
    }
    let preds = predictions(scores, options.threshold);
    let mut m = fit_reference_with_predictions(scores, &preds, labels, options.calibration)?;
    m.threshold = options.threshold;
    Ok(m)
}

/// As [`fit_reference`], with the model's predictions given per row (NannyML's
/// `y_pred` column) instead of a threshold. The returned model's
/// [`threshold`](CbpeModel::threshold) is 0.5; use
/// [`CbpeModel::estimate_with_predictions`] when the predictions do not come
/// from a threshold.
pub fn fit_reference_with_predictions(
    scores: &[f64],
    predictions: &[bool],
    labels: &[bool],
    calibration: CbpeCalibration,
) -> Result<CbpeModel> {
    check_scores(scores)?;
    check_len("predictions", predictions.len(), scores.len())?;
    check_len("labels", labels.len(), scores.len())?;
    let n_pos = labels.iter().filter(|&&l| l).count();
    if n_pos == 0 || n_pos == labels.len() {
        return Err(StatError::InvalidParameter(
            "the reference labels must contain both classes".into(),
        ));
    }
    let calibrator = match calibration {
        CbpeCalibration::Isotonic => Calibrator::Isotonic(IsotonicCalibrator::fit(scores, labels)?),
        CbpeCalibration::Platt => Calibrator::Platt(PlattCalibrator::fit(scores, labels)?),
        CbpeCalibration::None => Calibrator::Identity,
    };
    Ok(CbpeModel {
        calibrator,
        threshold: 0.5,
        sampling: sampling_components(scores, predictions, labels)?,
    })
}

/// NannyML's `estimate_roc_auc`: the calibrated probabilities, ordered by
/// decreasing raw score, are the expected true positives along the ROC
/// curve (cumulative sums rounded to 5 decimals, as NannyML), whose area is
/// the estimate. Tied raw scores have equal calibrated probabilities, so
/// their order does not matter.
fn estimate_roc_auc(calibrated: &[f64], scores: &[f64]) -> f64 {
    let n = scores.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
    let round5 = |v: f64| (v * 1e5).round() / 1e5;
    let mut tps = Vec::with_capacity(n + 1);
    let mut fps = Vec::with_capacity(n + 1);
    tps.push(0.0);
    fps.push(0.0);
    let mut cum = 0.0;
    for (k, &i) in idx.iter().enumerate() {
        cum += calibrated[i];
        tps.push(round5(cum));
        fps.push(round5((k + 1) as f64 - cum));
    }
    let (tmax, fmax) = (tps[n], fps[n]);
    if tmax <= 0.0 || fmax <= 0.0 {
        return f64::NAN;
    }
    let mut area = 0.0;
    for k in 1..=n {
        let dx = (fps[k] - fps[k - 1]) / fmax;
        area += dx * (tps[k] + tps[k - 1]) / (2.0 * tmax);
    }
    area
}

fn ratio_or_zero(num: f64, den: f64) -> f64 {
    if den != 0.0 {
        num / den
    } else {
        0.0
    }
}

impl CbpeModel {
    /// Rebuilds a model from its parts (the fields of a fitted model).
    pub fn from_parts(
        calibrator: Calibrator,
        threshold: f64,
        sampling: CbpeSamplingComponents,
    ) -> Self {
        Self {
            calibrator,
            threshold,
            sampling,
        }
    }

    /// Estimates the performance on a chunk of unlabelled scores, with
    /// predictions `score >= self.threshold`.
    ///
    /// Errors on an empty chunk or non-finite scores. A metric whose
    /// expected denominator is zero is 0 (NannyML); ROC AUC is NaN when all
    /// expected labels are of one class.
    pub fn estimate(&self, scores: &[f64]) -> Result<CbpeEstimate> {
        check_scores(scores)?;
        let preds = predictions(scores, self.threshold);
        self.estimate_with_predictions(scores, &preds)
    }

    /// As [`estimate`](Self::estimate), with the predictions given per row.
    pub fn estimate_with_predictions(
        &self,
        scores: &[f64],
        predictions: &[bool],
    ) -> Result<CbpeEstimate> {
        check_scores(scores)?;
        check_len("predictions", predictions.len(), scores.len())?;
        let n = scores.len();
        let nf = n as f64;
        let cal = self.calibrator.calibrate(scores);
        let (mut tp, mut fp, mut fn_, mut tn) = (0.0, 0.0, 0.0, 0.0);
        for (&p, &pr) in cal.iter().zip(predictions) {
            if pr {
                tp += p;
                fp += 1.0 - p;
            } else {
                fn_ += p;
                tn += 1.0 - p;
            }
        }
        let s = &self.sampling;
        let se = |std: f64, frac: f64| std / (nf * frac).sqrt();
        let unit = |v: f64, e: f64| MetricEstimate::new(v, e, 0.0, 1.0);
        let count = |v: f64, std: f64| MetricEstimate::new(v, std * nf.sqrt(), 0.0, f64::INFINITY);
        Ok(CbpeEstimate {
            n,
            roc_auc: unit(
                estimate_roc_auc(&cal, scores),
                se(s.roc_auc_std, s.roc_auc_fraction),
            ),
            f1: unit(
                ratio_or_zero(tp, tp + 0.5 * (fp + fn_)),
                se(s.f1_std, s.f1_fraction),
            ),
            precision: unit(
                ratio_or_zero(tp, tp + fp),
                se(s.precision_std, s.precision_fraction),
            ),
            recall: unit(
                ratio_or_zero(tp, tp + fn_),
                se(s.recall_std, s.recall_fraction),
            ),
            specificity: unit(
                ratio_or_zero(tn, tn + fp),
                se(s.specificity_std, s.specificity_fraction),
            ),
            accuracy: unit((tp + tn) / nf, s.accuracy_std / nf.sqrt()),
            true_positive: count(tp, s.true_positive_std),
            false_positive: count(fp, s.false_positive_std),
            false_negative: count(fn_, s.false_negative_std),
            true_negative: count(tn, s.true_negative_std),
        })
    }

    /// Realized performance of a labelled chunk (raw scores, predictions
    /// `score >= self.threshold`), e.g. on the reference chunks for alert
    /// thresholds (NannyML derives them from the realized values on the
    /// reference chunks; see [`threshold_band`](super::threshold_band)).
    pub fn realized(&self, scores: &[f64], labels: &[bool]) -> Result<RealizedPerformance> {
        check_scores(scores)?;
        let preds = predictions(scores, self.threshold);
        realized_performance(scores, &preds, labels)
    }
}

/// Realized binary-classification performance of a labelled chunk with the
/// rules of [`RealizedPerformance`] (NannyML's `_realized_performance`).
/// ROC AUC uses the raw `scores` (ties count half).
pub fn realized_performance(
    scores: &[f64],
    predictions: &[bool],
    labels: &[bool],
) -> Result<RealizedPerformance> {
    check_scores(scores)?;
    check_len("predictions", predictions.len(), scores.len())?;
    check_len("labels", labels.len(), scores.len())?;
    let n = scores.len();
    let (mut tp, mut fp, mut fn_, mut tn) = (0.0, 0.0, 0.0, 0.0);
    for (&p, &y) in predictions.iter().zip(labels) {
        match (p, y) {
            (true, true) => tp += 1.0,
            (true, false) => fp += 1.0,
            (false, true) => fn_ += 1.0,
            (false, false) => tn += 1.0,
        }
    }
    let one_label = tp + fn_ == 0.0 || fp + tn == 0.0;
    let one_pred = tp + fp == 0.0 || fn_ + tn == 0.0;
    let undefined = one_label || one_pred;
    let nan_if = |c: bool, v: f64| if c { f64::NAN } else { v };
    Ok(RealizedPerformance {
        n,
        roc_auc: if one_label {
            f64::NAN
        } else {
            crate::classification::auc(scores, labels)?
        },
        f1: nan_if(undefined, ratio_or_zero(2.0 * tp, 2.0 * tp + fp + fn_)),
        precision: nan_if(undefined, ratio_or_zero(tp, tp + fp)),
        recall: nan_if(undefined, ratio_or_zero(tp, tp + fn_)),
        specificity: nan_if(tn + fp == 0.0, ratio_or_zero(tn, tn + fp)),
        accuracy: (tp + tn) / n as f64,
        true_positive: tp,
        false_positive: fp,
        false_negative: fn_,
        true_negative: tn,
    })
}
