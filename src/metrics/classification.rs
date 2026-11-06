//! Classification metrics from true and predicted labels, as scikit-learn's
//! `sklearn.metrics` (`accuracy_score`, `precision_recall_fscore_support`,
//! `balanced_accuracy_score`, `matthews_corrcoef`, `cohen_kappa_score`,
//! `confusion_matrix`).

use crate::error::{Result, StatError};

/// How the per-class precision, recall, F1 and specificity are combined
/// (scikit-learn's `average`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Average<T> {
    /// The metrics of the given positive class only (`average = "binary"`,
    /// `pos_label`). Errors when more than two labels occur, or when two occur
    /// and neither is the positive class.
    Binary(T),
    /// Unweighted mean over the labels (`"macro"`).
    Macro,
    /// From the confusion counts summed over the labels (`"micro"`); for
    /// precision, recall and F1 this equals the accuracy.
    Micro,
    /// Mean over the labels weighted by their support (number of true
    /// instances, `"weighted"`).
    Weighted,
}

/// Options of [`classification_metrics`].
#[derive(Debug, Clone, PartialEq)]
pub struct ClassificationOptions<T> {
    /// Default [`Average::Macro`].
    pub average: Average<T>,
    /// Value of a ratio whose denominator is zero (precision without
    /// predicted positives, recall without true positives, F1 when
    /// `2 tp + fp + fn = 0`, specificity without true negatives), before
    /// averaging: scikit-learn's `zero_division` (its default `"warn"`
    /// returns 0). Default 0.0; 1.0 and NaN are the other scikit-learn
    /// choices.
    pub zero_division: f64,
}

impl<T> Default for ClassificationOptions<T> {
    fn default() -> Self {
        Self {
            average: Average::Macro,
            zero_division: 0.0,
        }
    }
}

/// Confusion matrix: `counts[i][j]` is the number of rows with true label
/// `labels[i]` and predicted label `labels[j]` (scikit-learn's layout).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfusionMatrix<T> {
    /// The sorted distinct labels of `y_true` and `y_pred` together.
    pub labels: Vec<T>,
    pub counts: Vec<Vec<u64>>,
}

/// Classification metrics returned by [`classification_metrics`].
#[derive(Debug, Clone, PartialEq)]
pub struct ClassificationMetrics<T> {
    /// Number of rows.
    pub n: usize,
    /// Share of correct predictions.
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    /// `tn / (tn + fp)` per class, averaged as the other per-class metrics
    /// (no scikit-learn counterpart; R `yardstick::spec`).
    pub specificity: f64,
    /// Mean recall over the labels that occur in `y_true`
    /// (`balanced_accuracy_score`, independent of `average`).
    pub balanced_accuracy: f64,
    /// Matthews correlation coefficient (multiclass `R_K` statistic); 0 when
    /// undefined (`matthews_corrcoef`).
    pub mcc: f64,
    /// Unweighted Cohen's kappa between the true and predicted labels; NaN
    /// when undefined (`cohen_kappa_score`).
    pub cohen_kappa: f64,
    pub confusion: ConfusionMatrix<T>,
}

/// Confusion matrix of `y_true` against `y_pred` over the sorted distinct
/// labels of both. Errors on empty or mismatched inputs.
///
/// Python equivalent: `sklearn.metrics.confusion_matrix(y_true, y_pred)`.
pub fn confusion_matrix<T: Ord + Clone>(y_true: &[T], y_pred: &[T]) -> Result<ConfusionMatrix<T>> {
    if y_true.len() != y_pred.len() {
        return Err(StatError::InvalidParameter(format!(
            "y_true ({}) and y_pred ({}) must have the same length",
            y_true.len(),
            y_pred.len()
        )));
    }
    if y_true.is_empty() {
        return Err(StatError::EmptyData);
    }
    let mut labels: Vec<T> = y_true.iter().chain(y_pred).cloned().collect();
    labels.sort();
    labels.dedup();
    let k = labels.len();
    let pos = |v: &T| labels.binary_search(v).expect("label present");
    let mut counts = vec![vec![0u64; k]; k];
    for (t, p) in y_true.iter().zip(y_pred) {
        counts[pos(t)][pos(p)] += 1;
    }
    Ok(ConfusionMatrix { labels, counts })
}

fn ratio(num: f64, den: f64, zero_division: f64) -> f64 {
    if den > 0.0 {
        num / den
    } else {
        zero_division
    }
}

/// Classification metrics of predicted against true labels (any ordered
/// label type: integers, strings, `bool`).
///
/// Precision, recall, F1 and specificity are per class (one-vs-rest
/// confusion counts) and combined by `options.average`; a zero denominator
/// gives `options.zero_division` for that class. Micro specificity is
/// `sum tn / sum (tn + fp)`; weighted averages weight by the true count of
/// each label. Accuracy, balanced accuracy, MCC, kappa and the confusion
/// matrix do not depend on the options.
///
/// Errors on empty or mismatched inputs, and for [`Average::Binary`] with
/// more than two labels or two labels without the positive class (as
/// scikit-learn).
///
/// Python equivalent: `accuracy_score`, `precision_score` / `recall_score` /
/// `f1_score(average = ..., zero_division = ...)`, `balanced_accuracy_score`,
/// `matthews_corrcoef`, `cohen_kappa_score`, `confusion_matrix`. R:
/// `yardstick::accuracy`, `precision`, `recall`, `f_meas`, `spec`, `mcc`,
/// `kap` (`estimator = "binary" / "macro" / "micro" / "macro_weighted"`).
///
/// # Example
/// ```
/// use anofox_statistics::metrics::{classification_metrics, Average, ClassificationOptions};
/// let y_true = [1, 1, 0, 0, 1];
/// let y_pred = [1, 0, 0, 1, 1];
/// let opts = ClassificationOptions { average: Average::Binary(1), zero_division: 0.0 };
/// let m = classification_metrics(&y_true, &y_pred, &opts).unwrap();
/// assert!((m.precision - 2.0 / 3.0).abs() < 1e-12);
/// assert!((m.specificity - 0.5).abs() < 1e-12);
/// assert_eq!(m.confusion.counts, vec![vec![1, 1], vec![1, 2]]);
/// ```
pub fn classification_metrics<T: Ord + Clone>(
    y_true: &[T],
    y_pred: &[T],
    options: &ClassificationOptions<T>,
) -> Result<ClassificationMetrics<T>> {
    let cm = confusion_matrix(y_true, y_pred)?;
    let k = cm.labels.len();
    let n = y_true.len();
    let nf = n as f64;
    let c: Vec<Vec<f64>> = cm
        .counts
        .iter()
        .map(|r| r.iter().map(|&v| v as f64).collect())
        .collect();
    let row: Vec<f64> = c.iter().map(|r| r.iter().sum()).collect();
    let col: Vec<f64> = (0..k).map(|j| c.iter().map(|r| r[j]).sum()).collect();
    let tp: Vec<f64> = (0..k).map(|i| c[i][i]).collect();
    let fp: Vec<f64> = (0..k).map(|i| col[i] - tp[i]).collect();
    let fneg: Vec<f64> = (0..k).map(|i| row[i] - tp[i]).collect();
    let tn: Vec<f64> = (0..k).map(|i| nf - tp[i] - fp[i] - fneg[i]).collect();
    let zd = options.zero_division;
    let per_class = |i: usize| -> [f64; 4] {
        [
            ratio(tp[i], tp[i] + fp[i], zd),
            ratio(tp[i], tp[i] + fneg[i], zd),
            ratio(2.0 * tp[i], 2.0 * tp[i] + fp[i] + fneg[i], zd),
            ratio(tn[i], tn[i] + fp[i], zd),
        ]
    };
    let [precision, recall, f1, specificity] = match &options.average {
        Average::Binary(positive) => {
            if k > 2 {
                return Err(StatError::InvalidParameter(format!(
                    "binary averaging needs at most two labels, got {}",
                    k
                )));
            }
            match cm.labels.binary_search(positive) {
                Ok(i) => per_class(i),
                Err(_) if k == 2 => {
                    return Err(StatError::InvalidParameter(
                        "the positive label does not occur in the data".into(),
                    ))
                }
                // The positive class never occurs: tp = fp = fn = 0.
                Err(_) => [zd, zd, zd, ratio(nf, nf, zd)],
            }
        }
        Average::Macro => {
            let mut acc = [0.0; 4];
            for i in 0..k {
                let m = per_class(i);
                for (a, v) in acc.iter_mut().zip(m) {
                    *a += v / k as f64;
                }
            }
            acc
        }
        Average::Weighted => {
            let mut acc = [0.0; 4];
            for (i, &support) in row.iter().enumerate() {
                if support > 0.0 {
                    for (a, v) in acc.iter_mut().zip(per_class(i)) {
                        *a += v * support / nf;
                    }
                }
            }
            acc
        }
        Average::Micro => {
            let (stp, sfp, sfn, stn) = (
                tp.iter().sum::<f64>(),
                fp.iter().sum::<f64>(),
                fneg.iter().sum::<f64>(),
                tn.iter().sum::<f64>(),
            );
            [
                ratio(stp, stp + sfp, zd),
                ratio(stp, stp + sfn, zd),
                ratio(2.0 * stp, 2.0 * stp + sfp + sfn, zd),
                ratio(stn, stn + sfp, zd),
            ]
        }
    };
    let correct: f64 = tp.iter().sum();
    let present: Vec<usize> = (0..k).filter(|&i| row[i] > 0.0).collect();
    let balanced_accuracy =
        present.iter().map(|&i| tp[i] / row[i]).sum::<f64>() / present.len() as f64;
    // Matthews correlation (sklearn's covariance form).
    let tdotp: f64 = (0..k).map(|i| row[i] * col[i]).sum();
    let cov_tp = correct * nf - tdotp;
    let cov_pp = nf * nf - col.iter().map(|v| v * v).sum::<f64>();
    let cov_tt = nf * nf - row.iter().map(|v| v * v).sum::<f64>();
    let mcc = if cov_pp * cov_tt == 0.0 {
        0.0
    } else {
        cov_tp / (cov_tt.sqrt() * cov_pp.sqrt())
    };
    // Cohen's kappa: 1 - observed / expected disagreement.
    let observed = nf - correct;
    let expected = nf - tdotp / nf;
    let cohen_kappa = if expected == 0.0 {
        f64::NAN
    } else {
        1.0 - observed / expected
    };
    Ok(ClassificationMetrics {
        n,
        accuracy: correct / nf,
        precision,
        recall,
        f1,
        specificity,
        balanced_accuracy,
        mcc,
        cohen_kappa,
        confusion: cm,
    })
}
