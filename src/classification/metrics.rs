use super::curves::roc_curve;
use crate::error::{Result, StatError};
use crate::utils::finite::ensure_finite;

/// Area under the ROC curve by the trapezoidal rule (scikit-learn
/// `roc_auc_score`, pROC `auc`).
///
/// Ties are counted as one half, so the value equals the Mann-Whitney
/// probability `P(score_pos > score_neg) + P(score_pos = score_neg) / 2`.
///
/// # Errors
/// Empty input, unequal lengths, NaN scores, or only one class present
/// (as sklearn).
pub fn auc(scores: &[f64], labels: &[bool]) -> Result<f64> {
    let roc = roc_curve(scores, labels, false)?;
    if roc.fpr[0].is_nan() || roc.tpr[0].is_nan() {
        return Err(StatError::InvalidParameter(
            "AUC is undefined when only one class is present".into(),
        ));
    }
    Ok(roc
        .fpr
        .windows(2)
        .zip(roc.tpr.windows(2))
        .map(|(f, t)| (f[1] - f[0]) * (t[1] + t[0]) / 2.0)
        .sum())
}

fn check_probs(probs: &[f64], labels: &[bool]) -> Result<()> {
    if probs.is_empty() {
        return Err(StatError::EmptyData);
    }
    if probs.len() != labels.len() {
        return Err(StatError::InvalidParameter(format!(
            "probs and labels must have equal length ({} vs {})",
            probs.len(),
            labels.len()
        )));
    }
    ensure_finite("probs", probs)?;
    if let Some(p) = probs.iter().find(|p| !(0.0..=1.0).contains(*p)) {
        return Err(StatError::InvalidParameter(format!(
            "probabilities must be in [0, 1], got {}",
            p
        )));
    }
    Ok(())
}

/// Brier score: mean squared difference between the predicted probability
/// and the outcome (scikit-learn `brier_score_loss`).
///
/// # Errors
/// Empty input, unequal lengths, or probabilities outside `[0, 1]`.
pub fn brier_score(probs: &[f64], labels: &[bool]) -> Result<f64> {
    check_probs(probs, labels)?;
    let s: f64 = probs
        .iter()
        .zip(labels)
        .map(|(&p, &y)| {
            let d = p - if y { 1.0 } else { 0.0 };
            d * d
        })
        .sum();
    Ok(s / probs.len() as f64)
}

/// Mean binary cross-entropy (scikit-learn `log_loss`).
///
/// Probabilities are clipped to `[eps, 1 - eps]` with `eps = f64::EPSILON`
/// (sklearn's `eps = "auto"` for float64), so 0 and 1 give a finite loss.
///
/// # Errors
/// Empty input, unequal lengths, or probabilities outside `[0, 1]`.
pub fn log_loss(probs: &[f64], labels: &[bool]) -> Result<f64> {
    check_probs(probs, labels)?;
    let eps = f64::EPSILON;
    let s: f64 = probs
        .iter()
        .zip(labels)
        .map(|(&p, &y)| {
            let p = p.clamp(eps, 1.0 - eps);
            if y {
                -p.ln()
            } else {
                -(1.0 - p).ln()
            }
        })
        .sum();
    Ok(s / probs.len() as f64)
}
