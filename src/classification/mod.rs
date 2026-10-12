//! Evaluation of binary classifiers: ROC and precision-recall curves,
//! calibration curves, ROC AUC, Brier score and log loss.
//!
//! The curve functions follow scikit-learn (`roc_curve`,
//! `precision_recall_curve`, `calibration_curve(strategy = "uniform")`), so
//! tied scores collapse into one curve point, as in sklearn and pROC.
//! Labels are `bool` (`true` = positive class).

mod curves;
mod metrics;

pub use curves::{
    calibration_curve, calibration_curve_with_conf_level, pr_curve, roc_curve, CalibrationBin,
    CalibrationCurve, PrCurve, RocCurve,
};
pub use metrics::{auc, brier_score, log_loss};
