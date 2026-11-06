//! Performance metrics of labelled predictions: classification
//! ([`classification_metrics`], binary and multiclass with binary / macro /
//! micro / weighted averaging), regression ([`regression_metrics`]) and
//! ranking ([`ranking_metrics`], [`ndcg_at_k`]).
//!
//! Definitions follow scikit-learn (validated against `sklearn.metrics`
//! and R `yardstick`); the ranking metrics other than NDCG follow evidently.
//! Probability-based metrics of binary classifiers (ROC AUC, Brier score, log
//! loss, curves) are in [`classification`](crate::classification); label-free
//! performance estimation is in [`monitoring`](crate::monitoring).

mod classification;
mod ranking;
mod regression;

pub use classification::{
    classification_metrics, confusion_matrix, Average, ClassificationMetrics,
    ClassificationOptions, ConfusionMatrix,
};
pub use ranking::{ndcg_at_k, ranking_metrics, RankingMetrics, RankingOptions};
pub use regression::{regression_metrics, RegressionMetrics};
