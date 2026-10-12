//! Non-parametric survival analysis: the Kaplan-Meier estimator with
//! Greenwood standard errors and pointwise confidence intervals, and the
//! k-group log-rank test. Results match R `survival::survfit` and
//! `survival::survdiff`.
//!
//! Events are `bool` (`true` = event observed, `false` = right-censored).

mod kaplan_meier;
mod log_rank;

pub use kaplan_meier::{kaplan_meier, KaplanMeierResult, KmConfType};
pub use log_rank::{log_rank, LogRankResult};
