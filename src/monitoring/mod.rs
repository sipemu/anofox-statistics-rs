//! Model monitoring without (or before) labels, as NannyML: confidence-based
//! performance estimation of a binary classifier ([`cbpe`]), multivariate
//! drift by PCA reconstruction error ([`reconstruction`]) and alert
//! thresholds from reference values ([`thresholds`]).
//!
//! The functions take plain slices (one chunk at a time); chunking, storage
//! and the period bookkeeping are left to the caller. Each fitted model has
//! public fields and a `from_parts` constructor, so it can be persisted and
//! rebuilt.
//!
//! The performance metrics of labelled data (classification, regression and
//! ranking) are in [`metrics`](crate::metrics).

pub mod cbpe;
pub mod reconstruction;
pub mod thresholds;

pub use cbpe::{
    fit_reference, fit_reference_with_predictions, realized_performance, Calibrator,
    CbpeCalibration, CbpeEstimate, CbpeModel, CbpeOptions, CbpeSamplingComponents, MetricEstimate,
    RealizedPerformance,
};
pub use reconstruction::{
    fit_reconstruction, reconstruction_thresholds, ComponentSelection, ReconstructionChunk,
    ReconstructionModel, ReconstructionOptions,
};
pub use thresholds::{threshold_band, ThresholdBand, ThresholdMethod, ValueLimits};
