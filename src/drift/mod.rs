//! Data-drift measures between a reference and a current sample, as in
//! evidently's `legacy/calculations/stattests`.
//!
//! The divergences ([`psi`], [`kl_divergence`]) bin both samples on common bins
//! ([`binned_proportions`], [`Binning`]) and compare the bin proportions `p`
//! (reference) and `q` (current). Each has a `*_from_proportions` form for
//! pre-binned data and a `*_categorical` form for labels of any ordered type.

pub mod binning;
pub mod kl;
pub mod psi;

pub use binning::{binned_proportions, category_proportions, zero_fill, Binning};
pub use kl::{kl_divergence, kl_divergence_categorical, kl_divergence_from_proportions};
pub use psi::{psi, psi_categorical, psi_from_proportions};

use crate::error::{Result, StatError};

/// Natural log of a logarithm base (`None` = `e`). Errors unless the base is
/// finite, positive and not 1.
pub(crate) fn ln_base(base: Option<f64>) -> Result<f64> {
    match base {
        None => Ok(1.0),
        Some(b) if b.is_finite() && b > 0.0 && b != 1.0 => Ok(b.ln()),
        Some(b) => Err(StatError::InvalidParameter(format!(
            "logarithm base must be finite, positive and not 1, got {}",
            b
        ))),
    }
}

/// Result of a binned divergence between a reference and a current sample.
#[derive(Debug, Clone, PartialEq)]
pub struct DivergenceResult {
    /// Name of the measure, e.g. `"Population Stability Index"`.
    pub method: String,
    /// Value of the measure.
    pub statistic: f64,
    /// Number of bins (categories) the proportions were computed on.
    pub n_bins: usize,
    /// Size of the reference sample.
    pub n_reference: usize,
    /// Size of the current sample.
    pub n_current: usize,
}

impl DivergenceResult {
    pub(crate) fn new(
        method: &str,
        statistic: f64,
        n_bins: usize,
        n_reference: usize,
        n_current: usize,
    ) -> Self {
        DivergenceResult {
            method: method.to_string(),
            statistic,
            n_bins,
            n_reference,
            n_current,
        }
    }
}
