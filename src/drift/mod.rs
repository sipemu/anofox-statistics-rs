//! Data-drift measures between a reference and a current sample, as in
//! evidently's `legacy/calculations/stattests`.
//!
//! The divergences ([`psi()`], [`kl_divergence()`], [`jensen_shannon()`],
//! [`hellinger()`]) bin both samples on common bins
//! ([`binned_proportions`], [`Binning`]) and compare the bin proportions `p`
//! (reference) and `q` (current). Each has a `*_from_proportions` form for
//! pre-binned data and a `*_categorical` form for labels of any ordered type.
//! [`tvd_test`] compares category distributions with a permutation p-value.
//!
//! The rank-based drift tests are in [`nonparametric`](crate::nonparametric)
//! ([`ks_test_2sample`](crate::ks_test_2sample),
//! [`ad_test_ksamp`](crate::ad_test_ksamp),
//! [`cvm_test_2sample`](crate::cvm_test_2sample),
//! [`epps_singleton_test`](crate::epps_singleton_test)) and the Wasserstein
//! distance in [`modern`](crate::modern)
//! ([`wasserstein_test_1d`](crate::wasserstein_test_1d)).

pub mod binning;
pub mod column;
pub mod gof;
pub mod hellinger;
pub mod js;
pub mod kl;
pub mod psi;
pub mod tvd;
pub mod z;

pub use binning::{binned_proportions, category_proportions, zero_fill, Binning, ZeroFill};
pub use column::{
    column_drift_categorical, column_drift_numeric, dataset_drift, default_drift_test_categorical,
    default_drift_test_numeric, empirical_mmd_bandwidth, energy_distance_1d, ColumnDrift,
    DatasetDrift, DriftOptions, DriftTest,
};
pub use gof::{chisq_drift, chisq_drift_numeric, g_drift, g_drift_numeric};
pub use hellinger::{hellinger, hellinger_categorical, hellinger_from_proportions};
pub use js::{
    jensen_shannon, jensen_shannon_categorical, jensen_shannon_divergence_from_proportions,
    jensen_shannon_from_proportions,
};
pub use kl::{kl_divergence, kl_divergence_categorical, kl_divergence_from_proportions};
pub use psi::{psi, psi_categorical, psi_from_proportions};
pub use tvd::{tvd_test, tvd_test_numeric, TvdResult};
pub use z::{z_test, z_test_numeric, ZTestResult};

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
    /// The divergence behind a distance, where the measure has one: the
    /// Jensen-Shannon divergence (`statistic^2`) for
    /// [`jensen_shannon()`]; `None` otherwise.
    pub divergence: Option<f64>,
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
            divergence: None,
            n_bins,
            n_reference,
            n_current,
        }
    }
}

/// `f64` with the total order, for using finite numbers as category labels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct OrdF64(pub(crate) f64);

impl OrdF64 {
    /// Category labels of finite values: one per distinct value (`-0.0` and
    /// `0.0` are one category).
    pub(crate) fn codes(values: &[f64]) -> Vec<OrdF64> {
        values
            .iter()
            .map(|&v| OrdF64(if v == 0.0 { 0.0 } else { v }))
            .collect()
    }
}

impl Eq for OrdF64 {}

impl PartialOrd for OrdF64 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdF64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// Counts `(reference, current)` per category over the sorted distinct
/// categories of both samples.
pub(crate) fn category_counts<T: Ord>(reference: &[T], current: &[T]) -> (Vec<usize>, Vec<usize>) {
    let mut counts: std::collections::BTreeMap<&T, (usize, usize)> =
        std::collections::BTreeMap::new();
    for v in reference {
        counts.entry(v).or_default().0 += 1;
    }
    for v in current {
        counts.entry(v).or_default().1 += 1;
    }
    counts.values().copied().unzip()
}
