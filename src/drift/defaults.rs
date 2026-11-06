//! evidently's default parameters for the drift module, so callers do not
//! have to copy them. The crate's own functions use these constants.
//!
//! The default *test* of a column depends on the data (reference size and
//! number of distinct values): [`default_drift_test_numeric`] /
//! [`default_drift_test_categorical`] implement the rule with the constants
//! below ([`SMALL_REFERENCE_ROWS`], [`NUMERIC_SMALL_MANY_VALUES`], ...).
//!
//! [`default_drift_test_numeric`]: super::default_drift_test_numeric
//! [`default_drift_test_categorical`]: super::default_drift_test_categorical

use super::{DriftTest, EmbeddingThreshold};

/// Default threshold of the p-value tests (KS, Epps-Singleton,
/// Mann-Whitney, t-test, chi-square, Z): drift when `p < 0.05`.
pub const P_VALUE_THRESHOLD: f64 = 0.05;
/// Default threshold of the distance scores (PSI, KL, JS, Wasserstein,
/// Hellinger, energy, TVD, ...): drift when the score is at least 0.1.
pub const DISTANCE_THRESHOLD: f64 = 0.1;

/// Reference size up to which evidently picks the small-sample tests.
pub const SMALL_REFERENCE_ROWS: usize = 1000;
/// Numeric columns with more distinct (pooled) values than this count as
/// continuous.
pub const NUMERIC_MANY_VALUES: usize = 5;
/// Numeric columns with more distinct values than this (and at most
/// [`NUMERIC_MANY_VALUES`]) get the chi-square test in small samples.
pub const NUMERIC_FEW_VALUES: usize = 2;
/// Default numeric test, small reference, more than [`NUMERIC_MANY_VALUES`]
/// distinct values.
pub const NUMERIC_SMALL_MANY_VALUES: DriftTest = DriftTest::Ks;
/// Default numeric test, small reference, 3 to [`NUMERIC_MANY_VALUES`]
/// distinct values.
pub const NUMERIC_SMALL_FEW_VALUES: DriftTest = DriftTest::ChiSquare;
/// Default numeric test, small reference, at most [`NUMERIC_FEW_VALUES`]
/// distinct values.
pub const NUMERIC_SMALL_BINARY: DriftTest = DriftTest::Z;
/// Default numeric test, large reference, more than
/// [`NUMERIC_MANY_VALUES`] distinct values.
pub const NUMERIC_LARGE_MANY_VALUES: DriftTest = DriftTest::Wasserstein;
/// Default numeric test, large reference, few distinct values.
pub const NUMERIC_LARGE_FEW_VALUES: DriftTest = DriftTest::JensenShannon;
/// Default categorical test, small reference, more than two categories.
pub const CATEGORICAL_SMALL_MANY: DriftTest = DriftTest::ChiSquare;
/// Default categorical test, small reference, at most two categories.
pub const CATEGORICAL_SMALL_BINARY: DriftTest = DriftTest::Z;
/// Default categorical test, large reference.
pub const CATEGORICAL_LARGE: DriftTest = DriftTest::JensenShannon;

/// Default number of permutations of the `TVD` / `empirical_mmd` p-values
/// ([`DriftOptions`](super::DriftOptions)).
pub const N_PERMUTATIONS: usize = 1000;
/// Default share of drifted columns for dataset drift
/// ([`dataset_drift`](super::dataset_drift)).
pub const DATASET_DRIFT_SHARE: f64 = 0.5;

/// Fixed threshold of [`embedding_distance_drift`](super::embedding_distance_drift).
pub const EMBEDDING_DISTANCE_THRESHOLD: f64 = 0.2;
/// Fixed threshold of [`embedding_mmd_drift`](super::embedding_mmd_drift).
pub const EMBEDDING_MMD_THRESHOLD: f64 = 0.015;
/// Bootstrap quantile of the distance method.
pub const EMBEDDING_DISTANCE_QUANTILE: f64 = 0.95;
/// Bootstrap quantile of the MMD method.
pub const EMBEDDING_MMD_QUANTILE: f64 = 0.05;
/// Number of bootstrap resamples of the embedding thresholds.
pub const EMBEDDING_BOOTSTRAP_RESAMPLES: usize = 100;
/// Component test of [`embedding_ratio_drift`](super::embedding_ratio_drift).
pub const EMBEDDING_RATIO_COMPONENT_TEST: DriftTest = DriftTest::Wasserstein;
/// Component threshold of the ratio method.
pub const EMBEDDING_RATIO_COMPONENT_THRESHOLD: f64 = 0.1;
/// Share of drifted components above which the ratio method flags drift.
pub const EMBEDDING_RATIO_THRESHOLD: f64 = 0.2;
/// Rows per sample used for the MMD median distance (larger samples are
/// subsampled).
pub const EMBEDDING_MEDIAN_MAX_ROWS: usize = 1000;

impl EmbeddingThreshold {
    /// evidently's fixed threshold of the distance method, `Fixed(0.2)`.
    pub const EVIDENTLY_DISTANCE: EmbeddingThreshold =
        EmbeddingThreshold::Fixed(EMBEDDING_DISTANCE_THRESHOLD);
    /// evidently's fixed threshold of the MMD method, `Fixed(0.015)`.
    pub const EVIDENTLY_MMD: EmbeddingThreshold =
        EmbeddingThreshold::Fixed(EMBEDDING_MMD_THRESHOLD);

    /// evidently's bootstrap threshold of the distance method: 100
    /// resamples, quantile 0.95.
    pub fn evidently_distance_bootstrap(seed: Option<u64>) -> Self {
        EmbeddingThreshold::Bootstrap {
            n_bootstrap: EMBEDDING_BOOTSTRAP_RESAMPLES,
            quantile: EMBEDDING_DISTANCE_QUANTILE,
            seed,
        }
    }

    /// evidently's bootstrap threshold of the MMD method: 100 resamples,
    /// quantile 0.05.
    pub fn evidently_mmd_bootstrap(seed: Option<u64>) -> Self {
        EmbeddingThreshold::Bootstrap {
            n_bootstrap: EMBEDDING_BOOTSTRAP_RESAMPLES,
            quantile: EMBEDDING_MMD_QUANTILE,
            seed,
        }
    }
}
