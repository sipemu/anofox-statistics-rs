//! Multivariate drift by PCA reconstruction error, as NannyML's
//! `DataReconstructionDriftCalculator` on numeric columns.
//!
//! The reference rows are standardised (column means, population standard
//! deviations as scikit-learn's `StandardScaler`; a constant column keeps
//! scale 1) and a PCA keeps the components explaining a share of the
//! variance (NannyML's default 0.65). Each row's reconstruction error is the
//! euclidean norm of its standardised residual outside those components; a
//! chunk's value is the mean error, its sampling error `sd / sqrt(n)`, and
//! the alert band comes from the chunk values of the reference period
//! (mean ± 3 sd, [`reconstruction_thresholds`]).
//!
//! NannyML count-encodes categorical columns; here all columns are numeric
//! (encode categories before). Missing values (NaN) are imputed by the
//! reference column means, as NannyML's `SimpleImputer(strategy = "mean")`.
//!
//! Validated against NannyML 0.13.1 on its synthetic car-loan dataset
//! (`R/generate_monitoring_refs.py`) and against `prcomp`
//! (`R/generate_monitoring_refs.R`).

use super::cbpe::SAMPLING_ERROR_RANGE;
use super::thresholds::{threshold_band, ThresholdBand, ThresholdMethod, ValueLimits};
use crate::error::{Result, StatError};
use crate::multivariate::{pca, pca_with_options, Pca, PcaOptions};

/// How many principal components the reconstruction keeps.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ComponentSelection {
    /// The fewest components whose explained-variance share exceeds the
    /// value, in `(0, 1)` (scikit-learn's `PCA(n_components = 0.65)`, see
    /// [`Pca::n_components_for_variance`]).
    VarianceShare(f64),
    /// A fixed number of components.
    Fixed(usize),
}

/// Options of [`fit_reconstruction`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReconstructionOptions {
    /// Default `VarianceShare(0.65)` (NannyML).
    pub n_components: ComponentSelection,
    /// Standardise the columns (population sd) before the PCA. Default
    /// `true` (NannyML); `false` only centres them.
    pub scale: bool,
    /// Replace NaN by the reference column mean (NannyML). Default `true`;
    /// with `false`, NaN is an error.
    pub impute_nan: bool,
}

impl Default for ReconstructionOptions {
    fn default() -> Self {
        Self {
            n_components: ComponentSelection::VarianceShare(0.65),
            scale: true,
            impute_nan: true,
        }
    }
}

/// A reconstruction-error model fitted by [`fit_reconstruction`]. The
/// fields are public, so a model can be stored and rebuilt from its parts.
#[derive(Debug, Clone, PartialEq)]
pub struct ReconstructionModel {
    /// The PCA; [`Pca::center`] are the reference column means (also the
    /// imputation values) and [`Pca::scale`] the population standard
    /// deviations (when scaled).
    pub pca: Pca,
    /// Components used for the reconstruction.
    pub n_components: usize,
    /// Whether NaN inputs are imputed by [`Pca::center`].
    pub impute_nan: bool,
}

/// Reconstruction error of one chunk.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReconstructionChunk {
    /// Rows in the chunk.
    pub n: usize,
    /// Mean reconstruction error (NannyML's `reconstruction_error`).
    pub value: f64,
    /// Standard deviation of the row errors (divisor `n - 1`; NaN for one row).
    pub std: f64,
    /// `std / sqrt(n)`.
    pub sampling_error: f64,
    /// `max(value - 3 * sampling_error, 0)`.
    pub lower_confidence: f64,
    /// `value + 3 * sampling_error`.
    pub upper_confidence: f64,
}

fn check_rows(data: &[Vec<f64>], d: Option<usize>) -> Result<usize> {
    if data.is_empty() {
        return Err(StatError::EmptyData);
    }
    let d = d.unwrap_or(data[0].len());
    if d == 0 {
        return Err(StatError::InvalidParameter(
            "rows must have at least one value".into(),
        ));
    }
    for (r, row) in data.iter().enumerate() {
        if row.len() != d {
            return Err(StatError::InvalidParameter(format!(
                "row {} has {} values, expected {}",
                r,
                row.len(),
                d
            )));
        }
        if let Some(v) = row.iter().find(|v| v.is_infinite()) {
            return Err(StatError::InvalidParameter(format!(
                "infinite value in row {}: {}",
                r, v
            )));
        }
    }
    Ok(d)
}

fn impute(data: &[Vec<f64>], values: &[f64], allow: bool) -> Result<Vec<Vec<f64>>> {
    data.iter()
        .enumerate()
        .map(|(r, row)| {
            row.iter()
                .zip(values)
                .map(|(&v, &m)| {
                    if v.is_nan() {
                        if allow {
                            Ok(m)
                        } else {
                            Err(StatError::InvalidParameter(format!("NaN in row {}", r)))
                        }
                    } else {
                        Ok(v)
                    }
                })
                .collect()
        })
        .collect()
}

/// Fits the reconstruction-error model on the reference rows (observations
/// x numeric variables).
///
/// Errors on fewer than two rows, rows of unequal or zero length, infinite
/// values, a column that is entirely NaN, NaN without `impute_nan`, and an
/// invalid component selection (a share outside `(0, 1)`, or a fixed count
/// of 0 or above `min(n - 1, d)`).
///
/// Python equivalent: `nannyml.DataReconstructionDriftCalculator(column_names,
/// n_components = 0.65).fit(reference)` (numeric columns).
pub fn fit_reconstruction(
    reference: &[Vec<f64>],
    options: &ReconstructionOptions,
) -> Result<ReconstructionModel> {
    let d = check_rows(reference, None)?;
    if reference.len() < 2 {
        return Err(StatError::InsufficientData {
            needed: 2,
            got: reference.len(),
        });
    }
    let means: Vec<f64> = (0..d)
        .map(|i| {
            let v: Vec<f64> = reference
                .iter()
                .map(|r| r[i])
                .filter(|x| !x.is_nan())
                .collect();
            if v.is_empty() {
                Err(StatError::InvalidParameter(format!(
                    "column {} has no non-missing reference values",
                    i
                )))
            } else {
                Ok(v.iter().sum::<f64>() / v.len() as f64)
            }
        })
        .collect::<Result<_>>()?;
    let x = impute(reference, &means, options.impute_nan)?;
    let n = x.len() as f64;
    let mut p = if options.scale {
        let sd: Vec<f64> = (0..d)
            .map(|i| {
                let s = (x.iter().map(|r| (r[i] - means[i]).powi(2)).sum::<f64>() / n).sqrt();
                // scikit-learn's _handle_zeros_in_scale
                if s < 10.0 * f64::EPSILON * means[i].abs().max(1.0) {
                    1.0
                } else {
                    s
                }
            })
            .collect();
        let z: Vec<Vec<f64>> = x
            .iter()
            .map(|r| (0..d).map(|i| (r[i] - means[i]) / sd[i]).collect())
            .collect();
        let mut p = pca(&z)?;
        p.center = means;
        p.scale = Some(sd);
        p
    } else {
        pca_with_options(&x, &PcaOptions { scale: false })?
    };
    let k = match options.n_components {
        ComponentSelection::VarianceShare(s) => p.n_components_for_variance(s)?,
        ComponentSelection::Fixed(k) => {
            if k == 0 || k > p.n_components() {
                return Err(StatError::InvalidParameter(format!(
                    "n_components must be in 1..={}, got {}",
                    p.n_components(),
                    k
                )));
            }
            k
        }
    };
    // Keep the components used, as scikit-learn's PCA(n_components) does.
    p.sdev.truncate(k);
    p.explained_variance_ratio.truncate(k);
    for row in p.rotation.iter_mut() {
        row.truncate(k);
    }
    Ok(ReconstructionModel {
        pca: p,
        n_components: k,
        impute_nan: options.impute_nan,
    })
}

impl ReconstructionModel {
    /// Rebuilds a model from its parts. Errors when `n_components` is 0 or
    /// exceeds the components of `pca`.
    pub fn from_parts(pca: Pca, n_components: usize, impute_nan: bool) -> Result<Self> {
        if n_components == 0 || n_components > pca.n_components() {
            return Err(StatError::InvalidParameter(format!(
                "n_components must be in 1..={}, got {}",
                pca.n_components(),
                n_components
            )));
        }
        Ok(Self {
            pca,
            n_components,
            impute_nan,
        })
    }

    /// Reconstruction error of each row (NaN imputed by the reference means
    /// when [`impute_nan`](Self::impute_nan)).
    ///
    /// Errors on an empty input, rows of the wrong length, infinite values
    /// and NaN without imputation.
    pub fn row_errors(&self, data: &[Vec<f64>]) -> Result<Vec<f64>> {
        check_rows(data, Some(self.pca.center.len()))?;
        let x = impute(data, &self.pca.center, self.impute_nan)?;
        self.pca.reconstruction_error(&x, self.n_components)
    }

    /// Mean reconstruction error of a chunk with its sampling error and
    /// NannyML's confidence band.
    ///
    /// Errors as [`row_errors`](Self::row_errors).
    pub fn chunk_error(&self, data: &[Vec<f64>]) -> Result<ReconstructionChunk> {
        let e = self.row_errors(data)?;
        let n = e.len();
        let value = e.iter().sum::<f64>() / n as f64;
        let std = if n > 1 {
            (e.iter().map(|v| (v - value).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt()
        } else {
            f64::NAN
        };
        let sampling_error = std / (n as f64).sqrt();
        Ok(ReconstructionChunk {
            n,
            value,
            std,
            sampling_error,
            lower_confidence: (value - SAMPLING_ERROR_RANGE * sampling_error).max(0.0),
            upper_confidence: value + SAMPLING_ERROR_RANGE * sampling_error,
        })
    }
}

/// Alert band of the reconstruction error from its values on the reference
/// chunks: `method` (NannyML's default [`ThresholdMethod::NANNYML_DEFAULT`],
/// mean ± 3 population sd), with a lower threshold at or below 0 dropped
/// (`None`), as NannyML. A chunk alerts when its value is outside the band
/// ([`ThresholdBand::alert`]).
pub fn reconstruction_thresholds(
    reference_chunk_values: &[f64],
    method: &ThresholdMethod,
) -> Result<ThresholdBand> {
    threshold_band(
        reference_chunk_values,
        method,
        &ValueLimits {
            lower: Some(0.0),
            upper: None,
            override_with_none: true,
        },
    )
}
