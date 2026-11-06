//! Alert thresholds derived from reference values (NannyML's
//! `StandardDeviationThreshold` / `ConstantThreshold` and
//! `calculate_threshold_values`).

use crate::error::{Result, StatError};

/// How a [`ThresholdBand`] is derived from the reference values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ThresholdMethod {
    /// `mean - lower * sd` and `mean + upper * sd` of the reference values
    /// (NaN ignored, population sd, as NannyML's `StandardDeviationThreshold`
    /// with `np.nanmean` / `np.nanstd`). `None` disables that side. NannyML's
    /// default is `lower = upper = Some(3.0)`.
    StandardDeviation {
        lower: Option<f64>,
        upper: Option<f64>,
    },
    /// Fixed limits, independent of the reference values (NannyML's
    /// `ConstantThreshold`). `None` disables that side.
    Constant {
        lower: Option<f64>,
        upper: Option<f64>,
    },
    /// Limits relative to the reference mean `m` (NaN ignored):
    /// `m - lower * |m|` and `m + upper * |m|`, e.g. `lower = Some(0.1)` alerts
    /// on a drop of more than 10%. `None` disables that side.
    Relative {
        lower: Option<f64>,
        upper: Option<f64>,
    },
}

impl ThresholdMethod {
    /// NannyML's default, `StandardDeviationThreshold(3, 3)`.
    pub const NANNYML_DEFAULT: ThresholdMethod = ThresholdMethod::StandardDeviation {
        lower: Some(3.0),
        upper: Some(3.0),
    };
}

impl Default for ThresholdMethod {
    fn default() -> Self {
        Self::NANNYML_DEFAULT
    }
}

/// Theoretical limits of a metric (e.g. `[0, 1]` for accuracy), applied as in
/// NannyML's `calculate_threshold_values`: a computed lower threshold
/// `<= lower` (upper threshold `>= upper`) is replaced by the limit, or
/// dropped (`None`) with `override_with_none`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ValueLimits {
    /// Lower limit of the metric, `None` for unbounded.
    pub lower: Option<f64>,
    /// Upper limit of the metric, `None` for unbounded.
    pub upper: Option<f64>,
    /// Drop a threshold beyond its limit instead of clamping it (NannyML's
    /// `override_using_none`, used by the reconstruction-error drift).
    pub override_with_none: bool,
}

impl ValueLimits {
    /// No limits.
    pub const NONE: ValueLimits = ValueLimits {
        lower: None,
        upper: None,
        override_with_none: false,
    };
    /// `[0, 1]`, clamped (NannyML's limits of the CBPE rate metrics).
    pub const UNIT_INTERVAL: ValueLimits = ValueLimits {
        lower: Some(0.0),
        upper: Some(1.0),
        override_with_none: false,
    };
}

/// Lower and upper alert thresholds; `None` is no threshold on that side.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThresholdBand {
    pub lower: Option<f64>,
    pub upper: Option<f64>,
}

impl ThresholdBand {
    /// `true` when `value` is below the lower or above the upper threshold
    /// (strictly, as NannyML's CBPE `alert`). NaN never alerts.
    pub fn alert(&self, value: f64) -> bool {
        self.lower.is_some_and(|l| value < l) || self.upper.is_some_and(|u| value > u)
    }
}

fn nan_mean_std(values: &[f64]) -> Result<(f64, f64)> {
    let v: Vec<f64> = values.iter().copied().filter(|x| !x.is_nan()).collect();
    if v.is_empty() {
        return Err(StatError::InsufficientData { needed: 1, got: 0 });
    }
    if v.iter().any(|x| x.is_infinite()) {
        return Err(StatError::InvalidParameter(
            "reference values must not be infinite".into(),
        ));
    }
    let n = v.len() as f64;
    let m = v.iter().sum::<f64>() / n;
    let sd = (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / n).sqrt();
    Ok((m, sd))
}

fn check_multiplier(name: &str, m: Option<f64>) -> Result<()> {
    match m {
        Some(k) if !(k.is_finite() && k >= 0.0) => Err(StatError::InvalidParameter(format!(
            "{} must be finite and non-negative, got {}",
            name, k
        ))),
        _ => Ok(()),
    }
}

/// Alert thresholds from the reference values of a metric (typically its
/// value on each reference chunk), then limited by `limits`.
///
/// NaN reference values are ignored (NannyML's `np.nanmean` / `np.nanstd`).
/// [`ThresholdMethod::Constant`] ignores `reference_values` (it may be empty);
/// the other methods need at least one non-NaN value.
///
/// Errors on negative or non-finite multipliers, a constant band with
/// `lower >= upper` (as NannyML), and infinite reference values.
///
/// # Example
/// ```
/// use anofox_statistics::monitoring::{threshold_band, ThresholdMethod, ValueLimits};
/// let reference = [0.91, 0.93, 0.92, 0.95, 0.90];
/// let band = threshold_band(&reference, &ThresholdMethod::NANNYML_DEFAULT,
///                           &ValueLimits::UNIT_INTERVAL).unwrap();
/// assert!(band.alert(0.80));
/// assert!(!band.alert(0.92));
/// ```
pub fn threshold_band(
    reference_values: &[f64],
    method: &ThresholdMethod,
    limits: &ValueLimits,
) -> Result<ThresholdBand> {
    let (lower, upper) = match *method {
        ThresholdMethod::StandardDeviation { lower, upper } => {
            check_multiplier("lower multiplier", lower)?;
            check_multiplier("upper multiplier", upper)?;
            let (m, sd) = nan_mean_std(reference_values)?;
            (lower.map(|k| m - k * sd), upper.map(|k| m + k * sd))
        }
        ThresholdMethod::Relative { lower, upper } => {
            check_multiplier("lower share", lower)?;
            check_multiplier("upper share", upper)?;
            let (m, _) = nan_mean_std(reference_values)?;
            (
                lower.map(|k| m - k * m.abs()),
                upper.map(|k| m + k * m.abs()),
            )
        }
        ThresholdMethod::Constant { lower, upper } => {
            for v in [lower, upper].into_iter().flatten() {
                if !v.is_finite() {
                    return Err(StatError::InvalidParameter(format!(
                        "constant thresholds must be finite, got {}",
                        v
                    )));
                }
            }
            if let (Some(l), Some(u)) = (lower, upper) {
                if l >= u {
                    return Err(StatError::InvalidParameter(format!(
                        "lower threshold {} must be less than upper threshold {}",
                        l, u
                    )));
                }
            }
            (lower, upper)
        }
    };
    let replace = |limit: f64| {
        if limits.override_with_none {
            None
        } else {
            Some(limit)
        }
    };
    let lower = match (lower, limits.lower) {
        (Some(l), Some(lim)) if l <= lim => replace(lim),
        (l, _) => l,
    };
    let upper = match (upper, limits.upper) {
        (Some(u), Some(lim)) if u >= lim => replace(lim),
        (u, _) => u,
    };
    Ok(ThresholdBand { lower, upper })
}
