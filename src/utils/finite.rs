//! Input validation for non-finite values (NaN, ±Inf).
//!
//! Rank-based tests accept `±Inf` (it ranks as an ordinary extreme value, as in
//! R) but reject `NaN`. Moment-based tests reject every non-finite value.

use crate::error::{Result, StatError};

/// Error if `data` contains NaN or ±Inf.
pub(crate) fn ensure_finite(name: &str, data: &[f64]) -> Result<()> {
    match data.iter().position(|v| !v.is_finite()) {
        Some(i) => Err(StatError::InvalidParameter(format!(
            "non-finite value in {} at index {}: {}",
            name, i, data[i]
        ))),
        None => Ok(()),
    }
}

/// Error if `data` contains NaN (±Inf is allowed).
pub(crate) fn ensure_no_nan(name: &str, data: &[f64]) -> Result<()> {
    match data.iter().position(|v| v.is_nan()) {
        Some(i) => Err(StatError::InvalidParameter(format!(
            "non-finite value in {} at index {}: NaN",
            name, i
        ))),
        None => Ok(()),
    }
}

/// Error if the scalar parameter `value` is NaN or ±Inf.
pub(crate) fn ensure_finite_param(name: &str, value: f64) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(StatError::InvalidParameter(format!(
            "{} must be finite, got {}",
            name, value
        )))
    }
}

/// Error unless `0 < value < 1` (rejects NaN).
pub(crate) fn ensure_open_unit(name: &str, value: f64) -> Result<()> {
    if value > 0.0 && value < 1.0 {
        Ok(())
    } else {
        Err(StatError::InvalidParameter(format!(
            "{} must be between 0 and 1, got {}",
            name, value
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finite_checks() {
        assert!(ensure_finite("x", &[1.0, 2.0]).is_ok());
        assert!(ensure_finite("x", &[1.0, f64::INFINITY]).is_err());
        assert!(ensure_finite("x", &[f64::NAN]).is_err());
        assert!(ensure_no_nan("x", &[f64::INFINITY, f64::NEG_INFINITY]).is_ok());
        assert!(ensure_no_nan("x", &[1.0, f64::NAN]).is_err());
        assert!(ensure_finite_param("mu", f64::NAN).is_err());
        assert!(ensure_open_unit("conf_level", f64::NAN).is_err());
        assert!(ensure_open_unit("conf_level", 0.95).is_ok());
    }
}
