//! NaN-safe wrappers around `statrs` distribution functions.
//!
//! `statrs` panics when, for example, `StudentsT::cdf` gets a NaN argument,
//! `FisherSnedecor::sf` gets a NaN statistic, or `inverse_cdf` gets a
//! probability outside `[0, 1]`. A NaN statistic can come from degenerate data
//! (zero variance), so these wrappers return NaN instead of panicking.

use statrs::distribution::{ChiSquared, ContinuousCDF, FisherSnedecor, Normal, StudentsT};

fn t_dist(df: f64) -> Option<StudentsT> {
    StudentsT::new(0.0, 1.0, df).ok()
}

/// Student t CDF; NaN for a NaN statistic or invalid df.
pub(crate) fn t_cdf(x: f64, df: f64) -> f64 {
    match t_dist(df) {
        Some(d) if !x.is_nan() => d.cdf(x),
        _ => f64::NAN,
    }
}

/// Student t survival function; NaN for a NaN statistic or invalid df.
pub(crate) fn t_sf(x: f64, df: f64) -> f64 {
    match t_dist(df) {
        Some(d) if !x.is_nan() => d.sf(x),
        _ => f64::NAN,
    }
}

/// Student t quantile; NaN for p outside [0, 1] or invalid df.
pub(crate) fn t_inv(p: f64, df: f64) -> f64 {
    match t_dist(df) {
        Some(d) if (0.0..=1.0).contains(&p) => d.inverse_cdf(p),
        _ => f64::NAN,
    }
}

/// F survival function; NaN for a NaN statistic or invalid df.
pub(crate) fn f_sf(x: f64, df1: f64, df2: f64) -> f64 {
    match FisherSnedecor::new(df1, df2) {
        Ok(d) if !x.is_nan() => d.sf(x),
        _ => f64::NAN,
    }
}

/// Chi-squared survival function; NaN for a NaN statistic or invalid df.
pub(crate) fn chisq_sf(x: f64, df: f64) -> f64 {
    match ChiSquared::new(df) {
        Ok(d) if !x.is_nan() => d.sf(x),
        _ => f64::NAN,
    }
}

/// Standard normal quantile; NaN for p outside [0, 1].
pub(crate) fn norm_inv(p: f64) -> f64 {
    if (0.0..=1.0).contains(&p) {
        Normal::new(0.0, 1.0).unwrap().inverse_cdf(p)
    } else {
        f64::NAN
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nan_inputs_do_not_panic() {
        assert!(t_cdf(f64::NAN, 5.0).is_nan());
        assert!(t_sf(f64::NAN, 5.0).is_nan());
        assert!(t_sf(1.0, f64::NAN).is_nan());
        assert!(t_inv(f64::NAN, 5.0).is_nan());
        assert!(f_sf(f64::NAN, 2.0, 10.0).is_nan());
        assert!(f_sf(1.0, 0.0, 10.0).is_nan());
        assert!(chisq_sf(f64::NAN, 2.0).is_nan());
        assert!(norm_inv(f64::NAN).is_nan());
        assert_eq!(t_sf(f64::INFINITY, 5.0), 0.0);
        assert!((t_sf(0.0, 5.0) - 0.5).abs() < 1e-12);
    }
}
