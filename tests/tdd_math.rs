mod common;

use approx::assert_relative_eq;
use libanostat::utils::math;

const EPSILON: f64 = 1e-12;

// ============================================
// Mean Tests
// ============================================

#[test]
fn test_mean_short_vector() {
    let refs = common::load_reference_scalars("math_basic.csv");
    let x_short = vec![1.2, 2.3, 3.4, 4.5, 5.6];

    let result = math::mean(&x_short).expect("mean should succeed");
    assert_relative_eq!(result, refs["mean_short"], epsilon = EPSILON);
}

#[test]
fn test_mean_long_vector() {
    let refs = common::load_reference_scalars("math_basic.csv");
    let x_long = common::load_reference_vector("vec_long.csv");

    let result = math::mean(&x_long).expect("mean should succeed");
    assert_relative_eq!(result, refs["mean_long"], epsilon = EPSILON);
}

#[test]
fn test_mean_single_element() {
    let refs = common::load_reference_scalars("math_basic.csv");
    let x_single = vec![42.0];

    let result = math::mean(&x_single).expect("mean should succeed");
    assert_relative_eq!(result, refs["mean_single"], epsilon = EPSILON);
}

#[test]
fn test_mean_empty_returns_error() {
    let empty: Vec<f64> = vec![];
    assert!(math::mean(&empty).is_err());
}

// ============================================
// Variance Tests
// ============================================

#[test]
fn test_variance_short_vector() {
    let refs = common::load_reference_scalars("math_basic.csv");
    let x_short = vec![1.2, 2.3, 3.4, 4.5, 5.6];

    let result = math::variance(&x_short).expect("variance should succeed");
    assert_relative_eq!(result, refs["var_short"], epsilon = EPSILON);
}

#[test]
fn test_variance_long_vector() {
    let refs = common::load_reference_scalars("math_basic.csv");
    let x_long = common::load_reference_vector("vec_long.csv");

    let result = math::variance(&x_long).expect("variance should succeed");
    assert_relative_eq!(result, refs["var_long"], epsilon = EPSILON);
}

#[test]
fn test_variance_outlier_vector() {
    let refs = common::load_reference_scalars("math_basic.csv");
    let x_outlier = vec![1.0, 1.0, 1.0, 1.0, 100.0];

    let result = math::variance(&x_outlier).expect("variance should succeed");
    assert_relative_eq!(result, refs["var_outlier"], epsilon = EPSILON);
}

#[test]
fn test_variance_single_element_returns_error() {
    let x_single = vec![42.0];
    // R's var() returns NA for single element; we return error
    assert!(math::variance(&x_single).is_err());
}

#[test]
fn test_variance_empty_returns_error() {
    let empty: Vec<f64> = vec![];
    assert!(math::variance(&empty).is_err());
}

// ============================================
// Median Tests
// ============================================

#[test]
fn test_median_short_vector() {
    let refs = common::load_reference_scalars("math_basic.csv");
    let x_short = vec![1.2, 2.3, 3.4, 4.5, 5.6];

    let result = math::median(&x_short).expect("median should succeed");
    assert_relative_eq!(result, refs["median_short"], epsilon = EPSILON);
}

#[test]
fn test_median_long_vector() {
    let refs = common::load_reference_scalars("math_basic.csv");
    let x_long = common::load_reference_vector("vec_long.csv");

    let result = math::median(&x_long).expect("median should succeed");
    assert_relative_eq!(result, refs["median_long"], epsilon = EPSILON);
}

#[test]
fn test_median_even_length() {
    let refs = common::load_reference_scalars("math_basic.csv");
    let x_even = vec![1.0, 2.0, 3.0, 4.0];

    let result = math::median(&x_even).expect("median should succeed");
    assert_relative_eq!(result, refs["median_even"], epsilon = EPSILON);
}

#[test]
fn test_median_single_element() {
    let refs = common::load_reference_scalars("math_basic.csv");
    let x_single = vec![42.0];

    let result = math::median(&x_single).expect("median should succeed");
    assert_relative_eq!(result, refs["median_single"], epsilon = EPSILON);
}

#[test]
fn test_median_outlier_robust() {
    let refs = common::load_reference_scalars("math_basic.csv");
    let x_outlier = vec![1.0, 1.0, 1.0, 1.0, 100.0];

    let result = math::median(&x_outlier).expect("median should succeed");
    assert_relative_eq!(result, refs["median_outlier"], epsilon = EPSILON);
}

#[test]
fn test_median_empty_returns_error() {
    let empty: Vec<f64> = vec![];
    assert!(math::median(&empty).is_err());
}

// ============================================
// Trimmed Mean Tests
// ============================================

#[test]
fn test_trimmed_mean_short_vector() {
    let refs = common::load_reference_scalars("math_basic.csv");
    let x_short = vec![1.2, 2.3, 3.4, 4.5, 5.6];

    let result = math::trimmed_mean(&x_short, 0.2).expect("trimmed_mean should succeed");
    assert_relative_eq!(result, refs["trim_20_short"], epsilon = EPSILON);
}

#[test]
fn test_trimmed_mean_long_vector() {
    let refs = common::load_reference_scalars("math_basic.csv");
    let x_long = common::load_reference_vector("vec_long.csv");

    let result = math::trimmed_mean(&x_long, 0.2).expect("trimmed_mean should succeed");
    assert_relative_eq!(result, refs["trim_20_long"], epsilon = EPSILON);
}

#[test]
fn test_trimmed_mean_outlier_robust() {
    let refs = common::load_reference_scalars("math_basic.csv");
    let x_outlier = vec![1.0, 1.0, 1.0, 1.0, 100.0];

    let result = math::trimmed_mean(&x_outlier, 0.2).expect("trimmed_mean should succeed");
    assert_relative_eq!(result, refs["trim_20_outlier"], epsilon = EPSILON);
}

#[test]
fn test_trimmed_mean_invalid_trim_too_high() {
    let x = vec![1.0, 2.0, 3.0];
    // trim >= 0.5 is invalid
    assert!(math::trimmed_mean(&x, 0.5).is_err());
    assert!(math::trimmed_mean(&x, 0.6).is_err());
}

#[test]
fn test_trimmed_mean_invalid_trim_negative() {
    let x = vec![1.0, 2.0, 3.0];
    assert!(math::trimmed_mean(&x, -0.1).is_err());
}

#[test]
fn test_trimmed_mean_zero_trim_equals_mean() {
    let x_short = vec![1.2, 2.3, 3.4, 4.5, 5.6];

    let mean_result = math::mean(&x_short).expect("mean should succeed");
    let trim_result = math::trimmed_mean(&x_short, 0.0).expect("trimmed_mean should succeed");

    assert_relative_eq!(mean_result, trim_result, epsilon = EPSILON);
}

#[test]
fn test_trimmed_mean_empty_returns_error() {
    let empty: Vec<f64> = vec![];
    assert!(math::trimmed_mean(&empty, 0.2).is_err());
}
