mod common;

use approx::assert_relative_eq;
use libanostat::shapiro_wilk;

// Shapiro-Wilk has known implementation variations across software packages.
// The W statistic typically matches to 3-4 decimal places between implementations.
const W_EPSILON: f64 = 1e-3;
// P-values can vary more due to different approximation methods.
const P_EPSILON: f64 = 0.05;

// ============================================
// Shapiro-Wilk Test
// ============================================

#[test]
fn test_shapiro_wilk_normal_small() {
    let refs = common::load_reference_scalars("shapiro_wilk.csv");
    let data = common::load_reference_vector("sw_normal_small.csv");

    let result = shapiro_wilk(&data).expect("shapiro_wilk should succeed");

    assert_relative_eq!(
        result.statistic,
        refs["w_normal_small"],
        epsilon = W_EPSILON
    );
    assert_relative_eq!(result.p_value, refs["p_normal_small"], epsilon = P_EPSILON);
}

#[test]
fn test_shapiro_wilk_normal_medium() {
    let refs = common::load_reference_scalars("shapiro_wilk.csv");
    let data = common::load_reference_vector("sw_normal_medium.csv");

    let result = shapiro_wilk(&data).expect("shapiro_wilk should succeed");

    assert_relative_eq!(
        result.statistic,
        refs["w_normal_medium"],
        epsilon = W_EPSILON
    );
    assert_relative_eq!(result.p_value, refs["p_normal_medium"], epsilon = P_EPSILON);
}

#[test]
fn test_shapiro_wilk_normal_large() {
    let refs = common::load_reference_scalars("shapiro_wilk.csv");
    let data = common::load_reference_vector("sw_normal_large.csv");

    let result = shapiro_wilk(&data).expect("shapiro_wilk should succeed");

    assert_relative_eq!(
        result.statistic,
        refs["w_normal_large"],
        epsilon = W_EPSILON
    );
    assert_relative_eq!(result.p_value, refs["p_normal_large"], epsilon = P_EPSILON);
}

#[test]
fn test_shapiro_wilk_uniform() {
    let refs = common::load_reference_scalars("shapiro_wilk.csv");
    let data = common::load_reference_vector("sw_uniform.csv");

    let result = shapiro_wilk(&data).expect("shapiro_wilk should succeed");

    assert_relative_eq!(result.statistic, refs["w_uniform"], epsilon = W_EPSILON);
    assert_relative_eq!(result.p_value, refs["p_uniform"], epsilon = P_EPSILON);
}

#[test]
fn test_shapiro_wilk_exponential() {
    let refs = common::load_reference_scalars("shapiro_wilk.csv");
    let data = common::load_reference_vector("sw_exp.csv");

    let result = shapiro_wilk(&data).expect("shapiro_wilk should succeed");

    assert_relative_eq!(result.statistic, refs["w_exp"], epsilon = W_EPSILON);
    assert_relative_eq!(result.p_value, refs["p_exp"], epsilon = P_EPSILON);
}

#[test]
fn test_shapiro_wilk_too_small_returns_error() {
    // Shapiro-Wilk requires at least 3 observations
    let data = vec![1.0, 2.0];
    assert!(shapiro_wilk(&data).is_err());
}

#[test]
fn test_shapiro_wilk_empty_returns_error() {
    let empty: Vec<f64> = vec![];
    assert!(shapiro_wilk(&empty).is_err());
}
