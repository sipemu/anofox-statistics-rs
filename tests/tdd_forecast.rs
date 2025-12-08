mod common;

use approx::assert_relative_eq;
use libanostat::{diebold_mariano, LossFunction};

const EPSILON: f64 = 1e-6;

// ============================================
// Diebold-Mariano Test
// ============================================

#[test]
fn test_dm_squared_error_h1() {
    let refs = common::load_reference_scalars("diebold_mariano.csv");
    let e1 = common::load_reference_vector("dm_e1.csv");
    let e2 = common::load_reference_vector("dm_e2.csv");

    let result = diebold_mariano(&e1, &e2, LossFunction::SquaredError, 1)
        .expect("diebold_mariano should succeed");

    assert_relative_eq!(result.statistic, refs["statistic_se_h1"], epsilon = EPSILON);
    assert_relative_eq!(result.p_value, refs["p_value_se_h1"], epsilon = EPSILON);
}

#[test]
fn test_dm_absolute_error_h1() {
    let refs = common::load_reference_scalars("diebold_mariano.csv");
    let e1 = common::load_reference_vector("dm_e1.csv");
    let e2 = common::load_reference_vector("dm_e2.csv");

    let result = diebold_mariano(&e1, &e2, LossFunction::AbsoluteError, 1)
        .expect("diebold_mariano should succeed");

    assert_relative_eq!(result.statistic, refs["statistic_ae_h1"], epsilon = EPSILON);
    assert_relative_eq!(result.p_value, refs["p_value_ae_h1"], epsilon = EPSILON);
}

#[test]
fn test_dm_squared_error_h3() {
    let refs = common::load_reference_scalars("diebold_mariano.csv");
    let e1 = common::load_reference_vector("dm_e1.csv");
    let e2 = common::load_reference_vector("dm_e2.csv");

    let result = diebold_mariano(&e1, &e2, LossFunction::SquaredError, 3)
        .expect("diebold_mariano should succeed");

    assert_relative_eq!(result.statistic, refs["statistic_se_h3"], epsilon = EPSILON);
    assert_relative_eq!(result.p_value, refs["p_value_se_h3"], epsilon = EPSILON);
}

#[test]
fn test_dm_empty_returns_error() {
    let e1: Vec<f64> = vec![];
    let e2: Vec<f64> = vec![];
    assert!(diebold_mariano(&e1, &e2, LossFunction::SquaredError, 1).is_err());
}

#[test]
fn test_dm_unequal_length_returns_error() {
    let e1 = vec![1.0, 2.0, 3.0];
    let e2 = vec![1.0, 2.0];
    assert!(diebold_mariano(&e1, &e2, LossFunction::SquaredError, 1).is_err());
}

#[test]
fn test_dm_insufficient_data_returns_error() {
    let e1 = vec![1.0, 2.0];
    let e2 = vec![1.5, 2.5];
    assert!(diebold_mariano(&e1, &e2, LossFunction::SquaredError, 1).is_err());
}
