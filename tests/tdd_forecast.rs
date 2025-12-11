mod common;

use anofox_statistics::{
    clark_west, diebold_mariano, model_confidence_set, spa_test, Alternative, LossFunction,
    MCSStatistic,
};
use approx::assert_relative_eq;

const EPSILON: f64 = 1e-6;

// ============================================
// Diebold-Mariano Test
// ============================================

#[test]
fn test_dm_squared_error_h1() {
    let refs = common::load_reference_scalars("diebold_mariano.csv");
    let e1 = common::load_reference_vector("dm_e1.csv");
    let e2 = common::load_reference_vector("dm_e2.csv");

    let result = diebold_mariano(&e1, &e2, LossFunction::SquaredError, 1, Alternative::TwoSided)
        .expect("diebold_mariano should succeed");

    assert_relative_eq!(result.statistic, refs["statistic_se_h1"], epsilon = EPSILON);
    assert_relative_eq!(result.p_value, refs["p_value_se_h1"], epsilon = EPSILON);
}

#[test]
fn test_dm_absolute_error_h1() {
    let refs = common::load_reference_scalars("diebold_mariano.csv");
    let e1 = common::load_reference_vector("dm_e1.csv");
    let e2 = common::load_reference_vector("dm_e2.csv");

    let result = diebold_mariano(&e1, &e2, LossFunction::AbsoluteError, 1, Alternative::TwoSided)
        .expect("diebold_mariano should succeed");

    assert_relative_eq!(result.statistic, refs["statistic_ae_h1"], epsilon = EPSILON);
    assert_relative_eq!(result.p_value, refs["p_value_ae_h1"], epsilon = EPSILON);
}

#[test]
fn test_dm_squared_error_h3() {
    let refs = common::load_reference_scalars("diebold_mariano.csv");
    let e1 = common::load_reference_vector("dm_e1.csv");
    let e2 = common::load_reference_vector("dm_e2.csv");

    let result = diebold_mariano(&e1, &e2, LossFunction::SquaredError, 3, Alternative::TwoSided)
        .expect("diebold_mariano should succeed");

    assert_relative_eq!(result.statistic, refs["statistic_se_h3"], epsilon = EPSILON);
    assert_relative_eq!(result.p_value, refs["p_value_se_h3"], epsilon = EPSILON);
}

#[test]
fn test_dm_less() {
    let refs = common::load_reference_scalars("diebold_mariano.csv");
    let e1 = common::load_reference_vector("dm_e1.csv");
    let e2 = common::load_reference_vector("dm_e2.csv");

    let result = diebold_mariano(&e1, &e2, LossFunction::SquaredError, 1, Alternative::Less)
        .expect("diebold_mariano should succeed");

    assert_relative_eq!(
        result.p_value,
        refs["p_value_less_se_h1"],
        epsilon = EPSILON
    );
}

#[test]
fn test_dm_greater() {
    let refs = common::load_reference_scalars("diebold_mariano.csv");
    let e1 = common::load_reference_vector("dm_e1.csv");
    let e2 = common::load_reference_vector("dm_e2.csv");

    let result = diebold_mariano(&e1, &e2, LossFunction::SquaredError, 1, Alternative::Greater)
        .expect("diebold_mariano should succeed");

    assert_relative_eq!(
        result.p_value,
        refs["p_value_greater_se_h1"],
        epsilon = EPSILON
    );
}

#[test]
fn test_dm_empty_returns_error() {
    let e1: Vec<f64> = vec![];
    let e2: Vec<f64> = vec![];
    assert!(diebold_mariano(&e1, &e2, LossFunction::SquaredError, 1, Alternative::TwoSided).is_err());
}

#[test]
fn test_dm_unequal_length_returns_error() {
    let e1 = vec![1.0, 2.0, 3.0];
    let e2 = vec![1.0, 2.0];
    assert!(diebold_mariano(&e1, &e2, LossFunction::SquaredError, 1, Alternative::TwoSided).is_err());
}

#[test]
fn test_dm_insufficient_data_returns_error() {
    let e1 = vec![1.0, 2.0];
    let e2 = vec![1.5, 2.5];
    assert!(diebold_mariano(&e1, &e2, LossFunction::SquaredError, 1, Alternative::TwoSided).is_err());
}

// ============================================
// Clark-West Test
// ============================================

#[test]
fn test_cw_h1() {
    let refs = common::load_reference_scalars("clark_west.csv");
    let e1 = common::load_reference_vector("dm_e1.csv");
    let e2 = common::load_reference_vector("dm_e2.csv");

    let result = clark_west(&e1, &e2, 1).expect("clark_west should succeed");

    assert_relative_eq!(result.statistic, refs["statistic_h1"], epsilon = EPSILON);
    assert_relative_eq!(result.p_value, refs["p_value_one_h1"], epsilon = EPSILON);
    assert_relative_eq!(
        result.p_value_two_sided,
        refs["p_value_two_h1"],
        epsilon = EPSILON
    );
}

#[test]
fn test_cw_h3() {
    let refs = common::load_reference_scalars("clark_west.csv");
    let e1 = common::load_reference_vector("dm_e1.csv");
    let e2 = common::load_reference_vector("dm_e2.csv");

    let result = clark_west(&e1, &e2, 3).expect("clark_west should succeed");

    assert_relative_eq!(result.statistic, refs["statistic_h3"], epsilon = EPSILON);
    assert_relative_eq!(result.p_value, refs["p_value_one_h3"], epsilon = EPSILON);
    assert_relative_eq!(
        result.p_value_two_sided,
        refs["p_value_two_h3"],
        epsilon = EPSILON
    );
}

#[test]
fn test_cw_empty_returns_error() {
    let e1: Vec<f64> = vec![];
    let e2: Vec<f64> = vec![];
    assert!(clark_west(&e1, &e2, 1).is_err());
}

#[test]
fn test_cw_unequal_length_returns_error() {
    let e1 = vec![1.0, 2.0, 3.0];
    let e2 = vec![1.0, 2.0];
    assert!(clark_west(&e1, &e2, 1).is_err());
}

// ============================================
// SPA Test (Superior Predictive Ability)
// ============================================

#[test]
fn test_spa_statistic_and_best_model() {
    let refs = common::load_reference_scalars("spa.csv");
    let benchmark = common::load_reference_vector("spa_benchmark.csv");
    let model1 = common::load_reference_vector("spa_model1.csv");
    let model2 = common::load_reference_vector("spa_model2.csv");
    let model3 = common::load_reference_vector("spa_model3.csv");

    let models = vec![model1, model2, model3];

    let result =
        spa_test(&benchmark, &models, 499, 5.0, Some(42)).expect("spa_test should succeed");

    // Verify the test statistic matches R
    assert_relative_eq!(result.statistic, refs["statistic"], epsilon = 0.01);

    // Verify best model index (model 1 with index 0 should be best)
    assert_eq!(
        result.best_model_idx,
        Some(refs["best_idx"] as usize),
        "Best model should match R"
    );
}

#[test]
fn test_spa_clearly_better_model_rejects_null() {
    // Model that is clearly better than benchmark should have low p-value
    let benchmark: Vec<f64> = vec![10.0; 50];
    let model: Vec<Vec<f64>> = vec![vec![1.0; 50]];

    let result = spa_test(&benchmark, &model, 499, 3.0, Some(42)).expect("spa_test should succeed");

    assert!(
        result.p_value_consistent < 0.05,
        "p_value {} should reject null for clearly better model",
        result.p_value_consistent
    );
}

#[test]
fn test_spa_empty_returns_error() {
    let benchmark: Vec<f64> = vec![];
    let models: Vec<Vec<f64>> = vec![vec![1.0, 2.0, 3.0]];
    assert!(spa_test(&benchmark, &models, 100, 3.0, None).is_err());
}

#[test]
fn test_spa_no_models_returns_error() {
    let benchmark: Vec<f64> = vec![1.0, 2.0, 3.0];
    let models: Vec<Vec<f64>> = vec![];
    assert!(spa_test(&benchmark, &models, 100, 3.0, None).is_err());
}

// ============================================
// Model Confidence Set (MCS)
// ============================================

#[test]
fn test_mcs_eliminates_worst_models() {
    // Load test data: model0 (best), model1 (good), model2 (medium), model3 (worst)
    let model0 = common::load_reference_vector("mcs_model0.csv");
    let model1 = common::load_reference_vector("mcs_model1.csv");
    let model2 = common::load_reference_vector("mcs_model2.csv");
    let model3 = common::load_reference_vector("mcs_model3.csv");

    let losses = vec![model0, model1, model2, model3];

    let result = model_confidence_set(&losses, 0.10, MCSStatistic::Range, 500, 5.0, Some(42))
        .expect("model_confidence_set should succeed");

    // Best models (0, 1) should be in the confidence set
    assert!(
        result.included_models.contains(&0),
        "Model 0 (best) should be included"
    );

    // Worst model (3) should be eliminated
    assert!(
        result.eliminated_models.contains(&3),
        "Model 3 (worst) should be eliminated"
    );
}

#[test]
fn test_mcs_identical_models_all_included() {
    // When all models are identical, none should be eliminated
    let base: Vec<f64> = (0..100)
        .map(|i| (i as f64 * 0.1).sin().abs() + 1.0)
        .collect();
    let losses = vec![base.clone(), base.clone(), base.clone()];

    let result = model_confidence_set(&losses, 0.10, MCSStatistic::Range, 500, 5.0, Some(42))
        .expect("model_confidence_set should succeed");

    assert_eq!(
        result.included_models.len(),
        3,
        "All identical models should be included"
    );
    assert!(
        result.eliminated_models.is_empty(),
        "No models should be eliminated when identical"
    );
}

#[test]
fn test_mcs_single_model() {
    let losses = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];

    let result = model_confidence_set(&losses, 0.10, MCSStatistic::Range, 100, 3.0, Some(42))
        .expect("model_confidence_set should succeed");

    assert_eq!(result.included_models, vec![0]);
    assert!(result.eliminated_models.is_empty());
    assert_eq!(result.mcs_p_value, 1.0);
}

#[test]
fn test_mcs_empty_returns_error() {
    let losses: Vec<Vec<f64>> = vec![];
    assert!(model_confidence_set(&losses, 0.10, MCSStatistic::Range, 100, 3.0, None).is_err());
}

#[test]
fn test_mcs_invalid_alpha_returns_error() {
    let losses = vec![vec![1.0, 2.0, 3.0]];
    assert!(model_confidence_set(&losses, 0.0, MCSStatistic::Range, 100, 3.0, None).is_err());
    assert!(model_confidence_set(&losses, 1.0, MCSStatistic::Range, 100, 3.0, None).is_err());
}
