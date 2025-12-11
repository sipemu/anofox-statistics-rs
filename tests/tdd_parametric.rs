mod common;

use anofox_statistics::{brown_forsythe, t_test, yuen_test, Alternative, TTestKind};
use approx::assert_relative_eq;

const EPSILON: f64 = 1e-10;

// ============================================
// Welch T-Test (Unequal Variances)
// ============================================

#[test]
fn test_welch_two_sided() {
    let refs = common::load_reference_scalars("ttest_welch.csv");
    let g1 = common::load_reference_vector("ttest_g1.csv");
    let g2 = common::load_reference_vector("ttest_g2.csv");

    let result =
        t_test(&g1, &g2, TTestKind::Welch, Alternative::TwoSided, 0.0).expect("t_test should succeed");

    assert_relative_eq!(result.statistic, refs["statistic_two"], epsilon = EPSILON);
    assert_relative_eq!(result.df, refs["df_two"], epsilon = EPSILON);
    assert_relative_eq!(result.p_value, refs["p_value_two"], epsilon = EPSILON);
    assert_relative_eq!(result.mean_x, refs["mean_x_two"], epsilon = EPSILON);
    assert_relative_eq!(
        result.mean_y.unwrap(),
        refs["mean_y_two"],
        epsilon = EPSILON
    );
}

#[test]
fn test_welch_less() {
    let refs = common::load_reference_scalars("ttest_welch.csv");
    let g1 = common::load_reference_vector("ttest_g1.csv");
    let g2 = common::load_reference_vector("ttest_g2.csv");

    let result =
        t_test(&g1, &g2, TTestKind::Welch, Alternative::Less, 0.0).expect("t_test should succeed");

    assert_relative_eq!(result.statistic, refs["statistic_less"], epsilon = EPSILON);
    assert_relative_eq!(result.df, refs["df_less"], epsilon = EPSILON);
    assert_relative_eq!(result.p_value, refs["p_value_less"], epsilon = EPSILON);
}

#[test]
fn test_welch_greater() {
    let refs = common::load_reference_scalars("ttest_welch.csv");
    let g1 = common::load_reference_vector("ttest_g1.csv");
    let g2 = common::load_reference_vector("ttest_g2.csv");

    let result =
        t_test(&g1, &g2, TTestKind::Welch, Alternative::Greater, 0.0).expect("t_test should succeed");

    assert_relative_eq!(
        result.statistic,
        refs["statistic_greater"],
        epsilon = EPSILON
    );
    assert_relative_eq!(result.df, refs["df_greater"], epsilon = EPSILON);
    assert_relative_eq!(result.p_value, refs["p_value_greater"], epsilon = EPSILON);
}

#[test]
fn test_welch_with_mu() {
    let refs = common::load_reference_scalars("ttest_welch.csv");
    let g1 = common::load_reference_vector("ttest_g1.csv");
    let g2 = common::load_reference_vector("ttest_g2.csv");

    let result =
        t_test(&g1, &g2, TTestKind::Welch, Alternative::TwoSided, 0.5).expect("t_test should succeed");

    assert_relative_eq!(result.statistic, refs["statistic_mu"], epsilon = EPSILON);
    assert_relative_eq!(result.p_value, refs["p_value_mu"], epsilon = EPSILON);
}

// ============================================
// Student T-Test (Equal Variances Assumed)
// ============================================

#[test]
fn test_student_two_sided() {
    let refs = common::load_reference_scalars("ttest_student.csv");
    let g1 = common::load_reference_vector("ttest_g1.csv");
    let g2 = common::load_reference_vector("ttest_g2.csv");

    let result =
        t_test(&g1, &g2, TTestKind::Student, Alternative::TwoSided, 0.0).expect("t_test should succeed");

    assert_relative_eq!(result.statistic, refs["statistic_two"], epsilon = EPSILON);
    assert_relative_eq!(result.df, refs["df_two"], epsilon = EPSILON);
    assert_relative_eq!(result.p_value, refs["p_value_two"], epsilon = EPSILON);
    assert_relative_eq!(result.mean_x, refs["mean_x_two"], epsilon = EPSILON);
    assert_relative_eq!(
        result.mean_y.unwrap(),
        refs["mean_y_two"],
        epsilon = EPSILON
    );
}

#[test]
fn test_student_less() {
    let refs = common::load_reference_scalars("ttest_student.csv");
    let g1 = common::load_reference_vector("ttest_g1.csv");
    let g2 = common::load_reference_vector("ttest_g2.csv");

    let result =
        t_test(&g1, &g2, TTestKind::Student, Alternative::Less, 0.0).expect("t_test should succeed");

    assert_relative_eq!(result.statistic, refs["statistic_less"], epsilon = EPSILON);
    assert_relative_eq!(result.df, refs["df_less"], epsilon = EPSILON);
    assert_relative_eq!(result.p_value, refs["p_value_less"], epsilon = EPSILON);
}

#[test]
fn test_student_greater() {
    let refs = common::load_reference_scalars("ttest_student.csv");
    let g1 = common::load_reference_vector("ttest_g1.csv");
    let g2 = common::load_reference_vector("ttest_g2.csv");

    let result =
        t_test(&g1, &g2, TTestKind::Student, Alternative::Greater, 0.0).expect("t_test should succeed");

    assert_relative_eq!(
        result.statistic,
        refs["statistic_greater"],
        epsilon = EPSILON
    );
    assert_relative_eq!(result.df, refs["df_greater"], epsilon = EPSILON);
    assert_relative_eq!(result.p_value, refs["p_value_greater"], epsilon = EPSILON);
}

// ============================================
// Paired T-Test
// ============================================

#[test]
fn test_paired_two_sided() {
    let refs = common::load_reference_scalars("ttest_paired.csv");
    let g1 = common::load_reference_vector("ttest_g1_paired.csv");
    let g2 = common::load_reference_vector("ttest_g2_paired.csv");

    let result =
        t_test(&g1, &g2, TTestKind::Paired, Alternative::TwoSided, 0.0).expect("t_test should succeed");

    assert_relative_eq!(result.statistic, refs["statistic_two"], epsilon = EPSILON);
    assert_relative_eq!(result.df, refs["df_two"], epsilon = EPSILON);
    assert_relative_eq!(result.p_value, refs["p_value_two"], epsilon = EPSILON);
    assert_relative_eq!(result.mean_x, refs["mean_diff_two"], epsilon = EPSILON);
}

#[test]
fn test_paired_less() {
    let refs = common::load_reference_scalars("ttest_paired.csv");
    let g1 = common::load_reference_vector("ttest_g1_paired.csv");
    let g2 = common::load_reference_vector("ttest_g2_paired.csv");

    let result =
        t_test(&g1, &g2, TTestKind::Paired, Alternative::Less, 0.0).expect("t_test should succeed");

    assert_relative_eq!(result.statistic, refs["statistic_less"], epsilon = EPSILON);
    assert_relative_eq!(result.df, refs["df_less"], epsilon = EPSILON);
    assert_relative_eq!(result.p_value, refs["p_value_less"], epsilon = EPSILON);
}

#[test]
fn test_paired_greater() {
    let refs = common::load_reference_scalars("ttest_paired.csv");
    let g1 = common::load_reference_vector("ttest_g1_paired.csv");
    let g2 = common::load_reference_vector("ttest_g2_paired.csv");

    let result =
        t_test(&g1, &g2, TTestKind::Paired, Alternative::Greater, 0.0).expect("t_test should succeed");

    assert_relative_eq!(
        result.statistic,
        refs["statistic_greater"],
        epsilon = EPSILON
    );
    assert_relative_eq!(result.df, refs["df_greater"], epsilon = EPSILON);
    assert_relative_eq!(result.p_value, refs["p_value_greater"], epsilon = EPSILON);
}

// ============================================
// Error Cases
// ============================================

#[test]
fn test_ttest_empty_x_returns_error() {
    let empty: Vec<f64> = vec![];
    let y = vec![1.0, 2.0, 3.0];
    assert!(t_test(&empty, &y, TTestKind::Welch, Alternative::TwoSided, 0.0).is_err());
}

#[test]
fn test_ttest_empty_y_returns_error() {
    let x = vec![1.0, 2.0, 3.0];
    let empty: Vec<f64> = vec![];
    assert!(t_test(&x, &empty, TTestKind::Welch, Alternative::TwoSided, 0.0).is_err());
}

#[test]
fn test_paired_unequal_length_returns_error() {
    let x = vec![1.0, 2.0, 3.0];
    let y = vec![1.0, 2.0];
    assert!(t_test(&x, &y, TTestKind::Paired, Alternative::TwoSided, 0.0).is_err());
}

#[test]
fn test_ttest_insufficient_data_returns_error() {
    let x = vec![1.0];
    let y = vec![2.0];
    assert!(t_test(&x, &y, TTestKind::Welch, Alternative::TwoSided, 0.0).is_err());
}

// ============================================
// Yuen's Test (Robust T-Test with Trimmed Means)
// ============================================

#[test]
fn test_yuen_20_percent_trim() {
    let refs = common::load_reference_scalars("yuen.csv");
    let g1 = common::load_reference_vector("yuen_g1.csv");
    let g2 = common::load_reference_vector("yuen_g2.csv");

    let result = yuen_test(&g1, &g2, 0.2).expect("yuen_test should succeed");

    assert_relative_eq!(result.statistic, refs["statistic_20"], epsilon = EPSILON);
    assert_relative_eq!(result.df, refs["df_20"], epsilon = EPSILON);
    assert_relative_eq!(result.p_value, refs["p_value_20"], epsilon = EPSILON);
    assert_relative_eq!(result.diff, refs["diff_20"], epsilon = EPSILON);
}

#[test]
fn test_yuen_10_percent_trim() {
    let refs = common::load_reference_scalars("yuen.csv");
    let g1 = common::load_reference_vector("yuen_g1.csv");
    let g2 = common::load_reference_vector("yuen_g2.csv");

    let result = yuen_test(&g1, &g2, 0.1).expect("yuen_test should succeed");

    assert_relative_eq!(result.statistic, refs["statistic_10"], epsilon = EPSILON);
    assert_relative_eq!(result.df, refs["df_10"], epsilon = EPSILON);
    assert_relative_eq!(result.p_value, refs["p_value_10"], epsilon = EPSILON);
    assert_relative_eq!(result.diff, refs["diff_10"], epsilon = EPSILON);
}

#[test]
fn test_yuen_invalid_trim_returns_error() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 3.0, 4.0, 5.0, 6.0];
    assert!(yuen_test(&x, &y, 0.5).is_err());
    assert!(yuen_test(&x, &y, -0.1).is_err());
}

#[test]
fn test_yuen_empty_returns_error() {
    let empty: Vec<f64> = vec![];
    let y = vec![1.0, 2.0, 3.0];
    assert!(yuen_test(&empty, &y, 0.2).is_err());
}

// ============================================
// Brown-Forsythe Test (Variance Homogeneity)
// ============================================

#[test]
fn test_brown_forsythe_two_groups() {
    let refs = common::load_reference_scalars("brown_forsythe.csv");
    let g1 = common::load_reference_vector("ttest_g1.csv");
    let g2 = common::load_reference_vector("ttest_g2.csv");

    let result = brown_forsythe(&[&g1, &g2]).expect("brown_forsythe should succeed");

    assert_relative_eq!(result.statistic, refs["statistic_2"], epsilon = EPSILON);
    assert_relative_eq!(result.df1, refs["df1_2"], epsilon = EPSILON);
    assert_relative_eq!(result.df2, refs["df2_2"], epsilon = EPSILON);
    assert_relative_eq!(result.p_value, refs["p_value_2"], epsilon = EPSILON);
}

#[test]
fn test_brown_forsythe_three_groups() {
    let refs = common::load_reference_scalars("brown_forsythe.csv");
    let g1 = common::load_reference_vector("ttest_g1.csv");
    let g2 = common::load_reference_vector("ttest_g2.csv");
    let g3 = common::load_reference_vector("levene_g3.csv");

    let result = brown_forsythe(&[&g1, &g2, &g3]).expect("brown_forsythe should succeed");

    assert_relative_eq!(result.statistic, refs["statistic_3"], epsilon = EPSILON);
    assert_relative_eq!(result.df1, refs["df1_3"], epsilon = EPSILON);
    assert_relative_eq!(result.df2, refs["df2_3"], epsilon = EPSILON);
    assert_relative_eq!(result.p_value, refs["p_value_3"], epsilon = EPSILON);
}

#[test]
fn test_brown_forsythe_single_group_returns_error() {
    let g1 = vec![1.0, 2.0, 3.0];
    assert!(brown_forsythe(&[&g1[..]]).is_err());
}

#[test]
fn test_brown_forsythe_empty_group_returns_error() {
    let g1 = vec![1.0, 2.0, 3.0];
    let empty: Vec<f64> = vec![];
    assert!(brown_forsythe(&[&g1[..], &empty[..]]).is_err());
}
