mod common;

use anofox_statistics::{
    brunner_munzel, kruskal_wallis, mann_whitney_u, rank, wilcoxon_signed_rank, Alternative,
};
use approx::assert_relative_eq;

const EPSILON: f64 = 1e-10;

// ============================================
// Ranking Tests
// ============================================

#[test]
fn test_rank_simple() {
    let refs = common::load_reference_scalars("ranks.csv");
    let data = common::load_reference_vector("rank_simple.csv");

    let ranks = rank(&data).expect("rank should succeed");

    assert_eq!(ranks.len(), 8);
    assert_relative_eq!(ranks[0], refs["rank_simple_1"], epsilon = EPSILON);
    assert_relative_eq!(ranks[1], refs["rank_simple_2"], epsilon = EPSILON);
    assert_relative_eq!(ranks[2], refs["rank_simple_3"], epsilon = EPSILON);
    assert_relative_eq!(ranks[3], refs["rank_simple_4"], epsilon = EPSILON);
    assert_relative_eq!(ranks[4], refs["rank_simple_5"], epsilon = EPSILON);
    assert_relative_eq!(ranks[5], refs["rank_simple_6"], epsilon = EPSILON);
    assert_relative_eq!(ranks[6], refs["rank_simple_7"], epsilon = EPSILON);
    assert_relative_eq!(ranks[7], refs["rank_simple_8"], epsilon = EPSILON);
}

#[test]
fn test_rank_with_ties() {
    let refs = common::load_reference_scalars("ranks.csv");
    let data = common::load_reference_vector("rank_ties.csv");

    let ranks = rank(&data).expect("rank should succeed");

    assert_eq!(ranks.len(), 7);
    // Ties get average rank
    assert_relative_eq!(ranks[0], refs["rank_ties_1"], epsilon = EPSILON);
    assert_relative_eq!(ranks[1], refs["rank_ties_2"], epsilon = EPSILON);
    assert_relative_eq!(ranks[2], refs["rank_ties_3"], epsilon = EPSILON);
    assert_relative_eq!(ranks[3], refs["rank_ties_4"], epsilon = EPSILON);
    assert_relative_eq!(ranks[4], refs["rank_ties_5"], epsilon = EPSILON);
    assert_relative_eq!(ranks[5], refs["rank_ties_6"], epsilon = EPSILON);
    assert_relative_eq!(ranks[6], refs["rank_ties_7"], epsilon = EPSILON);
}

#[test]
fn test_rank_empty_returns_error() {
    let empty: Vec<f64> = vec![];
    assert!(rank(&empty).is_err());
}

// ============================================
// Mann-Whitney U Test
// ============================================

#[test]
fn test_mann_whitney_u() {
    let refs = common::load_reference_scalars("mann_whitney.csv");
    let x = common::load_reference_vector("mw_x.csv");
    let y = common::load_reference_vector("mw_y.csv");

    let result = mann_whitney_u(&x, &y).expect("mann_whitney_u should succeed");

    assert_relative_eq!(result.statistic, refs["statistic"], epsilon = EPSILON);
    assert_relative_eq!(result.p_value, refs["p_value"], epsilon = 1e-6);
}

#[test]
fn test_mann_whitney_empty_x_returns_error() {
    let empty: Vec<f64> = vec![];
    let y = vec![1.0, 2.0, 3.0];
    assert!(mann_whitney_u(&empty, &y).is_err());
}

#[test]
fn test_mann_whitney_empty_y_returns_error() {
    let x = vec![1.0, 2.0, 3.0];
    let empty: Vec<f64> = vec![];
    assert!(mann_whitney_u(&x, &empty).is_err());
}

// ============================================
// Wilcoxon Signed-Rank Test
// ============================================

#[test]
fn test_wilcoxon_signed_rank() {
    let refs = common::load_reference_scalars("wilcoxon_signed_rank.csv");
    let x = common::load_reference_vector("wsr_x.csv");
    let y = common::load_reference_vector("wsr_y.csv");

    let result = wilcoxon_signed_rank(&x, &y).expect("wilcoxon_signed_rank should succeed");

    assert_relative_eq!(result.statistic, refs["statistic"], epsilon = EPSILON);
    assert_relative_eq!(result.p_value, refs["p_value"], epsilon = 1e-6);
}

#[test]
fn test_wilcoxon_signed_rank_unequal_length_returns_error() {
    let x = vec![1.0, 2.0, 3.0];
    let y = vec![1.0, 2.0];
    assert!(wilcoxon_signed_rank(&x, &y).is_err());
}

#[test]
fn test_wilcoxon_signed_rank_empty_returns_error() {
    let empty: Vec<f64> = vec![];
    let y = vec![1.0, 2.0, 3.0];
    assert!(wilcoxon_signed_rank(&empty, &y).is_err());
}

// ============================================
// Kruskal-Wallis Test
// ============================================

#[test]
fn test_kruskal_wallis() {
    let refs = common::load_reference_scalars("kruskal_wallis.csv");
    let a = common::load_reference_vector("kw_a.csv");
    let b = common::load_reference_vector("kw_b.csv");
    let c = common::load_reference_vector("kw_c.csv");

    let result = kruskal_wallis(&[&a, &b, &c]).expect("kruskal_wallis should succeed");

    assert_relative_eq!(result.statistic, refs["statistic"], epsilon = EPSILON);
    assert_relative_eq!(result.df, refs["df"], epsilon = EPSILON);
    assert_relative_eq!(result.p_value, refs["p_value"], epsilon = 1e-6);
}

#[test]
fn test_kruskal_wallis_single_group_returns_error() {
    let a = vec![1.0, 2.0, 3.0];
    assert!(kruskal_wallis(&[&a[..]]).is_err());
}

#[test]
fn test_kruskal_wallis_empty_group_returns_error() {
    let a = vec![1.0, 2.0, 3.0];
    let empty: Vec<f64> = vec![];
    assert!(kruskal_wallis(&[&a[..], &empty[..]]).is_err());
}

// ============================================
// Brunner-Munzel Test
// ============================================

#[test]
fn test_brunner_munzel() {
    let refs = common::load_reference_scalars("brunner_munzel.csv");
    let x = common::load_reference_vector("bm_x.csv");
    let y = common::load_reference_vector("bm_y.csv");

    let result =
        brunner_munzel(&x, &y, Alternative::TwoSided).expect("brunner_munzel should succeed");

    assert_relative_eq!(result.statistic, refs["statistic"], epsilon = 1e-6);
    assert_relative_eq!(result.df, refs["df"], epsilon = 1e-6);
    assert_relative_eq!(result.p_value, refs["p_value"], epsilon = 1e-6);
    assert_relative_eq!(result.estimate, refs["estimate"], epsilon = 1e-6);
}

#[test]
fn test_brunner_munzel_empty_x_returns_error() {
    let empty: Vec<f64> = vec![];
    let y = vec![1.0, 2.0, 3.0];
    assert!(brunner_munzel(&empty, &y, Alternative::TwoSided).is_err());
}

#[test]
fn test_brunner_munzel_empty_y_returns_error() {
    let x = vec![1.0, 2.0, 3.0];
    let empty: Vec<f64> = vec![];
    assert!(brunner_munzel(&x, &empty, Alternative::TwoSided).is_err());
}
