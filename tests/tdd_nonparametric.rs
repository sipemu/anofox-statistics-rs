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
fn test_mann_whitney_u_two_sided() {
    let refs = common::load_reference_scalars("mann_whitney.csv");
    let x = common::load_reference_vector("mw_x.csv");
    let y = common::load_reference_vector("mw_y.csv");

    let result = mann_whitney_u(&x, &y, Alternative::TwoSided, false, false, None, None)
        .expect("mann_whitney_u should succeed");

    assert_relative_eq!(result.statistic, refs["statistic"], epsilon = EPSILON);
    assert_relative_eq!(result.p_value, refs["p_value"], epsilon = 1e-6);
}

#[test]
fn test_mann_whitney_u_less() {
    let refs = common::load_reference_scalars("mann_whitney.csv");
    let x = common::load_reference_vector("mw_x.csv");
    let y = common::load_reference_vector("mw_y.csv");

    let result = mann_whitney_u(&x, &y, Alternative::Less, false, false, None, None)
        .expect("mann_whitney_u should succeed");

    assert_relative_eq!(result.p_value, refs["p_value_less"], epsilon = 1e-6);
}

#[test]
fn test_mann_whitney_u_greater() {
    let refs = common::load_reference_scalars("mann_whitney.csv");
    let x = common::load_reference_vector("mw_x.csv");
    let y = common::load_reference_vector("mw_y.csv");

    let result = mann_whitney_u(&x, &y, Alternative::Greater, false, false, None, None)
        .expect("mann_whitney_u should succeed");

    assert_relative_eq!(result.p_value, refs["p_value_greater"], epsilon = 1e-6);
}

#[test]
fn test_mann_whitney_u_corrected() {
    let refs = common::load_reference_scalars("mann_whitney.csv");
    let x = common::load_reference_vector("mw_x.csv");
    let y = common::load_reference_vector("mw_y.csv");

    let result = mann_whitney_u(&x, &y, Alternative::TwoSided, true, false, None, None)
        .expect("mann_whitney_u should succeed");

    assert_relative_eq!(result.p_value, refs["p_value_corrected"], epsilon = 1e-6);
}

#[test]
fn test_mann_whitney_u_with_mu() {
    let refs = common::load_reference_scalars("mann_whitney.csv");
    let x = common::load_reference_vector("mw_x.csv");
    let y = common::load_reference_vector("mw_y.csv");

    let result = mann_whitney_u(&x, &y, Alternative::TwoSided, false, false, None, Some(0.5))
        .expect("mann_whitney_u should succeed");

    assert_relative_eq!(result.p_value, refs["p_value_mu"], epsilon = 1e-6);
}

#[test]
fn test_mann_whitney_u_with_mu_less() {
    let refs = common::load_reference_scalars("mann_whitney.csv");
    let x = common::load_reference_vector("mw_x.csv");
    let y = common::load_reference_vector("mw_y.csv");

    let result = mann_whitney_u(&x, &y, Alternative::Less, false, false, None, Some(0.5))
        .expect("mann_whitney_u should succeed");

    assert_relative_eq!(result.p_value, refs["p_value_mu_less"], epsilon = 1e-6);
}

#[test]
fn test_mann_whitney_u_with_mu_greater() {
    let refs = common::load_reference_scalars("mann_whitney.csv");
    let x = common::load_reference_vector("mw_x.csv");
    let y = common::load_reference_vector("mw_y.csv");

    let result = mann_whitney_u(&x, &y, Alternative::Greater, false, false, None, Some(0.5))
        .expect("mann_whitney_u should succeed");

    assert_relative_eq!(result.p_value, refs["p_value_mu_greater"], epsilon = 1e-6);
}

/// Regression test for issue #6: extremely small p-values used to underflow to 0.0
/// because the normal approximation computed `1 - cdf(|z|)`, which loses all
/// precision once `cdf` saturates at 1.0 (|z| > ~8.3).
/// R's `wilcox.test(..., correct = FALSE, exact = FALSE)` returns p ≈ 1.92e-25
/// for this data.
#[test]
fn test_mann_whitney_u_small_pvalue_does_not_underflow() {
    let x = vec![
        203.0, 34.0, 612.0, 277.0, 204.0, 361.0, 558.0, 386.0, 165.0, 521.0, 241.0, 702.0, 10.0,
        133.0, 207.0, 98.0, 445.0, 356.0, 389.0, 3.0, 176.0, 287.0, 297.0, 156.0, 186.0, 411.0,
        111.0, 359.0, 204.0, 373.0, 382.0, 344.0, 632.0, 311.0, 11.0, 43.0, 364.0, 182.0, 7.0,
        976.0, 10.0, 125.0, 213.0, 520.0, 854.0, 39.0, 449.0, 12.0, 275.0, 294.0, 327.0, 587.0,
        76.0, 443.0, 270.0, 484.0, 458.0, 1.0, 265.0, 106.0, 515.0, 197.0, 634.0, 320.0, 487.0,
        382.0, 236.0, 9.0, 449.0, 510.0, 130.0, 4.0, 540.0, 25.0, 219.0, 337.0, 345.0, 250.0, 7.0,
        81.0, 233.0, 309.0, 35.0, 8.0, 892.0, 61.0, 196.0, 195.0, 1341.0, 7.0, 223.0, 106.0, 3.0,
        508.0, 220.0, 337.0, 670.0, 686.0, 1385.0, 451.0, 431.0, 101.0, 702.0, 262.0, 396.0, 654.0,
        291.0, 93.0, 467.0, 284.0, 138.0, 435.0, 16.0, 263.0, 92.0, 8.0, 99.0, 230.0, 341.0, 146.0,
        639.0, 355.0, 643.0, 394.0, 198.0, 676.0, 848.0, 83.0, 147.0, 114.0, 167.0, 382.0, 434.0,
        901.0, 615.0, 1269.0, 235.0, 146.0, 357.0, 772.0, 253.0, 104.0, 1864.0, 291.0, 181.0, 13.0,
        735.0, 1819.0, 208.0, 326.0, 61.0, 280.0, 523.0, 270.0, 505.0, 264.0, 225.0, 89.0, 344.0,
        878.0, 18.0, 283.0, 102.0, 414.0, 322.0, 332.0, 15.0, 130.0, 74.0, 314.0, 378.0, 219.0,
        223.0, 52.0, 103.0, 220.0, 35.0, 742.0, 1402.0, 403.0, 884.0, 219.0, 242.0, 636.0, 109.0,
        401.0, 29.0, 437.0, 382.0, 135.0, 315.0, 80.0, 481.0, 1100.0, 648.0, 528.0, 454.0, 168.0,
        718.0, 24.0, 1097.0, 21.0, 151.0, 734.0, 486.0, 274.0, 275.0, 268.0, 477.0, 509.0, 289.0,
        333.0, 388.0, 37.0, 12.0, 168.0, 43.0, 428.0, 297.0, 405.0, 131.0, 438.0, 324.0, 259.0,
        3.0, 808.0, 29.0, 357.0, 737.0, 356.0, 1680.0, 333.0, 76.0, 497.0, 396.0, 361.0, 6.0,
        448.0, 364.0, 19.0, 31.0, 521.0, 383.0, 407.0, 12.0, 702.0, 258.0, 188.0, 293.0, 225.0,
        217.0, 92.0, 1961.0, 439.0, 245.0, 629.0, 143.0, 292.0, 30.0, 184.0, 315.0, 197.0, 157.0,
        130.0, 1077.0, 220.0, 315.0, 694.0, 1260.0, 165.0, 98.0, 387.0, 216.0, 145.0, 910.0, 79.0,
        268.0, 221.0, 99.0, 3.0, 92.0, 200.0, 740.0, 524.0, 412.0, 120.0, 832.0, 68.0, 621.0,
        521.0, 686.0, 1148.0, 255.0, 284.0, 172.0, 646.0, 647.0, 1219.0, 204.0, 201.0, 24.0, 120.0,
        655.0, 52.0, 211.0, 630.0, 118.0, 174.0, 559.0, 447.0, 640.0, 50.0, 405.0, 53.0, 7.0,
        256.0, 174.0, 184.0, 177.0, 316.0, 721.0, 664.0, 82.0, 312.0, 625.0, 71.0, 457.0, 371.0,
        99.0, 413.0, 887.0, 298.0, 370.0, 175.0, 223.0, 251.0, 177.0, 56.0, 671.0, 149.0, 403.0,
        122.0, 138.0, 75.0, 80.0, 11.0, 354.0, 327.0, 234.0, 642.0, 1376.0, 440.0, 3.0, 720.0,
        311.0, 1356.0, 374.0, 116.0, 881.0, 159.0, 364.0, 897.0, 467.0, 708.0, 6.0, 858.0, 35.0,
        576.0, 311.0, 114.0, 270.0, 813.0, 117.0, 613.0, 434.0, 426.0, 80.0, 822.0, 128.0, 324.0,
        44.0, 114.0, 387.0, 667.0, 517.0, 90.0, 303.0, 5.0, 20.0, 223.0, 424.0, 263.0, 419.0, 74.0,
        388.0, 946.0, 35.0, 626.0, 156.0, 601.0, 102.0, 4.0,
    ];
    let y = vec![
        122.0, 9.0, 13.0, 5.0, 23.0, 27.0, 7.0, 45.0, 10.0, 62.0, 109.0, 43.0, 8.0, 5.0, 78.0, 3.0,
        14.0, 19.0, 45.0, 10.0, 13.0, 23.0, 5.0, 16.0, 4.0, 2.0, 9.0, 6.0, 21.0, 1.0, 2.0, 62.0,
        42.0, 21.0, 131.0, 6.0, 5.0, 2.0, 22.0, 881.0, 3.0, 4.0, 16.0, 150.0, 12.0, 77.0, 12.0,
        5.0, 4.0, 7.0, 7.0, 1.0, 16.0, 44.0, 64.0, 8.0, 9.0, 1.0, 27.0, 5.0, 13.0, 122.0, 215.0,
    ];

    let result = mann_whitney_u(&x, &y, Alternative::TwoSided, false, false, None, None)
        .expect("mann_whitney_u should succeed");

    // Before the fix this returned exactly 0.0.
    assert!(
        result.p_value > 0.0,
        "p_value underflowed to {} (regression of issue #6)",
        result.p_value
    );
    // Sanity-check: should still be vanishingly small.
    assert!(
        result.p_value < 1e-15,
        "p_value too large: {} (effect should be highly significant)",
        result.p_value
    );
}

#[test]
fn test_mann_whitney_empty_x_returns_error() {
    let empty: Vec<f64> = vec![];
    let y = vec![1.0, 2.0, 3.0];
    assert!(mann_whitney_u(&empty, &y, Alternative::TwoSided, false, false, None, None).is_err());
}

#[test]
fn test_mann_whitney_empty_y_returns_error() {
    let x = vec![1.0, 2.0, 3.0];
    let empty: Vec<f64> = vec![];
    assert!(mann_whitney_u(&x, &empty, Alternative::TwoSided, false, false, None, None).is_err());
}

// ============================================
// Wilcoxon Signed-Rank Test
// ============================================

#[test]
fn test_wilcoxon_signed_rank_two_sided() {
    let refs = common::load_reference_scalars("wilcoxon_signed_rank.csv");
    let x = common::load_reference_vector("wsr_x.csv");
    let y = common::load_reference_vector("wsr_y.csv");

    let result = wilcoxon_signed_rank(&x, &y, Alternative::TwoSided, false, false, None, None)
        .expect("wilcoxon_signed_rank should succeed");

    assert_relative_eq!(result.statistic, refs["statistic"], epsilon = EPSILON);
    assert_relative_eq!(result.p_value, refs["p_value"], epsilon = 1e-6);
}

#[test]
fn test_wilcoxon_signed_rank_less() {
    let refs = common::load_reference_scalars("wilcoxon_signed_rank.csv");
    let x = common::load_reference_vector("wsr_x.csv");
    let y = common::load_reference_vector("wsr_y.csv");

    let result = wilcoxon_signed_rank(&x, &y, Alternative::Less, false, false, None, None)
        .expect("wilcoxon_signed_rank should succeed");

    assert_relative_eq!(result.p_value, refs["p_value_less"], epsilon = 1e-6);
}

#[test]
fn test_wilcoxon_signed_rank_greater() {
    let refs = common::load_reference_scalars("wilcoxon_signed_rank.csv");
    let x = common::load_reference_vector("wsr_x.csv");
    let y = common::load_reference_vector("wsr_y.csv");

    let result = wilcoxon_signed_rank(&x, &y, Alternative::Greater, false, false, None, None)
        .expect("wilcoxon_signed_rank should succeed");

    assert_relative_eq!(result.p_value, refs["p_value_greater"], epsilon = 1e-6);
}

#[test]
fn test_wilcoxon_signed_rank_corrected() {
    let refs = common::load_reference_scalars("wilcoxon_signed_rank.csv");
    let x = common::load_reference_vector("wsr_x.csv");
    let y = common::load_reference_vector("wsr_y.csv");

    let result = wilcoxon_signed_rank(&x, &y, Alternative::TwoSided, true, false, None, None)
        .expect("wilcoxon_signed_rank should succeed");

    assert_relative_eq!(result.p_value, refs["p_value_corrected"], epsilon = 1e-6);
}

#[test]
fn test_wilcoxon_signed_rank_with_mu() {
    let refs = common::load_reference_scalars("wilcoxon_signed_rank.csv");
    let x = common::load_reference_vector("wsr_x.csv");
    let y = common::load_reference_vector("wsr_y.csv");

    let result = wilcoxon_signed_rank(&x, &y, Alternative::TwoSided, false, false, None, Some(0.3))
        .expect("wilcoxon_signed_rank should succeed");

    assert_relative_eq!(result.p_value, refs["p_value_mu"], epsilon = 1e-6);
}

#[test]
fn test_wilcoxon_signed_rank_with_mu_less() {
    let refs = common::load_reference_scalars("wilcoxon_signed_rank.csv");
    let x = common::load_reference_vector("wsr_x.csv");
    let y = common::load_reference_vector("wsr_y.csv");

    let result = wilcoxon_signed_rank(&x, &y, Alternative::Less, false, false, None, Some(0.3))
        .expect("wilcoxon_signed_rank should succeed");

    assert_relative_eq!(result.p_value, refs["p_value_mu_less"], epsilon = 1e-6);
}

#[test]
fn test_wilcoxon_signed_rank_with_mu_greater() {
    let refs = common::load_reference_scalars("wilcoxon_signed_rank.csv");
    let x = common::load_reference_vector("wsr_x.csv");
    let y = common::load_reference_vector("wsr_y.csv");

    let result = wilcoxon_signed_rank(&x, &y, Alternative::Greater, false, false, None, Some(0.3))
        .expect("wilcoxon_signed_rank should succeed");

    assert_relative_eq!(result.p_value, refs["p_value_mu_greater"], epsilon = 1e-6);
}

#[test]
fn test_wilcoxon_signed_rank_unequal_length_returns_error() {
    let x = vec![1.0, 2.0, 3.0];
    let y = vec![1.0, 2.0];
    assert!(wilcoxon_signed_rank(&x, &y, Alternative::TwoSided, false, false, None, None).is_err());
}

#[test]
fn test_wilcoxon_signed_rank_empty_returns_error() {
    let empty: Vec<f64> = vec![];
    let y = vec![1.0, 2.0, 3.0];
    assert!(
        wilcoxon_signed_rank(&empty, &y, Alternative::TwoSided, false, false, None, None).is_err()
    );
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
        brunner_munzel(&x, &y, Alternative::TwoSided, None).expect("brunner_munzel should succeed");

    assert_relative_eq!(result.statistic, refs["statistic"], epsilon = 1e-6);
    assert_relative_eq!(result.df, refs["df"], epsilon = 1e-6);
    assert_relative_eq!(result.p_value, refs["p_value"], epsilon = 1e-6);
    assert_relative_eq!(result.estimate, refs["estimate"], epsilon = 1e-6);
}

#[test]
fn test_brunner_munzel_less() {
    let refs = common::load_reference_scalars("brunner_munzel.csv");
    let x = common::load_reference_vector("bm_x.csv");
    let y = common::load_reference_vector("bm_y.csv");

    let result =
        brunner_munzel(&x, &y, Alternative::Less, None).expect("brunner_munzel should succeed");

    assert_relative_eq!(result.p_value, refs["p_value_less"], epsilon = 1e-6);
}

#[test]
fn test_brunner_munzel_greater() {
    let refs = common::load_reference_scalars("brunner_munzel.csv");
    let x = common::load_reference_vector("bm_x.csv");
    let y = common::load_reference_vector("bm_y.csv");

    let result =
        brunner_munzel(&x, &y, Alternative::Greater, None).expect("brunner_munzel should succeed");

    assert_relative_eq!(result.p_value, refs["p_value_greater"], epsilon = 1e-6);
}

#[test]
fn test_brunner_munzel_confidence_interval_95() {
    let refs = common::load_reference_scalars("brunner_munzel.csv");
    let x = common::load_reference_vector("bm_x.csv");
    let y = common::load_reference_vector("bm_y.csv");

    let result = brunner_munzel(&x, &y, Alternative::TwoSided, Some(0.05))
        .expect("brunner_munzel should succeed");

    let ci = result.conf_int.expect("conf_int should be present");

    assert_relative_eq!(ci.lower, refs["conf_low_95"], epsilon = 1e-6);
    assert_relative_eq!(ci.upper, refs["conf_high_95"], epsilon = 1e-6);
    assert_relative_eq!(ci.conf_level, 0.95, epsilon = EPSILON);
}

#[test]
fn test_brunner_munzel_confidence_interval_90() {
    let refs = common::load_reference_scalars("brunner_munzel.csv");
    let x = common::load_reference_vector("bm_x.csv");
    let y = common::load_reference_vector("bm_y.csv");

    let result = brunner_munzel(&x, &y, Alternative::TwoSided, Some(0.10))
        .expect("brunner_munzel should succeed");

    let ci = result.conf_int.expect("conf_int should be present");

    assert_relative_eq!(ci.lower, refs["conf_low_90"], epsilon = 1e-6);
    assert_relative_eq!(ci.upper, refs["conf_high_90"], epsilon = 1e-6);
    assert_relative_eq!(ci.conf_level, 0.90, epsilon = EPSILON);
}

#[test]
fn test_brunner_munzel_confidence_interval_99() {
    let refs = common::load_reference_scalars("brunner_munzel.csv");
    let x = common::load_reference_vector("bm_x.csv");
    let y = common::load_reference_vector("bm_y.csv");

    let result = brunner_munzel(&x, &y, Alternative::TwoSided, Some(0.01))
        .expect("brunner_munzel should succeed");

    let ci = result.conf_int.expect("conf_int should be present");

    assert_relative_eq!(ci.lower, refs["conf_low_99"], epsilon = 1e-6);
    assert_relative_eq!(ci.upper, refs["conf_high_99"], epsilon = 1e-6);
    assert_relative_eq!(ci.conf_level, 0.99, epsilon = EPSILON);
}

#[test]
fn test_brunner_munzel_empty_x_returns_error() {
    let empty: Vec<f64> = vec![];
    let y = vec![1.0, 2.0, 3.0];
    assert!(brunner_munzel(&empty, &y, Alternative::TwoSided, None).is_err());
}

#[test]
fn test_brunner_munzel_empty_y_returns_error() {
    let x = vec![1.0, 2.0, 3.0];
    let empty: Vec<f64> = vec![];
    assert!(brunner_munzel(&x, &empty, Alternative::TwoSided, None).is_err());
}

// ============================================
// Mann-Whitney U Exact Tests
// ============================================

#[test]
fn test_mann_whitney_u_exact_two_sided() {
    let refs = common::load_reference_scalars("mann_whitney_exact.csv");
    let x = common::load_reference_vector("mw_x_exact.csv");
    let y = common::load_reference_vector("mw_y_exact.csv");

    let result = mann_whitney_u(&x, &y, Alternative::TwoSided, false, true, None, None)
        .expect("mann_whitney_u exact should succeed");

    assert_relative_eq!(result.statistic, refs["statistic"], epsilon = EPSILON);
    assert_relative_eq!(result.p_value, refs["p_value"], epsilon = 1e-6);
}

#[test]
fn test_mann_whitney_u_exact_less() {
    let refs = common::load_reference_scalars("mann_whitney_exact.csv");
    let x = common::load_reference_vector("mw_x_exact.csv");
    let y = common::load_reference_vector("mw_y_exact.csv");

    let result = mann_whitney_u(&x, &y, Alternative::Less, false, true, None, None)
        .expect("mann_whitney_u exact should succeed");

    assert_relative_eq!(result.p_value, refs["p_value_less"], epsilon = 1e-6);
}

#[test]
fn test_mann_whitney_u_exact_greater() {
    let refs = common::load_reference_scalars("mann_whitney_exact.csv");
    let x = common::load_reference_vector("mw_x_exact.csv");
    let y = common::load_reference_vector("mw_y_exact.csv");

    let result = mann_whitney_u(&x, &y, Alternative::Greater, false, true, None, None)
        .expect("mann_whitney_u exact should succeed");

    assert_relative_eq!(result.p_value, refs["p_value_greater"], epsilon = 1e-6);
}

#[test]
fn test_mann_whitney_u_confidence_interval() {
    let refs = common::load_reference_scalars("mann_whitney_exact.csv");
    let x = common::load_reference_vector("mw_x_exact.csv");
    let y = common::load_reference_vector("mw_y_exact.csv");

    let result = mann_whitney_u(&x, &y, Alternative::TwoSided, false, true, Some(0.95), None)
        .expect("mann_whitney_u with CI should succeed");

    assert!(result.estimate.is_some());
    assert!(result.conf_int.is_some());

    let estimate = result.estimate.unwrap();
    let ci = result.conf_int.unwrap();

    assert_relative_eq!(estimate, refs["estimate"], epsilon = 1e-6);
    assert_relative_eq!(ci.lower, refs["conf_low_95"], epsilon = 1e-6);
    assert_relative_eq!(ci.upper, refs["conf_high_95"], epsilon = 1e-6);
    assert_relative_eq!(ci.conf_level, 0.95, epsilon = EPSILON);
}

#[test]
fn test_mann_whitney_u_confidence_interval_90() {
    let refs = common::load_reference_scalars("mann_whitney_exact.csv");
    let x = common::load_reference_vector("mw_x_exact.csv");
    let y = common::load_reference_vector("mw_y_exact.csv");

    let result = mann_whitney_u(&x, &y, Alternative::TwoSided, false, true, Some(0.90), None)
        .expect("mann_whitney_u with CI should succeed");

    let ci = result.conf_int.unwrap();

    assert_relative_eq!(ci.lower, refs["conf_low_90"], epsilon = 1e-6);
    assert_relative_eq!(ci.upper, refs["conf_high_90"], epsilon = 1e-6);
    assert_relative_eq!(ci.conf_level, 0.90, epsilon = EPSILON);
}

// ============================================
// Wilcoxon Signed-Rank Exact Tests
// ============================================

#[test]
fn test_wilcoxon_signed_rank_exact_two_sided() {
    let refs = common::load_reference_scalars("wilcoxon_signed_rank_exact.csv");
    let x = common::load_reference_vector("wsr_x_exact.csv");
    let y = common::load_reference_vector("wsr_y_exact.csv");

    let result = wilcoxon_signed_rank(&x, &y, Alternative::TwoSided, false, true, None, None)
        .expect("wilcoxon_signed_rank exact should succeed");

    assert_relative_eq!(result.statistic, refs["statistic"], epsilon = EPSILON);
    assert_relative_eq!(result.p_value, refs["p_value"], epsilon = 1e-6);
}

#[test]
fn test_wilcoxon_signed_rank_exact_less() {
    let refs = common::load_reference_scalars("wilcoxon_signed_rank_exact.csv");
    let x = common::load_reference_vector("wsr_x_exact.csv");
    let y = common::load_reference_vector("wsr_y_exact.csv");

    let result = wilcoxon_signed_rank(&x, &y, Alternative::Less, false, true, None, None)
        .expect("wilcoxon_signed_rank exact should succeed");

    assert_relative_eq!(result.p_value, refs["p_value_less"], epsilon = 1e-6);
}

#[test]
fn test_wilcoxon_signed_rank_exact_greater() {
    let refs = common::load_reference_scalars("wilcoxon_signed_rank_exact.csv");
    let x = common::load_reference_vector("wsr_x_exact.csv");
    let y = common::load_reference_vector("wsr_y_exact.csv");

    let result = wilcoxon_signed_rank(&x, &y, Alternative::Greater, false, true, None, None)
        .expect("wilcoxon_signed_rank exact should succeed");

    assert_relative_eq!(result.p_value, refs["p_value_greater"], epsilon = 1e-6);
}

#[test]
fn test_wilcoxon_signed_rank_confidence_interval() {
    let refs = common::load_reference_scalars("wilcoxon_signed_rank_exact.csv");
    let x = common::load_reference_vector("wsr_x_exact.csv");
    let y = common::load_reference_vector("wsr_y_exact.csv");

    let result = wilcoxon_signed_rank(&x, &y, Alternative::TwoSided, false, true, Some(0.95), None)
        .expect("wilcoxon_signed_rank with CI should succeed");

    assert!(result.estimate.is_some());
    assert!(result.conf_int.is_some());

    let estimate = result.estimate.unwrap();
    let ci = result.conf_int.unwrap();

    assert_relative_eq!(estimate, refs["estimate"], epsilon = 1e-6);
    assert_relative_eq!(ci.lower, refs["conf_low_95"], epsilon = 1e-6);
    assert_relative_eq!(ci.upper, refs["conf_high_95"], epsilon = 1e-6);
    assert_relative_eq!(ci.conf_level, 0.95, epsilon = EPSILON);
}

#[test]
fn test_wilcoxon_signed_rank_confidence_interval_90() {
    let refs = common::load_reference_scalars("wilcoxon_signed_rank_exact.csv");
    let x = common::load_reference_vector("wsr_x_exact.csv");
    let y = common::load_reference_vector("wsr_y_exact.csv");

    let result = wilcoxon_signed_rank(&x, &y, Alternative::TwoSided, false, true, Some(0.90), None)
        .expect("wilcoxon_signed_rank with CI should succeed");

    let ci = result.conf_int.unwrap();

    assert_relative_eq!(ci.lower, refs["conf_low_90"], epsilon = 1e-6);
    assert_relative_eq!(ci.upper, refs["conf_high_90"], epsilon = 1e-6);
    assert_relative_eq!(ci.conf_level, 0.90, epsilon = EPSILON);
}

// ============================================
// Mann-Whitney U Result Fields
// ============================================

#[test]
fn test_mann_whitney_u_result_contains_null_value() {
    let x = common::load_reference_vector("mw_x.csv");
    let y = common::load_reference_vector("mw_y.csv");

    let result_mu0 = mann_whitney_u(&x, &y, Alternative::TwoSided, false, false, None, None)
        .expect("mann_whitney_u should succeed");

    let result_mu05 = mann_whitney_u(&x, &y, Alternative::TwoSided, false, false, None, Some(0.5))
        .expect("mann_whitney_u should succeed");

    assert_relative_eq!(result_mu0.null_value, 0.0, epsilon = EPSILON);
    assert_relative_eq!(result_mu05.null_value, 0.5, epsilon = EPSILON);
}

// ============================================
// Wilcoxon Signed-Rank Result Fields
// ============================================

#[test]
fn test_wilcoxon_signed_rank_result_contains_null_value() {
    let x = common::load_reference_vector("wsr_x.csv");
    let y = common::load_reference_vector("wsr_y.csv");

    let result_mu0 = wilcoxon_signed_rank(&x, &y, Alternative::TwoSided, false, false, None, None)
        .expect("wilcoxon_signed_rank should succeed");

    let result_mu03 =
        wilcoxon_signed_rank(&x, &y, Alternative::TwoSided, false, false, None, Some(0.3))
            .expect("wilcoxon_signed_rank should succeed");

    assert_relative_eq!(result_mu0.null_value, 0.0, epsilon = EPSILON);
    assert_relative_eq!(result_mu03.null_value, 0.3, epsilon = EPSILON);
}
