//! Regression tests for GitHub issues #7-#15, one test per issue, using the
//! reproductions from the issue reports. Reference values: R 4.6.1
//! (cross-checked with scipy 1.17.0 where available).
#![allow(clippy::excessive_precision, clippy::approx_constant)]

use anofox_statistics::*;

fn close(actual: f64, expected: f64, tol: f64) -> bool {
    (actual - expected).abs() <= tol
}

/// #7: shapiro.test(...)
#[test]
fn issue_07_shapiro_wilk_small_n_matches_r() {
    let r = shapiro_wilk(&[4.1, 5.3, 3.8, 6.9, 5.0, 4.4, 9.2, 5.7, 4.9, 6.1]).unwrap();
    assert!(
        close(r.statistic, 0.88790406283629908, 1e-6),
        "{}",
        r.statistic
    );
    assert!(close(r.p_value, 0.16058545199963783, 1e-6), "{}", r.p_value);
    let r = shapiro_wilk(&[
        98.059831,
        100.258934,
        101.1010067,
        99.33949313,
        102.5085549,
        99.77152429,
        98.47658593,
        101.3298538,
        98.84786198,
    ])
    .unwrap();
    assert!(
        close(r.statistic, 0.96512149278140513, 1e-6),
        "{}",
        r.statistic
    );
    assert!(close(r.p_value, 0.8500927903451192, 1e-6), "{}", r.p_value);
    let r = shapiro_wilk(&[1.2, 3.4, 2.2, 5.9]).unwrap();
    assert!(close(r.p_value, 0.74255791273476768, 1e-6), "{}", r.p_value);
}

/// #8: cor.test(kx, ky, method = "kendall", exact = FALSE)
#[test]
fn issue_08_kendall_tie_corrected_variance() {
    let x = [1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.0];
    let y = [2.0, 1.0, 3.0, 3.0, 5.0, 4.0, 6.0, 6.0, 5.0, 8.0, 7.0, 9.0];
    let r = kendall(&x, &y, KendallVariant::TauB).unwrap();
    assert!(close(r.estimate, 0.80655653082637868, 1e-12));
    assert!(
        close(r.statistic, 3.4987518043922514, 1e-8),
        "{}",
        r.statistic
    );
    assert!(
        close(r.p_value, 0.00046744148056981807, 1e-10),
        "{}",
        r.p_value
    );
}

/// #9: binom.test(13, 20), binom.test(4500, 10000), binom.test(13, 20, alternative = "less")
#[test]
fn issue_09_binom_test_matches_r() {
    let r = binom_test(13, 20, 0.5, Alternative::TwoSided).unwrap();
    assert!(close(r.p_value, 0.26317596435546875, 1e-12));
    assert!(close(r.conf_int_lower, 0.40781146546717190, 1e-8));
    assert!(close(r.conf_int_upper, 0.84609079521545882, 1e-8));
    let r = binom_test(4500, 10000, 0.5, Alternative::TwoSided).unwrap();
    assert!(
        (r.p_value / 1.5510640568246068e-23 - 1.0).abs() < 1e-6,
        "{}",
        r.p_value
    );
    let r = binom_test(13, 20, 0.5, Alternative::Less).unwrap();
    assert_eq!(r.conf_int_lower, 0.0);
    assert!(close(r.conf_int_upper, 0.82268908242555083, 1e-8));
}

/// #10: prop.test(...)$conf.int
#[test]
fn issue_10_prop_test_intervals_match_r() {
    let r = prop_test_one(13, 20, 0.5, Alternative::Greater).unwrap();
    assert!(close(r.conf_int_lower, 0.46651268884847175, 1e-8));
    assert_eq!(r.conf_int_upper, 1.0);
    let r = prop_test_two([18, 11], [30, 28], Alternative::TwoSided, true).unwrap();
    assert!(close(r.conf_int_lower, -0.079284640161607245, 1e-8));
    assert!(close(r.conf_int_upper, 0.493570354447321502, 1e-8));
    let r = prop_test_two([18, 11], [30, 28], Alternative::TwoSided, false).unwrap();
    assert!(close(r.conf_int_lower, -0.044760830637797733, 1e-8));
    assert!(close(r.conf_int_upper, 0.459046544923511990, 1e-8));
    // prop.test(13, 20, correct = FALSE, conf.level = 0.9)$conf.int
    let r = prop_test_one_with_conf_level(13, 20, 0.5, Alternative::TwoSided, 0.9).unwrap();
    assert!(close(r.conf_int_lower, 0.4665126888484717, 1e-8));
    assert!(close(r.conf_int_upper, 0.79773995994323899, 1e-8));
}

/// #11: f <- fisher.test(matrix(c(18, 9, 7, 16), 2))
#[test]
fn issue_11_fisher_exact_conditional_matches_r() {
    let r = fisher_exact_conditional(&[[18, 7], [9, 16]], Alternative::TwoSided, 0.95).unwrap();
    assert!(close(r.p_value, 0.022241295691722542, 1e-9));
    assert!(
        close(r.odds_ratio, 4.4215633454100463, 1e-3),
        "{}",
        r.odds_ratio
    );
    assert!(
        close(r.conf_int_lower, 1.2001230942188943, 1e-3),
        "{}",
        r.conf_int_lower
    );
    assert!(
        close(r.conf_int_upper, 18.034801377448705, 2e-2),
        "{}",
        r.conf_int_upper
    );
    // fisher_exact keeps the sample odds ratio, with the exact p-value
    let r = fisher_exact(&[[18, 7], [9, 16]], Alternative::TwoSided).unwrap();
    assert!(close(r.p_value, 0.022241295691722542, 1e-9));
    assert!(close(r.odds_ratio, 4.5714285714285712, 1e-12));
}

/// #12: wilcox.test(x, y, exact = TRUE): W = 58, p = 0.3153781203316808
#[test]
fn issue_12_mann_whitney_exact_two_sided_upper_tail() {
    let x = [1.83, 0.50, 1.62, 2.48, 1.68, 1.88, 1.55, 3.06, 1.30];
    let y = [0.878, 0.647, 0.598, 2.05, 1.06, 1.29, 1.07, 3.14, 1.28, 4.1];
    let r = mann_whitney_u(&x, &y, Alternative::TwoSided, true, true, None, None).unwrap();
    assert_eq!(r.statistic, 58.0);
    assert!(
        close(r.p_value, 0.31537812033168078, 1e-12),
        "{}",
        r.p_value
    );
    let r = mann_whitney_u(&y, &x, Alternative::TwoSided, true, true, None, None).unwrap();
    assert!(
        close(r.p_value, 0.31537812033168078, 1e-12),
        "{}",
        r.p_value
    );
}

/// #13: wilcox.test(c(1, 4), c(2, 3), exact = FALSE)$p.value == 1, etc.
#[test]
fn issue_13_rank_tests_centre_and_all_tied() {
    let r = mann_whitney_u(
        &[1.0, 4.0],
        &[2.0, 3.0],
        Alternative::TwoSided,
        true,
        false,
        None,
        None,
    )
    .unwrap();
    assert_eq!(r.p_value, 1.0);
    let r = wilcoxon_signed_rank(
        &[1.0, 2.0, 3.0, 4.0],
        &[4.0, 3.0, 2.0, 1.0],
        Alternative::TwoSided,
        true,
        false,
        None,
        None,
    )
    .unwrap();
    assert_eq!(r.p_value, 1.0);
    for alt in [
        Alternative::TwoSided,
        Alternative::Less,
        Alternative::Greater,
    ] {
        let r = mann_whitney_u(&[5.0; 6], &[5.0; 7], alt, true, false, None, None).unwrap();
        assert_eq!(r.p_value, 1.0);
    }
}

/// #14: cor.test(1:5, rep(5, 5)) -> NA; scipy.stats.pearsonr -> (nan, nan)
#[test]
fn issue_14_pearson_constant_column_does_not_panic() {
    let res = std::panic::catch_unwind(|| pearson(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5.0; 5], None));
    let c = res.expect("pearson() panicked").unwrap();
    assert!(c.estimate.is_nan() && c.p_value.is_nan());
    let c = pearson(&[2.0; 5], &[5.0; 5], Some(0.95)).unwrap();
    assert!(c.estimate.is_nan() && c.conf_int.is_none());
}

/// #15: Cohen's d and rank-biserial r
#[test]
fn issue_15_effect_sizes() {
    let x = [5.1, 4.9, 6.2, 5.8, 6.05, 5.5, 5.3, 6.1];
    let y = [6.5, 7.1, 6.8, 7.4, 6.0, 7.9, 6.6, 7.2, 6.9, 7.05];
    // sp <- sqrt(((8-1)*var(x) + (10-1)*var(y)) / (8+10-2)); (mean(x) - mean(y)) / sp
    let d = cohens_d(&x, &y, TTestKind::Student, 0.0).unwrap();
    assert!(close(d, -2.606655378290357, 1e-12), "{d}");
    // (mean(x) - mean(y)) / sqrt((var(x) + var(y)) / 2)
    let d = cohens_d(&x, &y, TTestKind::Welch, 0.0).unwrap();
    assert!(close(d, -2.6164932281884563, 1e-12), "{d}");
    // d <- x - y[1:8]; mean(d) / sd(d)
    let d = cohens_d(&x, &y[..8], TTestKind::Paired, 0.0).unwrap();
    assert!(close(d, -1.6503380487140236, 1e-12), "{d}");
    // W <- wilcox.test(x, y, exact = FALSE)$statistic; 1 - 2*W/(8*10)
    let mw = mann_whitney_u(&x, &y, Alternative::TwoSided, true, false, None, None).unwrap();
    assert!(close(
        rank_biserial_from_u(mw.statistic, 8, 10),
        0.925,
        1e-12
    ));
    assert!(close(rank_biserial(&x, &y, None).unwrap(), 0.925, 1e-12));
}

/// #14 (hardening): other tests whose statistic is 0/0 on constant data must
/// not panic inside statrs either; they report a NaN p-value.
#[test]
fn issue_14_constant_inputs_do_not_panic() {
    let c = [3.0; 6];
    let c2 = [3.0; 5];
    for k in [TTestKind::Welch, TTestKind::Student, TTestKind::Paired] {
        let r = std::panic::catch_unwind(|| {
            t_test(&c, &c, k, Alternative::TwoSided, 0.0, Some(0.95)).unwrap()
        })
        .expect("t_test panicked");
        assert!(r.p_value.is_nan(), "{k:?}");
    }
    let r =
        std::panic::catch_unwind(|| one_way_anova(&[&c[..], &c2[..]], AnovaKind::Fisher).unwrap())
            .expect("one_way_anova panicked");
    assert!(r.p_value.is_nan());
    let r = std::panic::catch_unwind(|| brown_forsythe(&[&c[..], &c2[..]]).unwrap())
        .expect("brown_forsythe panicked");
    assert!(r.p_value.is_nan());
    let r = std::panic::catch_unwind(|| {
        yuen_test(&c, &c, 0.2, Alternative::TwoSided, Some(0.95)).unwrap()
    })
    .expect("yuen_test panicked");
    assert!(r.p_value.is_nan());
}
