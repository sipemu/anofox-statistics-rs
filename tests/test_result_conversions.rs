//! Conversions of the existing result types into the unified `TestResult`.

use anofox_statistics::{
    brown_forsythe, chisq_test, fisher_exact_with_conf_level, jarque_bera, kruskal_wallis,
    mann_whitney_u, one_way_anova, pearson, shapiro_wilk, t_test, tost_t_test_two_sample,
    two_way_anova, Alternative, AnovaKind, EquivalenceBounds, TTestKind, TestResult,
};

const A: [f64; 8] = [5.1, 4.9, 5.6, 5.8, 6.0, 5.5, 5.3, 6.2];
const B: [f64; 8] = [4.1, 4.5, 4.9, 5.0, 4.4, 4.7, 4.2, 5.1];
const C: [f64; 8] = [6.1, 5.9, 6.6, 6.4, 7.0, 6.5, 6.8, 6.2];

#[test]
fn t_test_conversion() {
    let r = t_test(&A, &B, TTestKind::Welch, Alternative::Less, 0.0, Some(0.9)).unwrap();
    let t = TestResult::from(&r)
        .with_alternative(Alternative::Less)
        .with_n(16);
    assert_eq!(t.statistic, Some(r.statistic));
    assert_eq!(t.df1, Some(r.df));
    assert_eq!(t.df2, None);
    assert_eq!(t.p_value, Some(r.p_value));
    assert_eq!(t.estimate, Some(r.mean_x - r.mean_y.unwrap()));
    let ci = r.conf_int.as_ref().unwrap();
    assert_eq!((t.conf_low, t.conf_high), (Some(ci.lower), Some(ci.upper)));
    assert_eq!(t.conf_level, Some(0.9));
    assert_eq!(t.alternative_name(), Some("less"));
    assert_eq!(t.n, Some(16));
    // owned conversion gives the same
    let owned: TestResult = r.clone().into();
    assert_eq!(owned.statistic, Some(r.statistic));
}

#[test]
fn anova_conversions() {
    let r = one_way_anova(&[&A, &B, &C], AnovaKind::Fisher).unwrap();
    let t = TestResult::from(&r);
    assert_eq!((t.df1, t.df2), (Some(2.0), Some(21.0)));
    assert_eq!(t.n, Some(24));
    assert_eq!(t.effect_name.as_deref(), Some("eta_squared"));
    let eta = r.ss_between.unwrap() / r.ss_total.unwrap();
    assert!((t.effect_size.unwrap() - eta).abs() < 1e-15);

    let w = one_way_anova(&[&A, &B, &C], AnovaKind::Welch).unwrap();
    assert_eq!(TestResult::from(&w).effect_size, None);

    let lev = TestResult::from(&brown_forsythe(&[&A, &B, &C]).unwrap());
    assert_eq!((lev.df1, lev.df2), (Some(2.0), Some(21.0)));

    let kw = TestResult::from(&kruskal_wallis(&[&A, &B, &C]).unwrap());
    assert_eq!(kw.method, "Kruskal-Wallis rank sum test");
    assert_eq!(kw.df1, Some(2.0));

    let values: Vec<f64> = A.iter().chain(&B).chain(&C).chain(&A).copied().collect();
    let fa: Vec<usize> = (0..32).map(|i| i / 16).collect();
    let fb: Vec<usize> = (0..32).map(|i| (i / 4) % 2).collect();
    let tw = two_way_anova(&values, &fa, &fb).unwrap();
    let rows = tw.to_test_results();
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0].statistic, tw.factor_a.f_statistic);
    assert_eq!(rows[2].df2, Some(tw.residual.df));
}

#[test]
fn other_conversions() {
    let mw = mann_whitney_u(&A, &B, Alternative::TwoSided, true, false, Some(0.95), None).unwrap();
    let t = TestResult::from(&mw);
    assert_eq!(t.estimate, mw.estimate);
    assert_eq!(t.conf_level, Some(0.95));

    let cor = TestResult::from(&pearson(&A, &C, Some(0.95)).unwrap());
    assert_eq!(cor.effect_name.as_deref(), Some("r"));
    assert_eq!(cor.n, Some(8));
    assert_eq!(cor.effect_size, cor.estimate);

    let sw = TestResult::from(&shapiro_wilk(&A).unwrap());
    assert!(sw.statistic.is_some() && sw.p_value.is_some() && sw.df1.is_none());

    let jb = TestResult::from(&jarque_bera(&A).unwrap());
    assert_eq!(jb.n, Some(8));

    let chi = TestResult::from(&chisq_test(&[vec![10, 20], vec![30, 15]], true).unwrap());
    assert_eq!(chi.df1, Some(1.0));

    let f = fisher_exact_with_conf_level(&[[10, 3], [2, 15]], Alternative::Greater, 0.95).unwrap();
    let ft = TestResult::from(&f);
    assert_eq!(ft.alternative, Some(Alternative::Greater));
    assert_eq!(ft.estimate, Some(f.odds_ratio));

    let tost = tost_t_test_two_sample(
        &A,
        &B,
        &EquivalenceBounds::Raw {
            lower: -2.0,
            upper: 2.0,
        },
        0.05,
        false,
    )
    .unwrap();
    let tt = TestResult::from(&tost);
    assert_eq!(tt.p_value, Some(tost.tost_p_value));
    assert!((tt.conf_level.unwrap() - 0.9).abs() < 1e-12);
}
