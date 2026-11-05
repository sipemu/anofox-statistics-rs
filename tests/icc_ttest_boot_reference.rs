//! ICC (all six Shrout-Fleiss types, psych::ICC), t-tests with one-sided
//! confidence intervals (t.test) and the block bootstrap of the mean
//! (boot::tsboot, Monte-Carlo tolerance) against R. Reference data:
//! `R/generate_icc_ttest_boot_refs.R` -> `R/data/icc_ttest_boot_reference.csv`.

use anofox_statistics::correlation::{icc, ICCType};
use anofox_statistics::{
    bootstrap_mean_ci, bootstrap_mean_ci_with_options, cohens_d_one_sample, t_test,
    t_test_one_sample, Alternative, BootstrapCIOptions, BootstrapMethod, TTestKind, TTestResult,
};
use std::collections::HashMap;

type Refs = HashMap<(String, String), Vec<f64>>;

fn refs() -> Refs {
    let p = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/R/data/icc_ttest_boot_reference.csv"
    );
    let mut m = HashMap::new();
    for r in csv::Reader::from_path(p).expect("reference file").records() {
        let r = r.unwrap();
        let v = r[2]
            .split_whitespace()
            .map(|s| match s {
                "Inf" => f64::INFINITY,
                "-Inf" => f64::NEG_INFINITY,
                s => s.parse().unwrap(),
            })
            .collect();
        m.insert((r[0].to_string(), r[1].to_string()), v);
    }
    m
}

fn get<'a>(m: &'a Refs, case: &str, key: &str) -> &'a [f64] {
    m.get(&(case.to_string(), key.to_string()))
        .unwrap_or_else(|| panic!("missing {} {}", case, key))
}

fn close(got: f64, want: f64, tol: f64, what: &str) {
    if want.is_infinite() {
        assert_eq!(got, want, "{}", what);
    } else {
        assert!(
            (got - want).abs() <= tol * want.abs().max(1.0),
            "{}: {} vs {}",
            what,
            got,
            want
        );
    }
}

#[test]
fn icc_all_types_match_psych() {
    let m = refs();
    let types = [
        ("ICC1", ICCType::ICC1),
        ("ICC2", ICCType::ICC2),
        ("ICC3", ICCType::ICC3),
        ("ICC1k", ICCType::ICC1k),
        ("ICC2k", ICCType::ICC2k),
        ("ICC3k", ICCType::ICC3k),
    ];
    for case in ["icc_sf", "icc_icc2", "icc_icc3"] {
        let dims = get(&m, case, "dims");
        let (n, k) = (dims[0] as usize, dims[1] as usize);
        let flat = get(&m, case, "in_data");
        let data: Vec<Vec<f64>> = (0..n).map(|i| flat[i * k..(i + 1) * k].to_vec()).collect();
        for (name, ty) in types {
            let want = get(&m, case, name);
            let r = icc(&data, ty).unwrap();
            let what = format!("{} {}", case, name);
            close(r.icc, want[0], 1e-10, &format!("{} icc", what));
            close(r.f_value, want[1], 1e-10, &format!("{} F", what));
            close(r.df1, want[2], 1e-12, &format!("{} df1", what));
            close(r.df2, want[3], 1e-12, &format!("{} df2", what));
            close(r.p_value, want[4], 1e-8, &format!("{} p", what));
            close(r.conf_int_lower, want[5], 1e-7, &format!("{} lower", what));
            close(r.conf_int_upper, want[6], 1e-7, &format!("{} upper", what));
        }
    }
}

fn check_t(r: &TTestResult, want: &[f64], what: &str) {
    close(r.statistic, want[0], 1e-10, &format!("{} t", what));
    close(r.df, want[1], 1e-10, &format!("{} df", what));
    close(r.p_value, want[2], 1e-9, &format!("{} p", what));
    let ci = r.conf_int.as_ref().unwrap();
    close(ci.lower, want[3], 1e-9, &format!("{} lower", what));
    close(ci.upper, want[4], 1e-9, &format!("{} upper", what));
}

#[test]
fn t_tests_match_r_including_one_sided_intervals() {
    let m = refs();
    let x = get(&m, "ttest", "in_x");
    let y = get(&m, "ttest", "in_y");
    let xp = get(&m, "ttest", "in_xp");
    let yp = get(&m, "ttest", "in_yp");
    for (alt, a) in [
        ("two.sided", Alternative::TwoSided),
        ("less", Alternative::Less),
        ("greater", Alternative::Greater),
    ] {
        for cl in [0.95, 0.9] {
            let sfx = format!("_{}_{}", alt, cl);
            let r = t_test_one_sample(x, a, 5.0, Some(cl)).unwrap();
            check_t(
                &r,
                get(&m, "ttest", &format!("one{}", sfx)),
                &format!("one{}", sfx),
            );
            assert!(r.mean_y.is_none());
            let r = t_test(xp, yp, TTestKind::Paired, a, 0.2, Some(cl)).unwrap();
            check_t(
                &r,
                get(&m, "ttest", &format!("paired{}", sfx)),
                &format!("paired{}", sfx),
            );
            let r = t_test(x, y, TTestKind::Welch, a, 0.3, Some(cl)).unwrap();
            check_t(
                &r,
                get(&m, "ttest", &format!("welch{}", sfx)),
                &format!("welch{}", sfx),
            );
            let r = t_test(x, y, TTestKind::Student, a, 0.3, Some(cl)).unwrap();
            check_t(
                &r,
                get(&m, "ttest", &format!("student{}", sfx)),
                &format!("student{}", sfx),
            );
        }
    }
    close(
        cohens_d_one_sample(x, 5.0).unwrap(),
        get(&m, "ttest", "cohens_d_one")[0],
        1e-12,
        "cohens_d_one_sample",
    );
    assert!(t_test_one_sample(&[1.0], Alternative::TwoSided, 0.0, None).is_err());
    assert!(t_test_one_sample(x, Alternative::TwoSided, 0.0, Some(1.0)).is_err());
}

/// The bootstrap of the mean against R with B = 20000 on both sides: the
/// standard error within 3 % and the percentile bounds within 0.1 SE (the
/// MC standard error of a bound difference is about 0.03 SE).
#[test]
fn block_bootstrap_mean_matches_tsboot() {
    let m = refs();
    let x = get(&m, "boot", "in_x");
    let cases: [(&str, BootstrapMethod); 6] = [
        ("iid", BootstrapMethod::Iid),
        (
            "geom_l1",
            BootstrapMethod::Stationary {
                expected_block_length: 1.0,
            },
        ),
        (
            "geom_l5",
            BootstrapMethod::Stationary {
                expected_block_length: 5.0,
            },
        ),
        (
            "geom_l10",
            BootstrapMethod::Stationary {
                expected_block_length: 10.0,
            },
        ),
        (
            "fixed_l5",
            BootstrapMethod::CircularBlock { block_length: 5 },
        ),
        (
            "fixed_l10",
            BootstrapMethod::CircularBlock { block_length: 10 },
        ),
    ];
    for (name, method) in cases {
        let want = get(&m, "boot", name);
        let opts = BootstrapCIOptions {
            n_bootstrap: 20000,
            conf_level: 0.95,
            method,
            seed: Some(2024),
        };
        let r = bootstrap_mean_ci_with_options(x, &opts).unwrap();
        println!(
            "{}: rust se {:.5} [{:.5}, {:.5}]  R se {:.5} [{:.5}, {:.5}]",
            name, r.se, r.conf_int_lower, r.conf_int_upper, want[0], want[1], want[2]
        );
        assert!((r.se / want[0] - 1.0).abs() < 0.03, "{} se", name);
        assert!(
            (r.conf_int_lower - want[1]).abs() < 0.1 * want[0],
            "{} lower",
            name
        );
        assert!(
            (r.conf_int_upper - want[2]).abs() < 0.1 * want[0],
            "{} upper",
            name
        );
    }
}

#[test]
fn bootstrap_mean_ci_defaults_are_unchanged() {
    let x: Vec<f64> = (0..40).map(|i| (i as f64 * 0.7).sin()).collect();
    for block in [0usize, 1, 4] {
        let a = bootstrap_mean_ci(&x, 500, 0.9, block, Some(9)).unwrap();
        let b = bootstrap_mean_ci_with_options(
            &x,
            &BootstrapCIOptions {
                n_bootstrap: 500,
                conf_level: 0.9,
                method: BootstrapMethod::from_block_length(block),
                seed: Some(9),
            },
        )
        .unwrap();
        assert_eq!(a.conf_int_lower, b.conf_int_lower);
        assert_eq!(a.conf_int_upper, b.conf_int_upper);
        assert_eq!(a.se, b.se);
    }
    let bad = BootstrapCIOptions {
        method: BootstrapMethod::Stationary {
            expected_block_length: 0.5,
        },
        ..Default::default()
    };
    assert!(bootstrap_mean_ci_with_options(&x, &bad).is_err());
    let bad = BootstrapCIOptions {
        method: BootstrapMethod::CircularBlock { block_length: 0 },
        ..Default::default()
    };
    assert!(bootstrap_mean_ci_with_options(&x, &bad).is_err());
}
