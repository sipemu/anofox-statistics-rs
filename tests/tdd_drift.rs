//! Data-drift tests against R. Reference data:
//! `R/generate_drift_refs.R` -> `R/data/drift_reference.csv`.

use anofox_statistics::drift::{
    binned_proportions, category_proportions, hellinger, hellinger_categorical,
    hellinger_from_proportions, jensen_shannon, jensen_shannon_categorical,
    jensen_shannon_divergence_from_proportions, jensen_shannon_from_proportions, kl_divergence,
    kl_divergence_categorical, kl_divergence_from_proportions, psi, psi_categorical,
    psi_from_proportions, tvd_test, tvd_test_numeric, Binning, ZeroFill,
};
use anofox_statistics::{
    ad_test_ksamp, cvm_test_2sample, epps_singleton_test, ks_test_2sample, wasserstein_distance_1d,
    wasserstein_test_1d, Alternative, CvmMethod, TestResult,
};
use approx::assert_relative_eq;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::OnceLock;

type Refs = HashMap<(String, String), Vec<f64>>;

fn refs() -> &'static Refs {
    static REFS: OnceLock<Refs> = OnceLock::new();
    REFS.get_or_init(|| {
        let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.push("R/data/drift_reference.csv");
        let mut m = HashMap::new();
        for r in csv::Reader::from_path(p).expect("reference file").records() {
            let r = r.unwrap();
            let v = r[2]
                .split_whitespace()
                .map(|s| match s {
                    "NA" => f64::NAN,
                    s => s.parse().unwrap(),
                })
                .collect();
            m.insert((r[0].to_string(), r[1].to_string()), v);
        }
        m
    })
}

fn get(case: &str, key: &str) -> &'static [f64] {
    refs()
        .get(&(case.to_string(), key.to_string()))
        .unwrap_or_else(|| panic!("missing {} {}", case, key))
}

fn get1(case: &str, key: &str) -> f64 {
    get(case, key)[0]
}

/// Numeric reference / current sample.
fn xy() -> (&'static [f64], &'static [f64]) {
    (get("data", "in_x"), get("data", "in_y"))
}

/// Categorical reference / current sample (as numbers).
fn xyc() -> (&'static [f64], &'static [f64]) {
    (get("data", "in_xc"), get("data", "in_yc"))
}

/// Tied numeric reference / current sample of unequal size.
fn xyt() -> (&'static [f64], &'static [f64]) {
    (get("data", "in_xt"), get("data", "in_yt"))
}

/// The numeric two-sample cases of the R script: name -> (x, y).
fn num_cases() -> Vec<(&'static str, &'static [f64], &'static [f64])> {
    let (x, y) = xy();
    let (xt, yt) = xyt();
    vec![
        ("main", x, y),
        ("tie", xt, yt),
        ("large", get("data", "in_xl"), get("data", "in_yl")),
        ("close", get("data", "in_xs"), get("data", "in_ys")),
    ]
}

const ALTS: [(Alternative, &str); 3] = [
    (Alternative::TwoSided, "two.sided"),
    (Alternative::Less, "less"),
    (Alternative::Greater, "greater"),
];

/// The categorical sample as string labels.
fn labels(v: &[f64]) -> Vec<String> {
    v.iter().map(|x| format!("level{}", x)).collect()
}

fn assert_vec(actual: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected) {
        assert_relative_eq!(*a, *e, epsilon = tol);
    }
}

/// The binning cases of the R script: name -> (reference, current, binning).
fn binning_cases() -> Vec<(&'static str, &'static [f64], &'static [f64], Binning)> {
    let (x, y) = xy();
    let (xc, yc) = xyc();
    let (xt, yt) = xyt();
    vec![
        ("num", x, y, Binning::Auto),
        ("cat", xc, yc, Binning::Auto),
        ("q10", x, y, Binning::Quantile(10)),
        ("tie_q5", xt, yt, Binning::Quantile(5)),
        ("tie_cat", xt, yt, Binning::Categorical),
    ]
}

// ---------------------------------------------------------------------------
// Binning
// ---------------------------------------------------------------------------

#[test]
fn binning_matches_r() {
    for (name, r, c, binning) in binning_cases() {
        let (p, q) = binned_proportions(r, c, binning).unwrap();
        let case = format!("bins_{}", name);
        assert_vec(&p, get(&case, "p"), 1e-15);
        assert_vec(&q, get(&case, "q"), 1e-15);
    }
    // Auto picks Sturges (8 bins) for x and one bin per value for xc.
    let (x, y) = xy();
    let sturges = binned_proportions(x, y, Binning::Sturges).unwrap();
    assert_eq!(sturges.0.len(), 8);
    assert_eq!(sturges, binned_proportions(x, y, Binning::Auto).unwrap());
    let (xc, yc) = xyc();
    assert_eq!(
        binned_proportions(xc, yc, Binning::Auto).unwrap(),
        category_proportions(&labels(xc), &labels(yc)).unwrap()
    );
}

#[test]
fn binning_rejects_bad_input() {
    assert!(binned_proportions(&[], &[1.0], Binning::Auto).is_err());
    assert!(binned_proportions(&[1.0], &[], Binning::Auto).is_err());
    assert!(binned_proportions(&[1.0, f64::NAN], &[1.0], Binning::Auto).is_err());
    assert!(binned_proportions(&[1.0], &[f64::INFINITY], Binning::Sturges).is_err());
    assert!(binned_proportions(&[1.0], &[1.0], Binning::Quantile(0)).is_err());
    assert!(category_proportions::<&str>(&[], &["a"]).is_err());
}

// ---------------------------------------------------------------------------
// #35 PSI
// ---------------------------------------------------------------------------

#[test]
fn psi_matches_r() {
    for (name, r, c, binning) in binning_cases() {
        let res = psi(r, c, binning, ZeroFill::Evidently).unwrap();
        assert_relative_eq!(
            res.statistic,
            get1(&format!("psi_{}", name), "statistic"),
            epsilon = 1e-10
        );
        assert_eq!((res.n_reference, res.n_current), (r.len(), c.len()));
    }
    // Issue values.
    let (x, y) = xy();
    assert_relative_eq!(
        psi(x, y, Binning::Auto, ZeroFill::Evidently)
            .unwrap()
            .statistic,
        0.570692693084,
        epsilon = 1e-11
    );
    let (xc, yc) = xyc();
    let cat = psi_categorical(&labels(xc), &labels(yc), ZeroFill::Evidently).unwrap();
    assert_relative_eq!(cat.statistic, 1.56548983794, epsilon = 1e-10);
    assert_eq!(cat.n_bins, 5);
    let (p, q) = (get("bins_cat", "p"), get("bins_cat", "q"));
    assert_relative_eq!(
        psi_from_proportions(p, q, ZeroFill::Evidently).unwrap(),
        1.56548983794,
        epsilon = 1e-10
    );
}

#[test]
fn constant_zero_fill_matches_r() {
    // #42: ZeroFill::Constant(eps), philentropy on manually filled proportions
    for (name, r, c, binning) in binning_cases() {
        for eps in [1e-3, 0.01] {
            let key = format!("statistic_eps{}", eps);
            let fill = ZeroFill::Constant(eps);
            let res = psi(r, c, binning, fill).unwrap();
            assert_relative_eq!(
                res.statistic,
                get1(&format!("psi_{}", name), &key),
                epsilon = 1e-10
            );
            let res = kl_divergence(r, c, binning, None, fill).unwrap();
            assert_relative_eq!(
                res.statistic,
                get1(&format!("kl_{}", name), &key),
                epsilon = 1e-10
            );
        }
    }
    let (xc, yc) = xyc();
    let fill = ZeroFill::Constant(0.01);
    assert_relative_eq!(
        psi_categorical(&labels(xc), &labels(yc), fill)
            .unwrap()
            .statistic,
        get1("psi_cat", "statistic_eps0.01"),
        epsilon = 1e-10
    );
    assert_relative_eq!(
        kl_divergence_categorical(&labels(xc), &labels(yc), None, fill)
            .unwrap()
            .statistic,
        get1("kl_cat", "statistic_eps0.01"),
        epsilon = 1e-10
    );
    assert_eq!(ZeroFill::default(), ZeroFill::Evidently);
    for eps in [0.0, -1e-3, f64::NAN, f64::INFINITY] {
        let fill = ZeroFill::Constant(eps);
        assert!(psi_from_proportions(&[0.5, 0.5], &[1.0, 0.0], fill).is_err());
        assert!(kl_divergence_from_proportions(&[0.5, 0.5], &[1.0, 0.0], None, fill).is_err());
    }
}

#[test]
fn psi_categorical_str_labels() {
    let r = ["a", "a", "b", "c"];
    let c = ["a", "b", "b", "b", "d"];
    let res = psi_categorical(&r, &c, ZeroFill::Evidently).unwrap();
    let (p, q) = category_proportions(&r, &c).unwrap();
    assert_eq!(res.n_bins, 4);
    assert_eq!(
        res.statistic,
        psi_from_proportions(&p, &q, ZeroFill::Evidently).unwrap()
    );
    let t = TestResult::from(&res);
    assert_eq!(t.method, "Population Stability Index");
    assert_eq!(t.statistic, Some(res.statistic));
    assert_eq!(t.p_value, None);
    assert_eq!(t.n, Some(9));
}

#[test]
fn psi_from_proportions_rejects_bad_input() {
    assert!(psi_from_proportions(&[0.5, 0.5], &[1.0], ZeroFill::Evidently).is_err());
    assert!(psi_from_proportions(&[], &[], ZeroFill::Evidently).is_err());
    assert!(psi_from_proportions(&[1.5, -0.5], &[0.5, 0.5], ZeroFill::Evidently).is_err());
    assert!(psi_from_proportions(&[0.0, 0.0], &[0.5, 0.5], ZeroFill::Evidently).is_err());
    assert!(psi_from_proportions(&[f64::NAN, 1.0], &[0.5, 0.5], ZeroFill::Evidently).is_err());
}

// ---------------------------------------------------------------------------
// #36 Kullback-Leibler
// ---------------------------------------------------------------------------

#[test]
fn kl_matches_r() {
    for (name, r, c, binning) in binning_cases() {
        let case = format!("kl_{}", name);
        let ln = kl_divergence(r, c, binning, None, ZeroFill::Evidently).unwrap();
        assert_relative_eq!(ln.statistic, get1(&case, "statistic"), epsilon = 1e-10);
        let log2 = kl_divergence(r, c, binning, Some(2.0), ZeroFill::Evidently).unwrap();
        assert_relative_eq!(
            log2.statistic,
            get1(&case, "statistic_log2"),
            epsilon = 1e-10
        );
    }
    let (x, y) = xy();
    assert_relative_eq!(
        kl_divergence(x, y, Binning::Auto, None, ZeroFill::Evidently)
            .unwrap()
            .statistic,
        0.264782319603,
        epsilon = 1e-11
    );
    // The categorical case needs the renormalization after the zero fill.
    let (xc, yc) = xyc();
    let cat =
        kl_divergence_categorical(&labels(xc), &labels(yc), None, ZeroFill::Evidently).unwrap();
    assert_relative_eq!(cat.statistic, 0.223061363402, epsilon = 1e-11);
    assert_eq!(TestResult::from(&cat).method, "Kullback-Leibler divergence");
}

#[test]
fn kl_rejects_bad_input() {
    let p = [0.5, 0.5];
    for b in [0.0, 1.0, -2.0, f64::NAN, f64::INFINITY] {
        assert!(kl_divergence_from_proportions(&p, &p, Some(b), ZeroFill::Evidently).is_err());
    }
    assert_eq!(
        kl_divergence_from_proportions(&p, &p, None, ZeroFill::Evidently).unwrap(),
        0.0
    );
    assert!(kl_divergence_from_proportions(&p, &[1.0], None, ZeroFill::Evidently).is_err());
    assert!(kl_divergence(
        &[1.0, f64::NAN],
        &[1.0],
        Binning::Auto,
        None,
        ZeroFill::Evidently
    )
    .is_err());
}

// ---------------------------------------------------------------------------
// #37 Jensen-Shannon distance
// ---------------------------------------------------------------------------

#[test]
fn jensen_shannon_matches_r() {
    for (name, r, c, binning) in binning_cases() {
        let case = format!("js_{}", name);
        let ln = jensen_shannon(r, c, binning, None).unwrap();
        assert_relative_eq!(ln.statistic, get1(&case, "statistic"), epsilon = 1e-10);
        let log2 = jensen_shannon(r, c, binning, Some(2.0)).unwrap();
        assert_relative_eq!(
            log2.statistic,
            get1(&case, "statistic_log2"),
            epsilon = 1e-10
        );
        // #44: the divergence, philentropy::JSD
        assert_relative_eq!(
            ln.divergence.unwrap(),
            get1(&case, "divergence"),
            epsilon = 1e-12
        );
        assert_relative_eq!(
            log2.divergence.unwrap(),
            get1(&case, "divergence_log2"),
            epsilon = 1e-12
        );
        let (p, q) = binned_proportions(r, c, binning).unwrap();
        assert_relative_eq!(
            jensen_shannon_divergence_from_proportions(&p, &q, Some(2.0)).unwrap(),
            get1(&case, "divergence_log2"),
            epsilon = 1e-12
        );
    }
    let (x, y) = xy();
    assert_relative_eq!(
        jensen_shannon(x, y, Binning::Auto, None).unwrap().statistic,
        0.257847900615,
        epsilon = 1e-11
    );
    let (xc, yc) = xyc();
    let cat = jensen_shannon_categorical(&labels(xc), &labels(yc), None).unwrap();
    assert_relative_eq!(cat.statistic, 0.27397012217, epsilon = 1e-10);
    assert_relative_eq!(
        cat.divergence.unwrap(),
        get1("js_cat", "divergence"),
        epsilon = 1e-12
    );
    let t = TestResult::from(&cat);
    assert_eq!(t.method, "Jensen-Shannon distance");
    assert_eq!(t.effect_name.as_deref(), Some("divergence"));
    assert_eq!(t.effect_size, cat.divergence);
    // Other divergences carry no secondary value.
    let h = hellinger_categorical(&labels(xc), &labels(yc)).unwrap();
    assert_eq!(h.divergence, None);
    assert_eq!(TestResult::from(&h).effect_size, None);
}

#[test]
fn jensen_shannon_bounds() {
    // Disjoint supports: sqrt(ln 2), or 1 in base 2; identical: 0.
    let (p, q) = ([1.0, 0.0], [0.0, 1.0]);
    assert_relative_eq!(
        jensen_shannon_from_proportions(&p, &q, None).unwrap(),
        2f64.ln().sqrt(),
        epsilon = 1e-15
    );
    assert_relative_eq!(
        jensen_shannon_from_proportions(&p, &q, Some(2.0)).unwrap(),
        1.0,
        epsilon = 1e-15
    );
    assert_eq!(jensen_shannon_from_proportions(&p, &p, None).unwrap(), 0.0);
    // Unnormalized input is normalized first.
    assert_relative_eq!(
        jensen_shannon_from_proportions(&[2.0, 6.0], &[1.0, 1.0], None).unwrap(),
        jensen_shannon_from_proportions(&[0.25, 0.75], &[0.5, 0.5], None).unwrap(),
        epsilon = 1e-15
    );
    assert!(jensen_shannon_from_proportions(&p, &q, Some(1.0)).is_err());
}

// ---------------------------------------------------------------------------
// #38 Hellinger distance
// ---------------------------------------------------------------------------

#[test]
fn hellinger_matches_r() {
    for (name, r, c, binning) in binning_cases() {
        // The numeric Auto case uses evidently's Hellinger bins, checked below.
        if name == "num" {
            continue;
        }
        let res = hellinger(r, c, binning).unwrap();
        assert_relative_eq!(
            res.statistic,
            get1(&format!("hellinger_{}", name), "statistic"),
            epsilon = 1e-10
        );
    }
    let (x, y) = xy();
    for b in [Binning::Auto, Binning::Sturges] {
        let res = hellinger(x, y, b).unwrap();
        assert_relative_eq!(
            res.statistic,
            get1("hellinger_num", "statistic"),
            epsilon = 1e-10
        );
        assert_relative_eq!(res.statistic, 0.262287362723, epsilon = 1e-11);
    }
    let (xc, yc) = xyc();
    let cat = hellinger_categorical(&labels(xc), &labels(yc)).unwrap();
    assert_relative_eq!(cat.statistic, 0.325194993744, epsilon = 1e-11);
    assert_eq!(TestResult::from(&cat).method, "Hellinger distance");
    // Integer codes under Auto: one bin per code, as psi / kl / js (#45).
    let auto = hellinger(xc, yc, Binning::Auto).unwrap();
    assert_relative_eq!(auto.statistic, 0.325194993744, epsilon = 1e-11);
    assert_eq!(auto.n_bins, cat.n_bins);
    // xt has 20 distinct values: Auto bins per value.
    let (xt, yt) = xyt();
    let res = hellinger(xt, yt, Binning::Auto).unwrap();
    assert_eq!(res.n_bins as f64, get1("hellinger_auto_tie", "n_bins"));
    assert_relative_eq!(
        res.statistic,
        get1("hellinger_auto_tie", "statistic"),
        epsilon = 1e-10
    );
    // With ties and > 20 distinct values, evidently's rule (bins from
    // distinct values) differs from the pooled-size Sturges rule.
    let (xtl, ytl) = (get("data", "in_xtl"), get("data", "in_ytl"));
    for (b, case) in [
        (Binning::Auto, "hellinger_auto_tiel"),
        (Binning::Sturges, "hellinger_sturges_tiel"),
    ] {
        let res = hellinger(xtl, ytl, b).unwrap();
        assert_eq!(res.n_bins as f64, get1(case, "n_bins"));
        assert_relative_eq!(res.statistic, get1(case, "statistic"), epsilon = 1e-10);
    }
}

#[test]
fn hellinger_bounds() {
    assert_eq!(
        hellinger_from_proportions(&[1.0, 0.0], &[0.0, 1.0]).unwrap(),
        1.0
    );
    assert_eq!(
        hellinger_from_proportions(&[0.3, 0.7], &[0.3, 0.7]).unwrap(),
        0.0
    );
    assert!(hellinger_from_proportions(&[0.3, 0.7], &[1.0]).is_err());
    assert!(hellinger(&[1.0], &[f64::NAN], Binning::Auto).is_err());
}

// ---------------------------------------------------------------------------
// #30 two-sample Kolmogorov-Smirnov
// ---------------------------------------------------------------------------

#[test]
fn ks_matches_r() {
    for (name, x, y) in num_cases() {
        for (alt, alt_name) in ALTS {
            for (exact, ex_name) in [
                (None, "auto"),
                (Some(true), "exact"),
                (Some(false), "asymp"),
            ] {
                let case = format!("ks_{}_{}_{}", name, alt_name, ex_name);
                let r = ks_test_2sample(x, y, alt, exact).unwrap();
                assert_relative_eq!(r.statistic, get1(&case, "statistic"), epsilon = 1e-12);
                assert_relative_eq!(r.p_value, get1(&case, "p_value"), epsilon = 1e-12);
                assert_eq!(r.exact, get1(&case, "exact") == 1.0, "{}", case);
                assert_eq!((r.n_x, r.n_y), (x.len(), y.len()));
            }
        }
    }
    // Issue values.
    let (x, y) = xy();
    let r = ks_test_2sample(x, y, Alternative::TwoSided, None).unwrap();
    assert_relative_eq!(r.statistic, 0.275, epsilon = 1e-15);
    assert_relative_eq!(r.p_value, 0.0574636491581, epsilon = 1e-9);
    let r = ks_test_2sample(x, y, Alternative::TwoSided, Some(false)).unwrap();
    assert_relative_eq!(r.p_value, 0.0693904727347, epsilon = 1e-9);
    let t = TestResult::from(&r);
    assert_eq!(t.method, "Asymptotic two-sample Kolmogorov-Smirnov test");
    assert_eq!(t.alternative_name(), Some("two.sided"));
    assert_eq!(t.n, Some(90));
}

#[test]
fn ks_edge_cases() {
    assert!(ks_test_2sample(&[], &[1.0], Alternative::TwoSided, None).is_err());
    assert!(ks_test_2sample(&[1.0, f64::NAN], &[1.0], Alternative::TwoSided, None).is_err());
    // Identical samples: D = 0, p = 1.
    let r = ks_test_2sample(&[1.0, 2.0], &[2.0, 1.0], Alternative::TwoSided, None).unwrap();
    assert_eq!((r.statistic, r.p_value), (0.0, 1.0));
    // +-Inf ranks as an extreme value.
    let r = ks_test_2sample(
        &[f64::NEG_INFINITY, 0.0],
        &[1.0, f64::INFINITY],
        Alternative::Greater,
        None,
    )
    .unwrap();
    assert_eq!(r.statistic, 1.0);
}

// ---------------------------------------------------------------------------
// #31 k-sample Anderson-Darling
// ---------------------------------------------------------------------------

#[test]
fn ad_ksamp_matches_ksamples() {
    for name in ["main", "tie", "k3", "close", "large", "k4", "shift"] {
        let case = format!("ad_{}", name);
        let groups: Vec<&[f64]> = (1..)
            .map_while(|j| refs().get(&(case.clone(), format!("in_g{}", j))))
            .map(|v| v.as_slice())
            .collect();
        for (v, midrank) in [(1, false), (2, true)] {
            let r = ad_test_ksamp(&groups, midrank).unwrap();
            assert_relative_eq!(r.ad, get1(&case, &format!("ad_v{}", v)), epsilon = 1e-10);
            assert_relative_eq!(
                r.statistic,
                get1(&case, &format!("statistic_v{}", v)),
                epsilon = 1e-10
            );
            let p = get1(&case, &format!("p_value_v{}", v));
            assert_relative_eq!(r.p_value, p, epsilon = 1e-9, max_relative = 1e-7);
            assert_eq!(
                (r.k, r.n),
                (groups.len(), groups.iter().map(|g| g.len()).sum())
            );
        }
    }
    // Issue values (kSamples, version 2 and 1).
    let (x, y) = xy();
    let r = ad_test_ksamp(&[x, y], true).unwrap();
    assert_relative_eq!(r.statistic, 3.61044012836, epsilon = 1e-10);
    assert_relative_eq!(r.p_value, 0.011626, epsilon = 1e-5);
    let r1 = ad_test_ksamp(&[x, y], false).unwrap();
    assert_relative_eq!(r1.statistic, 3.52653061328, epsilon = 1e-10);
    assert_relative_eq!(r1.p_value, 0.012463, epsilon = 1e-5);
    let t = TestResult::from(&r);
    assert_eq!(t.method, "k-sample Anderson-Darling test");
    assert_eq!(
        (t.statistic, t.p_value, t.n),
        (Some(r.statistic), Some(r.p_value), Some(90))
    );
}

#[test]
fn ad_ksamp_edge_cases() {
    assert!(ad_test_ksamp(&[&[1.0, 2.0]], true).is_err());
    assert!(ad_test_ksamp(&[&[1.0, 2.0], &[]], true).is_err());
    assert!(ad_test_ksamp(&[&[1.0, f64::NAN], &[1.0]], true).is_err());
    // N = 2: T undefined, p = 1 (kSamples).
    let r = ad_test_ksamp(&[&[1.0], &[2.0]], true).unwrap();
    assert!(r.statistic.is_nan());
    assert_eq!(r.p_value, 1.0);
}

// ---------------------------------------------------------------------------
// #32 two-sample Cramer-von Mises
// ---------------------------------------------------------------------------

#[test]
fn cvm_asymptotic_matches_r() {
    for (name, x, y) in num_cases() {
        let case = format!("cvm_{}", name);
        let r = cvm_test_2sample(x, y, CvmMethod::Asymptotic).unwrap();
        assert_relative_eq!(r.statistic, get1(&case, "statistic"), epsilon = 1e-12);
        assert_relative_eq!(r.p_value, get1(&case, "p_value"), epsilon = 1e-12);
        assert!(!r.exact);
        // All these cases have max(n_x, n_y) > 20: Auto is asymptotic.
        assert_eq!(r, cvm_test_2sample(x, y, CvmMethod::Auto).unwrap());
    }
    let (x, y) = xy();
    let r = cvm_test_2sample(x, y, CvmMethod::Auto).unwrap();
    assert_relative_eq!(r.statistic, 0.4722592592593, epsilon = 1e-12);
    assert_relative_eq!(r.p_value, 0.0468933196396, epsilon = 1e-12);
    let t = TestResult::from(&r);
    assert_eq!(t.method, "Asymptotic two-sample Cramer-von Mises test");
    assert_eq!(t.n, Some(90));
}

#[test]
fn cvm_exact_matches_enumeration() {
    for name in ["e54", "e76", "e88", "e107", "e39", "t108", "tcat"] {
        let case = format!("cvm_{}", name);
        let (x, y) = (get(&case, "in_x"), get(&case, "in_y"));
        let r = cvm_test_2sample(x, y, CvmMethod::Auto).unwrap();
        assert!(r.exact);
        assert_relative_eq!(r.statistic, get1(&case, "statistic"), epsilon = 1e-12);
        assert_relative_eq!(r.p_value, get1(&case, "p_value"), epsilon = 1e-12);
    }
    // Cross-check for samples too large to enumerate in R:
    // scipy.stats.cramervonmises_2samp(method = "exact") 1.17 (no ties).
    let (x, y) = xy();
    let r = cvm_test_2sample(x, y, CvmMethod::Exact).unwrap();
    assert_relative_eq!(r.p_value, 0.04728222743467917, epsilon = 1e-12);
}

#[test]
fn cvm_edge_cases() {
    assert!(cvm_test_2sample(&[], &[1.0], CvmMethod::Auto).is_err());
    assert!(cvm_test_2sample(&[1.0, f64::NAN], &[1.0], CvmMethod::Auto).is_err());
    for method in [CvmMethod::Exact, CvmMethod::Asymptotic] {
        let r = cvm_test_2sample(&[1.0], &[2.0], method).unwrap();
        assert_eq!(r.p_value, 1.0);
    }
}

// ---------------------------------------------------------------------------
// #33 Epps-Singleton
// ---------------------------------------------------------------------------

#[test]
fn epps_singleton_matches_r() {
    for name in ["main", "small", "discrete", "rankdef", "t3", "tie"] {
        let case = format!("es_{}", name);
        let r = epps_singleton_test(
            get(&case, "in_x"),
            get(&case, "in_y"),
            Some(get(&case, "in_t")),
        )
        .unwrap();
        assert_relative_eq!(r.statistic, get1(&case, "statistic"), max_relative = 1e-10);
        assert_relative_eq!(r.p_value, get1(&case, "p_value"), max_relative = 1e-9);
        assert_eq!(r.df as f64, get1(&case, "df"), "{}", case);
    }
    // Issue values, default t.
    let (x, y) = xy();
    let r = epps_singleton_test(x, y, None).unwrap();
    assert_relative_eq!(r.statistic, 19.25205304182, epsilon = 1e-9);
    assert_relative_eq!(r.p_value, 0.00070119804217, epsilon = 1e-12);
    let t = TestResult::from(&r);
    assert_eq!(t.method, "Epps-Singleton two-sample test");
    assert_eq!((t.df1, t.n), (Some(4.0), Some(90)));
}

#[test]
fn epps_singleton_rejects_bad_input() {
    let (x, y) = xy();
    assert!(epps_singleton_test(&x[..4], y, None).is_err());
    assert!(epps_singleton_test(x, &y[..4], None).is_err());
    assert!(epps_singleton_test(x, y, Some(&[])).is_err());
    assert!(epps_singleton_test(x, y, Some(&[0.4, -0.8])).is_err());
    assert!(epps_singleton_test(x, y, Some(&[f64::NAN])).is_err());
    let mut xn = x.to_vec();
    xn[2] = f64::INFINITY;
    assert!(epps_singleton_test(&xn, y, None).is_err());
    // zero pooled IQR
    let c = [1.0; 10];
    assert!(epps_singleton_test(&c, &c, None).is_err());
}

// ---------------------------------------------------------------------------
// #34 1-D Wasserstein
// ---------------------------------------------------------------------------

#[test]
fn wasserstein_matches_transport() {
    let mut cases = num_cases();
    cases.push((
        "equal",
        get("wass_equal", "in_x"),
        get("wass_equal", "in_y"),
    ));
    for (name, x, y) in cases {
        let case = format!("wass_{}", name);
        for p in [1, 2, 3] {
            let key = format!("distance_p{}", p);
            if let Some(e) = refs().get(&(case.clone(), key)) {
                let d = wasserstein_distance_1d(x, y, p as f64).unwrap();
                assert_relative_eq!(d, e[0], epsilon = 1e-12);
            }
        }
        if name != "equal" {
            let r = wasserstein_test_1d(x, y, 1.0, 0, None).unwrap();
            assert_relative_eq!(r.distance, get1(&case, "distance_p1"), epsilon = 1e-12);
            assert_relative_eq!(
                r.distance_normed,
                get1(&case, "distance_normed"),
                epsilon = 1e-12
            );
            // #43: W_2 and its normed score
            let r = wasserstein_test_1d(x, y, 2.0, 0, None).unwrap();
            assert_eq!(r.p, 2.0);
            assert_relative_eq!(r.distance, get1(&case, "distance_p2"), epsilon = 1e-12);
            assert_relative_eq!(
                r.distance_normed,
                get1(&case, "distance_normed_p2"),
                epsilon = 1e-12
            );
        }
    }
    let (x, y) = xy();
    let r = wasserstein_test_1d(x, y, 1.0, 0, None).unwrap();
    assert_relative_eq!(r.distance, 0.625792792352, epsilon = 1e-11);
    assert_relative_eq!(r.distance_normed, 0.43940285495, epsilon = 1e-10);
    assert_eq!(r.p_value, None);
    let t = TestResult::from(&r);
    assert_eq!(t.method, "Wasserstein distance");
    assert_eq!(t.effect_size, Some(r.distance_normed));
}

#[test]
fn wasserstein_permutation_p_value() {
    // twosamples::wass_test with 20000 permutations; 4000 here: the two
    // Monte Carlo standard errors combined are below 0.0085 for p ~ 0.5.
    for name in ["main", "close", "tie"] {
        let (_, x, y) = num_cases().into_iter().find(|c| c.0 == name).unwrap();
        let case = format!("wass_{}", name);
        let r = wasserstein_test_1d(x, y, 1.0, 4000, Some(42)).unwrap();
        let p = r.p_value.unwrap();
        let e = get1(&case, "p_value_mc");
        let se = (e * (1.0 - e) / 4000.0 + e * (1.0 - e) / 20000.0).sqrt();
        assert!((p - e).abs() < 4.0 * se + 1e-3, "{}: {} vs {}", name, p, e);
        assert_eq!(r.n_permutations, 4000);
        // reproducible with a seed
        assert_eq!(r, wasserstein_test_1d(x, y, 1.0, 4000, Some(42)).unwrap());
        // #43: permutation p-value of W_2 (R: 20000 permutations of
        // transport::wasserstein1d(p = 2))
        let r = wasserstein_test_1d(x, y, 2.0, 4000, Some(42)).unwrap();
        let p = r.p_value.unwrap();
        let e = get1(&case, "p_value_mc_p2");
        let se = (e * (1.0 - e) / 4000.0 + e * (1.0 - e) / 20000.0).sqrt();
        assert!(
            (p - e).abs() < 4.0 * se + 1e-3,
            "{} p=2: {} vs {}",
            name,
            p,
            e
        );
    }
}

#[test]
fn wasserstein_edge_cases() {
    assert!(wasserstein_distance_1d(&[], &[1.0], 1.0).is_err());
    assert!(wasserstein_distance_1d(&[1.0], &[f64::NAN], 1.0).is_err());
    assert!(wasserstein_distance_1d(&[1.0], &[2.0], 0.5).is_err());
    assert!(wasserstein_distance_1d(&[1.0], &[2.0], f64::INFINITY).is_err());
    assert!(wasserstein_test_1d(&[1.0], &[2.0], 0.5, 10, Some(1)).is_err());
    // identical samples: every split is at least as extreme
    let r = wasserstein_test_1d(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0], 2.0, 99, Some(1)).unwrap();
    assert_eq!((r.distance, r.p_value), (0.0, Some(1.0)));
    assert_eq!(
        wasserstein_distance_1d(&[1.0, 2.0], &[2.0, 1.0], 1.0).unwrap(),
        0.0
    );
    // constant reference: the normed distance divides by 0.001
    let r = wasserstein_test_1d(&[1.0, 1.0], &[2.0], 1.0, 0, None).unwrap();
    assert_relative_eq!(r.distance_normed, 1000.0, epsilon = 1e-9);
}

// ---------------------------------------------------------------------------
// #39 total variation distance
// ---------------------------------------------------------------------------

#[test]
fn tvd_matches_r() {
    let (xc, yc) = xyc();
    let (xt, yt) = xyt();
    for (name, r, c) in [("cat", xc, yc), ("tie", xt, yt)] {
        let case = format!("tvd_{}", name);
        let res = tvd_test_numeric(r, c, 1000, Some(7)).unwrap();
        assert_relative_eq!(res.statistic, get1(&case, "statistic"), epsilon = 1e-15);
        assert_eq!(res.n_categories as f64, get1(&case, "n_categories"));
        // R: B = 100000; here B = 1000, tolerance 4 combined Monte Carlo SEs.
        let e = get1(&case, "p_value_mc");
        let se = (e * (1.0 - e) / 1000.0 + e * (1.0 - e) / 100000.0).sqrt();
        assert!(
            (res.p_value - e).abs() < 4.0 * se + 2e-3,
            "{}: {} vs {}",
            name,
            res.p_value,
            e
        );
    }
    // statistic = 0.2 exactly (up to the rounding of 0.26 - 0.2 etc.), with labels
    let res = tvd_test(&labels(xc), &labels(yc), 1000, Some(1)).unwrap();
    assert_relative_eq!(res.statistic, 0.2, epsilon = 1e-15);
    assert_eq!(res.n_categories, 5);
    assert!((res.p_value - 0.28894).abs() < 0.06);
    // A fixed seed gives identical results; labels and numeric codes agree.
    assert_eq!(
        res,
        tvd_test(&labels(xc), &labels(yc), 1000, Some(1)).unwrap()
    );
    assert_eq!(res, tvd_test_numeric(xc, yc, 1000, Some(1)).unwrap());
    let t = TestResult::from(&res);
    assert_eq!(t.method, "Total variation distance (permutation test)");
    assert_eq!((t.p_value, t.n), (Some(res.p_value), Some(90)));
}

#[test]
fn tvd_edge_cases() {
    assert!(tvd_test::<&str>(&[], &["a"], 10, None).is_err());
    assert!(tvd_test(&["a"], &["a"], 0, None).is_err());
    assert!(tvd_test_numeric(&[1.0, f64::NAN], &[1.0], 10, None).is_err());
    assert!(tvd_test_numeric(&[1.0], &[f64::INFINITY], 10, None).is_err());
    // identical distributions: TVD 0, p = 1
    let r = tvd_test(&[1, 2, 1, 2], &[2, 1], 99, Some(3)).unwrap();
    assert_eq!((r.statistic, r.p_value), (0.0, 1.0));
    // disjoint categories: TVD 1
    let r = tvd_test(&["a", "a"], &["b", "b", "b"], 99, Some(3)).unwrap();
    assert_eq!(r.statistic, 1.0);
}
