//! Data-drift tests against R. Reference data:
//! `R/generate_drift_refs.R` -> `R/data/drift_reference.csv`.

use anofox_statistics::drift::{
    binned_proportions, category_proportions, hellinger, hellinger_categorical,
    hellinger_from_proportions, jensen_shannon, jensen_shannon_categorical,
    jensen_shannon_from_proportions, kl_divergence, kl_divergence_categorical,
    kl_divergence_from_proportions, psi, psi_categorical, psi_from_proportions, Binning,
};
use anofox_statistics::TestResult;
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
        let res = psi(r, c, binning).unwrap();
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
        psi(x, y, Binning::Auto).unwrap().statistic,
        0.570692693084,
        epsilon = 1e-11
    );
    let (xc, yc) = xyc();
    let cat = psi_categorical(&labels(xc), &labels(yc)).unwrap();
    assert_relative_eq!(cat.statistic, 1.56548983794, epsilon = 1e-10);
    assert_eq!(cat.n_bins, 5);
    let (p, q) = (get("bins_cat", "p"), get("bins_cat", "q"));
    assert_relative_eq!(
        psi_from_proportions(p, q).unwrap(),
        1.56548983794,
        epsilon = 1e-10
    );
}

#[test]
fn psi_categorical_str_labels() {
    let r = ["a", "a", "b", "c"];
    let c = ["a", "b", "b", "b", "d"];
    let res = psi_categorical(&r, &c).unwrap();
    let (p, q) = category_proportions(&r, &c).unwrap();
    assert_eq!(res.n_bins, 4);
    assert_eq!(res.statistic, psi_from_proportions(&p, &q).unwrap());
    let t = TestResult::from(&res);
    assert_eq!(t.method, "Population Stability Index");
    assert_eq!(t.statistic, Some(res.statistic));
    assert_eq!(t.p_value, None);
    assert_eq!(t.n, Some(9));
}

#[test]
fn psi_from_proportions_rejects_bad_input() {
    assert!(psi_from_proportions(&[0.5, 0.5], &[1.0]).is_err());
    assert!(psi_from_proportions(&[], &[]).is_err());
    assert!(psi_from_proportions(&[1.5, -0.5], &[0.5, 0.5]).is_err());
    assert!(psi_from_proportions(&[0.0, 0.0], &[0.5, 0.5]).is_err());
    assert!(psi_from_proportions(&[f64::NAN, 1.0], &[0.5, 0.5]).is_err());
}

// ---------------------------------------------------------------------------
// #36 Kullback-Leibler
// ---------------------------------------------------------------------------

#[test]
fn kl_matches_r() {
    for (name, r, c, binning) in binning_cases() {
        let case = format!("kl_{}", name);
        let ln = kl_divergence(r, c, binning, None).unwrap();
        assert_relative_eq!(ln.statistic, get1(&case, "statistic"), epsilon = 1e-10);
        let log2 = kl_divergence(r, c, binning, Some(2.0)).unwrap();
        assert_relative_eq!(
            log2.statistic,
            get1(&case, "statistic_log2"),
            epsilon = 1e-10
        );
    }
    let (x, y) = xy();
    assert_relative_eq!(
        kl_divergence(x, y, Binning::Auto, None).unwrap().statistic,
        0.264782319603,
        epsilon = 1e-11
    );
    // The categorical case needs the renormalization after the zero fill.
    let (xc, yc) = xyc();
    let cat = kl_divergence_categorical(&labels(xc), &labels(yc), None).unwrap();
    assert_relative_eq!(cat.statistic, 0.223061363402, epsilon = 1e-11);
    assert_eq!(TestResult::from(&cat).method, "Kullback-Leibler divergence");
}

#[test]
fn kl_rejects_bad_input() {
    let p = [0.5, 0.5];
    for b in [0.0, 1.0, -2.0, f64::NAN, f64::INFINITY] {
        assert!(kl_divergence_from_proportions(&p, &p, Some(b)).is_err());
    }
    assert_eq!(kl_divergence_from_proportions(&p, &p, None).unwrap(), 0.0);
    assert!(kl_divergence_from_proportions(&p, &[1.0], None).is_err());
    assert!(kl_divergence(&[1.0, f64::NAN], &[1.0], Binning::Auto, None).is_err());
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
    assert_eq!(TestResult::from(&cat).method, "Jensen-Shannon distance");
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
        // Auto means evidently's numeric Hellinger rule; the categorical
        // case of the R script uses one bin per value.
        let binning = match binning {
            Binning::Auto => Binning::Categorical,
            b => b,
        };
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
    // With ties, evidently's rule (bins from distinct values) differs from
    // the pooled-size Sturges rule.
    let (xt, yt) = xyt();
    for (b, case) in [
        (Binning::Auto, "hellinger_auto_tie"),
        (Binning::Sturges, "hellinger_sturges_tie"),
    ] {
        let res = hellinger(xt, yt, b).unwrap();
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
