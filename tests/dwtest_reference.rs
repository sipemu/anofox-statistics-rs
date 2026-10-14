//! `durbin_watson_test` against R `lmtest::dwtest`. Reference data:
//! `R/generate_dwtest_refs.R` -> `R/data/dwtest_reference.csv`.
//! Cases starting with `e_` use the exact p-value (`n < 100`), `a_` the
//! normal approximation (`n >= 100`); `forced_*` override R's default `exact`.
//!
//! Exact p-values are checked against dwtest's defaults (`p_*`, a 15-term
//! Pan series) and against high-precision exact values (`pexact_*`, Imhof's
//! formula with `stats::integrate`, cross-checked with a 1000-term Pan
//! series). Intended difference: in `e_three` dwtest's 15-term series returns
//! a value outside `[0, 1]` and R falls back to the normal approximation,
//! while `durbin_watson_test` returns the exact p-value; that case is checked
//! against `pexact_*` only.
//! Run with `--nocapture` to print the maximum absolute differences.

use anofox_statistics::{durbin_watson_test, Alternative};
use std::collections::HashMap;
use std::path::PathBuf;

type Refs = HashMap<(String, String), Vec<f64>>;

fn refs() -> Refs {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("R/data/dwtest_reference.csv");
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
}

fn get<'a>(m: &'a Refs, case: &str, key: &str) -> &'a [f64] {
    m.get(&(case.to_string(), key.to_string()))
        .unwrap_or_else(|| panic!("missing {} {}", case, key))
}

const CASES: [&str; 12] = [
    "e_int_ar",
    "e_trend",
    "e_two_neg",
    "e_three",
    "e_noint",
    "e_strong_ar",
    "e_tiny",
    "forced_approx",
    "a_int",
    "a_two_ar",
    "a_three_neg",
    "forced_exact",
];

const ALTS: [(Alternative, &str); 3] = [
    (Alternative::Greater, "greater"),
    (Alternative::TwoSided, "two.sided"),
    (Alternative::Less, "less"),
];

#[test]
fn durbin_watson_test_matches_dwtest() {
    let m = refs();
    let (mut d_stat, mut d_p, mut d_exact) = (0.0_f64, 0.0_f64, 0.0_f64);
    for c in CASES {
        let e = get(&m, c, "in_resid");
        let cols: Vec<&[f64]> = (1..)
            .map_while(|j| m.get(&(c.to_string(), format!("in_x{}", j))))
            .map(|v| v.as_slice())
            .collect();
        let exact = match get(&m, c, "in_exact")[0] {
            v if v.is_nan() => None,
            v => Some(v == 1.0),
        };
        let want_stat = get(&m, c, "stat")[0];
        for (alt, name) in ALTS {
            let r = durbin_watson_test(e, &cols, alt, exact).unwrap();
            assert_eq!(r.method, "Durbin-Watson test");
            assert_eq!(r.alternative, Some(alt));
            assert_eq!(r.n, Some(e.len()));
            d_stat = d_stat.max((r.statistic.unwrap() - want_stat).abs());
            let got = r.p_value.unwrap();
            if let Some(want) = m.get(&(c.to_string(), format!("pexact_{}", name))) {
                let d = (got - want[0]).abs();
                assert!(d <= 1e-6, "{} {}: p {} vs exact {}", c, name, got, want[0]);
                d_exact = d_exact.max(d);
            }
            let want_p = get(&m, c, &format!("p_{}", name))[0];
            let dp = (got - want_p).abs();
            if c == "e_three" {
                // R's series fails here; our exact value differs on purpose.
                assert!(
                    dp > 1e-3,
                    "e_three {}: expected to differ from dwtest",
                    name
                );
                continue;
            }
            assert!(dp <= 1e-6, "{} {}: p {} vs {}", c, name, got, want_p);
            d_p = d_p.max(dp);
        }
    }
    println!("dw statistic  max abs diff {:.3e}", d_stat);
    println!("dw p vs dwtest max abs diff {:.3e}", d_p);
    println!("dw p vs exact  max abs diff {:.3e}", d_exact);
    assert!(d_stat <= 1e-12);
}

#[test]
fn durbin_watson_test_errors_and_edge_cases() {
    let ones = [1.0; 5];
    let t = [1.0, 2.0, 3.0, 4.0, 5.0];
    let e = [0.3, -0.1, -0.4, 0.0, 0.2];
    // No columns, wrong length, k >= n, rank deficient, non-finite.
    assert!(durbin_watson_test(&e, &[], Alternative::Greater, None).is_err());
    assert!(durbin_watson_test(&e, &[&t[..4]], Alternative::Greater, None).is_err());
    let cols: Vec<&[f64]> = vec![&ones; 5];
    assert!(durbin_watson_test(&e, &cols, Alternative::Greater, None).is_err());
    assert!(durbin_watson_test(&e, &[&ones, &ones], Alternative::Greater, None).is_err());
    let bad = [1.0, f64::NAN, 3.0, 4.0, 5.0];
    assert!(durbin_watson_test(&e, &[&ones, &bad], Alternative::Greater, None).is_err());
    assert!(durbin_watson_test(&[0.0; 5], &[&ones], Alternative::Greater, None).is_err());
    // n < 3: p-value 1, as R.
    let r = durbin_watson_test(&[0.5, -0.5], &[&[1.0, 1.0]], Alternative::Greater, None).unwrap();
    assert_eq!(r.p_value, Some(1.0));
    // Approximation with n < max(5, k): p-value 1, as R.
    let r = durbin_watson_test(&e[..4], &[&ones[..4]], Alternative::Greater, Some(false)).unwrap();
    assert_eq!(r.p_value, Some(1.0));
}
