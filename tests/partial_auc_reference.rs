//! Partial / semi-partial correlation (ppcor) and the AUC null quantile
//! (qwilcox) against R. Reference data: `R/generate_partial_auc_refs.R` ->
//! `R/data/partial_auc_reference.csv`.

use anofox_statistics::{auc_null_quantile, partial_cor, rank, semi_partial_cor};
use std::collections::HashMap;

type Refs = HashMap<(String, String), Vec<f64>>;

fn refs() -> Refs {
    let p = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/R/data/partial_auc_reference.csv"
    );
    let mut m = HashMap::new();
    for r in csv::Reader::from_path(p).expect("reference file").records() {
        let r = r.unwrap();
        let v = r[2]
            .split_whitespace()
            .map(|s| s.parse().unwrap())
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
    assert!(
        (got - want).abs() <= tol * want.abs().max(1e-300),
        "{}: {} vs {}",
        what,
        got,
        want
    );
}

#[test]
fn partial_and_semi_partial_match_ppcor() {
    let m = refs();
    let raw: Vec<Vec<f64>> = ["x", "y", "z1", "z2", "z3"]
        .iter()
        .map(|k| get(&m, "data", k).to_vec())
        .collect();
    for method in ["pearson", "spearman"] {
        let v: Vec<Vec<f64>> = if method == "spearman" {
            raw.iter().map(|c| rank(c).unwrap()).collect()
        } else {
            raw.clone()
        };
        for k in 1..=3 {
            let z: Vec<&[f64]> = v[2..2 + k].iter().map(|c| c.as_slice()).collect();
            for (name, r) in [
                ("pcor", partial_cor(&v[0], &v[1], &z).unwrap()),
                ("spcor", semi_partial_cor(&v[0], &v[1], &z).unwrap()),
            ] {
                let want = get(&m, &format!("{}_{}", name, method), &format!("k{}", k));
                let what = format!("{} {} k={}", name, method, k);
                close(r.estimate, want[0], 1e-10, &format!("{} estimate", what));
                close(r.statistic, want[1], 1e-10, &format!("{} statistic", what));
                close(r.p_value, want[2], 1e-8, &format!("{} p", what));
                assert_eq!(r.df, want[3], "{} df", what);
                assert_eq!(r.n_controls, k);
            }
        }
    }
}

#[test]
fn auc_null_quantile_matches_qwilcox() {
    let m = refs();
    let probs = get(&m, "auc_q", "probs");
    for (a, b) in [
        (1, 1),
        (3, 4),
        (10, 10),
        (7, 23),
        (25, 25),
        (50, 40),
        (40, 50),
    ] {
        let want = get(&m, "auc_q", &format!("n{}_{}", a, b));
        for (&p, &w) in probs.iter().zip(want) {
            let got = auc_null_quantile(a, b, p).unwrap();
            assert!(
                (got - w).abs() < 1e-12,
                "({}, {}) p = {}: {} vs {}",
                a,
                b,
                p,
                got,
                w
            );
        }
    }
    assert_eq!(auc_null_quantile(50, 40, 1.0).unwrap(), 1.0);
    assert_eq!(auc_null_quantile(50, 40, 0.0).unwrap(), 0.0);
}
