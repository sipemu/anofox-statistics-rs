//! `p_adjust`, `ptukey`/`qtukey`, `pairwise_t_test`, `tukey_hsd` and
//! `dunn_test` against R 4.6 (`p.adjust`, `ptukey`, `qtukey`,
//! `pairwise.t.test`, `TukeyHSD(aov(...))`, `dunn.test(altp = TRUE)` +
//! `p.adjust` as in `FSA::dunnTest`). Reference data:
//! `R/generate_posthoc_refs.R`.

use anofox_statistics::{
    dunn_test, p_adjust, pairwise_t_test, ptukey, qtukey, tukey_hsd, Alternative, PAdjustMethod,
};
use std::path::PathBuf;

fn path(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("R/data");
    p.push(name);
    p
}

fn num(s: &str) -> f64 {
    if s == "NA" {
        f64::NAN
    } else {
        s.parse().unwrap()
    }
}

fn vec_of(s: &str) -> Vec<f64> {
    s.split_whitespace().map(num).collect()
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a.is_nan() && b.is_nan()) || (a - b).abs() <= tol * (1.0 + b.abs())
}

fn records(name: &str) -> Vec<csv::StringRecord> {
    csv::Reader::from_path(path(name))
        .expect("reference file")
        .records()
        .map(|r| r.unwrap())
        .collect()
}

#[test]
fn p_adjust_matches_r() {
    let recs = records("p_adjust_reference.csv");
    assert!(recs.len() >= 50);
    for r in recs {
        let method: PAdjustMethod = r[1].parse().unwrap();
        let got = p_adjust(&vec_of(&r[2]), method);
        let want = vec_of(&r[3]);
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(&want) {
            assert!(
                close(*g, *w, 1e-14),
                "case {} {}: {} vs {}",
                &r[0],
                &r[1],
                g,
                w
            );
        }
    }
}

/// R accumulates parts of `wprob` in `long double`; f64 differs by up to
/// ~1e-10 (df > 25000, nranges > 1), ~1e-11 elsewhere.
const TOL_P: f64 = 1e-10;

#[test]
fn ptukey_matches_r() {
    let mut max_err: f64 = 0.0;
    let recs = records("ptukey_reference.csv");
    for r in recs {
        let (q, k, df, nr) = (num(&r[0]), num(&r[1]), num(&r[2]), num(&r[3]));
        let lower = ptukey(q, k, df, nr, true);
        let upper = ptukey(q, k, df, nr, false);
        max_err = max_err.max((lower - num(&r[4])).abs());
        assert!(
            (lower - num(&r[4])).abs() < TOL_P,
            "lower {:?}: {}",
            r,
            lower
        );
        assert!(
            (upper - num(&r[5])).abs() < TOL_P,
            "upper {:?}: {}",
            r,
            upper
        );
    }
    eprintln!("ptukey max abs error vs R: {:e}", max_err);
}

#[test]
fn qtukey_matches_r() {
    let recs = records("qtukey_reference.csv");
    for r in recs {
        let (p, k, df, nr) = (num(&r[0]), num(&r[1]), num(&r[2]), num(&r[3]));
        let q = qtukey(p, k, df, nr, true);
        assert!(close(q, num(&r[4]), 1e-8), "{:?}: {}", r, q);
        // upper tail is the complement
        let qu = qtukey(1.0 - p, k, df, nr, false);
        assert!(close(qu, q, 1e-6), "upper {:?}: {} vs {}", r, qu, q);
    }
}

struct DataSet {
    x: Vec<f64>,
    g: Vec<String>,
}

fn data_sets() -> Vec<DataSet> {
    records("posthoc_data.csv")
        .into_iter()
        .map(|r| DataSet {
            x: vec_of(&r[1]),
            g: r[2].split_whitespace().map(String::from).collect(),
        })
        .collect()
}

fn alt(s: &str) -> Alternative {
    match s {
        "two.sided" => Alternative::TwoSided,
        "less" => Alternative::Less,
        _ => Alternative::Greater,
    }
}

#[test]
fn posthoc_matches_r() {
    let sets = data_sets();
    let recs = records("posthoc_reference.csv");
    let mut counts = [0usize; 3];
    for r in &recs {
        let set = &sets[r[0].parse::<usize>().unwrap() - 1];
        let (g1, g2) = (&r[5], &r[6]);
        let tol = 1e-9;
        match &r[1] {
            "pairwise_t" => {
                let res = pairwise_t_test(
                    &set.x,
                    &set.g,
                    &r[2] == "TRUE",
                    r[4].parse().unwrap(),
                    alt(&r[3]),
                )
                .unwrap();
                let c = res
                    .comparisons
                    .iter()
                    .find(|c| c.group1 == g1 && c.group2 == g2)
                    .expect("pair");
                assert!(close(c.estimate, num(&r[7]), tol), "{:?}", r);
                assert!(close(c.statistic, num(&r[8]), tol), "{:?}", r);
                assert!(close(c.df.unwrap(), num(&r[9]), tol), "{:?}", r);
                assert!(close(c.p_value, num(&r[10]), tol), "{:?} {}", r, c.p_value);
                assert!(close(c.p_adj, num(&r[11]), tol), "{:?} {}", r, c.p_adj);
                assert!(c.conf_low.is_none());
                counts[0] += 1;
            }
            "tukey" => {
                let res = tukey_hsd(&set.x, &set.g, num(&r[4])).unwrap();
                let c = res
                    .comparisons
                    .iter()
                    .find(|c| c.group1 == g1 && c.group2 == g2)
                    .expect("pair");
                assert!(close(c.estimate, num(&r[7]), tol), "{:?}", r);
                assert!(close(c.df.unwrap(), num(&r[9]), tol), "{:?}", r);
                assert!(
                    (c.p_value - num(&r[10])).abs() < 1e-10,
                    "{:?} {}",
                    r,
                    c.p_value
                );
                assert!((c.p_adj - num(&r[11])).abs() < 1e-10, "{:?}", r);
                assert!(close(c.conf_low.unwrap(), num(&r[12]), tol), "{:?}", r);
                assert!(close(c.conf_high.unwrap(), num(&r[13]), tol), "{:?}", r);
                counts[1] += 1;
            }
            "dunn" => {
                let res = dunn_test(&set.x, &set.g, r[4].parse().unwrap()).unwrap();
                // dunn.test labels "A - B" with z for A - B; we report group2 - group1.
                let c = res
                    .comparisons
                    .iter()
                    .find(|c| c.group1 == g1 && c.group2 == g2)
                    .expect("pair");
                assert!(
                    close(c.statistic, -num(&r[8]), tol),
                    "{:?} {}",
                    r,
                    c.statistic
                );
                assert!(close(c.p_value, num(&r[10]), tol), "{:?}", r);
                assert!(close(c.p_adj, num(&r[11]), tol), "{:?}", r);
                counts[2] += 1;
            }
            other => panic!("unknown test {}", other),
        }
    }
    assert!(counts.iter().all(|&c| c > 50), "{:?}", counts);
}

#[test]
fn row_order_and_memory_shape() {
    // k groups -> k(k-1)/2 rows, ordered like TukeyHSD: B-A, C-A, D-A, C-B, D-B, D-C.
    let x: Vec<f64> = (0..40).map(|i| (i as f64 * 0.37).sin()).collect();
    let g: Vec<u8> = (0..40).map(|i| (i % 4) as u8).collect();
    let res = tukey_hsd(&x, &g, 0.95).unwrap();
    let pairs: Vec<(u8, u8)> = res
        .comparisons
        .iter()
        .map(|c| (c.group1, c.group2))
        .collect();
    assert_eq!(pairs, vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]);
    assert_eq!(res.groups, vec![0, 1, 2, 3]);
    assert_eq!(res.group_sizes, vec![10; 4]);
}

#[test]
fn input_validation() {
    assert!(tukey_hsd(&[1.0, 2.0], &["a"], 0.95).is_err());
    assert!(tukey_hsd(&[1.0, 2.0], &["a", "a"], 0.95).is_err());
    assert!(tukey_hsd(&[1.0, 2.0, 3.0], &["a", "a", "b"], 1.5).is_err());
    assert!(pairwise_t_test(
        &[1.0, f64::NAN, 3.0],
        &["a", "b", "b"],
        true,
        PAdjustMethod::Holm,
        Alternative::TwoSided
    )
    .is_err());
    assert!(dunn_test(&[1.0, f64::NAN], &["a", "b"], PAdjustMethod::Holm).is_err());
    // A singleton group makes the pooled SD undefined (R: NA).
    let r = pairwise_t_test(
        &[1.0, 2.0, 3.0, 4.0],
        &["a", "a", "a", "b"],
        true,
        PAdjustMethod::Holm,
        Alternative::TwoSided,
    )
    .unwrap();
    assert!(r.comparisons[0].p_value.is_nan());
}
