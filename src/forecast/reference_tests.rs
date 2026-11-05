//! Exact checks of the SPA and MCS cores against R with the same bootstrap
//! indices (`R/generate_spa_mcs_refs.R` -> `R/data/spa_mcs_reference.csv`,
//! cases `exact_*`). Monte-Carlo comparisons with R's own random numbers are
//! in `tests/spa_mcs_reference.rs`.

use super::mcs::mcs_core;
use super::spa_common::{spa_core, spa_omega2};
use super::MCSStatistic;
use std::collections::HashMap;

type Refs = HashMap<(String, String), Vec<f64>>;

fn refs() -> Refs {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/R/data/spa_mcs_reference.csv");
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

fn models(m: &Refs, case: &str) -> Vec<Vec<f64>> {
    (0..)
        .map_while(|k| {
            m.get(&(case.to_string(), format!("in_model{}", k)))
                .cloned()
        })
        .collect()
}

/// Index vectors of length n, replication by replication.
fn index_draws(idx: &[f64], n: usize) -> impl FnMut() -> Vec<usize> + '_ {
    let mut b = 0;
    move || {
        let v = idx[b * n..(b + 1) * n]
            .iter()
            .map(|&x| x as usize)
            .collect();
        b += 1;
        v
    }
}

#[test]
fn spa_matches_r_with_same_indices() {
    let m = refs();
    let bench = get(&m, "spa", "in_bench");
    let d: Vec<Vec<f64>> = models(&m, "spa")
        .iter()
        .map(|l| bench.iter().zip(l).map(|(b, x)| b - x).collect())
        .collect();
    let n = bench.len();
    let l = get(&m, "exact_spa", "block_length")[0];
    let idx = get(&m, "exact_spa", "in_idx");
    let n_boot = idx.len() / n;

    for (k, &w) in get(&m, "exact_spa", "omega").iter().enumerate() {
        let got = spa_omega2(&d[k], l).sqrt();
        assert!(
            (got - w).abs() < 1e-10 * w,
            "omega[{}]: {} vs {}",
            k,
            got,
            w
        );
    }
    let out = spa_core(&d, l, n_boot, index_draws(idx, n));
    let stat = get(&m, "exact_spa", "statistic")[0];
    assert!(
        (out.statistic - stat).abs() < 1e-10,
        "{} vs {}",
        out.statistic,
        stat
    );
    let p = get(&m, "exact_spa", "p");
    let got = [out.p_value_lower, out.p_value_consistent, out.p_value_upper];
    for i in 0..3 {
        assert!(
            (got[i] - p[i]).abs() < 1e-12,
            "p[{}]: {} vs {}",
            i,
            got[i],
            p[i]
        );
    }
    assert!(got[0] < got[1] && got[1] < got[2], "{:?}", got);
    assert_eq!(out.best_model_idx, Some(0));
}

#[test]
fn mcs_matches_r_with_same_indices() {
    let m = refs();
    let losses = models(&m, "mcs");
    let n = losses[0].len();
    for (stat, name) in [(MCSStatistic::Max, "Tmax"), (MCSStatistic::Range, "TR")] {
        let case = format!("exact_mcs_{}", name);
        let idx = get(&m, &case, "in_idx");
        let n_boot = get(&m, &case, "B")[0] as usize;
        let r = mcs_core(&losses, 0.1, stat, n_boot, index_draws(idx, n)).unwrap();
        let order: Vec<usize> = get(&m, &case, "order")
            .iter()
            .map(|&x| x as usize)
            .collect();
        assert_eq!(r.elimination_order, order, "{}", name);
        let step_p = get(&m, &case, "step_p");
        for (s, step) in r.elimination_sequence.iter().enumerate() {
            assert!(
                (step.p_value - step_p[s]).abs() < 1e-12,
                "{} step {}",
                name,
                s
            );
        }
        for (i, (&g, &w)) in r
            .mcs_p_values
            .iter()
            .zip(get(&m, &case, "mcs_p"))
            .enumerate()
        {
            assert!(
                (g - w).abs() < 1e-12,
                "{} model {}: {} vs {}",
                name,
                i,
                g,
                w
            );
        }
        let included: Vec<usize> = (0..losses.len())
            .filter(|&i| get(&m, &case, "mcs_p")[i] >= 0.1)
            .collect();
        assert_eq!(r.included_models, included, "{}", name);
    }
}
