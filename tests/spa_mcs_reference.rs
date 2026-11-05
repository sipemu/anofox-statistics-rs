//! SPA (Hansen 2005), MSPE-adjusted SPA and the Model Confidence Set against
//! R, with each side's own random numbers: the statistics must agree to
//! rounding and the bootstrap p-values within Monte-Carlo tolerance
//! (B = 20000 on both sides; the MC standard error of a difference is below
//! 0.005). Reference data: `R/generate_spa_mcs_refs.R` ->
//! `R/data/spa_mcs_reference.csv` (MCS: R package `MCS`; SPA: base R).
//! Exact comparisons with shared bootstrap indices are unit tests in
//! `src/forecast/reference_tests.rs`.

use anofox_statistics::{
    model_confidence_set, model_confidence_set_with_method, mspe_adjusted_spa, spa_test,
    BootstrapMethod, MCSStatistic,
};
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

// SPA: the MC SD of a p-value difference is <= 0.005; the MCS p-values
// vary a little more because the variances are bootstrap estimates too.
const MC_TOL: f64 = 0.025;
const B: usize = 20000;

#[test]
fn spa_test_matches_r_hansen_2005() {
    let m = refs();
    let bench = get(&m, "spa", "in_bench");
    let l = get(&m, "mc_spa", "block_length")[0];
    let r = spa_test(bench, &models(&m, "spa"), B, l, Some(11)).unwrap();
    let stat = get(&m, "mc_spa", "statistic")[0];
    assert!(
        (r.statistic - stat).abs() < 1e-10,
        "{} vs {}",
        r.statistic,
        stat
    );
    let p = get(&m, "mc_spa", "p");
    let got = [r.p_value_lower, r.p_value_consistent, r.p_value_upper];
    println!("SPA p (l, c, u): rust {:?} R {:?}", got, p);
    for i in 0..3 {
        assert!(
            (got[i] - p[i]).abs() < MC_TOL,
            "p[{}]: {} vs {}",
            i,
            got[i],
            p[i]
        );
    }
    assert!(got[0] <= got[1] && got[1] <= got[2]);
    assert_eq!(r.best_model_idx, Some(0));
}

#[test]
fn mspe_adjusted_spa_matches_r() {
    let m = refs();
    let bench = get(&m, "mspe", "in_bench");
    let l = get(&m, "mc_mspe", "block_length")[0];
    let r = mspe_adjusted_spa(bench, &models(&m, "mspe"), B, l, Some(12)).unwrap();
    let stat = get(&m, "mc_mspe", "statistic")[0];
    assert!(
        (r.statistic - stat).abs() < 1e-10,
        "{} vs {}",
        r.statistic,
        stat
    );
    let p = get(&m, "mc_mspe", "p");
    let got = [r.p_value_lower, r.p_value_consistent, r.p_value_upper];
    println!("MSPE-SPA p (l, c, u): rust {:?} R {:?}", got, p);
    for i in 0..3 {
        assert!(
            (got[i] - p[i]).abs() < MC_TOL,
            "p[{}]: {} vs {}",
            i,
            got[i],
            p[i]
        );
    }
}

#[test]
fn mcs_matches_r_mcsprocedure() {
    let m = refs();
    let losses = models(&m, "mcs");
    for (stat, name) in [(MCSStatistic::Max, "Tmax"), (MCSStatistic::Range, "TR")] {
        for k in [1usize, 3] {
            let case = format!("mc_mcs_{}_k{}", name, k);
            let r = model_confidence_set_with_method(
                &losses,
                0.1,
                stat,
                B,
                BootstrapMethod::MovingBlock { block_length: k },
                Some(3),
            )
            .unwrap();
            let order: Vec<usize> = get(&m, &case, "order")
                .iter()
                .map(|&x| x as usize)
                .collect();
            assert_eq!(r.elimination_order, order, "{}", case);
            let want = get(&m, &case, "mcs_p");
            println!("{}: rust {:?} R {:?}", case, r.mcs_p_values, want);
            for i in 0..losses.len() {
                assert!(
                    (r.mcs_p_values[i] - want[i]).abs() < MC_TOL,
                    "{} model {}: {} vs {}",
                    case,
                    i,
                    r.mcs_p_values[i],
                    want[i]
                );
            }
            let included: Vec<usize> = (0..losses.len()).filter(|&i| want[i] >= 0.1).collect();
            assert_eq!(r.included_models, included, "{}", case);
        }
    }
}

#[test]
fn mcs_is_reproducible_with_many_models() {
    // Previously the model pairs were iterated in HashMap order, so a fixed
    // seed did not give a fixed result with >= 3 models.
    let m = refs();
    let losses = models(&m, "mcs");
    for stat in [MCSStatistic::Max, MCSStatistic::Range] {
        let a = model_confidence_set(&losses, 0.1, stat, 500, 3.0, Some(42)).unwrap();
        for _ in 0..5 {
            let b = model_confidence_set(&losses, 0.1, stat, 500, 3.0, Some(42)).unwrap();
            assert_eq!(a.mcs_p_values, b.mcs_p_values);
            assert_eq!(a.elimination_order, b.elimination_order);
            assert_eq!(a.included_models, b.included_models);
        }
    }
}
