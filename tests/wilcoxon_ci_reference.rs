//! Hodges-Lehmann estimates and confidence intervals of `mann_whitney_u` and
//! `wilcoxon_signed_rank` against R's `wilcox.test(..., conf.int = TRUE)`
//! (reference data: `R/generate_wilcox_ci_refs.R`), plus large-sample
//! regression tests for issue #19 (O(n^2) memory).

use anofox_statistics::{mann_whitney_u, wilcoxon_signed_rank, Alternative};
use std::path::PathBuf;

struct Case {
    kind: String,
    alternative: Alternative,
    correct: bool,
    exact: bool,
    conf_level: f64,
    mu: f64,
    estimate: f64,
    lower: f64,
    upper: f64,
    x: Vec<f64>,
    y: Vec<f64>,
}

fn parse_vec(s: &str) -> Vec<f64> {
    s.split_whitespace().map(|v| v.parse().unwrap()).collect()
}

fn load_cases() -> Vec<Case> {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("R/data/wilcox_ci_reference.csv");
    let mut rdr = csv::Reader::from_path(&path).expect("reference file");
    rdr.records()
        .map(|r| {
            let r = r.unwrap();
            Case {
                kind: r[0].to_string(),
                alternative: match &r[1] {
                    "two.sided" => Alternative::TwoSided,
                    "less" => Alternative::Less,
                    _ => Alternative::Greater,
                },
                correct: &r[2] == "TRUE",
                exact: &r[3] == "TRUE",
                conf_level: r[4].parse().unwrap(),
                mu: r[5].parse().unwrap(),
                estimate: r[6].parse().unwrap(),
                lower: r[7].parse().unwrap(),
                upper: r[8].parse().unwrap(),
                x: parse_vec(&r[9]),
                y: parse_vec(&r[10]),
            }
        })
        .collect()
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    if a.is_infinite() || b.is_infinite() {
        return a == b;
    }
    (a - b).abs() <= tol * (1.0 + b.abs())
}

#[test]
fn hodges_lehmann_ci_matches_r() {
    let cases = load_cases();
    assert!(cases.len() > 100);
    let mut failures = Vec::new();
    for (i, c) in cases.iter().enumerate() {
        let (est, ci) = if c.kind == "mw" {
            let r = mann_whitney_u(
                &c.x,
                &c.y,
                c.alternative,
                c.correct,
                c.exact,
                Some(c.conf_level),
                Some(c.mu),
            )
            .unwrap();
            (r.estimate.unwrap(), r.conf_int.unwrap())
        } else {
            let r = wilcoxon_signed_rank(
                &c.x,
                &c.y,
                c.alternative,
                c.correct,
                c.exact,
                Some(c.conf_level),
                Some(c.mu),
            )
            .unwrap();
            (r.estimate.unwrap(), r.conf_int.unwrap())
        };
        // Exact intervals are order statistics (bit-identical); the
        // asymptotic ones replicate R's uniroot iterations.
        let tol = if c.exact { 1e-14 } else { 1e-10 };
        if !(close(est, c.estimate, tol)
            && close(ci.lower, c.lower, tol)
            && close(ci.upper, c.upper, tol))
        {
            failures.push(format!(
                "case {i} ({} {:?} correct={} exact={} n={}): got {est} [{}, {}], R {} [{}, {}]",
                c.kind,
                c.alternative,
                c.correct,
                c.exact,
                c.x.len(),
                ci.lower,
                ci.upper,
                c.estimate,
                c.lower,
                c.upper
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} mismatches:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// Deterministic pseudo-random uniforms (no RNG dependency).
fn lcg(n: usize, seed: u64, shift: f64) -> Vec<f64> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64) / (1u64 << 53) as f64 + shift
        })
        .collect()
}

/// Issue #19: 0.4.4 materialised all n1*n2 differences (80 GB here). Now
/// O(n log n) time and O(n) memory: ~0.4 s in debug, ~60 ms / 13 MB peak RSS
/// in release.
#[test]
fn large_sample_ci_completes() {
    let n = 100_000;
    let x = lcg(n, 1, 0.0);
    let y = lcg(n, 2, 0.1);
    let t = std::time::Instant::now();
    let mw = mann_whitney_u(&x, &y, Alternative::TwoSided, true, true, Some(0.95), None).unwrap();
    let ci = mw.conf_int.unwrap();
    let est = mw.estimate.unwrap();
    assert!(ci.lower < est && est < ci.upper);
    assert!((est + 0.1).abs() < 0.01, "estimate {est}");
    let wsr =
        wilcoxon_signed_rank(&x, &y, Alternative::TwoSided, true, true, Some(0.95), None).unwrap();
    let ci = wsr.conf_int.unwrap();
    let est = wsr.estimate.unwrap();
    assert!(ci.lower < est && est < ci.upper);
    assert!((est + 0.1).abs() < 0.01, "estimate {est}");
    eprintln!("n = {n} per group: MW + WSR with CI in {:?}", t.elapsed());
}

/// Moderate sizes run in the normal test suite: the old implementation needed
/// n1*n2*8 bytes (here ~200 MB) and seconds; the new one is milliseconds.
#[test]
fn moderate_sample_ci_is_fast() {
    let n = 5_000;
    let x = lcg(n, 3, 0.0);
    let y = lcg(n, 4, 0.05);
    let mw = mann_whitney_u(&x, &y, Alternative::Less, false, false, Some(0.9), None).unwrap();
    let ci = mw.conf_int.unwrap();
    assert!(ci.lower == f64::NEG_INFINITY && ci.upper.is_finite());
    let wsr =
        wilcoxon_signed_rank(&x, &y, Alternative::Greater, false, false, Some(0.9), None).unwrap();
    let ci = wsr.conf_int.unwrap();
    assert!(ci.lower.is_finite() && ci.upper == f64::INFINITY);
}

/// `exact = true` on a sample far beyond the exact-distribution limits falls
/// back to the normal approximation instead of running the exact DP.
#[test]
fn exact_request_on_large_sample_uses_normal_approximation() {
    let x = lcg(3_000, 5, 0.0);
    let y = lcg(3_000, 6, 0.0);
    let a = mann_whitney_u(&x, &y, Alternative::TwoSided, false, true, None, None).unwrap();
    let b = mann_whitney_u(&x, &y, Alternative::TwoSided, false, false, None, None).unwrap();
    assert_eq!(a.p_value, b.p_value);
    let a = wilcoxon_signed_rank(&x, &y, Alternative::TwoSided, false, true, None, None).unwrap();
    let b = wilcoxon_signed_rank(&x, &y, Alternative::TwoSided, false, false, None, None).unwrap();
    assert_eq!(a.p_value, b.p_value);
}
