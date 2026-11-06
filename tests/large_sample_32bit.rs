//! Order statistics of ~N^2 / 2 pairs for N > 65 536 values: the pair counts
//! and ranks overflow `usize` on 32-bit targets (wasm32, i686), so they are
//! `u64` internally. CI runs these on i686-unknown-linux-gnu. The samples are
//! integer grids whose pair-difference distributions are known in closed
//! form; the selection is O(N log^2 N), so the tests stay fast.

use anofox_statistics::drift::empirical_mmd_bandwidth;
use anofox_statistics::modern::median_heuristic_bandwidth_1d;
use anofox_statistics::{tost_wilcoxon_two_sample, EquivalenceBounds};

/// k-th smallest (0-based) |i - j|, i < j, over 0..n: difference d occurs
/// n - d times.
fn kth_abs_diff_grid(n: u64, k: u64) -> f64 {
    let mut acc = 0u64;
    for d in 1..n {
        acc += n - d;
        if k < acc {
            return d as f64;
        }
    }
    unreachable!()
}

#[test]
fn mmd_1d_median_heuristic_beyond_65k_values() {
    let n: u64 = 70_000;
    let x: Vec<f64> = (0..35_000).map(|i| i as f64).collect();
    let y: Vec<f64> = (35_000..n).map(|i| i as f64).collect();
    let pairs = n * (n - 1) / 2; // 2.45e9 > 2^31: overflowed n * (n - 1) on 32-bit
    let want = (kth_abs_diff_grid(n, pairs / 2 - 1) + kth_abs_diff_grid(n, pairs / 2)) / 2.0;
    let got = median_heuristic_bandwidth_1d(&x, &y).unwrap();
    assert_eq!(got, want);
}

#[test]
fn empirical_mmd_bandwidth_beyond_65k_values() {
    // m^2 = 4.9e9 > u32::MAX.
    let m: u64 = 70_000;
    let x: Vec<f64> = (0..m).map(|i| i as f64).collect();
    // numpy "nearest" index into the m^2 sorted squared differences
    // (m zeros, then every pair difference twice).
    let count = m * m;
    let lo = (count - 2) / 2;
    let idx = lo + (lo & 1); // round half to even (no is_multiple_of: MSRV 1.86)
    let d = kth_abs_diff_grid(m, (idx - m) / 2);
    let want = (0.5 * d * d).sqrt();
    assert_eq!(empirical_mmd_bandwidth(&x).unwrap(), want);
}

#[test]
fn wilcoxon_tost_hodges_lehmann_beyond_2_pow_32_pairs() {
    // 70 000 x 70 000 = 4.9e9 pairwise differences.
    let x: Vec<f64> = (0..70_000).map(|i| i as f64 + 0.25).collect();
    let y: Vec<f64> = (0..70_000).map(|i| i as f64).collect();
    let r = tost_wilcoxon_two_sample(
        &x,
        &y,
        &EquivalenceBounds::Symmetric { delta: 1000.0 },
        0.05,
    )
    .unwrap();
    // The differences x_i - y_j are symmetric around 0.25.
    assert_eq!(r.estimate, 0.25);
    assert!(r.ci.0 <= 0.25 && 0.25 <= r.ci.1);
}
