//! Seeded resampling must give the same results on every target (64-bit,
//! 32-bit, wasm32). `rand`'s `gen_range` over `usize` depends on the pointer
//! width, so all index draws sample `u64` (see `src/utils/rng.rs`). These
//! values are pinned from x86_64 (identical before and after the fix);
//! CI also runs the test suite on i686-unknown-linux-gnu (32-bit).

use anofox_statistics::drift::{embedding_distance_drift, EmbeddingDistance, EmbeddingThreshold};
use anofox_statistics::{
    bootstrap_mean_ci, distance_cor_test, model_confidence_set_with_method, permutation_t_test,
    Alternative, BootstrapMethod, CircularBlockBootstrap, MCSStatistic, MovingBlockBootstrap,
    StationaryBootstrap,
};

fn series(n: usize, a: f64, b: f64) -> Vec<f64> {
    (0..n)
        .map(|i| (i as f64 * a).sin() + 0.3 * (i as f64 * b).cos())
        .collect()
}

#[test]
fn bootstrap_indices_are_pinned() {
    let s = StationaryBootstrap::new(3.0, Some(7)).sample_indices(50, 12);
    let c = CircularBlockBootstrap::new(4, Some(7)).sample_indices(50, 12);
    let m = MovingBlockBootstrap::new(3, Some(7)).sample_indices(50, 12);
    println!("{:?}\n{:?}\n{:?}", s, c, m);
    assert_eq!(s, PINNED_STATIONARY);
    assert_eq!(c, PINNED_CIRCULAR);
    assert_eq!(m, PINNED_MOVING);
}

#[test]
fn seeded_results_are_pinned() {
    let x = series(40, 0.37, 1.3);
    let y = series(35, 0.53, 0.7);
    let iid = bootstrap_mean_ci(&x, 200, 0.9, 0, Some(11)).unwrap();
    let blk = bootstrap_mean_ci(&x, 200, 0.9, 4, Some(11)).unwrap();
    let perm = permutation_t_test(&x, &y, Alternative::TwoSided, 199, Some(3)).unwrap();
    let dcor = distance_cor_test(&x[..35], &y, 199, Some(5)).unwrap();

    let losses: Vec<Vec<f64>> = (0..5)
        .map(|k| {
            series(60, 0.3 + 0.07 * k as f64, 1.1)
                .iter()
                .map(|v| v * v + 0.05 * k as f64)
                .collect()
        })
        .collect();
    let mcs = model_confidence_set_with_method(
        &losses,
        0.1,
        MCSStatistic::Max,
        499,
        BootstrapMethod::MovingBlock { block_length: 3 },
        Some(7),
    )
    .unwrap();

    let emb_ref: Vec<Vec<f64>> = (0..30).map(|i| vec![x[i], y[i]]).collect();
    let emb_cur: Vec<Vec<f64>> = (0..20).map(|i| vec![y[i] + 0.2, x[i]]).collect();
    let emb = embedding_distance_drift(
        &emb_ref,
        &emb_cur,
        EmbeddingDistance::Euclidean,
        EmbeddingThreshold::Bootstrap {
            n_bootstrap: 50,
            quantile: 0.95,
            seed: Some(9),
        },
    )
    .unwrap();

    let got = [
        iid.conf_int_lower,
        iid.conf_int_upper,
        blk.conf_int_lower,
        blk.conf_int_upper,
        perm.p_value,
        dcor.p_value.unwrap(),
        mcs.mcs_p_values[0],
        mcs.mcs_p_values[1],
        emb.threshold,
    ];
    println!("{:?}", got.map(|v| format!("{:.17e}", v)));
    for (i, (g, w)) in got.iter().zip(PINNED_VALUES).enumerate() {
        assert_eq!(*g, w, "value {}: {} vs {}", i, g, w);
    }
}

const PINNED_STATIONARY: [usize; 12] = [8, 9, 10, 11, 12, 42, 43, 44, 19, 20, 21, 22];
const PINNED_CIRCULAR: [usize; 12] = [8, 9, 10, 11, 35, 36, 37, 38, 36, 37, 38, 39];
const PINNED_MOVING: [usize; 12] = [7, 8, 9, 8, 9, 10, 17, 18, 19, 17, 18, 19];
#[allow(clippy::excessive_precision)]
const PINNED_VALUES: [f64; 9] = [
    -7.95089646008166895e-2,
    2.91638698708687916e-1,
    -2.23453709441231441e-1,
    4.76276809066645934e-1,
    5.10000000000000009e-1,
    4.14999999999999980e-1,
    1.0,
    6.61322645290581157e-1,
    5.73436420319634044e-1,
];
