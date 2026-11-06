//! PCA with scaling (prcomp(scale. = TRUE), predict.prcomp), the MMD
//! median-heuristic bandwidths and the evidently drift defaults. Reference
//! data: `R/generate_pca_mmd_refs.R` -> `R/data/pca_mmd_reference.csv`.

use anofox_statistics::drift::{
    defaults, embedding_distance_drift, embedding_mmd_bandwidth, embedding_mmd_drift, DriftOptions,
    DriftTest, EmbeddingDistance, EmbeddingThreshold,
};
use anofox_statistics::modern::{
    median_heuristic_bandwidth, median_heuristic_bandwidth_1d, mmd_test, mmd_test_1d, Kernel,
};
use anofox_statistics::multivariate::{pca, pca_with_options, PcaOptions};
use std::collections::HashMap;

type Refs = HashMap<(String, String), Vec<f64>>;

fn refs() -> Refs {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/R/data/pca_mmd_reference.csv");
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

fn rows(flat: &[f64], d: usize) -> Vec<Vec<f64>> {
    flat.chunks(d).map(|c| c.to_vec()).collect()
}

fn close_all(got: &[f64], want: &[f64], tol: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{} length", what);
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tol * w.abs().max(1.0),
            "{}[{}]: {} vs {}",
            what,
            i,
            g,
            w
        );
    }
}

#[test]
fn pca_scaled_matches_prcomp() {
    let m = refs();
    for case in ["pca_tall", "pca_wide"] {
        let d = get(&m, case, "dims")[1] as usize;
        let x = rows(get(&m, case, "in_x"), d);
        let p = pca_with_options(&x, &PcaOptions { scale: true }).unwrap();
        close_all(
            &p.center,
            get(&m, case, "center"),
            1e-12,
            &format!("{} center", case),
        );
        close_all(
            p.scale.as_ref().unwrap(),
            get(&m, case, "scale"),
            1e-12,
            &format!("{} scale", case),
        );
        close_all(
            &p.sdev,
            get(&m, case, "sdev"),
            1e-10,
            &format!("{} sdev", case),
        );
        let rot: Vec<f64> = p.rotation.iter().flatten().copied().collect();
        close_all(
            &rot,
            get(&m, case, "rotation"),
            1e-8,
            &format!("{} rotation", case),
        );
        close_all(
            &p.explained_variance_ratio,
            get(&m, case, "explained"),
            1e-10,
            &format!("{} explained", case),
        );
        let newdata = rows(get(&m, case, "in_new"), d);
        let sc: Vec<f64> = p
            .transform(&newdata, p.n_components())
            .unwrap()
            .into_iter()
            .flatten()
            .collect();
        close_all(
            &sc,
            get(&m, case, "scores"),
            1e-8,
            &format!("{} scores", case),
        );
    }
    // Default and `pca` are unscaled and identical.
    let x = rows(get(&m, "pca_tall", "in_x"), 4);
    let a = pca(&x).unwrap();
    assert_eq!(a, pca_with_options(&x, &PcaOptions::default()).unwrap());
    assert!(a.scale.is_none());
    // A constant column cannot be scaled (as prcomp).
    let c: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64, 1.0]).collect();
    assert!(pca_with_options(&c, &PcaOptions { scale: true }).is_err());
    assert!(pca(&c).is_ok());
}

#[test]
fn mmd_median_heuristic_bandwidths_match_r() {
    let m = refs();
    let d = get(&m, "mmd", "dim")[0] as usize;
    let x = rows(get(&m, "mmd", "in_x"), d);
    let y = rows(get(&m, "mmd", "in_y"), d);
    let bw = median_heuristic_bandwidth(&x, &y).unwrap();
    close_all(&[bw], get(&m, "mmd", "bandwidth"), 1e-12, "bandwidth");
    let r = mmd_test(&x, &y, Kernel::GaussianMedianHeuristic, 99, Some(1)).unwrap();
    close_all(
        &[r.statistic],
        get(&m, "mmd", "statistic"),
        1e-10,
        "statistic",
    );
    // Same as passing the bandwidth explicitly (same seed, same permutations).
    let e = mmd_test(&x, &y, Kernel::Gaussian { bandwidth: bw }, 99, Some(1)).unwrap();
    assert_eq!(r.statistic, e.statistic);
    assert_eq!(r.p_value, e.p_value);

    let eb = embedding_mmd_bandwidth(&x, &y).unwrap();
    close_all(
        &[eb],
        get(&m, "mmd", "embedding_bandwidth"),
        1e-12,
        "embedding bandwidth",
    );

    let x1 = get(&m, "mmd1", "in_x");
    let y1 = get(&m, "mmd1", "in_y");
    let b1 = median_heuristic_bandwidth_1d(x1, y1).unwrap();
    close_all(&[b1], get(&m, "mmd1", "bandwidth"), 1e-12, "1-D bandwidth");
    // 1-D rows: the multivariate heuristic and mmd_test_1d agree.
    let xr: Vec<Vec<f64>> = x1.iter().map(|&v| vec![v]).collect();
    let yr: Vec<Vec<f64>> = y1.iter().map(|&v| vec![v]).collect();
    assert_eq!(median_heuristic_bandwidth(&xr, &yr).unwrap(), b1);
    let a = mmd_test_1d(x1, y1, 99, Some(4)).unwrap();
    let b = mmd_test(&xr, &yr, Kernel::GaussianMedianHeuristic, 99, Some(4)).unwrap();
    assert_eq!(a.statistic, b.statistic);
    assert_eq!(a.p_value, b.p_value);
}

#[test]
fn evidently_defaults_are_exposed() {
    assert_eq!(defaults::EMBEDDING_DISTANCE_THRESHOLD, 0.2);
    assert_eq!(defaults::EMBEDDING_MMD_THRESHOLD, 0.015);
    assert_eq!(defaults::EMBEDDING_DISTANCE_QUANTILE, 0.95);
    assert_eq!(defaults::EMBEDDING_MMD_QUANTILE, 0.05);
    assert_eq!(defaults::EMBEDDING_BOOTSTRAP_RESAMPLES, 100);
    assert_eq!(
        defaults::EMBEDDING_RATIO_COMPONENT_TEST,
        DriftTest::Wasserstein
    );
    assert_eq!(
        DriftTest::Ks.default_threshold(),
        defaults::P_VALUE_THRESHOLD
    );
    assert_eq!(
        DriftTest::Wasserstein.default_threshold(),
        defaults::DISTANCE_THRESHOLD
    );
    assert_eq!(
        DriftOptions::default().n_permutations,
        defaults::N_PERMUTATIONS
    );
    assert_eq!(
        EmbeddingThreshold::EVIDENTLY_DISTANCE,
        EmbeddingThreshold::Fixed(0.2)
    );
    assert_eq!(
        EmbeddingThreshold::evidently_mmd_bootstrap(Some(3)),
        EmbeddingThreshold::Bootstrap {
            n_bootstrap: 100,
            quantile: 0.05,
            seed: Some(3)
        }
    );
    let r: Vec<Vec<f64>> = (0..30)
        .map(|i| vec![i as f64 * 0.1, (i % 7) as f64])
        .collect();
    let c: Vec<Vec<f64>> = (0..25)
        .map(|i| vec![i as f64 * 0.1 + 1.0, (i % 5) as f64])
        .collect();
    let a = embedding_distance_drift(
        &r,
        &c,
        EmbeddingDistance::Euclidean,
        EmbeddingThreshold::EVIDENTLY_DISTANCE,
    )
    .unwrap();
    assert_eq!(a.threshold, 0.2);
    assert_eq!(
        EmbeddingThreshold::evidently_distance_bootstrap(None),
        EmbeddingThreshold::Bootstrap {
            n_bootstrap: 100,
            quantile: 0.95,
            seed: None
        }
    );
    let a = embedding_distance_drift(
        &r,
        &c,
        EmbeddingDistance::Euclidean,
        EmbeddingThreshold::evidently_distance_bootstrap(Some(2)),
    )
    .unwrap();
    assert!(a.threshold.is_finite());
    let b =
        embedding_mmd_drift(&r, &c, EmbeddingThreshold::evidently_mmd_bootstrap(Some(1))).unwrap();
    assert!(b.threshold.is_finite());
}
