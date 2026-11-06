//! Metric pack (classification, regression, ranking) against scikit-learn /
//! evidently (`R/generate_metrics_refs.py` ->
//! `R/data/metrics_sklearn_reference.csv`) and R yardstick
//! (`R/generate_monitoring_refs.R` -> `R/data/monitoring_r_reference.csv`).

use anofox_statistics::metrics::{
    classification_metrics, confusion_matrix, ndcg_at_k, ranking_metrics, regression_metrics,
    Average, ClassificationOptions, RankingOptions,
};
use std::collections::HashMap;

type Refs = HashMap<(String, String), Vec<f64>>;

fn refs(file: &str) -> Refs {
    let p = format!("{}/R/data/{}", env!("CARGO_MANIFEST_DIR"), file);
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

fn ints(v: &[f64]) -> Vec<i64> {
    v.iter().map(|&x| x as i64).collect()
}

fn close(got: f64, want: f64, tol: f64, what: &str) {
    assert!(
        (got - want).abs() <= tol * want.abs().max(1.0),
        "{}: {} vs {}",
        what,
        got,
        want
    );
}

fn opts(average: Average<i64>, zero_division: f64) -> ClassificationOptions<i64> {
    ClassificationOptions {
        average,
        zero_division,
    }
}

#[test]
fn classification_matches_sklearn() {
    let m = refs("metrics_sklearn_reference.csv");
    for (case, averages) in [
        ("binary", vec!["binary", "macro", "micro", "weighted"]),
        ("multiclass", vec!["macro", "micro", "weighted"]),
        ("multiclass_unsupported", vec!["macro", "micro", "weighted"]),
        ("constant_pred", vec!["binary", "macro"]),
    ] {
        let yt = ints(get(&m, case, "y_true"));
        let yp = ints(get(&m, case, "y_pred"));
        for avg in averages {
            for zd in [0.0, 1.0] {
                let a = match avg {
                    "binary" => Average::Binary(1),
                    "macro" => Average::Macro,
                    "micro" => Average::Micro,
                    _ => Average::Weighted,
                };
                let r = classification_metrics(&yt, &yp, &opts(a, zd)).unwrap();
                let want = get(&m, case, &format!("{}_zd{}", avg, zd as i32));
                let what = format!("{} {} zd{}", case, avg, zd);
                close(r.precision, want[0], 1e-12, &format!("{} precision", what));
                close(r.recall, want[1], 1e-12, &format!("{} recall", what));
                close(r.f1, want[2], 1e-12, &format!("{} f1", what));
                close(r.accuracy, get(&m, case, "accuracy")[0], 1e-12, &what);
                close(
                    r.balanced_accuracy,
                    get(&m, case, "balanced_accuracy")[0],
                    1e-12,
                    &what,
                );
                close(r.mcc, get(&m, case, "mcc")[0], 1e-12, &what);
                close(r.cohen_kappa, get(&m, case, "kappa")[0], 1e-12, &what);
                let cm: Vec<f64> = r
                    .confusion
                    .counts
                    .iter()
                    .flatten()
                    .map(|&c| c as f64)
                    .collect();
                assert_eq!(cm, get(&m, case, "confusion"), "{} confusion", what);
            }
        }
    }
}

#[test]
fn classification_matches_yardstick() {
    let m = refs("monitoring_r_reference.csv");
    for (case, estimators) in [
        ("binary", vec!["binary", "macro", "micro", "macro_weighted"]),
        ("multiclass", vec!["macro", "micro", "macro_weighted"]),
    ] {
        let yt = ints(get(&m, case, "y_true"));
        let yp = ints(get(&m, case, "y_pred"));
        for e in estimators {
            let a = match e {
                "binary" => Average::Binary(1),
                "macro" => Average::Macro,
                "micro" => Average::Micro,
                _ => Average::Weighted,
            };
            let r = classification_metrics(&yt, &yp, &opts(a, 0.0)).unwrap();
            let want = get(&m, case, e);
            let got = [r.precision, r.recall, r.f1, r.specificity];
            for (i, name) in ["precision", "recall", "f1", "spec"].iter().enumerate() {
                close(got[i], want[i], 1e-12, &format!("{} {} {}", case, e, name));
            }
            close(r.accuracy, get(&m, case, "accuracy")[0], 1e-12, case);
            close(r.mcc, get(&m, case, "mcc")[0], 1e-12, case);
            close(r.cohen_kappa, get(&m, case, "kappa")[0], 1e-12, case);
        }
    }
}

#[test]
fn classification_label_types_and_errors() {
    let yt = ["cat", "dog", "dog", "bird"];
    let yp = ["cat", "dog", "cat", "bird"];
    let cm = confusion_matrix(&yt, &yp).unwrap();
    assert_eq!(cm.labels, vec!["bird", "cat", "dog"]);
    assert_eq!(cm.counts, vec![vec![1, 0, 0], vec![0, 1, 0], vec![0, 1, 1]]);
    let o = ClassificationOptions {
        average: Average::Binary("dog"),
        zero_division: 0.0,
    };
    assert!(classification_metrics(&yt, &yp, &o).is_err()); // three labels
    let b = classification_metrics(
        &[true, true, false],
        &[true, false, false],
        &ClassificationOptions {
            average: Average::Binary(true),
            zero_division: 0.0,
        },
    )
    .unwrap();
    assert!((b.recall - 0.5).abs() < 1e-12 && (b.specificity - 1.0).abs() < 1e-12);
    // A positive class that never occurs: zero_division for P / R / F1.
    let z = classification_metrics(
        &[0, 0],
        &[0, 0],
        &ClassificationOptions {
            average: Average::Binary(1),
            zero_division: 1.0,
        },
    )
    .unwrap();
    assert_eq!(
        (z.precision, z.recall, z.f1, z.specificity),
        (1.0, 1.0, 1.0, 1.0)
    );
    assert!(z.cohen_kappa.is_nan());
    assert_eq!(z.mcc, 0.0);
    assert!(classification_metrics::<i64>(&[], &[], &Default::default()).is_err());
    assert!(classification_metrics(&[1], &[1, 2], &Default::default()).is_err());
}

#[test]
fn regression_matches_sklearn_and_yardstick() {
    let m = refs("metrics_sklearn_reference.csv");
    let r = regression_metrics(
        get(&m, "regression", "y_true"),
        get(&m, "regression", "y_pred"),
    )
    .unwrap();
    for (name, got) in [
        ("mae", r.mae),
        ("mse", r.mse),
        ("rmse", r.rmse),
        ("mape", r.mape),
        ("median_ae", r.median_ae),
        ("r2", r.r2),
        ("explained_variance", r.explained_variance),
        ("max_error", r.max_error),
    ] {
        close(got, get(&m, "regression", name)[0], 1e-12, name);
    }
    let c1 = regression_metrics(&[2.0, 2.0, 2.0], &[2.0, 2.0, 2.0]).unwrap();
    let c2 = regression_metrics(&[2.0, 2.0, 2.0], &[2.0, 2.0, 3.0]).unwrap();
    let c3 = regression_metrics(&[2.0, 2.0, 2.0], &[1.0, 2.0, 3.0]).unwrap();
    assert_eq!([c1.r2, c2.r2], [1.0, 0.0]);
    assert_eq!(get(&m, "regression_const", "r2"), &[1.0, 0.0]);
    assert_eq!([c1.explained_variance, c3.explained_variance], [1.0, 0.0]);
    assert_eq!(
        get(&m, "regression_const", "explained_variance"),
        &[1.0, 0.0]
    );

    let y = refs("monitoring_r_reference.csv");
    let r = regression_metrics(
        get(&y, "regression", "y_true"),
        get(&y, "regression", "y_pred"),
    )
    .unwrap();
    for (name, got) in [
        ("mae", r.mae),
        ("rmse", r.rmse),
        ("mape", r.mape),
        ("smape", r.smape),
        ("median_ae", r.median_ae),
        ("r2", r.r2),
        ("explained_variance", r.explained_variance),
        ("max_error", r.max_error),
    ] {
        close(got, get(&y, "regression", name)[0], 1e-12, name);
    }
    let zero = regression_metrics(&[0.0, 1.0], &[0.0, 3.0]).unwrap();
    assert!((zero.smape - 0.5).abs() < 1e-12); // (0 + 2 / 2) / 2
    assert!(regression_metrics(&[1.0], &[f64::NAN]).is_err());
    assert!(regression_metrics(&[1.0], &[1.0, 2.0]).is_err());
    assert!(regression_metrics(&[1.0], &[1.0]).unwrap().r2.is_nan());
}

#[test]
fn ranking_matches_sklearn_and_evidently() {
    let m = refs("metrics_sklearn_reference.csv");
    let q = ints(get(&m, "ranking", "query"));
    let rel = get(&m, "ranking", "relevance");
    let score = get(&m, "ranking", "score");
    for k in [3usize, 5] {
        for min_rel in [None, Some(2.0)] {
            for nf in [false, true] {
                let key = format!(
                    "k{}_min{}_nf{}",
                    k,
                    min_rel.unwrap_or(0.0) as i32,
                    nf as i32
                );
                let r = ranking_metrics(
                    &q,
                    rel,
                    score,
                    &RankingOptions {
                        k,
                        min_relevance: min_rel,
                        include_no_feedback: nf,
                    },
                )
                .unwrap();
                let ev = get(&m, "ranking", &format!("evidently_{}", key));
                let got = [r.ndcg, r.map, r.mrr, r.precision, r.recall, r.hit_rate];
                for (i, name) in ["ndcg", "map", "mrr", "precision", "recall", "hit_rate"]
                    .iter()
                    .enumerate()
                {
                    close(got[i], ev[i], 1e-12, &format!("{} {}", key, name));
                }
                close(
                    r.ndcg,
                    get(&m, "ranking", &format!("sklearn_ndcg_{}", key))[0],
                    1e-12,
                    &key,
                );
            }
        }
    }
    let tied = get(&m, "ranking", "tied_score");
    for k in [3usize, 5] {
        let r = ranking_metrics(
            &q,
            rel,
            tied,
            &RankingOptions {
                k,
                min_relevance: None,
                include_no_feedback: true,
            },
        )
        .unwrap();
        close(
            r.ndcg,
            get(&m, "ranking", &format!("sklearn_ndcg_tied_k{}", k))[0],
            1e-12,
            "tied ndcg",
        );
    }
}

#[test]
fn ndcg_single_query_and_errors() {
    // sklearn: ndcg_score([[10, 0, 0, 1, 5]], [[.1, .2, .3, 4, 70]]) = 0.6956940443813076
    let v = ndcg_at_k(&[10.0, 0.0, 0.0, 1.0, 5.0], &[0.1, 0.2, 0.3, 4.0, 70.0], 5).unwrap();
    assert!((v - 0.6956940443813076).abs() < 1e-12);
    // ties: ndcg_score([[10, 0, 0, 1, 5]], [[1, 0, 0, 0, 1]], k=1) = 0.75
    let t = ndcg_at_k(&[10.0, 0.0, 0.0, 1.0, 5.0], &[1.0, 0.0, 0.0, 0.0, 1.0], 1).unwrap();
    assert!((t - 0.75).abs() < 1e-12);
    assert_eq!(ndcg_at_k(&[0.0, 0.0], &[1.0, 2.0], 2).unwrap(), 0.0);
    assert!(ndcg_at_k(&[1.0], &[1.0], 0).is_err());
    assert!(ndcg_at_k(&[-1.0], &[1.0], 1).is_err());
    let o = RankingOptions::default();
    assert!(ranking_metrics(&[1, 1], &[0.0, 0.0], &[1.0, 2.0], &o).is_err());
    assert!(ranking_metrics(&[1], &[1.0, 0.0], &[1.0, 2.0], &o).is_err());
}
