//! Monitoring module (CBPE, PCA reconstruction-error drift, reference-band
//! thresholds) and the calibrators, against NannyML 0.13.1 / scikit-learn
//! (`R/generate_monitoring_refs.py` -> `R/data/monitoring_nannyml_reference.csv`)
//! and R (`R/generate_monitoring_refs.R` -> `R/data/monitoring_r_reference.csv`).

use anofox_statistics::classification::{IsotonicCalibrator, PlattCalibrator};
use anofox_statistics::monitoring::{
    fit_reconstruction, fit_reference, fit_reference_with_predictions, realized_performance,
    reconstruction_thresholds, threshold_band, CbpeCalibration, CbpeModel, CbpeOptions,
    ComponentSelection, MetricEstimate, ReconstructionModel, ReconstructionOptions,
    ThresholdMethod, ValueLimits,
};
use anofox_statistics::multivariate::{pca_with_options, PcaOptions};
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

fn rows(flat: &[f64], d: usize) -> Vec<Vec<f64>> {
    flat.chunks(d).map(|c| c.to_vec()).collect()
}

fn bools(v: &[f64]) -> Vec<bool> {
    v.iter().map(|&x| x != 0.0).collect()
}

fn close(got: f64, want: f64, tol: f64, what: &str) {
    if want.is_nan() {
        assert!(got.is_nan(), "{}: {} vs NaN", what, got);
        return;
    }
    assert!(
        (got - want).abs() <= tol * want.abs().max(1.0),
        "{}: {} vs {}",
        what,
        got,
        want
    );
}

fn close_all(got: &[f64], want: &[f64], tol: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{} length", what);
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        close(*g, *w, tol, &format!("{}[{}]", what, i));
    }
}

// ------------------------------------------------------------ calibrators

#[test]
fn isotonic_matches_sklearn_with_ties() {
    let m = refs("monitoring_nannyml_reference.csv");
    let c = IsotonicCalibrator::fit(get(&m, "isotonic", "x"), &bools(get(&m, "isotonic", "y")))
        .unwrap();
    close_all(
        &c.calibrate(get(&m, "isotonic", "new")),
        get(&m, "isotonic", "pred"),
        1e-12,
        "isotonic",
    );
    let rebuilt = IsotonicCalibrator::from_parts(c.x.clone(), c.y.clone()).unwrap();
    assert_eq!(rebuilt, c);
}

#[test]
fn isotonic_matches_r_isoreg() {
    let m = refs("monitoring_r_reference.csv");
    let c = IsotonicCalibrator::fit(get(&m, "isotonic", "x"), &bools(get(&m, "isotonic", "y")))
        .unwrap();
    close_all(
        &c.calibrate(get(&m, "isotonic", "new")),
        get(&m, "isotonic", "pred"),
        1e-12,
        "isoreg",
    );
}

#[test]
fn platt_matches_glm_and_sklearn() {
    let r = refs("monitoring_r_reference.csv");
    let c = PlattCalibrator::fit(get(&r, "platt", "x"), &bools(get(&r, "platt", "y"))).unwrap();
    close_all(&[c.a, c.b], get(&r, "platt", "ab"), 1e-9, "platt vs glm");
    // scikit-learn stops L-BFGS-B at gtol = 1e-6.
    let s = refs("monitoring_nannyml_reference.csv");
    let c = PlattCalibrator::fit(get(&s, "platt", "x"), &bools(get(&s, "platt", "y"))).unwrap();
    close_all(
        &[c.a, c.b],
        get(&s, "platt", "ab"),
        1e-5,
        "platt vs sklearn",
    );
}

// ------------------------------------------------------------------ CBPE

fn check_metric(e: &MetricEstimate, m: &Refs, case: &str, name: &str, i: usize, tol_se: f64) {
    close(
        e.value,
        get(m, case, &format!("{}_value", name))[i],
        1e-9,
        &format!("{} {} value [{}]", case, name, i),
    );
    close(
        e.sampling_error,
        get(m, case, &format!("{}_sampling_error", name))[i],
        tol_se,
        &format!("{} {} sampling error [{}]", case, name, i),
    );
    close(
        e.lower_confidence,
        get(m, case, &format!("{}_lower_confidence_boundary", name))[i],
        tol_se,
        &format!("{} {} lower [{}]", case, name, i),
    );
    close(
        e.upper_confidence,
        get(m, case, &format!("{}_upper_confidence_boundary", name))[i],
        tol_se,
        &format!("{} {} upper [{}]", case, name, i),
    );
}

const METRICS: [&str; 10] = [
    "roc_auc",
    "f1",
    "precision",
    "recall",
    "specificity",
    "accuracy",
    "true_positive",
    "false_positive",
    "false_negative",
    "true_negative",
];

fn cbpe_case(case: &str, auc_se_tol: f64) {
    let m = refs("monitoring_nannyml_reference.csv");
    assert_eq!(get(&m, case, "needs_calibration")[0], 1.0);
    let ref_score = get(&m, case, "ref_score");
    let ref_label = bools(get(&m, case, "ref_label"));
    let ana_score = get(&m, case, "ana_score");
    let ana_pred = bools(get(&m, case, "ana_pred"));
    let ana_label = bools(get(&m, case, "ana_label"));
    let chunk = get(&m, case, "chunk_size")[0] as usize;
    // NannyML's y_pred column; in the synthetic dataset it comes from the
    // unrounded probabilities, so it is not always score >= 0.5.
    let model = fit_reference_with_predictions(
        ref_score,
        &bools(get(&m, case, "ref_pred")),
        &ref_label,
        CbpeCalibration::Isotonic,
    )
    .unwrap();
    let thresholded = ref_score
        .iter()
        .zip(get(&m, case, "ref_pred"))
        .all(|(&s, &p)| (s >= 0.5) == (p != 0.0));
    if thresholded {
        let by_threshold = fit_reference(ref_score, &ref_label, &CbpeOptions::default()).unwrap();
        assert_eq!(model, by_threshold);
    }
    close_all(
        &model.calibrator.calibrate(ana_score),
        get(&m, case, "ana_calibrated"),
        1e-12,
        "calibrated",
    );
    // Reference thresholds from the realized metrics on the reference chunks.
    let ref_pred = bools(get(&m, case, "ref_pred"));
    let realized_ref: Vec<_> = ref_score
        .chunks(chunk)
        .zip(ref_label.chunks(chunk))
        .zip(ref_pred.chunks(chunk))
        .map(|((s, l), p)| realized_performance(s, p, l).unwrap())
        .collect();
    let n_chunks = ana_score.len() / chunk;
    for i in 0..n_chunks {
        let r = i * chunk..(i + 1) * chunk;
        let est = model
            .estimate_with_predictions(&ana_score[r.clone()], &ana_pred[r.clone()])
            .unwrap();
        assert_eq!(est.n, chunk);
        if thresholded {
            assert_eq!(model.estimate(&ana_score[r.clone()]).unwrap(), est);
        }
        let ests = [
            est.roc_auc,
            est.f1,
            est.precision,
            est.recall,
            est.specificity,
            est.accuracy,
            est.true_positive,
            est.false_positive,
            est.false_negative,
            est.true_negative,
        ];
        let real = realized_performance(
            &ana_score[r.clone()],
            &ana_pred[r.clone()],
            &ana_label[r.clone()],
        )
        .unwrap();
        if thresholded {
            let by_model = model
                .realized(&ana_score[r.clone()], &ana_label[r.clone()])
                .unwrap();
            assert_eq!(by_model, real);
        }
        let reals = [
            real.roc_auc,
            real.f1,
            real.precision,
            real.recall,
            real.specificity,
            real.accuracy,
            real.true_positive,
            real.false_positive,
            real.false_negative,
            real.true_negative,
        ];
        for (k, name) in METRICS.iter().enumerate() {
            let tol = if *name == "roc_auc" {
                auc_se_tol
            } else {
                1e-10
            };
            check_metric(&ests[k], &m, case, name, i, tol);
            close(
                reals[k],
                get(&m, case, &format!("{}_realized", name))[i],
                1e-12,
                &format!("{} {} realized [{}]", case, name, i),
            );
        }
        // Alert thresholds (rate metrics: limits [0, 1]).
        for (k, name) in METRICS.iter().take(6).enumerate() {
            let vals: Vec<f64> = realized_ref
                .iter()
                .map(|p| {
                    [
                        p.roc_auc,
                        p.f1,
                        p.precision,
                        p.recall,
                        p.specificity,
                        p.accuracy,
                    ][k]
                })
                .collect();
            let band = threshold_band(
                &vals,
                &ThresholdMethod::NANNYML_DEFAULT,
                &ValueLimits::UNIT_INTERVAL,
            )
            .unwrap();
            close(
                band.lower.unwrap(),
                get(&m, case, &format!("{}_lower_threshold", name))[i],
                1e-12,
                &format!("{} lower threshold", name),
            );
            close(
                band.upper.unwrap(),
                get(&m, case, &format!("{}_upper_threshold", name))[i],
                1e-12,
                &format!("{} upper threshold", name),
            );
            assert_eq!(
                band.alert(ests[k].value),
                get(&m, case, &format!("{}_alert", name))[i] != 0.0,
                "{} alert [{}]",
                name,
                i
            );
        }
    }
}

#[test]
fn cbpe_matches_nannyml_synthetic_dataset() {
    // Scores with two decimals: NannyML orders tied scores of the ROC AUC
    // sampling error by an unstable sort, this counts them half; the sampling
    // error differs by 5e-5 (0.003556 vs 0.003606), the bounds by 3 times that.
    cbpe_case("cbpe_synthetic", 2e-4);
}

#[test]
fn cbpe_matches_nannyml_continuous_scores() {
    cbpe_case("cbpe_continuous", 1e-10);
}

#[test]
fn cbpe_model_rebuilds_from_parts_and_validates() {
    let m = refs("monitoring_nannyml_reference.csv");
    let s = get(&m, "cbpe_continuous", "ref_score");
    let l = bools(get(&m, "cbpe_continuous", "ref_label"));
    for cal in [
        CbpeCalibration::Isotonic,
        CbpeCalibration::Platt,
        CbpeCalibration::None,
    ] {
        let model = fit_reference(
            s,
            &l,
            &CbpeOptions {
                calibration: cal,
                threshold: 0.5,
            },
        )
        .unwrap();
        let rebuilt =
            CbpeModel::from_parts(model.calibrator.clone(), model.threshold, model.sampling);
        assert_eq!(
            rebuilt.estimate(&s[..100]).unwrap(),
            model.estimate(&s[..100]).unwrap()
        );
    }
    assert!(fit_reference(&[0.1, 0.9], &[true, true], &CbpeOptions::default()).is_err());
    assert!(fit_reference(&[0.1, f64::NAN], &[true, false], &CbpeOptions::default()).is_err());
    let model = fit_reference(s, &l, &CbpeOptions::default()).unwrap();
    assert!(model.estimate(&[]).is_err());
    assert!(model.estimate_with_predictions(&[0.5], &[]).is_err());
}

// ------------------------------------------- reconstruction-error drift

fn dre_case(case: &str, inputs: &str) {
    let m = refs("monitoring_nannyml_reference.csv");
    let d = get(&m, inputs, "dims")[0] as usize;
    let reference = rows(get(&m, inputs, "ref_x"), d);
    let analysis = rows(get(&m, inputs, "ana_x"), d);
    let chunk = get(&m, case, "chunk_size")[0] as usize;
    let opt = get(&m, case, "n_components_option")[0];
    let selection = if opt < 1.0 {
        ComponentSelection::VarianceShare(opt)
    } else {
        ComponentSelection::Fixed(opt as usize)
    };
    let model = fit_reconstruction(
        &reference,
        &ReconstructionOptions {
            n_components: selection,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        model.n_components,
        get(&m, case, "n_components")[0] as usize
    );
    close_all(
        &model.row_errors(&analysis[..20]).unwrap(),
        get(&m, case, "row_errors"),
        1e-9,
        &format!("{} row errors", case),
    );
    let ref_chunks: Vec<_> = reference
        .chunks(chunk)
        .map(|c| model.chunk_error(c).unwrap())
        .collect();
    let band = reconstruction_thresholds(
        &ref_chunks.iter().map(|c| c.value).collect::<Vec<_>>(),
        &ThresholdMethod::NANNYML_DEFAULT,
    )
    .unwrap();
    for (period, chunks) in [
        ("reference", ref_chunks.clone()),
        (
            "analysis",
            analysis
                .chunks(chunk)
                .map(|c| model.chunk_error(c).unwrap())
                .collect(),
        ),
    ] {
        for (i, c) in chunks.iter().enumerate() {
            let w = |k: &str| get(&m, case, &format!("{}_{}", period, k))[i];
            let what = format!("{} {} [{}]", case, period, i);
            close(c.value, w("value"), 1e-9, &what);
            close(c.sampling_error, w("sampling_error"), 1e-9, &what);
            close(
                c.lower_confidence,
                w("lower_confidence_boundary"),
                1e-9,
                &what,
            );
            close(
                c.upper_confidence,
                w("upper_confidence_boundary"),
                1e-9,
                &what,
            );
            close(
                band.lower.unwrap_or(f64::NAN),
                w("lower_threshold"),
                1e-9,
                &what,
            );
            close(
                band.upper.unwrap_or(f64::NAN),
                w("upper_threshold"),
                1e-9,
                &what,
            );
            assert_eq!(band.alert(c.value), w("alert") != 0.0, "{} alert", what);
        }
    }
    let rebuilt =
        ReconstructionModel::from_parts(model.pca.clone(), model.n_components, true).unwrap();
    assert_eq!(rebuilt, model);
}

#[test]
fn reconstruction_drift_matches_nannyml_car_loan() {
    dre_case("dre_car_loan", "dre_car_loan");
}

#[test]
fn reconstruction_drift_matches_nannyml_fixed_components() {
    dre_case("dre_car_loan_k2", "dre_car_loan");
}

#[test]
fn reconstruction_drift_matches_nannyml_with_missing_values() {
    dre_case("dre_car_loan_nan", "dre_car_loan_nan");
}

#[test]
fn reconstruction_rejects_bad_input() {
    let x = vec![vec![1.0, 2.0], vec![2.0, 1.0], vec![3.0, 5.0]];
    let no_impute = ReconstructionOptions {
        impute_nan: false,
        ..Default::default()
    };
    let model = fit_reconstruction(&x, &no_impute).unwrap();
    assert!(model.row_errors(&[vec![f64::NAN, 1.0]]).is_err());
    assert!(model.row_errors(&[vec![1.0]]).is_err());
    assert!(fit_reconstruction(&x[..1], &Default::default()).is_err());
    let fixed = |k| ReconstructionOptions {
        n_components: ComponentSelection::Fixed(k),
        ..Default::default()
    };
    assert!(fit_reconstruction(&x, &fixed(0)).is_err());
    assert!(fit_reconstruction(&x, &fixed(3)).is_err());
    let share = ReconstructionOptions {
        n_components: ComponentSelection::VarianceShare(1.0),
        ..Default::default()
    };
    assert!(fit_reconstruction(&x, &share).is_err());
    let all_nan = vec![vec![f64::NAN, 1.0], vec![f64::NAN, 2.0]];
    assert!(fit_reconstruction(&all_nan, &Default::default()).is_err());
}

#[test]
fn pca_reconstruction_matches_prcomp() {
    let m = refs("monitoring_r_reference.csv");
    let d = get(&m, "reconstruction", "dims")[1] as usize;
    let x = rows(get(&m, "reconstruction", "in_x"), d);
    let new = rows(get(&m, "reconstruction", "in_new"), d);
    let p = pca_with_options(&x, &PcaOptions { scale: true }).unwrap();
    for k in 1..=3 {
        close_all(
            &p.reconstruction_error(&new, k).unwrap(),
            get(&m, "reconstruction", &format!("error_k{}", k)),
            1e-10,
            &format!("error k{}", k),
        );
        let rec: Vec<f64> = p
            .reconstruct(&new, k)
            .unwrap()
            .into_iter()
            .flatten()
            .collect();
        close_all(
            &rec,
            get(&m, "reconstruction", &format!("reconstruct_k{}", k)),
            1e-10,
            &format!("reconstruct k{}", k),
        );
    }
    // Reconstructing with every component returns the data.
    let full: Vec<f64> = p
        .reconstruct(&new, 4)
        .unwrap()
        .into_iter()
        .flatten()
        .collect();
    close_all(&full, get(&m, "reconstruction", "in_new"), 1e-10, "full");
    let pu = pca_with_options(&x, &PcaOptions { scale: false }).unwrap();
    close_all(
        &pu.reconstruction_error(&new, 2).unwrap(),
        get(&m, "reconstruction", "error_unscaled_k2"),
        1e-10,
        "unscaled",
    );
    // sklearn's rule: the smallest k whose cumulative share exceeds the value.
    let cum: Vec<f64> = p
        .explained_variance_ratio
        .iter()
        .scan(0.0, |s, r| {
            *s += r;
            Some(*s)
        })
        .collect();
    assert_eq!(p.n_components_for_variance(cum[0] - 1e-9).unwrap(), 1);
    assert_eq!(p.n_components_for_variance(cum[0] + 1e-9).unwrap(), 2);
    assert!(p.n_components_for_variance(0.0).is_err());
}

// ------------------------------------------------------------ thresholds

#[test]
fn thresholds_match_nannyml() {
    let m = refs("monitoring_nannyml_reference.csv");
    let vals = get(&m, "threshold_std", "values");
    let opt = |v: f64| if v.is_nan() { None } else { Some(v) };
    for (name, lo, up) in [
        ("k3", Some(3.0), Some(3.0)),
        ("k2_lower_only", Some(2.0), None),
        ("k15_05", Some(1.5), Some(0.5)),
    ] {
        let method = ThresholdMethod::StandardDeviation {
            lower: lo,
            upper: up,
        };
        for (suffix, limits) in [
            ("", ValueLimits::NONE),
            (
                "_limited",
                ValueLimits {
                    lower: Some(0.9),
                    upper: Some(0.96),
                    override_with_none: false,
                },
            ),
            (
                "_limited_none",
                ValueLimits {
                    lower: Some(0.9),
                    upper: Some(0.96),
                    override_with_none: true,
                },
            ),
        ] {
            let band = threshold_band(vals, &method, &limits).unwrap();
            let want = get(&m, "threshold_std", &format!("{}{}", name, suffix));
            for (g, w) in [(band.lower, want[0]), (band.upper, want[1])] {
                match (g, opt(w)) {
                    (Some(g), Some(w)) => close(g, w, 1e-12, name),
                    (None, None) => {}
                    other => panic!("{}{}: {:?}", name, suffix, other),
                }
            }
        }
    }
}

#[test]
fn constant_and_relative_thresholds() {
    let c = threshold_band(
        &[],
        &ThresholdMethod::Constant {
            lower: Some(0.8),
            upper: None,
        },
        &ValueLimits::UNIT_INTERVAL,
    )
    .unwrap();
    assert_eq!((c.lower, c.upper), (Some(0.8), None));
    assert!(c.alert(0.79) && !c.alert(0.8) && !c.alert(f64::NAN));
    assert!(threshold_band(
        &[],
        &ThresholdMethod::Constant {
            lower: Some(1.0),
            upper: Some(1.0)
        },
        &ValueLimits::NONE
    )
    .is_err());
    let r = threshold_band(
        &[-2.0, -4.0, f64::NAN],
        &ThresholdMethod::Relative {
            lower: Some(0.1),
            upper: Some(0.5),
        },
        &ValueLimits::NONE,
    )
    .unwrap();
    close(r.lower.unwrap(), -3.3, 1e-12, "relative lower");
    close(r.upper.unwrap(), -1.5, 1e-12, "relative upper");
    assert!(threshold_band(
        &[1.0],
        &ThresholdMethod::StandardDeviation {
            lower: Some(-1.0),
            upper: None
        },
        &ValueLimits::NONE
    )
    .is_err());
    assert!(threshold_band(&[f64::NAN], &ThresholdMethod::default(), &ValueLimits::NONE).is_err());
}
