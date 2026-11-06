//! Edge cases and input validation of the monitoring, calibration and metric
//! functions (the reference values are in `monitoring_reference.rs` and
//! `metrics_reference.rs`).

use anofox_statistics::classification::{IsotonicCalibrator, PlattCalibrator};
use anofox_statistics::metrics::{
    classification_metrics, ndcg_at_k, ranking_metrics, regression_metrics, Average,
    ClassificationOptions, RankingOptions,
};
use anofox_statistics::monitoring::{
    fit_reconstruction, fit_reference, fit_reference_with_predictions, realized_performance,
    threshold_band, Calibrator, CbpeCalibration, CbpeOptions, ComponentSelection,
    ReconstructionModel, ReconstructionOptions, ThresholdMethod, ValueLimits,
};

#[test]
fn calibrator_validation() {
    assert!(IsotonicCalibrator::fit(&[0.1], &[true, false]).is_err());
    assert!(IsotonicCalibrator::fit(&[], &[]).is_err());
    assert!(PlattCalibrator::fit(&[f64::INFINITY], &[true]).is_err());
    assert!(IsotonicCalibrator::from_parts(vec![], vec![]).is_err());
    assert!(IsotonicCalibrator::from_parts(vec![0.0, 0.0], vec![0.1, 0.2]).is_err());
    assert!(IsotonicCalibrator::from_parts(vec![0.0, 1.0], vec![0.3, 0.2]).is_err());
    let c = IsotonicCalibrator::from_parts(vec![0.0, 1.0], vec![0.2, 0.6]).unwrap();
    assert!(c.calibrate_one(f64::NAN).is_nan());
    assert_eq!(c.calibrate(&[-1.0, 0.5, 2.0]), vec![0.2, 0.4, 0.6]);
    let p = PlattCalibrator { a: -2.0, b: 0.0 };
    assert_eq!(p.calibrate(&[0.0]), vec![0.5]);
    assert!(p.calibrate_one(f64::NAN).is_nan());
    // Perfectly separated scores still converge (smoothed targets).
    let p = PlattCalibrator::fit(&[0.0, 0.1, 0.9, 1.0], &[false, false, true, true]).unwrap();
    assert!(p.a < 0.0 && p.calibrate_one(1.0) > 0.5);
}

#[test]
fn cbpe_majority_positive_reference_is_symmetric() {
    // The ROC AUC sampling error relabels the minority class as positive.
    let s: Vec<f64> = (0..40).map(|i| ((i * 37) % 40) as f64 / 40.0).collect();
    let l: Vec<bool> = s
        .iter()
        .enumerate()
        .map(|(i, &v)| v > 0.25 || i % 9 == 0)
        .collect();
    let flipped_s: Vec<f64> = s.iter().map(|v| 1.0 - v).collect();
    let flipped_l: Vec<bool> = l.iter().map(|v| !v).collect();
    let none = |t| CbpeOptions {
        calibration: CbpeCalibration::None,
        threshold: t,
    };
    let a = fit_reference(&s, &l, &none(0.5)).unwrap();
    let b = fit_reference(&flipped_s, &flipped_l, &none(0.5)).unwrap();
    assert!((a.sampling.roc_auc_std - b.sampling.roc_auc_std).abs() < 1e-12);
    assert!((a.sampling.roc_auc_fraction - b.sampling.roc_auc_fraction).abs() < 1e-12);
    assert_eq!(a.calibrator, Calibrator::Identity);
    assert!(fit_reference(&s, &l, &none(f64::NAN)).is_err());
}

#[test]
fn cbpe_degenerate_chunks() {
    let s = [0.1, 0.2, 0.8, 0.9];
    let l = [false, true, false, true];
    // Everything predicted negative on the reference: F1 observations are the
    // two false negatives (value 0).
    let m = fit_reference_with_predictions(&s, &[false; 4], &l, CbpeCalibration::None).unwrap();
    assert_eq!((m.sampling.f1_std, m.sampling.f1_fraction), (0.0, 0.5));
    // A chunk whose calibrated probabilities are all 0: ROC AUC undefined,
    // precision / recall / F1 0 (NannyML).
    let e = m.estimate(&[0.0, 0.0]).unwrap();
    assert!(e.roc_auc.value.is_nan());
    assert_eq!(
        (e.precision.value, e.recall.value, e.f1.value),
        (0.0, 0.0, 0.0)
    );
    assert_eq!(e.specificity.value, 1.0);
    // Realized metrics with a single class / single prediction.
    let r = realized_performance(&[0.1, 0.9], &[false, true], &[true, true]).unwrap();
    assert!(r.roc_auc.is_nan() && r.f1.is_nan() && r.specificity.is_nan());
    assert_eq!(r.accuracy, 0.5);
    let r = realized_performance(&[0.1, 0.9], &[true, true], &[false, true]).unwrap();
    assert!(r.precision.is_nan() && !r.roc_auc.is_nan());
    assert!(realized_performance(&[0.1], &[true], &[]).is_err());
}

#[test]
fn reconstruction_options_and_validation() {
    let x: Vec<Vec<f64>> = (0..30)
        .map(|i| {
            let t = i as f64;
            vec![t, 2.0 * t + (t * 0.7).sin(), 5.0, (t * 1.3).cos()]
        })
        .collect();
    // A constant column keeps scale 1 (scikit-learn's StandardScaler).
    let m = fit_reconstruction(&x, &ReconstructionOptions::default()).unwrap();
    assert_eq!(m.pca.scale.as_ref().unwrap()[2], 1.0);
    let unscaled = fit_reconstruction(
        &x,
        &ReconstructionOptions {
            scale: false,
            n_components: ComponentSelection::Fixed(2),
            impute_nan: true,
        },
    )
    .unwrap();
    assert!(unscaled.pca.scale.is_none());
    assert!(unscaled
        .row_errors(&x)
        .unwrap()
        .iter()
        .all(|e| e.is_finite()));
    let one = m.chunk_error(&x[..1]).unwrap();
    assert!(one.std.is_nan() && one.sampling_error.is_nan());
    assert!(m.row_errors(&[]).is_err());
    assert!(m.row_errors(&[vec![f64::INFINITY, 0.0, 0.0, 0.0]]).is_err());
    assert!(fit_reconstruction(&[vec![], vec![]], &Default::default()).is_err());
    assert!(ReconstructionModel::from_parts(m.pca.clone(), 0, true).is_err());
    assert!(ReconstructionModel::from_parts(m.pca.clone(), 99, true).is_err());
}

#[test]
fn threshold_validation() {
    assert!(threshold_band(
        &[1.0, f64::INFINITY],
        &ThresholdMethod::default(),
        &ValueLimits::NONE
    )
    .is_err());
    assert!(threshold_band(
        &[],
        &ThresholdMethod::Constant {
            lower: Some(f64::NAN),
            upper: None
        },
        &ValueLimits::NONE
    )
    .is_err());
    let b = threshold_band(
        &[],
        &ThresholdMethod::Constant {
            lower: None,
            upper: Some(2.0),
        },
        &ValueLimits::UNIT_INTERVAL,
    )
    .unwrap();
    assert_eq!(b.upper, Some(1.0));
}

#[test]
fn metric_validation() {
    // Two labels without the positive class.
    let o = ClassificationOptions {
        average: Average::Binary(9),
        zero_division: 0.0,
    };
    assert!(classification_metrics(&[0, 1], &[1, 0], &o).is_err());
    assert!(regression_metrics(&[], &[]).is_err());
    let even = regression_metrics(&[1.0, 2.0, 3.0, 4.0], &[1.0, 2.5, 2.0, 6.0]).unwrap();
    assert!((even.median_ae - 0.75).abs() < 1e-12);
    assert!(ndcg_at_k(&[1.0], &[1.0, 2.0], 1).is_err());
    assert!(ndcg_at_k(&[], &[], 1).is_err());
    let rel_opt = |m| RankingOptions {
        k: 2,
        min_relevance: Some(m),
        include_no_feedback: false,
    };
    assert!(ranking_metrics(&[1], &[1.0], &[1.0], &rel_opt(f64::NAN)).is_err());
    assert!(ranking_metrics(&[1], &[-1.0], &[1.0], &RankingOptions::default()).is_err());
    // A negative relevance is fine with a binary threshold.
    let r = ranking_metrics(&[1, 1], &[-1.0, 3.0], &[0.2, 0.1], &rel_opt(2.0)).unwrap();
    assert!((r.mrr - 0.5).abs() < 1e-12);
    // k beyond the list: precision over the list length.
    let r = ranking_metrics(
        &[1, 1],
        &[1.0, 0.0],
        &[0.9, 0.1],
        &RankingOptions {
            k: 10,
            ..Default::default()
        },
    )
    .unwrap();
    assert!((r.precision - 0.5).abs() < 1e-12 && (r.map - 1.0).abs() < 1e-12);
}
