//! `classification`, `timeseries` and `survival` against R 4.x reference
//! values (pROC, stats, lmtest, survival). Reference data:
//! `R/generate_curves_ts_surv_refs.R` -> `R/data/curves_ts_surv_reference.csv`.
//! PR and calibration references are sklearn's definitions recomputed in base
//! R (see the generator header). Run with `--nocapture` to print the maximum
//! absolute difference per function.

use anofox_statistics::{
    acf, auc, box_pierce, brier_score, calibration_curve_with_conf_level, durbin_watson,
    kaplan_meier, ljung_box, log_loss, log_rank, pacf, pr_curve, roc_curve, KmConfType, TestResult,
};
use std::collections::HashMap;
use std::path::PathBuf;

type Refs = HashMap<(String, String), Vec<f64>>;

fn refs() -> Refs {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("R/data/curves_ts_surv_reference.csv");
    let mut m = HashMap::new();
    for r in csv::Reader::from_path(p).expect("reference file").records() {
        let r = r.unwrap();
        let v = r[2]
            .split_whitespace()
            .map(|s| match s {
                "NA" | "NaN" => f64::NAN,
                "Inf" => f64::INFINITY,
                s => s.parse().unwrap(),
            })
            .collect();
        m.insert((r[0].to_string(), r[1].to_string()), v);
    }
    m
}

fn get<'a>(m: &'a Refs, case: &str, key: &str) -> &'a [f64] {
    m.get(&(case.to_string(), key.to_string()))
        .unwrap_or_else(|| panic!("missing {} {}", case, key))
}

fn bools(v: &[f64]) -> Vec<bool> {
    v.iter().map(|&x| x == 1.0).collect()
}

/// Max abs difference; NaN and inf must match exactly.
fn diff(got: &[f64], want: &[f64], what: &str) -> f64 {
    assert_eq!(got.len(), want.len(), "{}: length", what);
    let mut d: f64 = 0.0;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        if w.is_nan() || w.is_infinite() {
            assert!(
                (g.is_nan() && w.is_nan()) || g == w,
                "{}[{}]: {} vs {}",
                what,
                i,
                g,
                w
            );
        } else {
            d = d.max((g - w).abs());
        }
    }
    d
}

fn check(name: &str, d: f64, tol: f64) {
    println!("{:<14} max abs diff {:.3e}", name, d);
    assert!(d <= tol, "{}: max abs diff {} > {}", name, d, tol);
}

const CLS: [&str; 4] = ["c1", "c2", "c3", "c4"];

#[test]
fn roc_auc_match_proc() {
    let m = refs();
    let (mut d_roc, mut d_auc) = (0.0_f64, 0.0_f64);
    for c in CLS {
        let s = get(&m, c, "in_scores");
        let y = bools(get(&m, c, "in_labels"));
        let roc = roc_curve(s, &y, false).unwrap();
        d_roc = d_roc.max(diff(&roc.fpr, get(&m, c, "roc_fpr"), "fpr"));
        d_roc = d_roc.max(diff(&roc.tpr, get(&m, c, "roc_tpr"), "tpr"));
        assert!(roc.thresholds[0].is_infinite());
        d_auc = d_auc.max((auc(s, &y).unwrap() - get(&m, c, "auc")[0]).abs());
        // drop_intermediate keeps the curve: same AUC from fewer points.
        let r2 = roc_curve(s, &y, true).unwrap();
        assert!(r2.fpr.len() <= roc.fpr.len());
        let a2: f64 = r2
            .fpr
            .windows(2)
            .zip(r2.tpr.windows(2))
            .map(|(f, t)| (f[1] - f[0]) * (t[1] + t[0]) / 2.0)
            .sum();
        d_auc = d_auc.max((a2 - get(&m, c, "auc")[0]).abs());
    }
    check("roc_curve", d_roc, 1e-14);
    check("auc", d_auc, 1e-14);
}

#[test]
fn pr_brier_logloss_match_r() {
    let m = refs();
    let (mut d_pr, mut d_b, mut d_l) = (0.0_f64, 0.0_f64, 0.0_f64);
    for c in CLS {
        let s = get(&m, c, "in_scores");
        let y = bools(get(&m, c, "in_labels"));
        let pr = pr_curve(s, &y).unwrap();
        d_pr = d_pr.max(diff(&pr.thresholds, get(&m, c, "pr_thresholds"), "thr"));
        d_pr = d_pr.max(diff(&pr.precision, get(&m, c, "pr_precision"), "prec"));
        d_pr = d_pr.max(diff(&pr.recall, get(&m, c, "pr_recall"), "rec"));
        d_b = d_b.max((brier_score(s, &y).unwrap() - get(&m, c, "brier")[0]).abs());
        d_l = d_l.max((log_loss(s, &y).unwrap() - get(&m, c, "log_loss")[0]).abs());
    }
    check("pr_curve", d_pr, 1e-14);
    check("brier_score", d_b, 1e-14);
    check("log_loss", d_l, 1e-13);
}

#[test]
fn calibration_matches_r() {
    let m = refs();
    let (mut d, mut d_ci) = (0.0_f64, 0.0_f64);
    for c in CLS {
        let s = get(&m, c, "in_scores");
        let y = bools(get(&m, c, "in_labels"));
        for nb in [5usize, 10] {
            for cl in [0.95, 0.9] {
                let k = format!("cal_{}_{}_", nb, cl);
                let cc = calibration_curve_with_conf_level(s, &y, nb, cl).unwrap();
                let col = |f: &dyn Fn(&anofox_statistics::CalibrationBin) -> f64| -> Vec<f64> {
                    cc.bins.iter().map(f).collect()
                };
                d = d.max(diff(
                    &col(&|b| b.bin as f64),
                    get(&m, c, &(k.clone() + "bin")),
                    "bin",
                ));
                d = d.max(diff(
                    &col(&|b| b.n as f64),
                    get(&m, c, &(k.clone() + "n")),
                    "n",
                ));
                d = d.max(diff(
                    &col(&|b| b.mean_pred),
                    get(&m, c, &(k.clone() + "mean_pred")),
                    "mean_pred",
                ));
                d = d.max(diff(
                    &col(&|b| b.frac_pos),
                    get(&m, c, &(k.clone() + "frac_pos")),
                    "frac_pos",
                ));
                d_ci = d_ci.max(diff(
                    &col(&|b| b.conf_low),
                    get(&m, c, &(k.clone() + "low")),
                    "low",
                ));
                d_ci = d_ci.max(diff(
                    &col(&|b| b.conf_high),
                    get(&m, c, &(k + "high")),
                    "high",
                ));
            }
        }
    }
    check("calibration", d, 1e-14);
    check("wilson_ci", d_ci, 1e-14);
}

const TS: [&str; 4] = ["t1", "t2", "t3", "t4"];

#[test]
fn acf_pacf_match_r() {
    let m = refs();
    let (mut da, mut dp, mut db) = (0.0_f64, 0.0_f64, 0.0_f64);
    for c in TS {
        let x = get(&m, c, "in_x");
        let ml = get(&m, c, "max_lag")[0] as usize;
        let a = acf(x, ml).unwrap();
        assert_eq!(a.lags, (0..=ml).collect::<Vec<_>>());
        da = da.max(diff(&a.values, get(&m, c, "acf"), "acf"));
        let p = pacf(x, ml).unwrap();
        assert_eq!(p.lags[0], 1);
        dp = dp.max(diff(&p.values, get(&m, c, "pacf"), "pacf"));
        db = db.max((a.conf_bound - get(&m, c, "bound")[0]).abs());
    }
    check("acf", da, 1e-13);
    check("pacf", dp, 1e-12);
    check("acf_bound", db, 1e-15);
}

#[test]
fn box_tests_and_dw_match_r() {
    let m = refs();
    let (mut dl, mut dbp, mut ddw) = (0.0_f64, 0.0_f64, 0.0_f64);
    let mut count = 0;
    for c in TS {
        let x = get(&m, c, "in_x");
        for lag in [1usize, 5, 10] {
            for fitdf in [0usize, 1, 2] {
                let key = format!("lb_{}_{}", lag, fitdf);
                let Some(want) = m.get(&(c.to_string(), key)) else {
                    continue;
                };
                count += 1;
                let r = ljung_box(x, lag, fitdf).unwrap();
                dl = dl.max(diff(&[r.statistic, r.df, r.p_value], want, "lb"));
                let want = get(&m, c, &format!("bp_{}_{}", lag, fitdf));
                let r = box_pierce(x, lag, fitdf).unwrap();
                dbp = dbp.max(diff(&[r.statistic, r.df, r.p_value], want, "bp"));
                let tr = TestResult::from(&r);
                assert_eq!(tr.statistic, Some(r.statistic));
            }
        }
        if let Some(want) = m.get(&(c.to_string(), "dw".to_string())) {
            let e = get(&m, c, "in_dw_resid");
            ddw = ddw.max((durbin_watson(e).unwrap() - want[0]).abs());
        }
    }
    assert!(count >= 20);
    check("ljung_box", dl, 1e-12);
    check("box_pierce", dbp, 1e-12);
    check("durbin_watson", ddw, 1e-13);
}

const SURV: [&str; 4] = ["s1", "s2", "s3", "s4"];

#[test]
fn kaplan_meier_matches_survfit() {
    let m = refs();
    let (mut d, mut dse, mut dci) = (0.0_f64, 0.0_f64, 0.0_f64);
    for c in SURV {
        let t = get(&m, c, "in_time");
        let ev = bools(get(&m, c, "in_event"));
        for (name, ct) in [
            ("log", KmConfType::Log),
            ("log-log", KmConfType::LogLog),
            ("plain", KmConfType::Plain),
        ] {
            for cl in [0.95, 0.9] {
                let km = kaplan_meier(t, &ev, ct, cl).unwrap();
                if name == "log" && cl == 0.95 {
                    let us = |v: &[usize]| v.iter().map(|&x| x as f64).collect::<Vec<_>>();
                    d = d.max(diff(&km.time, get(&m, c, "km_time"), "time"));
                    d = d.max(diff(&us(&km.n_risk), get(&m, c, "km_n_risk"), "n_risk"));
                    d = d.max(diff(&us(&km.n_event), get(&m, c, "km_n_event"), "n_event"));
                    d = d.max(diff(
                        &us(&km.n_censor),
                        get(&m, c, "km_n_censor"),
                        "n_censor",
                    ));
                    d = d.max(diff(&km.survival, get(&m, c, "km_surv"), "surv"));
                    dse = dse.max(diff(
                        &km.std_err_log,
                        get(&m, c, "km_std_err_log"),
                        "se_log",
                    ));
                    dse = dse.max(diff(&km.std_err, get(&m, c, "km_std_err"), "se"));
                }
                let k = format!("km_{}_{}_", name, cl);
                dci = dci.max(diff(&km.conf_low, get(&m, c, &(k.clone() + "low")), "low"));
                dci = dci.max(diff(&km.conf_high, get(&m, c, &(k + "high")), "high"));
            }
        }
    }
    check("km_survival", d, 1e-14);
    check("km_std_err", dse, 1e-13);
    check("km_conf_int", dci, 1e-13);
}

#[test]
fn log_rank_matches_survdiff() {
    let m = refs();
    let (mut d_oe, mut d_stat) = (0.0_f64, 0.0_f64);
    for c in SURV {
        let t = get(&m, c, "in_time");
        let ev = bools(get(&m, c, "in_event"));
        let g: Vec<i64> = get(&m, c, "in_group").iter().map(|&v| v as i64).collect();
        let r = log_rank(t, &ev, &g).unwrap();
        d_oe = d_oe.max(diff(&r.observed, get(&m, c, "lr_obs"), "obs"));
        d_oe = d_oe.max(diff(&r.expected, get(&m, c, "lr_exp"), "exp"));
        let n: Vec<f64> = r.n.iter().map(|&v| v as f64).collect();
        d_oe = d_oe.max(diff(&n, get(&m, c, "lr_n"), "n"));
        let want = get(&m, c, "lr_stat");
        assert_eq!(r.df, want[1]);
        d_stat = d_stat.max((r.chisq - want[0]).abs());
        assert!((r.p_value - want[2]).abs() <= 1e-12 * (1.0 + want[2]));
        let tr: TestResult = r.into();
        assert_eq!(tr.df1, Some(want[1]));
    }
    check("log_rank_O/E", d_oe, 1e-12);
    check("log_rank_chisq", d_stat, 1e-10);
}

/// Examples from the scikit-learn documentation of `roc_curve` and
/// `precision_recall_curve` (sklearn 1.5).
#[test]
fn sklearn_doc_examples() {
    let s = [0.1, 0.4, 0.35, 0.8];
    let y = [false, false, true, true];
    let r = roc_curve(&s, &y, true).unwrap();
    assert_eq!(r.fpr, vec![0.0, 0.0, 0.5, 0.5, 1.0]);
    assert_eq!(r.tpr, vec![0.0, 0.5, 0.5, 1.0, 1.0]);
    assert_eq!(r.thresholds, vec![f64::INFINITY, 0.8, 0.4, 0.35, 0.1]);
    let p = pr_curve(&s, &y).unwrap();
    assert_eq!(p.precision, vec![0.5, 2.0 / 3.0, 0.5, 1.0, 1.0]);
    assert_eq!(p.recall, vec![1.0, 1.0, 0.5, 0.5, 0.0]);
    assert_eq!(p.thresholds, vec![0.1, 0.35, 0.4, 0.8]);
    assert_eq!(auc(&s, &y).unwrap(), 0.75);
}

#[test]
fn edge_cases() {
    // Constant scores: one ROC point (0,0) -> (1,1), AUC 0.5.
    let y = [true, false, true, false];
    assert_eq!(auc(&[0.3; 4], &y).unwrap(), 0.5);
    // One class only: sklearn gives NaN rates and auc raises.
    let r = roc_curve(&[0.1, 0.2], &[true, true], true).unwrap();
    assert!(r.fpr.iter().all(|v| v.is_nan()));
    assert!(auc(&[0.1, 0.2], &[true, true]).is_err());
    assert!(roc_curve(&[f64::NAN, 0.2], &[true, false], true).is_err());
    assert!(roc_curve(&[], &[], true).is_err());
    assert!(calibration_curve_with_conf_level(&[1.2], &[true], 5, 0.95).is_err());
    assert!(calibration_curve_with_conf_level(&[0.2], &[true], 0, 0.95).is_err());
    assert!(log_loss(&[0.0, 1.0], &[false, true]).unwrap() < 1e-12);
    assert!(acf(&[1.0, 2.0, 3.0], 3).is_err());
    assert!(pacf(&[1.0, 2.0, 3.0], 0).is_err());
    assert!(ljung_box(&[1.0, 2.0, 3.0, 1.0], 2, 2).is_err());
    assert!(durbin_watson(&[0.0, 0.0]).is_err());
    assert!(log_rank(&[1.0, 2.0], &[true, true], &[1, 1]).is_err());
    assert!(kaplan_meier(&[1.0], &[true], KmConfType::Log, 1.0).is_err());
    assert_eq!("log-log".parse::<KmConfType>().unwrap(), KmConfType::LogLog);
}
