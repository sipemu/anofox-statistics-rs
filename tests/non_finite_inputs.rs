//! Property-style regression test for GitHub issue #17: no public function may
//! panic on non-finite input (NaN, +Inf, -Inf) in data or scalar parameters.
//! Each call must return `Ok` (with a NaN or in-range result) or `Err`.

use anofox_statistics::resampling::bootstrap::{CircularBlockBootstrap, StationaryBootstrap};
use anofox_statistics::utils::math;
use anofox_statistics::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::panic::{catch_unwind, AssertUnwindSafe};

const NAN: f64 = f64::NAN;
const INF: f64 = f64::INFINITY;
const NINF: f64 = f64::NEG_INFINITY;

const ALTS: [Alternative; 3] = [
    Alternative::TwoSided,
    Alternative::Less,
    Alternative::Greater,
];

/// Scalar parameter values tried for every float parameter.
const SCALARS: [f64; 6] = [NAN, INF, NINF, 0.0, -1.0, 0.5];

fn base(n: usize, offset: f64) -> Vec<f64> {
    (0..n)
        .map(|i| offset + (i as f64 * 1.7).sin() * 3.0 + i as f64 * 0.3)
        .collect()
}

/// Hand-picked (x, y) pairs covering the interesting non-finite patterns.
fn fixed_cases() -> Vec<(Vec<f64>, Vec<f64>)> {
    let mut cases = Vec::new();
    for &n in &[3usize, 5, 10, 25] {
        let x = base(n, 0.0);
        let y = base(n, 1.0);
        for &s in &[NAN, INF, NINF] {
            // single special in x
            let mut xs = x.clone();
            xs[0] = s;
            cases.push((xs.clone(), y.clone()));
            // special in both at the same index (paired Inf - Inf = NaN)
            let mut ys = y.clone();
            ys[0] = s;
            cases.push((xs.clone(), ys.clone()));
            // all special
            cases.push((vec![s; n], vec![s; n]));
            cases.push((vec![s; n], y.clone()));
        }
        // +Inf and -Inf mixed
        let mut xm = x.clone();
        xm[0] = INF;
        xm[n - 1] = NINF;
        let mut ym = y.clone();
        ym[0] = NINF;
        cases.push((xm.clone(), ym.clone()));
        cases.push((xm, y.clone()));
        let mut ym2 = y.clone();
        ym2[n / 2] = NAN;
        cases.push((x.clone(), ym2));
        // degenerate: constant data (zero variance)
        cases.push((vec![5.0; n], vec![5.0; n]));
        cases.push((vec![5.0; n], vec![7.0; n]));
        cases.push((x.clone(), x.clone()));
    }
    cases
}

/// Random (x, y) pairs with ~25% special values.
fn random_cases(count: usize) -> Vec<(Vec<f64>, Vec<f64>)> {
    let mut rng = ChaCha8Rng::seed_from_u64(17);
    let specials = [NAN, INF, NINF, 0.0, 1e308, -1e308, 5e-324];
    let mut gen = |rng: &mut ChaCha8Rng, n: usize| -> Vec<f64> {
        (0..n)
            .map(|_| {
                if rng.gen::<f64>() < 0.25 {
                    specials[rng.gen_range(0..specials.len())]
                } else {
                    rng.gen_range(-10.0..10.0)
                }
            })
            .collect()
    };
    (0..count)
        .map(|_| {
            let n = rng.gen_range(0..16);
            let paired = rng.gen::<bool>();
            let m = if paired { n } else { rng.gen_range(0..16) };
            let x = gen(&mut rng, n);
            let y = gen(&mut rng, m);
            (x, y)
        })
        .collect()
}

fn check_p(name: &str, p: f64) {
    assert!(
        p.is_nan() || (0.0..=1.0).contains(&p),
        "{name}: p-value out of range: {p}"
    );
}

/// Runs every public data-taking function on (x, y). Returns names that panicked.
fn run_all(x: &[f64], y: &[f64]) -> Vec<String> {
    let mut failures = Vec::new();
    let mut run = |name: &str, f: &mut dyn FnMut()| {
        if catch_unwind(AssertUnwindSafe(f)).is_err() {
            failures.push(format!("{name} x={x:?} y={y:?}"));
        }
    };
    let z = base(x.len(), 2.0);
    let zs: [&[f64]; 1] = [&z];
    let groups: [&[f64]; 3] = [x, y, &z];
    let bounds = EquivalenceBounds::symmetric(0.5).unwrap();

    for alt in ALTS {
        for exact in [false, true] {
            for cc in [false, true] {
                run("mann_whitney_u", &mut || {
                    if let Ok(r) = mann_whitney_u(x, y, alt, cc, exact, Some(0.95), None) {
                        check_p("mann_whitney_u", r.p_value);
                    }
                    let _ = mann_whitney_u(x, y, alt, cc, exact, None, Some(0.5));
                });
                run("wilcoxon_signed_rank", &mut || {
                    if let Ok(r) = wilcoxon_signed_rank(x, y, alt, cc, exact, Some(0.95), None) {
                        check_p("wilcoxon_signed_rank", r.p_value);
                    }
                    let _ = wilcoxon_signed_rank(x, &[], alt, cc, exact, Some(0.95), None);
                    let _ = wilcoxon_signed_rank(x, y, alt, cc, exact, None, Some(0.5));
                });
            }
        }
        run("brunner_munzel", &mut || {
            if let Ok(r) = brunner_munzel(x, y, alt, Some(0.05)) {
                check_p("brunner_munzel", r.p_value);
            }
        });
        for kind in [TTestKind::Welch, TTestKind::Student, TTestKind::Paired] {
            run("t_test", &mut || {
                if let Ok(r) = t_test(x, y, kind, alt, 0.0, Some(0.95)) {
                    check_p("t_test", r.p_value);
                }
                let _ = t_test(x, &[], kind, alt, 0.0, Some(0.95));
            });
            run("cohens_d", &mut || {
                let _ = cohens_d(x, y, kind, 0.0);
            });
        }
        run("yuen_test", &mut || {
            if let Ok(r) = yuen_test(x, y, 0.2, alt, Some(0.95)) {
                check_p("yuen_test", r.p_value);
            }
        });
        run("permutation_t_test", &mut || {
            let _ = permutation_t_test(x, y, alt, 50, Some(1));
        });
        for loss in [LossFunction::SquaredError, LossFunction::AbsoluteError] {
            for ve in [VarEstimator::Acf, VarEstimator::Bartlett] {
                run("diebold_mariano", &mut || {
                    let _ = diebold_mariano(x, y, loss, 1, alt, ve);
                    let _ = diebold_mariano(x, y, loss, 2, alt, ve);
                });
            }
        }
    }

    run("rank", &mut || {
        let _ = rank(x);
    });
    run("rank_biserial", &mut || {
        let _ = rank_biserial(x, y, None);
        let _ = rank_biserial(x, y, Some(0.5));
        let _ = matched_pairs_rank_biserial(x, y, None);
        let _ = matched_pairs_rank_biserial(x, y, Some(0.5));
        let _ = rank_biserial_from_u(x.first().copied().unwrap_or(NAN), x.len(), y.len());
    });
    run("kruskal_wallis", &mut || {
        if let Ok(r) = kruskal_wallis(&groups) {
            check_p("kruskal_wallis", r.p_value);
        }
    });
    for kind in [AnovaKind::Fisher, AnovaKind::Welch] {
        run("one_way_anova", &mut || {
            let _ = one_way_anova(&groups, kind);
        });
    }
    run("two_way_anova", &mut || {
        let fa: Vec<usize> = (0..x.len()).map(|i| i % 2).collect();
        let fb: Vec<usize> = (0..x.len()).map(|i| (i / 2) % 2).collect();
        let _ = two_way_anova(x, &fa, &fb);
    });
    run("repeated_measures_anova", &mut || {
        let _ = repeated_measures_anova(&groups, true);
        let _ = repeated_measures_anova(&[x, y], false);
    });
    run("brown_forsythe", &mut || {
        let _ = brown_forsythe(&groups);
    });
    run("cohens_d_one_sample", &mut || {
        let _ = cohens_d_one_sample(x, 0.0);
    });
    run("shapiro_wilk", &mut || {
        if let Ok(r) = shapiro_wilk(x) {
            check_p("shapiro_wilk", r.p_value);
        }
    });
    run("jarque_bera", &mut || {
        let _ = jarque_bera(x);
    });
    run("dagostino_k_squared", &mut || {
        let _ = dagostino_k_squared(x);
    });
    run("pearson", &mut || {
        let _ = pearson(x, y, Some(0.95));
    });
    run("spearman", &mut || {
        let _ = spearman(x, y, Some(0.95));
    });
    for v in [
        KendallVariant::TauA,
        KendallVariant::TauB,
        KendallVariant::TauC,
    ] {
        run("kendall", &mut || {
            let _ = kendall(x, y, v);
        });
    }
    run("partial_cor", &mut || {
        let _ = partial_cor(x, y, &zs);
        let _ = semi_partial_cor(x, y, &zs);
    });
    run("distance_cor", &mut || {
        let _ = distance_cor(x, y);
        let _ = distance_cor_test(x, y, 30, Some(1));
    });
    for t in [
        ICCType::ICC1,
        ICCType::ICC2,
        ICCType::ICC3,
        ICCType::ICC1k,
        ICCType::ICC2k,
    ] {
        run("icc", &mut || {
            let rows: Vec<Vec<f64>> = x.iter().zip(y.iter()).map(|(&a, &b)| vec![a, b]).collect();
            let _ = icc(&rows, t);
        });
    }
    run("energy_distance_test", &mut || {
        let _ = energy_distance_test_1d(x, y, 30, Some(1));
        let xv: Vec<Vec<f64>> = x.iter().map(|&v| vec![v, 1.0]).collect();
        let yv: Vec<Vec<f64>> = y.iter().map(|&v| vec![v, 1.0]).collect();
        let _ = energy_distance_test(&xv, &yv, 30, Some(1));
    });
    run("mmd_test", &mut || {
        let _ = mmd_test_1d(x, y, 30, Some(1));
        let xv: Vec<Vec<f64>> = x.iter().map(|&v| vec![v]).collect();
        let yv: Vec<Vec<f64>> = y.iter().map(|&v| vec![v]).collect();
        for k in [
            Kernel::Gaussian { bandwidth: 1.0 },
            Kernel::Linear,
            Kernel::Polynomial {
                degree: 2,
                scale: 1.0,
                offset: 1.0,
            },
            Kernel::Laplacian { bandwidth: 1.0 },
        ] {
            let _ = mmd_test(&xv, &yv, k, 30, Some(1));
        }
    });
    run("clark_west", &mut || {
        let _ = clark_west(x, y, 1);
    });
    run("spa_test", &mut || {
        let _ = spa_test(x, &[y.to_vec(), z.clone()], 30, 2.0, Some(1));
        let _ = mspe_adjusted_spa(x, &[y.to_vec(), z.clone()], 30, 2.0, Some(1));
    });
    for stat in [MCSStatistic::Range, MCSStatistic::Max] {
        run("model_confidence_set", &mut || {
            let _ = model_confidence_set(
                &[x.to_vec(), y.to_vec(), z.clone()],
                0.1,
                stat,
                30,
                2.0,
                Some(1),
            );
        });
    }
    run("bootstrap_ci", &mut || {
        let _ = bootstrap_ci(x, |d| d.iter().sum::<f64>(), 50, 0.95, Some(1));
        let _ = bootstrap_mean_ci(x, 50, 0.95, 0, Some(1));
        let _ = bootstrap_mean_ci(x, 50, 0.95, 2, Some(1));
    });
    run("tost_t_test", &mut || {
        let _ = tost_t_test_one_sample(x, 0.0, &bounds, 0.05);
        let _ = tost_t_test_two_sample(x, y, &bounds, 0.05, true);
        let _ = tost_t_test_two_sample(x, y, &bounds, 0.05, false);
        let _ = tost_t_test_paired(x, y, &bounds, 0.05);
    });
    run("tost_wilcoxon", &mut || {
        let _ = tost_wilcoxon_paired(x, y, &bounds, 0.05);
        let _ = tost_wilcoxon_two_sample(x, y, &bounds, 0.05);
    });
    run("tost_yuen", &mut || {
        let _ = tost_yuen(x, y, &bounds, 0.05, 0.2);
    });
    run("tost_bootstrap", &mut || {
        let _ = tost_bootstrap(x, y, &bounds, 0.05, 50, Some(1));
    });
    for m in [
        CorrelationTostMethod::Pearson,
        CorrelationTostMethod::Spearman,
    ] {
        run("tost_correlation", &mut || {
            let _ = tost_correlation(x, y, 0.0, &bounds, 0.05, m);
        });
    }
    run("utils::math", &mut || {
        let _ = math::mean(x);
        let _ = math::stable_mean(x);
        let _ = math::variance(x);
        let _ = math::stable_variance(x);
        let _ = math::median(x);
        let _ = math::trimmed_mean(x, 0.2);
        let _ = math::std_dev(x);
        let _ = math::skewness(x);
        let _ = math::kurtosis(x);
    });
    run("bootstrap engines", &mut || {
        let _ = StationaryBootstrap::new(2.0, Some(1)).samples(x, x.len(), 3);
        let _ = CircularBlockBootstrap::new(2, Some(1)).samples(x, x.len(), 3);
    });
    failures
}

/// Runs every function taking float scalar parameters with value `s`.
fn run_scalars(s: f64) -> Vec<String> {
    let mut failures = Vec::new();
    let mut run = |name: &str, f: &mut dyn FnMut()| {
        if catch_unwind(AssertUnwindSafe(f)).is_err() {
            failures.push(format!("{name} scalar={s}"));
        }
    };
    let x = base(12, 0.0);
    let y = base(12, 1.0);
    let z = base(12, 2.0);
    let groups: [&[f64]; 3] = [&x, &y, &z];
    let good = EquivalenceBounds::symmetric(0.5).unwrap();
    let all_bounds = [
        EquivalenceBounds::Symmetric { delta: s },
        EquivalenceBounds::Raw {
            lower: s,
            upper: 0.5,
        },
        EquivalenceBounds::Raw {
            lower: -0.5,
            upper: s,
        },
        EquivalenceBounds::CohenD { d: s },
    ];

    for alt in ALTS {
        run("mann_whitney_u", &mut || {
            let _ = mann_whitney_u(&x, &y, alt, true, false, Some(s), None);
            let _ = mann_whitney_u(&x, &y, alt, true, true, Some(s), None);
            let _ = mann_whitney_u(&x, &y, alt, true, false, Some(0.95), Some(s));
            let _ = mann_whitney_u(&x, &y, alt, true, true, Some(0.95), Some(s));
        });
        run("wilcoxon_signed_rank", &mut || {
            let _ = wilcoxon_signed_rank(&x, &y, alt, true, false, Some(s), None);
            let _ = wilcoxon_signed_rank(&x, &y, alt, true, true, Some(s), None);
            let _ = wilcoxon_signed_rank(&x, &y, alt, true, false, Some(0.95), Some(s));
            let _ = wilcoxon_signed_rank(&x, &y, alt, true, true, Some(0.95), Some(s));
            let _ = wilcoxon_signed_rank(&x, &[], alt, true, true, Some(0.95), Some(s));
        });
        run("brunner_munzel", &mut || {
            let _ = brunner_munzel(&x, &y, alt, Some(s));
        });
        for kind in [TTestKind::Welch, TTestKind::Student, TTestKind::Paired] {
            run("t_test", &mut || {
                let _ = t_test(&x, &y, kind, alt, s, Some(0.95));
                let _ = t_test(&x, &y, kind, alt, 0.0, Some(s));
            });
            run("cohens_d", &mut || {
                let _ = cohens_d(&x, &y, kind, s);
            });
        }
        run("yuen_test", &mut || {
            let _ = yuen_test(&x, &y, s, alt, Some(0.95));
            let _ = yuen_test(&x, &y, 0.2, alt, Some(s));
        });
        run("prop_test", &mut || {
            let _ = prop_test_one(6, 10, s, alt);
            let _ = prop_test_one_with_conf_level(6, 10, 0.5, alt, s);
            let _ = prop_test_two_with_conf_level([3, 6], [10, 12], alt, true, s);
            let _ = binom_test(6, 10, s, alt);
            let _ = binom_test_with_conf_level(6, 10, 0.5, alt, s);
        });
        run("fisher_exact", &mut || {
            let _ = fisher_exact_with_conf_level(&[[3, 1], [1, 3]], alt, s);
            let _ = fisher_exact_conditional(&[[3, 1], [1, 3]], alt, s);
        });
    }
    run("rank_biserial", &mut || {
        let _ = rank_biserial(&x, &y, Some(s));
        let _ = matched_pairs_rank_biserial(&x, &y, Some(s));
        let _ = rank_biserial_from_u(s, 5, 5);
    });
    run("cohens_d_one_sample", &mut || {
        let _ = cohens_d_one_sample(&x, s);
    });
    run("pearson/spearman", &mut || {
        let _ = pearson(&x, &y, Some(s));
        let _ = spearman(&x, &y, Some(s));
    });
    run("chisq_goodness_of_fit", &mut || {
        let _ = chisq_goodness_of_fit(&[3, 4, 5], Some(&[s, 0.5, 0.5]));
    });
    run("bootstrap_ci", &mut || {
        let _ = bootstrap_ci(&x, |d| d.iter().sum::<f64>(), 50, s, Some(1));
        let _ = bootstrap_ci(&x, |_| s, 50, 0.95, Some(1));
        let _ = bootstrap_mean_ci(&x, 50, s, 2, Some(1));
    });
    run("spa_test", &mut || {
        let _ = spa_test(&x, &[y.clone(), z.clone()], 30, s, Some(1));
        let _ = mspe_adjusted_spa(&x, &[y.clone(), z.clone()], 30, s, Some(1));
        let _ = model_confidence_set(
            &[x.clone(), y.clone(), z.clone()],
            s,
            MCSStatistic::Range,
            30,
            2.0,
            Some(1),
        );
        let _ = model_confidence_set(
            &[x.clone(), y.clone(), z.clone()],
            0.1,
            MCSStatistic::Max,
            30,
            s,
            Some(1),
        );
    });
    run("mmd_test", &mut || {
        let xv: Vec<Vec<f64>> = x.iter().map(|&v| vec![v]).collect();
        let yv: Vec<Vec<f64>> = y.iter().map(|&v| vec![v]).collect();
        for k in [
            Kernel::Gaussian { bandwidth: s },
            Kernel::Polynomial {
                degree: 2,
                scale: s,
                offset: s,
            },
            Kernel::Laplacian { bandwidth: s },
        ] {
            let _ = mmd_test(&xv, &yv, k, 30, Some(1));
        }
    });
    run("equivalence bounds", &mut || {
        let _ = EquivalenceBounds::symmetric(s);
        let _ = EquivalenceBounds::raw(s, 1.0);
        let _ = EquivalenceBounds::cohen_d(s);
        let _ = good.to_raw(Some(s));
        let _ = EquivalenceBounds::CohenD { d: 0.5 }.to_raw(Some(s));
    });
    for bounds in all_bounds.iter().chain(std::iter::once(&good)) {
        run("tost", &mut || {
            for alpha in [0.05, s] {
                let _ = tost_t_test_one_sample(&x, s, bounds, alpha);
                let _ = tost_t_test_two_sample(&x, &y, bounds, alpha, true);
                let _ = tost_t_test_two_sample(&x, &y, bounds, alpha, false);
                let _ = tost_t_test_paired(&x, &y, bounds, alpha);
                let _ = tost_wilcoxon_paired(&x, &y, bounds, alpha);
                let _ = tost_wilcoxon_two_sample(&x, &y, bounds, alpha);
                let _ = tost_yuen(&x, &y, bounds, alpha, 0.2);
                let _ = tost_yuen(&x, &y, bounds, 0.05, s);
                let _ = tost_bootstrap(&x, &y, bounds, alpha, 50, Some(1));
                let _ = tost_correlation(&x, &y, s, bounds, alpha, CorrelationTostMethod::Pearson);
                let _ =
                    tost_correlation(&x, &y, 0.0, bounds, alpha, CorrelationTostMethod::Spearman);
                let _ = tost_prop_one(6, 10, s, bounds, alpha);
                let _ = tost_prop_one(6, 10, 0.5, bounds, alpha);
                let _ = tost_prop_two(3, 10, 6, 12, bounds, alpha);
            }
        });
    }
    run("utils::math", &mut || {
        let _ = math::trimmed_mean(&x, s);
    });
    run("bootstrap engines", &mut || {
        let _ = StationaryBootstrap::new(s, Some(1)).samples(&x, x.len(), 3);
    });
    run("anova groups", &mut || {
        let _ = one_way_anova(&groups, AnovaKind::Welch);
    });
    failures
}

#[test]
fn issue_17_no_panic_on_fixed_non_finite_cases() {
    let mut failures = Vec::new();
    for (x, y) in fixed_cases() {
        failures.extend(run_all(&x, &y));
    }
    assert!(failures.is_empty(), "panics:\n{}", failures.join("\n"));
}

#[test]
fn issue_17_no_panic_on_random_non_finite_cases() {
    let mut failures = Vec::new();
    for (x, y) in random_cases(150) {
        failures.extend(run_all(&x, &y));
    }
    assert!(failures.is_empty(), "panics:\n{}", failures.join("\n"));
}

#[test]
fn issue_17_no_panic_on_non_finite_scalar_parameters() {
    let mut failures = Vec::new();
    for s in SCALARS {
        failures.extend(run_scalars(s));
    }
    assert!(failures.is_empty(), "panics:\n{}", failures.join("\n"));
}
