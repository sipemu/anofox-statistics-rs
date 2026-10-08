# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.6] - 2026-10-08

### Fixed

- **`mcnemar_exact` no longer returns NaN for more than ~1,074 discordant pairs** (#20):
  binomial probabilities were computed as `exp(log C(n, k)) * 0.5^n`, where `0.5^n`
  underflows to 0 and `exp(log C(n, k))` overflows to infinity once `n = b + c`
  exceeds ~1074 (0 * inf = NaN); the log-factorials also used Stirling's approximation
  above 20 without correction terms. The two-sided p-value is now
  `min(1, 2 * P(X <= min(b, c)))`, `X ~ Binomial(b + c, 0.5)`, via the regularized
  incomplete beta function (statrs `Binomial::cdf`), matching R's
  `binom.test(b, b + c, 0.5)$p.value` (= `exact2x2::mcnemar.exact`) to < 1e-9 relative
  error from `b + c = 1` up to `b + c = 10^6`.
- Audited the other exact tests for the same failure: `binom_test` (statrs
  Binomial), `fisher_exact` (log-space hypergeometric with `ln_factorial`) and the
  exact Mann-Whitney / signed-rank distributions (`f64` counts, gated to small
  samples) were already safe; regression tests against R at `n = 10^6` / ~200,000
  now guard them.

## [0.4.5] - 2026-10-08

### Fixed

- **Wilcoxon / Mann-Whitney confidence intervals no longer need O(n²) memory** (#19):
  `mann_whitney_u` and `wilcoxon_signed_rank` with `conf_level` materialised all
  `n1 * n2` pairwise differences (all `n(n+1)/2` Walsh averages) and sorted them —
  a 31 GB allocation (process abort / OOM kill) at 62,500 observations per group,
  even with `exact = false`. The estimate and interval now follow R's
  `wilcox.test(..., conf.int = TRUE)` algorithm: on the normal-approximation path the
  shift is root-found with a port of R's `zeroin` (`uniroot`, tolerance 1e-4) on the
  standardised rank statistic, each evaluation an O(n) merge of the presorted
  samples — **O(n log n) time, O(n) memory** (100,000 per group: ~60 ms, 13 MB peak
  RSS in release; 0.4.4 needed ~80 GB). The exact path (no ties, small samples) keeps
  the order statistics with R's quantile rule.
- **Intervals now match R** (126 reference cases, agreement < 1e-10): the asymptotic
  estimate/limits are R's root-finding values (0.4.4 used order statistics with a
  normal-approximation index, off by ~1e-4); one-sided alternatives give one-sided
  intervals (`(-Inf, u]`, `[l, Inf)`); the continuity correction is applied to the
  interval as in R; the signed-rank interval is computed on `x - y` (not shifted by
  `mu`, zeros kept) as R does. With an infinite observation the asymptotic
  estimate/interval is NaN (R's root finder cannot search an infinite bracket).
- **Exact null distributions no longer overflow**: the `u64` counts overflowed for
  `n1 + n2 > ~62` / `n ≥ 64`; the pmf is now computed in `f64` with additions only.
  `exact = true` is honoured for `n1 * n2 ≤ 10,000` (rank sum) and `n ≤ 300` non-zero
  differences (signed rank); larger samples use the normal approximation, so an
  explicit `exact = true` on database-sized input never runs the exact DP.
- **No O(n²) memory anywhere on a single sample** (crate-wide audit for #19); results
  are unchanged (exact integer counts / identical order statistics, otherwise within
  1e-10), and every touched function documents its time and memory complexity:
  - `kendall`: Knight's merge-sort algorithm, O(n log n) time (was O(n²)), exact
    tie counts for tau-b/tau-c and the variance (100k: 0.03 s).
  - `distance_cor` / `distance_cor_test`: Huo & Székely (2016) O(n log n) algorithm
    with O(n) memory; 0.4.4 built four n×n matrices (~80 GB each at n = 100,000).
  - `tost_wilcoxon_paired` / `tost_wilcoxon_two_sample`: Hodges-Lehmann order
    statistics are selected without materialising all Walsh averages / pairwise
    differences (new `utils::select`, O(n) memory); out-of-range CI indices for
    `alpha > 0.5` no longer panic.
  - `mmd_test_1d`: median-heuristic bandwidth by selection, O(n) memory (was n²/2
    stored distances).
  - `energy_distance_test_1d` (and d = 1 inputs): O(N log N + B·N) via sorted
    prefix sums instead of O(B·N²).
  - Inherently quadratic-time statistics (MMD, multivariate energy distance) stream
    their pairwise sums with O(N·d) memory; documented.

## [0.4.4] - 2026-10-08

### Fixed

- **No panics on non-finite input** (#17): `wilcoxon_signed_rank` panicked on data
  containing `±Inf` (`Inf - Inf = NaN` reached `partial_cmp(..).unwrap()` in the
  ranking sort), and about 20 other sort sites, plus several `statrs` calls, could
  panic in the same way on NaN. The crate was audited, and every public function
  now returns a result or an error for NaN/`±Inf` data and scalar parameters:
  - All sorts use `f64::total_cmp`. `statrs` survival functions and quantiles go
    through NaN-safe wrappers, so a NaN statistic gives a NaN p-value instead of a
    panic. This covers ANOVA (including sphericity corrections), TOST t/Yuen,
    partial/Pearson/Spearman correlation, ICC, Brunner-Munzel and Kruskal-Wallis.
  - **Rank-based tests match R:** `wilcoxon_signed_rank`, `mann_whitney_u`,
    `kruskal_wallis`, `brunner_munzel`, `rank` and the rank-biserial helpers rank
    `±Inf` as the largest/smallest value, as R's `wilcox.test`/`kruskal.test` do.
    Paired `Inf - Inf` differences are dropped, as R does. `NaN` input returns the
    `non-finite value` error. A Hodges-Lehmann estimate/CI made undefined by
    `Inf - Inf` is NaN.
  - Non-finite data now returns the `non-finite value` error instead of a
    meaningless finite p-value in `permutation_t_test`, the energy-distance and MMD
    tests, `clark_west`, `diebold_mariano`, `spa_test`, `mspe_adjusted_spa`,
    `model_confidence_set` and the Wilcoxon TOSTs. `yuen_test`, `tost_yuen`,
    `utils::math::median` and `trimmed_mean` reject NaN (`±Inf` can be trimmed).
  - NaN parameters are rejected: equivalence bounds, MMD kernel parameters, MCS
    `alpha`, Wilcoxon/Mann-Whitney `mu`, and `chisq_goodness_of_fit` expected
    proportions (which also must be non-negative). A bootstrap percentile interval
    is NaN when a replicate is NaN.
- Added a property-style regression test that feeds NaN/`±Inf` (fixed patterns,
  random fuzzing and constant data) to every public function and asserts that none
  of them panic.

## [0.4.3] - 2026-10-07

### Fixed

- **Shapiro-Wilk p-value for n <= 11** (#7): `shapiro_wilk` now ports R's `swilk.c`
  (Royston 1995, AS R94) line for line, including the small-sample (4 <= n <= 11)
  gamma transform and the exact n <= 5 coefficient path. W and p agree with
  `shapiro.test()` to ~1e-12 (previously p = 0.070 vs R's 0.161 for an n = 10 sample).
- **Kendall tau test with ties** (#8): the variance of S now uses the full tie-corrected
  formula (R `cor.test(method = "kendall", exact = FALSE)`); previously an ad-hoc
  scaling was used, giving wrong z / p-values whenever either variable had ties.
- **Mann-Whitney / Wilcoxon continuity correction at z = 0** (#13): like R
  (`CORRECTION = sign(z) * 0.5`) no correction is applied when the statistic equals
  its null expectation, so p = 1 instead of < 1. Mann-Whitney with every observation
  tied (Var(U) = 0) now returns p = 1 instead of p = 0.
- **Binomial test** (#9): the Clopper-Pearson interval now uses the exact beta quantile
  (the previous approximation was off by ~5e-3 and clamped to [0.001, 0.999]); the
  two-sided p-value uses R's relative tolerance instead of an absolute 1e-10, which
  floored p-values for large n. Intervals for one-sided alternatives are one-sided,
  as in R `binom.test`.
- **Proportion tests** (#10): `prop_test_two` interval uses `qnorm` instead of the rounded
  1.96 and includes R's continuity correction when `correction = true`;
  one-sided alternatives give one-sided intervals (R `prop.test`). Same for the
  Wilson interval of `prop_test_one`.
- **Fisher's exact test** (#11): Woolf interval uses `qnorm` instead of 1.96; the two-sided
  p-value uses R's relative tolerance; log-factorials are exact (statrs) instead of a
  Stirling approximation. Cohen's kappa CI uses `qnorm(0.975)`.
- **Exact two-sided Mann-Whitney p-value** (#12): `mann_whitney_u(.., exact = true)`
  returned p = 1 whenever U was above its mean n1*n2/2 (the "upper" tail mirrored back
  onto the lower one). It now follows R `wilcox.test`: the tail on the side of the
  observed U, doubled and capped at 1 (W = 58: p = 0.3153781203 instead of 1).
- **Pearson correlation on a constant variable** (#14): `pearson` panicked inside
  statrs (`XOutOfRange`); it now returns NaN estimate / statistic / p-value and no
  interval (R: `NA`, scipy: `nan`). `t_test`, `one_way_anova` (Fisher),
  `brown_forsythe` and `yuen_test` likewise return a NaN p-value instead of panicking
  when the statistic is 0/0 (constant samples).

### Added

- `binom_test_with_conf_level`, `prop_test_one_with_conf_level`,
  `prop_test_two_with_conf_level`, `fisher_exact_with_conf_level`: variants taking the
  confidence level of the reported interval (the existing functions use 0.95).
- `fisher_exact_conditional(table, alternative, conf_level)` (#11): R `fisher.test`
  semantics, i.e. the conditional maximum-likelihood odds ratio and the exact
  conditional confidence interval (one-sided for one-sided alternatives).
  `fisher_exact` keeps reporting the sample odds ratio with a Woolf interval
  (now documented as such).
- Effect sizes (#15): `cohens_d(x, y, kind, mu)` (Student: pooled SD; Welch:
  average-variance standardiser; Paired: d_z) and `cohens_d_one_sample(x, mu)`;
  `rank_biserial(x, y, mu)` / `rank_biserial_from_u(u1, n1, n2)` for Mann-Whitney
  (`r = 1 - 2 U1 / (n1 n2)`, positive when `y` tends to be larger) and
  `matched_pairs_rank_biserial(x, y, mu)` for the signed-rank test.
- `jarque_bera(data)`: Jarque-Bera normality test (R `tseries::jarque.bera.test`).
- `bootstrap_ci(data, statistic, n_bootstrap, conf_level, seed)` and
  `bootstrap_mean_ci(data, n_bootstrap, conf_level, block_length, seed)`: percentile
  bootstrap confidence intervals (IID, or stationary / circular block bootstrap for the
  mean).

These additions move method code out of the DuckDB extension
[anofox-statistics](https://github.com/DataZooDE/anofox-statistics), which now only
needs to delegate to the crate.

## [0.4.2] - 2026-06-01

### Fixed

- **Numerical stability for extreme p-values** (#6): Replaced `1.0 - cdf(x)` with the
  distribution's survival function (`sf`) throughout the library. The old form lost
  all precision once `cdf` saturated at 1.0 (around |z| > 8.3 for the normal
  distribution), causing p-values smaller than ~1.4e-14 to underflow to exactly 0.0.
  Affected tests include Mann-Whitney U, Wilcoxon signed-rank, t-tests (one/two
  sample, Welch, paired), Yuen's trimmed-mean test, ANOVA (one-way, two-way,
  repeated-measures, Mauchly, Greenhouse-Geisser, Huynh-Feldt), Levene, Brunner-
  Munzel, Kruskal-Wallis, chi-square (Pearson, G-test, McNemar's), Shapiro-Wilk,
  D'Agostino, Pearson/Spearman/Kendall/partial correlation, ICC, one-/two-proportion
  tests, Cohen's kappa, Diebold-Mariano, Clark-West, and all TOST variants.

## [0.4.0] - 2025-12-17

### Added

- **Equivalence Testing Module**: New `equivalence` module providing TOST (Two One-Sided Tests) procedures for establishing practical equivalence

- **TOST for T-Tests**:
  - `tost_t_test_one_sample()`: One-sample equivalence test
  - `tost_t_test_two_sample()`: Two-sample equivalence test (Welch/Student)
  - `tost_t_test_paired()`: Paired samples equivalence test

- **TOST for Correlations**:
  - `tost_correlation()`: Test if correlation is practically equivalent to zero (or other value)
  - Supports Pearson and Spearman methods via `CorrelationTostMethod` enum
  - Uses Fisher's z-transformation for inference

- **TOST for Proportions**:
  - `tost_prop_one()`: Single proportion equivalence test
  - `tost_prop_two()`: Two proportions equivalence test

- **Non-parametric TOST**:
  - `tost_wilcoxon_paired()`: Paired Wilcoxon signed-rank equivalence test
  - `tost_wilcoxon_two_sample()`: Two-sample Wilcoxon rank-sum equivalence test
  - Uses Hodges-Lehmann estimator for location

- **Robust and Resampling TOST**:
  - `tost_bootstrap()`: Bootstrap-based equivalence test with percentile CI
  - `tost_yuen()`: Robust trimmed means equivalence test

- **New Types**:
  - `TostResult`: Comprehensive result struct with estimates, CIs, bounds, and test statistics
  - `OneSidedTestResult`: Result for each one-sided test within TOST
  - `EquivalenceBounds`: Enum supporting Raw, Symmetric, and Cohen's d bounds
  - `CorrelationTostMethod`: Enum for Pearson/Spearman correlation methods

- **Documentation**:
  - Full API documentation for all 10 TOST functions
  - R validation reference (TOSTER package)
  - Usage examples in README and API reference

### R Equivalents

All TOST functions are designed to match the TOSTER package in R:
- `TOSTER::TOSTone()`, `TOSTtwo()`, `TOSTpaired()` for t-tests
- `TOSTER::TOSTr()` for correlations
- `TOSTER::TOSTtwo.prop()` for proportions
- `TOSTER::wilcox_TOST()` for Wilcoxon tests
- `TOSTER::boot_t_TOST()` for bootstrap

## [0.3.0] - 2025-12-12

### Added

- **Full R Output Parity**: All statistical test result structs now include all fields returned by R's `htest` objects (except `method` and `data.name`)
  - `DMResult`: Added `horizon`, `loss_function`, `varestimator`, `alternative` fields
  - `TTestResult`: Added `null_value` field
  - `MannWhitneyResult`: Added `null_value` field
  - `WilcoxonResult`: Added `null_value` field
  - `YuenResult`: Added `conf_int` field with `YuenConfInt` struct

- **Full R Input Parameter Parity**: All statistical tests now support the same parameters as their R equivalents
  - `mann_whitney_u()`: Added `alternative`, `continuity_correction`, `exact`, `conf_level`, `mu` parameters
  - `wilcoxon_signed_rank()`: Added `alternative`, `continuity_correction`, `exact`, `conf_level`, `mu` parameters
  - `diebold_mariano()`: Added `alternative`, `varestimator` parameters
  - `yuen_test()`: Added `alternative`, `conf_level` parameters
  - `t_test()`: Added `mu`, `conf_level` parameters
  - `brunner_munzel()`: Added `alternative`, `alpha` parameters

- **Exact P-Values**: Mann-Whitney U and Wilcoxon signed-rank tests now support exact p-value computation for small samples without ties

- **Confidence Intervals**: Added Hodges-Lehmann confidence intervals for Mann-Whitney U and Wilcoxon signed-rank tests

- **Numerically Stable Algorithms**: Added `stable_mean()` and `stable_variance()` using Welford's online algorithm

- **CI/CD**: Added automatic cargo publish to crates.io on GitHub release

- **Documentation**:
  - Comprehensive API reference in `doc/API_REFERENCE.md`
  - Scientific references for all statistical tests
  - Runnable examples for all test categories
  - R validation documentation in `R/VALIDATION.md`

### Changed

- Refactored complex modules to improve code maintainability
- Consolidated documentation structure

## [0.2.0] - 2025-12-09

### Added

- **Complete R Validation Coverage**: All 31 statistical tests now have comprehensive R validation
- **New Test Modules**:
  - `tdd_modern.rs` - Tests for Energy Distance and MMD
  - `tdd_resampling.rs` - Tests for Permutation T-Test and Bootstrap methods

### Validated Against R

The following tests now have R reference validation:

#### Distributional Tests
- D'Agostino's K-squared test (validated against `moments::agostino.test()`)

#### Nonparametric Tests
- Brunner-Munzel test (validated against `lawstat::brunner.munzel.test()`)

#### Forecast Evaluation
- Clark-West test for nested model comparison
- Superior Predictive Ability (SPA) test
- Model Confidence Set (MCS) with Range and Semi-Quadratic statistics

#### Modern Distribution Tests
- Energy Distance test (validated against `energy::eqdist.etest()`)
- Maximum Mean Discrepancy (MMD) test

#### Resampling Methods
- Permutation T-test (validated against `coin::independence_test()`)
- Stationary Bootstrap
- Circular Block Bootstrap

### Test Coverage

- Total tests: 194 (all passing)
- R validation coverage: 100% of statistical tests

## [0.1.0] - 2025-01-XX

### Added

- Initial release
- **Parametric Tests**: T-tests (Welch, Student, Paired), Yuen's test, Brown-Forsythe
- **Nonparametric Tests**: Mann-Whitney U, Wilcoxon signed-rank, Kruskal-Wallis
- **Distributional Tests**: Shapiro-Wilk normality test
- **Forecast Evaluation**: Diebold-Mariano test
- **Math Primitives**: Mean, variance, std_dev, median, trimmed_mean, skewness, kurtosis
- R validation framework with CSV-based reference data
