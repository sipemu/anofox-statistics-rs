# anofox-statistics

[![CI](https://github.com/sipemu/anofox-statistics-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/sipemu/anofox-statistics-rs/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/anofox-statistics.svg)](https://crates.io/crates/anofox-statistics)
[![Documentation](https://docs.rs/anofox-statistics/badge.svg)](https://docs.rs/anofox-statistics)
[![codecov](https://codecov.io/gh/sipemu/anofox-statistics-rs/branch/main/graph/badge.svg)](https://codecov.io/gh/sipemu/stattests-rs)
[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
[![Code Health](https://img.shields.io/badge/code%20health-85%25-brightgreen)](.)

A comprehensive statistical hypothesis testing library for Rust, validated against R.

This library provides a wide range of statistical tests commonly used in data analysis, all validated against R's implementations to ensure numerical accuracy.

## Features

- **Math Primitives**
  - Mean, variance, standard deviation, median
  - Trimmed mean (robust to outliers)
  - Skewness and kurtosis (Fisher's definition, matching R's e1071)

- **Parametric Tests**
  - T-tests (Welch, Student, Paired) with all alternatives
  - Yuen's test (robust t-test using trimmed means)
  - Brown-Forsythe test (homogeneity of variances)

- **Nonparametric Tests**
  - Ranking with average tie handling
  - Mann-Whitney U test (Wilcoxon rank-sum)
  - Wilcoxon signed-rank test (paired)
  - Kruskal-Wallis test (k-sample)
  - Brunner-Munzel test (robust rank-based test for stochastic equality)

- **Distributional Tests**
  - Shapiro-Wilk normality test (Royston AS R94)
  - D'Agostino's K-squared test (omnibus normality test using skewness and kurtosis)

- **Resampling Methods**
  - Permutation engine with custom statistics
  - Permutation t-test
  - Stationary bootstrap (for dependent data)
  - Circular block bootstrap

- **Modern Distribution Tests**
  - Energy distance test (univariate and multivariate)
  - Maximum Mean Discrepancy (MMD) with multiple kernels (Gaussian, Linear, Polynomial, Laplacian)

- **Forecast Evaluation**
  - Diebold-Mariano test for comparing predictive accuracy
  - Clark-West test for nested model comparison
  - Superior Predictive Ability (SPA) test for multiple model comparison
  - MSPE-Adjusted SPA test for multiple nested models (Clark-West + bootstrap)
  - Model Confidence Set (Hansen, Lunde, & Nason, 2011)

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
anofox-statistics = "0.2"
```

## Examples

The library includes runnable examples demonstrating each major feature:

```bash
cargo run --example quickstart      # Overview: t-test, Mann-Whitney, Shapiro-Wilk, permutation test
cargo run --example parametric      # T-tests, Yuen's robust test, Brown-Forsythe
cargo run --example nonparametric   # Ranking, Mann-Whitney, Wilcoxon, Kruskal-Wallis, Brunner-Munzel
cargo run --example normality       # Shapiro-Wilk, D'Agostino's K-squared
cargo run --example resampling      # Permutation tests, bootstrap methods
cargo run --example modern          # Energy distance, MMD with different kernels
cargo run --example forecast        # Diebold-Mariano, Clark-West, SPA, MCS
```

## Quick Start

### T-Tests

```rust
use anofox_statistics::{t_test, TTestKind, Alternative};

let group1 = vec![1.2, 2.3, 3.4, 4.5, 5.6];
let group2 = vec![2.1, 3.2, 4.3, 5.4, 6.5];

// Welch t-test (unequal variances)
let result = t_test(&group1, &group2, TTestKind::Welch, Alternative::TwoSided)
    .expect("t-test should succeed");

println!("t-statistic: {:.4}", result.statistic);
println!("p-value: {:.4}", result.p_value);
println!("degrees of freedom: {:.4}", result.df);

// Student t-test (equal variances assumed)
let result = t_test(&group1, &group2, TTestKind::Student, Alternative::TwoSided)?;

// Paired t-test
let result = t_test(&group1, &group2, TTestKind::Paired, Alternative::Less)?;
```

### Yuen's Robust T-Test

```rust
use anofox_statistics::yuen_test;

// 20% trimmed means (robust to outliers)
let result = yuen_test(&group1, &group2, 0.2)?;

println!("Test statistic: {:.4}", result.statistic);
println!("p-value: {:.4}", result.p_value);
```

### Brown-Forsythe Test

```rust
use anofox_statistics::brown_forsythe;

let groups = vec![
    vec![1.0, 2.0, 3.0],
    vec![4.0, 5.0, 6.0],
    vec![7.0, 8.0, 9.0],
];

let result = brown_forsythe(&groups)?;

println!("F-statistic: {:.4}", result.statistic);
println!("p-value: {:.4}", result.p_value);
```

### Nonparametric Tests

```rust
use anofox_statistics::{mann_whitney_u, wilcoxon_signed_rank, kruskal_wallis, rank, brunner_munzel};

// Ranking
let data = vec![3.0, 1.0, 4.0, 1.0, 5.0];
let ranks = rank(&data)?;

// Mann-Whitney U test
let result = mann_whitney_u(&group1, &group2)?;

// Wilcoxon signed-rank test (paired)
let result = wilcoxon_signed_rank(&group1, &group2)?;

// Kruskal-Wallis test
let result = kruskal_wallis(&groups)?;

// Brunner-Munzel test (robust alternative to Mann-Whitney)
let result = brunner_munzel(&group1, &group2)?;
println!("Estimate P(X < Y): {:.4}", result.estimate);
```

### Normality Tests

```rust
use anofox_statistics::{shapiro_wilk, dagostino_k_squared};

let data = vec![1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8];

// Shapiro-Wilk test
let result = shapiro_wilk(&data)?;
println!("W statistic: {:.4}", result.statistic);
println!("p-value: {:.4}", result.p_value);

// D'Agostino's K-squared test (omnibus test using skewness and kurtosis)
let result = dagostino_k_squared(&data)?;
println!("K-squared: {:.4}", result.statistic);
println!("p-value: {:.4}", result.p_value);
```

### Resampling Methods

```rust
use anofox_statistics::resampling::{permutation_t_test, StationaryBootstrap, CircularBlockBootstrap};

// Permutation t-test
let result = permutation_t_test(&group1, &group2, 10000, Some(42))?;
println!("p-value: {:.4}", result.p_value);

// Stationary bootstrap for time series
let bootstrap = StationaryBootstrap::new(&time_series, 10.0, Some(42))?;
let samples: Vec<Vec<f64>> = bootstrap.take(1000).collect();

// Circular block bootstrap
let bootstrap = CircularBlockBootstrap::new(&time_series, 5, Some(42))?;
```

### Modern Distribution Tests

```rust
use anofox_statistics::modern::{energy_distance_test, mmd_test, Kernel};

// Energy distance test
let result = energy_distance_test(&sample1, &sample2, 1000, Some(42))?;
println!("Energy distance: {:.4}", result.statistic);
println!("p-value: {:.4}", result.p_value);

// Maximum Mean Discrepancy with Gaussian kernel
let result = mmd_test(&sample1, &sample2, Kernel::Gaussian(1.0), 1000, Some(42))?;
println!("MMD: {:.4}", result.statistic);
println!("p-value: {:.4}", result.p_value);

// MMD with automatic bandwidth selection
let result = mmd_test(&sample1, &sample2, Kernel::GaussianMedian, 1000, Some(42))?;
```

### Forecast Evaluation

```rust
use anofox_statistics::{diebold_mariano, clark_west, spa_test, mspe_adjusted_spa,
                        model_confidence_set, LossFunction, MCSStatistic};

// Forecast errors from two competing models
let errors_model1 = vec![0.1, -0.2, 0.3, -0.1, 0.2];
let errors_model2 = vec![0.2, -0.3, 0.4, -0.2, 0.3];

// Diebold-Mariano test
let result = diebold_mariano(&errors_model1, &errors_model2, LossFunction::SquaredError, 1)?;
println!("DM statistic: {:.4}", result.statistic);
println!("p-value: {:.4}", result.p_value);

// Clark-West test for nested models (e.g., AR(1) vs AR(2))
let restricted_errors = vec![0.3, -0.2, 0.4, -0.3, 0.2];   // Benchmark/restricted model
let unrestricted_errors = vec![0.2, -0.1, 0.3, -0.2, 0.1]; // Alternative/unrestricted model
let result = clark_west(&restricted_errors, &unrestricted_errors, 1)?;
println!("CW statistic: {:.4}", result.statistic);
println!("p-value (one-sided): {:.4}", result.p_value);

// Superior Predictive Ability test (compare benchmark vs multiple models)
let benchmark_losses = vec![0.5, 0.6, 0.4, 0.7, 0.5];
let model_losses = vec![
    vec![0.4, 0.5, 0.3, 0.6, 0.4],  // Model 1
    vec![0.6, 0.7, 0.5, 0.8, 0.6],  // Model 2
];
let result = spa_test(&benchmark_losses, &model_losses, 1000, 10.0, Some(42))?;
println!("SPA statistic: {:.4}", result.statistic);
println!("p-value: {:.4}", result.p_value_consistent);

// MSPE-Adjusted SPA for multiple nested models
// Combines Clark-West adjustment with bootstrap for multiple testing
let benchmark_errors = vec![0.5, 0.4, 0.6, 0.3, 0.5];
let nested_model_errors = vec![
    vec![0.4, 0.3, 0.5, 0.2, 0.4],  // Nested model 1
    vec![0.3, 0.2, 0.4, 0.1, 0.3],  // Nested model 2
];
let result = mspe_adjusted_spa(&benchmark_errors, &nested_model_errors, 1000, 5.0, Some(42))?;
println!("Best model: {:?}", result.best_model_idx);
println!("p-value (adjusted): {:.4}", result.p_value_consistent);

// Model Confidence Set - identify the set of best models
let losses = vec![
    vec![0.5, 0.6, 0.4, 0.7, 0.5],  // Model 0
    vec![0.4, 0.5, 0.3, 0.6, 0.4],  // Model 1
    vec![0.8, 0.9, 0.7, 1.0, 0.8],  // Model 2 (worst)
];
let result = model_confidence_set(&losses, 0.10, MCSStatistic::Range, 1000, 5.0, Some(42))?;
println!("Models in MCS: {:?}", result.included_models);
println!("Eliminated: {:?}", result.eliminated_models);
```

## API Reference

### Parametric Tests

| Function | Description |
|----------|-------------|
| `t_test(x, y, kind, alternative)` | T-test (Welch, Student, or Paired) |
| `yuen_test(x, y, trim)` | Yuen's trimmed mean t-test |
| `brown_forsythe(groups)` | Brown-Forsythe test for homogeneity of variances |

### Nonparametric Tests

| Function | Description |
|----------|-------------|
| `rank(data)` | Compute ranks with average tie handling |
| `mann_whitney_u(x, y)` | Mann-Whitney U test (Wilcoxon rank-sum) |
| `wilcoxon_signed_rank(x, y)` | Wilcoxon signed-rank test for paired samples |
| `kruskal_wallis(groups)` | Kruskal-Wallis H test for k independent samples |
| `brunner_munzel(x, y)` | Brunner-Munzel test for stochastic equality |

### Distributional Tests

| Function | Description |
|----------|-------------|
| `shapiro_wilk(data)` | Shapiro-Wilk test for normality |
| `dagostino_k_squared(data)` | D'Agostino's K-squared omnibus normality test |

### Resampling Methods

| Function | Description |
|----------|-------------|
| `permutation_t_test(x, y, n_permutations, seed)` | Permutation-based t-test |
| `PermutationEngine::new(x, y, seed)` | Generic permutation testing engine |
| `StationaryBootstrap::new(data, expected_length, seed)` | Stationary bootstrap for dependent data |
| `CircularBlockBootstrap::new(data, block_length, seed)` | Circular block bootstrap |

### Modern Distribution Tests

| Function | Description |
|----------|-------------|
| `energy_distance_test(x, y, n_permutations, seed)` | Energy distance two-sample test |
| `mmd_test(x, y, kernel, n_permutations, seed)` | Maximum Mean Discrepancy test |

### Forecast Evaluation

| Function | Description |
|----------|-------------|
| `diebold_mariano(e1, e2, loss, h)` | Diebold-Mariano test for predictive accuracy |
| `clark_west(e1, e2, h)` | Clark-West test for nested model comparison |
| `spa_test(benchmark, models, n_bootstrap, block_length, seed)` | Superior Predictive Ability test |
| `mspe_adjusted_spa(benchmark, models, n_bootstrap, block_length, seed)` | MSPE-Adjusted SPA for multiple nested models |
| `model_confidence_set(losses, alpha, statistic, n_bootstrap, block_length, seed)` | Model Confidence Set (Hansen et al., 2011) |

### Math Primitives

| Function | Description |
|----------|-------------|
| `mean(data)` | Arithmetic mean |
| `variance(data)` | Sample variance |
| `std_dev(data)` | Sample standard deviation |
| `median(data)` | Median |
| `trimmed_mean(data, trim)` | Trimmed mean |
| `skewness(data)` | Sample skewness (Fisher's, type 2) |
| `kurtosis(data)` | Sample excess kurtosis (Fisher's, type 2) |

## Validation

This library is developed using Test-Driven Development (TDD) with R as the oracle (ground truth). All implementations are validated against R's statistical functions:

| Rust Function | R Equivalent | Package |
|---------------|--------------|---------|
| `t_test()` | `t.test()` | stats |
| `yuen_test()` | `yuen()` | WRS2 |
| `brown_forsythe()` | `leveneTest(center=median)` | car |
| `mann_whitney_u()`, `wilcoxon_signed_rank()` | `wilcox.test()` | stats |
| `kruskal_wallis()` | `kruskal.test()` | stats |
| `brunner_munzel()` | `brunner.munzel.test()` | lawstat |
| `shapiro_wilk()` | `shapiro.test()` | stats |
| `dagostino_k_squared()` | `agostino.test()`, `anscombe.test()` | moments |
| `skewness()`, `kurtosis()` | `skewness()`, `kurtosis()` | e1071 |
| `diebold_mariano()` | `dm.test()` | forecast |

All 134 test cases ensure numerical agreement with R within appropriate tolerances (typically 1e-10, with documented exceptions for algorithm-dependent tests like Shapiro-Wilk).

**For complete transparency on the validation process, see [`R/VALIDATION.md`](R/VALIDATION.md)**, which documents:
- All 76 reference data files and their R generation code
- Tolerance rationale for each test category
- Step-by-step reproduction instructions
- R package dependencies

## Dependencies

- [statrs](https://crates.io/crates/statrs) - Statistical distributions
- [thiserror](https://crates.io/crates/thiserror) - Error handling
- [rand](https://crates.io/crates/rand) - Random number generation for resampling

## License

MIT License
