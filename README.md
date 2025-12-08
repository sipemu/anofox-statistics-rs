# libanostat

[![CI](https://github.com/sipemu/libanostat/actions/workflows/ci.yml/badge.svg)](https://github.com/sipemu/libanostat/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/libanostat.svg)](https://crates.io/crates/libanostat)
[![Documentation](https://docs.rs/libanostat/badge.svg)](https://docs.rs/libanostat)
[![codecov](https://codecov.io/gh/sipemu/libanostat/branch/main/graph/badge.svg)](https://codecov.io/gh/sipemu/libanostat)
[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

A comprehensive statistical hypothesis testing library for Rust, validated against R.

This library provides a wide range of statistical tests commonly used in data analysis, all validated against R's implementations to ensure numerical accuracy.

## Features

- **Math Primitives**
  - Mean, variance, median
  - Trimmed mean (robust to outliers)

- **Parametric Tests**
  - T-tests (Welch, Student, Paired) with all alternatives
  - Yuen's test (robust t-test using trimmed means)
  - Brown-Forsythe test (homogeneity of variances)

- **Nonparametric Tests**
  - Ranking with average tie handling
  - Mann-Whitney U test (Wilcoxon rank-sum)
  - Wilcoxon signed-rank test (paired)
  - Kruskal-Wallis test (k-sample)

- **Distributional Tests**
  - Shapiro-Wilk normality test (Royston AS R94)

- **Forecast Evaluation**
  - Diebold-Mariano test for comparing predictive accuracy

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
libanostat = "0.1"
```

## Quick Start

### T-Tests

```rust
use libanostat::{t_test, TTestKind, Alternative};

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
use libanostat::yuen_test;

// 20% trimmed means (robust to outliers)
let result = yuen_test(&group1, &group2, 0.2)?;

println!("Test statistic: {:.4}", result.statistic);
println!("p-value: {:.4}", result.p_value);
```

### Brown-Forsythe Test

```rust
use libanostat::brown_forsythe;

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
use libanostat::{mann_whitney_u, wilcoxon_signed_rank, kruskal_wallis, rank};

// Ranking
let data = vec![3.0, 1.0, 4.0, 1.0, 5.0];
let ranks = rank(&data)?;

// Mann-Whitney U test
let result = mann_whitney_u(&group1, &group2)?;

// Wilcoxon signed-rank test (paired)
let result = wilcoxon_signed_rank(&group1, &group2)?;

// Kruskal-Wallis test
let result = kruskal_wallis(&groups)?;
```

### Shapiro-Wilk Normality Test

```rust
use libanostat::shapiro_wilk;

let data = vec![1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8];

let result = shapiro_wilk(&data)?;

println!("W statistic: {:.4}", result.statistic);
println!("p-value: {:.4}", result.p_value);

// High p-value suggests data is consistent with normal distribution
if result.p_value > 0.05 {
    println!("Cannot reject normality at 5% level");
}
```

### Diebold-Mariano Forecast Test

```rust
use libanostat::{diebold_mariano, LossFunction};

// Forecast errors from two competing models
let errors_model1 = vec![0.1, -0.2, 0.3, -0.1, 0.2];
let errors_model2 = vec![0.2, -0.3, 0.4, -0.2, 0.3];

// Test with squared error loss, h=1 step ahead
let result = diebold_mariano(&errors_model1, &errors_model2, LossFunction::SquaredError, 1)?;

println!("DM statistic: {:.4}", result.statistic);
println!("p-value: {:.4}", result.p_value);

// Test with absolute error loss
let result = diebold_mariano(&errors_model1, &errors_model2, LossFunction::AbsoluteError, 1)?;
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

### Distributional Tests

| Function | Description |
|----------|-------------|
| `shapiro_wilk(data)` | Shapiro-Wilk test for normality |

### Forecast Evaluation

| Function | Description |
|----------|-------------|
| `diebold_mariano(e1, e2, loss, h)` | Diebold-Mariano test for predictive accuracy |

### Math Primitives

| Function | Description |
|----------|-------------|
| `mean(data)` | Arithmetic mean |
| `variance(data)` | Sample variance |
| `median(data)` | Median |
| `trimmed_mean(data, trim)` | Trimmed mean |

## Validation

This library is developed using Test-Driven Development (TDD) with R as the oracle (ground truth). All implementations are validated against R's statistical functions:

- `t.test()` for t-tests
- `WRS2::yuen()` for Yuen's test
- `car::leveneTest()` for Brown-Forsythe
- `wilcox.test()` for Mann-Whitney and Wilcoxon tests
- `kruskal.test()` for Kruskal-Wallis
- `shapiro.test()` for Shapiro-Wilk
- `forecast::dm.test()` for Diebold-Mariano

All tests ensure numerical agreement with R within appropriate tolerances.

## Dependencies

- [statrs](https://crates.io/crates/statrs) - Statistical distributions
- [thiserror](https://crates.io/crates/thiserror) - Error handling

## License

MIT License
