# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
