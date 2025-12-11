# R Validation Report

This document provides complete transparency into how the Rust statistical implementations are validated against R.

## Overview

All statistical tests in this library are validated using **Test-Driven Development (TDD)** with R as the ground truth oracle. The validation process ensures numerical agreement between Rust implementations and established R packages.

## Validation Architecture

```
┌─────────────────────────┐     ┌─────────────────────────┐
│   R/generate_refs.R     │     │   tests/tdd_*.rs        │
│   (Reference Generator) │     │   (Rust Test Suite)     │
└───────────┬─────────────┘     └───────────┬─────────────┘
            │                               │
            │  set.seed(42)                 │  load CSV files
            │  compute expected values      │  compute Rust values
            │                               │  assert_relative_eq!
            ▼                               ▼
┌─────────────────────────────────────────────────────────┐
│                    R/data/*.csv                         │
│              (76 Reference Data Files)                  │
│                                                         │
│   Test Vectors:  Input data generated in R              │
│   Expected Values: R-computed reference results         │
└─────────────────────────────────────────────────────────┘
```

## How to Reproduce Validation

### Step 1: Generate R Reference Data

```bash
cd /path/to/anofox-statistics
Rscript R/generate_refs.R
```

This generates 76 CSV files in `R/data/` containing:
- Test vectors (input data)
- Expected results (R-computed values)

### Step 2: Run Rust Tests

```bash
cargo test
```

Tests load the CSV files and compare Rust results against R references.

## Reference Data Files

### Test Vector Files (Input Data)

| File | Description | R Generation Code |
|------|-------------|-------------------|
| `vec_short.csv` | 5-element vector | `c(1.2, 2.3, 3.4, 4.5, 5.6)` |
| `vec_long.csv` | 100-element normal sample | `rnorm(100, mean=10, sd=2)` |
| `vec_outlier.csv` | Vector with outlier | `c(1, 1, 1, 1, 100)` |
| `vec_even.csv` | Even-length vector | `c(1.0, 2.0, 3.0, 4.0)` |
| `vec_skewed.csv` | Right-skewed data | `c(1,1,1,1,1,2,2,3,5,10,20)` |
| `ttest_g1.csv` | T-test group 1 (n=20) | `rnorm(20, mean=5, sd=1)` |
| `ttest_g2.csv` | T-test group 2 (n=25) | `rnorm(25, mean=5.5, sd=1.5)` |
| `ttest_g1_paired.csv` | Paired t-test group 1 | `rnorm(15, mean=10, sd=2)` |
| `ttest_g2_paired.csv` | Paired t-test group 2 | `g1_paired + rnorm(15, mean=0.5, sd=0.5)` |
| `yuen_g1.csv` | Yuen test group 1 | `c(g1, 50)` (with outlier) |
| `yuen_g2.csv` | Yuen test group 2 | `c(g2, -20)` (with outlier) |
| `levene_g3.csv` | Third group for Brown-Forsythe | `rnorm(22, mean=6, sd=2)` |
| `rank_simple.csv` | Simple ranking data | `c(3, 1, 4, 1, 5, 9, 2, 6)` |
| `rank_ties.csv` | Data with ties | `c(1, 2, 2, 3, 3, 3, 4)` |
| `mw_x.csv` | Mann-Whitney X (n=30) | `rnorm(30, mean=5, sd=1)` |
| `mw_y.csv` | Mann-Whitney Y (n=35) | `rnorm(35, mean=5.8, sd=1.2)` |
| `mw_x_exact.csv` | Mann-Whitney exact X (n=8) | `c(1.1, 2.3, 3.5, 4.7, 5.9, 6.2, 7.4, 8.6)` |
| `mw_y_exact.csv` | Mann-Whitney exact Y (n=9) | `c(2.2, 3.4, 4.6, 5.8, 7.0, 8.2, 9.4, 10.6, 11.8)` |
| `wsr_x.csv` | Wilcoxon signed-rank X | `rnorm(25, mean=10, sd=2)` |
| `wsr_y.csv` | Wilcoxon signed-rank Y | `wsr_x + rnorm(25, mean=0.5, sd=0.8)` |
| `wsr_x_exact.csv` | Wilcoxon SR exact X (n=10) | `seq(10, 20, length.out=10)` |
| `wsr_y_exact.csv` | Wilcoxon SR exact Y (n=10) | `wsr_x_exact - c(0.3, 0.5, ..., 2.1)` |
| `kw_a.csv` | Kruskal-Wallis group A | `rnorm(20, mean=5, sd=1)` |
| `kw_b.csv` | Kruskal-Wallis group B | `rnorm(22, mean=6, sd=1.2)` |
| `kw_c.csv` | Kruskal-Wallis group C | `rnorm(18, mean=5.5, sd=0.9)` |
| `sw_normal_small.csv` | Shapiro-Wilk normal (n=20) | `rnorm(20, mean=5, sd=2)` |
| `sw_normal_medium.csv` | Shapiro-Wilk normal (n=50) | `rnorm(50, mean=10, sd=3)` |
| `sw_normal_large.csv` | Shapiro-Wilk normal (n=100) | `rnorm(100, mean=0, sd=1)` |
| `sw_uniform.csv` | Shapiro-Wilk uniform (n=30) | `runif(30, min=0, max=10)` |
| `sw_exp.csv` | Shapiro-Wilk exponential (n=40) | `rexp(40, rate=1)` |
| `sw_n3.csv` | Shapiro-Wilk edge case (n=3) | `rnorm(3)` |
| `sw_n4.csv` | Shapiro-Wilk edge case (n=4) | `rnorm(4)` |
| `sw_n5.csv` | Shapiro-Wilk edge case (n=5) | `rnorm(5)` |
| `sw_n10.csv` | Shapiro-Wilk edge case (n=10) | `rnorm(10)` |
| `dag_normal.csv` | D'Agostino normal data | `rnorm(50, mean=10, sd=2)` |
| `dag_skewed.csv` | D'Agostino skewed data | `rexp(50, rate=0.5)` |
| `bm_x.csv` | Brunner-Munzel X | `c(1,2,1,1,1,1,1,1,1,1,2,4,1,1)` |
| `bm_y.csv` | Brunner-Munzel Y | `c(3,3,4,3,1,2,3,1,1,5,4)` |
| `dm_e1.csv` | Diebold-Mariano errors 1 | `actual - forecast1` |
| `dm_e2.csv` | Diebold-Mariano errors 2 | `actual - forecast2` |
| `spa_benchmark.csv` | SPA benchmark losses | `abs(rnorm(100, 2, 0.5))` |
| `spa_model1.csv` | SPA model 1 losses | `abs(rnorm(100, 1, 0.3))` |
| `spa_model2.csv` | SPA model 2 losses | `abs(rnorm(100, 2.1, 0.5))` |
| `spa_model3.csv` | SPA model 3 losses | `abs(rnorm(100, 3, 0.6))` |
| `mcs_model0.csv` | MCS model 0 losses | `abs(rnorm(200, 1.0, 0.3))` |
| `mcs_model1.csv` | MCS model 1 losses | `abs(rnorm(200, 1.1, 0.35))` |
| `mcs_model2.csv` | MCS model 2 losses | `abs(rnorm(200, 2.0, 0.5))` |
| `mcs_model3.csv` | MCS model 3 losses | `abs(rnorm(200, 3.5, 0.8))` |
| `ed_x.csv` | Energy distance X | `rnorm(30, mean=0, sd=1)` |
| `ed_y.csv` | Energy distance Y | `rnorm(30, mean=3, sd=1)` |
| `perm_x.csv` | Permutation test X | `rnorm(20, mean=5, sd=1)` |
| `perm_y.csv` | Permutation test Y | `rnorm(25, mean=7, sd=1.2)` |

### Expected Value Files (R Results)

| File | Test | R Function | Values Stored |
|------|------|------------|---------------|
| `math_basic.csv` | Math primitives | `mean()`, `var()`, `median()`, `mean(trim=0.2)` | mean, variance, median, trimmed mean for each test vector |
| `math_extended.csv` | Extended math | `sd()`, `e1071::skewness()`, `e1071::kurtosis()` | std_dev, skewness (type=2), kurtosis (type=2) |
| `math_skewed.csv` | Skewed data stats | `e1071::skewness()`, `e1071::kurtosis()` | skewness, kurtosis for skewed vector |
| `ttest_welch.csv` | Welch t-test | `t.test(var.equal=FALSE)` | statistic, df, p_value for all 3 alternatives |
| `ttest_student.csv` | Student t-test | `t.test(var.equal=TRUE)` | statistic, df, p_value for all 3 alternatives |
| `ttest_paired.csv` | Paired t-test | `t.test(paired=TRUE)` | statistic, df, p_value for all 3 alternatives |
| `yuen.csv` | Yuen's test | `WRS2::yuen()` | statistic, df, p_value, diff for 20% and 10% trim |
| `brown_forsythe.csv` | Brown-Forsythe | `car::leveneTest(center=median)` | F-statistic, df1, df2, p_value for 2 and 3 groups |
| `ranks.csv` | Ranking | `rank(ties.method="average")` | All rank values for simple and tied data |
| `mann_whitney.csv` | Mann-Whitney U | `wilcox.test(exact=FALSE, correct=FALSE/TRUE)` | statistic, p_value, p_value_less, p_value_greater, p_value_corrected, estimate, conf_low, conf_high |
| `mann_whitney_exact.csv` | Mann-Whitney exact | `wilcox.test(exact=TRUE, conf.int=TRUE)` | statistic, p_value, p_value_less, p_value_greater, estimate, conf_low_95, conf_high_95, conf_low_90, conf_high_90 |
| `wilcoxon_signed_rank.csv` | Wilcoxon SR | `wilcox.test(paired=TRUE, exact=FALSE, correct=FALSE/TRUE)` | statistic, p_value, p_value_less, p_value_greater, p_value_corrected, estimate, conf_low, conf_high |
| `wilcoxon_signed_rank_exact.csv` | Wilcoxon SR exact | `wilcox.test(paired=TRUE, exact=TRUE, conf.int=TRUE)` | statistic, p_value, p_value_less, p_value_greater, estimate, conf_low_95, conf_high_95, conf_low_90, conf_high_90 |
| `kruskal_wallis.csv` | Kruskal-Wallis | `kruskal.test()` | statistic, df, p_value |
| `shapiro_wilk.csv` | Shapiro-Wilk | `shapiro.test()` | W, p_value for 5 test vectors |
| `shapiro_wilk_edge.csv` | Shapiro-Wilk edge | `shapiro.test()` | W, p_value for n=3,4,5,10 |
| `dagostino.csv` | D'Agostino | `moments::agostino.test()`, `moments::anscombe.test()` | Z_skew, Z_kurt, p_values |
| `brunner_munzel.csv` | Brunner-Munzel | `lawstat::brunner.munzel.test()` | statistic, df, p_value, estimate, p_value_less, p_value_greater |
| `diebold_mariano.csv` | Diebold-Mariano | `forecast::dm.test()` | statistic, p_value for SE/AE loss, h=1/h=3, alternative=less/greater |
| `ttest_welch.csv` | Welch t-test with mu | `t.test(var.equal=FALSE, mu=0.5)` | statistic_mu, p_value_mu |
| `clark_west.csv` | Clark-West | Manual HAC computation | statistic, p_value for h=1 and h=3 |
| `spa.csv` | SPA test | Manual computation | t-statistics, means for all models |
| `mcs.csv` | Model Confidence Set | Manual computation | pairwise t-statistics, range statistic |
| `energy_distance.csv` | Energy distance | Manual computation | statistic, mean_xy, mean_xx, mean_yy |
| `permutation_t.csv` | Permutation t-test | Manual t-statistic | t_statistic, means, variances |

## Validation by Test Category

### 1. Math Primitives

**R Packages:** `base`, `e1071`

| Function | R Equivalent | Tolerance | Test Cases |
|----------|-------------|-----------|------------|
| `mean()` | `base::mean()` | 1e-12 | short, long, outlier, single |
| `variance()` | `base::var()` | 1e-12 | short, long, outlier |
| `std_dev()` | `base::sd()` | 1e-12 | short, long, outlier |
| `median()` | `base::median()` | 1e-12 | short, long, even, single |
| `trimmed_mean()` | `base::mean(trim=0.2)` | 1e-12 | short, long, outlier |
| `skewness()` | `e1071::skewness(type=2)` | 1e-12 | short, long, outlier, skewed |
| `kurtosis()` | `e1071::kurtosis(type=2)` | 1e-12 | short, long, outlier, skewed |

### 2. Parametric Tests

**R Packages:** `stats`, `WRS2`, `car`

| Function | R Equivalent | Tolerance | Test Cases |
|----------|-------------|-----------|------------|
| `t_test(..., Welch)` | `t.test(var.equal=FALSE)` | 1e-10 | two-sided, less, greater, mu=0.5 |
| `t_test(..., Student)` | `t.test(var.equal=TRUE)` | 1e-10 | two-sided, less, greater |
| `t_test(..., Paired)` | `t.test(paired=TRUE)` | 1e-10 | two-sided, less, greater |
| `yuen_test()` | `WRS2::yuen()` | 1e-10 | 20% trim, 10% trim |
| `brown_forsythe()` | `car::leveneTest(center=median)` | 1e-10 | 2 groups, 3 groups |

### 3. Nonparametric Tests

**R Packages:** `stats`, `lawstat`

| Function | R Equivalent | Tolerance | Test Cases |
|----------|-------------|-----------|------------|
| `rank()` | `base::rank(ties.method="average")` | 1e-10 | simple, with ties |
| `mann_whitney_u()` | `wilcox.test(exact=FALSE)` | 1e-6 | two-sided, less, greater, continuity correction |
| `wilcoxon_signed_rank()` | `wilcox.test(paired=TRUE, exact=FALSE)` | 1e-6 | two-sided, less, greater, continuity correction |
| `kruskal_wallis()` | `kruskal.test()` | 1e-10 | 3 groups |
| `brunner_munzel()` | `lawstat::brunner.munzel.test()` | 1e-10 | two-sided, less, greater |

### 4. Distributional Tests

**R Packages:** `stats`, `moments`

| Function | R Equivalent | Tolerance | Notes |
|----------|-------------|-----------|-------|
| `shapiro_wilk()` | `shapiro.test()` | W: 1e-3, p: 0.05 | Algorithm variations across implementations |
| `dagostino_k_squared()` | `moments::agostino.test()` + `moments::anscombe.test()` | 1e-6 | Combines skewness and kurtosis |

**Note on Shapiro-Wilk Tolerance:** The relaxed tolerances are intentional. Different implementations of the Royston AS R94 algorithm can produce slightly different results due to:
- Different polynomial approximations for small samples
- Different p-value approximation methods
- Floating-point precision in the algorithm's iterative calculations

### 5. Forecast Evaluation Tests

**R Packages:** `forecast`

| Function | R Equivalent | Tolerance | Test Cases |
|----------|-------------|-----------|------------|
| `diebold_mariano()` | `forecast::dm.test()` | 1e-6 | SE/AE loss, h=1/h=3, alternative=two-sided/less/greater |
| `clark_west()` | Manual HAC computation | 1e-6 | h=1, h=3 |
| `spa_test()` | Manual computation | 1e-6 | 3 competing models |
| `model_confidence_set()` | Manual computation | 1e-6 | 4 models |

### 6. Modern Distribution Tests

| Function | R Equivalent | Tolerance | Notes |
|----------|-------------|-----------|-------|
| `energy_distance_test()` | Manual distance calculation | 1e-6 | Permutation p-value |
| `mmd_test()` | Kernel-based computation | 1e-6 | Multiple kernel types |

## Example: Complete Validation Flow

### Welch T-Test Validation

**Step 1: R generates test data** (`R/generate_refs.R`, lines 83-84)
```r
set.seed(42)
g1 <- rnorm(20, mean = 5, sd = 1)
g2 <- rnorm(25, mean = 5.5, sd = 1.5)
```

**Step 2: R computes expected values** (`R/generate_refs.R`, lines 98-100)
```r
welch_two <- t.test(g1, g2, var.equal = FALSE, alternative = "two.sided")
welch_less <- t.test(g1, g2, var.equal = FALSE, alternative = "less")
welch_greater <- t.test(g1, g2, var.equal = FALSE, alternative = "greater")
```

**Step 3: R saves to CSV** (`R/data/ttest_welch.csv`)
```csv
"statistic_two","df_two","p_value_two","mean_x_two","mean_y_two",...
-0.0643125318393203,42.9656281823889,0.949019668509795,5.01533768967797,5.03745840727566,...
```

**Step 4: Rust test loads and validates** (`tests/tdd_parametric.rs`, lines 13-30)
```rust
#[test]
fn test_welch_two_sided() {
    let refs = common::load_reference_scalars("ttest_welch.csv");
    let g1 = common::load_reference_vector("ttest_g1.csv");
    let g2 = common::load_reference_vector("ttest_g2.csv");

    let result = t_test(&g1, &g2, TTestKind::Welch, Alternative::TwoSided, 0.0)
        .expect("t_test should succeed");

    assert_relative_eq!(result.statistic, refs["statistic_two"], epsilon = 1e-10);
    assert_relative_eq!(result.df, refs["df_two"], epsilon = 1e-10);
    assert_relative_eq!(result.p_value, refs["p_value_two"], epsilon = 1e-10);
}
```

## R Package Dependencies

The following R packages are required to regenerate reference data:

| Package | CRAN | Purpose |
|---------|------|---------|
| `stats` | Base R | Core statistical functions (t.test, wilcox.test, etc.) |
| `WRS2` | [CRAN](https://cran.r-project.org/package=WRS2) | Robust statistics (yuen) |
| `car` | [CRAN](https://cran.r-project.org/package=car) | Companion to Applied Regression (leveneTest) |
| `lawstat` | [CRAN](https://cran.r-project.org/package=lawstat) | Brunner-Munzel test |
| `e1071` | [CRAN](https://cran.r-project.org/package=e1071) | Skewness, kurtosis (type 2) |
| `moments` | [CRAN](https://cran.r-project.org/package=moments) | D'Agostino tests |
| `forecast` | [CRAN](https://cran.r-project.org/package=forecast) | Diebold-Mariano test |

Install all dependencies in R:
```r
install.packages(c("WRS2", "car", "lawstat", "e1071", "moments", "forecast"))
```

## Tolerance Rationale

| Tolerance | Used For | Rationale |
|-----------|----------|-----------|
| 1e-12 | Math primitives | Machine epsilon level; direct floating-point operations |
| 1e-10 | Most statistical tests | High precision; allows for minor floating-point accumulation |
| 1e-6 | Bootstrap/permutation tests | Randomness introduces variation; seed controls but precision varies |
| 1e-3 | Shapiro-Wilk W statistic | Known algorithm variations between implementations |
| 0.05 | Shapiro-Wilk p-value | Different p-value approximation methods produce larger variation |

## Test Coverage Summary

| Category | Tests | Test Cases | Reference Files |
|----------|-------|------------|-----------------|
| Math Primitives | 7 | 41 | 8 |
| Parametric | 5 | 22 | 13 |
| Nonparametric | 5 | 23 | 14 |
| Distributional | 2 | 17 | 12 |
| Resampling | 3 | 13 | 4 |
| Modern | 2 | 8 | 3 |
| Forecast | 5 | 21 | 22 |
| **Total** | **29** | **145** | **76** |

## Reproducibility

All reference data is reproducible:

1. **Fixed seed:** `set.seed(42)` in R ensures identical random data
2. **Version controlled:** CSV files are committed to the repository
3. **CI/CD verified:** GitHub Actions runs `cargo test` against these references
4. **Transparent:** All R code and CSV data are publicly available

## Verifying Your Installation

```bash
# 1. Clone repository
git clone https://github.com/sipemu/anofox-statistics-rs.git
cd anofox-statistics-rs

# 2. (Optional) Regenerate R references
Rscript R/generate_refs.R

# 3. Run validation tests
cargo test

# All 136 tests should pass (60 unit + 76 TDD integration)
```
