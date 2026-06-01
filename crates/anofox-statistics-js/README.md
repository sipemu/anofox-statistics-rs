# @sipemu/anofox-statistics

A comprehensive statistical hypothesis testing library compiled to WebAssembly for JavaScript/TypeScript applications. All tests are validated against R.

## Installation

```bash
npm install @sipemu/anofox-statistics
```

## Usage

### Web/Browser

```javascript
import init, {
  tTest,
  shapiroWilk,
  mannWhitneyU,
  oneWayAnova,
  JsTTestKind,
  JsAlternative,
  JsAnovaKind
} from '@sipemu/anofox-statistics';

// Initialize the WASM module
await init();

// Two-sample Welch's t-test
const result = tTest(
  new Float64Array([1.2, 2.3, 3.1, 4.5, 5.2]),
  new Float64Array([2.1, 3.4, 4.2, 5.6, 6.1]),
  JsTTestKind.Welch,
  JsAlternative.TwoSided,
  0.0,  // null hypothesis value
  0.95  // confidence level
);

console.log(result);
// { statistic: -2.34, df: 7.89, p_value: 0.047, mean_x: 3.26, mean_y: 4.28, ... }
```

### Node.js

```javascript
const { tTest, JsTTestKind, JsAlternative } = require('@sipemu/anofox-statistics');

const result = tTest(
  new Float64Array([1.2, 2.3, 3.1]),
  new Float64Array([2.1, 3.4, 4.2]),
  JsTTestKind.Welch,
  JsAlternative.TwoSided
);
```

## Available Tests

### Parametric Tests

| Function | Description |
|----------|-------------|
| `tTest` | Student's, Welch's, or paired t-test |
| `yuenTest` | Yuen's test for trimmed means (robust) |
| `oneWayAnova` | One-way ANOVA (Fisher or Welch) |
| `twoWayAnova` | Two-way ANOVA with interaction |
| `repeatedMeasuresAnova` | One-way RM-ANOVA with sphericity test and GG/HF corrections |
| `brownForsythe` | Brown-Forsythe test for homogeneity of variances |

### Nonparametric Tests

| Function | Description |
|----------|-------------|
| `mannWhitneyU` | Mann-Whitney U test (Wilcoxon rank-sum) |
| `wilcoxonSignedRank` | Wilcoxon signed-rank test |
| `kruskalWallis` | Kruskal-Wallis H test |
| `brunnerMunzel` | Brunner-Munzel test |
| `rank` | Rank a sample, with average rank assigned to ties |

### Distributional Tests

| Function | Description |
|----------|-------------|
| `shapiroWilk` | Shapiro-Wilk normality test |
| `dagostinoKSquared` | D'Agostino's K² omnibus normality test |

### Categorical Tests

| Function | Description |
|----------|-------------|
| `chiSquareTest` | Chi-square test of independence (with optional Yates correction) |
| `chiSquareGoodnessOfFit` | Chi-square goodness-of-fit test |
| `gTest` | G-test (likelihood-ratio chi-square) |
| `fisherExact` | Fisher's exact test (2×2 tables) |
| `binomTest` | Exact binomial test |
| `mcnemarTest` | McNemar's test for paired proportions |
| `mcnemarExact` | Exact McNemar test for small samples |
| `propTestOne` | One-sample proportion test |
| `propTestTwo` | Two-sample proportion test |
| `cohenKappa` | Cohen's kappa for inter-rater agreement |
| `cramersV` | Cramér's V effect size |
| `phiCoefficient` | Phi coefficient (2×2 tables) |
| `contingencyCoefficient` | Pearson's contingency coefficient |

### Correlation Tests

| Function | Description |
|----------|-------------|
| `pearsonCorrelation` | Pearson product-moment correlation |
| `spearmanCorrelation` | Spearman rank correlation |
| `kendallCorrelation` | Kendall's tau correlation |
| `partialCorrelation` | Partial correlation (controlling for covariates) |
| `semiPartialCorrelation` | Semi-partial (part) correlation |
| `intraclassCorrelation` | Intraclass correlation coefficient |
| `distanceCorrelation` | Distance correlation (Székely-Rizzo) |
| `distanceCorrelationTest` | Distance correlation independence test |

### Equivalence Tests (TOST)

| Function | Description |
|----------|-------------|
| `tostTTestOneSample` | One-sample TOST |
| `tostTTestTwoSample` | Two-sample TOST (Welch / Student) |
| `tostTTestPaired` | Paired TOST |
| `tostCorrelation` | Correlation equivalence (Pearson / Spearman) |
| `tostPropOne` | One-proportion TOST |
| `tostPropTwo` | Two-proportion TOST |
| `tostWilcoxonPaired` | Paired Wilcoxon TOST |
| `tostWilcoxonTwoSample` | Two-sample Wilcoxon (Mann-Whitney) TOST |
| `tostYuen` | Trimmed-mean (Yuen) TOST |
| `tostBootstrap` | Bootstrap TOST with percentile CI |

### Modern / Kernel Tests

| Function | Description |
|----------|-------------|
| `energyDistanceTest1d` | Energy distance test (1D) |
| `energyDistanceTest` | Energy distance test (multivariate) |
| `mmdTest1d` | Maximum Mean Discrepancy test (1D) |
| `mmdTest` | MMD test (multivariate) |

### Forecast Comparison Tests

| Function | Description |
|----------|-------------|
| `dieboldMariano` | Diebold-Mariano test |
| `clarkWest` | Clark-West test for nested models |
| `spaTest` | Superior Predictive Ability test |
| `mspeAdjustedSpa` | MSPE-adjusted SPA test |
| `modelConfidenceSet` | Model Confidence Set procedure |

### Resampling

| Function | Description |
|----------|-------------|
| `permutationTTest` | Permutation t-test |

## Enums

```typescript
// T-test variants
enum JsTTestKind { Welch, Student, Paired }

// Alternative hypotheses
enum JsAlternative { TwoSided, Less, Greater }

// ANOVA variants
enum JsAnovaKind { Fisher, Welch }

// Kendall correlation variants
enum JsKendallVariant { TauA, TauB, TauC }

// Intraclass correlation forms
enum JsICCType { ICC1, ICC2, ICC3, ICC1k, ICC2k, ICC3k }

// Method for correlation TOST
enum JsCorrelationTostMethod { Pearson, Spearman }

// Kernel types for MMD
enum JsKernel { Gaussian, Laplacian, Linear }

// Loss functions for forecast comparison
enum JsLossFunction { SquaredError, AbsoluteError }

// Variance estimators (Diebold-Mariano)
enum JsVarEstimator { Acf, Bartlett }

// MCS statistics
enum JsMCSStatistic { Max, Range }
```

## Examples

### Normality Testing

```javascript
import init, { shapiroWilk, dagostinoKSquared } from '@sipemu/anofox-statistics';

await init();

const data = new Float64Array([2.3, 3.1, 2.8, 3.5, 2.9, 3.2, 2.7]);

const sw = shapiroWilk(data);
console.log(`Shapiro-Wilk: W=${sw.statistic.toFixed(4)}, p=${sw.p_value.toFixed(4)}`);

const dk = dagostinoKSquared(data);
console.log(`D'Agostino K²: K²=${dk.statistic.toFixed(4)}, p=${dk.p_value.toFixed(4)}`);
```

### One-Way ANOVA

```javascript
import init, { oneWayAnova, JsAnovaKind } from '@sipemu/anofox-statistics';

await init();

const groups = [
  new Float64Array([23, 25, 28, 31, 27]),
  new Float64Array([31, 33, 35, 37, 34]),
  new Float64Array([41, 43, 45, 47, 44])
];

const result = oneWayAnova(groups, JsAnovaKind.Fisher);
console.log(`F(${result.df_between}, ${result.df_within}) = ${result.statistic.toFixed(2)}`);
console.log(`p-value: ${result.p_value.toFixed(6)}`);
```

### Two-Way ANOVA

```javascript
import init, { twoWayAnova } from '@sipemu/anofox-statistics';

await init();

// 2 (factor A) × 3 (factor B) design, 2 reps per cell
const values   = new Float64Array([10, 12,  20, 22,  30, 32,  15, 17,  25, 27,  35, 37]);
const factorA  = new Uint32Array([   0,  0,   0,  0,   0,  0,   1,  1,   1,  1,   1,  1]);
const factorB  = new Uint32Array([   0,  0,   1,  1,   2,  2,   0,  0,   1,  1,   2,  2]);

const result = twoWayAnova(values, factorA, factorB);
console.log(`Factor A: F=${result.factor_a.f_statistic.toFixed(2)}, p=${result.factor_a.p_value.toFixed(4)}`);
console.log(`Factor B: F=${result.factor_b.f_statistic.toFixed(2)}, p=${result.factor_b.p_value.toFixed(4)}`);
console.log(`A×B:      F=${result.interaction.f_statistic.toFixed(2)}, p=${result.interaction.p_value.toFixed(4)}`);
```

### Repeated-Measures ANOVA

```javascript
import init, { repeatedMeasuresAnova } from '@sipemu/anofox-statistics';

await init();

// Rows = subjects, columns = conditions
const data = [
  new Float64Array([10, 12, 15]),
  new Float64Array([11, 14, 17]),
  new Float64Array([ 9, 13, 16]),
  new Float64Array([12, 15, 18]),
];

const result = repeatedMeasuresAnova(data, /* computeSphericity */ true);
console.log(`Within-subjects: F=${result.within_subjects.f_statistic.toFixed(2)}, p=${result.within_subjects.p_value.toFixed(4)}`);
if (result.sphericity) {
  console.log(`Mauchly W=${result.sphericity.w.toFixed(4)}, p=${result.sphericity.p_value.toFixed(4)}`);
  console.log(`Greenhouse-Geisser p=${result.greenhouse_geisser.p_value.toFixed(4)}`);
}
```

### Equivalence Testing (TOST)

```javascript
import init, { tostTTestTwoSample, JsTTestKind } from '@sipemu/anofox-statistics';

await init();

const treatment = new Float64Array([10.2, 11.1, 9.8, 10.5, 10.9]);
const control   = new Float64Array([10.0, 10.8, 9.9, 10.3, 10.7]);

// Test equivalence within ±1.0 of zero difference
const result = tostTTestTwoSample(treatment, control, -1.0, 1.0, JsTTestKind.Welch, 0.05);
console.log(`Equivalent: ${result.equivalent}`);
console.log(`TOST p-value: ${result.tost_p_value.toFixed(4)}`);
```

### Forecast Model Comparison

```javascript
import init, {
  dieboldMariano,
  JsLossFunction,
  JsAlternative,
  JsVarEstimator
} from '@sipemu/anofox-statistics';

await init();

const errors1 = new Float64Array([0.5, -0.3, 0.2, -0.1, 0.4]);
const errors2 = new Float64Array([0.3, -0.5, 0.1, 0.2, -0.2]);

const dm = dieboldMariano(
  errors1,
  errors2,
  JsLossFunction.SquaredError,
  1,                       // horizon
  JsAlternative.TwoSided,
  JsVarEstimator.Bartlett
);
console.log(`DM statistic: ${dm.statistic.toFixed(4)}, p-value: ${dm.p_value.toFixed(4)}`);
```

## TypeScript Support

Full TypeScript definitions are included. All functions and enums are properly typed:

```typescript
import init, {
  tTest,
  JsTTestKind,
  JsAlternative
} from '@sipemu/anofox-statistics';

await init();

const result = tTest(
  new Float64Array([1, 2, 3]),
  new Float64Array([4, 5, 6]),
  JsTTestKind.Welch,
  JsAlternative.TwoSided,
  0.0,
  0.95
);

// TypeScript knows result has: statistic, df, p_value, mean_x, mean_y, conf_int, etc.
console.log(result.p_value);
```

## Validation

All statistical tests are validated against R's implementation using extensive test suites. Results match R's output within numerical precision (typically 10⁻¹⁰).

## Changelog

### 0.4.2

- **Fixed:** p-values smaller than ~1.4×10⁻¹⁴ no longer underflow to `0.0`. The
  underlying Rust library now uses each distribution's survival function (`sf`)
  instead of `1 - cdf(x)` everywhere a tail probability is computed. Affects all
  tests that use a normal, Student's t, chi-square, or F approximation
  (Mann-Whitney, Wilcoxon, t/Welch/paired, Yuen, every ANOVA variant, Levene,
  Brunner-Munzel, Kruskal-Wallis, chi-square, McNemar, Shapiro-Wilk, D'Agostino,
  Pearson/Spearman/Kendall/partial correlation, ICC, proportion tests, Cohen's
  kappa, Diebold-Mariano, Clark-West, and all TOST variants).
- **Added:** `rank`, `twoWayAnova`, `repeatedMeasuresAnova` bindings — the JS
  package now exposes the same statistical surface as the Rust crate.

## License

MIT License — see [LICENSE](LICENSE) for details.

## Links

- [GitHub Repository](https://github.com/sipemu/anofox-statistics-rs)
- [Rust crate on crates.io](https://crates.io/crates/anofox-statistics)
- [API Documentation](https://docs.rs/anofox-statistics)
