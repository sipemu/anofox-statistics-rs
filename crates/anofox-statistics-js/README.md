# anofox-statistics

A comprehensive statistical hypothesis testing library compiled to WebAssembly for JavaScript/TypeScript applications. All tests are validated against R.

## Installation

```bash
npm install anofox-statistics
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
} from 'anofox-statistics';

// Initialize the WASM module
await init();

// Two-sample t-test
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
const { tTest, JsTTestKind, JsAlternative } = require('anofox-statistics');

// Use directly (WASM auto-initializes)
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
| `brownForsythe` | Brown-Forsythe test for homogeneity of variances |

### Nonparametric Tests

| Function | Description |
|----------|-------------|
| `mannWhitneyU` | Mann-Whitney U test (Wilcoxon rank-sum) |
| `wilcoxonSignedRank` | Wilcoxon signed-rank test |
| `kruskalWallis` | Kruskal-Wallis H test |
| `brunnerMunzel` | Brunner-Munzel test |

### Distributional Tests

| Function | Description |
|----------|-------------|
| `shapiroWilk` | Shapiro-Wilk normality test |
| `kolmogorovSmirnov` | Kolmogorov-Smirnov test |
| `andersonDarling` | Anderson-Darling normality test |
| `jarqueBera` | Jarque-Bera normality test |

### Categorical Tests

| Function | Description |
|----------|-------------|
| `chiSquaredTest` | Chi-squared test for independence |
| `fisherExact` | Fisher's exact test |
| `binomialTest` | Exact binomial test |
| `mcnemarTest` | McNemar's test for paired proportions |

### Correlation Tests

| Function | Description |
|----------|-------------|
| `pearsonCorrelation` | Pearson correlation with significance |
| `spearmanCorrelation` | Spearman rank correlation |
| `kendallCorrelation` | Kendall's tau correlation |
| `icc` | Intraclass correlation coefficient |

### Equivalence Tests (TOST)

| Function | Description |
|----------|-------------|
| `tostTwoSample` | TOST for two independent samples |
| `tostPaired` | TOST for paired samples |
| `tostCorrelation` | TOST for correlation equivalence |
| `tostBootstrap` | Bootstrap TOST |

### Modern/Kernel Tests

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

### Resampling Tests

| Function | Description |
|----------|-------------|
| `permutationTTest` | Permutation t-test |
| `permutationCorrelation` | Permutation correlation test |
| `bootstrapCI` | Bootstrap confidence intervals |

## Enums

```typescript
// T-test variants
enum JsTTestKind { Welch, Student, Paired }

// Alternative hypotheses
enum JsAlternative { TwoSided, Less, Greater }

// ANOVA variants
enum JsAnovaKind { Fisher, Welch }

// Kernel types for MMD
enum JsKernel { Gaussian, Laplacian, Linear }

// Loss functions for forecast comparison
enum JsLossFunction { SquaredError, AbsoluteError }

// Variance estimators
enum JsVarEstimator { Acf, Bartlett }

// MCS statistics
enum JsMCSStatistic { Max, Range }
```

## Examples

### Normality Testing

```javascript
import init, { shapiroWilk, andersonDarling } from 'anofox-statistics';

await init();

const data = new Float64Array([2.3, 3.1, 2.8, 3.5, 2.9, 3.2, 2.7]);

const sw = shapiroWilk(data);
console.log(`Shapiro-Wilk: W=${sw.statistic.toFixed(4)}, p=${sw.p_value.toFixed(4)}`);

const ad = andersonDarling(data);
console.log(`Anderson-Darling: A=${ad.statistic.toFixed(4)}, p=${ad.p_value.toFixed(4)}`);
```

### ANOVA with Post-hoc

```javascript
import init, { oneWayAnova, JsAnovaKind } from 'anofox-statistics';

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

### Equivalence Testing (TOST)

```javascript
import init, { tostTwoSample, JsAlternative } from 'anofox-statistics';

await init();

const treatment = new Float64Array([10.2, 11.1, 9.8, 10.5, 10.9]);
const control = new Float64Array([10.0, 10.8, 9.9, 10.3, 10.7]);

// Test equivalence within bounds of -1 to +1
const result = tostTwoSample(treatment, control, -1.0, 1.0, 0.95);
console.log(`Equivalent: ${result.equivalent}`);
console.log(`TOST p-value: ${result.p_value.toFixed(4)}`);
```

### Forecast Model Comparison

```javascript
import init, {
  dieboldMariano,
  JsLossFunction,
  JsAlternative,
  JsVarEstimator
} from 'anofox-statistics';

await init();

const errors1 = new Float64Array([0.5, -0.3, 0.2, -0.1, 0.4]);
const errors2 = new Float64Array([0.3, -0.5, 0.1, 0.2, -0.2]);

const dm = dieboldMariano(
  errors1,
  errors2,
  JsLossFunction.SquaredError,
  1,  // horizon
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
} from 'anofox-statistics';

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

All statistical tests are validated against R's implementation using extensive test suites. Results match R's output within numerical precision (typically 10^-10).

## License

MIT License - see [LICENSE](LICENSE) for details.

## Links

- [GitHub Repository](https://github.com/sipemu/anofox-statistics-rs)
- [Rust crate on crates.io](https://crates.io/crates/anofox-statistics)
- [API Documentation](https://docs.rs/anofox-statistics)
