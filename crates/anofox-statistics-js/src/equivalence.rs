use anofox_statistics::{
    tost_bootstrap, tost_correlation, tost_prop_one, tost_prop_two, tost_t_test_one_sample,
    tost_t_test_paired, tost_t_test_two_sample, tost_wilcoxon_paired, tost_wilcoxon_two_sample,
    tost_yuen, CorrelationTostMethod, EquivalenceBounds,
};
use serde::Serialize;
use wasm_bindgen::prelude::*;

#[derive(Serialize)]
struct TostResultJs {
    estimate: f64,
    ci: (f64, f64),
    bounds: (f64, f64),
    lower_test: OneSidedTestJs,
    upper_test: OneSidedTestJs,
    tost_p_value: f64,
    equivalent: bool,
    alpha: f64,
    n: usize,
    df: Option<f64>,
    method: String,
}

#[derive(Serialize)]
struct OneSidedTestJs {
    hypothesis: String,
    statistic: f64,
    p_value: f64,
    rejected: bool,
}

fn convert_tost_result(result: anofox_statistics::TostResult) -> TostResultJs {
    TostResultJs {
        estimate: result.estimate,
        ci: result.ci,
        bounds: result.bounds,
        lower_test: OneSidedTestJs {
            hypothesis: result.lower_test.hypothesis,
            statistic: result.lower_test.statistic,
            p_value: result.lower_test.p_value,
            rejected: result.lower_test.rejected,
        },
        upper_test: OneSidedTestJs {
            hypothesis: result.upper_test.hypothesis,
            statistic: result.upper_test.statistic,
            p_value: result.upper_test.p_value,
            rejected: result.upper_test.rejected,
        },
        tost_p_value: result.tost_p_value,
        equivalent: result.equivalent,
        alpha: result.alpha,
        n: result.n,
        df: result.df,
        method: result.method,
    }
}

/// TOST t-test for one sample.
///
/// @param x - Sample data as Float64Array
/// @param mu - Hypothesized mean
/// @param lowerBound - Lower equivalence bound (e.g., -0.5)
/// @param upperBound - Upper equivalence bound (e.g., 0.5)
/// @param alpha - Significance level (default: 0.05)
/// @returns TOST result object
#[wasm_bindgen(js_name = tostTTestOneSample)]
pub fn js_tost_t_test_one_sample(
    x: &[f64],
    mu: f64,
    lower_bound: f64,
    upper_bound: f64,
    alpha: Option<f64>,
) -> Result<JsValue, JsError> {
    let bounds = EquivalenceBounds::raw(lower_bound, upper_bound)
        .map_err(|e| JsError::new(&e.to_string()))?;

    let result = tost_t_test_one_sample(x, mu, &bounds, alpha.unwrap_or(0.05))
        .map_err(|e| JsError::new(&e.to_string()))?;

    serde_wasm_bindgen::to_value(&convert_tost_result(result))
        .map_err(|e| JsError::new(&e.to_string()))
}

/// TOST t-test for two independent samples.
///
/// @param x - First sample as Float64Array
/// @param y - Second sample as Float64Array
/// @param lowerBound - Lower equivalence bound
/// @param upperBound - Upper equivalence bound
/// @param alpha - Significance level (default: 0.05)
/// @param pooled - Use pooled variance (default: false for Welch)
/// @returns TOST result object
#[wasm_bindgen(js_name = tostTTestTwoSample)]
pub fn js_tost_t_test_two_sample(
    x: &[f64],
    y: &[f64],
    lower_bound: f64,
    upper_bound: f64,
    alpha: Option<f64>,
    pooled: Option<bool>,
) -> Result<JsValue, JsError> {
    let bounds = EquivalenceBounds::raw(lower_bound, upper_bound)
        .map_err(|e| JsError::new(&e.to_string()))?;

    let result = tost_t_test_two_sample(
        x,
        y,
        &bounds,
        alpha.unwrap_or(0.05),
        pooled.unwrap_or(false),
    )
    .map_err(|e| JsError::new(&e.to_string()))?;

    serde_wasm_bindgen::to_value(&convert_tost_result(result))
        .map_err(|e| JsError::new(&e.to_string()))
}

/// TOST t-test for paired samples.
///
/// @param x - First sample as Float64Array
/// @param y - Second sample as Float64Array
/// @param lowerBound - Lower equivalence bound
/// @param upperBound - Upper equivalence bound
/// @param alpha - Significance level (default: 0.05)
/// @returns TOST result object
#[wasm_bindgen(js_name = tostTTestPaired)]
pub fn js_tost_t_test_paired(
    x: &[f64],
    y: &[f64],
    lower_bound: f64,
    upper_bound: f64,
    alpha: Option<f64>,
) -> Result<JsValue, JsError> {
    let bounds = EquivalenceBounds::raw(lower_bound, upper_bound)
        .map_err(|e| JsError::new(&e.to_string()))?;

    let result = tost_t_test_paired(x, y, &bounds, alpha.unwrap_or(0.05))
        .map_err(|e| JsError::new(&e.to_string()))?;

    serde_wasm_bindgen::to_value(&convert_tost_result(result))
        .map_err(|e| JsError::new(&e.to_string()))
}

/// Correlation TOST method for JavaScript
#[wasm_bindgen]
#[derive(Clone, Copy)]
pub enum JsCorrelationTostMethod {
    Pearson,
    Spearman,
}

impl From<JsCorrelationTostMethod> for CorrelationTostMethod {
    fn from(m: JsCorrelationTostMethod) -> Self {
        match m {
            JsCorrelationTostMethod::Pearson => CorrelationTostMethod::Pearson,
            JsCorrelationTostMethod::Spearman => CorrelationTostMethod::Spearman,
        }
    }
}

/// TOST for correlation equivalence.
///
/// @param x - First variable as Float64Array
/// @param y - Second variable as Float64Array
/// @param rhoNull - Null value for correlation (usually 0)
/// @param lowerBound - Lower equivalence bound for correlation (e.g., -0.1)
/// @param upperBound - Upper equivalence bound for correlation (e.g., 0.1)
/// @param method - Correlation method (Pearson or Spearman)
/// @param alpha - Significance level (default: 0.05)
/// @returns TOST result object
#[wasm_bindgen(js_name = tostCorrelation)]
pub fn js_tost_correlation(
    x: &[f64],
    y: &[f64],
    rho_null: f64,
    lower_bound: f64,
    upper_bound: f64,
    method: JsCorrelationTostMethod,
    alpha: Option<f64>,
) -> Result<JsValue, JsError> {
    let bounds = EquivalenceBounds::raw(lower_bound, upper_bound)
        .map_err(|e| JsError::new(&e.to_string()))?;

    let result = tost_correlation(
        x,
        y,
        rho_null,
        &bounds,
        alpha.unwrap_or(0.05),
        method.into(),
    )
    .map_err(|e| JsError::new(&e.to_string()))?;

    serde_wasm_bindgen::to_value(&convert_tost_result(result))
        .map_err(|e| JsError::new(&e.to_string()))
}

/// TOST for one-sample proportion.
///
/// @param successes - Number of successes
/// @param trials - Number of trials
/// @param p0 - Hypothesized proportion
/// @param lowerBound - Lower equivalence bound
/// @param upperBound - Upper equivalence bound
/// @param alpha - Significance level (default: 0.05)
/// @returns TOST result object
#[wasm_bindgen(js_name = tostPropOne)]
pub fn js_tost_prop_one(
    successes: usize,
    trials: usize,
    p0: f64,
    lower_bound: f64,
    upper_bound: f64,
    alpha: Option<f64>,
) -> Result<JsValue, JsError> {
    let bounds = EquivalenceBounds::raw(lower_bound, upper_bound)
        .map_err(|e| JsError::new(&e.to_string()))?;

    let result = tost_prop_one(successes, trials, p0, &bounds, alpha.unwrap_or(0.05))
        .map_err(|e| JsError::new(&e.to_string()))?;

    serde_wasm_bindgen::to_value(&convert_tost_result(result))
        .map_err(|e| JsError::new(&e.to_string()))
}

/// TOST for two-sample proportions.
///
/// @param successes1 - Successes in group 1
/// @param trials1 - Trials in group 1
/// @param successes2 - Successes in group 2
/// @param trials2 - Trials in group 2
/// @param lowerBound - Lower equivalence bound
/// @param upperBound - Upper equivalence bound
/// @param alpha - Significance level (default: 0.05)
/// @returns TOST result object
#[wasm_bindgen(js_name = tostPropTwo)]
pub fn js_tost_prop_two(
    successes1: usize,
    trials1: usize,
    successes2: usize,
    trials2: usize,
    lower_bound: f64,
    upper_bound: f64,
    alpha: Option<f64>,
) -> Result<JsValue, JsError> {
    let bounds = EquivalenceBounds::raw(lower_bound, upper_bound)
        .map_err(|e| JsError::new(&e.to_string()))?;

    let result = tost_prop_two(
        successes1,
        trials1,
        successes2,
        trials2,
        &bounds,
        alpha.unwrap_or(0.05),
    )
    .map_err(|e| JsError::new(&e.to_string()))?;

    serde_wasm_bindgen::to_value(&convert_tost_result(result))
        .map_err(|e| JsError::new(&e.to_string()))
}

/// TOST Wilcoxon test for two independent samples.
///
/// @param x - First sample as Float64Array
/// @param y - Second sample as Float64Array
/// @param lowerBound - Lower equivalence bound
/// @param upperBound - Upper equivalence bound
/// @param alpha - Significance level (default: 0.05)
/// @returns TOST result object
#[wasm_bindgen(js_name = tostWilcoxonTwoSample)]
pub fn js_tost_wilcoxon_two_sample(
    x: &[f64],
    y: &[f64],
    lower_bound: f64,
    upper_bound: f64,
    alpha: Option<f64>,
) -> Result<JsValue, JsError> {
    let bounds = EquivalenceBounds::raw(lower_bound, upper_bound)
        .map_err(|e| JsError::new(&e.to_string()))?;

    let result = tost_wilcoxon_two_sample(x, y, &bounds, alpha.unwrap_or(0.05))
        .map_err(|e| JsError::new(&e.to_string()))?;

    serde_wasm_bindgen::to_value(&convert_tost_result(result))
        .map_err(|e| JsError::new(&e.to_string()))
}

/// TOST Wilcoxon test for paired samples.
///
/// @param x - First sample as Float64Array
/// @param y - Second sample as Float64Array
/// @param lowerBound - Lower equivalence bound
/// @param upperBound - Upper equivalence bound
/// @param alpha - Significance level (default: 0.05)
/// @returns TOST result object
#[wasm_bindgen(js_name = tostWilcoxonPaired)]
pub fn js_tost_wilcoxon_paired(
    x: &[f64],
    y: &[f64],
    lower_bound: f64,
    upper_bound: f64,
    alpha: Option<f64>,
) -> Result<JsValue, JsError> {
    let bounds = EquivalenceBounds::raw(lower_bound, upper_bound)
        .map_err(|e| JsError::new(&e.to_string()))?;

    let result = tost_wilcoxon_paired(x, y, &bounds, alpha.unwrap_or(0.05))
        .map_err(|e| JsError::new(&e.to_string()))?;

    serde_wasm_bindgen::to_value(&convert_tost_result(result))
        .map_err(|e| JsError::new(&e.to_string()))
}

/// TOST Yuen test using trimmed means.
///
/// @param x - First sample as Float64Array
/// @param y - Second sample as Float64Array
/// @param lowerBound - Lower equivalence bound
/// @param upperBound - Upper equivalence bound
/// @param trim - Trim proportion (default: 0.2)
/// @param alpha - Significance level (default: 0.05)
/// @returns TOST result object
#[wasm_bindgen(js_name = tostYuen)]
pub fn js_tost_yuen(
    x: &[f64],
    y: &[f64],
    lower_bound: f64,
    upper_bound: f64,
    trim: Option<f64>,
    alpha: Option<f64>,
) -> Result<JsValue, JsError> {
    let bounds = EquivalenceBounds::raw(lower_bound, upper_bound)
        .map_err(|e| JsError::new(&e.to_string()))?;

    let result = tost_yuen(x, y, &bounds, trim.unwrap_or(0.2), alpha.unwrap_or(0.05))
        .map_err(|e| JsError::new(&e.to_string()))?;

    serde_wasm_bindgen::to_value(&convert_tost_result(result))
        .map_err(|e| JsError::new(&e.to_string()))
}

/// TOST using bootstrap resampling.
///
/// @param x - First sample as Float64Array
/// @param y - Second sample as Float64Array
/// @param lowerBound - Lower equivalence bound
/// @param upperBound - Upper equivalence bound
/// @param alpha - Significance level (default: 0.05)
/// @param nBootstrap - Number of bootstrap samples (default: 1000)
/// @param seed - Random seed (optional)
/// @returns TOST result object
#[wasm_bindgen(js_name = tostBootstrap)]
pub fn js_tost_bootstrap(
    x: &[f64],
    y: &[f64],
    lower_bound: f64,
    upper_bound: f64,
    alpha: Option<f64>,
    n_bootstrap: Option<usize>,
    seed: Option<u64>,
) -> Result<JsValue, JsError> {
    let bounds = EquivalenceBounds::raw(lower_bound, upper_bound)
        .map_err(|e| JsError::new(&e.to_string()))?;

    let result = tost_bootstrap(
        x,
        y,
        &bounds,
        alpha.unwrap_or(0.05),
        n_bootstrap.unwrap_or(1000),
        seed,
    )
    .map_err(|e| JsError::new(&e.to_string()))?;

    serde_wasm_bindgen::to_value(&convert_tost_result(result))
        .map_err(|e| JsError::new(&e.to_string()))
}
