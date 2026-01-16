use anofox_statistics::{
    clark_west, diebold_mariano, model_confidence_set, mspe_adjusted_spa, spa_test, LossFunction,
    MCSStatistic, VarEstimator,
};
use serde::Serialize;
use wasm_bindgen::prelude::*;

use crate::parametric::JsAlternative;
use anofox_statistics::Alternative;

/// Loss function for forecast comparison
#[wasm_bindgen]
#[derive(Clone, Copy)]
pub enum JsLossFunction {
    SquaredError,
    AbsoluteError,
}

impl From<JsLossFunction> for LossFunction {
    fn from(lf: JsLossFunction) -> Self {
        match lf {
            JsLossFunction::SquaredError => LossFunction::SquaredError,
            JsLossFunction::AbsoluteError => LossFunction::AbsoluteError,
        }
    }
}

/// Variance estimator for forecast tests
#[wasm_bindgen]
#[derive(Clone, Copy)]
pub enum JsVarEstimator {
    Acf,
    Bartlett,
}

impl From<JsVarEstimator> for VarEstimator {
    fn from(ve: JsVarEstimator) -> Self {
        match ve {
            JsVarEstimator::Acf => VarEstimator::Acf,
            JsVarEstimator::Bartlett => VarEstimator::Bartlett,
        }
    }
}

#[derive(Serialize)]
struct DMResultJs {
    statistic: f64,
    p_value: f64,
    horizon: usize,
}

/// Diebold-Mariano test for comparing forecast accuracy.
///
/// @param e1 - Forecast errors from model 1 as Float64Array
/// @param e2 - Forecast errors from model 2 as Float64Array
/// @param lossFunction - Loss function (SquaredError or AbsoluteError)
/// @param horizon - Forecast horizon (h=1 for 1-step ahead)
/// @param alternative - Alternative hypothesis direction
/// @param varEstimator - Variance estimator (Acf or Bartlett)
/// @returns Object with statistic, p_value, horizon
#[wasm_bindgen(js_name = dieboldMariano)]
pub fn js_diebold_mariano(
    e1: &[f64],
    e2: &[f64],
    loss_function: JsLossFunction,
    horizon: usize,
    alternative: JsAlternative,
    var_estimator: JsVarEstimator,
) -> Result<JsValue, JsError> {
    let alt: Alternative = match alternative {
        JsAlternative::TwoSided => Alternative::TwoSided,
        JsAlternative::Less => Alternative::Less,
        JsAlternative::Greater => Alternative::Greater,
    };

    let result = diebold_mariano(
        e1,
        e2,
        loss_function.into(),
        horizon,
        alt,
        var_estimator.into(),
    )
    .map_err(|e| JsError::new(&e.to_string()))?;

    let js_result = DMResultJs {
        statistic: result.statistic,
        p_value: result.p_value,
        horizon: result.horizon,
    };

    serde_wasm_bindgen::to_value(&js_result).map_err(|e| JsError::new(&e.to_string()))
}

#[derive(Serialize)]
struct CWResultJs {
    statistic: f64,
    p_value: f64,
    p_value_two_sided: f64,
}

/// Clark-West test for comparing nested forecast models.
///
/// @param e1 - Forecast errors from restricted model as Float64Array
/// @param e2 - Forecast errors from unrestricted model as Float64Array
/// @param horizon - Forecast horizon for HAC variance adjustment
/// @returns Object with statistic, p_value, p_value_two_sided
#[wasm_bindgen(js_name = clarkWest)]
pub fn js_clark_west(e1: &[f64], e2: &[f64], horizon: usize) -> Result<JsValue, JsError> {
    let result = clark_west(e1, e2, horizon).map_err(|e| JsError::new(&e.to_string()))?;

    let js_result = CWResultJs {
        statistic: result.statistic,
        p_value: result.p_value,
        p_value_two_sided: result.p_value_two_sided,
    };

    serde_wasm_bindgen::to_value(&js_result).map_err(|e| JsError::new(&e.to_string()))
}

#[derive(Serialize)]
struct SPAResultJs {
    statistic: f64,
    p_value_consistent: f64,
    p_value_upper: f64,
    n_bootstrap: usize,
    best_model_idx: Option<usize>,
}

/// Superior Predictive Ability (SPA) test.
///
/// Tests if a benchmark forecast is not inferior to a set of alternative forecasts.
///
/// @param benchmarkLosses - Loss values from benchmark model as Float64Array
/// @param modelLosses - Array of loss value arrays from competing models
/// @param nBootstrap - Number of bootstrap samples (default: 1000)
/// @param blockLength - Expected block length for stationary bootstrap
/// @param seed - Random seed (optional)
/// @returns Object with statistic, p_values, best_model_idx
#[wasm_bindgen(js_name = spaTest)]
pub fn js_spa_test(
    benchmark_losses: &[f64],
    model_losses: js_sys::Array,
    n_bootstrap: Option<usize>,
    block_length: f64,
    seed: Option<u64>,
) -> Result<JsValue, JsError> {
    let models_vec: Result<Vec<Vec<f64>>, _> = model_losses
        .iter()
        .map(|row| {
            let arr = js_sys::Float64Array::new(&row);
            Ok(arr.to_vec())
        })
        .collect();
    let models_vec = models_vec.map_err(|_: JsValue| JsError::new("Invalid model losses array"))?;

    let result = spa_test(
        benchmark_losses,
        &models_vec,
        n_bootstrap.unwrap_or(1000),
        block_length,
        seed,
    )
    .map_err(|e| JsError::new(&e.to_string()))?;

    let js_result = SPAResultJs {
        statistic: result.statistic,
        p_value_consistent: result.p_value_consistent,
        p_value_upper: result.p_value_upper,
        n_bootstrap: result.n_bootstrap,
        best_model_idx: result.best_model_idx,
    };

    serde_wasm_bindgen::to_value(&js_result).map_err(|e| JsError::new(&e.to_string()))
}

#[derive(Serialize)]
struct MSPEAdjustedResultJs {
    statistic: f64,
    p_value_consistent: f64,
    p_value_upper: f64,
    n_bootstrap: usize,
    best_model_idx: Option<usize>,
}

/// MSPE-adjusted SPA test.
///
/// @param benchmarkErrors - Forecast errors from benchmark model as Float64Array
/// @param modelErrors - Array of forecast error arrays from alternative models
/// @param nBootstrap - Number of bootstrap samples (default: 1000)
/// @param blockLength - Expected block length for stationary bootstrap
/// @param seed - Random seed (optional)
/// @returns Object with statistic, p_value_consistent, p_value_upper
#[wasm_bindgen(js_name = mspeAdjustedSpa)]
pub fn js_mspe_adjusted_spa(
    benchmark_errors: &[f64],
    model_errors: js_sys::Array,
    n_bootstrap: Option<usize>,
    block_length: f64,
    seed: Option<u64>,
) -> Result<JsValue, JsError> {
    let models_vec: Result<Vec<Vec<f64>>, _> = model_errors
        .iter()
        .map(|row| {
            let arr = js_sys::Float64Array::new(&row);
            Ok(arr.to_vec())
        })
        .collect();
    let models_vec = models_vec.map_err(|_: JsValue| JsError::new("Invalid model errors array"))?;

    let result = mspe_adjusted_spa(
        benchmark_errors,
        &models_vec,
        n_bootstrap.unwrap_or(1000),
        block_length,
        seed,
    )
    .map_err(|e| JsError::new(&e.to_string()))?;

    let js_result = MSPEAdjustedResultJs {
        statistic: result.statistic,
        p_value_consistent: result.p_value_consistent,
        p_value_upper: result.p_value_upper,
        n_bootstrap: result.n_bootstrap,
        best_model_idx: result.best_model_idx,
    };

    serde_wasm_bindgen::to_value(&js_result).map_err(|e| JsError::new(&e.to_string()))
}

/// MCS statistic type
#[wasm_bindgen]
#[derive(Clone, Copy)]
pub enum JsMCSStatistic {
    Max,
    Range,
}

impl From<JsMCSStatistic> for MCSStatistic {
    fn from(s: JsMCSStatistic) -> Self {
        match s {
            JsMCSStatistic::Max => MCSStatistic::Max,
            JsMCSStatistic::Range => MCSStatistic::Range,
        }
    }
}

#[derive(Serialize)]
struct MCSResultJs {
    included_models: Vec<usize>,
    eliminated_models: Vec<usize>,
    mcs_p_value: f64,
    elimination_sequence: Vec<EliminationStepJs>,
    n_bootstrap: usize,
}

#[derive(Serialize)]
struct EliminationStepJs {
    model_idx: usize,
    p_value: f64,
    eliminated: bool,
}

/// Model Confidence Set (MCS) procedure.
///
/// Identifies the set of models that contains the best model with a given confidence.
///
/// @param losses - Array of loss value arrays (one per model)
/// @param alpha - Significance level (default: 0.1)
/// @param statistic - MCS statistic (TMax or TR)
/// @param nBootstrap - Number of bootstrap samples (default: 1000)
/// @param blockLength - Expected block length for stationary bootstrap
/// @param seed - Random seed (optional)
/// @returns Object with included_models, excluded_models, p_values, elimination_order
#[wasm_bindgen(js_name = modelConfidenceSet)]
pub fn js_model_confidence_set(
    losses: js_sys::Array,
    alpha: Option<f64>,
    statistic: JsMCSStatistic,
    n_bootstrap: Option<usize>,
    block_length: f64,
    seed: Option<u64>,
) -> Result<JsValue, JsError> {
    let losses_vec: Result<Vec<Vec<f64>>, _> = losses
        .iter()
        .map(|row| {
            let arr = js_sys::Float64Array::new(&row);
            Ok(arr.to_vec())
        })
        .collect();
    let losses_vec = losses_vec.map_err(|_: JsValue| JsError::new("Invalid losses array"))?;

    let result = model_confidence_set(
        &losses_vec,
        alpha.unwrap_or(0.1),
        statistic.into(),
        n_bootstrap.unwrap_or(1000),
        block_length,
        seed,
    )
    .map_err(|e| JsError::new(&e.to_string()))?;

    let js_result = MCSResultJs {
        included_models: result.included_models,
        eliminated_models: result.eliminated_models,
        mcs_p_value: result.mcs_p_value,
        elimination_sequence: result
            .elimination_sequence
            .into_iter()
            .map(|s| EliminationStepJs {
                model_idx: s.model_idx,
                p_value: s.p_value,
                eliminated: s.eliminated,
            })
            .collect(),
        n_bootstrap: result.n_bootstrap,
    };

    serde_wasm_bindgen::to_value(&js_result).map_err(|e| JsError::new(&e.to_string()))
}
