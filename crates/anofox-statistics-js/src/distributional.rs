use anofox_statistics::{dagostino_k_squared, shapiro_wilk};
use serde::Serialize;
use wasm_bindgen::prelude::*;

#[derive(Serialize)]
struct ShapiroWilkResultJs {
    statistic: f64,
    p_value: f64,
}

/// Shapiro-Wilk test for normality.
///
/// @param x - Sample data as Float64Array
/// @returns Object with statistic (W) and p_value
#[wasm_bindgen(js_name = shapiroWilk)]
pub fn js_shapiro_wilk(x: &[f64]) -> Result<JsValue, JsError> {
    let result = shapiro_wilk(x).map_err(|e| JsError::new(&e.to_string()))?;

    let js_result = ShapiroWilkResultJs {
        statistic: result.statistic,
        p_value: result.p_value,
    };

    serde_wasm_bindgen::to_value(&js_result).map_err(|e| JsError::new(&e.to_string()))
}

#[derive(Serialize)]
struct DAgostinoResultJs {
    k2_statistic: f64,
    p_value: f64,
    skewness_z: f64,
    kurtosis_z: f64,
}

/// D'Agostino-Pearson K² test for normality.
///
/// Tests normality using both skewness and kurtosis.
///
/// @param x - Sample data as Float64Array
/// @returns Object with k2_statistic, p_value, skewness_z, kurtosis_z
#[wasm_bindgen(js_name = dagostinoKSquared)]
pub fn js_dagostino_k_squared(x: &[f64]) -> Result<JsValue, JsError> {
    let result = dagostino_k_squared(x).map_err(|e| JsError::new(&e.to_string()))?;

    let js_result = DAgostinoResultJs {
        k2_statistic: result.statistic,
        p_value: result.p_value,
        skewness_z: result.z_skewness,
        kurtosis_z: result.z_kurtosis,
    };

    serde_wasm_bindgen::to_value(&js_result).map_err(|e| JsError::new(&e.to_string()))
}
