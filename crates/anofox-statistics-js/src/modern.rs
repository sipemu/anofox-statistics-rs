use anofox_statistics::{energy_distance_test, energy_distance_test_1d, mmd_test, mmd_test_1d, Kernel};
use serde::Serialize;
use wasm_bindgen::prelude::*;

#[derive(Serialize)]
struct EnergyDistanceResultJs {
    statistic: f64,
    p_value: f64,
    n_permutations: usize,
}

/// Energy distance two-sample test (1D data).
///
/// Tests if two samples come from the same distribution using energy distance.
///
/// @param x - First sample as Float64Array
/// @param y - Second sample as Float64Array
/// @param nPermutations - Number of permutations (default: 999)
/// @param seed - Random seed (optional)
/// @returns Object with statistic, p_value, n_permutations
#[wasm_bindgen(js_name = energyDistanceTest1d)]
pub fn js_energy_distance_test_1d(
    x: &[f64],
    y: &[f64],
    n_permutations: Option<usize>,
    seed: Option<u64>,
) -> Result<JsValue, JsError> {
    let result = energy_distance_test_1d(x, y, n_permutations.unwrap_or(999), seed)
        .map_err(|e| JsError::new(&e.to_string()))?;

    let js_result = EnergyDistanceResultJs {
        statistic: result.statistic,
        p_value: result.p_value,
        n_permutations: result.n_permutations,
    };

    serde_wasm_bindgen::to_value(&js_result).map_err(|e| JsError::new(&e.to_string()))
}

/// Energy distance two-sample test (multivariate data).
///
/// @param x - First sample as 2D array (each row is an observation)
/// @param y - Second sample as 2D array
/// @param nPermutations - Number of permutations (default: 999)
/// @param seed - Random seed (optional)
/// @returns Object with statistic, p_value, n_permutations
#[wasm_bindgen(js_name = energyDistanceTest)]
pub fn js_energy_distance_test(
    x: js_sys::Array,
    y: js_sys::Array,
    n_permutations: Option<usize>,
    seed: Option<u64>,
) -> Result<JsValue, JsError> {
    let x_vec: Result<Vec<Vec<f64>>, _> = x
        .iter()
        .map(|row| {
            let arr = js_sys::Float64Array::new(&row);
            Ok(arr.to_vec())
        })
        .collect();
    let x_vec = x_vec.map_err(|_: JsValue| JsError::new("Invalid input array x"))?;

    let y_vec: Result<Vec<Vec<f64>>, _> = y
        .iter()
        .map(|row| {
            let arr = js_sys::Float64Array::new(&row);
            Ok(arr.to_vec())
        })
        .collect();
    let y_vec = y_vec.map_err(|_: JsValue| JsError::new("Invalid input array y"))?;

    let result = energy_distance_test(&x_vec, &y_vec, n_permutations.unwrap_or(999), seed)
        .map_err(|e| JsError::new(&e.to_string()))?;

    let js_result = EnergyDistanceResultJs {
        statistic: result.statistic,
        p_value: result.p_value,
        n_permutations: result.n_permutations,
    };

    serde_wasm_bindgen::to_value(&js_result).map_err(|e| JsError::new(&e.to_string()))
}

/// Kernel type for MMD test
#[wasm_bindgen]
#[derive(Clone, Copy)]
pub enum JsKernel {
    Gaussian,
    Laplacian,
    Linear,
}

#[derive(Serialize)]
struct MMDResultJs {
    statistic: f64,
    p_value: f64,
    n_permutations: usize,
}

/// Maximum Mean Discrepancy two-sample test (1D data).
///
/// Uses median heuristic for bandwidth selection automatically.
///
/// @param x - First sample as Float64Array
/// @param y - Second sample as Float64Array
/// @param nPermutations - Number of permutations (default: 999)
/// @param seed - Random seed (optional)
/// @returns Object with statistic, p_value, n_permutations
#[wasm_bindgen(js_name = mmdTest1d)]
pub fn js_mmd_test_1d(
    x: &[f64],
    y: &[f64],
    n_permutations: Option<usize>,
    seed: Option<u64>,
) -> Result<JsValue, JsError> {
    let result = mmd_test_1d(x, y, n_permutations.unwrap_or(999), seed)
        .map_err(|e| JsError::new(&e.to_string()))?;

    let js_result = MMDResultJs {
        statistic: result.statistic,
        p_value: result.p_value,
        n_permutations: result.n_permutations,
    };

    serde_wasm_bindgen::to_value(&js_result).map_err(|e| JsError::new(&e.to_string()))
}

/// Maximum Mean Discrepancy two-sample test (multivariate data).
///
/// @param x - First sample as 2D array
/// @param y - Second sample as 2D array
/// @param kernel - Kernel type (Gaussian, Laplacian, or Linear)
/// @param bandwidth - Kernel bandwidth (required for Gaussian and Laplacian)
/// @param nPermutations - Number of permutations (default: 999)
/// @param seed - Random seed (optional)
/// @returns Object with statistic, p_value, n_permutations
#[wasm_bindgen(js_name = mmdTest)]
pub fn js_mmd_test(
    x: js_sys::Array,
    y: js_sys::Array,
    kernel: JsKernel,
    bandwidth: Option<f64>,
    n_permutations: Option<usize>,
    seed: Option<u64>,
) -> Result<JsValue, JsError> {
    let x_vec: Result<Vec<Vec<f64>>, _> = x
        .iter()
        .map(|row| {
            let arr = js_sys::Float64Array::new(&row);
            Ok(arr.to_vec())
        })
        .collect();
    let x_vec = x_vec.map_err(|_: JsValue| JsError::new("Invalid input array x"))?;

    let y_vec: Result<Vec<Vec<f64>>, _> = y
        .iter()
        .map(|row| {
            let arr = js_sys::Float64Array::new(&row);
            Ok(arr.to_vec())
        })
        .collect();
    let y_vec = y_vec.map_err(|_: JsValue| JsError::new("Invalid input array y"))?;

    let bw = bandwidth.unwrap_or(1.0);
    let kern = match kernel {
        JsKernel::Gaussian => Kernel::Gaussian { bandwidth: bw },
        JsKernel::Laplacian => Kernel::Laplacian { bandwidth: bw },
        JsKernel::Linear => Kernel::Linear,
    };

    let result = mmd_test(&x_vec, &y_vec, kern, n_permutations.unwrap_or(999), seed)
        .map_err(|e| JsError::new(&e.to_string()))?;

    let js_result = MMDResultJs {
        statistic: result.statistic,
        p_value: result.p_value,
        n_permutations: result.n_permutations,
    };

    serde_wasm_bindgen::to_value(&js_result).map_err(|e| JsError::new(&e.to_string()))
}
