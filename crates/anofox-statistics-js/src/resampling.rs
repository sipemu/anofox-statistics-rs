use anofox_statistics::permutation_t_test;
use serde::Serialize;
use wasm_bindgen::prelude::*;

use crate::parametric::JsAlternative;
use anofox_statistics::Alternative;

#[derive(Serialize)]
struct PermutationResultJs {
    statistic: f64,
    p_value: f64,
    n_permutations: usize,
}

/// Permutation t-test.
///
/// @param x - First sample as Float64Array
/// @param y - Second sample as Float64Array
/// @param alternative - Alternative hypothesis direction
/// @param nPermutations - Number of permutations (default: 9999)
/// @param seed - Random seed (optional)
/// @returns Object with statistic, p_value, n_permutations
#[wasm_bindgen(js_name = permutationTTest)]
pub fn js_permutation_t_test(
    x: &[f64],
    y: &[f64],
    alternative: JsAlternative,
    n_permutations: Option<usize>,
    seed: Option<u64>,
) -> Result<JsValue, JsError> {
    let alt: Alternative = match alternative {
        JsAlternative::TwoSided => Alternative::TwoSided,
        JsAlternative::Less => Alternative::Less,
        JsAlternative::Greater => Alternative::Greater,
    };

    let result = permutation_t_test(x, y, alt, n_permutations.unwrap_or(9999), seed)
        .map_err(|e| JsError::new(&e.to_string()))?;

    let js_result = PermutationResultJs {
        statistic: result.statistic,
        p_value: result.p_value,
        n_permutations: result.n_permutations,
    };

    serde_wasm_bindgen::to_value(&js_result).map_err(|e| JsError::new(&e.to_string()))
}
