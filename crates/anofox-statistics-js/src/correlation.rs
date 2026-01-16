use anofox_statistics::{
    distance_cor, distance_cor_test, icc, kendall, partial_cor, pearson, semi_partial_cor,
    spearman, ICCType, KendallVariant,
};
use serde::Serialize;
use wasm_bindgen::prelude::*;

#[derive(Serialize)]
struct CorrelationResultJs {
    estimate: f64,
    statistic: f64,
    df: Option<f64>,
    p_value: f64,
    conf_int: Option<ConfIntJs>,
    method: String,
    n: usize,
}

#[derive(Serialize)]
struct ConfIntJs {
    lower: f64,
    upper: f64,
    conf_level: f64,
}

/// Pearson correlation coefficient with significance test.
///
/// @param x - First variable as Float64Array
/// @param y - Second variable as Float64Array
/// @param confLevel - Confidence level for CI (optional, e.g., 0.95)
/// @returns Object with estimate, statistic, df, p_value, conf_int
#[wasm_bindgen(js_name = pearsonCorrelation)]
pub fn js_pearson(x: &[f64], y: &[f64], conf_level: Option<f64>) -> Result<JsValue, JsError> {
    let result = pearson(x, y, conf_level).map_err(|e| JsError::new(&e.to_string()))?;

    let js_result = CorrelationResultJs {
        estimate: result.estimate,
        statistic: result.statistic,
        df: result.df,
        p_value: result.p_value,
        conf_int: result.conf_int.map(|ci| ConfIntJs {
            lower: ci.lower,
            upper: ci.upper,
            conf_level: ci.conf_level,
        }),
        method: format!("{}", result.method),
        n: result.n,
    };

    serde_wasm_bindgen::to_value(&js_result).map_err(|e| JsError::new(&e.to_string()))
}

/// Spearman rank correlation coefficient with significance test.
///
/// @param x - First variable as Float64Array
/// @param y - Second variable as Float64Array
/// @param confLevel - Confidence level for CI (optional)
/// @returns Object with estimate, statistic, df, p_value, conf_int
#[wasm_bindgen(js_name = spearmanCorrelation)]
pub fn js_spearman(x: &[f64], y: &[f64], conf_level: Option<f64>) -> Result<JsValue, JsError> {
    let result = spearman(x, y, conf_level).map_err(|e| JsError::new(&e.to_string()))?;

    let js_result = CorrelationResultJs {
        estimate: result.estimate,
        statistic: result.statistic,
        df: result.df,
        p_value: result.p_value,
        conf_int: result.conf_int.map(|ci| ConfIntJs {
            lower: ci.lower,
            upper: ci.upper,
            conf_level: ci.conf_level,
        }),
        method: format!("{}", result.method),
        n: result.n,
    };

    serde_wasm_bindgen::to_value(&js_result).map_err(|e| JsError::new(&e.to_string()))
}

/// Kendall variant for JavaScript
#[wasm_bindgen]
#[derive(Clone, Copy)]
pub enum JsKendallVariant {
    TauA,
    TauB,
    TauC,
}

impl From<JsKendallVariant> for KendallVariant {
    fn from(v: JsKendallVariant) -> Self {
        match v {
            JsKendallVariant::TauA => KendallVariant::TauA,
            JsKendallVariant::TauB => KendallVariant::TauB,
            JsKendallVariant::TauC => KendallVariant::TauC,
        }
    }
}

/// Kendall's tau correlation coefficient.
///
/// @param x - First variable as Float64Array
/// @param y - Second variable as Float64Array
/// @param variant - Tau variant (TauA, TauB, TauC)
/// @returns Object with estimate, statistic, p_value, conf_int
#[wasm_bindgen(js_name = kendallCorrelation)]
pub fn js_kendall(x: &[f64], y: &[f64], variant: JsKendallVariant) -> Result<JsValue, JsError> {
    let result = kendall(x, y, variant.into()).map_err(|e| JsError::new(&e.to_string()))?;

    let js_result = CorrelationResultJs {
        estimate: result.estimate,
        statistic: result.statistic,
        df: result.df,
        p_value: result.p_value,
        conf_int: result.conf_int.map(|ci| ConfIntJs {
            lower: ci.lower,
            upper: ci.upper,
            conf_level: ci.conf_level,
        }),
        method: format!("{}", result.method),
        n: result.n,
    };

    serde_wasm_bindgen::to_value(&js_result).map_err(|e| JsError::new(&e.to_string()))
}

#[derive(Serialize)]
struct DistanceCorResultJs {
    dcor: f64,
    dcov: f64,
    dvar_x: f64,
    dvar_y: f64,
    statistic: f64,
}

/// Distance correlation (measures non-linear dependence).
///
/// @param x - First variable as Float64Array
/// @param y - Second variable as Float64Array
/// @returns Object with dcor, dcov, dvar_x, dvar_y, statistic
#[wasm_bindgen(js_name = distanceCorrelation)]
pub fn js_distance_cor(x: &[f64], y: &[f64]) -> Result<JsValue, JsError> {
    let result = distance_cor(x, y).map_err(|e| JsError::new(&e.to_string()))?;

    let js_result = DistanceCorResultJs {
        dcor: result.dcor,
        dcov: result.dcov,
        dvar_x: result.dvar_x,
        dvar_y: result.dvar_y,
        statistic: result.statistic,
    };

    serde_wasm_bindgen::to_value(&js_result).map_err(|e| JsError::new(&e.to_string()))
}

#[derive(Serialize)]
struct DistanceCorTestResultJs {
    dcor: f64,
    statistic: f64,
    p_value: f64,
}

/// Distance correlation with permutation test.
///
/// @param x - First variable as Float64Array
/// @param y - Second variable as Float64Array
/// @param nPermutations - Number of permutations (default: 999)
/// @param seed - Random seed (optional)
/// @returns Object with dcor, statistic, p_value
#[wasm_bindgen(js_name = distanceCorrelationTest)]
pub fn js_distance_cor_test(
    x: &[f64],
    y: &[f64],
    n_permutations: Option<usize>,
    seed: Option<u64>,
) -> Result<JsValue, JsError> {
    let result = distance_cor_test(x, y, n_permutations.unwrap_or(999), seed)
        .map_err(|e| JsError::new(&e.to_string()))?;

    let js_result = DistanceCorTestResultJs {
        dcor: result.dcor,
        statistic: result.statistic,
        p_value: result.p_value.unwrap_or(f64::NAN),
    };

    serde_wasm_bindgen::to_value(&js_result).map_err(|e| JsError::new(&e.to_string()))
}

#[derive(Serialize)]
struct PartialCorResultJs {
    estimate: f64,
    statistic: f64,
    df: f64,
    p_value: f64,
}

/// Partial correlation controlling for confounders.
///
/// @param x - First variable as Float64Array
/// @param y - Second variable as Float64Array
/// @param z - Array of confounding variables (each as Float64Array)
/// @returns Object with estimate, statistic, df, p_value
#[wasm_bindgen(js_name = partialCorrelation)]
pub fn js_partial_cor(x: &[f64], y: &[f64], z: js_sys::Array) -> Result<JsValue, JsError> {
    let z_vecs: Result<Vec<Vec<f64>>, _> = z
        .iter()
        .map(|row| {
            let arr = js_sys::Float64Array::new(&row);
            Ok(arr.to_vec())
        })
        .collect();
    let z_vecs = z_vecs.map_err(|_: JsValue| JsError::new("Invalid control variables"))?;
    let z_refs: Vec<&[f64]> = z_vecs.iter().map(|v| v.as_slice()).collect();

    let result = partial_cor(x, y, &z_refs).map_err(|e| JsError::new(&e.to_string()))?;

    let js_result = PartialCorResultJs {
        estimate: result.estimate,
        statistic: result.statistic,
        df: result.df,
        p_value: result.p_value,
    };

    serde_wasm_bindgen::to_value(&js_result).map_err(|e| JsError::new(&e.to_string()))
}

/// Semi-partial (part) correlation.
///
/// @param x - First variable as Float64Array
/// @param y - Second variable as Float64Array
/// @param z - Array of confounding variables to control for x only (each as Float64Array)
/// @returns Object with estimate, statistic, df, p_value
#[wasm_bindgen(js_name = semiPartialCorrelation)]
pub fn js_semi_partial_cor(x: &[f64], y: &[f64], z: js_sys::Array) -> Result<JsValue, JsError> {
    let z_vecs: Result<Vec<Vec<f64>>, _> = z
        .iter()
        .map(|row| {
            let arr = js_sys::Float64Array::new(&row);
            Ok(arr.to_vec())
        })
        .collect();
    let z_vecs = z_vecs.map_err(|_: JsValue| JsError::new("Invalid control variables"))?;
    let z_refs: Vec<&[f64]> = z_vecs.iter().map(|v| v.as_slice()).collect();

    let result = semi_partial_cor(x, y, &z_refs).map_err(|e| JsError::new(&e.to_string()))?;

    let js_result = PartialCorResultJs {
        estimate: result.estimate,
        statistic: result.statistic,
        df: result.df,
        p_value: result.p_value,
    };

    serde_wasm_bindgen::to_value(&js_result).map_err(|e| JsError::new(&e.to_string()))
}

/// ICC type for JavaScript
#[wasm_bindgen]
#[derive(Clone, Copy)]
pub enum JsICCType {
    ICC1,
    ICC2,
    ICC3,
    ICC1k,
    ICC2k,
    ICC3k,
}

impl From<JsICCType> for ICCType {
    fn from(t: JsICCType) -> Self {
        match t {
            JsICCType::ICC1 => ICCType::ICC1,
            JsICCType::ICC2 => ICCType::ICC2,
            JsICCType::ICC3 => ICCType::ICC3,
            JsICCType::ICC1k => ICCType::ICC1k,
            JsICCType::ICC2k => ICCType::ICC2k,
            JsICCType::ICC3k => ICCType::ICC3k,
        }
    }
}

#[derive(Serialize)]
struct ICCResultJs {
    icc: f64,
    f_value: f64,
    df1: f64,
    df2: f64,
    p_value: f64,
    conf_int_lower: f64,
    conf_int_upper: f64,
    n_subjects: usize,
    n_raters: usize,
    icc_type: String,
}

/// Intraclass correlation coefficient.
///
/// @param data - 2D array where rows are subjects and columns are raters
/// @param iccType - ICC type (ICC1, ICC2, ICC3, ICC1k, ICC2k, ICC3k)
/// @returns Object with icc, f_value, df1, df2, p_value, conf_int_lower, conf_int_upper
#[wasm_bindgen(js_name = intraclassCorrelation)]
pub fn js_icc(data: js_sys::Array, icc_type: JsICCType) -> Result<JsValue, JsError> {
    let data_vec: Result<Vec<Vec<f64>>, _> = data
        .iter()
        .map(|row| {
            let arr = js_sys::Float64Array::new(&row);
            Ok(arr.to_vec())
        })
        .collect();

    let data_vec = data_vec.map_err(|_: JsValue| JsError::new("Invalid input arrays"))?;

    let result = icc(&data_vec, icc_type.into()).map_err(|e| JsError::new(&e.to_string()))?;

    let js_result = ICCResultJs {
        icc: result.icc,
        f_value: result.f_value,
        df1: result.df1,
        df2: result.df2,
        p_value: result.p_value,
        conf_int_lower: result.conf_int_lower,
        conf_int_upper: result.conf_int_upper,
        n_subjects: result.n_subjects,
        n_raters: result.n_raters,
        icc_type: format!("{}", result.icc_type),
    };

    serde_wasm_bindgen::to_value(&js_result).map_err(|e| JsError::new(&e.to_string()))
}
