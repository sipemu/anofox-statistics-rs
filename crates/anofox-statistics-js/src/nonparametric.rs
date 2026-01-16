use anofox_statistics::{brunner_munzel, kruskal_wallis, mann_whitney_u, wilcoxon_signed_rank};
use serde::Serialize;
use wasm_bindgen::prelude::*;

use crate::parametric::JsAlternative;
use anofox_statistics::Alternative;

#[derive(Serialize)]
struct MannWhitneyResultJs {
    statistic: f64,
    p_value: f64,
    estimate: Option<f64>,
    null_value: f64,
    conf_int: Option<ConfIntJs>,
}

#[derive(Serialize)]
struct ConfIntJs {
    lower: f64,
    upper: f64,
    conf_level: f64,
}

/// Mann-Whitney U test (Wilcoxon rank-sum test).
///
/// @param x - First sample as Float64Array
/// @param y - Second sample as Float64Array
/// @param alternative - Alternative hypothesis direction
/// @param continuityCorrectionParam - Whether to apply continuity correction (default: true)
/// @param exact - Whether to compute exact p-value (default: false)
/// @param confLevel - Confidence level for Hodges-Lehmann CI (optional)
/// @param mu - Location shift under H0 (default: 0)
/// @returns Object with statistic, p_value, estimate, null_value, conf_int
#[wasm_bindgen(js_name = mannWhitneyU)]
pub fn js_mann_whitney_u(
    x: &[f64],
    y: &[f64],
    alternative: JsAlternative,
    continuity_correction: Option<bool>,
    exact: Option<bool>,
    conf_level: Option<f64>,
    mu: Option<f64>,
) -> Result<JsValue, JsError> {
    let alt: Alternative = match alternative {
        JsAlternative::TwoSided => Alternative::TwoSided,
        JsAlternative::Less => Alternative::Less,
        JsAlternative::Greater => Alternative::Greater,
    };

    let result = mann_whitney_u(
        x,
        y,
        alt,
        continuity_correction.unwrap_or(true),
        exact.unwrap_or(false),
        conf_level,
        mu,
    )
    .map_err(|e| JsError::new(&e.to_string()))?;

    let js_result = MannWhitneyResultJs {
        statistic: result.statistic,
        p_value: result.p_value,
        estimate: result.estimate,
        null_value: result.null_value,
        conf_int: result.conf_int.map(|ci| ConfIntJs {
            lower: ci.lower,
            upper: ci.upper,
            conf_level: ci.conf_level,
        }),
    };

    serde_wasm_bindgen::to_value(&js_result).map_err(|e| JsError::new(&e.to_string()))
}

#[derive(Serialize)]
struct WilcoxonResultJs {
    statistic: f64,
    p_value: f64,
    estimate: Option<f64>,
    null_value: f64,
    conf_int: Option<ConfIntJs>,
}

/// Wilcoxon signed-rank test.
///
/// @param x - First sample as Float64Array
/// @param y - Second sample as Float64Array (optional, for paired test)
/// @param alternative - Alternative hypothesis direction
/// @param continuityCorrectionParam - Whether to apply continuity correction (default: true)
/// @param exact - Whether to compute exact p-value (default: false)
/// @param confLevel - Confidence level for CI (optional)
/// @param mu - Null hypothesis value (default: 0.0)
/// @returns Object with statistic, p_value, estimate, null_value, conf_int
#[wasm_bindgen(js_name = wilcoxonSignedRank)]
pub fn js_wilcoxon_signed_rank(
    x: &[f64],
    y: &[f64],
    alternative: JsAlternative,
    continuity_correction: Option<bool>,
    exact: Option<bool>,
    conf_level: Option<f64>,
    mu: Option<f64>,
) -> Result<JsValue, JsError> {
    let alt: Alternative = match alternative {
        JsAlternative::TwoSided => Alternative::TwoSided,
        JsAlternative::Less => Alternative::Less,
        JsAlternative::Greater => Alternative::Greater,
    };

    let result = wilcoxon_signed_rank(
        x,
        y,
        alt,
        continuity_correction.unwrap_or(true),
        exact.unwrap_or(false),
        conf_level,
        mu,
    )
    .map_err(|e| JsError::new(&e.to_string()))?;

    let js_result = WilcoxonResultJs {
        statistic: result.statistic,
        p_value: result.p_value,
        estimate: result.estimate,
        null_value: result.null_value,
        conf_int: result.conf_int.map(|ci| ConfIntJs {
            lower: ci.lower,
            upper: ci.upper,
            conf_level: ci.conf_level,
        }),
    };

    serde_wasm_bindgen::to_value(&js_result).map_err(|e| JsError::new(&e.to_string()))
}

#[derive(Serialize)]
struct KruskalResultJs {
    statistic: f64,
    df: f64,
    p_value: f64,
}

/// Kruskal-Wallis H test (non-parametric one-way ANOVA).
///
/// @param groups - Array of groups, each group is a Float64Array
/// @returns Object with statistic (H), df, p_value
#[wasm_bindgen(js_name = kruskalWallis)]
pub fn js_kruskal_wallis(groups: js_sys::Array) -> Result<JsValue, JsError> {
    let groups_vec: Result<Vec<Vec<f64>>, _> = groups
        .iter()
        .map(|g| {
            let arr = js_sys::Float64Array::new(&g);
            Ok(arr.to_vec())
        })
        .collect();

    let groups_vec = groups_vec.map_err(|_: JsValue| JsError::new("Invalid input arrays"))?;
    let group_refs: Vec<&[f64]> = groups_vec.iter().map(|g| g.as_slice()).collect();

    let result = kruskal_wallis(&group_refs).map_err(|e| JsError::new(&e.to_string()))?;

    let js_result = KruskalResultJs {
        statistic: result.statistic,
        df: result.df,
        p_value: result.p_value,
    };

    serde_wasm_bindgen::to_value(&js_result).map_err(|e| JsError::new(&e.to_string()))
}

#[derive(Serialize)]
struct BrunnerMunzelResultJs {
    statistic: f64,
    df: f64,
    p_value: f64,
    estimate: f64,
    conf_int: Option<BmConfIntJs>,
}

#[derive(Serialize)]
struct BmConfIntJs {
    lower: f64,
    upper: f64,
    conf_level: f64,
}

/// Brunner-Munzel test for stochastic equality.
///
/// @param x - First sample as Float64Array
/// @param y - Second sample as Float64Array
/// @param alternative - Alternative hypothesis direction
/// @param alpha - If specified, compute CI at (1-alpha) level (e.g., 0.05 for 95% CI)
/// @returns Object with statistic, df, p_value, estimate, conf_int
#[wasm_bindgen(js_name = brunnerMunzel)]
pub fn js_brunner_munzel(
    x: &[f64],
    y: &[f64],
    alternative: JsAlternative,
    alpha: Option<f64>,
) -> Result<JsValue, JsError> {
    let alt: Alternative = match alternative {
        JsAlternative::TwoSided => Alternative::TwoSided,
        JsAlternative::Less => Alternative::Less,
        JsAlternative::Greater => Alternative::Greater,
    };

    let result = brunner_munzel(x, y, alt, alpha).map_err(|e| JsError::new(&e.to_string()))?;

    let js_result = BrunnerMunzelResultJs {
        statistic: result.statistic,
        df: result.df,
        p_value: result.p_value,
        estimate: result.estimate,
        conf_int: result.conf_int.map(|ci| BmConfIntJs {
            lower: ci.lower,
            upper: ci.upper,
            conf_level: ci.conf_level,
        }),
    };

    serde_wasm_bindgen::to_value(&js_result).map_err(|e| JsError::new(&e.to_string()))
}
