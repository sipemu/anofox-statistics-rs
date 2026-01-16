use anofox_statistics::{
    brown_forsythe, one_way_anova, t_test, yuen_test, Alternative, AnovaKind, TTestKind,
};
use serde::Serialize;
use wasm_bindgen::prelude::*;

/// T-test kind for JavaScript
#[wasm_bindgen]
#[derive(Clone, Copy)]
pub enum JsTTestKind {
    Welch,
    Student,
    Paired,
}

/// Alternative hypothesis for JavaScript
#[wasm_bindgen]
#[derive(Clone, Copy)]
pub enum JsAlternative {
    TwoSided,
    Less,
    Greater,
}

impl From<JsTTestKind> for TTestKind {
    fn from(kind: JsTTestKind) -> Self {
        match kind {
            JsTTestKind::Welch => TTestKind::Welch,
            JsTTestKind::Student => TTestKind::Student,
            JsTTestKind::Paired => TTestKind::Paired,
        }
    }
}

impl From<JsAlternative> for Alternative {
    fn from(alt: JsAlternative) -> Self {
        match alt {
            JsAlternative::TwoSided => Alternative::TwoSided,
            JsAlternative::Less => Alternative::Less,
            JsAlternative::Greater => Alternative::Greater,
        }
    }
}

#[derive(Serialize)]
struct TTestResultJs {
    statistic: f64,
    df: f64,
    p_value: f64,
    mean_x: f64,
    mean_y: Option<f64>,
    null_value: f64,
    conf_int: Option<ConfIntJs>,
}

#[derive(Serialize)]
struct ConfIntJs {
    lower: f64,
    upper: f64,
    conf_level: f64,
}

/// Perform a t-test comparing two samples.
///
/// @param x - First sample as Float64Array
/// @param y - Second sample as Float64Array
/// @param kind - Type of t-test (Welch, Student, or Paired)
/// @param alternative - Alternative hypothesis direction
/// @param mu - Null hypothesis value (default: 0.0)
/// @param confLevel - Confidence level for CI (optional, e.g., 0.95)
/// @returns Object with statistic, df, p_value, mean_x, mean_y, conf_int
#[wasm_bindgen(js_name = tTest)]
pub fn js_t_test(
    x: &[f64],
    y: &[f64],
    kind: JsTTestKind,
    alternative: JsAlternative,
    mu: Option<f64>,
    conf_level: Option<f64>,
) -> Result<JsValue, JsError> {
    let result = t_test(
        x,
        y,
        kind.into(),
        alternative.into(),
        mu.unwrap_or(0.0),
        conf_level,
    )
    .map_err(|e| JsError::new(&e.to_string()))?;

    let js_result = TTestResultJs {
        statistic: result.statistic,
        df: result.df,
        p_value: result.p_value,
        mean_x: result.mean_x,
        mean_y: result.mean_y,
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
struct YuenResultJs {
    statistic: f64,
    df: f64,
    p_value: f64,
    diff: f64,
    trimmed_mean_x: f64,
    trimmed_mean_y: f64,
    conf_int: Option<YuenConfIntJs>,
}

#[derive(Serialize)]
struct YuenConfIntJs {
    lower: f64,
    upper: f64,
    conf_level: f64,
}

/// Yuen's test for trimmed means (robust alternative to t-test).
///
/// @param x - First sample as Float64Array
/// @param y - Second sample as Float64Array
/// @param trim - Trim proportion (default: 0.2 for 20% trimming)
/// @param alternative - Alternative hypothesis direction
/// @param confLevel - Confidence level for CI (optional)
/// @returns Object with statistic, df, p_value, diff, trimmed_mean_x, trimmed_mean_y
#[wasm_bindgen(js_name = yuenTest)]
pub fn js_yuen_test(
    x: &[f64],
    y: &[f64],
    trim: Option<f64>,
    alternative: JsAlternative,
    conf_level: Option<f64>,
) -> Result<JsValue, JsError> {
    let alt: Alternative = match alternative {
        JsAlternative::TwoSided => Alternative::TwoSided,
        JsAlternative::Less => Alternative::Less,
        JsAlternative::Greater => Alternative::Greater,
    };

    let result = yuen_test(x, y, trim.unwrap_or(0.2), alt, conf_level)
        .map_err(|e| JsError::new(&e.to_string()))?;

    let js_result = YuenResultJs {
        statistic: result.statistic,
        df: result.df,
        p_value: result.p_value,
        diff: result.diff,
        trimmed_mean_x: result.trimmed_mean_x,
        trimmed_mean_y: result.trimmed_mean_y,
        conf_int: result.conf_int.map(|ci| YuenConfIntJs {
            lower: ci.lower,
            upper: ci.upper,
            conf_level: ci.conf_level,
        }),
    };

    serde_wasm_bindgen::to_value(&js_result).map_err(|e| JsError::new(&e.to_string()))
}

#[derive(Serialize)]
struct LeveneResultJs {
    statistic: f64,
    df1: f64,
    df2: f64,
    p_value: f64,
}

/// Brown-Forsythe test for homogeneity of variances.
///
/// @param groups - Array of groups, each group is a Float64Array
/// @returns Object with statistic, df1, df2, p_value
#[wasm_bindgen(js_name = brownForsythe)]
pub fn js_brown_forsythe(groups: js_sys::Array) -> Result<JsValue, JsError> {
    let groups_vec: Result<Vec<Vec<f64>>, _> = groups
        .iter()
        .map(|g| {
            let arr = js_sys::Float64Array::new(&g);
            Ok(arr.to_vec())
        })
        .collect();

    let groups_vec = groups_vec.map_err(|_: JsValue| JsError::new("Invalid input arrays"))?;
    let group_refs: Vec<&[f64]> = groups_vec.iter().map(|g| g.as_slice()).collect();

    let result = brown_forsythe(&group_refs).map_err(|e| JsError::new(&e.to_string()))?;

    let js_result = LeveneResultJs {
        statistic: result.statistic,
        df1: result.df1,
        df2: result.df2,
        p_value: result.p_value,
    };

    serde_wasm_bindgen::to_value(&js_result).map_err(|e| JsError::new(&e.to_string()))
}

/// ANOVA kind for JavaScript
#[wasm_bindgen]
#[derive(Clone, Copy)]
pub enum JsAnovaKind {
    Fisher,
    Welch,
}

impl From<JsAnovaKind> for AnovaKind {
    fn from(kind: JsAnovaKind) -> Self {
        match kind {
            JsAnovaKind::Fisher => AnovaKind::Fisher,
            JsAnovaKind::Welch => AnovaKind::Welch,
        }
    }
}

#[derive(Serialize)]
struct OneWayAnovaResultJs {
    statistic: f64,
    df_between: f64,
    df_within: f64,
    p_value: f64,
    ss_between: Option<f64>,
    ss_within: Option<f64>,
    ss_total: Option<f64>,
    ms_between: Option<f64>,
    ms_within: Option<f64>,
    n_groups: usize,
    group_sizes: Vec<usize>,
    group_means: Vec<f64>,
    grand_mean: Option<f64>,
}

/// One-way ANOVA test.
///
/// @param groups - Array of groups, each group is a Float64Array
/// @param kind - Type of ANOVA (Fisher or Welch)
/// @returns Object with statistic, df_between, df_within, p_value, SS and MS components
#[wasm_bindgen(js_name = oneWayAnova)]
pub fn js_one_way_anova(groups: js_sys::Array, kind: JsAnovaKind) -> Result<JsValue, JsError> {
    let groups_vec: Result<Vec<Vec<f64>>, _> = groups
        .iter()
        .map(|g| {
            let arr = js_sys::Float64Array::new(&g);
            Ok(arr.to_vec())
        })
        .collect();

    let groups_vec = groups_vec.map_err(|_: JsValue| JsError::new("Invalid input arrays"))?;
    let group_refs: Vec<&[f64]> = groups_vec.iter().map(|g| g.as_slice()).collect();

    let result =
        one_way_anova(&group_refs, kind.into()).map_err(|e| JsError::new(&e.to_string()))?;

    let js_result = OneWayAnovaResultJs {
        statistic: result.statistic,
        df_between: result.df_between,
        df_within: result.df_within,
        p_value: result.p_value,
        ss_between: result.ss_between,
        ss_within: result.ss_within,
        ss_total: result.ss_total,
        ms_between: result.ms_between,
        ms_within: result.ms_within,
        n_groups: result.n_groups,
        group_sizes: result.group_sizes,
        group_means: result.group_means,
        grand_mean: result.grand_mean,
    };

    serde_wasm_bindgen::to_value(&js_result).map_err(|e| JsError::new(&e.to_string()))
}
