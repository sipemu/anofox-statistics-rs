use anofox_statistics::{
    binom_test, chisq_goodness_of_fit, chisq_test, cohen_kappa, contingency_coef, cramers_v,
    fisher_exact, g_test, mcnemar_exact, mcnemar_test, phi_coefficient, prop_test_one,
    prop_test_two, Alternative,
};
use serde::Serialize;
use wasm_bindgen::prelude::*;

use crate::parametric::JsAlternative;

#[derive(Serialize)]
struct ChiSquareResultJs {
    statistic: f64,
    df: f64,
    p_value: f64,
    expected: Vec<Vec<f64>>,
}

/// Chi-square test for independence.
///
/// @param observed - 2D array of observed counts (contingency table)
/// @param yatesCorrection - Apply Yates continuity correction for 2x2 tables (default: false)
/// @returns Object with statistic, df, p_value, expected frequencies
#[wasm_bindgen(js_name = chiSquareTest)]
pub fn js_chisq_test(
    observed: js_sys::Array,
    yates_correction: Option<bool>,
) -> Result<JsValue, JsError> {
    let table: Result<Vec<Vec<usize>>, _> = observed
        .iter()
        .map(|row| {
            let arr = js_sys::Array::from(&row);
            arr.iter()
                .map(|v| v.as_f64().map(|n| n as usize).ok_or(()))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect();

    let table = table.map_err(|_| JsError::new("Invalid contingency table"))?;

    let result = chisq_test(&table, yates_correction.unwrap_or(false))
        .map_err(|e| JsError::new(&e.to_string()))?;

    let js_result = ChiSquareResultJs {
        statistic: result.statistic,
        df: result.df,
        p_value: result.p_value,
        expected: result.expected,
    };

    serde_wasm_bindgen::to_value(&js_result).map_err(|e| JsError::new(&e.to_string()))
}

#[derive(Serialize)]
struct ChiSquareGofResultJs {
    statistic: f64,
    df: f64,
    p_value: f64,
}

/// Chi-square goodness-of-fit test.
///
/// @param observed - Observed counts as array of numbers
/// @param expectedProps - Expected proportions as array of numbers (must sum to 1), or null for uniform
/// @returns Object with statistic, df, p_value
#[wasm_bindgen(js_name = chiSquareGoodnessOfFit)]
pub fn js_chisq_goodness_of_fit(
    observed: &[f64],
    expected_props: Option<Vec<f64>>,
) -> Result<JsValue, JsError> {
    let obs: Vec<usize> = observed.iter().map(|&x| x as usize).collect();

    let result = chisq_goodness_of_fit(&obs, expected_props.as_deref())
        .map_err(|e| JsError::new(&e.to_string()))?;

    let js_result = ChiSquareGofResultJs {
        statistic: result.statistic,
        df: result.df,
        p_value: result.p_value,
    };

    serde_wasm_bindgen::to_value(&js_result).map_err(|e| JsError::new(&e.to_string()))
}

/// G-test (likelihood ratio test) for independence.
///
/// @param observed - 2D array of observed counts (contingency table)
/// @returns Object with statistic, df, p_value, expected frequencies
#[wasm_bindgen(js_name = gTest)]
pub fn js_g_test(observed: js_sys::Array) -> Result<JsValue, JsError> {
    let table: Result<Vec<Vec<usize>>, _> = observed
        .iter()
        .map(|row| {
            let arr = js_sys::Array::from(&row);
            arr.iter()
                .map(|v| v.as_f64().map(|n| n as usize).ok_or(()))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect();

    let table = table.map_err(|_| JsError::new("Invalid contingency table"))?;

    let result = g_test(&table).map_err(|e| JsError::new(&e.to_string()))?;

    let js_result = ChiSquareResultJs {
        statistic: result.statistic,
        df: result.df,
        p_value: result.p_value,
        expected: result.expected,
    };

    serde_wasm_bindgen::to_value(&js_result).map_err(|e| JsError::new(&e.to_string()))
}

#[derive(Serialize)]
struct FisherResultJs {
    odds_ratio: f64,
    p_value: f64,
    conf_int_lower: f64,
    conf_int_upper: f64,
}

/// Fisher's exact test for 2x2 contingency tables.
///
/// @param table - 2x2 contingency table as [[a, b], [c, d]]
/// @param alternative - Alternative hypothesis direction
/// @returns Object with odds_ratio, p_value, conf_int_lower, conf_int_upper
#[wasm_bindgen(js_name = fisherExact)]
pub fn js_fisher_exact(
    table: js_sys::Array,
    alternative: JsAlternative,
) -> Result<JsValue, JsError> {
    let rows: Vec<js_sys::Array> = table.iter().map(|r| js_sys::Array::from(&r)).collect();

    if rows.len() != 2 {
        return Err(JsError::new("Fisher's exact test requires a 2x2 table"));
    }

    let a = rows[0]
        .get(0)
        .as_f64()
        .ok_or_else(|| JsError::new("Invalid table"))? as usize;
    let b = rows[0]
        .get(1)
        .as_f64()
        .ok_or_else(|| JsError::new("Invalid table"))? as usize;
    let c = rows[1]
        .get(0)
        .as_f64()
        .ok_or_else(|| JsError::new("Invalid table"))? as usize;
    let d = rows[1]
        .get(1)
        .as_f64()
        .ok_or_else(|| JsError::new("Invalid table"))? as usize;

    let table_arr = [[a, b], [c, d]];

    let alt: Alternative = match alternative {
        JsAlternative::TwoSided => Alternative::TwoSided,
        JsAlternative::Less => Alternative::Less,
        JsAlternative::Greater => Alternative::Greater,
    };

    let result = fisher_exact(&table_arr, alt).map_err(|e| JsError::new(&e.to_string()))?;

    let js_result = FisherResultJs {
        odds_ratio: result.odds_ratio,
        p_value: result.p_value,
        conf_int_lower: result.conf_int_lower,
        conf_int_upper: result.conf_int_upper,
    };

    serde_wasm_bindgen::to_value(&js_result).map_err(|e| JsError::new(&e.to_string()))
}

#[derive(Serialize)]
struct AssociationResultJs {
    estimate: f64,
}

/// Cramér's V measure of association.
///
/// @param observed - 2D array of observed counts
/// @returns Object with estimate (Cramér's V)
#[wasm_bindgen(js_name = cramersV)]
pub fn js_cramers_v(observed: js_sys::Array) -> Result<JsValue, JsError> {
    let table: Result<Vec<Vec<usize>>, _> = observed
        .iter()
        .map(|row| {
            let arr = js_sys::Array::from(&row);
            arr.iter()
                .map(|v| v.as_f64().map(|n| n as usize).ok_or(()))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect();

    let table = table.map_err(|_| JsError::new("Invalid contingency table"))?;

    let result = cramers_v(&table).map_err(|e| JsError::new(&e.to_string()))?;

    let js_result = AssociationResultJs {
        estimate: result.estimate,
    };

    serde_wasm_bindgen::to_value(&js_result).map_err(|e| JsError::new(&e.to_string()))
}

/// Phi coefficient for 2x2 tables.
///
/// @param table - 2x2 contingency table as [[a, b], [c, d]]
/// @returns Object with estimate (phi coefficient)
#[wasm_bindgen(js_name = phiCoefficient)]
pub fn js_phi_coefficient(table: js_sys::Array) -> Result<JsValue, JsError> {
    let rows: Vec<js_sys::Array> = table.iter().map(|r| js_sys::Array::from(&r)).collect();

    if rows.len() != 2 {
        return Err(JsError::new("Phi coefficient requires a 2x2 table"));
    }

    let a = rows[0]
        .get(0)
        .as_f64()
        .ok_or_else(|| JsError::new("Invalid table"))? as usize;
    let b = rows[0]
        .get(1)
        .as_f64()
        .ok_or_else(|| JsError::new("Invalid table"))? as usize;
    let c = rows[1]
        .get(0)
        .as_f64()
        .ok_or_else(|| JsError::new("Invalid table"))? as usize;
    let d = rows[1]
        .get(1)
        .as_f64()
        .ok_or_else(|| JsError::new("Invalid table"))? as usize;

    let table_arr = [[a, b], [c, d]];

    let result = phi_coefficient(&table_arr).map_err(|e| JsError::new(&e.to_string()))?;

    let js_result = AssociationResultJs {
        estimate: result.estimate,
    };

    serde_wasm_bindgen::to_value(&js_result).map_err(|e| JsError::new(&e.to_string()))
}

/// Contingency coefficient.
///
/// @param observed - 2D array of observed counts
/// @returns Object with estimate
#[wasm_bindgen(js_name = contingencyCoefficient)]
pub fn js_contingency_coef(observed: js_sys::Array) -> Result<JsValue, JsError> {
    let table: Result<Vec<Vec<usize>>, _> = observed
        .iter()
        .map(|row| {
            let arr = js_sys::Array::from(&row);
            arr.iter()
                .map(|v| v.as_f64().map(|n| n as usize).ok_or(()))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect();

    let table = table.map_err(|_| JsError::new("Invalid contingency table"))?;

    let result = contingency_coef(&table).map_err(|e| JsError::new(&e.to_string()))?;

    let js_result = AssociationResultJs {
        estimate: result.estimate,
    };

    serde_wasm_bindgen::to_value(&js_result).map_err(|e| JsError::new(&e.to_string()))
}

#[derive(Serialize)]
struct KappaResultJs {
    kappa: f64,
    se: f64,
    z: f64,
    p_value: f64,
    conf_int_lower: f64,
    conf_int_upper: f64,
    weighted: bool,
}

/// Cohen's Kappa for inter-rater agreement.
///
/// @param table - 2D agreement matrix (square, same categories for both raters)
/// @param weighted - Use weighted kappa with linear weights (default: false)
/// @returns Object with kappa, se, z, p_value, conf_int_lower, conf_int_upper, weighted
#[wasm_bindgen(js_name = cohenKappa)]
pub fn js_cohen_kappa(table: js_sys::Array, weighted: Option<bool>) -> Result<JsValue, JsError> {
    let matrix: Result<Vec<Vec<usize>>, _> = table
        .iter()
        .map(|row| {
            let arr = js_sys::Array::from(&row);
            arr.iter()
                .map(|v| v.as_f64().map(|n| n as usize).ok_or(()))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect();

    let matrix = matrix.map_err(|_| JsError::new("Invalid agreement matrix"))?;

    let result = cohen_kappa(&matrix, weighted.unwrap_or(false))
        .map_err(|e| JsError::new(&e.to_string()))?;

    let js_result = KappaResultJs {
        kappa: result.kappa,
        se: result.se,
        z: result.z,
        p_value: result.p_value,
        conf_int_lower: result.conf_int_lower,
        conf_int_upper: result.conf_int_upper,
        weighted: result.weighted,
    };

    serde_wasm_bindgen::to_value(&js_result).map_err(|e| JsError::new(&e.to_string()))
}

#[derive(Serialize)]
struct McNemarkResultJs {
    statistic: f64,
    df: f64,
    p_value: f64,
}

/// McNemar's test for paired proportions.
///
/// @param table - 2x2 paired table as [[a, b], [c, d]]
/// @param correction - Apply continuity correction (default: true)
/// @returns Object with statistic, df, p_value
#[wasm_bindgen(js_name = mcnemarTest)]
pub fn js_mcnemar_test(table: js_sys::Array, correction: Option<bool>) -> Result<JsValue, JsError> {
    let rows: Vec<js_sys::Array> = table.iter().map(|r| js_sys::Array::from(&r)).collect();

    if rows.len() != 2 {
        return Err(JsError::new("McNemar's test requires a 2x2 table"));
    }

    let a = rows[0]
        .get(0)
        .as_f64()
        .ok_or_else(|| JsError::new("Invalid table"))? as usize;
    let b = rows[0]
        .get(1)
        .as_f64()
        .ok_or_else(|| JsError::new("Invalid table"))? as usize;
    let c = rows[1]
        .get(0)
        .as_f64()
        .ok_or_else(|| JsError::new("Invalid table"))? as usize;
    let d = rows[1]
        .get(1)
        .as_f64()
        .ok_or_else(|| JsError::new("Invalid table"))? as usize;

    let table_arr = [[a, b], [c, d]];

    let result = mcnemar_test(&table_arr, correction.unwrap_or(true))
        .map_err(|e| JsError::new(&e.to_string()))?;

    let js_result = McNemarkResultJs {
        statistic: result.statistic,
        df: result.df,
        p_value: result.p_value,
    };

    serde_wasm_bindgen::to_value(&js_result).map_err(|e| JsError::new(&e.to_string()))
}

#[derive(Serialize)]
struct McNemarkExactResultJs {
    p_value: f64,
    b: usize,
    c: usize,
}

/// McNemar's exact test for paired proportions (small samples).
///
/// @param table - 2x2 paired table as [[a, b], [c, d]]
/// @returns Object with p_value, b, c
#[wasm_bindgen(js_name = mcnemarExact)]
pub fn js_mcnemar_exact(table: js_sys::Array) -> Result<JsValue, JsError> {
    let rows: Vec<js_sys::Array> = table.iter().map(|r| js_sys::Array::from(&r)).collect();

    if rows.len() != 2 {
        return Err(JsError::new("McNemar's test requires a 2x2 table"));
    }

    let a = rows[0]
        .get(0)
        .as_f64()
        .ok_or_else(|| JsError::new("Invalid table"))? as usize;
    let b = rows[0]
        .get(1)
        .as_f64()
        .ok_or_else(|| JsError::new("Invalid table"))? as usize;
    let c = rows[1]
        .get(0)
        .as_f64()
        .ok_or_else(|| JsError::new("Invalid table"))? as usize;
    let d = rows[1]
        .get(1)
        .as_f64()
        .ok_or_else(|| JsError::new("Invalid table"))? as usize;

    let table_arr = [[a, b], [c, d]];

    let result = mcnemar_exact(&table_arr).map_err(|e| JsError::new(&e.to_string()))?;

    let js_result = McNemarkExactResultJs {
        p_value: result.p_value,
        b: result.b,
        c: result.c,
    };

    serde_wasm_bindgen::to_value(&js_result).map_err(|e| JsError::new(&e.to_string()))
}

#[derive(Serialize)]
struct BinomTestResultJs {
    p_value: f64,
    estimate: f64,
    conf_int_lower: f64,
    conf_int_upper: f64,
}

/// Exact binomial test.
///
/// @param successes - Number of successes
/// @param trials - Number of trials
/// @param p0 - Hypothesized probability (default: 0.5)
/// @param alternative - Alternative hypothesis direction
/// @returns Object with p_value, estimate, conf_int_lower, conf_int_upper
#[wasm_bindgen(js_name = binomTest)]
pub fn js_binom_test(
    successes: usize,
    trials: usize,
    p0: Option<f64>,
    alternative: JsAlternative,
) -> Result<JsValue, JsError> {
    let alt: Alternative = match alternative {
        JsAlternative::TwoSided => Alternative::TwoSided,
        JsAlternative::Less => Alternative::Less,
        JsAlternative::Greater => Alternative::Greater,
    };

    let result = binom_test(successes, trials, p0.unwrap_or(0.5), alt)
        .map_err(|e| JsError::new(&e.to_string()))?;

    let js_result = BinomTestResultJs {
        p_value: result.p_value,
        estimate: result.estimate,
        conf_int_lower: result.conf_int_lower,
        conf_int_upper: result.conf_int_upper,
    };

    serde_wasm_bindgen::to_value(&js_result).map_err(|e| JsError::new(&e.to_string()))
}

#[derive(Serialize)]
struct PropTestResultJs {
    statistic: f64,
    p_value: f64,
    estimate: f64,
    conf_int_lower: f64,
    conf_int_upper: f64,
}

/// One-sample proportion test.
///
/// @param successes - Number of successes
/// @param trials - Number of trials
/// @param p0 - Hypothesized probability
/// @param alternative - Alternative hypothesis direction
/// @returns Object with statistic, p_value, estimate, conf_int_lower, conf_int_upper
#[wasm_bindgen(js_name = propTestOne)]
pub fn js_prop_test_one(
    successes: usize,
    trials: usize,
    p0: f64,
    alternative: JsAlternative,
) -> Result<JsValue, JsError> {
    let alt: Alternative = match alternative {
        JsAlternative::TwoSided => Alternative::TwoSided,
        JsAlternative::Less => Alternative::Less,
        JsAlternative::Greater => Alternative::Greater,
    };

    let result =
        prop_test_one(successes, trials, p0, alt).map_err(|e| JsError::new(&e.to_string()))?;

    let js_result = PropTestResultJs {
        statistic: result.statistic,
        p_value: result.p_value,
        estimate: result.estimate[0],
        conf_int_lower: result.conf_int_lower,
        conf_int_upper: result.conf_int_upper,
    };

    serde_wasm_bindgen::to_value(&js_result).map_err(|e| JsError::new(&e.to_string()))
}

#[derive(Serialize)]
struct PropTestTwoResultJs {
    statistic: f64,
    p_value: f64,
    estimate1: f64,
    estimate2: f64,
    conf_int_lower: f64,
    conf_int_upper: f64,
}

/// Two-sample proportion test.
///
/// @param successes1 - Number of successes in group 1
/// @param trials1 - Number of trials in group 1
/// @param successes2 - Number of successes in group 2
/// @param trials2 - Number of trials in group 2
/// @param alternative - Alternative hypothesis direction
/// @param correction - Apply Yates' continuity correction (default: false)
/// @returns Object with statistic, p_value, estimates, conf_int_lower, conf_int_upper
#[wasm_bindgen(js_name = propTestTwo)]
pub fn js_prop_test_two(
    successes1: usize,
    trials1: usize,
    successes2: usize,
    trials2: usize,
    alternative: JsAlternative,
    correction: Option<bool>,
) -> Result<JsValue, JsError> {
    let alt: Alternative = match alternative {
        JsAlternative::TwoSided => Alternative::TwoSided,
        JsAlternative::Less => Alternative::Less,
        JsAlternative::Greater => Alternative::Greater,
    };

    let result = prop_test_two(
        [successes1, successes2],
        [trials1, trials2],
        alt,
        correction.unwrap_or(false),
    )
    .map_err(|e| JsError::new(&e.to_string()))?;

    let js_result = PropTestTwoResultJs {
        statistic: result.statistic,
        p_value: result.p_value,
        estimate1: result.estimate[0],
        estimate2: result.estimate[1],
        conf_int_lower: result.conf_int_lower,
        conf_int_upper: result.conf_int_upper,
    };

    serde_wasm_bindgen::to_value(&js_result).map_err(|e| JsError::new(&e.to_string()))
}
