//! WASM integration tests for anofox-statistics-js

#![cfg(target_arch = "wasm32")]

use wasm_bindgen_test::*;

use anofox_statistics_js::*;

#[wasm_bindgen_test]
fn test_version() {
    let version = get_version();
    assert!(!version.is_empty());
    assert!(version.starts_with("0."));
}

#[wasm_bindgen_test]
fn test_t_test() {
    let x = js_sys::Float64Array::new_with_length(5);
    x.copy_from(&[1.0, 2.0, 3.0, 4.0, 5.0]);

    let y = js_sys::Float64Array::new_with_length(5);
    y.copy_from(&[2.0, 3.0, 4.0, 5.0, 6.0]);

    let result = js_t_test(
        x.to_vec().as_slice(),
        y.to_vec().as_slice(),
        JsTTestKind::Welch,
        JsAlternative::TwoSided,
        Some(0.0),
        Some(0.95),
    );

    assert!(result.is_ok());
}

#[wasm_bindgen_test]
fn test_shapiro_wilk() {
    let data = js_sys::Float64Array::new_with_length(10);
    data.copy_from(&[2.3, 3.1, 2.8, 3.5, 2.9, 3.2, 2.7, 3.0, 2.6, 3.3]);

    let result = js_shapiro_wilk(data.to_vec().as_slice());
    assert!(result.is_ok());
}

#[wasm_bindgen_test]
fn test_mann_whitney() {
    let x = js_sys::Float64Array::new_with_length(5);
    x.copy_from(&[1.0, 2.0, 3.0, 4.0, 5.0]);

    let y = js_sys::Float64Array::new_with_length(5);
    y.copy_from(&[3.0, 4.0, 5.0, 6.0, 7.0]);

    let result = js_mann_whitney_u(
        x.to_vec().as_slice(),
        y.to_vec().as_slice(),
        JsAlternative::TwoSided,
        Some(true),
        Some(false),
        None,
        Some(0.0),
    );

    assert!(result.is_ok());
}

#[wasm_bindgen_test]
fn test_pearson_correlation() {
    let x = js_sys::Float64Array::new_with_length(5);
    x.copy_from(&[1.0, 2.0, 3.0, 4.0, 5.0]);

    let y = js_sys::Float64Array::new_with_length(5);
    y.copy_from(&[2.0, 4.0, 6.0, 8.0, 10.0]);

    let result = js_pearson(x.to_vec().as_slice(), y.to_vec().as_slice(), Some(0.95));
    assert!(result.is_ok());
}
