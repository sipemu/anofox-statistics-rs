mod categorical;
mod correlation;
mod distributional;
mod equivalence;
mod forecast;
mod modern;
mod nonparametric;
mod parametric;
mod resampling;

pub use categorical::*;
pub use correlation::*;
pub use distributional::*;
pub use equivalence::*;
pub use forecast::*;
pub use modern::*;
pub use nonparametric::*;
pub use parametric::*;
pub use resampling::*;

use wasm_bindgen::prelude::*;

#[wasm_bindgen(start)]
pub fn start() {
    // Initialize WASM module
}

#[wasm_bindgen(js_name = getVersion)]
pub fn get_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
