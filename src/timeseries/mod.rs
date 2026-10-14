//! Time-series and residual diagnostics: sample autocorrelation (ACF),
//! partial autocorrelation (PACF), the Ljung-Box and Box-Pierce portmanteau
//! tests and the Durbin-Watson statistic and test. Results match R's `acf`,
//! `pacf`, `Box.test` and `lmtest::dwtest`.

mod autocorrelation;
mod dwtest;
mod portmanteau;

pub use autocorrelation::{acf, durbin_watson, pacf, AcfResult};
pub use dwtest::durbin_watson_test;
pub use portmanteau::{box_pierce, ljung_box, PortmanteauResult};
