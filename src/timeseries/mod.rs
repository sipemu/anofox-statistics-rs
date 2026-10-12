//! Time-series and residual diagnostics: sample autocorrelation (ACF),
//! partial autocorrelation (PACF), the Ljung-Box and Box-Pierce portmanteau
//! tests and the Durbin-Watson statistic. Results match R's `acf`, `pacf`,
//! `Box.test` and `lmtest::dwtest` (statistic).

mod autocorrelation;
mod portmanteau;

pub use autocorrelation::{acf, durbin_watson, pacf, AcfResult};
pub use portmanteau::{box_pierce, ljung_box, PortmanteauResult};
