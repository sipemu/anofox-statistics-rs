pub mod dagostino;
pub mod jarque_bera;
pub mod normality;

pub use dagostino::{dagostino_k_squared, DAgostinoResult};
pub use jarque_bera::{jarque_bera, JarqueBeraResult};
pub use normality::{shapiro_wilk, ShapiroWilkResult};
