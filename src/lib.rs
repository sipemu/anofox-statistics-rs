pub mod distributional;
pub mod error;
pub mod forecast;
pub mod nonparametric;
pub mod parametric;
pub mod utils;

pub use distributional::{shapiro_wilk, ShapiroWilkResult};
pub use error::{Result, StatError};
pub use forecast::{diebold_mariano, DMResult, LossFunction};
pub use nonparametric::{
    kruskal_wallis, mann_whitney_u, rank, wilcoxon_signed_rank, KruskalResult, MannWhitneyResult,
    WilcoxonResult,
};
pub use parametric::{
    brown_forsythe, t_test, yuen_test, Alternative, LeveneResult, TTestKind, TTestResult,
    YuenResult,
};
