pub mod distributional;
pub mod error;
pub mod forecast;
pub mod modern;
pub mod nonparametric;
pub mod parametric;
pub mod resampling;
pub mod utils;

pub use distributional::{dagostino_k_squared, shapiro_wilk, DAgostinoResult, ShapiroWilkResult};
pub use error::{Result, StatError};
pub use forecast::{
    clark_west, diebold_mariano, model_confidence_set, mspe_adjusted_spa, spa_test, CWResult,
    DMResult, LossFunction, MCSEliminationStep, MCSResult, MCSStatistic, MSPEAdjustedResult,
    SPAResult,
};
pub use modern::{
    energy_distance_test, energy_distance_test_1d, mmd_test, mmd_test_1d, EnergyDistanceResult,
    Kernel, MMDResult,
};
pub use nonparametric::{
    brunner_munzel, kruskal_wallis, mann_whitney_u, rank, wilcoxon_signed_rank,
    BrunnerMunzelResult, KruskalResult, MannWhitneyResult, WilcoxonResult,
};
pub use parametric::{
    brown_forsythe, t_test, yuen_test, Alternative, LeveneResult, TTestKind, TTestResult,
    YuenResult,
};
pub use resampling::{
    permutation_t_test, CircularBlockBootstrap, PermutationEngine, PermutationResult,
    StationaryBootstrap,
};
