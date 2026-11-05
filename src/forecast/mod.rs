mod clark_west;
mod diebold_mariano;
mod mcs;
mod mspe_adjusted;
mod spa;
mod spa_common;

#[cfg(test)]
mod reference_tests;

pub use clark_west::{clark_west, CWResult};
pub use diebold_mariano::{diebold_mariano, DMResult, LossFunction, VarEstimator};
pub use mcs::{
    model_confidence_set, model_confidence_set_with_method, MCSEliminationStep, MCSResult,
    MCSStatistic,
};
pub use mspe_adjusted::{mspe_adjusted_spa, MSPEAdjustedResult};
pub use spa::{spa_test, SPAResult};
