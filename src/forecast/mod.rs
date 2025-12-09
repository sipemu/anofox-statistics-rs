mod clark_west;
mod diebold_mariano;
mod mcs;
mod mspe_adjusted;
mod spa;

pub use clark_west::{clark_west, CWResult};
pub use diebold_mariano::{diebold_mariano, DMResult, LossFunction};
pub use mcs::{model_confidence_set, MCSEliminationStep, MCSResult, MCSStatistic};
pub use mspe_adjusted::{mspe_adjusted_spa, MSPEAdjustedResult};
pub use spa::{spa_test, SPAResult};
