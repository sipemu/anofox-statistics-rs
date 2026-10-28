pub mod energy;
pub mod mmd;
pub mod wasserstein;

pub use energy::{energy_distance_test, energy_distance_test_1d, EnergyDistanceResult};
pub use mmd::{mmd_test, mmd_test_1d, Kernel, MMDResult};
pub use wasserstein::{wasserstein_distance_1d, wasserstein_test_1d, WassersteinResult};
