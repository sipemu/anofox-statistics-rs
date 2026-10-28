pub mod anderson_darling;
mod anderson_darling_table;
pub mod brunner_munzel;
pub mod cramer_von_mises;
mod hodges_lehmann;
pub mod kruskal;
pub mod ks;
pub mod ranks;
pub mod wilcoxon;
mod wilcoxon_dist;

pub use anderson_darling::{ad_test_ksamp, AndersonDarlingKResult};
pub use brunner_munzel::{brunner_munzel, BrunnerMunzelResult};
pub use cramer_von_mises::{cvm_test_2sample, CvmMethod, CvmTestResult};
pub use kruskal::{kruskal_wallis, KruskalResult};
pub use ks::{ks_test_2sample, KsTestResult};
pub use ranks::rank;
pub use wilcoxon::{
    mann_whitney_u, matched_pairs_rank_biserial, rank_biserial, rank_biserial_from_u,
    wilcoxon_signed_rank, ConfidenceInterval, MannWhitneyResult, WilcoxonResult,
};
