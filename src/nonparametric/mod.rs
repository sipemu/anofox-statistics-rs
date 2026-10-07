pub mod brunner_munzel;
pub mod kruskal;
pub mod ranks;
pub mod wilcoxon;

pub use brunner_munzel::{brunner_munzel, BrunnerMunzelResult};
pub use kruskal::{kruskal_wallis, KruskalResult};
pub use ranks::rank;
pub use wilcoxon::{
    mann_whitney_u, matched_pairs_rank_biserial, rank_biserial, rank_biserial_from_u,
    wilcoxon_signed_rank, ConfidenceInterval, MannWhitneyResult, WilcoxonResult,
};
