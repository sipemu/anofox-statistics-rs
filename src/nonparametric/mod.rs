pub mod kruskal;
pub mod ranks;
pub mod wilcoxon;

pub use kruskal::{kruskal_wallis, KruskalResult};
pub use ranks::rank;
pub use wilcoxon::{mann_whitney_u, wilcoxon_signed_rank, MannWhitneyResult, WilcoxonResult};
