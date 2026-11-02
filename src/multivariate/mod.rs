//! Multivariate methods: principal component analysis ([`pca`]), used to
//! reduce embeddings before the embedding drift measures of
//! [`drift`](crate::drift).

mod pca;

pub use pca::{pca, Pca};
