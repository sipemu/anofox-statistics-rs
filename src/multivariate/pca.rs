//! Principal component analysis (R's `prcomp`, scikit-learn's `PCA`).

use crate::error::{Result, StatError};
use crate::utils::finite::ensure_finite;
use crate::utils::math::symmetric_eigen;

/// A principal component analysis fitted by [`pca`].
#[derive(Debug, Clone, PartialEq)]
pub struct Pca {
    /// Column means of the data the analysis was fitted on.
    pub center: Vec<f64>,
    /// Standard deviations of the components (divisor `n - 1`), decreasing.
    pub sdev: Vec<f64>,
    /// Loadings: `rotation[i][j]` is variable `i` of component `j`. Each
    /// component is a unit vector whose entry of largest absolute value is
    /// positive.
    pub rotation: Vec<Vec<f64>>,
    /// Share of the total variance per component.
    pub explained_variance_ratio: Vec<f64>,
}

impl Pca {
    /// Number of components.
    pub fn n_components(&self) -> usize {
        self.sdev.len()
    }

    /// Scores of `data` on the first `n_components` components: the rows,
    /// centred by the fitted [`center`](Pca::center), times the loadings.
    ///
    /// Errors when `n_components` is 0 or exceeds the fitted components, or
    /// when a row has the wrong length or non-finite values.
    ///
    /// R equivalent: `predict(p, newdata)[, 1:n_components]`.
    pub fn transform(&self, data: &[Vec<f64>], n_components: usize) -> Result<Vec<Vec<f64>>> {
        if n_components == 0 || n_components > self.n_components() {
            return Err(StatError::InvalidParameter(format!(
                "n_components must be in 1..={}, got {}",
                self.n_components(),
                n_components
            )));
        }
        let d = self.center.len();
        data.iter()
            .enumerate()
            .map(|(r, row)| {
                if row.len() != d {
                    return Err(StatError::InvalidParameter(format!(
                        "row {} has {} values, expected {}",
                        r,
                        row.len(),
                        d
                    )));
                }
                ensure_finite(&format!("row {}", r), row)?;
                Ok((0..n_components)
                    .map(|j| {
                        (0..d)
                            .map(|i| (row[i] - self.center[i]) * self.rotation[i][j])
                            .sum()
                    })
                    .collect())
            })
            .collect()
    }
}

/// Principal component analysis of the rows of `data` (observations x
/// variables), centred but not scaled.
///
/// The components are the eigenvectors of the sample covariance matrix
/// (divisor `n - 1`), ordered by decreasing variance. With `n` rows and `d`
/// variables there are `min(n - 1, d)` components; with `n <= d` they come
/// from the `n x n` Gram matrix of the centred rows, which is cheaper for
/// embeddings with fewer rows than dimensions. Each component's sign is
/// fixed so that its entry of largest absolute value is positive
/// (scikit-learn's `svd_flip` on the components); R's `prcomp` signs are
/// arbitrary.
///
/// evidently's embedding drift methods reduce both samples with
/// `PCA(k).fit_transform(reference)` and `.transform(current)`, which is
/// `pca(reference)` followed by [`Pca::transform`] of each sample.
///
/// The eigendecomposition uses cyclic Jacobi rotations: `O(n d min(n, d) +
/// min(n, d)^3)` time per sweep, intended for up to a few hundred
/// dimensions or rows.
///
/// Errors on fewer than two rows, rows of unequal or zero length and
/// non-finite values.
///
/// R equivalent: `prcomp(data, center = TRUE, scale. = FALSE)`.
///
/// # Example
/// ```
/// use anofox_statistics::multivariate::pca;
/// let data = vec![vec![1.0, 1.0], vec![2.0, 2.0], vec![3.0, 3.0]];
/// let p = pca(&data).unwrap();
/// assert!((p.sdev[0] - 2f64.sqrt()).abs() < 1e-12);
/// assert!((p.explained_variance_ratio[0] - 1.0).abs() < 1e-12);
/// let scores = p.transform(&data, 1).unwrap();
/// assert!((scores[2][0] - 2f64.sqrt()).abs() < 1e-12);
/// ```
pub fn pca(data: &[Vec<f64>]) -> Result<Pca> {
    let n = data.len();
    if n < 2 {
        return Err(StatError::InsufficientData { needed: 2, got: n });
    }
    let d = data[0].len();
    if d == 0 {
        return Err(StatError::InvalidParameter(
            "rows must have at least one value".into(),
        ));
    }
    for (r, row) in data.iter().enumerate() {
        if row.len() != d {
            return Err(StatError::InvalidParameter(format!(
                "row {} has {} values, expected {}",
                r,
                row.len(),
                d
            )));
        }
        ensure_finite(&format!("row {}", r), row)?;
    }
    let center: Vec<f64> = (0..d)
        .map(|i| data.iter().map(|row| row[i]).sum::<f64>() / n as f64)
        .collect();
    let xc: Vec<Vec<f64>> = data
        .iter()
        .map(|row| row.iter().zip(&center).map(|(v, c)| v - c).collect())
        .collect();
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    let k = (n - 1).min(d);

    // (eigenvalue of Xc' Xc, unit loading vector) pairs.
    let mut comps: Vec<(f64, Vec<f64>)> = if d < n {
        let mut s = vec![vec![0.0; d]; d];
        for row in &xc {
            for i in 0..d {
                for j in i..d {
                    s[i][j] += row[i] * row[j];
                }
            }
        }
        let (vals, vecs) = symmetric_eigen(&s);
        (0..d)
            .map(|j| (vals[j], (0..d).map(|i| vecs[i][j]).collect()))
            .collect()
    } else {
        let g: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| dot(&xc[i], &xc[j])).collect())
            .collect();
        let (vals, vecs) = symmetric_eigen(&g);
        (0..n)
            .map(|j| {
                // v = Xc' u / |Xc' u|
                let mut v: Vec<f64> = (0..d)
                    .map(|i| (0..n).map(|r| xc[r][i] * vecs[r][j]).sum())
                    .collect();
                let norm = dot(&v, &v).sqrt();
                if norm > 0.0 {
                    v.iter_mut().for_each(|x| *x /= norm);
                }
                (vals[j], v)
            })
            .collect()
    };
    comps.sort_by(|a, b| b.0.total_cmp(&a.0));
    comps.truncate(k);

    let total: f64 = xc.iter().map(|row| dot(row, row)).sum();
    let mut rotation = vec![vec![0.0; k]; d];
    let mut sdev = Vec::with_capacity(k);
    let mut ratio = Vec::with_capacity(k);
    for (j, (val, mut v)) in comps.into_iter().enumerate() {
        let val = val.max(0.0);
        let big = v
            .iter()
            .copied()
            .fold(0.0f64, |acc, x| if x.abs() > acc.abs() { x } else { acc });
        if big < 0.0 {
            v.iter_mut().for_each(|x| *x = -*x);
        }
        for i in 0..d {
            rotation[i][j] = v[i];
        }
        sdev.push((val / (n - 1) as f64).sqrt());
        ratio.push(if total > 0.0 { val / total } else { 0.0 });
    }
    Ok(Pca {
        center,
        sdev,
        rotation,
        explained_variance_ratio: ratio,
    })
}
