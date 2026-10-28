use crate::error::{Result, StatError};
use crate::utils::finite::ensure_no_nan;

/// Calculate the arithmetic mean of a slice.
/// Returns `StatError::EmptyData` if the slice is empty.
pub fn mean(data: &[f64]) -> Result<f64> {
    if data.is_empty() {
        return Err(StatError::EmptyData);
    }
    let sum: f64 = data.iter().sum();
    Ok(sum / data.len() as f64)
}

/// Calculate the arithmetic mean using Welford's online algorithm.
///
/// This implementation uses B.P. Welford's numerically stable algorithm
/// (Technometrics, 1962) which incrementally updates the mean to avoid
/// precision loss when summing large values or many small values.
///
/// Prefer this over `mean()` when:
/// - Data values have large magnitude with small relative differences
/// - Processing streaming/online data
/// - Numerical stability is critical
///
/// Returns `StatError::EmptyData` if the slice is empty.
pub fn stable_mean(data: &[f64]) -> Result<f64> {
    if data.is_empty() {
        return Err(StatError::EmptyData);
    }
    let sum: f64 = data.iter().enumerate().fold(0_f64, |mean_km1, (k, xk)| {
        mean_km1 + (xk - mean_km1) / (k + 1) as f64
    });
    Ok(sum)
}

/// Calculate the sample variance (using n-1 denominator, matching R's `var()`).
/// Returns `StatError::InsufficientData` if fewer than 2 elements.
pub fn variance(data: &[f64]) -> Result<f64> {
    let n = data.len();
    if n < 2 {
        return Err(StatError::InsufficientData { needed: 2, got: n });
    }

    let mean_val = mean(data)?;
    let sum_sq: f64 = data.iter().map(|x| (x - mean_val).powi(2)).sum();
    Ok(sum_sq / (n - 1) as f64)
}

/// Calculate the sample variance using Welford's online algorithm.
///
/// This implementation uses B.P. Welford's numerically stable algorithm
/// (Technometrics, 1962) which computes mean and variance in a single pass
/// while avoiding catastrophic cancellation.
///
/// The algorithm maintains a running mean and sum of squared deviations,
/// updating both incrementally. This is more stable than the two-pass
/// algorithm when variance is small relative to the mean magnitude.
///
/// Prefer this over `variance()` when:
/// - Data values have large magnitude with small variance
/// - Processing streaming/online data
/// - Numerical stability is critical
///
/// Returns `StatError::InsufficientData` if fewer than 2 elements.
pub fn stable_variance(data: &[f64]) -> Result<f64> {
    let n = data.len();
    if n < 2 {
        return Err(StatError::InsufficientData { needed: 2, got: n });
    }

    let (_mean, snd_moment): (f64, f64) =
        data.iter()
            .enumerate()
            .fold((0_f64, 0_f64), |(mean, snd_moment), (count, xk)| {
                let delta = xk - mean;
                let mean = mean + delta / (count + 1) as f64;
                let delta2 = xk - mean;
                let snd_moment = snd_moment + delta * delta2;
                (mean, snd_moment)
            });
    Ok(snd_moment / (n as f64 - 1.0_f64))
}
/// Calculate the median of a slice.
/// Returns `StatError::EmptyData` if the slice is empty.
pub fn median(data: &[f64]) -> Result<f64> {
    if data.is_empty() {
        return Err(StatError::EmptyData);
    }

    ensure_no_nan("data", data)?;
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));

    let n = sorted.len();
    if n % 2 == 1 {
        // Odd length: return middle element
        Ok(sorted[n / 2])
    } else {
        // Even length: return average of two middle elements
        Ok((sorted[n / 2 - 1] + sorted[n / 2]) / 2.0)
    }
}

/// Calculate the trimmed mean, removing `trim` proportion from each tail.
/// `trim` should be in [0, 0.5).
/// Returns `StatError::InvalidParameter` if trim is out of range.
/// Returns `StatError::EmptyData` if the slice is empty after trimming.
pub fn trimmed_mean(data: &[f64], trim: f64) -> Result<f64> {
    if !(0.0..0.5).contains(&trim) {
        return Err(StatError::InvalidParameter(format!(
            "trim must be in [0, 0.5), got {}",
            trim
        )));
    }

    if data.is_empty() {
        return Err(StatError::EmptyData);
    }

    ensure_no_nan("data", data)?;
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));

    let n = sorted.len();
    let k = (n as f64 * trim).floor() as usize;

    // Trim k elements from each end
    let trimmed = &sorted[k..n - k];

    if trimmed.is_empty() {
        return Err(StatError::EmptyData);
    }

    mean(trimmed)
}

/// Calculate the sample standard deviation (using n-1 denominator, matching R's `sd()`).
/// Returns `StatError::InsufficientData` if fewer than 2 elements.
pub fn std_dev(data: &[f64]) -> Result<f64> {
    Ok(variance(data)?.sqrt())
}

/// Calculate the sample skewness (Fisher's definition, matching R's `e1071::skewness(type=2)`).
/// Uses the formula from e1071:
///   y = sqrt(n) * sum(x^3) / sum(x^2)^1.5  (type 1)
///   type 2: y * sqrt(n*(n-1)) / (n-2)
/// Returns `StatError::InsufficientData` if fewer than 3 elements.
pub fn skewness(data: &[f64]) -> Result<f64> {
    let n = data.len();
    if n < 3 {
        return Err(StatError::InsufficientData { needed: 3, got: n });
    }

    let mean_val = mean(data)?;
    let n_f = n as f64;

    // Sum of squared deviations and cubed deviations
    let ss2: f64 = data.iter().map(|x| (x - mean_val).powi(2)).sum();
    let ss3: f64 = data.iter().map(|x| (x - mean_val).powi(3)).sum();

    if ss2 < 1e-28 {
        // Constant data has zero skewness
        return Ok(0.0);
    }

    // Type 1: y = sqrt(n) * sum(x^3) / sum(x^2)^1.5
    let y = n_f.sqrt() * ss3 / ss2.powf(1.5);

    // Type 2: y * sqrt(n*(n-1)) / (n-2)
    let g1 = y * (n_f * (n_f - 1.0)).sqrt() / (n_f - 2.0);

    Ok(g1)
}

/// Calculate the sample excess kurtosis (Fisher's definition, matching R's `e1071::kurtosis(type=2)`).
/// Uses the formula from e1071:
///   r = n * sum(x^4) / sum(x^2)^2
///   type 2: ((n+1)*(r-3) + 6) * (n-1) / ((n-2)*(n-3))
/// Returns `StatError::InsufficientData` if fewer than 4 elements.
pub fn kurtosis(data: &[f64]) -> Result<f64> {
    let n = data.len();
    if n < 4 {
        return Err(StatError::InsufficientData { needed: 4, got: n });
    }

    let mean_val = mean(data)?;
    let n_f = n as f64;

    // Sum of squared and fourth-power deviations
    let ss2: f64 = data.iter().map(|x| (x - mean_val).powi(2)).sum();
    let ss4: f64 = data.iter().map(|x| (x - mean_val).powi(4)).sum();

    if ss2 < 1e-28 {
        // Constant data - kurtosis is undefined but we return 0
        return Ok(0.0);
    }

    // r = n * sum(x^4) / sum(x^2)^2 (this is the raw kurtosis ratio)
    let r = n_f * ss4 / (ss2 * ss2);

    // Type 2: ((n+1)*(r-3) + 6) * (n-1) / ((n-2)*(n-3))
    let g2 = ((n_f + 1.0) * (r - 3.0) + 6.0) * (n_f - 1.0) / ((n_f - 2.0) * (n_f - 3.0));

    Ok(g2)
}

/// Eigen-decomposition of a small symmetric matrix by cyclic Jacobi
/// rotations. Returns the eigenvalues and the eigenvectors as the columns of
/// the second matrix (`vectors[i][j]` is component `i` of eigenvector `j`).
/// Only the upper triangle of `a` is read.
#[allow(clippy::needless_range_loop)]
pub(crate) fn symmetric_eigen(a: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = a.len();
    let mut m: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| if i <= j { a[i][j] } else { a[j][i] })
                .collect()
        })
        .collect();
    let mut v: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect();
    for _sweep in 0..100 {
        let off: f64 = (0..n)
            .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
            .map(|(i, j)| m[i][j] * m[i][j])
            .sum();
        let diag: f64 = (0..n).map(|i| m[i][i] * m[i][i]).sum();
        if off == 0.0 || off <= 1e-36 * diag {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                if m[p][q] == 0.0 {
                    continue;
                }
                // Rotation that zeroes m[p][q] (Golub and Van Loan, 8.4.2).
                let theta = (m[q][q] - m[p][p]) / (2.0 * m[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (mkp, mkq) = (m[k][p], m[k][q]);
                    m[k][p] = c * mkp - s * mkq;
                    m[k][q] = s * mkp + c * mkq;
                }
                for k in 0..n {
                    let (mpk, mqk) = (m[p][k], m[q][k]);
                    m[p][k] = c * mpk - s * mqk;
                    m[q][k] = s * mpk + c * mqk;
                }
                for k in 0..n {
                    let (vkp, vkq) = (v[k][p], v[k][q]);
                    v[k][p] = c * vkp - s * vkq;
                    v[k][q] = s * vkp + c * vkq;
                }
            }
        }
    }
    ((0..n).map(|i| m[i][i]).collect(), v)
}

#[cfg(test)]
mod eigen_tests {
    use super::symmetric_eigen;

    #[test]
    fn reconstructs_matrix() {
        let a = vec![
            vec![4.0, 1.0, -2.0, 0.5],
            vec![1.0, 3.0, 0.0, 1.0],
            vec![-2.0, 0.0, 5.0, -1.0],
            vec![0.5, 1.0, -1.0, 2.0],
        ];
        let (w, v) = symmetric_eigen(&a);
        for i in 0..4 {
            for j in 0..4 {
                let r: f64 = (0..4).map(|k| v[i][k] * w[k] * v[j][k]).sum();
                assert!((r - a[i][j]).abs() < 1e-13);
            }
        }
        // trace is preserved
        assert!((w.iter().sum::<f64>() - 14.0).abs() < 1e-13);
        // rank-one matrix: one non-zero eigenvalue
        let b = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        let (w, _) = symmetric_eigen(&b);
        let mut w = w;
        w.sort_by(|a, b| a.total_cmp(b));
        assert!(w[0].abs() < 1e-15 && (w[1] - 5.0).abs() < 1e-14);
    }
}
