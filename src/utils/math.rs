use crate::error::{Result, StatError};

/// Calculate the arithmetic mean of a slice.
/// Returns `StatError::EmptyData` if the slice is empty.
pub fn mean(data: &[f64]) -> Result<f64> {
    if data.is_empty() {
        return Err(StatError::EmptyData);
    }
    let sum: f64 = data.iter().sum();
    Ok(sum / data.len() as f64)
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

/// Calculate the median of a slice.
/// Returns `StatError::EmptyData` if the slice is empty.
pub fn median(data: &[f64]) -> Result<f64> {
    if data.is_empty() {
        return Err(StatError::EmptyData);
    }

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

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

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = sorted.len();
    let k = (n as f64 * trim).floor() as usize;

    // Trim k elements from each end
    let trimmed = &sorted[k..n - k];

    if trimmed.is_empty() {
        return Err(StatError::EmptyData);
    }

    mean(trimmed)
}
