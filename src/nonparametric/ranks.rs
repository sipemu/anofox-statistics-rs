use crate::error::{Result, StatError};

/// Compute ranks of data with average tie handling (matching R's rank(ties.method="average")).
///
/// # Arguments
/// * `data` - The data to rank
///
/// # Returns
/// * Vector of ranks (1-indexed, ties get average rank)
pub fn rank(data: &[f64]) -> Result<Vec<f64>> {
    if data.is_empty() {
        return Err(StatError::EmptyData);
    }

    let n = data.len();

    // Create index-value pairs and sort by value
    let mut indexed: Vec<(usize, f64)> = data.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let mut ranks = vec![0.0; n];

    // Assign ranks, handling ties by averaging
    let mut i = 0;
    while i < n {
        let mut j = i;
        // Find all elements with the same value (ties)
        while j < n && indexed[j].1 == indexed[i].1 {
            j += 1;
        }

        // Average rank for ties: (start_rank + end_rank) / 2
        // Ranks are 1-indexed: positions i..j get ranks (i+1)..(j+1)
        let avg_rank = (i + 1 + j) as f64 / 2.0;

        // Assign the average rank to all tied elements
        for item in indexed.iter().take(j).skip(i) {
            ranks[item.0] = avg_rank;
        }

        i = j;
    }

    Ok(ranks)
}

/// Internal helper: compute ranks and return tie information for correction
pub(crate) fn rank_with_ties(data: &[f64]) -> Result<(Vec<f64>, Vec<usize>)> {
    if data.is_empty() {
        return Err(StatError::EmptyData);
    }

    let n = data.len();

    // Create index-value pairs and sort by value
    let mut indexed: Vec<(usize, f64)> = data.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let mut ranks = vec![0.0; n];
    let mut tie_sizes = Vec::new();

    // Assign ranks, handling ties by averaging
    let mut i = 0;
    while i < n {
        let mut j = i;
        // Find all elements with the same value (ties)
        while j < n && indexed[j].1 == indexed[i].1 {
            j += 1;
        }

        let tie_size = j - i;
        if tie_size > 1 {
            tie_sizes.push(tie_size);
        }

        // Average rank for ties
        let avg_rank = (i + 1 + j) as f64 / 2.0;

        // Assign the average rank to all tied elements
        for item in indexed.iter().take(j).skip(i) {
            ranks[item.0] = avg_rank;
        }

        i = j;
    }

    Ok((ranks, tie_sizes))
}
