//! Kendall's tau correlation coefficient.

use crate::correlation::{validate_correlation_input, CorrelationMethod, CorrelationResult};
use crate::error::Result;
use statrs::distribution::{ContinuousCDF, Normal};

/// Variant of Kendall's tau to compute
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KendallVariant {
    /// Tau-a: No tie adjustment (simple ratio of concordant-discordant pairs)
    TauA,
    /// Tau-b: Adjusted for ties (default, matches R's cor.test)
    #[default]
    TauB,
    /// Tau-c: Stuart's tau-c for rectangular tables (adjusts for table size)
    TauC,
}

/// Compute Kendall's tau correlation coefficient with significance test.
///
/// Kendall's tau measures the strength of association between two variables
/// based on concordant and discordant pairs. It's particularly useful for
/// small samples and ordinal data.
///
/// # Variants
/// - `TauA`: Simple ratio, no tie adjustment: (C - D) / (n*(n-1)/2)
/// - `TauB`: Tie-adjusted (default, matches R): (C - D) / sqrt((C+D+Tx)(C+D+Ty))
/// - `TauC`: Stuart's tau-c: 2(C - D) / (n² * (m-1)/m) where m = min(rows, cols)
///
/// # Arguments
/// * `x` - First variable (must have at least 3 observations)
/// * `y` - Second variable (same length as x)
/// * `variant` - Which variant of tau to compute
///
/// # Returns
/// * `CorrelationResult` containing the correlation coefficient, z-statistic,
///   p-value (using normal approximation)
///
/// # Examples
/// ```
/// use anofox_statistics::correlation::{kendall, KendallVariant};
///
/// let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = vec![5.0, 6.0, 7.0, 8.0, 7.0];
///
/// let result = kendall(&x, &y, KendallVariant::TauB).unwrap();
/// println!("Kendall tau = {:.4}", result.estimate);
/// println!("p-value = {:.4}", result.p_value);
/// ```
///
/// # R equivalent
/// `cor.test(x, y, method = "kendall")` (uses tau-b)
pub fn kendall(x: &[f64], y: &[f64], variant: KendallVariant) -> Result<CorrelationResult> {
    let n = validate_correlation_input(x, y)?;

    // Count concordant, discordant, and tied pairs
    let (concordant, discordant, ties_x, ties_y, _ties_xy) = count_pairs(x, y);

    // Total number of pairs
    let n_pairs = (n * (n - 1)) / 2;

    // Compute tau based on variant
    let tau = match variant {
        KendallVariant::TauA => {
            // Tau-a: simple ratio
            (concordant as f64 - discordant as f64) / n_pairs as f64
        }
        KendallVariant::TauB => {
            // Tau-b: tie-adjusted (matches R)
            let c_minus_d = concordant as f64 - discordant as f64;
            let denom1 = n_pairs as f64 - ties_x as f64;
            let denom2 = n_pairs as f64 - ties_y as f64;

            if denom1 == 0.0 || denom2 == 0.0 {
                0.0
            } else {
                c_minus_d / (denom1 * denom2).sqrt()
            }
        }
        KendallVariant::TauC => {
            // Tau-c: Stuart's tau-c
            // For continuous data, we use the number of unique values
            let unique_x = count_unique(x);
            let unique_y = count_unique(y);
            let m = unique_x.min(unique_y);

            if m <= 1 {
                0.0
            } else {
                let c_minus_d = concordant as f64 - discordant as f64;
                2.0 * c_minus_d * m as f64 / ((n * n * (m - 1)) as f64)
            }
        }
    };

    // Clamp tau to [-1, 1]
    let tau = tau.clamp(-1.0, 1.0);

    // Compute z-statistic and p-value using normal approximation
    // Variance formula for tau-b (R's method):
    // var(tau) = (4n + 10) / (9n(n-1)) for no ties
    // With ties, use more complex formula

    let (z_stat, p_value) = compute_kendall_significance(x, y, n, concordant, discordant, variant);

    Ok(CorrelationResult {
        estimate: tau,
        statistic: z_stat,
        df: None, // Kendall uses normal approximation, no df
        p_value,
        conf_int: None, // CI for Kendall is complex, not commonly provided
        method: CorrelationMethod::Kendall,
        n,
    })
}

/// Count concordant, discordant, and tied pairs.
///
/// Returns (concordant, discordant, ties_in_x, ties_in_y, ties_in_both)
/// Note: ties_in_x includes all pairs tied in x (including those also tied in y)
/// Same for ties_in_y. This is needed for the tau-b denominator.
fn count_pairs(x: &[f64], y: &[f64]) -> (usize, usize, usize, usize, usize) {
    let n = x.len();
    let mut concordant = 0usize;
    let mut discordant = 0usize;
    let mut ties_x = 0usize; // All pairs tied in x
    let mut ties_y = 0usize; // All pairs tied in y
    let mut ties_xy = 0usize; // Tied in both

    for i in 0..n {
        for j in (i + 1)..n {
            let dx = x[i] - x[j];
            let dy = y[i] - y[j];

            let tied_x = dx == 0.0;
            let tied_y = dy == 0.0;

            if tied_x {
                ties_x += 1;
            }
            if tied_y {
                ties_y += 1;
            }
            if tied_x && tied_y {
                ties_xy += 1;
            }

            // For concordant/discordant, only count pairs not tied in either
            if !tied_x && !tied_y {
                if (dx > 0.0 && dy > 0.0) || (dx < 0.0 && dy < 0.0) {
                    concordant += 1;
                } else {
                    discordant += 1;
                }
            }
        }
    }

    (concordant, discordant, ties_x, ties_y, ties_xy)
}

/// Count unique values in a slice
fn count_unique(data: &[f64]) -> usize {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted.dedup();
    sorted.len()
}

/// Sums over tie groups of size t: (sum t(t-1)(2t+5), sum t(t-1), sum t(t-1)(t-2)).
fn tie_sums(data: &[f64]) -> (f64, f64, f64) {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let (mut v, mut s1, mut s2) = (0.0, 0.0, 0.0);
    let mut i = 0;
    while i < sorted.len() {
        let mut j = i + 1;
        while j < sorted.len() && sorted[j] == sorted[i] {
            j += 1;
        }
        let t = (j - i) as f64;
        if t > 1.0 {
            v += t * (t - 1.0) * (2.0 * t + 5.0);
            s1 += t * (t - 1.0);
            s2 += t * (t - 1.0) * (t - 2.0);
        }
        i = j;
    }
    (v, s1, s2)
}

/// Compute z-statistic and p-value for Kendall's tau using normal approximation.
#[allow(clippy::too_many_arguments)]
fn compute_kendall_significance(
    x: &[f64],
    y: &[f64],
    n: usize,
    concordant: usize,
    discordant: usize,
    _variant: KendallVariant,
) -> (f64, f64) {
    let n_f = n as f64;

    // S = concordant - discordant
    let s = concordant as f64 - discordant as f64;

    // Tie-corrected variance of S (Kendall 1970; identical to R's
    // cor.test(method = "kendall", exact = FALSE)):
    //   var(S) = (v0 - vt - vu) / 18
    //          + v1 / (2 n (n-1))
    //          + v2 / (9 n (n-1) (n-2))
    // with, over tie groups of size t (in x) and u (in y):
    //   v0 = n(n-1)(2n+5), vt = sum t(t-1)(2t+5), vu = sum u(u-1)(2u+5)
    //   v1 = sum t(t-1) * sum u(u-1)
    //   v2 = sum t(t-1)(t-2) * sum u(u-1)(u-2)
    let (vt, sx1, sx2) = tie_sums(x);
    let (vu, sy1, sy2) = tie_sums(y);
    let v0 = n_f * (n_f - 1.0) * (2.0 * n_f + 5.0);
    let mut variance = (v0 - vt - vu) / 18.0;
    if n > 1 {
        variance += sx1 * sy1 / (2.0 * n_f * (n_f - 1.0));
    }
    if n > 2 {
        variance += sx2 * sy2 / (9.0 * n_f * (n_f - 1.0) * (n_f - 2.0));
    }

    // Z-statistic
    let z_stat = if variance <= 0.0 {
        0.0
    } else {
        s / variance.sqrt()
    };

    // Two-sided p-value using normal approximation
    let p_value = if z_stat == 0.0 {
        1.0
    } else {
        let normal = Normal::new(0.0, 1.0).unwrap();
        2.0 * normal.sf(z_stat.abs())
    };

    (z_stat, p_value)
}

#[cfg(test)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;

    #[test]
    fn test_kendall_basic() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];

        let result = kendall(&x, &y, KendallVariant::TauB).unwrap();

        assert!((result.estimate - 1.0).abs() < 1e-10);
        assert_eq!(result.method, CorrelationMethod::Kendall);
        assert_eq!(result.n, 5);
    }

    #[test]
    fn test_kendall_negative() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![5.0, 4.0, 3.0, 2.0, 1.0];

        let result = kendall(&x, &y, KendallVariant::TauB).unwrap();

        assert!((result.estimate - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_kendall_tau_a_vs_tau_b() {
        // With no ties, tau-a and tau-b should be equal
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 3.0, 2.0, 5.0, 4.0];

        let tau_a = kendall(&x, &y, KendallVariant::TauA).unwrap();
        let tau_b = kendall(&x, &y, KendallVariant::TauB).unwrap();

        assert!((tau_a.estimate - tau_b.estimate).abs() < 1e-10);
    }

    #[test]
    fn test_kendall_with_ties() {
        // Data where ties reduce tau-b below 1
        let x = vec![1.0, 2.0, 2.0, 4.0, 5.0, 3.0];
        let y = vec![1.0, 3.0, 2.0, 4.0, 5.0, 4.0];

        let result = kendall(&x, &y, KendallVariant::TauB).unwrap();

        // Should be positive but less than 1 due to ties and discordant pairs
        assert!(result.estimate > 0.0);
        assert!(result.estimate < 1.0);
    }

    #[test]
    fn test_kendall_zero_correlation() {
        // Random-looking data with no clear correlation
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y = vec![3.0, 1.0, 4.0, 2.0, 6.0, 5.0];

        let result = kendall(&x, &y, KendallVariant::TauB).unwrap();

        // Should be close to zero or small
        assert!(result.estimate.abs() < 0.5);
    }

    /// R: cor.test(x, y, method = "kendall", exact = FALSE) with ties in both variables.
    #[test]
    fn test_kendall_tie_corrected_variance_matches_r() {
        let x = [1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.0];
        let y = [2.0, 1.0, 3.0, 3.0, 5.0, 4.0, 6.0, 6.0, 5.0, 8.0, 7.0, 9.0];
        let r = kendall(&x, &y, KendallVariant::TauB).unwrap();
        assert!((r.estimate - 0.80655653082637868).abs() < 1e-12);
        assert!((r.statistic - 3.4987518043922514).abs() < 1e-10);
        assert!(
            (r.p_value - 0.00046744148056981807).abs() < 1e-13,
            "{}",
            r.p_value
        );

        let x = [1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 4.0];
        let y = [3.0, 1.0, 2.0, 2.0, 2.0, 5.0, 5.0];
        let r = kendall(&x, &y, KendallVariant::TauB).unwrap();
        assert!((r.estimate - 0.58823529411764708).abs() < 1e-12);
        assert!(
            (r.statistic - 1.6717604707611915).abs() < 1e-10,
            "{}",
            r.statistic
        );
        assert!(
            (r.p_value - 0.094571564500939773).abs() < 1e-10,
            "{}",
            r.p_value
        );
    }
}
