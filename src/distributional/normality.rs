use crate::error::{Result, StatError};
use statrs::distribution::{ContinuousCDF, Normal};

/// Result of Shapiro-Wilk test
#[derive(Debug, Clone)]
pub struct ShapiroWilkResult {
    /// The W statistic
    pub statistic: f64,
    /// The p-value
    pub p_value: f64,
}

/// Perform the Shapiro-Wilk test for normality.
///
/// Tests the null hypothesis that the data was drawn from a normal distribution.
/// Implementation follows Algorithm AS R94 (Royston, 1995).
///
/// # Arguments
/// * `data` - Sample data (3 <= n <= 5000)
///
/// # Returns
/// * `ShapiroWilkResult` containing W statistic and p-value
pub fn shapiro_wilk(data: &[f64]) -> Result<ShapiroWilkResult> {
    let n = data.len();

    if n < 3 {
        return Err(StatError::InsufficientData { needed: 3, got: n });
    }

    if n > 5000 {
        return Err(StatError::InvalidParameter(
            "Shapiro-Wilk test is limited to n <= 5000".to_string(),
        ));
    }

    // Sort the data
    let mut x = data.to_vec();
    x.sort_by(|a, b| a.total_cmp(b));

    // Check for constant data
    let range = x[n - 1] - x[0];
    if range < 1e-10 {
        return Ok(ShapiroWilkResult {
            statistic: 1.0,
            p_value: 1.0,
        });
    }

    let (w, p_value) = swilk(&x);

    Ok(ShapiroWilkResult {
        statistic: w,
        p_value,
    })
}

/// Implementation of the SWILK algorithm (Royston 1995, AS R94), ported
/// line-for-line from R's `src/library/stats/src/swilk.c` so that both the
/// W statistic and the p-value (including the small-sample branch for
/// 4 <= n <= 11) agree with `shapiro.test()` / `scipy.stats.shapiro`.
///
/// `x` must be sorted ascending and have non-zero range.
fn swilk(x: &[f64]) -> (f64, f64) {
    let n = x.len();
    let an = n as f64;
    let nn2 = n / 2;
    let normal = Normal::new(0.0, 1.0).unwrap();

    // Polynomial coefficients (ascending powers), as in swilk.c
    const G: [f64; 2] = [-2.273, 0.459];
    const C1: [f64; 6] = [0.0, 0.221157, -0.147981, -2.07119, 4.434685, -2.706056];
    const C2: [f64; 6] = [0.0, 0.042981, -0.293762, -1.752461, 5.682633, -3.582633];
    const C3: [f64; 4] = [0.544, -0.39978, 0.025054, -6.714e-4];
    const C4: [f64; 4] = [1.3822, -0.77857, 0.062767, -0.0020322];
    const C5: [f64; 4] = [-1.5861, -0.31082, -0.083751, 0.0038915];
    const C6: [f64; 3] = [-0.4803, -0.082676, 0.0030302];

    // a is 1-based like the C code: a[1..=nn2]
    let mut a = vec![0.0; nn2 + 1];
    if n == 3 {
        a[1] = std::f64::consts::FRAC_1_SQRT_2;
    } else {
        let an25 = an + 0.25;
        let mut summ2 = 0.0;
        for (i, ai) in a.iter_mut().enumerate().skip(1) {
            *ai = normal.inverse_cdf((i as f64 - 0.375) / an25);
            summ2 += *ai * *ai;
        }
        summ2 *= 2.0;
        let ssumm2 = summ2.sqrt();
        let rsn = 1.0 / an.sqrt();
        let a1 = poly(&C1, rsn) - a[1] / ssumm2;

        let (i1, fac) = if n > 5 {
            let a2 = -a[2] / ssumm2 + poly(&C2, rsn);
            let fac = ((summ2 - 2.0 * a[1] * a[1] - 2.0 * a[2] * a[2])
                / (1.0 - 2.0 * a1 * a1 - 2.0 * a2 * a2))
                .sqrt();
            a[2] = a2;
            (3, fac)
        } else {
            let fac = ((summ2 - 2.0 * a[1] * a[1]) / (1.0 - 2.0 * a1 * a1)).sqrt();
            (2, fac)
        };
        a[1] = a1;
        for ai in a.iter_mut().take(nn2 + 1).skip(i1) {
            *ai /= -fac;
        }
    }

    let range = x[n - 1] - x[0];

    // Full antisymmetric coefficient vector: coef(i) = sign(i - j) * a[1 + min(i, j)],
    // j = n - 1 - i (0 for the middle element of odd n).
    let coef = |i: usize| -> f64 {
        let j = n - 1 - i;
        match i.cmp(&j) {
            std::cmp::Ordering::Less => -a[1 + i],
            std::cmp::Ordering::Greater => a[1 + j],
            std::cmp::Ordering::Equal => 0.0,
        }
    };

    // W as squared correlation between data (range-scaled) and coefficients
    let sa: f64 = (0..n).map(coef).sum::<f64>() / an;
    let sx: f64 = x.iter().map(|xi| xi / range).sum::<f64>() / an;
    let (mut ssa, mut ssx, mut sax) = (0.0, 0.0, 0.0);
    for (i, xi) in x.iter().enumerate() {
        let asa = coef(i) - sa;
        let xsx = xi / range - sx;
        ssa += asa * asa;
        ssx += xsx * xsx;
        sax += asa * xsx;
    }

    // w1 = 1 - W, computed to avoid rounding error for W near 1
    let ssassx = (ssa * ssx).sqrt();
    let w1 = (ssassx - sax) * (ssassx + sax) / (ssa * ssx);
    let w = 1.0 - w1;

    // Significance level for W
    if n == 3 {
        // 6/pi and asin(sqrt(3/4)) = pi/3
        let pi6 = 6.0 / std::f64::consts::PI;
        let stqr = std::f64::consts::FRAC_PI_3;
        let pw = pi6 * (w.sqrt().asin() - stqr);
        return (w, pw.clamp(0.0, 1.0));
    }

    let mut y = w1.ln();
    let xx = an.ln();
    let (m, s) = if n <= 11 {
        let gamma = poly(&G, an);
        if y >= gamma {
            return (w, 1e-99);
        }
        y = -(gamma - y).ln();
        (poly(&C3, an), poly(&C4, an).exp())
    } else {
        (poly(&C5, xx), poly(&C6, xx).exp())
    };

    let z = (y - m) / s;
    (w, normal.sf(z).clamp(0.0, 1.0))
}

/// Evaluate a polynomial with ascending coefficients: c[0] + c[1]*x + c[2]*x^2 + ...
fn poly(c: &[f64], x: f64) -> f64 {
    c.iter().rev().fold(0.0, |acc, &ci| acc * x + ci)
}

#[cfg(test)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;

    /// Reference values from R 4.x `shapiro.test()` (scipy.stats.shapiro agrees).
    fn check(data: &[f64], w_ref: f64, p_ref: f64) {
        let r = shapiro_wilk(data).unwrap();
        assert!(
            (r.statistic - w_ref).abs() < 1e-10,
            "W: got {}, R {}",
            r.statistic,
            w_ref
        );
        assert!(
            (r.p_value - p_ref).abs() < 1e-10,
            "p: got {}, R {}",
            r.p_value,
            p_ref
        );
    }

    #[test]
    fn matches_r_n3_exact() {
        check(&[1.0, 2.0, 4.0], 0.96428571428571419, 0.6368868450289632);
    }

    #[test]
    fn matches_r_n4_n5_small_coefficients() {
        check(
            &[1.2, 3.4, 2.2, 5.9],
            0.95422076646118381,
            0.74255791273476768,
        );
        check(
            &[1.2, 3.4, 2.2, 5.9, 2.0],
            0.89215500173725426,
            0.36805560604568577,
        );
    }

    #[test]
    fn matches_r_small_sample_branch() {
        check(
            &[3.1, 2.7, 4.4, 1.9, 5.6, 3.3, 2.8],
            0.9246674424398178,
            0.50651413318800143,
        );
        check(
            &[4.1, 5.3, 3.8, 6.9, 5.0, 4.4, 9.2, 5.7, 4.9, 6.1],
            0.88790406283629908,
            0.16058545199963783,
        );
    }

    #[test]
    fn matches_r_large_sample_branch() {
        check(
            &[
                2.31, 3.85, 1.97, 4.42, 3.10, 2.76, 5.94, 3.33, 2.05, 4.88, 3.61, 2.49, 7.12, 3.02,
                2.88, 4.15, 3.47, 1.64, 5.21, 2.95,
            ],
            0.93325039684015909,
            0.1783034135065848,
        );
    }
}
