//! Multiple-comparison p-value adjustment, a port of R's `stats::p.adjust`.

use crate::error::{Result, StatError};
use std::str::FromStr;

/// P-value adjustment method, as in R's `p.adjust.methods`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PAdjustMethod {
    /// Holm (1979) step-down; R `"holm"` (R's default).
    Holm,
    /// Hochberg (1988) step-up; R `"hochberg"`.
    Hochberg,
    /// Hommel (1988); R `"hommel"`.
    Hommel,
    /// Bonferroni; R `"bonferroni"`.
    Bonferroni,
    /// Benjamini & Hochberg (1995) false discovery rate; R `"BH"` / `"fdr"`.
    BH,
    /// Benjamini & Yekutieli (2001) false discovery rate; R `"BY"`.
    BY,
    /// No adjustment; R `"none"`.
    None,
}

impl PAdjustMethod {
    /// The method's name as R spells it (`"holm"`, `"BH"`, ...).
    pub fn r_name(&self) -> &'static str {
        match self {
            PAdjustMethod::Holm => "holm",
            PAdjustMethod::Hochberg => "hochberg",
            PAdjustMethod::Hommel => "hommel",
            PAdjustMethod::Bonferroni => "bonferroni",
            PAdjustMethod::BH => "BH",
            PAdjustMethod::BY => "BY",
            PAdjustMethod::None => "none",
        }
    }
}

impl FromStr for PAdjustMethod {
    type Err = StatError;

    /// Parse an R method name, case-insensitively; `"fdr"` is an alias of `"BH"`.
    fn from_str(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "holm" => Ok(PAdjustMethod::Holm),
            "hochberg" => Ok(PAdjustMethod::Hochberg),
            "hommel" => Ok(PAdjustMethod::Hommel),
            "bonferroni" => Ok(PAdjustMethod::Bonferroni),
            "bh" | "fdr" => Ok(PAdjustMethod::BH),
            "by" => Ok(PAdjustMethod::BY),
            "none" => Ok(PAdjustMethod::None),
            other => Err(StatError::InvalidParameter(format!(
                "unknown p-value adjustment method '{}' (expected holm, hochberg, hommel, \
                 bonferroni, BH, fdr, BY or none)",
                other
            ))),
        }
    }
}

/// Stable ascending order of `p` (R's `order(p)`; `p` has no NaN).
fn order_asc(p: &[f64]) -> Vec<usize> {
    let mut o: Vec<usize> = (0..p.len()).collect();
    o.sort_by(|&a, &b| p[a].total_cmp(&p[b]));
    o
}

/// Stable descending order of `p` (R's `order(p, decreasing = TRUE)`).
fn order_desc(p: &[f64]) -> Vec<usize> {
    let mut o: Vec<usize> = (0..p.len()).collect();
    o.sort_by(|&a, &b| p[b].total_cmp(&p[a]));
    o
}

/// Adjust p-values for multiple comparisons, matching R's
/// `p.adjust(p, method)` exactly.
///
/// NaN entries (R's `NA`) are kept as NaN in the output and are not counted:
/// the number of comparisons `n` is the number of non-NaN values, as in R.
/// With at most one non-NaN value the input is returned unchanged, and
/// Hommel with two values falls back to Hochberg (both as in R). Values are
/// not range-checked, just as in R.
///
/// Memory and time are O(n) and O(n log n) (Hommel: O(n²) time).
///
/// # Example
/// ```
/// use anofox_statistics::{p_adjust, PAdjustMethod};
/// let adj = p_adjust(&[0.01, 0.02, 0.03, f64::NAN], PAdjustMethod::Holm);
/// assert!((adj[0] - 0.03).abs() < 1e-12);
/// assert!(adj[3].is_nan());
/// ```
pub fn p_adjust(p: &[f64], method: PAdjustMethod) -> Vec<f64> {
    let mut out = p.to_vec();
    let idx: Vec<usize> = (0..p.len()).filter(|&i| !p[i].is_nan()).collect();
    let pv: Vec<f64> = idx.iter().map(|&i| p[i]).collect();
    let n = pv.len();
    if n <= 1 {
        return out;
    }
    let method = if n == 2 && method == PAdjustMethod::Hommel {
        PAdjustMethod::Hochberg
    } else {
        method
    };
    let nf = n as f64;
    let adj: Vec<f64> = match method {
        PAdjustMethod::None => pv,
        PAdjustMethod::Bonferroni => pv.iter().map(|&x| (nf * x).min(1.0)).collect(),
        PAdjustMethod::Holm => {
            // pmin(1, cummax((n + 1 - i) * p[o]))[ro]
            let o = order_asc(&pv);
            let mut res = vec![0.0; n];
            let mut run = f64::NEG_INFINITY;
            for (i, &oi) in o.iter().enumerate() {
                let v = (nf - i as f64) * pv[oi];
                run = run.max(v);
                res[oi] = run.min(1.0);
            }
            res
        }
        PAdjustMethod::Hochberg | PAdjustMethod::BH | PAdjustMethod::BY => {
            // i <- n:1; o <- order(p, decreasing = TRUE); pmin(1, cummin(f(i) * p[o]))[ro]
            let q = if method == PAdjustMethod::BY {
                (1..=n).map(|k| 1.0 / k as f64).sum::<f64>()
            } else {
                1.0
            };
            let o = order_desc(&pv);
            let mut res = vec![0.0; n];
            let mut run = f64::INFINITY;
            for (pos, &oi) in o.iter().enumerate() {
                let i = (n - pos) as f64;
                let v = match method {
                    PAdjustMethod::Hochberg => (nf + 1.0 - i) * pv[oi],
                    _ => q * nf / i * pv[oi],
                };
                run = run.min(v);
                res[oi] = run.min(1.0);
            }
            res
        }
        PAdjustMethod::Hommel => hommel(&pv),
    };
    for (k, &i) in idx.iter().enumerate() {
        out[i] = adj[k];
    }
    out
}

/// R's Hommel branch (`n >= 3`, no NaN).
fn hommel(p_in: &[f64]) -> Vec<f64> {
    let n = p_in.len();
    let o = order_asc(p_in);
    let p: Vec<f64> = o.iter().map(|&i| p_in[i]).collect();
    // q <- pa <- rep.int(min(n * p / i), n)
    let init = (0..n)
        .map(|i| n as f64 * p[i] / (i + 1) as f64)
        .fold(f64::INFINITY, f64::min);
    let mut q = vec![init; n];
    let mut pa = vec![init; n];
    for m in (2..n).rev() {
        // i1 <- 1:(n - m + 1); i2 <- (n - m + 2):n  (1-based)
        let n1 = n - m + 1; // length of i1
        let mf = m as f64;
        // q1 <- min(m * p[i2] / (2:m))
        let q1 = (n1..n)
            .enumerate()
            .map(|(k, j)| mf * p[j] / (k + 2) as f64)
            .fold(f64::INFINITY, f64::min);
        for (j, qj) in q.iter_mut().enumerate().take(n1) {
            *qj = (mf * p[j]).min(q1);
        }
        let fill = q[n1 - 1];
        for qj in q.iter_mut().skip(n1) {
            *qj = fill;
        }
        for j in 0..n {
            pa[j] = pa[j].max(q[j]);
        }
    }
    // pmax(pa, p)[ro]
    let mut res = vec![0.0; n];
    for (j, &oi) in o.iter().enumerate() {
        res[oi] = pa[j].max(p[j]);
    }
    res
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_names() {
        assert_eq!("fdr".parse::<PAdjustMethod>().unwrap(), PAdjustMethod::BH);
        assert_eq!("BY".parse::<PAdjustMethod>().unwrap(), PAdjustMethod::BY);
        assert!("foo".parse::<PAdjustMethod>().is_err());
    }

    #[test]
    fn trivial_inputs() {
        assert!(p_adjust(&[], PAdjustMethod::Holm).is_empty());
        assert_eq!(p_adjust(&[0.3], PAdjustMethod::Bonferroni), vec![0.3]);
        let r = p_adjust(&[f64::NAN, 0.2], PAdjustMethod::BY);
        assert!(r[0].is_nan());
        assert_eq!(r[1], 0.2);
    }
}
