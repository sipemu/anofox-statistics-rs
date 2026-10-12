use crate::error::{Result, StatError};
use crate::utils::dist::chisq_sf;
use crate::utils::finite::ensure_finite;
use std::collections::BTreeMap;

/// Result of the k-group log-rank test.
#[derive(Debug, Clone, PartialEq)]
pub struct LogRankResult<G> {
    /// Group labels, sorted (as the levels of R's factor).
    pub groups: Vec<G>,
    /// Number of observations per group.
    pub n: Vec<usize>,
    /// Observed number of events per group.
    pub observed: Vec<f64>,
    /// Expected number of events per group under H0.
    pub expected: Vec<f64>,
    /// Chi-squared statistic.
    pub chisq: f64,
    /// Degrees of freedom: groups with `expected > 0`, minus one.
    pub df: f64,
    /// Upper-tail chi-squared p-value.
    pub p_value: f64,
}

/// Log-rank test of equal survival in `k >= 2` groups, as R
/// `survdiff(Surv(time, event) ~ group)` (`rho = 0`).
///
/// At each distinct event time with `n` at risk and `d` events, group `g`
/// with `n_g` at risk contributes `E_g += d n_g / n` and
/// `V_gh += d (n - d) / (n - 1) * n_g / n * (δ_gh - n_h / n)`. The statistic
/// is `(O - E)' V⁻ (O - E)` over the groups with `E > 0`, dropping one.
///
/// # Errors
/// Empty input, unequal lengths, non-finite times, fewer than 2 groups, or
/// a singular variance matrix.
pub fn log_rank<G: Ord + Clone>(
    time: &[f64],
    event: &[bool],
    group: &[G],
) -> Result<LogRankResult<G>> {
    if time.is_empty() {
        return Err(StatError::EmptyData);
    }
    if time.len() != event.len() || time.len() != group.len() {
        return Err(StatError::InvalidParameter(
            "time, event and group must have equal length".into(),
        ));
    }
    ensure_finite("time", time)?;
    let mut levels: BTreeMap<G, usize> = BTreeMap::new();
    for g in group {
        levels.entry(g.clone()).or_insert(0);
    }
    if levels.len() < 2 {
        return Err(StatError::InvalidParameter(
            "log-rank test needs at least 2 groups".into(),
        ));
    }
    for (i, v) in levels.values_mut().enumerate() {
        *v = i;
    }
    let k = levels.len();
    let gi: Vec<usize> = group.iter().map(|g| levels[g]).collect();

    let mut idx: Vec<usize> = (0..time.len()).collect();
    idx.sort_by(|&a, &b| time[a].total_cmp(&time[b]));

    let mut at_risk = vec![0.0; k];
    let mut n_g = vec![0usize; k];
    for &g in &gi {
        at_risk[g] += 1.0;
        n_g[g] += 1;
    }
    let mut obs = vec![0.0; k];
    let mut exp = vec![0.0; k];
    let mut var = vec![vec![0.0; k]; k];
    let mut i = 0;
    while i < idx.len() {
        let t = time[idx[i]];
        let mut j = i;
        let mut d_g = vec![0.0; k];
        let mut leave = vec![0.0; k];
        while j < idx.len() && time[idx[j]] == t {
            let g = gi[idx[j]];
            if event[idx[j]] {
                d_g[g] += 1.0;
            }
            leave[g] += 1.0;
            j += 1;
        }
        let d: f64 = d_g.iter().sum();
        let n: f64 = at_risk.iter().sum();
        if d > 0.0 {
            for g in 0..k {
                obs[g] += d_g[g];
                exp[g] += d * at_risk[g] / n;
            }
            if n > 1.0 {
                let f = d * (n - d) / (n - 1.0);
                for g in 0..k {
                    let pg = at_risk[g] / n;
                    for h in 0..k {
                        let ph = at_risk[h] / n;
                        let delta = if g == h { 1.0 } else { 0.0 };
                        var[g][h] += f * pg * (delta - ph);
                    }
                }
            }
        }
        for g in 0..k {
            at_risk[g] -= leave[g];
        }
        i = j;
    }

    // Groups with positive expected counts; drop the first, as survdiff.
    let keep: Vec<usize> = (0..k).filter(|&g| exp[g] > 0.0).collect();
    let df = keep.len().saturating_sub(1);
    let chisq = if keep.len() < 2 {
        0.0
    } else {
        let sub = &keep[1..];
        let m = sub.len();
        let a: Vec<Vec<f64>> = sub
            .iter()
            .map(|&g| sub.iter().map(|&h| var[g][h]).collect())
            .collect();
        let b: Vec<f64> = sub.iter().map(|&g| obs[g] - exp[g]).collect();
        let x = solve(a, b.clone()).ok_or_else(|| {
            StatError::InvalidParameter("singular log-rank variance matrix".into())
        })?;
        (0..m).map(|r| x[r] * b[r]).sum()
    };
    Ok(LogRankResult {
        groups: levels.into_keys().collect(),
        n: n_g,
        observed: obs,
        expected: exp,
        chisq,
        df: df as f64,
        p_value: if df == 0 {
            f64::NAN
        } else {
            chisq_sf(chisq, df as f64)
        },
    })
}

/// Solve `a x = b` by Gaussian elimination with partial pivoting.
fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let m = b.len();
    let scale = a
        .iter()
        .flatten()
        .fold(0.0_f64, |s, v| s.max(v.abs()))
        .max(f64::MIN_POSITIVE);
    for c in 0..m {
        let p = (c..m).max_by(|&r, &s| a[r][c].abs().total_cmp(&a[s][c].abs()))?;
        if a[p][c].abs() <= 1e-12 * scale {
            return None;
        }
        a.swap(c, p);
        b.swap(c, p);
        for r in c + 1..m {
            let f = a[r][c] / a[c][c];
            let (top, bottom) = a.split_at_mut(r);
            for (x, &y) in bottom[0][c..].iter_mut().zip(&top[c][c..]) {
                *x -= f * y;
            }
            b[r] -= f * b[c];
        }
    }
    let mut x = vec![0.0; m];
    for r in (0..m).rev() {
        let s: f64 = (r + 1..m).map(|c| a[r][c] * x[c]).sum();
        x[r] = (b[r] - s) / a[r][r];
    }
    Some(x)
}
