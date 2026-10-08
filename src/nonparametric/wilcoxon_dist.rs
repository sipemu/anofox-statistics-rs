//! Exact null distributions of the Wilcoxon rank-sum (Mann-Whitney U) and
//! signed-rank (V) statistics.
//!
//! Both are computed with additions only in `f64` (counts up to ~1e308), so
//! they neither overflow (as `u64` counts do beyond `n1 + n2 ≈ 62` or
//! `n ≥ 64`) nor lose precision through cancellation. They are only used for
//! small samples: callers fall back to the normal approximation above
//! [`MW_EXACT_MAX_PAIRS`] / [`WSR_EXACT_MAX_N`], so an explicit `exact = true`
//! on database-sized input never triggers the (super-linear) dynamic program.

/// Largest `n1 * n2` for which the exact Mann-Whitney distribution is used.
///
/// R's automatic rule is `n1 < 50 && n2 < 50` (at most 2,401 pairs). The
/// Gaussian-binomial recursion below costs about `(n1 + n2) * min(n1, n2)^2 *
/// max(n1, n2) / 2` additions; at 10,000 pairs that is at most ~1.5e8 (100 x
/// 100), with at most a few MB of memory.
pub(crate) const MW_EXACT_MAX_PAIRS: usize = 10_000;

/// Largest number of non-zero differences for which the exact signed-rank
/// distribution is used (R's automatic rule is `n < 50`). Cost
/// `n * n (n + 1) / 4` additions, memory `n (n + 1) / 2` values.
pub(crate) const WSR_EXACT_MAX_N: usize = 300;

/// Whether the exact Mann-Whitney distribution is affordable for `(n1, n2)`.
pub(crate) fn mann_whitney_exact_feasible(n1: usize, n2: usize) -> bool {
    n1.checked_mul(n2).is_some_and(|p| p <= MW_EXACT_MAX_PAIRS)
}

/// Whether the exact signed-rank distribution is affordable for `n`.
pub(crate) fn wilcoxon_exact_feasible(n: usize) -> bool {
    n <= WSR_EXACT_MAX_N
}

/// Probability mass function of the Mann-Whitney U statistic under H0 for
/// sample sizes `(n1, n2)`, indexed by `u = 0..=n1*n2`.
///
/// Uses the Gaussian binomial coefficient `[n1+n2 choose m]_q` (whose
/// coefficient of `q^u` counts the arrangements with `U = u`), built with the
/// q-Pascal rule `[N, k] = [N-1, k-1] + q^k [N-1, k]`, i.e. additions only.
/// Time `O((n1+n2) * m^2 * M)`, memory `O(m^2 * M)` with `m = min(n1, n2)`,
/// `M = max(n1, n2)`.
pub(crate) fn mann_whitney_pmf(n1: usize, n2: usize) -> Vec<f64> {
    let m = n1.min(n2);
    let total_n = n1 + n2;
    // polys[k] = coefficients of [N, k]_q for the current N (degree k(N-k)).
    let mut polys: Vec<Vec<f64>> = vec![Vec::new(); m + 1];
    polys[0] = vec![1.0];
    for big_n in 1..=total_n {
        let k_max = m.min(big_n);
        // Only k >= m - (total_n - big_n) can still reach k = m.
        let k_min = (m + big_n).saturating_sub(total_n).max(1);
        for k in (k_min..=k_max).rev() {
            let deg = k * (big_n - k);
            let mut next = vec![0.0; deg + 1];
            // [N-1, k-1]: degree (k-1)(N-k)
            for (u, &c) in polys[k - 1].iter().enumerate() {
                next[u] += c;
            }
            // q^k [N-1, k]: degree k(N-1-k), empty when k > N-1
            for (u, &c) in polys[k].iter().enumerate() {
                next[u + k] += c;
            }
            polys[k] = next;
        }
        if k_min > 1 {
            // Polynomials below k_min are no longer needed.
            for p in polys.iter_mut().take(k_min - 1) {
                *p = Vec::new();
            }
        }
    }
    let mut pmf = std::mem::take(&mut polys[m]);
    let total: f64 = pmf.iter().sum();
    for p in pmf.iter_mut() {
        *p /= total;
    }
    pmf
}

/// Probability mass function of the signed-rank statistic V under H0 for `n`
/// non-zero differences, indexed by `v = 0..=n(n+1)/2`. Time `O(n^3)`,
/// memory `O(n^2)` (subset-sum counts in `f64`).
pub(crate) fn wilcoxon_pmf(n: usize) -> Vec<f64> {
    let max_v = n * (n + 1) / 2;
    let mut dp = vec![0.0f64; max_v + 1];
    dp[0] = 1.0;
    let mut reach = 0usize;
    for rank in 1..=n {
        reach += rank;
        for j in (rank..=reach).rev() {
            dp[j] += dp[j - rank];
        }
    }
    let total: f64 = dp.iter().sum();
    for p in dp.iter_mut() {
        *p /= total;
    }
    dp
}

/// `P(S <= q)` for a pmf on `0..len`.
pub(crate) fn cdf_le(pmf: &[f64], q: f64) -> f64 {
    if q < 0.0 {
        return 0.0;
    }
    let qi = (q + 1e-7).floor() as usize;
    if qi + 1 >= pmf.len() {
        return 1.0;
    }
    // Sum the shorter tail for accuracy (as R's pwilcox / psignrank do).
    if 2 * qi <= pmf.len() {
        pmf[..=qi].iter().sum::<f64>()
    } else {
        1.0 - pmf[qi + 1..].iter().sum::<f64>()
    }
}

/// `P(S >= q)` for a pmf on `0..len`.
pub(crate) fn cdf_ge(pmf: &[f64], q: f64) -> f64 {
    let qi = q.ceil();
    if qi <= 0.0 {
        return 1.0;
    }
    1.0 - cdf_le(pmf, qi - 1.0)
}

/// Quantile as R's `qwilcox` / `qsignrank`: the smallest `q` with
/// `P(S <= q) >= p` (with R's `10 * DBL_EPSILON` fuzz).
pub(crate) fn quantile(pmf: &[f64], p: f64) -> usize {
    let max_q = pmf.len() - 1;
    let mut acc = 0.0;
    if p <= 0.5 {
        let x = p - 10.0 * f64::EPSILON;
        for (q, &f) in pmf.iter().enumerate() {
            acc += f;
            if acc >= x {
                return q;
            }
        }
        max_q
    } else {
        let x = 1.0 - p + 10.0 * f64::EPSILON;
        for (q, &f) in pmf.iter().enumerate() {
            acc += f;
            if acc > x {
                return max_q - q;
            }
        }
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binom(n: u64, k: u64) -> f64 {
        (0..k).fold(1.0, |acc, i| acc * (n - i) as f64 / (i + 1) as f64)
    }

    /// Brute-force U distribution by enumerating all subsets.
    fn brute_mw(n1: usize, n2: usize) -> Vec<f64> {
        let n = n1 + n2;
        let mut counts = vec![0.0; n1 * n2 + 1];
        for mask in 0u32..(1 << n) {
            if mask.count_ones() as usize != n1 {
                continue;
            }
            // U = sum of ranks of sample 1 - n1(n1+1)/2
            let rs: usize = (0..n).filter(|i| mask >> i & 1 == 1).map(|i| i + 1).sum();
            counts[rs - n1 * (n1 + 1) / 2] += 1.0;
        }
        let total = binom(n as u64, n1 as u64);
        counts.iter().map(|c| c / total).collect()
    }

    #[test]
    fn mw_pmf_matches_enumeration() {
        for (a, b) in [(1, 1), (1, 5), (3, 4), (5, 3), (6, 6), (2, 9)] {
            let p = mann_whitney_pmf(a, b);
            let q = brute_mw(a, b);
            assert_eq!(p.len(), q.len());
            for (x, y) in p.iter().zip(q.iter()) {
                assert!((x - y).abs() < 1e-14, "{a},{b}");
            }
        }
    }

    #[test]
    fn mw_pmf_large_no_overflow() {
        // C(200, 100) ~ 9e58 overflows u64 but not f64.
        let p = mann_whitney_pmf(100, 100);
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        // symmetric
        assert!((p[10] - p[10_000 - 10]).abs() < 1e-20);
    }

    #[test]
    fn wsr_pmf_small() {
        // n = 3: V in 0..=6 with counts 1,1,1,2,1,1,1 / 8
        let p = wilcoxon_pmf(3);
        let expect = [1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0];
        for (x, y) in p.iter().zip(expect.iter()) {
            assert!((x - y / 8.0).abs() < 1e-15);
        }
        let big = wilcoxon_pmf(200);
        assert!((big.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }
}
