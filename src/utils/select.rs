//! Order statistics of implicitly defined pairwise quantities (Walsh
//! averages, pairwise differences) without materialising all O(n^2) values.

/// Return the `k`-th smallest (0-based) value of the multiset
/// `{ f(i, j) : i in 0..ranges.len(), j in ranges[i].0..ranges[i].1 }`.
///
/// Requirement: for every row `i`, `f(i, j)` is non-decreasing in `j` and
/// never NaN. The returned value is one of the `f(i, j)` and equals
/// `sorted[k]` of the fully materialised, sorted multiset (up to the sign of
/// zero).
///
/// Algorithm: randomized selection in a matrix with sorted rows (in the
/// spirit of Monahan 1984 / Johnson & Mizoguchi 1978). Each round picks a
/// uniformly random pivot among the still active entries, counts entries
/// `< pivot` and `<= pivot` per row by binary search and shrinks the active
/// column window of every row. Once at most `O(rows)` entries remain they are
/// materialised and selected directly. The pivot generator is a fixed-seed
/// xorshift, so results and run time are deterministic.
///
/// Complexity: expected O(r log(m) log(N)) time for `r` rows of length at
/// most `m` and `N` entries in total; O(r) memory.
///
/// # Panics
/// If `k` is not smaller than the total number of entries.
pub(crate) fn select_kth_sorted_rows<F>(ranges: &[(usize, usize)], f: F, k: usize) -> f64
where
    F: Fn(usize, usize) -> f64,
{
    let rows = ranges.len();
    let mut lo: Vec<usize> = ranges.iter().map(|r| r.0).collect();
    let mut hi: Vec<usize> = ranges.iter().map(|r| r.1.max(r.0)).collect();
    let total: usize = lo.iter().zip(&hi).map(|(l, h)| h - l).sum();
    assert!(k < total, "select_kth_sorted_rows: k = {k} >= {total}");

    let threshold = (4 * rows).max(4096);
    // Number of entries known to be smaller than every active entry.
    let mut below = 0usize;
    let mut rng_state = 0x9E37_79B9_7F4A_7C15u64;
    let mut lt = vec![0usize; rows];
    let mut le = vec![0usize; rows];

    loop {
        let active: usize = lo.iter().zip(&hi).map(|(l, h)| h - l).sum();
        if active <= threshold {
            let mut vals: Vec<f64> = Vec::with_capacity(active);
            for i in 0..rows {
                vals.extend((lo[i]..hi[i]).map(|j| f(i, j)));
            }
            let (_, v, _) = vals.select_nth_unstable_by(k - below, |a, b| a.total_cmp(b));
            return *v;
        }

        // Uniformly random active entry as pivot.
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        let mut r = (rng_state % active as u64) as usize;
        let mut pivot = f64::NAN;
        for i in 0..rows {
            let w = hi[i] - lo[i];
            if r < w {
                pivot = f(i, lo[i] + r);
                break;
            }
            r -= w;
        }

        // Count active entries < pivot and <= pivot.
        let (mut n_lt, mut n_le) = (below, below);
        for i in 0..rows {
            lt[i] = lower_bound(lo[i], hi[i], |j| f(i, j) < pivot);
            le[i] = lower_bound(lt[i], hi[i], |j| f(i, j) <= pivot);
            n_lt += lt[i] - lo[i];
            n_le += le[i] - lo[i];
        }

        if k < n_lt {
            hi.copy_from_slice(&lt);
        } else if k < n_le {
            return pivot;
        } else {
            lo.copy_from_slice(&le);
            below = n_le;
        }
    }
}

/// First index in `[lo, hi)` for which `pred` is false, assuming `pred` holds
/// on a prefix of the range.
#[inline]
fn lower_bound<P: Fn(usize) -> bool>(mut lo: usize, mut hi: usize, pred: P) -> usize {
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if pred(mid) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

/// `k`-th smallest (0-based) Walsh average `(s_i + s_j) / 2`, `i <= j`, of an
/// ascending sorted sample `s`. O(n log^2 n) expected time, O(n) memory.
pub(crate) fn kth_walsh_average(sorted: &[f64], k: usize) -> f64 {
    let n = sorted.len();
    let ranges: Vec<(usize, usize)> = (0..n).map(|i| (i, n)).collect();
    select_kth_sorted_rows(&ranges, |i, j| (sorted[i] + sorted[j]) / 2.0, k)
}

/// `k`-th smallest (0-based) pairwise difference `x_i - y_j` over all
/// `(i, j)`, given `y` sorted descending. O(n log^2 n) expected time, O(nx)
/// memory.
pub(crate) fn kth_pairwise_difference(x: &[f64], y_desc: &[f64], k: usize) -> f64 {
    let ny = y_desc.len();
    let ranges: Vec<(usize, usize)> = (0..x.len()).map(|_| (0, ny)).collect();
    select_kth_sorted_rows(&ranges, |i, j| x[i] - y_desc[j], k)
}

/// `k`-th smallest (0-based) absolute pairwise difference `|s_i - s_j|`,
/// `i < j`, of an ascending sorted sample `s`. O(n log^2 n) expected time,
/// O(n) memory.
pub(crate) fn kth_abs_difference(sorted: &[f64], k: usize) -> f64 {
    let n = sorted.len();
    let ranges: Vec<(usize, usize)> = (0..n).map(|i| (i + 1, n)).collect();
    select_kth_sorted_rows(&ranges, |i, j| sorted[j] - sorted[i], k)
}

/// Median of `count` values given their order statistics `kth(k)`, using
/// `(v[c/2 - 1] + v[c/2]) / 2` for even `count`; NaN when `count == 0`.
pub(crate) fn median_by<F: Fn(usize) -> f64>(count: usize, kth: F) -> f64 {
    if count == 0 {
        return f64::NAN;
    }
    if count % 2 == 0 {
        (kth(count / 2 - 1) + kth(count / 2)) / 2.0
    } else {
        kth(count / 2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 11) as f64 / (1u64 << 53) as f64
    }

    fn sample(s: &mut u64, n: usize, levels: usize) -> Vec<f64> {
        (0..n)
            .map(|_| {
                let u = lcg(s);
                if levels == 0 {
                    u * 20.0 - 7.0
                } else {
                    (u * levels as f64).floor() * 0.5
                }
            })
            .collect()
    }

    fn sorted(mut v: Vec<f64>) -> Vec<f64> {
        v.sort_by(|a, b| a.total_cmp(b));
        v
    }

    /// Naive O(n^2)-memory references.
    fn naive_all(x: &[f64], y: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = x.len();
        let mut walsh = Vec::new();
        let mut absd = Vec::new();
        for i in 0..n {
            for j in i..n {
                walsh.push((x[i] + x[j]) / 2.0);
                if j > i {
                    absd.push((x[i] - x[j]).abs());
                }
            }
        }
        let diffs: Vec<f64> = x
            .iter()
            .flat_map(|a| y.iter().map(move |b| a - b))
            .collect();
        (sorted(walsh), sorted(absd), sorted(diffs))
    }

    #[test]
    fn test_selection_matches_materialised_sort() {
        let mut s = 21u64;
        for &n in &[1usize, 2, 3, 10, 60, 150, 400] {
            for &levels in &[0usize, 1, 3, 25] {
                let x = sample(&mut s, n, levels);
                let y = sample(&mut s, n / 2 + 1, levels);
                let xs = sorted(x.clone());
                let yd: Vec<f64> = sorted(y.clone()).into_iter().rev().collect();
                let (walsh, absd, diffs) = naive_all(&x, &y);

                let ks = |len: usize| -> Vec<usize> {
                    let mut v = vec![0, len / 3, len / 2, len.saturating_sub(1)];
                    v.retain(|&k| k < len);
                    v
                };
                for k in ks(walsh.len()) {
                    assert_eq!(kth_walsh_average(&xs, k), walsh[k], "walsh n={n} k={k}");
                }
                for k in ks(absd.len()) {
                    assert_eq!(kth_abs_difference(&xs, k), absd[k], "abs n={n} k={k}");
                }
                for k in ks(diffs.len()) {
                    assert_eq!(
                        kth_pairwise_difference(&x, &yd, k),
                        diffs[k],
                        "diff n={n} k={k}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_selection_exercises_pivot_rounds() {
        // Large enough to exceed the materialisation threshold.
        let mut s = 5u64;
        for levels in [0usize, 4] {
            let x = sample(&mut s, 300, levels);
            let y = sample(&mut s, 250, levels);
            let xs = sorted(x.clone());
            let yd: Vec<f64> = sorted(y.clone()).into_iter().rev().collect();
            let (walsh, absd, diffs) = naive_all(&x, &y);
            for k in (0..walsh.len()).step_by(997) {
                assert_eq!(kth_walsh_average(&xs, k), walsh[k]);
            }
            for k in (0..absd.len()).step_by(331) {
                assert_eq!(kth_abs_difference(&xs, k), absd[k]);
            }
            for k in (0..diffs.len()).step_by(773) {
                assert_eq!(kth_pairwise_difference(&x, &yd, k), diffs[k]);
            }
        }
    }

    #[test]
    fn test_median_by() {
        let v = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(median_by(4, |k| v[k]), 2.5);
        assert_eq!(median_by(3, |k| v[k]), 2.0);
        assert!(median_by(0, |k| v[k]).is_nan());
    }

    /// 5e9 Walsh averages / pairwise differences (40 GB if materialised) at
    /// n = 100_000.
    #[test]
    #[ignore]
    fn test_select_large_n() {
        let mut s = 13u64;
        let xs = sorted(sample(&mut s, 100_000, 0));
        let n = xs.len();
        let m = median_by(n * (n + 1) / 2, |k| kth_walsh_average(&xs, k));
        assert!(m > 0.0 && m < 6.0, "{m}");
        let d = median_by(n * (n - 1) / 2, |k| kth_abs_difference(&xs, k));
        assert!(d > 4.0 && d < 9.0, "{d}");
    }
}
