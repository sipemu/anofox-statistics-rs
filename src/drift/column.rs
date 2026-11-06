//! Per-column drift detection with evidently's tests, default test choice,
//! thresholds and drift rules, and the dataset drift share.

use super::{
    chisq_drift, g_drift, hellinger, hellinger_categorical, jensen_shannon,
    jensen_shannon_categorical, kl_divergence, kl_divergence_categorical, psi, psi_categorical,
    tvd_test, z_test, Binning, OrdF64, ZeroFill,
};
use crate::categorical::fisher_exact;
use crate::error::{Result, StatError};
use crate::modern::{mmd_test, wasserstein_distance_1d, Kernel};
use crate::nonparametric::{
    ad_test_ksamp, cvm_test_2sample, epps_singleton_test, ks_test_2sample, mann_whitney_u,
    CvmMethod,
};
use crate::parametric::{t_test, Alternative, TTestKind};
use crate::utils::finite::ensure_finite;
use crate::utils::select::kth_abs_difference;

/// A column drift test of evidently's `stattests` registry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DriftTest {
    /// `"ks"`: two-sample Kolmogorov-Smirnov p-value
    /// ([`ks_test_2sample`]).
    Ks,
    /// `"anderson"`: k-sample Anderson-Darling p-value, midrank version
    /// ([`ad_test_ksamp`]).
    Anderson,
    /// `"cramer_von_mises"`: two-sample Cramér-von Mises p-value
    /// ([`cvm_test_2sample`]).
    CramerVonMises,
    /// `"es"`: Epps-Singleton p-value
    /// ([`epps_singleton_test`]).
    EppsSingleton,
    /// `"mannw"`: one-sided (`less`) Mann-Whitney p-value
    /// ([`mann_whitney_u`]).
    MannWhitney,
    /// `"t_test"`: Student two-sample t-test p-value
    /// ([`t_test`]).
    TTest,
    /// `"ed"`: energy distance ([`energy_distance_1d`]).
    EnergyDistance,
    /// `"empirical_mmd"`: permutation p-value of the 1-D MMD² with
    /// evidently's bandwidth ([`empirical_mmd_bandwidth`]).
    EmpiricalMmd,
    /// `"wasserstein"`: Wasserstein-1 distance over the population standard
    /// deviation of the reference (floored at 0.001).
    Wasserstein,
    /// `"psi"`: Population Stability Index ([`psi`](super::psi())).
    Psi,
    /// `"kl_div"`: Kullback-Leibler divergence
    /// ([`kl_divergence`](super::kl_divergence())).
    KlDiv,
    /// `"jensenshannon"`: Jensen-Shannon distance
    /// ([`jensen_shannon`](super::jensen_shannon())).
    JensenShannon,
    /// `"hellinger"`: Hellinger distance ([`hellinger`](super::hellinger())).
    Hellinger,
    /// `"chisquare"`: goodness-of-fit chi-squared p-value
    /// ([`chisq_drift`]).
    ChiSquare,
    /// `"g_test"`: goodness-of-fit G-test p-value
    /// ([`g_drift`]).
    GTest,
    /// `"z"`: two-proportion z test p-value ([`z_test`]).
    Z,
    /// `"fisher_exact"`: Fisher's exact test p-value of the 2x2 table of a
    /// binary column ([`fisher_exact`]).
    FisherExact,
    /// `"TVD"`: total variation distance permutation p-value
    /// ([`tvd_test`]).
    Tvd,
}

impl DriftTest {
    /// All tests, in evidently's registry order.
    pub const ALL: [DriftTest; 18] = [
        DriftTest::Anderson,
        DriftTest::ChiSquare,
        DriftTest::CramerVonMises,
        DriftTest::EnergyDistance,
        DriftTest::EppsSingleton,
        DriftTest::FisherExact,
        DriftTest::GTest,
        DriftTest::Hellinger,
        DriftTest::JensenShannon,
        DriftTest::KlDiv,
        DriftTest::Ks,
        DriftTest::MannWhitney,
        DriftTest::EmpiricalMmd,
        DriftTest::Psi,
        DriftTest::TTest,
        DriftTest::Tvd,
        DriftTest::Wasserstein,
        DriftTest::Z,
    ];

    /// evidently's name of the test (`"ks"`, `"wasserstein"`, ...).
    pub fn name(self) -> &'static str {
        match self {
            DriftTest::Ks => "ks",
            DriftTest::Anderson => "anderson",
            DriftTest::CramerVonMises => "cramer_von_mises",
            DriftTest::EppsSingleton => "es",
            DriftTest::MannWhitney => "mannw",
            DriftTest::TTest => "t_test",
            DriftTest::EnergyDistance => "ed",
            DriftTest::EmpiricalMmd => "empirical_mmd",
            DriftTest::Wasserstein => "wasserstein",
            DriftTest::Psi => "psi",
            DriftTest::KlDiv => "kl_div",
            DriftTest::JensenShannon => "jensenshannon",
            DriftTest::Hellinger => "hellinger",
            DriftTest::ChiSquare => "chisquare",
            DriftTest::GTest => "g_test",
            DriftTest::Z => "z",
            DriftTest::FisherExact => "fisher_exact",
            DriftTest::Tvd => "TVD",
        }
    }

    /// The test with evidently's name `name`, if any.
    ///
    /// # Example
    /// ```
    /// use anofox_statistics::drift::DriftTest;
    /// assert_eq!(DriftTest::from_name("jensenshannon"), Some(DriftTest::JensenShannon));
    /// assert_eq!(DriftTest::from_name("nope"), None);
    /// ```
    pub fn from_name(name: &str) -> Option<DriftTest> {
        DriftTest::ALL.into_iter().find(|t| t.name() == name)
    }

    /// evidently's default threshold of the test.
    pub fn default_threshold(self) -> f64 {
        match self {
            DriftTest::Ks
            | DriftTest::EppsSingleton
            | DriftTest::MannWhitney
            | DriftTest::TTest
            | DriftTest::ChiSquare
            | DriftTest::Z => super::defaults::P_VALUE_THRESHOLD,
            _ => super::defaults::DISTANCE_THRESHOLD,
        }
    }

    /// Whether the score is a p-value (drift at small scores) rather than a
    /// distance (drift at large scores).
    pub fn is_p_value(self) -> bool {
        !matches!(
            self,
            DriftTest::EnergyDistance
                | DriftTest::Wasserstein
                | DriftTest::Psi
                | DriftTest::KlDiv
                | DriftTest::JensenShannon
                | DriftTest::Hellinger
        )
    }

    /// evidently's drift rule: `score <= t` for `ks` and
    /// `cramer_von_mises`, `score < t` for the other p-values, `score > t`
    /// for `ed` and `score >= t` for the other distances.
    pub fn drifted(self, score: f64, threshold: f64) -> bool {
        match self {
            DriftTest::Ks | DriftTest::CramerVonMises => score <= threshold,
            DriftTest::EnergyDistance => score > threshold,
            t if t.is_p_value() => score < threshold,
            _ => score >= threshold,
        }
    }

    /// Whether the test applies to categorical columns (in evidently, the
    /// tests that only compare category frequencies, plus the binned
    /// divergences).
    pub fn supports_categorical(self) -> bool {
        matches!(
            self,
            DriftTest::Psi
                | DriftTest::KlDiv
                | DriftTest::JensenShannon
                | DriftTest::Hellinger
                | DriftTest::ChiSquare
                | DriftTest::GTest
                | DriftTest::Z
                | DriftTest::FisherExact
                | DriftTest::Tvd
        )
    }
}

impl std::fmt::Display for DriftTest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// Options of [`column_drift_numeric`] and [`column_drift_categorical`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DriftOptions {
    /// Test to run; `None` picks evidently's default
    /// ([`default_drift_test_numeric`], [`default_drift_test_categorical`]).
    pub test: Option<DriftTest>,
    /// Drift threshold; `None` uses the test's
    /// [`default_threshold`](DriftTest::default_threshold).
    pub threshold: Option<f64>,
    /// Binning of numeric columns for the divergences (default
    /// [`Binning::Auto`], evidently's).
    pub binning: Binning,
    /// Zero fill of PSI and the Kullback-Leibler divergence (default
    /// evidently's).
    pub zero_fill: ZeroFill,
    /// Permutations of the `TVD` and `empirical_mmd` p-values (default 1000;
    /// evidently uses 1000 and 100).
    pub n_permutations: usize,
    /// Seed of the permutations (`None`: from entropy).
    pub seed: Option<u64>,
}

impl Default for DriftOptions {
    fn default() -> Self {
        DriftOptions {
            test: None,
            threshold: None,
            binning: Binning::Auto,
            zero_fill: ZeroFill::Evidently,
            n_permutations: super::defaults::N_PERMUTATIONS,
            seed: None,
        }
    }
}

/// Drift result of one column.
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnDrift {
    /// The test that was run.
    pub test: DriftTest,
    /// The drift score: a p-value or a distance
    /// ([`DriftTest::is_p_value`]).
    pub score: f64,
    /// The threshold the score was compared with.
    pub threshold: f64,
    /// Whether drift was detected ([`DriftTest::drifted`]).
    pub drifted: bool,
}

use super::defaults::{
    CATEGORICAL_LARGE, CATEGORICAL_SMALL_BINARY, CATEGORICAL_SMALL_MANY, NUMERIC_FEW_VALUES,
    NUMERIC_LARGE_FEW_VALUES, NUMERIC_LARGE_MANY_VALUES, NUMERIC_MANY_VALUES, NUMERIC_SMALL_BINARY,
    NUMERIC_SMALL_FEW_VALUES, NUMERIC_SMALL_MANY_VALUES, SMALL_REFERENCE_ROWS as SMALL_REFERENCE,
};

/// evidently's default test for a numeric column.
///
/// With `u` distinct values over both samples:
/// - reference of at most 1000 rows: [`DriftTest::Ks`] for `u > 5`,
///   [`DriftTest::ChiSquare`] for `3 <= u <= 5`, [`DriftTest::Z`] for
///   `u <= 2`;
/// - larger reference: [`DriftTest::Wasserstein`] for `u > 5`,
///   [`DriftTest::JensenShannon`] otherwise.
///
/// # Example
/// ```
/// use anofox_statistics::drift::{default_drift_test_numeric, DriftTest};
/// let x: Vec<f64> = (0..100).map(|i| i as f64).collect();
/// assert_eq!(default_drift_test_numeric(&x, &x), DriftTest::Ks);
/// assert_eq!(default_drift_test_numeric(&[0.0, 1.0], &[1.0]), DriftTest::Z);
/// ```
pub fn default_drift_test_numeric(reference: &[f64], current: &[f64]) -> DriftTest {
    let pooled: Vec<f64> = reference.iter().chain(current).copied().collect();
    let u = super::binning::n_distinct(&pooled);
    match (reference.len() <= SMALL_REFERENCE, u) {
        (true, u) if u > NUMERIC_MANY_VALUES => NUMERIC_SMALL_MANY_VALUES,
        (true, u) if u > NUMERIC_FEW_VALUES => NUMERIC_SMALL_FEW_VALUES,
        (true, _) => NUMERIC_SMALL_BINARY,
        (false, u) if u > NUMERIC_MANY_VALUES => NUMERIC_LARGE_MANY_VALUES,
        (false, _) => NUMERIC_LARGE_FEW_VALUES,
    }
}

/// evidently's default test for a categorical column:
/// [`DriftTest::ChiSquare`] (more than two categories) or [`DriftTest::Z`]
/// for a reference of at most 1000 rows, [`DriftTest::JensenShannon`] for a
/// larger one.
pub fn default_drift_test_categorical<T: Ord>(reference: &[T], current: &[T]) -> DriftTest {
    if reference.len() > SMALL_REFERENCE {
        return CATEGORICAL_LARGE;
    }
    let (r, _) = super::category_counts(reference, current);
    if r.len() > 2 {
        CATEGORICAL_SMALL_MANY
    } else {
        CATEGORICAL_SMALL_BINARY
    }
}

/// Energy distance of two univariate samples as scipy's `energy_distance`:
/// `sqrt(2 E|X - Y| - E|X - X'| - E|Y - Y'|)` with all expectations over all
/// ordered pairs (V-statistics, including `i = j`). This is evidently's
/// `"ed"` score. O(N log N).
///
/// It differs from the statistic of
/// [`energy_distance_test_1d`](crate::energy_distance_test_1d), which
/// excludes the `i = j` pairs and is not square-rooted.
///
/// Errors on an empty sample or non-finite values.
///
/// R equivalent:
/// `sqrt(2 * mean(abs(outer(x, y, "-"))) - mean(abs(outer(x, x, "-"))) - mean(abs(outer(y, y, "-"))))`.
///
/// # Example
/// ```
/// use anofox_statistics::drift::energy_distance_1d;
/// // 2 * 1 - 0 - 0
/// assert!((energy_distance_1d(&[0.0], &[1.0]).unwrap() - 2f64.sqrt()).abs() < 1e-15);
/// ```
pub fn energy_distance_1d(reference: &[f64], current: &[f64]) -> Result<f64> {
    if reference.is_empty() || current.is_empty() {
        return Err(StatError::EmptyData);
    }
    ensure_finite("reference", reference)?;
    ensure_finite("current", current)?;
    let pooled: Vec<f64> = reference.iter().chain(current).copied().collect();
    let (sa, sb, sz) = (
        pair_abs_sum(reference),
        pair_abs_sum(current),
        pair_abs_sum(&pooled),
    );
    let (m, n) = (reference.len() as f64, current.len() as f64);
    let cross = (sz - sa - sb) / (m * n);
    let e = 2.0 * cross - 2.0 * sa / (m * m) - 2.0 * sb / (n * n);
    Ok(e.max(0.0).sqrt())
}

/// `sum_{i < j} |v_i - v_j|`, from the sorted sample.
fn pair_abs_sum(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_unstable_by(|a, b| a.total_cmp(b));
    let n = s.len() as f64;
    s.iter()
        .enumerate()
        .map(|(i, &x)| x * (2.0 * i as f64 - n + 1.0))
        .sum()
}

/// Bandwidth of evidently's `"empirical_mmd"` test:
/// `sigma = sqrt(0.5 * d)`, with `d` the median (numpy `method = "nearest"`,
/// ties to the even index) of the `m^2` squared differences
/// `(x_i - x_j)^2` of the reference, including the `m` zeros of `i = j`;
/// `sqrt(0.5)` when that median is 0. The kernel
/// `exp(-(a - b)^2 / (2 sigma^2))` is [`Kernel::Gaussian`] with
/// `bandwidth = sigma`. O(m log^2 m).
///
/// Errors on an empty sample or non-finite values.
pub fn empirical_mmd_bandwidth(reference: &[f64]) -> Result<f64> {
    if reference.is_empty() {
        return Err(StatError::EmptyData);
    }
    ensure_finite("reference", reference)?;
    let m = reference.len();
    let count = m as u64 * m as u64;
    // numpy "nearest": index round_half_even(0.5 * (count - 1)).
    let idx = if count % 2 == 1 {
        (count - 1) / 2
    } else {
        let lo = (count - 2) / 2;
        if lo % 2 == 0 {
            lo
        } else {
            lo + 1
        }
    };
    // Sorted multiset: the m diagonal zeros, then every pair difference
    // twice.
    let d = if idx < m as u64 {
        0.0
    } else {
        let mut s = reference.to_vec();
        s.sort_unstable_by(|a, b| a.total_cmp(b));
        kth_abs_difference(&s, (idx - m as u64) / 2).powi(2)
    };
    Ok(if d == 0.0 {
        0.5f64.sqrt()
    } else {
        (0.5 * d).sqrt()
    })
}

/// Wasserstein-1 distance over `max(population sd of the reference, 0.001)`.
fn wasserstein_normed(reference: &[f64], current: &[f64]) -> Result<f64> {
    let w = wasserstein_distance_1d(reference, current, 1.0)?;
    let m = reference.len() as f64;
    let mean = reference.iter().sum::<f64>() / m;
    let sd = (reference.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / m).sqrt();
    Ok(w / sd.max(0.001))
}

/// Fisher's exact test of the 2x2 table (sample x category) of a column
/// with at most two categories.
fn fisher_binary<T: Ord>(reference: &[T], current: &[T]) -> Result<f64> {
    let (r, c) = super::category_counts(reference, current);
    match r.len() {
        1 => Ok(1.0),
        2 => Ok(fisher_exact(&[[r[0], r[1]], [c[0], c[1]]], Alternative::TwoSided)?.p_value),
        k => Err(StatError::InvalidParameter(format!(
            "fisher_exact needs a binary column, got {} categories",
            k
        ))),
    }
}

fn finish(test: DriftTest, score: f64, threshold: Option<f64>) -> Result<ColumnDrift> {
    let threshold = threshold.unwrap_or_else(|| test.default_threshold());
    if !threshold.is_finite() {
        return Err(StatError::InvalidParameter(format!(
            "threshold must be finite, got {}",
            threshold
        )));
    }
    Ok(ColumnDrift {
        test,
        score,
        threshold,
        drifted: test.drifted(score, threshold),
    })
}

/// Drift of a numeric column between a reference and a current sample, as
/// evidently's column drift.
///
/// Runs `options.test` (evidently's [`default_drift_test_numeric`] when
/// `None`) and compares its score with `options.threshold` (the test's
/// default when `None`) using evidently's [`DriftTest::drifted`] rule. The
/// frequency tests (`chisquare`, `g_test`, `z`, `fisher_exact`, `TVD`) use
/// one category per distinct value.
///
/// The scores follow evidently, computed with this crate's R-validated
/// routines. Where scipy and R differ, R is followed:
/// - `anderson`: kSamples' asymptotic p-value, not capped to
///   `[0.001, 0.25]` as scipy's `anderson_ksamp`;
/// - `ks`, `mannw`, `cramer_von_mises`: exact or asymptotic p-values with R's
///   rules (`ks.test`, `wilcox.test` exact for `n < 50` without ties) and
///   scipy's rule for the Cramér-von Mises test;
/// - `g_test`: expected counts scaled to the current size
///   ([`g_drift`]);
/// - `TVD`, `empirical_mmd`: permutation p-values `(1 + #) / (1 + B)` with
///   this crate's seeded RNG.
///
/// Errors on empty or non-finite input, on a test that does not accept the
/// data (`es` with fewer than 5 values per sample, `fisher_exact` with more
/// than two categories) and on a non-finite threshold.
///
/// # Example
/// ```
/// use anofox_statistics::drift::{column_drift_numeric, DriftOptions, DriftTest};
/// let x: Vec<f64> = (0..50).map(|i| (0.37 * i as f64).sin()).collect();
/// let y: Vec<f64> = (0..40).map(|i| (0.53 * i as f64).sin() + 0.5).collect();
/// let r = column_drift_numeric(&x, &y, &DriftOptions::default()).unwrap();
/// assert_eq!(r.test, DriftTest::Ks);
/// assert_eq!(r.threshold, 0.05);
/// assert!(r.drifted);
/// ```
pub fn column_drift_numeric(
    reference: &[f64],
    current: &[f64],
    options: &DriftOptions,
) -> Result<ColumnDrift> {
    if reference.is_empty() || current.is_empty() {
        return Err(StatError::EmptyData);
    }
    ensure_finite("reference", reference)?;
    ensure_finite("current", current)?;
    let test = options
        .test
        .unwrap_or_else(|| default_drift_test_numeric(reference, current));
    let (x, y) = (reference, current);
    let score = match test {
        DriftTest::Ks => ks_test_2sample(x, y, Alternative::TwoSided, None)?.p_value,
        DriftTest::Anderson => ad_test_ksamp(&[x, y], true)?.p_value,
        DriftTest::CramerVonMises => cvm_test_2sample(x, y, CvmMethod::Auto)?.p_value,
        DriftTest::EppsSingleton => epps_singleton_test(x, y, None)?.p_value,
        DriftTest::MannWhitney => {
            let exact = x.len() < 50 && y.len() < 50;
            mann_whitney_u(x, y, Alternative::Less, true, exact, None, None)?.p_value
        }
        DriftTest::TTest => {
            t_test(x, y, TTestKind::Student, Alternative::TwoSided, 0.0, None)?.p_value
        }
        DriftTest::EnergyDistance => energy_distance_1d(x, y)?,
        DriftTest::EmpiricalMmd => {
            let bandwidth = empirical_mmd_bandwidth(x)?;
            let xv: Vec<Vec<f64>> = x.iter().map(|&v| vec![v]).collect();
            let yv: Vec<Vec<f64>> = y.iter().map(|&v| vec![v]).collect();
            mmd_test(
                &xv,
                &yv,
                Kernel::Gaussian { bandwidth },
                options.n_permutations,
                options.seed,
            )?
            .p_value
        }
        DriftTest::Wasserstein => wasserstein_normed(x, y)?,
        DriftTest::Psi => psi(x, y, options.binning, options.zero_fill)?.statistic,
        DriftTest::KlDiv => {
            kl_divergence(x, y, options.binning, None, options.zero_fill)?.statistic
        }
        DriftTest::JensenShannon => jensen_shannon(x, y, options.binning, None)?.statistic,
        DriftTest::Hellinger => hellinger(x, y, options.binning)?.statistic,
        t => {
            return column_drift_categorical(&OrdF64::codes(x), &OrdF64::codes(y), &{
                let mut o = *options;
                o.test = Some(t);
                o
            })
        }
    };
    finish(test, score, options.threshold)
}

/// Drift of a categorical column between a reference and a current sample,
/// as evidently's column drift.
///
/// As [`column_drift_numeric`], for labels of any ordered type; the default
/// test is [`default_drift_test_categorical`]. Only the tests with
/// [`DriftTest::supports_categorical`] apply; the divergences use one bin
/// per category.
///
/// # Example
/// ```
/// use anofox_statistics::drift::{column_drift_categorical, DriftOptions, DriftTest};
/// let x = ["a", "b", "c", "a", "b", "c"];
/// let y = ["a", "a", "a", "a", "b", "c"];
/// let r = column_drift_categorical(&x, &y, &DriftOptions::default()).unwrap();
/// assert_eq!(r.test, DriftTest::ChiSquare);
/// assert!(!r.drifted);
/// ```
pub fn column_drift_categorical<T: Ord>(
    reference: &[T],
    current: &[T],
    options: &DriftOptions,
) -> Result<ColumnDrift> {
    if reference.is_empty() || current.is_empty() {
        return Err(StatError::EmptyData);
    }
    let test = options
        .test
        .unwrap_or_else(|| default_drift_test_categorical(reference, current));
    let (x, y) = (reference, current);
    let score = match test {
        DriftTest::Psi => psi_categorical(x, y, options.zero_fill)?.statistic,
        DriftTest::KlDiv => kl_divergence_categorical(x, y, None, options.zero_fill)?.statistic,
        DriftTest::JensenShannon => jensen_shannon_categorical(x, y, None)?.statistic,
        DriftTest::Hellinger => hellinger_categorical(x, y)?.statistic,
        DriftTest::ChiSquare => chisq_drift(x, y)?.p_value,
        DriftTest::GTest => g_drift(x, y)?.p_value,
        DriftTest::Z => z_test(x, y, Alternative::TwoSided)?.p_value,
        DriftTest::FisherExact => fisher_binary(x, y)?,
        DriftTest::Tvd => tvd_test(x, y, options.n_permutations, options.seed)?.p_value,
        t => {
            return Err(StatError::InvalidParameter(format!(
                "drift test '{}' does not apply to categorical columns",
                t
            )))
        }
    };
    finish(test, score, options.threshold)
}

/// Share of drifted columns and whether the dataset drifted.
#[derive(Debug, Clone, PartialEq)]
pub struct DatasetDrift {
    /// Number of columns.
    pub n_columns: usize,
    /// Number of drifted columns.
    pub n_drifted: usize,
    /// `n_drifted / n_columns`.
    pub share_drifted: f64,
    /// `share_drifted >= drift_share`.
    pub dataset_drift: bool,
}

/// evidently's dataset drift: the dataset drifted when the share of drifted
/// columns is at least `drift_share` (evidently's default 0.5).
///
/// Errors on no columns or a `drift_share` outside `[0, 1]`.
///
/// # Example
/// ```
/// use anofox_statistics::drift::dataset_drift;
/// let d = dataset_drift(&[true, false, true, false], 0.5).unwrap();
/// assert_eq!(d.share_drifted, 0.5);
/// assert!(d.dataset_drift);
/// ```
pub fn dataset_drift(drifted: &[bool], drift_share: f64) -> Result<DatasetDrift> {
    if drifted.is_empty() {
        return Err(StatError::EmptyData);
    }
    if !(0.0..=1.0).contains(&drift_share) {
        return Err(StatError::InvalidParameter(format!(
            "drift_share must be in [0, 1], got {}",
            drift_share
        )));
    }
    let n_drifted = drifted.iter().filter(|&&d| d).count();
    let share_drifted = n_drifted as f64 / drifted.len() as f64;
    Ok(DatasetDrift {
        n_columns: drifted.len(),
        n_drifted,
        share_drifted,
        dataset_drift: share_drifted >= drift_share,
    })
}
