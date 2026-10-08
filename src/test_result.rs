//! A unified, flat result shape for every hypothesis test.
//!
//! [`TestResult`] is the `test` schema of the anofox integration contract:
//! one struct with the same fields for every test, `None` where a field does
//! not apply. Every existing result type converts into it with `From`
//! (`TestResult::from(&res)` or `res.into()`); the multi-test ANOVA tables
//! provide `to_test_results()`. The existing result types are unchanged.
//!
//! A conversion fills only what the source result stores. Facts the source
//! does not record (for example the alternative of a t-test, or the sample
//! size of a Kruskal-Wallis test) are `None`; set them with the `with_*`
//! builders:
//!
//! ```
//! use anofox_statistics::{t_test, Alternative, TTestKind, TestResult};
//! let x = [5.1, 4.9, 5.6, 5.8, 6.0, 5.5];
//! let y = [4.1, 4.5, 4.9, 5.0, 4.4, 4.7];
//! let res = t_test(&x, &y, TTestKind::Welch, Alternative::TwoSided, 0.0, Some(0.95)).unwrap();
//! let tr = TestResult::from(&res)
//!     .with_method("Welch Two Sample t-test")
//!     .with_alternative(Alternative::TwoSided)
//!     .with_n(x.len() + y.len());
//! assert_eq!(tr.df1, Some(res.df));
//! assert_eq!(tr.alternative_name(), Some("two.sided"));
//! ```

use crate::categorical::{
    AssociationResult, BinomTestResult, ChiSquareResult, FisherResult, KappaResult,
    McNemarkExactResult, McNemarkResult, PropTestResult,
};
use crate::correlation::{
    CorrelationMethod, CorrelationResult, DistanceCorResult, ICCResult, PartialCorResult,
};
use crate::distributional::{DAgostinoResult, JarqueBeraResult, ShapiroWilkResult};
use crate::equivalence::{OneSidedTestResult, TostResult};
use crate::forecast::{CWResult, DMResult, MCSResult, MSPEAdjustedResult, SPAResult};
use crate::modern::{EnergyDistanceResult, MMDResult};
use crate::nonparametric::{BrunnerMunzelResult, KruskalResult, MannWhitneyResult, WilcoxonResult};
use crate::parametric::anova::AnovaTableRow;
use crate::parametric::{
    Alternative, CorrectedResult, LeveneResult, OneWayAnovaResult, RmAnovaResult, SphericityResult,
    TTestResult, TwoWayAnovaResult, YuenResult,
};
use crate::resampling::{BootstrapCIResult, PermutationResult};

/// Unified result of a hypothesis test (the contract's `test` row).
///
/// All numeric fields are `Option`: `None` means "not applicable / not
/// computed" (SQL `NULL`), while `Some(NaN)` means the test computed an
/// undefined value.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TestResult {
    /// Name of the test, e.g. `"Kruskal-Wallis rank sum test"`.
    pub method: String,
    /// Alternative hypothesis, if the test has one and it is known.
    pub alternative: Option<Alternative>,
    /// Test statistic.
    pub statistic: Option<f64>,
    /// (Numerator) degrees of freedom of the statistic.
    pub df1: Option<f64>,
    /// Denominator degrees of freedom (F-type statistics).
    pub df2: Option<f64>,
    /// P-value.
    pub p_value: Option<f64>,
    /// Point estimate (difference, odds ratio, correlation, ...).
    pub estimate: Option<f64>,
    /// Lower confidence limit for `estimate`.
    pub conf_low: Option<f64>,
    /// Upper confidence limit for `estimate`.
    pub conf_high: Option<f64>,
    /// Confidence level of `conf_low` / `conf_high`.
    pub conf_level: Option<f64>,
    /// Effect size.
    pub effect_size: Option<f64>,
    /// Name of the effect size, e.g. `"eta_squared"`, `"r"`, `"kappa"`.
    pub effect_name: Option<String>,
    /// Number of observations (or subjects / pairs / trials, as documented
    /// for the conversion).
    pub n: Option<usize>,
}

impl TestResult {
    /// An empty result with the given method name.
    pub fn new(method: impl Into<String>) -> Self {
        TestResult {
            method: method.into(),
            ..Default::default()
        }
    }

    /// Replace the method name.
    pub fn with_method(mut self, method: impl Into<String>) -> Self {
        self.method = method.into();
        self
    }

    /// Set the alternative hypothesis.
    pub fn with_alternative(mut self, alternative: Alternative) -> Self {
        self.alternative = Some(alternative);
        self
    }

    /// Set the number of observations.
    pub fn with_n(mut self, n: usize) -> Self {
        self.n = Some(n);
        self
    }

    /// Set the confidence level of the interval.
    pub fn with_conf_level(mut self, conf_level: f64) -> Self {
        self.conf_level = Some(conf_level);
        self
    }

    /// Set the effect size and its name.
    pub fn with_effect_size(mut self, name: impl Into<String>, value: f64) -> Self {
        self.effect_name = Some(name.into());
        self.effect_size = Some(value);
        self
    }

    /// The alternative as R spells it: `"two.sided"`, `"less"` or `"greater"`.
    pub fn alternative_name(&self) -> Option<&'static str> {
        self.alternative.map(|a| match a {
            Alternative::TwoSided => "two.sided",
            Alternative::Less => "less",
            Alternative::Greater => "greater",
        })
    }

    fn stat(mut self, statistic: f64) -> Self {
        self.statistic = Some(statistic);
        self
    }

    fn df(mut self, df1: f64) -> Self {
        self.df1 = Some(df1);
        self
    }

    fn p(mut self, p: f64) -> Self {
        self.p_value = Some(p);
        self
    }

    fn est(mut self, estimate: f64) -> Self {
        self.estimate = Some(estimate);
        self
    }

    fn ci(mut self, low: f64, high: f64, level: Option<f64>) -> Self {
        self.conf_low = Some(low);
        self.conf_high = Some(high);
        self.conf_level = level;
        self
    }
}

/// Implements `From<T>` (owned) in terms of `From<&T>`.
macro_rules! from_owned {
    ($($t:ty),* $(,)?) => {
        $(impl From<$t> for TestResult {
            fn from(r: $t) -> Self {
                TestResult::from(&r)
            }
        })*
    };
}

from_owned!(
    TTestResult,
    YuenResult,
    OneWayAnovaResult,
    LeveneResult,
    SphericityResult,
    CorrectedResult,
    MannWhitneyResult,
    WilcoxonResult,
    KruskalResult,
    BrunnerMunzelResult,
    CorrelationResult,
    PartialCorResult,
    DistanceCorResult,
    ICCResult,
    ShapiroWilkResult,
    JarqueBeraResult,
    DAgostinoResult,
    ChiSquareResult,
    FisherResult,
    McNemarkResult,
    McNemarkExactResult,
    PropTestResult,
    BinomTestResult,
    AssociationResult,
    KappaResult,
    TostResult,
    OneSidedTestResult,
    PermutationResult,
    BootstrapCIResult,
    EnergyDistanceResult,
    MMDResult,
    DMResult,
    CWResult,
    SPAResult,
    MSPEAdjustedResult,
    MCSResult,
);

/// `estimate` is the mean difference (two-sample: `mean_x - mean_y`;
/// paired: the mean difference; one-sample: the mean). The t-test kind,
/// alternative and `n` are not stored in `TTestResult`: `method` is the
/// generic `"t-test"`.
impl From<&TTestResult> for TestResult {
    fn from(r: &TTestResult) -> Self {
        let estimate = r.mean_x - r.mean_y.unwrap_or(0.0);
        let mut t = TestResult::new("t-test")
            .stat(r.statistic)
            .df(r.df)
            .p(r.p_value)
            .est(estimate);
        if let Some(ci) = &r.conf_int {
            t = t.ci(ci.lower, ci.upper, Some(ci.conf_level));
        }
        t
    }
}

/// `estimate` is the difference in trimmed means.
impl From<&YuenResult> for TestResult {
    fn from(r: &YuenResult) -> Self {
        let mut t = TestResult::new("Yuen's trimmed means test")
            .stat(r.statistic)
            .df(r.df)
            .p(r.p_value)
            .est(r.diff);
        if let Some(ci) = &r.conf_int {
            t = t.ci(ci.lower, ci.upper, Some(ci.conf_level));
        }
        t
    }
}

/// F test with `df1 = df_between`, `df2 = df_within`; `n` is the total
/// sample size and the effect size is η² = SS_between / SS_total when the
/// sums of squares are available (Fisher's ANOVA, not Welch's).
impl From<&OneWayAnovaResult> for TestResult {
    fn from(r: &OneWayAnovaResult) -> Self {
        let mut t = TestResult::new("One-way ANOVA")
            .stat(r.statistic)
            .df(r.df_between)
            .p(r.p_value)
            .with_n(r.group_sizes.iter().sum());
        t.df2 = Some(r.df_within);
        if let (Some(b), Some(tot)) = (r.ss_between, r.ss_total) {
            t = t.with_effect_size("eta_squared", b / tot);
        }
        t
    }
}

fn anova_row(method: String, row: &AnovaTableRow, err: &AnovaTableRow) -> TestResult {
    let mut t = TestResult::new(method).df(row.df);
    t.df2 = Some(err.df);
    t.statistic = row.f_statistic;
    t.p_value = row.p_value;
    // partial eta squared: SS_effect / (SS_effect + SS_error)
    t = t.with_effect_size("partial_eta_squared", row.ss / (row.ss + err.ss));
    t
}

impl TwoWayAnovaResult {
    /// One [`TestResult`] per F test (factor A, factor B, interaction), in that
    /// order, with partial η² as effect size and `n` the number of observations.
    pub fn to_test_results(&self) -> Vec<TestResult> {
        [
            ("Two-way ANOVA: factor A", &self.factor_a),
            ("Two-way ANOVA: factor B", &self.factor_b),
            ("Two-way ANOVA: interaction", &self.interaction),
        ]
        .iter()
        .map(|(name, row)| anova_row(name.to_string(), row, &self.residual).with_n(self.n))
        .collect()
    }
}

impl RmAnovaResult {
    /// [`TestResult`]s for the within-subjects F test: uncorrected first, then
    /// the Greenhouse-Geisser and Huynh-Feldt corrected tests when computed.
    /// `n` is the number of subjects.
    pub fn to_test_results(&self) -> Vec<TestResult> {
        let n_subjects = self.subject_means.len();
        let mut out = vec![anova_row(
            "Repeated measures ANOVA".to_string(),
            &self.within_subjects,
            &self.error,
        )
        .with_n(n_subjects)];
        let eff = out[0].effect_size;
        for (name, c) in [
            ("Greenhouse-Geisser", &self.greenhouse_geisser),
            ("Huynh-Feldt", &self.huynh_feldt),
        ] {
            if let Some(c) = c {
                let mut t = TestResult::from(c)
                    .with_method(format!("Repeated measures ANOVA ({} corrected)", name))
                    .with_n(n_subjects);
                t.effect_size = eff;
                t.effect_name = Some("partial_eta_squared".to_string());
                out.push(t);
            }
        }
        out
    }
}

impl From<&LeveneResult> for TestResult {
    fn from(r: &LeveneResult) -> Self {
        let mut t = TestResult::new("Levene's test (Brown-Forsythe)")
            .stat(r.statistic)
            .df(r.df1)
            .p(r.p_value);
        t.df2 = Some(r.df2);
        t
    }
}

/// Mauchly's W is the statistic (its χ² approximation is not kept).
impl From<&SphericityResult> for TestResult {
    fn from(r: &SphericityResult) -> Self {
        TestResult::new("Mauchly's test for sphericity")
            .stat(r.w)
            .df(r.df)
            .p(r.p_value)
    }
}

impl From<&CorrectedResult> for TestResult {
    fn from(r: &CorrectedResult) -> Self {
        let mut t = TestResult::new("Sphericity-corrected F test")
            .stat(r.f_statistic)
            .df(r.df_num_corrected)
            .p(r.p_value)
            .with_effect_size("epsilon", r.epsilon);
        t.df2 = Some(r.df_den_corrected);
        t
    }
}

/// `estimate` / CI: the Hodges-Lehmann location shift, when computed.
impl From<&MannWhitneyResult> for TestResult {
    fn from(r: &MannWhitneyResult) -> Self {
        let mut t = TestResult::new("Wilcoxon rank sum test")
            .stat(r.statistic)
            .p(r.p_value);
        t.estimate = r.estimate;
        if let Some(ci) = &r.conf_int {
            t = t.ci(ci.lower, ci.upper, Some(ci.conf_level));
        }
        t
    }
}

/// `estimate` / CI: the pseudo-median, when computed.
impl From<&WilcoxonResult> for TestResult {
    fn from(r: &WilcoxonResult) -> Self {
        let mut t = TestResult::new("Wilcoxon signed rank test")
            .stat(r.statistic)
            .p(r.p_value);
        t.estimate = r.estimate;
        if let Some(ci) = &r.conf_int {
            t = t.ci(ci.lower, ci.upper, Some(ci.conf_level));
        }
        t
    }
}

impl From<&KruskalResult> for TestResult {
    fn from(r: &KruskalResult) -> Self {
        TestResult::new("Kruskal-Wallis rank sum test")
            .stat(r.statistic)
            .df(r.df)
            .p(r.p_value)
    }
}

/// `estimate` is P(X < Y) + 0.5 P(X = Y).
impl From<&BrunnerMunzelResult> for TestResult {
    fn from(r: &BrunnerMunzelResult) -> Self {
        let mut t = TestResult::new("Brunner-Munzel test")
            .stat(r.statistic)
            .df(r.df)
            .p(r.p_value)
            .est(r.estimate);
        if let Some(ci) = &r.conf_int {
            t = t.ci(ci.lower, ci.upper, Some(ci.conf_level));
        }
        t
    }
}

/// The correlation is both `estimate` and `effect_size` (named `r`, `rho`
/// or `tau`); `n` is the number of pairs.
impl From<&CorrelationResult> for TestResult {
    fn from(r: &CorrelationResult) -> Self {
        let (method, name) = match r.method {
            CorrelationMethod::Pearson => ("Pearson's product-moment correlation", "r"),
            CorrelationMethod::Spearman => ("Spearman's rank correlation rho", "rho"),
            CorrelationMethod::Kendall => ("Kendall's rank correlation tau", "tau"),
        };
        let mut t = TestResult::new(method)
            .stat(r.statistic)
            .p(r.p_value)
            .est(r.estimate)
            .with_effect_size(name, r.estimate)
            .with_n(r.n);
        t.df1 = r.df;
        if let Some(ci) = &r.conf_int {
            t = t.ci(ci.lower, ci.upper, Some(ci.conf_level));
        }
        t
    }
}

impl From<&PartialCorResult> for TestResult {
    fn from(r: &PartialCorResult) -> Self {
        TestResult::new(r.method.clone())
            .stat(r.statistic)
            .df(r.df)
            .p(r.p_value)
            .est(r.estimate)
            .with_effect_size("r", r.estimate)
            .with_n(r.n)
    }
}

impl From<&DistanceCorResult> for TestResult {
    fn from(r: &DistanceCorResult) -> Self {
        let mut t = TestResult::new(r.method.clone())
            .stat(r.statistic)
            .est(r.dcor)
            .with_effect_size("dcor", r.dcor)
            .with_n(r.n);
        t.p_value = r.p_value;
        t
    }
}

/// F test of the ICC with `df1`, `df2`; `n` is the number of subjects. The
/// confidence level of the ICC interval is not stored (`None`).
impl From<&ICCResult> for TestResult {
    fn from(r: &ICCResult) -> Self {
        let mut t = TestResult::new(r.method.clone())
            .stat(r.f_value)
            .df(r.df1)
            .p(r.p_value)
            .est(r.icc)
            .ci(r.conf_int_lower, r.conf_int_upper, None)
            .with_effect_size("icc", r.icc)
            .with_n(r.n_subjects);
        t.df2 = Some(r.df2);
        t
    }
}

impl From<&ShapiroWilkResult> for TestResult {
    fn from(r: &ShapiroWilkResult) -> Self {
        TestResult::new("Shapiro-Wilk normality test")
            .stat(r.statistic)
            .p(r.p_value)
    }
}

impl From<&JarqueBeraResult> for TestResult {
    fn from(r: &JarqueBeraResult) -> Self {
        TestResult::new("Jarque-Bera test")
            .stat(r.statistic)
            .df(2.0)
            .p(r.p_value)
            .with_n(r.n)
    }
}

impl From<&DAgostinoResult> for TestResult {
    fn from(r: &DAgostinoResult) -> Self {
        TestResult::new("D'Agostino-Pearson K² test")
            .stat(r.statistic)
            .df(2.0)
            .p(r.p_value)
    }
}

impl From<&ChiSquareResult> for TestResult {
    fn from(r: &ChiSquareResult) -> Self {
        TestResult::new(r.method.clone())
            .stat(r.statistic)
            .df(r.df)
            .p(r.p_value)
    }
}

/// `estimate` / CI: the conditional MLE odds ratio (level not stored).
impl From<&FisherResult> for TestResult {
    fn from(r: &FisherResult) -> Self {
        TestResult::new(r.method.clone())
            .with_alternative(r.alternative)
            .p(r.p_value)
            .est(r.odds_ratio)
            .ci(r.conf_int_lower, r.conf_int_upper, None)
    }
}

impl From<&McNemarkResult> for TestResult {
    fn from(r: &McNemarkResult) -> Self {
        TestResult::new(r.method.clone())
            .stat(r.statistic)
            .df(r.df)
            .p(r.p_value)
    }
}

/// `n` is the number of discordant pairs `b + c`.
impl From<&McNemarkExactResult> for TestResult {
    fn from(r: &McNemarkExactResult) -> Self {
        TestResult::new(r.method.clone())
            .p(r.p_value)
            .with_n(r.b + r.c)
    }
}

/// `estimate` is the proportion (one sample) or the difference `p1 - p2`
/// (two samples), matching the confidence interval.
impl From<&PropTestResult> for TestResult {
    fn from(r: &PropTestResult) -> Self {
        let estimate = match r.estimate.as_slice() {
            [p] => *p,
            [p1, p2, ..] => p1 - p2,
            [] => f64::NAN,
        };
        let mut t = TestResult::new(r.method.clone())
            .with_alternative(r.alternative)
            .stat(r.statistic)
            .p(r.p_value)
            .est(estimate)
            .ci(r.conf_int_lower, r.conf_int_upper, None);
        t.df1 = r.df;
        t
    }
}

/// `statistic` is the number of successes, `n` the number of trials.
impl From<&BinomTestResult> for TestResult {
    fn from(r: &BinomTestResult) -> Self {
        TestResult::new(r.method.clone())
            .with_alternative(r.alternative)
            .stat(r.successes as f64)
            .p(r.p_value)
            .est(r.estimate)
            .ci(r.conf_int_lower, r.conf_int_upper, None)
            .with_n(r.n)
    }
}

/// A measure of association without a test: no statistic or p-value.
impl From<&AssociationResult> for TestResult {
    fn from(r: &AssociationResult) -> Self {
        let mut t = TestResult::new(r.method.clone())
            .est(r.estimate)
            .with_effect_size(r.method.clone(), r.estimate);
        t.conf_low = r.conf_int_lower;
        t.conf_high = r.conf_int_upper;
        t
    }
}

/// `statistic` is the z statistic of kappa.
impl From<&KappaResult> for TestResult {
    fn from(r: &KappaResult) -> Self {
        TestResult::new(r.method.clone())
            .stat(r.z)
            .p(r.p_value)
            .est(r.kappa)
            .ci(r.conf_int_lower, r.conf_int_upper, None)
            .with_effect_size("kappa", r.kappa)
    }
}

/// TOST: `p_value` is the TOST p-value (the larger one-sided p-value), the
/// interval is the 1 - 2α interval (`conf_level = 1 - 2α`); no single
/// statistic (see `lower_test` / `upper_test`).
impl From<&TostResult> for TestResult {
    fn from(r: &TostResult) -> Self {
        let mut t = TestResult::new(r.method.clone())
            .p(r.tost_p_value)
            .est(r.estimate)
            .ci(r.ci.0, r.ci.1, Some(1.0 - 2.0 * r.alpha))
            .with_n(r.n);
        t.df1 = r.df;
        t
    }
}

impl From<&OneSidedTestResult> for TestResult {
    fn from(r: &OneSidedTestResult) -> Self {
        TestResult::new(r.hypothesis.clone())
            .stat(r.statistic)
            .p(r.p_value)
    }
}

impl From<&PermutationResult> for TestResult {
    fn from(r: &PermutationResult) -> Self {
        TestResult::new("Permutation test")
            .stat(r.statistic)
            .p(r.p_value)
    }
}

/// A bootstrap interval without a test: no statistic or p-value.
impl From<&BootstrapCIResult> for TestResult {
    fn from(r: &BootstrapCIResult) -> Self {
        TestResult::new("Bootstrap confidence interval")
            .est(r.estimate)
            .ci(r.conf_int_lower, r.conf_int_upper, Some(r.conf_level))
    }
}

impl From<&EnergyDistanceResult> for TestResult {
    fn from(r: &EnergyDistanceResult) -> Self {
        TestResult::new("Energy distance test")
            .stat(r.statistic)
            .p(r.p_value)
    }
}

impl From<&MMDResult> for TestResult {
    fn from(r: &MMDResult) -> Self {
        TestResult::new("Maximum mean discrepancy test")
            .stat(r.statistic)
            .p(r.p_value)
    }
}

impl From<&DMResult> for TestResult {
    fn from(r: &DMResult) -> Self {
        TestResult::new("Diebold-Mariano test")
            .with_alternative(r.alternative)
            .stat(r.statistic)
            .p(r.p_value)
    }
}

/// Clark-West is one-sided (`p_value`, alternative `Greater`).
impl From<&CWResult> for TestResult {
    fn from(r: &CWResult) -> Self {
        TestResult::new("Clark-West test")
            .with_alternative(Alternative::Greater)
            .stat(r.statistic)
            .p(r.p_value)
    }
}

/// `p_value` is the consistent SPA p-value.
impl From<&SPAResult> for TestResult {
    fn from(r: &SPAResult) -> Self {
        TestResult::new("Superior predictive ability test")
            .stat(r.statistic)
            .p(r.p_value_consistent)
    }
}

/// `p_value` is the consistent p-value.
impl From<&MSPEAdjustedResult> for TestResult {
    fn from(r: &MSPEAdjustedResult) -> Self {
        TestResult::new("MSPE-adjusted SPA test")
            .stat(r.statistic)
            .p(r.p_value_consistent)
    }
}

/// `p_value` is the MCS p-value; `n` is the number of models in the set.
impl From<&MCSResult> for TestResult {
    fn from(r: &MCSResult) -> Self {
        TestResult::new("Model confidence set")
            .p(r.mcs_p_value)
            .with_n(r.included_models.len())
    }
}
