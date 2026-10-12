use crate::error::{Result, StatError};
use crate::utils::dist::inv_or_nan;
use crate::utils::finite::ensure_finite;
use statrs::distribution::Normal;
use std::str::FromStr;

/// Transformation used for the pointwise Kaplan-Meier confidence interval
/// (R `survfit(conf.type = ...)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KmConfType {
    /// `exp(log S ± z se)`, upper bound capped at 1 (R's default).
    #[default]
    Log,
    /// `S^exp(± z se / log S)` (complementary log-log).
    LogLog,
    /// `S ± z S se`, clipped to `[0, 1]`.
    Plain,
}

impl FromStr for KmConfType {
    type Err = StatError;
    fn from_str(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "log" => Ok(KmConfType::Log),
            "log-log" | "loglog" | "log_log" => Ok(KmConfType::LogLog),
            "plain" => Ok(KmConfType::Plain),
            _ => Err(StatError::InvalidParameter(format!(
                "unknown conf_type '{}' (expected log, log-log or plain)",
                s
            ))),
        }
    }
}

/// Kaplan-Meier survival curve: one row per distinct observed time (event
/// or censoring), as the rows of an R `survfit` object.
#[derive(Debug, Clone, PartialEq)]
pub struct KaplanMeierResult {
    /// Distinct observed times, increasing.
    pub time: Vec<f64>,
    /// Number at risk just before each time.
    pub n_risk: Vec<usize>,
    /// Number of events at each time.
    pub n_event: Vec<usize>,
    /// Number censored at each time.
    pub n_censor: Vec<usize>,
    /// Kaplan-Meier survival estimate `S(t)`.
    pub survival: Vec<f64>,
    /// Greenwood standard error of `S(t)` (R `summary(survfit)$std.err`);
    /// NaN once `S(t) = 0`.
    pub std_err: Vec<f64>,
    /// Greenwood standard error of `log S(t)`, i.e. of the cumulative
    /// hazard (R `survfit()$std.err`); `+inf` once `S(t) = 0`.
    pub std_err_log: Vec<f64>,
    /// Lower confidence bound (NaN where R gives `NA`).
    pub conf_low: Vec<f64>,
    /// Upper confidence bound (NaN where R gives `NA`).
    pub conf_high: Vec<f64>,
    /// Interval transformation.
    pub conf_type: KmConfType,
    /// Confidence level.
    pub conf_level: f64,
    /// Number of observations.
    pub n: usize,
}

/// Kaplan-Meier estimator with Greenwood variance, as R
/// `survfit(Surv(time, event) ~ 1, conf.type, conf.int)`.
///
/// Ties are handled as in R: subjects censored at an event time are still at
/// risk at that time. The pointwise interval uses `std_err_log`; where it is
/// 0 (before the first event) the interval collapses to `S(t)`.
///
/// # Errors
/// Empty input, unequal lengths, non-finite times, or `conf_level` outside
/// `(0, 1)`.
pub fn kaplan_meier(
    time: &[f64],
    event: &[bool],
    conf_type: KmConfType,
    conf_level: f64,
) -> Result<KaplanMeierResult> {
    if time.is_empty() {
        return Err(StatError::EmptyData);
    }
    if time.len() != event.len() {
        return Err(StatError::InvalidParameter(format!(
            "time and event must have equal length ({} vs {})",
            time.len(),
            event.len()
        )));
    }
    ensure_finite("time", time)?;
    if !(conf_level > 0.0 && conf_level < 1.0) {
        return Err(StatError::InvalidParameter(format!(
            "conf_level must be in (0, 1), got {}",
            conf_level
        )));
    }
    let z = inv_or_nan(&Normal::standard(), 1.0 - (1.0 - conf_level) / 2.0);
    let mut idx: Vec<usize> = (0..time.len()).collect();
    idx.sort_by(|&a, &b| time[a].total_cmp(&time[b]));

    let n = time.len();
    let mut res = KaplanMeierResult {
        time: Vec::new(),
        n_risk: Vec::new(),
        n_event: Vec::new(),
        n_censor: Vec::new(),
        survival: Vec::new(),
        std_err: Vec::new(),
        std_err_log: Vec::new(),
        conf_low: Vec::new(),
        conf_high: Vec::new(),
        conf_type,
        conf_level,
        n,
    };
    let mut at_risk = n;
    let mut surv = 1.0;
    let mut green = 0.0;
    let mut i = 0;
    while i < n {
        let t = time[idx[i]];
        let mut d = 0;
        let mut c = 0;
        while i < n && time[idx[i]] == t {
            if event[idx[i]] {
                d += 1;
            } else {
                c += 1;
            }
            i += 1;
        }
        if d > 0 {
            surv *= 1.0 - d as f64 / at_risk as f64;
            green += if d == at_risk {
                f64::INFINITY
            } else {
                d as f64 / (at_risk as f64 * (at_risk - d) as f64)
            };
        }
        let se = green.sqrt();
        let (lo, hi) = km_ci(surv, se, z, conf_type);
        res.time.push(t);
        res.n_risk.push(at_risk);
        res.n_event.push(d);
        res.n_censor.push(c);
        res.survival.push(surv);
        res.std_err_log.push(se);
        res.std_err
            .push(if surv == 0.0 { f64::NAN } else { surv * se });
        res.conf_low.push(lo);
        res.conf_high.push(hi);
        at_risk -= d + c;
    }
    Ok(res)
}

/// R `survival:::survfit_confint` with `logse = TRUE`.
fn km_ci(p: f64, se: f64, z: f64, conf_type: KmConfType) -> (f64, f64) {
    if p == 0.0 {
        return (f64::NAN, f64::NAN);
    }
    if se == 0.0 {
        return (p, p);
    }
    match conf_type {
        KmConfType::Plain => {
            let h = se * p * z;
            ((p - h).max(0.0), (p + h).min(1.0))
        }
        KmConfType::Log => {
            let h = z * se;
            ((p.ln() - h).exp(), (p.ln() + h).exp().min(1.0))
        }
        KmConfType::LogLog => {
            if p == 1.0 {
                return (f64::NAN, f64::NAN);
            }
            let h = z * se / p.ln();
            let c = (-p.ln()).ln();
            ((-(c - h).exp()).exp(), (-(c + h).exp()).exp())
        }
    }
}
