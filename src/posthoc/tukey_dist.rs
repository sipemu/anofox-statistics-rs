//! Studentized range distribution: ports of R's `ptukey.c` and `qtukey.c`
//! (Copenhaver & Holland 1988, AS 190), which R's `TukeyHSD` uses.

// Quadrature constants are copied verbatim from R's sources.
#![allow(clippy::excessive_precision)]

use statrs::function::gamma::ln_gamma;
use std::f64::consts::{LN_2, SQRT_2};

const M_1_SQRT_2PI: f64 = 0.398_942_280_401_432_677_939_946_059_934;

/// Standard normal CDF, Φ(x) = erfc(-x/√2)/2.
fn pnorm(x: f64) -> f64 {
    0.5 * statrs::function::erf::erfc(-x / SQRT_2)
}

/// Probability integral of Hartley's form of the range (R's `wprob`).
fn wprob(w: f64, rr: f64, cc: f64) -> f64 {
    const NLEG: usize = 12;
    const IHALF: usize = 6;
    const C1: f64 = -30.0;
    const C2: f64 = -50.0;
    const C3: f64 = 60.0;
    const BB: f64 = 8.0;
    const WLAR: f64 = 3.0;
    const WINCR1: f64 = 2.0;
    const WINCR2: f64 = 3.0;
    const XLEG: [f64; IHALF] = [
        0.981_560_634_246_719_250_690_549_090_149,
        0.904_117_256_370_474_856_678_465_866_119,
        0.769_902_674_194_304_687_036_893_833_213,
        0.587_317_954_286_617_447_296_702_418_941,
        0.367_831_498_998_180_193_752_691_536_644,
        0.125_233_408_511_468_915_472_441_369_464,
    ];
    const ALEG: [f64; IHALF] = [
        0.047_175_336_386_511_827_194_615_961_485,
        0.106_939_325_995_318_430_960_254_718_194,
        0.160_078_328_543_346_226_334_652_529_543,
        0.203_167_426_723_065_921_749_064_455_810,
        0.233_492_536_538_354_808_760_849_898_925,
        0.249_147_045_813_402_785_000_562_436_043,
    ];

    let qsqz = w * 0.5;
    if qsqz >= BB {
        return 1.0;
    }
    let mut pr_w = 2.0 * pnorm(qsqz) - 1.0;
    if pr_w >= (C2 / cc).exp() {
        pr_w = pr_w.powf(cc);
    } else {
        pr_w = 0.0;
    }
    let wincr = if w > WLAR { WINCR1 } else { WINCR2 };

    let mut blb = qsqz;
    let binc = (BB - qsqz) / wincr;
    let mut bub = blb + binc;
    let mut einsum = 0.0;
    let cc1 = cc - 1.0;
    let mut wi = 1.0;
    while wi <= wincr {
        let mut elsum = 0.0;
        let a = 0.5 * (bub + blb);
        let b = 0.5 * (bub - blb);
        for jj in 1..=NLEG {
            let (j, xx) = if IHALF < jj {
                let j = NLEG - jj + 1;
                (j, XLEG[j - 1])
            } else {
                (jj, -XLEG[jj - 1])
            };
            let c = b * xx;
            let ac = a + c;
            let qexpo = ac * ac;
            if qexpo > C3 {
                break;
            }
            let pplus = 2.0 * pnorm(ac);
            let pminus = 2.0 * pnorm(ac - w);
            let mut rinsum = pplus * 0.5 - pminus * 0.5;
            if rinsum >= (C1 / cc1).exp() {
                rinsum = (ALEG[j - 1] * (-(0.5 * qexpo)).exp()) * rinsum.powf(cc1);
                elsum += rinsum;
            }
        }
        elsum *= ((2.0 * b) * cc) * M_1_SQRT_2PI;
        einsum += elsum;
        blb = bub;
        bub += binc;
        wi += 1.0;
    }
    pr_w += einsum;
    if pr_w <= (C1 / rr).exp() {
        return 0.0;
    }
    pr_w = pr_w.powf(rr);
    if pr_w >= 1.0 {
        return 1.0;
    }
    pr_w
}

/// Lower-tail CDF of the studentized range, P(Q <= q), for `nmeans` means,
/// `df` error degrees of freedom and `nranges` ranges (1 for Tukey HSD).
fn ptukey_lower(q: f64, nranges: f64, nmeans: f64, df: f64) -> f64 {
    const NLEGQ: usize = 16;
    const IHALFQ: usize = 8;
    const EPS1: f64 = -30.0;
    const EPS2: f64 = 1.0e-14;
    const DHAF: f64 = 100.0;
    const DQUAR: f64 = 800.0;
    const DEIGH: f64 = 5000.0;
    const DLARG: f64 = 25000.0;
    const XLEGQ: [f64; IHALFQ] = [
        0.989_400_934_991_649_932_596_154_173_450,
        0.944_575_023_073_232_576_077_988_415_535,
        0.865_631_202_387_831_743_880_467_897_712,
        0.755_404_408_355_003_033_895_101_194_847,
        0.617_876_244_402_643_748_446_671_764_049,
        0.458_016_777_657_227_386_342_419_442_984,
        0.281_603_550_779_258_913_230_460_501_460,
        0.950_125_098_376_374_401_853_193_354_250e-1,
    ];
    const ALEGQ: [f64; IHALFQ] = [
        0.271_524_594_117_540_948_517_805_724_560e-1,
        0.622_535_239_386_478_928_628_438_369_944e-1,
        0.951_585_116_824_927_848_099_251_076_022e-1,
        0.124_628_971_255_533_872_052_476_282_192,
        0.149_595_988_816_576_732_081_501_730_547,
        0.169_156_519_395_002_538_189_312_079_030,
        0.182_603_415_044_923_588_866_763_667_969,
        0.189_450_610_455_068_496_285_396_723_208,
    ];
    let (rr, cc) = (nranges, nmeans);

    if df > DLARG {
        return wprob(q, rr, cc);
    }
    let f2 = df * 0.5;
    let mut f2lf = ((f2 * df.ln()) - (df * LN_2)) - ln_gamma(f2);
    let f21 = f2 - 1.0;
    let ff4 = df * 0.25;
    let ulen: f64 = if df <= DHAF {
        1.0
    } else if df <= DQUAR {
        0.5
    } else if df <= DEIGH {
        0.25
    } else {
        0.125
    };
    f2lf += ulen.ln();

    let mut ans = 0.0;
    for i in 1..=50 {
        let mut otsum = 0.0;
        let twa1 = (2 * i - 1) as f64 * ulen;
        for jj in 1..=NLEGQ {
            let (j, upper) = if IHALFQ < jj {
                (jj - IHALFQ - 1, true)
            } else {
                (jj - 1, false)
            };
            let t1 = if upper {
                (f2lf + (f21 * (twa1 + (XLEGQ[j] * ulen)).ln()))
                    - (((XLEGQ[j] * ulen) + twa1) * ff4)
            } else {
                (f2lf + (f21 * (twa1 - (XLEGQ[j] * ulen)).ln()))
                    + (((XLEGQ[j] * ulen) - twa1) * ff4)
            };
            if t1 >= EPS1 {
                let qsqz = if upper {
                    q * (((XLEGQ[j] * ulen) + twa1) * 0.5).sqrt()
                } else {
                    q * (((-(XLEGQ[j] * ulen)) + twa1) * 0.5).sqrt()
                };
                let wprb = wprob(qsqz, rr, cc);
                otsum += (wprb * ALEGQ[j]) * t1.exp();
            }
        }
        if i as f64 * ulen >= 1.0 && otsum <= EPS2 {
            break;
        }
        ans += otsum;
    }
    ans.min(1.0)
}

/// Distribution function of the studentized range statistic, a port of R's
/// `ptukey(q, nmeans, df, nranges = 1, lower.tail)`.
///
/// Returns NaN for NaN arguments or invalid parameters (`df < 2`,
/// `nranges < 1`, `nmeans < 2`). Accuracy is that of R's algorithm (about
/// 1e-14 absolute in the lower tail); the upper tail is computed as
/// `1 - lower`, again as in R.
pub fn ptukey(q: f64, nmeans: f64, df: f64, nranges: f64, lower_tail: bool) -> f64 {
    if q.is_nan() || nranges.is_nan() || nmeans.is_nan() || df.is_nan() {
        return f64::NAN;
    }
    let tail = |p: f64| if lower_tail { p } else { 0.5 - p + 0.5 };
    if q <= 0.0 {
        return tail(0.0);
    }
    if df < 2.0 || nranges < 1.0 || nmeans < 2.0 {
        return f64::NAN;
    }
    if !q.is_finite() {
        return tail(1.0);
    }
    tail(ptukey_lower(q, nranges, nmeans, df))
}

/// Initial approximation for `qtukey` (R's `qinv`).
fn qinv(p: f64, c: f64, v: f64) -> f64 {
    const P0: f64 = 0.322232421088;
    const Q0: f64 = 0.993484626060e-01;
    const P1: f64 = -1.0;
    const Q1: f64 = 0.588581570495;
    const P2: f64 = -0.342242088547;
    const Q2: f64 = 0.531103462366;
    const P3: f64 = -0.204231210125;
    const Q3: f64 = 0.103537752850;
    const P4: f64 = -0.453642210148e-04;
    const Q4: f64 = 0.38560700634e-02;
    const C1: f64 = 0.8832;
    const C2: f64 = 0.2368;
    const C3: f64 = 1.214;
    const C4: f64 = 1.208;
    #[allow(clippy::approx_constant)] // R's truncated constant, kept for parity
    const C5: f64 = 1.4142;
    const VMAX: f64 = 120.0;

    let ps = 0.5 - 0.5 * p;
    let yi = (1.0 / (ps * ps)).ln().sqrt();
    let mut t = yi
        + ((((yi * P4 + P3) * yi + P2) * yi + P1) * yi + P0)
            / ((((yi * Q4 + Q3) * yi + Q2) * yi + Q1) * yi + Q0);
    if v < VMAX {
        t += (t * t * t + t) / v / 4.0;
    }
    let mut q = C1 - C2 * t;
    if v < VMAX {
        q += -C3 / v + C4 * t / v;
    }
    t * (q * (c - 1.0).ln() + C5)
}

/// Quantile function of the studentized range statistic, a port of R's
/// `qtukey(p, nmeans, df, nranges = 1, lower.tail)`.
///
/// Like R, this uses a secant iteration that stops once successive iterates
/// differ by less than 1e-4, so the result matches R's `qtukey` (not the
/// exact quantile) to floating-point accuracy.
pub fn qtukey(p: f64, nmeans: f64, df: f64, nranges: f64, lower_tail: bool) -> f64 {
    const EPS: f64 = 0.0001;
    const MAXITER: usize = 50;
    if p.is_nan() || nranges.is_nan() || nmeans.is_nan() || df.is_nan() {
        return f64::NAN;
    }
    if df < 2.0 || nranges < 1.0 || nmeans < 2.0 {
        return f64::NAN;
    }
    // R_Q_P01_boundaries(p, 0, ML_POSINF)
    if !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    let (at_zero, at_one) = if lower_tail { (0.0, 1.0) } else { (1.0, 0.0) };
    if p == at_zero {
        return 0.0;
    }
    if p == at_one {
        return f64::INFINITY;
    }
    let p = if lower_tail { p } else { 0.5 - p + 0.5 };

    let mut x0 = qinv(p, nmeans, df);
    let mut valx0 = ptukey(x0, nmeans, df, nranges, true) - p;
    let mut x1 = if valx0 > 0.0 {
        (x0 - 1.0).max(0.0)
    } else {
        x0 + 1.0
    };
    let mut valx1 = ptukey(x1, nmeans, df, nranges, true) - p;
    let mut ans = 0.0;
    for _ in 1..MAXITER {
        ans = x1 - ((valx1 * (x1 - x0)) / (valx1 - valx0));
        valx0 = valx1;
        x0 = x1;
        if ans < 0.0 {
            ans = 0.0;
        }
        valx1 = ptukey(ans, nmeans, df, nranges, true) - p;
        x1 = ans;
        if (x1 - x0).abs() < EPS {
            return ans;
        }
    }
    ans
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boundaries() {
        assert_eq!(ptukey(0.0, 3.0, 10.0, 1.0, true), 0.0);
        assert_eq!(ptukey(-1.0, 3.0, 10.0, 1.0, false), 1.0);
        assert_eq!(ptukey(f64::INFINITY, 3.0, 10.0, 1.0, true), 1.0);
        assert!(ptukey(1.0, 1.0, 10.0, 1.0, true).is_nan());
        assert!(ptukey(1.0, 3.0, 1.0, 1.0, true).is_nan());
        assert_eq!(qtukey(0.0, 3.0, 10.0, 1.0, true), 0.0);
        assert_eq!(qtukey(1.0, 3.0, 10.0, 1.0, true), f64::INFINITY);
        assert!(qtukey(1.5, 3.0, 10.0, 1.0, true).is_nan());
    }
}
