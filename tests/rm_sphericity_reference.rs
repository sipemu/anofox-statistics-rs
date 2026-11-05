//! Sphericity test (Mauchly) and Greenhouse-Geisser / Huynh-Feldt corrections
//! of the repeated-measures ANOVA against afex::aov_ez and stats::mauchly.test
//! (k = 3 and k = 4, two datasets each). Reference data:
//! `R/generate_rm_sphericity_refs.R` -> `R/data/rm_sphericity_reference.csv`.

use anofox_statistics::repeated_measures_anova;
use std::collections::HashMap;

type Refs = HashMap<(String, String), Vec<f64>>;

fn refs() -> Refs {
    let p = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/R/data/rm_sphericity_reference.csv"
    );
    let mut m = HashMap::new();
    for r in csv::Reader::from_path(p).expect("reference file").records() {
        let r = r.unwrap();
        let v = r[2]
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        m.insert((r[0].to_string(), r[1].to_string()), v);
    }
    m
}

fn get<'a>(m: &'a Refs, case: &str, key: &str) -> &'a [f64] {
    m.get(&(case.to_string(), key.to_string()))
        .unwrap_or_else(|| panic!("missing {} {}", case, key))
}

fn close(got: f64, want: f64, tol: f64, what: &str) {
    assert!(
        (got - want).abs() <= tol * want.abs().max(1e-300),
        "{}: {} vs {}",
        what,
        got,
        want
    );
}

#[test]
fn sphericity_matches_afex_and_mauchly_test() {
    let m = refs();
    for case in ["k3_a", "k3_b", "k4_a", "k4_b"] {
        let dims = get(&m, case, "dims");
        let (n, k) = (dims[0] as usize, dims[1] as usize);
        let flat = get(&m, case, "in_data");
        let rows: Vec<&[f64]> = (0..n).map(|i| &flat[i * k..(i + 1) * k]).collect();
        let r = repeated_measures_anova(&rows, true).unwrap();
        close(
            r.within_subjects.f_statistic.unwrap(),
            get(&m, case, "F")[0],
            1e-10,
            &format!("{} F", case),
        );
        let sph = r.sphericity.as_ref().unwrap();
        let ms = get(&m, case, "mauchly_stats");
        close(sph.w, ms[0], 1e-9, &format!("{} W", case));
        close(sph.p_value, ms[1], 1e-8, &format!("{} Mauchly p", case));
        let ma = get(&m, case, "mauchly_afex");
        close(sph.w, ma[0], 1e-9, &format!("{} W (afex)", case));
        close(
            sph.p_value,
            ma[1],
            1e-8,
            &format!("{} Mauchly p (afex)", case),
        );

        let gg = r.greenhouse_geisser.as_ref().unwrap();
        let want = get(&m, case, "gg");
        close(gg.epsilon, want[0], 1e-9, &format!("{} GG eps", case));
        close(gg.p_value, want[1], 1e-8, &format!("{} GG p", case));
        let hf = r.huynh_feldt.as_ref().unwrap();
        let want = get(&m, case, "hf");
        // afex reports the raw HF epsilon and uses min(1, eps) for the p-value.
        close(
            hf.epsilon,
            want[0].min(1.0),
            1e-9,
            &format!("{} HF eps", case),
        );
        close(hf.p_value, want[1], 1e-8, &format!("{} HF p", case));
    }
}
