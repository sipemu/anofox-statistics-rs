# Reference values for the monitoring module (CBPE, PCA reconstruction-error
# drift, reference-band thresholds) from NannyML, and the isotonic / Platt
# calibrators from scikit-learn.
#
#   python3 -m venv /tmp/nml && /tmp/nml/bin/pip install nannyml==0.13.1 \
#       scikit-learn==1.9.1 statsmodels
#   /tmp/nml/bin/python R/generate_monitoring_refs.py   (from the crate root)
#
# Writes R/data/monitoring_nannyml_reference.csv (case, key, values; shortest
# round-trip floats, space separated; booleans 0/1; a missing threshold is NaN). NannyML has no R
# counterpart; R/generate_monitoring_refs.R re-derives the calibrators and the
# PCA reconstruction errors with isoreg / glm / prcomp.
import warnings

import numpy as np
import pandas as pd
import nannyml as nml
from sklearn.calibration import _sigmoid_calibration
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
rows = []


def add(case, key, values):
    v = np.atleast_1d(np.asarray(values, dtype=float))
    rows.append((case, key, " ".join(repr(float(x)) for x in v)))


def nan_none(x):
    return np.nan if x is None else x


# ---------------------------------------------------------------- CBPE
CBPE_METRICS = ["roc_auc", "f1", "precision", "recall", "specificity", "accuracy"]
CM = ["true_positive", "false_positive", "false_negative", "true_negative"]


def cbpe_case(case, ref, ana, chunk_size):
    add(case, "ref_score", ref["y_pred_proba"])
    add(case, "ref_pred", ref["y_pred"])
    add(case, "ref_label", ref["y_true"])
    add(case, "ana_score", ana["y_pred_proba"])
    add(case, "ana_pred", ana["y_pred"])
    add(case, "ana_label", ana["y_true"])
    add(case, "chunk_size", chunk_size)
    est = nml.CBPE(
        y_pred_proba="y_pred_proba",
        y_pred="y_pred",
        y_true="y_true",
        problem_type="classification_binary",
        metrics=CBPE_METRICS + ["confusion_matrix"],
        chunk_size=chunk_size,
    )
    est.fit(ref)
    add(case, "needs_calibration", float(est.needs_calibration))
    res = est.estimate(ana).to_df()
    res = res[res[("chunk", "period")] == "analysis"]
    for m in CBPE_METRICS + CM:
        for col in ["value", "sampling_error", "realized", "upper_confidence_boundary",
                    "lower_confidence_boundary", "upper_threshold", "lower_threshold",
                    "alert"]:
            add(case, f"{m}_{col}", res[(m, col)].astype(float))
    # The calibrated scores of the analysis rows (isotonic, as NannyML fits it).
    iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
    iso.fit(ref["y_pred_proba"], ref["y_true"])
    add(case, "ana_calibrated", iso.predict(ana["y_pred_proba"]))


r, a, t = nml.load_synthetic_binary_classification_dataset()
a = a.merge(t, on="id")
r = r.rename(columns={"work_home_actual": "y_true"})
a = a.rename(columns={"work_home_actual": "y_true"})
cbpe_case("cbpe_synthetic", r.iloc[:5000].reset_index(drop=True),
          a.iloc[:5000].reset_index(drop=True), 1000)

# Continuous, miscalibrated scores (no ties) with drift in the analysis period.
rng = np.random.default_rng(20261106)


def sim(n, shift):
    x = rng.normal(shift, 1.0, n)
    p_true = 1 / (1 + np.exp(-(1.5 * x - 0.7)))
    y = (rng.uniform(size=n) < p_true).astype(int)
    score = 1 / (1 + np.exp(-(0.8 * x + 0.3)))  # miscalibrated model score
    return pd.DataFrame({"y_pred_proba": score, "y_pred": (score >= 0.5).astype(int),
                         "y_true": y})


cbpe_case("cbpe_continuous", sim(3000, 0.0), pd.concat([sim(1500, 0.0), sim(1500, 0.8)],
          ignore_index=True), 500)

# ---------------------------------------------------------- calibrators
s = rng.uniform(size=400) ** 1.5
yb = (rng.uniform(size=400) < np.sqrt(s)).astype(int)
s = np.round(s, 2)  # ties
newx = np.r_[-0.5, np.linspace(0, 1, 41), 1.5]
add("isotonic", "x", s)
add("isotonic", "y", yb)
add("isotonic", "new", newx)
add("isotonic", "pred", IsotonicRegression(out_of_bounds="clip").fit(s, yb).predict(newx))
sc = rng.normal(size=300)
yp = (rng.uniform(size=300) < 1 / (1 + np.exp(-(2 * sc + 0.5)))).astype(int)
A, B = _sigmoid_calibration(sc, yp)
add("platt", "x", sc)
add("platt", "y", yp)
add("platt", "ab", [A, B])

# ----------------------------------------- PCA reconstruction-error drift
COLS = ["car_value", "debt_to_income_ratio", "loan_length", "driver_tenure"]


def dre_case(case, ref, ana, chunk_size, n_components, store_inputs=True):
    if store_inputs:  # otherwise the test reuses the inputs of "dre_car_loan"
        add(case, "ref_x", ref[COLS].to_numpy().ravel())
        add(case, "ana_x", ana[COLS].to_numpy().ravel())
    add(case, "dims", [len(COLS)])
    add(case, "chunk_size", chunk_size)
    add(case, "n_components_option", n_components)
    calc = nml.DataReconstructionDriftCalculator(column_names=COLS, chunk_size=chunk_size,
                                                 n_components=n_components)
    calc.fit(ref)
    res = calc.calculate(ana).to_df()
    for period in ["reference", "analysis"]:
        p = res[res[("chunk", "period")] == period]
        for col in ["value", "sampling_error", "upper_confidence_boundary",
                    "lower_confidence_boundary", "upper_threshold", "lower_threshold", "alert"]:
            add(case, f"{period}_{col}", p[("reconstruction_error", col)].astype(float))
    add(case, "n_components", calc._pca.n_components_)
    means = ref[COLS].mean()  # NannyML's SimpleImputer(strategy="mean")
    sc = StandardScaler().fit(ref[COLS].fillna(means))
    pca = PCA(n_components=n_components, random_state=16).fit(sc.transform(ref[COLS].fillna(means)))
    z = sc.transform(ana[COLS].iloc[:20].fillna(means))
    add(case, "row_errors", np.linalg.norm(z - pca.inverse_transform(pca.transform(z)), axis=1))


cl = nml.load_synthetic_car_loan_dataset()
ref_cl = cl[0].iloc[:5000].reset_index(drop=True)
ana_cl = cl[1].iloc[:5000].reset_index(drop=True)
dre_case("dre_car_loan", ref_cl, ana_cl, 1000, 0.65)
dre_case("dre_car_loan_k2", ref_cl, ana_cl, 1000, 2, store_inputs=False)
# Missing values, imputed by the reference column means.
ref_nan, ana_nan = ref_cl.copy(), ana_cl.copy()
ref_nan.loc[::37, "car_value"] = np.nan
ref_nan.loc[5::53, "driver_tenure"] = np.nan
ana_nan.loc[::11, "debt_to_income_ratio"] = np.nan
ana_nan.loc[3::17, "car_value"] = np.nan
dre_case("dre_car_loan_nan", ref_nan, ana_nan, 1000, 0.65)

# ------------------------------------------------------------ thresholds
vals = np.array([0.91, 0.93, np.nan, 0.92, 0.95, 0.90])
add("threshold_std", "values", vals)
for lo, up, name in [(3, 3, "k3"), (2, None, "k2_lower_only"), (1.5, 0.5, "k15_05")]:
    lower, upper = nml.thresholds.StandardDeviationThreshold(lo, up).thresholds(vals)
    add("threshold_std", name, [nan_none(lower), nan_none(upper)])
    lower, upper = nml.thresholds.calculate_threshold_values(
        nml.thresholds.StandardDeviationThreshold(lo, up), vals,
        lower_threshold_value_limit=0.9, upper_threshold_value_limit=0.96)
    add("threshold_std", name + "_limited", [nan_none(lower), nan_none(upper)])
    lower, upper = nml.thresholds.calculate_threshold_values(
        nml.thresholds.StandardDeviationThreshold(lo, up), vals,
        lower_threshold_value_limit=0.9, upper_threshold_value_limit=0.96,
        override_using_none=True)
    add("threshold_std", name + "_limited_none", [nan_none(lower), nan_none(upper)])

out = pd.DataFrame(rows, columns=["case", "key", "values"])
out.to_csv("R/data/monitoring_nannyml_reference.csv", index=False)
print(len(out), "rows")
