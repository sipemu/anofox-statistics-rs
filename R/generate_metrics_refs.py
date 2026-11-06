# Reference values for the metric pack (anofox_statistics::metrics):
# classification and regression metrics from scikit-learn, NDCG from
# scikit-learn's ndcg_score, and the other ranking metrics from evidently's
# legacy recsys metrics.
#
#   python3 -m venv /tmp/ev && /tmp/ev/bin/pip install evidently==0.7.23 \
#       scikit-learn==1.9.1
#   /tmp/ev/bin/python R/generate_metrics_refs.py      (from the crate root)
#
# Writes R/data/metrics_sklearn_reference.csv (case, key, values; shortest
# round-trip floats, space separated). R/generate_metrics_refs.R cross-checks
# the classification and regression metrics with yardstick.
import warnings

import numpy as np
import pandas as pd
from sklearn import metrics as skm

warnings.filterwarnings("ignore")
rows = []


def add(case, key, values):
    v = np.atleast_1d(np.asarray(values, dtype=float))
    rows.append((case, key, " ".join(repr(float(x)) for x in v)))


rng = np.random.default_rng(368)

# --------------------------------------------------------- classification
def clf_case(case, yt, yp, averages, zero_divisions=(0.0, 1.0)):
    add(case, "y_true", yt)
    add(case, "y_pred", yp)
    add(case, "accuracy", skm.accuracy_score(yt, yp))
    add(case, "balanced_accuracy", skm.balanced_accuracy_score(yt, yp))
    add(case, "mcc", skm.matthews_corrcoef(yt, yp))
    add(case, "kappa", skm.cohen_kappa_score(yt, yp))
    add(case, "confusion", skm.confusion_matrix(yt, yp).ravel())
    for avg in averages:
        for zd in zero_divisions:
            kw = dict(average=avg, zero_division=zd)
            if avg == "binary":
                kw["pos_label"] = 1
            key = f"{avg}_zd{int(zd)}"
            add(case, key, [skm.precision_score(yt, yp, **kw), skm.recall_score(yt, yp, **kw),
                            skm.f1_score(yt, yp, **kw)])


n = 200
yt = (rng.uniform(size=n) < 0.35).astype(int)
yp = np.where(rng.uniform(size=n) < 0.8, yt, 1 - yt)
clf_case("binary", yt, yp, ["binary", "macro", "micro", "weighted"])
yt = rng.choice(4, size=150, p=[0.4, 0.3, 0.2, 0.1])
yp = np.where(rng.uniform(size=150) < 0.6, yt, rng.choice(3, size=150))  # class 3 never predicted
clf_case("multiclass", yt, yp, ["macro", "micro", "weighted"])
yt = np.array([0, 0, 1, 1, 2, 2, 2])
yp = np.array([0, 3, 1, 3, 2, 2, 0])  # label 3 only predicted (no support)
clf_case("multiclass_unsupported", yt, yp, ["macro", "micro", "weighted"])
clf_case("constant_pred", np.array([0, 1, 1, 0, 1]), np.array([1, 1, 1, 1, 1]),
         ["binary", "macro"])

# ------------------------------------------------------------ regression
y = rng.normal(3, 2, size=60)
y[7] = 0.0  # MAPE uses max(|y|, eps)
yhat = y + rng.normal(0, 0.8, size=60)
reg = {
    "mae": skm.mean_absolute_error(y, yhat),
    "mse": skm.mean_squared_error(y, yhat),
    "rmse": skm.root_mean_squared_error(y, yhat),
    "mape": skm.mean_absolute_percentage_error(y, yhat),
    "median_ae": skm.median_absolute_error(y, yhat),
    "r2": skm.r2_score(y, yhat),
    "explained_variance": skm.explained_variance_score(y, yhat),
    "max_error": skm.max_error(y, yhat),
}
add("regression", "y_true", y)
add("regression", "y_pred", yhat)
for k, v in reg.items():
    add("regression", k, v)
add("regression_const", "r2", [skm.r2_score([2, 2, 2], [2, 2, 2]), skm.r2_score([2, 2, 2], [2, 2, 3])])
add("regression_const", "explained_variance",
    [skm.explained_variance_score([2, 2, 2], [2, 2, 2]),
     skm.explained_variance_score([2, 2, 2], [1, 2, 3])])

# --------------------------------------------------------------- ranking
from evidently.legacy.metrics import (HitRateKMetric, MAPKMetric, MRRKMetric, NDCGKMetric,
                                      PrecisionTopKMetric, RecallTopKMetric)
from evidently.legacy.pipeline.column_mapping import ColumnMapping
from evidently.legacy.report import Report

recs = []
for q in range(30):
    m = int(rng.integers(5, 16))
    rel = rng.choice([0, 0, 0, 1, 2, 3], size=m)
    if q % 7 == 3:
        rel[:] = 0  # no-feedback queries
    score = rng.uniform(size=m)
    for i in range(m):
        recs.append((q, i, rel[i], score[i]))
df = pd.DataFrame(recs, columns=["user_id", "item_id", "target", "score"])
# Ranks by decreasing score; evidently's "last" aggregation needs rank order.
df["rank"] = df.groupby("user_id")["score"].rank(ascending=False).astype(int)
df = df.sort_values(["user_id", "rank"]).reset_index(drop=True)
add("ranking", "query", df["user_id"])
add("ranking", "relevance", df["target"])
add("ranking", "score", df["score"])


def ev(k, min_rel, no_feedback):
    cm = ColumnMapping(recommendations_type="rank", target="target", prediction="rank",
                       user_id="user_id", item_id="item_id")
    ms = [NDCGKMetric(k=k, min_rel_score=min_rel, no_feedback_users=no_feedback),
          MAPKMetric(k=k, min_rel_score=min_rel, no_feedback_users=no_feedback),
          MRRKMetric(k=k, min_rel_score=min_rel, no_feedback_users=no_feedback),
          PrecisionTopKMetric(k=k, min_rel_score=min_rel, no_feedback_users=no_feedback),
          RecallTopKMetric(k=k, min_rel_score=min_rel, no_feedback_users=no_feedback),
          HitRateKMetric(k=k, min_rel_score=min_rel, no_feedback_users=no_feedback)]
    rep = Report(metrics=ms)
    rep.run(reference_data=None, current_data=df[["user_id", "item_id", "target", "rank"]],
            column_mapping=cm)
    out = []
    for r in rep.as_dict()["metrics"]:
        cur = pd.Series(r["result"]["current"])
        cur.index = cur.index.astype(int)
        # evidently's own rule (TopKMetricRenderer): 0-based series hold k at k - 1.
        out.append(float(cur[k - 1] if cur.index.min() == 0 else cur[k]))
    return out


def sk_ndcg(k, min_rel, include_no_feedback, scores):
    vals = []
    for q, g in df.assign(s=scores).groupby("user_id"):
        t = g["target"].to_numpy(dtype=float)
        if min_rel is not None:
            t = (t >= min_rel).astype(float)
        if t.sum() == 0 and not include_no_feedback:
            continue
        vals.append(skm.ndcg_score([t], [g["s"].to_numpy()], k=k) if len(t) > 1 else
                    (1.0 if t.sum() > 0 else 0.0))
    return np.mean(vals)


for k in [3, 5]:
    for min_rel in [None, 2]:
        for nf in [False, True]:
            key = f"k{k}_min{min_rel or 0}_nf{int(nf)}"
            add("ranking", "evidently_" + key, ev(k, min_rel, nf))
            add("ranking", "sklearn_ndcg_" + key, sk_ndcg(k, min_rel, nf, df["score"]))
# Tied scores (NDCG tie averaging).
tied = np.round(df["score"].to_numpy() * 4) / 4
add("ranking", "tied_score", tied)
for k in [3, 5]:
    add("ranking", f"sklearn_ndcg_tied_k{k}", sk_ndcg(k, None, True, tied))

pd.DataFrame(rows, columns=["case", "key", "values"]).to_csv(
    "R/data/metrics_sklearn_reference.csv", index=False)
print(len(rows), "rows")
