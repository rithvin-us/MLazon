"""Hunt for information the model does not use: on labelled val pairs, fit XGB with base_margin = logit(current p)
on raw-data properties the pipeline never looks at. Out-of-fold (by S1 block) logloss vs the current p tells whether
any of them carries residual signal; gain per feature says which. Raw strings, ids and file row positions only."""
import sys

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.model_selection import GroupKFold

sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
import pipeline as pp  # noqa: E402

R = pp.RUNS_DIR
ce = sorted(R.glob("*-v10_ce"))[-1]
vt = pl.read_parquet(R / "20260926-141000-v9" / "val_truth.parquet").select("s1_idx", "block", "country_n")
va = pl.read_parquet(ce / "val_scored_ce.parquet").join(vt, on="s1_idx")
s1n = pp.scan_norm("train", "s1").select(pl.col("idx").alias("s1_idx"), pl.col("entity_id").alias("sid")).collect()
pn = pp.scan_norm("train", "pool").select(pl.col("idx").alias("cand_idx"), pl.col("entity_id").alias("cid")).collect()
va = va.join(s1n, on="s1_idx").join(pn, on="cand_idx")
raws = []
for i in (1, 2, 3):
    r = pl.read_parquet(f"cache/raw_train_s{i}.parquet", columns=["entity_id", "business_name", "business_address"]).with_row_index("row")
    raws.append(r.with_columns((pl.col("row") / r.height).alias("rowq"), pl.lit(i).alias("src")))
r1 = raws[0].rename({"entity_id": "sid", "business_name": "n1", "business_address": "a1", "row": "row1", "rowq": "q1"}).drop("src")
rp = pl.concat(raws[1:]).rename({"entity_id": "cid", "business_name": "n2", "business_address": "a2", "row": "row2", "rowq": "q2"})
d = va.join(r1, on="sid").join(rp, on="cid").with_columns(pl.col("n1", "a1", "n2", "a2").fill_null(""))
num = lambda c: pl.col(c).str.extract(r"(\d+)$").cast(pl.Int64)  # noqa: E731
up = lambda c: pl.col(c).str.count_matches(r"[A-Z]") / (pl.col(c).str.count_matches(r"[A-Za-z]") + 1)  # noqa: E731
d = d.with_columns(
    (pl.col("n1") == pl.col("n2")).cast(pl.Float32).alias("raw_name_eq"),
    (pl.col("n1").str.to_lowercase() == pl.col("n2").str.to_lowercase()).cast(pl.Float32).alias("raw_name_ci_eq"),
    (pl.col("a1") == pl.col("a2")).cast(pl.Float32).alias("raw_addr_eq"),
    (pl.col("a1").str.to_lowercase() == pl.col("a2").str.to_lowercase()).cast(pl.Float32).alias("raw_addr_ci_eq"),
    up("n2").alias("c_upper_share"), (up("n1") - up("n2")).abs().alias("upper_share_diff"),
    pl.col("n2").str.count_matches(r"[^\w\s]").alias("c_punct"), (pl.col("n1").str.count_matches(r"[^\w\s]") - pl.col("n2").str.count_matches(r"[^\w\s]")).alias("punct_diff"),
    pl.col("n2").str.count_matches("  ").alias("c_dblspace"), pl.col("a2").str.count_matches(",").alias("c_addr_commas"),
    (pl.col("a1").str.count_matches(",") - pl.col("a2").str.count_matches(",")).alias("comma_diff"),
    (pl.col("n2").str.len_chars() - pl.col("n1").str.len_chars()).alias("name_len_diff"),
    (pl.col("a2").str.len_chars() - pl.col("a1").str.len_chars()).alias("addr_len_diff"),
    pl.col("n2").str.contains(r"[^\x00-\x7F]").cast(pl.Float32).alias("c_nonascii"),
    pl.col("src").cast(pl.Float32).alias("src"),
    (num("sid") - num("cid")).abs().cast(pl.Float64).log1p().alias("id_absdiff_log"),
    (num("cid") % 97).cast(pl.Float32).alias("cid_mod97"), (num("sid") % 97 == num("cid") % 97).cast(pl.Float32).alias("id_mod97_eq"),
    num("cid").cast(pl.Float64).log1p().alias("cid_log"),
    (pl.col("q1") - pl.col("q2")).abs().alias("row_q_absdiff"), pl.col("q2").alias("cand_rowq"),
    pl.col("row2").rank("ordinal").over("s1_idx", "src").alias("row_rank_in_s1"),
)
d = d.with_columns((pl.col("row2") - pl.col("row2").shift(1).over(pl.col("s1_idx"), pl.col("src"), order_by="row2")).abs().cast(pl.Float64).log1p().alias("row_gap_prev"))
FEATS = ["raw_name_eq", "raw_name_ci_eq", "raw_addr_eq", "raw_addr_ci_eq", "c_upper_share", "upper_share_diff", "c_punct", "punct_diff",
         "c_dblspace", "c_addr_commas", "comma_diff", "name_len_diff", "addr_len_diff", "c_nonascii", "src", "id_absdiff_log",
         "cid_mod97", "id_mod97_eq", "cid_log", "row_q_absdiff", "cand_rowq", "row_rank_in_s1", "row_gap_prev"]
d = d.filter(pl.col("p") >= 0.02)
p = d["p"].clip(1e-5, 1 - 1e-5).to_numpy()
m = np.log(p / (1 - p))
y = d["label"].to_numpy()
X = d.select(FEATS).cast(pl.Float32).to_numpy()
g = d["block"].fill_null("?").to_numpy()
ll = lambda y, q: float(-np.mean(y * np.log(np.clip(q, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - q, 1e-6, 1))))  # noqa: E731
base = ll(y, p)
oof = np.zeros(len(y))
gain = {}
prm = {"objective": "binary:logistic", "max_depth": 4, "eta": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 20, "tree_method": "hist"}
for trn, tst in GroupKFold(5).split(X, y, g):
    dt = xgb.DMatrix(X[trn], y[trn], base_margin=m[trn], feature_names=FEATS)
    b = xgb.train(prm, dt, 300)
    oof[tst] = b.predict(xgb.DMatrix(X[tst], base_margin=m[tst], feature_names=FEATS))
    for k, v in b.get_score(importance_type="total_gain").items():
        gain[k] = gain.get(k, 0) + v
print(f"val pairs (p>=0.02) {len(y):,}  logloss current {base:.5f}  + raw/id/row residual model (OOF) {ll(y, oof):.5f}  "
      f"({(base - ll(y, oof)) / base * 100:.2f}% better)")
tot = sum(gain.values())
for k, v in sorted(gain.items(), key=lambda x: -x[1])[:12]:
    print(f"  {k:18s} {v / tot:.3f}")
# does it change decisions? F0.5 with the same decision on OOF-adjusted p
tvc = pl.read_parquet(R / "20260926-141000-v9" / "val_truth.parquet").select("s1_idx", "n_true")
import json  # noqa: E402
mm = json.loads((ce / "metrics.json").read_text())
dec = {"mode": mm["decision_mode"], "param": mm["decision_param"], "excl": mm["decision_excl"]}
a0 = d.select("s1_idx", "cand_idx", "label", "p")
a1 = a0.with_columns(pl.Series("p", oof.astype(np.float32)))
class _Q:
    def log(self, m): pass
print("F0.5 current decision:", round(pp.eval_selection(pp.apply_decision(a0, dec, 0.02), tvc)["f05"], 5),
      "| residual-adjusted, re-tuned:", round(max(pp.tune_decision(a1, tvc, 0.02, _Q()).values(), key=lambda z: z["f05"])["f05"], 5))
