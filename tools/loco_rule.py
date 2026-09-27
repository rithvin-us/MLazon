"""Suggestion #4 checked on the unseen-country proxy: override the model in its uncertain band with a deterministic
string rule for the held-out country. Train SRC (v10s features), score TGT, France-style decision (matches/S1 = SRC's).
  strict  = core name token-set >= 95, address token-set >= 90, same house number, no legal-form conflict
  R_both   band pairs -> accept if strict else reject
  R_accept band pairs -> accept if strict (others keep the model p)
  R_reject band pairs -> reject if name token-set < 90 (others keep the model p)
-> runs/loco_rule.json"""
import json
import sys

import numpy as np
import polars as pl
import xgboost as xgb

sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
import pipeline as pp  # noqa: E402
from features import FEATURES, IDF_FEATURES  # noqa: E402

R = pp.RUNS_DIR / sys.argv[1]
NAME = ["n_idf_jacc", "n_idf_miss1", "n_idf_miss2", "n_rep", "n_idf_rank"]
F = [f for f in FEATURES if f not in IDF_FEATURES] + NAME
PRM = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", device="cuda", eta=0.08, max_depth=8,
           min_child_weight=5, subsample=0.8, colsample_bytree=0.8, max_bin=256, seed=42)


class _Q:
    def log(self, m):
        pass


def shape_f(t, tvt, k_src):
    ex = pp.exclusive(t.filter(pl.col("p") >= 0.02))
    ths = np.arange(0.2, 0.996, 0.005)
    ks = np.array([ex.filter(pl.col("p") >= th).height / tvt.height for th in ths])
    th = float(ths[np.argmin(np.abs(ks - k_src))])
    return round(pp.eval_selection(ex.filter(pl.col("p") >= th), tvt)["f05"], 5)


cmap = pp.scan_norm("train", "s1").select(pl.col("idx").alias("s1_idx"), "country_n").collect()
vt = pl.read_parquet(R / "val_truth.parquet")
lf = pl.scan_parquet(sorted((R / "train_feats").glob("part-*.parquet"))).select("s1_idx", "cand_idx", "label", "is_val", "is_es", *F).join(cmap.lazy(), on="s1_idx")
res = {}
for src, tgt in (("us", "india"), ("india", "us")):
    tr = lf.filter(~pl.col("is_val") & ~pl.col("is_es") & (pl.col("country_n") == src)).collect()
    es = lf.filter(pl.col("is_es") & (pl.col("country_n") == src)).collect()
    sv = lf.filter(pl.col("is_val") & (pl.col("country_n") == src)).collect()
    tv = lf.filter(pl.col("is_val") & (pl.col("country_n") == tgt)).collect()
    d = xgb.QuantileDMatrix(tr.select(F).to_numpy(), tr["label"].to_numpy(), feature_names=F, max_bin=256)
    e = xgb.QuantileDMatrix(es.select(F).to_numpy(), es["label"].to_numpy(), ref=d, feature_names=F)
    b = xgb.train(PRM, d, 4000, evals=[(e, "es")], early_stopping_rounds=80, verbose_eval=False)
    del d, e, tr
    it = (0, b.best_iteration + 1)
    pr = lambda df: df.select("s1_idx", "cand_idx", "label", "nc_tset", "ad_tset", "hn_eq", "legal_conflict").with_columns(  # noqa: E731
        pl.Series("p", b.predict(xgb.DMatrix(df.select(F).to_numpy(), feature_names=F), iteration_range=it), dtype=pl.Float32))
    s, t = pr(sv), pr(tv)
    tvs, tvt = vt.filter(pl.col("country_n") == src).select("s1_idx", "n_true"), vt.filter(pl.col("country_n") == tgt).select("s1_idx", "n_true")
    dec = max(pp.tune_decision(s.select("s1_idx", "cand_idx", "label", "p"), tvs, 0.02, _Q()).values(), key=lambda z: z["f05"])
    k_src = pp.apply_decision(s.select("s1_idx", "cand_idx", "label", "p"), dec, 0.02).height / tvs.height
    strict = (pl.col("nc_tset") >= 95) & (pl.col("ad_tset") >= 90) & (pl.col("hn_eq") == 1) & (pl.col("legal_conflict") == 0)
    out = {"model": shape_f(t, tvt, k_src)}
    for lo, hi in ((0.2, 0.8), (0.1, 0.9)):
        band = (pl.col("p") >= lo) & (pl.col("p") < hi)
        variants = {
            "R_both": pl.when(band).then(pl.when(strict).then(0.99).otherwise(0.01)).otherwise(pl.col("p")),
            "R_accept": pl.when(band & strict).then(0.99).otherwise(pl.col("p")),
            "R_reject": pl.when(band & (pl.col("nc_tset") < 90)).then(0.01).otherwise(pl.col("p")),
        }
        for name, expr in variants.items():
            out[f"{name}[{lo},{hi})"] = shape_f(t.with_columns(expr.cast(pl.Float32).alias("p")), tvt, k_src)
    res[f"{src}->{tgt}"] = out
    print(f"{src}->{tgt}", out, flush=True)
    (pp.RUNS_DIR / "loco_rule.json").write_text(json.dumps(res, indent=1))
print("rule done")
