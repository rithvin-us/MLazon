"""Unseen-country proxy for transfer-robust training settings on the v10s feature set (v9 + name IDF):
monotone constraints, shallower trees, stronger regularisation. Same protocol as loco_idf.py (base results there).

  python loco_robust.py <run_with_train_feats> [names]   -> runs/loco_robust.json
"""
import json
import sys
import time

import numpy as np
import polars as pl
import xgboost as xgb

sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
import pipeline as pp  # noqa: E402
from features import FEATURES, IDF_FEATURES, monotone  # noqa: E402

R = pp.RUNS_DIR / sys.argv[1]
NAME = ["n_idf_jacc", "n_idf_miss1", "n_idf_miss2", "n_rep", "n_idf_rank"]  # the IDF subset that transfers (v10s)
BASE = [f for f in FEATURES if f not in IDF_FEATURES] + NAME
PRM = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", device="cuda", eta=0.08, max_depth=8,
           min_child_weight=5, subsample=0.8, colsample_bytree=0.8, max_bin=256, seed=42)
VARIANTS = {
    "mono": {"monotone_constraints": str(monotone(BASE)).replace(" ", "")},
    "d6": {"max_depth": 6},
    "mono_d6": {"max_depth": 6, "monotone_constraints": str(monotone(BASE)).replace(" ", "")},
    "reg": {"min_child_weight": 50, "colsample_bytree": 0.5, "reg_lambda": 10.0},
}
names = sys.argv[2].split(",") if len(sys.argv) > 2 else list(VARIANTS)
OUT = pp.RUNS_DIR / "loco_robust.json"


class _Q:
    def log(self, m):
        pass


cmap = pp.scan_norm("train", "s1").select(pl.col("idx").alias("s1_idx"), "country_n").collect()
vt = pl.read_parquet(R / "val_truth.parquet")
parts = sorted((R / "train_feats").glob("part-*.parquet"))
cols = ["s1_idx", "cand_idx", "label", "is_val", "is_es", *BASE]
res = json.loads(OUT.read_text()) if OUT.exists() else {}
for src, tgt in (("us", "india"), ("india", "us")):
    todo = [n for n in names if f"{src}->{tgt} {n}" not in res]
    if not todo:
        continue
    lf = pl.scan_parquet(parts).select(cols).join(cmap.lazy(), on="s1_idx")
    tr = lf.filter(~pl.col("is_val") & ~pl.col("is_es") & (pl.col("country_n") == src)).collect()
    es = lf.filter(pl.col("is_es") & (pl.col("country_n") == src)).collect()
    sv = lf.filter(pl.col("is_val") & (pl.col("country_n") == src)).collect()
    tv = lf.filter(pl.col("is_val") & (pl.col("country_n") == tgt)).collect()
    tvs = vt.filter(pl.col("country_n") == src).select("s1_idx", "n_true")
    tvt = vt.filter(pl.col("country_n") == tgt).select("s1_idx", "n_true")
    dtr = xgb.QuantileDMatrix(tr.select(BASE).to_numpy(), tr["label"].to_numpy(), feature_names=BASE, max_bin=256)
    des = xgb.QuantileDMatrix(es.select(BASE).to_numpy(), es["label"].to_numpy(), ref=dtr, feature_names=BASE)
    Xs, Xt = xgb.DMatrix(sv.select(BASE).to_numpy(), feature_names=BASE), xgb.DMatrix(tv.select(BASE).to_numpy(), feature_names=BASE)
    for n in todo:
        t0 = time.time()
        b = xgb.train({**PRM, **VARIANTS[n]}, dtr, 4000, evals=[(des, "es")], early_stopping_rounds=80, verbose_eval=False)
        it = (0, b.best_iteration + 1)
        s = sv.select("s1_idx", "cand_idx", "label").with_columns(pl.Series("p", b.predict(Xs, iteration_range=it), dtype=pl.Float32))
        t = tv.select("s1_idx", "cand_idx", "label").with_columns(pl.Series("p", b.predict(Xt, iteration_range=it), dtype=pl.Float32))
        dec = max(pp.tune_decision(s, tvs, 0.02, _Q()).values(), key=lambda d: d["f05"])
        k_src = pp.apply_decision(s, dec, 0.02).height / max(tvs.height, 1)
        best = min(((abs(pp.apply_decision(t, {"mode": "threshold", "param": th, "excl": dec.get("excl", False)}, 0.02).height
                         / max(tvt.height, 1) - k_src), th) for th in np.arange(0.30, 0.995, 0.01)), key=lambda x: x[0])
        shape = pp.eval_selection(pp.apply_decision(t, {"mode": "threshold", "param": best[1], "excl": dec.get("excl", False)}, 0.02), tvt)["f05"]
        res[f"{src}->{tgt} {n}"] = {"src_val": round(dec["f05"], 5),
                                    "tgt_val": round(pp.eval_selection(pp.apply_decision(t, dec, 0.02), tvt)["f05"], 5),
                                    "tgt_shape": round(shape, 5), "iters": b.best_iteration, "secs": round(time.time() - t0)}
        print(f"{src}->{tgt} {n}", res[f"{src}->{tgt} {n}"], flush=True)
        OUT.write_text(json.dumps(res, indent=1))
    del dtr, des, tr
print("robust done")
