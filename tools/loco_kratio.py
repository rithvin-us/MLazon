"""Unseen-country proxy: should an unlabelled country predict the SAME matches per S1 as the labelled one, or a
fraction of it? Train on SRC (v10s features), tune on SRC val, then on TGT pick the threshold whose matches/S1 =
r x SRC's, for r in a grid; F0.5 on TGT. -> runs/loco_kratio.json"""
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
RS = [0.85, 0.9, 0.93, 0.96, 0.98, 1.0, 1.02, 1.05, 1.1]


class _Q:
    def log(self, m):
        pass


cmap = pp.scan_norm("train", "s1").select(pl.col("idx").alias("s1_idx"), "country_n").collect()
vt = pl.read_parquet(R / "val_truth.parquet")
parts = sorted((R / "train_feats").glob("part-*.parquet"))
lf = pl.scan_parquet(parts).select("s1_idx", "cand_idx", "label", "is_val", "is_es", *F).join(cmap.lazy(), on="s1_idx")
res = {}
for src, tgt in (("us", "india"), ("india", "us")):
    tr = lf.filter(~pl.col("is_val") & ~pl.col("is_es") & (pl.col("country_n") == src)).collect()
    es = lf.filter(pl.col("is_es") & (pl.col("country_n") == src)).collect()
    sv = lf.filter(pl.col("is_val") & (pl.col("country_n") == src)).collect()
    tv = lf.filter(pl.col("is_val") & (pl.col("country_n") == tgt)).collect()
    dtr = xgb.QuantileDMatrix(tr.select(F).to_numpy(), tr["label"].to_numpy(), feature_names=F, max_bin=256)
    des = xgb.QuantileDMatrix(es.select(F).to_numpy(), es["label"].to_numpy(), ref=dtr, feature_names=F)
    b = xgb.train(PRM, dtr, 4000, evals=[(des, "es")], early_stopping_rounds=80, verbose_eval=False)
    del dtr, des, tr
    it = (0, b.best_iteration + 1)
    pr = lambda d: d.select("s1_idx", "cand_idx", "label").with_columns(pl.Series("p", b.predict(xgb.DMatrix(d.select(F).to_numpy(), feature_names=F), iteration_range=it), dtype=pl.Float32))  # noqa: E731
    s, t = pr(sv), pr(tv)
    tvs, tvt = vt.filter(pl.col("country_n") == src).select("s1_idx", "n_true"), vt.filter(pl.col("country_n") == tgt).select("s1_idx", "n_true")
    dec = max(pp.tune_decision(s, tvs, 0.02, _Q()).values(), key=lambda z: z["f05"])
    k_src = pp.apply_decision(s, dec, 0.02).height / tvs.height
    ex = pp.exclusive(t.filter(pl.col("p") >= 0.02))
    ths = np.arange(0.2, 0.996, 0.005)
    ks = np.array([ex.filter(pl.col("p") >= th).height / tvt.height for th in ths])
    out = {}
    for r in RS:
        th = float(ths[np.argmin(np.abs(ks - r * k_src))])
        out[str(r)] = round(pp.eval_selection(ex.filter(pl.col("p") >= th), tvt)["f05"], 5)
    best_th = max(ths, key=lambda th: pp.eval_selection(ex.filter(pl.col("p") >= th), tvt)["f05"])
    res[f"{src}->{tgt}"] = {"k_src": round(k_src, 3), "by_r": out, "oracle_thr": round(float(best_th), 3),
                            "oracle_k_ratio": round(ex.filter(pl.col("p") >= best_th).height / tvt.height / k_src, 3)}
    print(f"{src}->{tgt}", res[f"{src}->{tgt}"], flush=True)
    (pp.RUNS_DIR / "loco_kratio.json").write_text(json.dumps(res, indent=1))
print("kratio done")
