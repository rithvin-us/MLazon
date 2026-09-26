"""Unseen-country proxy for the v10 IDF features: train on one country only, evaluate the other country's val.
France has no labels, so "train US -> score India" (and India -> US) is the closest measurable stand-in.

  python loco_idf.py <run_with_augmented_train_feats>   -> runs/loco_idf.json + printed table
"""
import json
import sys
import time

import numpy as np
import polars as pl
import xgboost as xgb

sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
import pipeline as pp  # noqa: E402
from features import FEATURES, IDF_FEATURES  # noqa: E402

R = pp.RUNS_DIR / sys.argv[1]
BASE = [f for f in FEATURES if f not in IDF_FEATURES]
MAG = ["n_rep_idf", "n_miss_idf", "a_rep_idf", "a_miss_idf"]
SUBSETS = {
    "base": BASE, "idf": FEATURES,
    "ratio": [f for f in FEATURES if f not in MAG],
    "ranks": BASE + ["n_idf_rank", "a_idf_rank"],
    "name": BASE + [f for f in IDF_FEATURES if f.startswith("n_") and f not in MAG],
    "addr": BASE + [f for f in IDF_FEATURES if f.startswith("a_") and f not in MAG],
    "jacc": BASE + ["n_idf_jacc", "a_idf_jacc", "n_idf_rank", "a_idf_rank"],
}
if len(sys.argv) > 2:  # extra subsets, both directions: python loco_idf.py <run> ratio,ranks,...
    CONFIGS = [(s_, t_, n_, SUBSETS[n_], None) for s_, t_ in (("us", "india"), ("india", "us")) for n_ in sys.argv[2].split(",")]
else:
    CONFIGS = [  # (src, tgt, name, features, crowd_w)
        ("us", "india", "base", BASE, None),
        ("us", "india", "idf", FEATURES, None),
        ("us", "india", "idf+crowd3", FEATURES, 3.0),
        ("india", "us", "base", BASE, None),
        ("india", "us", "idf", FEATURES, None),
    ]
PRM = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", device="cuda", eta=0.08, max_depth=8,
           min_child_weight=5, subsample=0.8, colsample_bytree=0.8, max_bin=256, seed=42)


class _Q:
    def log(self, m):
        pass


cmap = pp.scan_norm("train", "s1").select(pl.col("idx").alias("s1_idx"), "country_n").collect()
vt = pl.read_parquet(R / "val_truth.parquet")
parts = sorted((R / "train_feats").glob("part-*.parquet"))
cols = ["s1_idx", "cand_idx", "label", "is_val", "is_es", *FEATURES]
res = json.loads((pp.RUNS_DIR / "loco_idf.json").read_text()) if (pp.RUNS_DIR / "loco_idf.json").exists() else {}
cache = {}
for src, tgt, name, F, cw in CONFIGS:
    key = f"{src}->{tgt} {name}"
    if key in res:
        print(key, res[key], "(cached)")
        continue
    t0 = time.time()
    if (src, tgt) not in cache:
        cache.clear()
        lf = pl.scan_parquet(parts).select(cols).join(cmap.lazy(), on="s1_idx")
        tr = lf.filter(~pl.col("is_val") & ~pl.col("is_es") & (pl.col("country_n") == src)).collect()
        es = lf.filter(pl.col("is_es") & (pl.col("country_n") == src)).collect()
        sv = lf.filter(pl.col("is_val") & (pl.col("country_n") == src)).collect()
        tv = lf.filter(pl.col("is_val") & (pl.col("country_n") == tgt)).collect()
        cache[(src, tgt)] = (tr, es, sv, tv)
    tr, es, sv, tv = cache[(src, tgt)]
    w = np.where(tr["n_cands"].to_numpy() >= 15, cw, 1.0).astype(np.float32) if cw else None
    dtr = xgb.QuantileDMatrix(tr.select(F).to_numpy(), tr["label"].to_numpy(), weight=w, feature_names=F, max_bin=256)
    des = xgb.QuantileDMatrix(es.select(F).to_numpy(), es["label"].to_numpy(), ref=dtr, feature_names=F)
    b = xgb.train(PRM, dtr, 4000, evals=[(des, "es")], early_stopping_rounds=80, verbose_eval=False)
    del dtr, des
    sc = lambda d: d.select("s1_idx", "cand_idx", "label").with_columns(  # noqa: E731
        pl.Series("p", b.predict(xgb.DMatrix(d.select(F).to_numpy(), feature_names=F),
                                 iteration_range=(0, b.best_iteration + 1)), dtype=pl.Float32))
    s, t = sc(sv), sc(tv)
    tvs = vt.filter(pl.col("country_n") == src).select("s1_idx", "n_true")
    tvt = vt.filter(pl.col("country_n") == tgt).select("s1_idx", "n_true")
    dec = max(pp.tune_decision(s, tvs, 0.02, _Q()).values(), key=lambda d: d["f05"])
    tgt_f = pp.eval_selection(pp.apply_decision(t, dec, 0.02), tvt)["f05"]
    oracle = max(pp.tune_decision(t, tvt, 0.02, _Q()).values(), key=lambda d: d["f05"])["f05"]
    # France-style decision: threshold whose matches per S1 on the target equal the source's
    k_src = pp.apply_decision(s, dec, 0.02).height / max(tvs.height, 1)
    best = min(((abs(pp.apply_decision(t, {"mode": "threshold", "param": th, "excl": dec.get("excl", False)}, 0.02).height
                     / max(tvt.height, 1) - k_src), th) for th in np.arange(0.30, 0.995, 0.01)), key=lambda x: x[0])
    shape = pp.eval_selection(pp.apply_decision(t, {"mode": "threshold", "param": best[1], "excl": dec.get("excl", False)}, 0.02), tvt)["f05"]
    res[key] = {"src_val": round(dec["f05"], 5), "tgt_val": round(tgt_f, 5), "tgt_shape": round(shape, 5),
                "shape_thr": round(float(best[1]), 2), "tgt_oracle": round(oracle, 5),
                "iters": b.best_iteration, "secs": round(time.time() - t0)}
    print(key, res[key], flush=True)
    (pp.RUNS_DIR / "loco_idf.json").write_text(json.dumps(res, indent=1))
print("loco done")
