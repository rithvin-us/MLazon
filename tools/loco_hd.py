"""Is the low score of different-house-number pairs a transfer artefact? Train on SRC only (M0) and on SRC + TGT
pseudo-labels (M1, current recipe); on TGT val, the slice "numbers differ, same core name, core address >= 90" by
context (A: the S1 has a same-house-number copy; B: it has not): true match rate vs mean p of M0 / M1 / an in-domain
model (trained on TGT's own labels). Runs after loco_adapt.py.  -> runs/loco_hd.json"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
import pipeline as pp  # noqa: E402
from features import FEATURES, IDF_FEATURES  # noqa: E402

while "adapt done" not in (Path(r"D:\amazon-ml\runs\loco_adapt.out").read_text(encoding="utf-8", errors="replace")
                           if Path(r"D:\amazon-ml\runs\loco_adapt.out").exists() else ""):
    time.sleep(30)
R = pp.RUNS_DIR / sys.argv[1]
NAME = ["n_idf_jacc", "n_idf_miss1", "n_idf_miss2", "n_rep", "n_idf_rank"]
F = [f for f in FEATURES if f not in IDF_FEATURES] + NAME
PRM = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", device="cuda", eta=0.08, max_depth=8,
           min_child_weight=5, subsample=0.8, colsample_bytree=0.8, max_bin=256, seed=42)
HD = (pl.col("hn_both") == 1) & (pl.col("hn_eq") == 0) & (pl.col("num_jacc") == 0) & (pl.col("nc_tset") == 100) & (pl.col("ad_core_tset") >= 90)


def train(frame, es):
    X = frame.select(F).to_numpy()
    d = xgb.QuantileDMatrix(X, frame["y"].to_numpy(), weight=frame["w"].to_numpy(), feature_names=F, max_bin=256)
    del X
    e = xgb.QuantileDMatrix(es.select(F).to_numpy(), es["label"].to_numpy(), ref=d, feature_names=F)
    return xgb.train(PRM, d, 4000, evals=[(e, "es")], early_stopping_rounds=80, verbose_eval=False)


def p_of(b, d):
    return b.predict(xgb.DMatrix(d.select(F).to_numpy(), feature_names=F), iteration_range=(0, b.best_iteration + 1))


cmap = pp.scan_norm("train", "s1").select(pl.col("idx").alias("s1_idx"), "country_n").collect()
lf = pl.scan_parquet(sorted((R / "train_feats").glob("part-*.parquet"))).select("s1_idx", "cand_idx", "label", "is_val", "is_es", *F).join(cmap.lazy(), on="s1_idx")
res = {}
for SRC, TGT in (("us", "india"), ("india", "us")):
    base = lf.filter(~pl.col("is_val") & ~pl.col("is_es") & (pl.col("country_n") == SRC)).select(*F, pl.col("label").cast(pl.Float32).alias("y"), pl.lit(1.0, pl.Float32).alias("w")).collect()
    es = lf.filter(pl.col("is_es") & (pl.col("country_n") == SRC)).collect()
    tv = lf.filter(pl.col("is_val") & (pl.col("country_n") == TGT)).collect()
    same = tv.filter((pl.col("hn_eq") == 1) & (pl.col("nc_tset") == 100)).select("s1_idx").unique().with_columns(pl.lit(True).alias("copy"))
    sl = tv.join(same, on="s1_idx", how="left").with_columns(pl.col("copy").fill_null(False)).with_columns(HD.alias("hd"))
    m0 = train(base, es)
    sl = sl.with_columns(pl.Series("p_m0", p_of(m0, sl), dtype=pl.Float32))
    del m0
    tu = lf.filter(~pl.col("is_val") & (pl.col("country_n") == TGT)).collect()
    tu = tu.filter(pl.col("s1_idx").is_in(tu["s1_idx"].unique().sample(min(300000, tu["s1_idx"].n_unique()), seed=5).implode()))
    # in-domain reference: TGT's own labelled non-val pairs (same size budget as SRC)
    tgt_base = tu.select(*F, pl.col("label").cast(pl.Float32).alias("y"), pl.lit(1.0, pl.Float32).alias("w"))
    es_t = lf.filter(pl.col("is_es") & (pl.col("country_n") == TGT)).collect()
    mt = train(tgt_base, es_t)
    sl = sl.with_columns(pl.Series("p_in", p_of(mt, sl), dtype=pl.Float32))
    del mt, tgt_base, es_t
    out = sl.filter(pl.col("hd")).group_by("copy").agg(pl.len().alias("n"), pl.col("label").mean().round(3).alias("true_rate"),
                                                       pl.col("p_m0").mean().round(3).alias("p_transfer"), pl.col("p_in").mean().round(3).alias("p_in_domain"))
    res[f"{SRC}->{TGT}"] = out.to_dicts()
    print(f"{SRC}->{TGT} hd slice: {out.to_dicts()}", flush=True)
    Path(pp.RUNS_DIR / "loco_hd.json").write_text(json.dumps(res, indent=1))
    del base, es, tv, sl, tu
print("hd done")
