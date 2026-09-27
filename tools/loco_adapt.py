"""Domain-adaptation variants on the unseen-country proxy (train SRC with labels, adapt to TGT without, France-style
decision, F0.5/P/R on TGT val), each with and without the rule fallback:
  M0        SRC only
  hard98    pseudo-labels, current recipe: pos p>=0.98 & exclusive owner, hard neg = owned by another S1 with p>=0.98,
            easy neg p<=0.02 (30%), weight 0.5
  hard90    same with 0.90 / 0.10 cut-offs
  soft      soft targets: every TGT pair p>=0.02 enters twice (label 1 weight W*y, label 0 weight W*(1-y)),
            y = p for the record's best S1, 0 when another S1 owns the record with p>=0.5; easy neg as hard98
  iw        importance weighting: a domain classifier (SRC pairs vs TGT pairs, same features) gives each SRC training
            row the weight P(tgt|x)/P(src|x) (clipped 0.1-10, mean 1); no TGT rows
-> runs/loco_adapt.json      python loco_adapt.py <run with train_feats>"""
import gc
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
NAME = ["n_idf_jacc", "n_idf_miss1", "n_idf_miss2", "n_rep", "n_idf_rank"]
F = [f for f in FEATURES if f not in IDF_FEATURES] + NAME
RC = ["nc_tset", "ad_tset", "hn_eq", "legal_conflict"]
PRM = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", device="cuda", eta=0.08, max_depth=8,
           min_child_weight=5, subsample=0.8, colsample_bytree=0.8, max_bin=256, seed=42)
W = 0.5
OUT = pp.RUNS_DIR / "loco_adapt.json"
BAND = (pl.col("p") >= 0.2) & (pl.col("p") < 0.8)
STRICT = (pl.col("nc_tset") >= 95) & (pl.col("ad_tset") >= 90) & (pl.col("hn_eq") == 1) & (pl.col("legal_conflict") == 0)


class _Q:
    def log(self, m):
        pass


def shape(t, tvt, k_src):
    ex = pp.exclusive(t.filter(pl.col("p") >= 0.02))
    ths = np.arange(0.2, 0.996, 0.005)
    ks = np.array([ex.filter(pl.col("p") >= th).height / tvt.height for th in ths])
    th = float(ths[np.argmin(np.abs(ks - k_src))])
    m = pp.eval_selection(ex.filter(pl.col("p") >= th), tvt)
    return {"f05": round(m["f05"], 5), "P": round(m["precision"], 4), "R": round(m["recall"], 4), "thr": round(th, 3)}


def both(t, tvt, k_src):
    ruled = t.with_columns(pl.when(BAND).then(pl.when(STRICT).then(0.99).otherwise(0.01)).otherwise(pl.col("p")).cast(pl.Float32).alias("p"))
    return shape(t, tvt, k_src), shape(ruled, tvt, k_src)


def train(frame, es):
    """frame: F columns + y (target) + w (weight)."""
    X = frame.select(F).to_numpy()
    d = xgb.QuantileDMatrix(X, frame["y"].to_numpy(), weight=frame["w"].to_numpy(), feature_names=F, max_bin=256)
    del X
    e = xgb.QuantileDMatrix(es.select(F).to_numpy(), es["label"].to_numpy(), ref=d, feature_names=F)
    return xgb.train(PRM, d, 4000, evals=[(e, "es")], early_stopping_rounds=80, verbose_eval=False)


def pred(b, d):
    return d.select("s1_idx", "cand_idx", "label", *RC).with_columns(
        pl.Series("p", b.predict(xgb.DMatrix(d.select(F).to_numpy(), feature_names=F), iteration_range=(0, b.best_iteration + 1)), dtype=pl.Float32))


def evaluate(m, sv, tv, tvs, tvt):
    s = pred(m, sv).select("s1_idx", "cand_idx", "label", "p")
    dec = max(pp.tune_decision(s, tvs, 0.02, _Q()).values(), key=lambda z: z["f05"])
    k = pp.apply_decision(s, dec, 0.02).height / tvs.height
    plain, ruled = both(pred(m, tv), tvt, k)
    return {"plain": plain, "rule": ruled, "src_val": round(dec["f05"], 5)}


cmap = pp.scan_norm("train", "s1").select(pl.col("idx").alias("s1_idx"), "country_n").collect()
vt = pl.read_parquet(R / "val_truth.parquet")
lf = pl.scan_parquet(sorted((R / "train_feats").glob("part-*.parquet"))).select("s1_idx", "cand_idx", "label", "is_val", "is_es", *F).join(cmap.lazy(), on="s1_idx")
res = {}
for SRC, TGT in (("us", "india"), ("india", "us")):
    t0 = time.time()
    tr = lf.filter(~pl.col("is_val") & ~pl.col("is_es") & (pl.col("country_n") == SRC)).collect()
    es = lf.filter(pl.col("is_es") & (pl.col("country_n") == SRC)).collect()
    sv = lf.filter(pl.col("is_val") & (pl.col("country_n") == SRC)).collect()
    tv = lf.filter(pl.col("is_val") & (pl.col("country_n") == TGT)).collect()
    tu = lf.filter(~pl.col("is_val") & (pl.col("country_n") == TGT)).collect()
    tu = tu.filter(pl.col("s1_idx").is_in(tu["s1_idx"].unique().sample(min(300000, tu["s1_idx"].n_unique()), seed=5).implode()))
    tvs, tvt = vt.filter(pl.col("country_n") == SRC).select("s1_idx", "n_true"), vt.filter(pl.col("country_n") == TGT).select("s1_idx", "n_true")
    out = {}
    base = tr.select(*F, pl.col("label").cast(pl.Float32).alias("y"), pl.lit(1.0, pl.Float32).alias("w"))
    del tr
    gc.collect()
    m0 = train(base, es)
    out["M0"] = evaluate(m0, sv, tv, tvs, tvt)
    print(f"{SRC}->{TGT} M0 {out['M0']}", flush=True)
    u = pred(m0, tu).select("s1_idx", "cand_idx", "label", "p").with_columns(
        pl.col("p").max().over("cand_idx").alias("pmax"), pl.col("p").rank("ordinal", descending=True).over("cand_idx").alias("own"))
    del m0
    gc.collect()
    eneg = u.filter(pl.col("p") <= 0.02).sample(fraction=0.3, seed=1).select("s1_idx", "cand_idx", pl.lit(0.0).alias("y"), pl.lit(W).alias("w"))
    variants = {}
    for name, hi, lo in (("hard98", 0.98, 0.02), ("hard90", 0.90, 0.10)):
        pos = u.filter((pl.col("p") >= hi) & (pl.col("own") == 1)).select("s1_idx", "cand_idx", pl.lit(1.0).alias("y"), pl.lit(W).alias("w"))
        hn = u.filter((pl.col("pmax") >= hi) & (pl.col("own") > 1)).select("s1_idx", "cand_idx", pl.lit(0.0).alias("y"), pl.lit(W).alias("w"))
        en = eneg if lo == 0.02 else u.filter(pl.col("p") <= lo).sample(fraction=0.3, seed=1).select("s1_idx", "cand_idx", pl.lit(0.0).alias("y"), pl.lit(W).alias("w"))
        en = en.join(pl.concat([pos, hn]).select("s1_idx", "cand_idx"), on=["s1_idx", "cand_idx"], how="anti")
        variants[name] = pl.concat([pos, hn, en], how="vertical_relaxed")
    y = (u.filter(pl.col("p") > 0.02).with_columns(
        pl.when((pl.col("own") > 1) & (pl.col("pmax") >= 0.5)).then(0.0).otherwise(pl.col("p")).alias("yy")))
    variants["soft"] = pl.concat([y.select("s1_idx", "cand_idx", pl.lit(1.0).alias("y"), (W * pl.col("yy")).alias("w")),
                                  y.select("s1_idx", "cand_idx", pl.lit(0.0).alias("y"), (W * (1 - pl.col("yy"))).alias("w")), eneg], how="vertical_relaxed").filter(pl.col("w") > 0)
    for name, ps in variants.items():
        extra = tu.select("s1_idx", "cand_idx", *F).join(ps, on=["s1_idx", "cand_idx"])
        comb = pl.concat([base, extra.select(*F, pl.col("y").cast(pl.Float32), pl.col("w").cast(pl.Float32))])
        n_extra = extra.height
        del extra
        m = train(comb, es)
        del comb
        out[name] = evaluate(m, sv, tv, tvs, tvt)
        out[name]["n_extra"] = n_extra
        print(f"{SRC}->{TGT} {name} {out[name]}", flush=True)
        res[f"{SRC}->{TGT}"] = out
        OUT.write_text(json.dumps(res, indent=1))
        del m
        gc.collect()
    # importance weighting with a domain classifier (SRC vs TGT pairs), SRC rows only
    ns = min(base.height, tu.height, 2_000_000)
    ds = pl.concat([base.select(F).sample(ns, seed=3).with_columns(pl.lit(0).alias("dom")), tu.select(F).sample(ns, seed=4).with_columns(pl.lit(1).alias("dom"))])
    dm = xgb.train({**PRM, "max_depth": 6}, xgb.QuantileDMatrix(ds.select(F).to_numpy(), ds["dom"].to_numpy(), feature_names=F, max_bin=256), 300)
    del ds
    q = dm.predict(xgb.DMatrix(base.select(F).to_numpy(), feature_names=F))
    iw = np.clip(q / np.clip(1 - q, 1e-6, None), 0.1, 10).astype(np.float32)
    iw /= iw.mean()
    m = train(base.with_columns(pl.Series("w", iw)), es)
    out["iw"] = evaluate(m, sv, tv, tvs, tvt)
    out["iw"]["weight_p99"] = round(float(np.percentile(iw, 99)), 2)
    out["secs"] = round(time.time() - t0)
    print(f"{SRC}->{TGT} iw {out['iw']}", flush=True)
    res[f"{SRC}->{TGT}"] = out
    OUT.write_text(json.dumps(res, indent=1))
    del base, es, sv, tv, tu, u, variants, dm, m, iw, q
    gc.collect()
print("adapt done")
