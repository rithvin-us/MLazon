"""Structure-based pseudo-labels for an unlabelled country, checked on the proxy (train SRC, adapt to TGT).
M0 = SRC-only model. Score TGT's non-val pairs (treated as unlabelled). Pseudo-labels:
  pos       p >= POS and this S1 is the candidate's best-scoring S1 (exclusive owner)
  hard neg  the candidate is owned by ANOTHER S1 with p >= POS  (look-alike of a different business)
  easy neg  p <= 0.02 (subsample)
Pseudo-label precision is measured against TGT's true labels. M1 = SRC + TGT pseudo pairs (weight W). TGT val F0.5
with the France-style decision (matches/S1 = SRC's), M0 vs M1.   -> runs/loco_pseudo.json"""
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
SRC, TGT = (sys.argv[2], sys.argv[3]) if len(sys.argv) > 3 else ("us", "india")
NAME = ["n_idf_jacc", "n_idf_miss1", "n_idf_miss2", "n_rep", "n_idf_rank"]
F = [f for f in FEATURES if f not in IDF_FEATURES] + NAME
PRM = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", device="cuda", eta=0.08, max_depth=8,
           min_child_weight=5, subsample=0.8, colsample_bytree=0.8, max_bin=256, seed=42)
OUT = pp.RUNS_DIR / "loco_pseudo.json"


class _Q:
    def log(self, m):
        pass


def shape_f(t, tvt, k_src):
    ex = pp.exclusive(t.filter(pl.col("p") >= 0.02))
    ths = np.arange(0.2, 0.996, 0.005)
    ks = np.array([ex.filter(pl.col("p") >= th).height / tvt.height for th in ths])
    th = float(ths[np.argmin(np.abs(ks - k_src))])
    return pp.eval_selection(ex.filter(pl.col("p") >= th), tvt)["f05"]


def train(df, w=None):
    d = xgb.QuantileDMatrix(df.select(F).to_numpy(), df["label"].to_numpy(), weight=w, feature_names=F, max_bin=256)
    e = xgb.QuantileDMatrix(es.select(F).to_numpy(), es["label"].to_numpy(), ref=d, feature_names=F)
    return xgb.train(PRM, d, 4000, evals=[(e, "es")], early_stopping_rounds=80, verbose_eval=False)


def pred(b, d):
    return d.select("s1_idx", "cand_idx", "label").with_columns(
        pl.Series("p", b.predict(xgb.DMatrix(d.select(F).to_numpy(), feature_names=F), iteration_range=(0, b.best_iteration + 1)), dtype=pl.Float32))


cmap = pp.scan_norm("train", "s1").select(pl.col("idx").alias("s1_idx"), "country_n").collect()
vt = pl.read_parquet(R / "val_truth.parquet")
lf = pl.scan_parquet(sorted((R / "train_feats").glob("part-*.parquet"))).select("s1_idx", "cand_idx", "label", "is_val", "is_es", *F).join(cmap.lazy(), on="s1_idx")
tr = lf.filter(~pl.col("is_val") & ~pl.col("is_es") & (pl.col("country_n") == SRC)).collect()
es = lf.filter(pl.col("is_es") & (pl.col("country_n") == SRC)).collect()
sv = lf.filter(pl.col("is_val") & (pl.col("country_n") == SRC)).collect()
tv = lf.filter(pl.col("is_val") & (pl.col("country_n") == TGT)).collect()
tu = lf.filter(~pl.col("is_val") & (pl.col("country_n") == TGT)).collect()  # "unlabelled" target pairs
tu = tu.filter(pl.col("s1_idx").is_in(tu["s1_idx"].unique().sample(min(300000, tu["s1_idx"].n_unique()), seed=5).implode()))
tvs, tvt = vt.filter(pl.col("country_n") == SRC).select("s1_idx", "n_true"), vt.filter(pl.col("country_n") == TGT).select("s1_idx", "n_true")
t0 = time.time()
m0 = train(tr)
s0 = pred(m0, sv)
dec = max(pp.tune_decision(s0, tvs, 0.02, _Q()).values(), key=lambda z: z["f05"])
k_src = pp.apply_decision(s0, dec, 0.02).height / tvs.height
f0 = shape_f(pred(m0, tv), tvt, k_src)
res = {"src->tgt": f"{SRC}->{TGT}", "M0_tgt_shape": round(f0, 5), "secs_m0": round(time.time() - t0)}
print(res, flush=True)
u = pred(m0, tu).drop("label").join(tu.select("s1_idx", "cand_idx", "label"), on=["s1_idx", "cand_idx"])
u = u.with_columns(pl.col("p").max().over("cand_idx").alias("pmax"), pl.col("p").rank("ordinal", descending=True).over("cand_idx").alias("own_rank"))
for POS in (0.98, 0.95):
    pos = u.filter((pl.col("p") >= POS) & (pl.col("own_rank") == 1)).with_columns(pl.lit(1, pl.Int8).alias("pl"))
    hneg = u.filter((pl.col("pmax") >= POS) & (pl.col("own_rank") > 1)).with_columns(pl.lit(0, pl.Int8).alias("pl"))
    eneg = u.filter(pl.col("p") <= 0.02).sample(fraction=0.3, seed=1).with_columns(pl.lit(0, pl.Int8).alias("pl"))
    ps = pl.concat([pos, hneg, eneg])
    acc = {"pos_n": pos.height, "pos_prec": round(pos["label"].mean(), 4), "hneg_n": hneg.height,
           "hneg_true_neg": round(1 - hneg["label"].mean(), 4), "eneg_true_neg": round(1 - eneg["label"].mean(), 4)}
    extra = tu.join(ps.select("s1_idx", "cand_idx", "pl"), on=["s1_idx", "cand_idx"]).with_columns(pl.col("pl").alias("label")).drop("pl")
    for W in (0.5, 1.0):
        comb = pl.concat([tr.select("label", *F), extra.select(pl.col("label").cast(tr["label"].dtype), *F)])
        w = np.concatenate([np.ones(tr.height, np.float32), np.full(extra.height, W, np.float32)])
        m1 = train(comb, w)
        s1 = pred(m1, sv)
        dec1 = max(pp.tune_decision(s1, tvs, 0.02, _Q()).values(), key=lambda z: z["f05"])
        k1 = pp.apply_decision(s1, dec1, 0.02).height / tvs.height
        f1 = shape_f(pred(m1, tv), tvt, k1)
        res[f"pos{POS}_w{W}"] = {**acc, "src_val": round(dec1["f05"], 5), "tgt_shape": round(f1, 5), "gain": round(f1 - f0, 5)}
        print(f"pos{POS} w{W}", res[f"pos{POS}_w{W}"], flush=True)
        OUT.write_text(json.dumps(res, indent=1))
print("pseudo done")
