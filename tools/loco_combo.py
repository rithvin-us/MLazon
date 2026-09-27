"""Do the two France fixes stack? Proxy for an unseen country: train on SRC only, adapt to TGT without its labels,
France-style decision (matches/S1 = SRC's), score TGT val. Mirrors the final France file:
  M0      SRC-only model (v10s features)
  M1      + TGT structure-based pseudo-labels from M0 (pos: p >= 0.98 and exclusive owner; hard neg: owned by another
          S1 with p >= 0.98; easy neg: p <= 0.02, 30%), weight 0.5              (= v10sfr, LB-verified)
  M2      round 2: positives from M1, hard negatives from M0 and M1, weight 1.0  (= v10sfr2)
  +rule   band [0.2, 0.8) decided by the strict string rule                       (= PROBE_france_rule; M2+rule = combo)
-> runs/loco_combo.json (F0.5, precision, recall per variant)
  python loco_combo.py <run with train_feats>"""
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
POS, W = 0.98, {1: 0.5, 2: 1.0}  # pseudo-label weight per round, as for France (frpseudo 143915: 0.5, 163600: 1.0)
OUT = pp.RUNS_DIR / "loco_combo.json"
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


def train(df, es, w=None):
    d = xgb.QuantileDMatrix(df.select(F).to_numpy(), df["label"].to_numpy(), weight=w, feature_names=F, max_bin=256)
    e = xgb.QuantileDMatrix(es.select(F).to_numpy(), es["label"].to_numpy(), ref=d, feature_names=F)
    return xgb.train(PRM, d, 4000, evals=[(e, "es")], early_stopping_rounds=80, verbose_eval=False)


def pred(b, d):
    return d.select("s1_idx", "cand_idx", "label", *RC).with_columns(
        pl.Series("p", b.predict(xgb.DMatrix(d.select(F).to_numpy(), feature_names=F), iteration_range=(0, b.best_iteration + 1)), dtype=pl.Float32))


def owners(u):
    return u.with_columns(pl.col("p").max().over("cand_idx").alias("pmax"), pl.col("p").rank("ordinal", descending=True).over("cand_idx").alias("own"))


def pseudo(us):
    """us: TGT unlabelled pairs scored by each round's model so far. Positives + easy negatives from the last model,
    hard negatives from all of them (minus positives), as build_fr_pseudo.py does for France."""
    us = [owners(u) for u in us]
    pos = us[-1].filter((pl.col("p") >= POS) & (pl.col("own") == 1)).select("s1_idx", "cand_idx", pl.lit(1, pl.Int8).alias("pl"))
    hneg = (pl.concat([u.filter((pl.col("pmax") >= POS) & (pl.col("own") > 1)).select("s1_idx", "cand_idx") for u in us]).unique()
            .join(pos, on=["s1_idx", "cand_idx"], how="anti").with_columns(pl.lit(0, pl.Int8).alias("pl")))
    eneg = (us[-1].filter(pl.col("p") <= 0.02).join(hneg, on=["s1_idx", "cand_idx"], how="anti").sample(fraction=0.3, seed=1)
            .select("s1_idx", "cand_idx", pl.lit(0, pl.Int8).alias("pl")))
    lab = us[-1].select("s1_idx", "cand_idx", "label")
    acc = {"pos_n": pos.height, "pos_prec": round(pos.join(lab, on=["s1_idx", "cand_idx"])["label"].mean(), 4),
           "hneg_n": hneg.height, "hneg_true_neg": round(1 - hneg.join(lab, on=["s1_idx", "cand_idx"])["label"].mean(), 4)}
    return pl.concat([pos, hneg, eneg]), acc


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
    tu = lf.filter(~pl.col("is_val") & (pl.col("country_n") == TGT)).collect()  # "unlabelled" target pairs
    tu = tu.filter(pl.col("s1_idx").is_in(tu["s1_idx"].unique().sample(min(300000, tu["s1_idx"].n_unique()), seed=5).implode()))
    tvs, tvt = vt.filter(pl.col("country_n") == SRC).select("s1_idx", "n_true"), vt.filter(pl.col("country_n") == TGT).select("s1_idx", "n_true")
    out, scored_u, m = {}, [], None
    for rnd in range(3):
        if rnd == 0:
            m = train(tr, es)
        else:
            ps, acc = pseudo(scored_u)
            out[f"M{rnd}_pseudo"] = acc
            extra = tu.join(ps, on=["s1_idx", "cand_idx"]).with_columns(pl.col("pl").cast(tr["label"].dtype).alias("label"))
            comb = pl.concat([tr.select("label", *F), extra.select("label", *F)])
            w = np.concatenate([np.ones(tr.height, np.float32), np.full(extra.height, W[rnd], np.float32)])
            del m
            m = train(comb, es, w)
            del comb, extra, w
        s = pred(m, sv)
        dec = max(pp.tune_decision(s.select("s1_idx", "cand_idx", "label", "p"), tvs, 0.02, _Q()).values(), key=lambda z: z["f05"])
        k = pp.apply_decision(s.select("s1_idx", "cand_idx", "label", "p"), dec, 0.02).height / tvs.height
        plain, ruled = both(pred(m, tv), tvt, k)
        out[f"M{rnd}"], out[f"M{rnd}+rule"], out[f"M{rnd}_src_val"] = plain, ruled, round(dec["f05"], 5)
        if rnd < 2:
            scored_u.append(pred(m, tu).select("s1_idx", "cand_idx", "label", "p"))
        out["secs"] = round(time.time() - t0)
        res[f"{SRC}->{TGT}"] = out
        OUT.write_text(json.dumps(res, indent=1))
        print(f"{SRC}->{TGT} M{rnd}: {plain} | +rule {ruled}", flush=True)
        gc.collect()
    del tr, es, sv, tv, tu, scored_u, m
    gc.collect()
print("combo done")
