"""Unseen-country proxy: country-invariant feature transforms on the v10s feature set (v9 + name IDF).
  rel     + within-S1 relative versions of key similarities (gap to the S1's best, gap to its median, z-score)
  pct     continuous features replaced by their percentile within the record's own country
  rel+pct both
Same protocol/params as loco_idf.py; reference = "name" results there. -> runs/loco_rel.json
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
NAME = ["n_idf_jacc", "n_idf_miss1", "n_idf_miss2", "n_rep", "n_idf_rank"]
BASE = [f for f in FEATURES if f not in IDF_FEATURES] + NAME
KEY = ["nf_tset", "nf_ratio", "nc_tset", "ncc_ratio", "ad_tset", "ad_core_tset", "hn_sim", "rr", "n_idf_jacc", "bscore_norm", "nsk_ratio"]
REL = [f"{k}_{s}" for k in KEY for s in ("gbest", "gmed", "z")]
PRM = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", device="cuda", eta=0.08, max_depth=8,
           min_child_weight=5, subsample=0.8, colsample_bytree=0.8, max_bin=256, seed=42)
OUT = pp.RUNS_DIR / "loco_rel.json"


class _Q:
    def log(self, m):
        pass


def add_rel(df):
    ex = []
    for k in KEY:
        x = pl.col(k).fill_nan(None)
        ex += [(x.max().over("s1_idx") - x).alias(f"{k}_gbest"), (x - x.median().over("s1_idx")).alias(f"{k}_gmed"),
               ((x - x.mean().over("s1_idx")) / (x.std().over("s1_idx") + 1e-3)).alias(f"{k}_z")]
    return df.with_columns(ex)


def pct(frames, cols):
    """percentile of each col within the union of frames (one country), applied to every frame"""
    allv = pl.concat([f.select(cols) for f in frames])
    out = []
    for f in frames:
        f2 = f
        for c in cols:
            ref = np.sort(allv[c].drop_nulls().drop_nans().to_numpy())
            v = f[c].to_numpy()
            q = np.searchsorted(ref, v, side="right") / max(len(ref), 1)
            f2 = f2.with_columns(pl.Series(c, np.where(np.isnan(v), np.nan, q).astype(np.float32)))
        out.append(f2)
    return out


cmap = pp.scan_norm("train", "s1").select(pl.col("idx").alias("s1_idx"), "country_n").collect()
vt = pl.read_parquet(R / "val_truth.parquet")
parts = sorted((R / "train_feats").glob("part-*.parquet"))
cols = ["s1_idx", "cand_idx", "label", "is_val", "is_es", *BASE]
res = json.loads(OUT.read_text()) if OUT.exists() else {}
binary = {"nc_first_eq", "pc_eq", "pc_both", "num_both", "addr_empty_any", "is_s3", "same_country", "hn_eq", "hn_both",
          "hn_prefix", "init_pref", "ad_core_empty", "legal_eq", "legal_conflict", "twin_better", "n_rep"}
for src, tgt in (("us", "india"), ("india", "us")):
    lf = pl.scan_parquet(parts).select(cols).join(cmap.lazy(), on="s1_idx")
    tr = add_rel(lf.filter(~pl.col("is_val") & ~pl.col("is_es") & (pl.col("country_n") == src)).collect())
    es = add_rel(lf.filter(pl.col("is_es") & (pl.col("country_n") == src)).collect())
    sv = add_rel(lf.filter(pl.col("is_val") & (pl.col("country_n") == src)).collect())
    tv = add_rel(lf.filter(pl.col("is_val") & (pl.col("country_n") == tgt)).collect())
    tvs = vt.filter(pl.col("country_n") == src).select("s1_idx", "n_true")
    tvt = vt.filter(pl.col("country_n") == tgt).select("s1_idx", "n_true")
    cont = [c for c in BASE if c not in binary]
    trp, esp, svp = pct([tr, es, sv], cont)
    (tvp,) = pct([tv], cont)
    for name, F, (a, b, c, d) in (("rel", BASE + REL, (tr, es, sv, tv)), ("pct", BASE, (trp, esp, svp, tvp)),
                                  ("rel+pct", BASE + REL, (trp, esp, svp, tvp))):
        key = f"{src}->{tgt} {name}"
        if key in res:
            continue
        t0 = time.time()
        dtr = xgb.QuantileDMatrix(a.select(F).to_numpy(), a["label"].to_numpy(), feature_names=F, max_bin=256)
        des = xgb.QuantileDMatrix(b.select(F).to_numpy(), b["label"].to_numpy(), ref=dtr, feature_names=F)
        bst = xgb.train(PRM, dtr, 4000, evals=[(des, "es")], early_stopping_rounds=80, verbose_eval=False)
        it = (0, bst.best_iteration + 1)
        s = c.select("s1_idx", "cand_idx", "label").with_columns(pl.Series("p", bst.predict(xgb.DMatrix(c.select(F).to_numpy(), feature_names=F), iteration_range=it), dtype=pl.Float32))
        t = d.select("s1_idx", "cand_idx", "label").with_columns(pl.Series("p", bst.predict(xgb.DMatrix(d.select(F).to_numpy(), feature_names=F), iteration_range=it), dtype=pl.Float32))
        dec = max(pp.tune_decision(s, tvs, 0.02, _Q()).values(), key=lambda z: z["f05"])
        k_src = pp.apply_decision(s, dec, 0.02).height / max(tvs.height, 1)
        best = min(((abs(pp.apply_decision(t, {"mode": "threshold", "param": th, "excl": dec.get("excl", False)}, 0.02).height
                         / max(tvt.height, 1) - k_src), th) for th in np.arange(0.30, 0.995, 0.01)), key=lambda x: x[0])
        shape = pp.eval_selection(pp.apply_decision(t, {"mode": "threshold", "param": best[1], "excl": dec.get("excl", False)}, 0.02), tvt)["f05"]
        res[key] = {"src_val": round(dec["f05"], 5), "tgt_val": round(pp.eval_selection(pp.apply_decision(t, dec, 0.02), tvt)["f05"], 5),
                    "tgt_shape": round(shape, 5), "iters": bst.best_iteration, "secs": round(time.time() - t0)}
        print(key, res[key], flush=True)
        OUT.write_text(json.dumps(res, indent=1))
        del dtr, des
ref = json.loads((pp.RUNS_DIR / "loco_idf.json").read_text())
for k in ("us->india", "india->us"):
    print(k, "name(ref)", ref[f"{k} name"]["src_val"], ref[f"{k} name"]["tgt_shape"], "| base", ref[f"{k} base"]["tgt_shape"])
print("rel done")
