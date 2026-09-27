"""Type-rate calibration transfer, checked on the unseen-country proxy.
The generator is the same in every country, so P(match | pair type) should be too, for types defined by
vocabulary-free relations (exact/near name, address present/equal/different, house number equal, S1 name unique).
1. are type rates equal in US and India (labelled)?  2. train on SRC, score TGT; shift TGT logits per type so each
type's mean p equals SRC's labelled rate; France-style decision; F0.5 before/after.   -> runs/loco_typecal.json
"""
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
TYPE = pl.concat_str([
    pl.when(pl.col("nc_tset") >= 100).then(pl.lit("n100")).when(pl.col("nc_tset") >= 90).then(pl.lit("n90"))
    .when(pl.col("nc_tset") >= 75).then(pl.lit("n75")).otherwise(pl.lit("n0")),
    pl.when(pl.col("len_addr_c") == 0).then(pl.lit("aE")).when(pl.col("ad_tset") >= 95).then(pl.lit("a95"))
    .when(pl.col("ad_tset") >= 70).then(pl.lit("a70")).otherwise(pl.lit("a0")),
    pl.when(pl.col("hn_both") == 0).then(pl.lit("h?")).when(pl.col("hn_eq") == 1).then(pl.lit("h=")).otherwise(pl.lit("h!")),
    pl.when(pl.col("s1_same_name") <= 1).then(pl.lit("u")).otherwise(pl.lit("s"))], separator="|").alias("type")


class _Q:
    def log(self, m):
        pass


def shape_f(t, tvt, k_src, excl):
    best = min(((abs(pp.apply_decision(t, {"mode": "threshold", "param": th, "excl": excl}, 0.02).height / max(tvt.height, 1) - k_src), th)
                for th in np.arange(0.30, 0.995, 0.01)), key=lambda x: x[0])
    return pp.eval_selection(pp.apply_decision(t, {"mode": "threshold", "param": best[1], "excl": excl}, 0.02), tvt)["f05"], best[1]


cmap = pp.scan_norm("train", "s1").select(pl.col("idx").alias("s1_idx"), "country_n").collect()
vt = pl.read_parquet(R / "val_truth.parquet")
parts = sorted((R / "train_feats").glob("part-*.parquet"))
cols = ["s1_idx", "cand_idx", "label", "is_val", "is_es", *F]
lf = pl.scan_parquet(parts).select(cols).join(cmap.lazy(), on="s1_idx").with_columns(TYPE)
res = {}
# 1. rate invariance on all labelled train rows
rates = lf.group_by("country_n", "type").agg(pl.len().alias("n"), pl.col("label").mean().alias("rate")).collect()
w = rates.pivot(on="country_n", index="type", values=["n", "rate"]).filter((pl.col("n_us") >= 300) & (pl.col("n_india") >= 300))
diff = (w["rate_us"] - w["rate_india"]).abs()
wt = (w["n_us"] + w["n_india"]).to_numpy()
res["rate_invariance"] = {"types": w.height, "weighted_mean_abs_diff": round(float((diff.to_numpy() * wt).sum() / wt.sum()), 4),
                          "max_abs_diff": round(float(diff.max()), 4)}
print("rate invariance US vs India:", res["rate_invariance"], flush=True)
print(w.with_columns(diff.alias("absdiff")).sort("absdiff", descending=True).head(12))
for src, tgt in (("us", "india"), ("india", "us")):
    tr = lf.filter(~pl.col("is_val") & ~pl.col("is_es") & (pl.col("country_n") == src)).collect()
    es = lf.filter(pl.col("is_es") & (pl.col("country_n") == src)).collect()
    sv = lf.filter(pl.col("is_val") & (pl.col("country_n") == src)).collect()
    tv = lf.filter(pl.col("is_val") & (pl.col("country_n") == tgt)).collect()
    src_rate = tr.group_by("type").agg(pl.col("label").mean().alias("rate"), pl.len().alias("n_src"))
    dtr = xgb.QuantileDMatrix(tr.select(F).to_numpy(), tr["label"].to_numpy(), feature_names=F, max_bin=256)
    des = xgb.QuantileDMatrix(es.select(F).to_numpy(), es["label"].to_numpy(), ref=dtr, feature_names=F)
    b = xgb.train(PRM, dtr, 4000, evals=[(des, "es")], early_stopping_rounds=80, verbose_eval=False)
    it = (0, b.best_iteration + 1)
    pr = lambda d: d.select("s1_idx", "cand_idx", "label", "type").with_columns(  # noqa: E731
        pl.Series("p", b.predict(xgb.DMatrix(d.select(F).to_numpy(), feature_names=F), iteration_range=it), dtype=pl.Float32))
    s, t = pr(sv), pr(tv)
    tvs = vt.filter(pl.col("country_n") == src).select("s1_idx", "n_true")
    tvt = vt.filter(pl.col("country_n") == tgt).select("s1_idx", "n_true")
    dec = max(pp.tune_decision(s, tvs, 0.02, _Q()).values(), key=lambda z: z["f05"])
    k_src = pp.apply_decision(s, dec, 0.02).height / max(tvs.height, 1)
    f0, th0 = shape_f(t, tvt, k_src, dec.get("excl", False))
    # per-type logit shift on the target: mean p of the type -> source rate (types with >= 200 target pairs)
    lg = lambda x: np.log(np.clip(x, 1e-4, 1 - 1e-4) / (1 - np.clip(x, 1e-4, 1 - 1e-4)))  # noqa: E731
    out = {}
    for shrink in (1.0, 0.5):
        tm = t.group_by("type").agg(pl.col("p").mean().alias("mp"), pl.len().alias("n_t")).join(src_rate, on="type", how="left")
        tm = tm.with_columns(pl.when((pl.col("n_t") >= 200) & (pl.col("n_src") >= 200))
                             .then(pl.Series(np.nan_to_num(shrink * (lg(tm["rate"].fill_null(0.5).to_numpy()) - lg(tm["mp"].to_numpy())))))
                             .otherwise(0.0).alias("shift"))
        t2 = t.join(tm.select("type", "shift"), on="type", how="left").with_columns(
            (1 / (1 + (-(pl.col("p").clip(1e-4, 1 - 1e-4).log() - (1 - pl.col("p").clip(1e-4, 1 - 1e-4)).log() + pl.col("shift"))).exp())).cast(pl.Float32).alias("p"))
        f1, th1 = shape_f(t2.drop("shift"), tvt, k_src, dec.get("excl", False))
        out[f"shrink{shrink}"] = {"tgt_shape": round(f1, 5), "thr": round(float(th1), 2)}
    res[f"{src}->{tgt}"] = {"src_val": round(dec["f05"], 5), "tgt_shape_before": round(f0, 5), "thr_before": round(float(th0), 2), **out}
    print(f"{src}->{tgt}", res[f"{src}->{tgt}"], flush=True)
    (pp.RUNS_DIR / "loco_typecal.json").write_text(json.dumps(res, indent=1))
print("typecal done")
