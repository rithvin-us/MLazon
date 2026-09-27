"""Stage 2 on competition features (the other pipeline's u_margin idea), checked on labelled val (US/India) with the
final stacked scores (v10_bge2: L12 + bge). Per (S1, record): p, best competing S1's p for the record, margin,
competitors >= 0.2, rank among the record's S1s, the S1's best p, rank within the S1, the S1's sum of p, candidates
>= 0.2, gap to the next candidate. XGBoost depth 4, 5-fold out-of-fold by S1 (competitor rows scored by the fold
models' average). Decision family re-tuned on p and on p2 exactly as the pipeline does (tune_decision, exclusivity).
Also per crowding: S1 whose records are contested (the slice France has 3.5x more of)."""
import sys

import numpy as np
import polars as pl
import xgboost as xgb

sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
import pipeline as pp  # noqa: E402

R = pp.RUNS_DIR
SRC = sys.argv[1] if len(sys.argv) > 1 else "v10_bge2"
v = pl.read_parquet(sorted(R.glob(f"*-{SRC}"))[-1] / "val_scored_ce.parquet")
vt = pl.read_parquet(sorted(R.glob("*-v10"))[-1] / "val_truth.parquet").select("s1_idx", "n_true")
FEAT = ["p", "max_other", "margin", "n_other", "rank_rec", "s1_max", "rank_s1", "s1_sum", "s1_n02", "gap_next"]


def feats(df):
    df = df.with_columns(
        pl.col("p").rank("ordinal", descending=True).over("cand_idx").alias("rank_rec"),
        pl.col("p").rank("ordinal", descending=True).over("s1_idx").alias("rank_s1"),
        pl.col("p").max().over("s1_idx").alias("s1_max"), pl.col("p").sum().over("s1_idx").alias("s1_sum"),
        (pl.col("p") >= 0.2).sum().over("s1_idx").alias("s1_n02"),
        ((pl.col("p") >= 0.2).sum().over("cand_idx") - (pl.col("p") >= 0.2).cast(pl.UInt32)).alias("n_other"))
    top2 = df.group_by("cand_idx").agg(pl.col("p").top_k(2).alias("t"))
    df = df.join(top2, on="cand_idx").with_columns(
        pl.when(pl.col("p") >= pl.col("t").list.first()).then(pl.col("t").list.get(1, null_on_oob=True)).otherwise(pl.col("t").list.first()).fill_null(0.0).alias("max_other")).drop("t")
    nxt = df.sort(["s1_idx", "p"], descending=[False, True]).with_columns(pl.col("p").shift(-1).over("s1_idx").fill_null(0.0).alias("p_next"))
    df = nxt.with_columns((pl.col("p") - pl.col("p_next")).alias("gap_next"), (pl.col("p") - pl.col("max_other")).alias("margin")).drop("p_next")
    return df


class _Q:
    def log(self, m):
        pass


d = feats(v).with_columns(pl.col(c).cast(pl.Float32) for c in FEAT)
is_val = d.join(vt.select("s1_idx"), on="s1_idx", how="semi")
comp = d.join(vt.select("s1_idx"), on="s1_idx", how="anti")
fold = (pl.col("s1_idx") * 7919 % 5).alias("fold")
is_val = is_val.with_columns(fold)
PRM = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", device="cuda", eta=0.05, max_depth=4, min_child_weight=20,
           subsample=0.8, colsample_bytree=1.0, monotone_constraints="(" + ",".join("1" if f in ("p", "margin", "s1_max") else ("-1" if f in ("max_other", "rank_rec") else "0") for f in FEAT) + ")")
oof = np.zeros(is_val.height, np.float32)
cp = np.zeros(comp.height, np.float32)
for k in range(5):
    tr, te = is_val.filter(pl.col("fold") != k), is_val.filter(pl.col("fold") == k)
    b = xgb.train(PRM, xgb.DMatrix(tr.select(FEAT).to_numpy(), tr["label"].to_numpy(), feature_names=FEAT), 400)
    idx = np.flatnonzero((is_val["fold"] == k).to_numpy())
    oof[idx] = b.predict(xgb.DMatrix(te.select(FEAT).to_numpy(), feature_names=FEAT))
    cp += b.predict(xgb.DMatrix(comp.select(FEAT).to_numpy(), feature_names=FEAT)) / 5
s2 = pl.concat([is_val.with_columns(pl.Series("p2", oof)).drop("fold"), comp.with_columns(pl.Series("p2", cp))])
base = max(pp.tune_decision(s2.select("s1_idx", "cand_idx", "label", "p"), vt, 0.02, _Q()).values(), key=lambda z: z["f05"])
new = max(pp.tune_decision(s2.select("s1_idx", "cand_idx", "label", pl.col("p2").alias("p")), vt, 0.02, _Q()).values(), key=lambda z: z["f05"])
print(f"[val {SRC}] stacked p: {base['mode']} {base['param']} excl={base['excl']} F0.5 {base['f05']:.5f} P {base['precision']:.4f} R {base['recall']:.4f}")
print(f"[val {SRC}] stage 2 : {new['mode']} {new['param']} excl={new['excl']} F0.5 {new['f05']:.5f} P {new['precision']:.4f} R {new['recall']:.4f}  (delta {new['f05'] - base['f05']:+.5f})")
# contested slice: val S1 with >= 1 candidate that another S1 scores >= 0.2
cs = is_val.filter(pl.col("n_other") >= 1).select("s1_idx").unique()
for name, sub in (("contested S1", cs), ("uncontested S1", vt.select("s1_idx").join(cs, on="s1_idx", how="anti"))):
    t = vt.join(sub, on="s1_idx")
    f0 = pp.eval_selection(pp.apply_decision(s2.select("s1_idx", "cand_idx", "label", "p"), base, 0.02).join(sub, on="s1_idx"), t)["f05"]
    f1 = pp.eval_selection(pp.apply_decision(s2.select("s1_idx", "cand_idx", "label", pl.col("p2").alias("p")), new, 0.02).join(sub, on="s1_idx"), t)["f05"]
    print(f"[val {SRC}] {name}: {t.height:,} S1, F0.5 {f0:.5f} -> {f1:.5f} ({f1 - f0:+.5f})")
