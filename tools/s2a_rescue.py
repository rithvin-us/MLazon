"""Labelled val: pairs whose decision flips between the stacked-p decision and stage 2 (OOF, trained on dense val,
applied to plain val as in stage2_cross). True match rate of each flip direction, overall and for near-certain copies
(identical core name, core address 100, same house number)."""
import sys
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline as pp  # noqa: E402
from stage2_margin_lib import feats, FEAT  # noqa: E402

R = pp.RUNS_DIR
COLS = [f for f in FEAT if f not in ("s1_sum", "s1_n02")]
MONO = "(" + ",".join("1" if f in ("p", "margin", "s1_max") else ("-1" if f in ("max_other", "rank_rec") else "0") for f in COLS) + ")"
PRM = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", device="cuda", eta=0.05, max_depth=4, min_child_weight=20,
           subsample=0.8, monotone_constraints=MONO, seed=0)


class _Q:
    def log(self, m):
        pass


v = pl.read_parquet(sorted(R.glob("*-v10_bge2"))[-1] / "val_scored_ce.parquet").filter(pl.col("p") >= 0.02)
vt = pl.read_parquet(sorted(R.glob("*-v10"))[-1] / "val_truth.parquet")
tvc = vt.select("s1_idx", "n_true")
test_sc = pl.read_parquet(sorted(R.glob("*-v10_bge2_dense"))[-1] / "pred" / "scored-*.parquet", columns=["s1_idx", "cand_idx", "p"])
aug, _ = pp.density_augment(v, vt, test_sc, seed=42)
rng = np.random.default_rng(1)
aug = aug.with_columns(pl.when(pl.col("cand_idx") < 0).then((pl.col("p") + pl.Series(rng.uniform(-0.01, 0.01, aug.height).astype(np.float32))).clip(0.02, 1.0))
                       .otherwise(pl.col("p")).cast(pl.Float32).alias("p"))
da, dp = feats(aug), feats(v)
fold = (pl.col("s1_idx") * 7919 % 5).alias("fold")
trn = da.join(vt.select("s1_idx"), on="s1_idx", how="semi").with_columns(fold)
dv = dp.join(vt.select("s1_idx"), on="s1_idx", how="semi").with_columns(fold)
dc = dp.join(vt.select("s1_idx"), on="s1_idx", how="anti")
oof, cp = np.zeros(dv.height, np.float32), np.zeros(dc.height, np.float32)
for k in range(5):
    tr = trn.filter(pl.col("fold") != k)
    b = xgb.train(PRM, xgb.DMatrix(tr.select(COLS).to_numpy(), tr["label"].to_numpy(), feature_names=COLS), 400)
    m = (dv["fold"] == k).to_numpy()
    oof[m] = b.predict(xgb.DMatrix(dv.filter(pl.col("fold") == k).select(COLS).to_numpy(), feature_names=COLS))
    cp += b.predict(xgb.DMatrix(dc.select(COLS).to_numpy(), feature_names=COLS)) / 5
s = pl.concat([dv.drop("fold").with_columns(pl.Series("p2", oof)), dc.with_columns(pl.Series("p2", cp))])
d0 = max(pp.tune_decision(s.select("s1_idx", "cand_idx", "label", "p"), tvc, 0.02, _Q()).values(), key=lambda z: z["f05"])
d1 = max(pp.tune_decision(s.select("s1_idx", "cand_idx", "label", pl.col("p2").alias("p")), tvc, 0.02, _Q()).values(), key=lambda z: z["f05"])
a0 = pp.apply_decision(s.select("s1_idx", "cand_idx", "label", "p"), d0, 0.02).select("s1_idx", "cand_idx", pl.lit(True).alias("a0"))
a1 = pp.apply_decision(s.select("s1_idx", "cand_idx", "label", pl.col("p2").alias("p")), d1, 0.02).select("s1_idx", "cand_idx", pl.lit(True).alias("a1"))
x = (s.join(vt.select("s1_idx"), on="s1_idx", how="semi").join(a0, on=["s1_idx", "cand_idx"], how="left").join(a1, on=["s1_idx", "cand_idx"], how="left")
     .with_columns(pl.col("a0").fill_null(False), pl.col("a1").fill_null(False)))
evv = s.join(vt.select("s1_idx"), on="s1_idx", how="semi")
base1 = pp.eval_selection(pp.apply_decision(s.select("s1_idx", "cand_idx", "label", pl.col("p2").alias("p")), d1, 0.02), tvc)["f05"]
top = s.with_columns(pl.col("p").rank("ordinal", descending=True).over("cand_idx").alias("rr"))
for tau in (0.98, 0.99, 0.995, 0.999):
    sel = pp.apply_decision(s.select("s1_idx", "cand_idx", "label", pl.col("p2").alias("p")), d1, 0.02).select("s1_idx", "cand_idx", "label")
    resc = top.filter((pl.col("p") >= tau) & (pl.col("rr") == 1)).select("s1_idx", "cand_idx", "label")
    u = pl.concat([sel, resc]).unique(["s1_idx", "cand_idx"])
    print(f"stage 2 {base1:.5f} | + rescue stacked p >= {tau} & best claimant: {pp.eval_selection(u, tvc)['f05']:.5f} (+{resc.join(sel, on=['s1_idx', 'cand_idx'], how='anti').height} pairs)")