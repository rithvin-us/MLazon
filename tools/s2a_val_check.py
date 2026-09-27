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
ft = (pl.scan_parquet(sorted((sorted(R.glob("*-v10src"))[-1] / "train_feats").glob("part-*.parquet"))).filter(pl.col("is_val"))
      .select("s1_idx", "cand_idx", "hn_eq", "ad_core_tset").collect())
n1 = pp.scan_norm("train", "s1").select(pl.col("idx").alias("s1_idx"), pl.col("name_core").alias("n1")).collect()
n2 = pp.scan_norm("train", "pool").select(pl.col("idx").alias("cand_idx"), pl.col("name_core").alias("n2")).collect()
x = x.join(ft, on=["s1_idx", "cand_idx"], how="left").join(n1, on="s1_idx").join(n2, on="cand_idx", how="left")
x = x.with_columns(((pl.col("n1") == pl.col("n2")) & (pl.col("ad_core_tset") == 100) & (pl.col("hn_eq") == 1)).fill_null(False).alias("nc"))
print(f"decisions: stacked {d0['mode']} {d0['param']} F0.5 {d0['f05']:.5f} | stage 2 {d1['mode']} {d1['param']} F0.5 {d1['f05']:.5f}")
for name, sub in (("all pairs", x), ("near-certain copies", x.filter("nc"))):
    lost = sub.filter(pl.col("a0") & ~pl.col("a1"))
    won = sub.filter(~pl.col("a0") & pl.col("a1"))
    print(f"[{name}] dropped by stage 2 {lost.height:,} (true {lost['label'].mean() if lost.height else 0:.3f}) | added by stage 2 {won.height:,} (true {won['label'].mean() if won.height else 0:.3f})")
