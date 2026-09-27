"""France pseudo-labelled training parts from structure (no labels): from the current France scores (v10s + CE,
test-density run) and the saved test features (all FEATURES).
  pos       p >= POS and this S1 is the record's best-scoring S1
  hard neg  the record is owned by ANOTHER French S1 with p >= POS
  easy neg  p < 0.02 (sampled)
Writes runs/<ts>-frpseudo/part-*.parquet with FEATURES + label + w + is_val/is_es=False (for retrain --extra-parts).
  python build_fr_pseudo.py [POS=0.98] [W=0.5]"""
import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
import pipeline as pp  # noqa: E402
from features import FEATURES  # noqa: E402

POS = float(sys.argv[1]) if len(sys.argv) > 1 else 0.98
W = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
R = pp.RUNS_DIR
out = R / f"{time.strftime('%Y%m%d-%H%M%S')}-frpseudo"
out.mkdir()
cm = pp.scan_norm("test", "s1").select(pl.col("idx").alias("s1_idx"), "country_n").collect()
fr = cm.filter(pl.col("country_n") == "france").select("s1_idx")
sc = pl.read_parquet(sorted(R.glob("*-v10s_ce_dense"))[-1] / "pred" / "scored-*.parquet").select("s1_idx", "cand_idx", "p").join(fr, on="s1_idx")
sc = sc.with_columns(pl.col("p").max().over("cand_idx").alias("pmax"), pl.col("p").rank("ordinal", descending=True).over("cand_idx").alias("own"))
pos = sc.filter((pl.col("p") >= POS) & (pl.col("own") == 1)).select("s1_idx", "cand_idx", pl.lit(1, pl.Int8).alias("label"))
hneg = sc.filter((pl.col("pmax") >= POS) & (pl.col("own") > 1)).select("s1_idx", "cand_idx", pl.lit(0, pl.Int8).alias("label"))
lab = pl.concat([pos, hneg])
feats = sorted((sorted(R.glob("*-v10testsrc"))[-1] / "test_feats").glob("part-*.parquet"))
n = {"pos": pos.height, "hneg": hneg.height, "eneg": 0}
for i, fp in enumerate(feats):
    f = pl.read_parquet(fp).join(fr, on="s1_idx")
    if not f.height:
        continue
    a = f.join(lab, on=["s1_idx", "cand_idx"])
    e = f.join(sc.select("s1_idx", "cand_idx"), on=["s1_idx", "cand_idx"], how="anti").sample(fraction=0.15, seed=i).with_columns(pl.lit(0, pl.Int8).alias("label"))
    n["eneg"] += e.height
    part = pl.concat([a.select("s1_idx", "cand_idx", *FEATURES, "label"), e.select("s1_idx", "cand_idx", *FEATURES, "label")])
    part.with_columns(pl.lit(W, pl.Float32).alias("w"), pl.lit(False).alias("is_val"), pl.lit(False).alias("is_es")).write_parquet(out / f"part-{i:05d}.parquet")
print(f"France pseudo parts -> {out.name}: {n}, POS {POS}, W {W}")
