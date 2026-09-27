"""Who deflates French different-house-number pairs: stage 1 (XGBoost) or the cross-encoder stack? Same slice
(numbers differ, same core name, core address >= 90), per context, stage-1 p (v10sfr_test) vs final stacked p
(v10sfr_bge2_dense). Labelled val reference: the same slice's stage-1 p vs truth."""
import sys

import polars as pl

sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
import pipeline as pp  # noqa: E402

R = pp.RUNS_DIR
COLS = ["s1_idx", "cand_idx", "hn_eq", "hn_both", "num_jacc", "nc_tset", "ad_core_tset"]
fr = pp.scan_norm("test", "s1").filter(pl.col("country_n") == "france").select(pl.col("idx").alias("s1_idx")).collect()
tf = pl.scan_parquet(sorted((sorted(R.glob("*-v10testsrc"))[-1] / "test_feats").glob("part-*.parquet"))).select(COLS).join(fr.lazy(), on="s1_idx").collect()
p1 = pl.read_parquet(sorted(R.glob("*-v10sfr_test"))[-1] / "pred" / "scored-*.parquet").select("s1_idx", "cand_idx", pl.col("p").alias("p1")).join(fr, on="s1_idx")
p2 = pl.read_parquet(sorted(R.glob("*-v10sfr_bge2_dense"))[-1] / "pred" / "scored-*.parquet").select("s1_idx", "cand_idx", pl.col("p").alias("p2")).join(fr, on="s1_idx")
t = tf.join(p1, on=["s1_idx", "cand_idx"], how="left").join(p2, on=["s1_idx", "cand_idx"], how="left").with_columns(pl.col("p1").fill_null(0.0), pl.col("p2").fill_null(0.0))
same = t.filter((pl.col("hn_eq") == 1) & (pl.col("nc_tset") == 100)).select("s1_idx").unique().with_columns(pl.lit(True).alias("copy"))
h = t.filter((pl.col("hn_both") == 1) & (pl.col("hn_eq") == 0) & (pl.col("num_jacc") == 0) & (pl.col("nc_tset") == 100) & (pl.col("ad_core_tset") >= 90)).join(same, on="s1_idx", how="left")
h = h.with_columns(pl.when(pl.col("copy").fill_null(False)).then(pl.lit("A: same-hn copy exists")).otherwise(pl.lit("B: no same-hn copy")).alias("ctx"))
print("[France] numbers differ, same core name, core address >= 90: stage-1 p vs final p")
print(h.group_by("ctx").agg(pl.len().alias("n"), pl.col("p1").mean().round(3).alias("stage1_mean"), (pl.col("p1") >= 0.5).mean().round(3).alias("stage1>=.5"),
                            pl.col("p2").mean().round(3).alias("final_mean"), (pl.col("p2") >= 0.865).mean().round(3).alias("final_accepted")).sort("ctx"))
cm = pp.scan_norm("train", "s1").select(pl.col("idx").alias("s1_idx"), "country_n").collect()
vs = pl.read_parquet(sorted(R.glob("*-v10sfr"))[-1] / "val_scored.parquet").select("s1_idx", "cand_idx", "label", pl.col("p").alias("p1")).filter(pl.col("cand_idx") >= 0)
vf = pl.scan_parquet(sorted((sorted(R.glob("*-v10src"))[-1] / "train_feats").glob("part-*.parquet"))).filter(pl.col("is_val")).select(COLS).collect()
v = vf.join(vs, on=["s1_idx", "cand_idx"]).join(cm, on="s1_idx")
vsame = v.filter((pl.col("hn_eq") == 1) & (pl.col("nc_tset") == 100)).select("s1_idx").unique().with_columns(pl.lit(True).alias("copy"))
w = v.filter((pl.col("hn_both") == 1) & (pl.col("hn_eq") == 0) & (pl.col("num_jacc") == 0) & (pl.col("nc_tset") == 100) & (pl.col("ad_core_tset") >= 90)).join(vsame, on="s1_idx", how="left")
w = w.with_columns(pl.when(pl.col("copy").fill_null(False)).then(pl.lit("A: same-hn copy exists")).otherwise(pl.lit("B: no same-hn copy")).alias("ctx"))
print("[val, same France model (v10sfr) stage 1] same slice: n, match rate, stage-1 mean p")
print(w.group_by("country_n", "ctx").agg(pl.len().alias("n"), pl.col("label").mean().round(3).alias("match"), pl.col("p1").mean().round(3).alias("stage1_mean")).sort("country_n", "ctx"))
