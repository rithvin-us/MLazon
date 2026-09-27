"""Differing house number, same core name, core address >= 90 ("hd pairs"): is the candidate a nudged distractor or a
copy with a house-number typo? Split by structural context of the S1:
  A  the S1 has another candidate with the SAME house number and the same core name (a copy exists -> hd = nudged?)
  B  the S1 has no candidate with its house number
Labelled val (US, India): match rate and mean p per context. France test: count and mean p / accepted per context."""
import sys

import polars as pl

sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
import pipeline as pp  # noqa: E402

R = pp.RUNS_DIR
COLS = ["s1_idx", "cand_idx", "hn_eq", "hn_both", "hn_sim", "nc_tset", "ad_core_tset"]


def ctx(df):
    same = df.filter((pl.col("hn_eq") == 1) & (pl.col("nc_tset") == 100)).select("s1_idx").unique().with_columns(pl.lit(True).alias("has_copy"))
    anyhn = df.filter(pl.col("hn_eq") == 1).select("s1_idx").unique().with_columns(pl.lit(True).alias("has_hn"))
    hd = df.filter((pl.col("hn_both") == 1) & (pl.col("hn_eq") == 0) & (pl.col("nc_tset") == 100) & (pl.col("ad_core_tset") >= 90))
    hd = hd.join(same, on="s1_idx", how="left").join(anyhn, on="s1_idx", how="left").with_columns(
        pl.when(pl.col("has_copy").fill_null(False)).then(pl.lit("A: same-hn copy exists"))
        .when(pl.col("has_hn").fill_null(False)).then(pl.lit("A2: same-hn cand, other name"))
        .otherwise(pl.lit("B: no same-hn candidate")).alias("ctx"))
    return hd


cm = pp.scan_norm("train", "s1").select(pl.col("idx").alias("s1_idx"), "country_n").collect()
vs = pl.read_parquet(sorted(R.glob("*-v10"))[-1] / "val_scored.parquet").select("s1_idx", "cand_idx", "label", "p").filter(pl.col("cand_idx") >= 0)
ft = pl.scan_parquet(sorted((sorted(R.glob("*-v10src"))[-1] / "train_feats").glob("part-*.parquet"))).filter(pl.col("is_val")).select(COLS).collect()
v = ctx(ft.join(vs, on=["s1_idx", "cand_idx"]).join(cm, on="s1_idx"))
print("[val] hd pairs by context: n, match rate, mean p")
print(v.group_by("country_n", "ctx").agg(pl.len().alias("n"), pl.col("label").mean().round(3).alias("match"), pl.col("p").mean().round(3).alias("mean_p")).sort("country_n", "ctx"))
del vs, ft, v
fr = pp.scan_norm("test", "s1").filter(pl.col("country_n") == "france").select(pl.col("idx").alias("s1_idx")).collect()
tf = pl.scan_parquet(sorted((sorted(R.glob("*-v10testsrc"))[-1] / "test_feats").glob("part-*.parquet"))).select(COLS).join(fr.lazy(), on="s1_idx").collect()
sc = pl.read_parquet(sorted(R.glob("*-v10sfr_bge2_dense"))[-1] / "pred" / "scored-*.parquet").select("s1_idx", "cand_idx", "p").join(fr, on="s1_idx")
t = ctx(tf.join(sc, on=["s1_idx", "cand_idx"], how="left").with_columns(pl.col("p").fill_null(0.0)))
print("[France] hd pairs by context: n, mean p, accepted (p>=0.865)")
print(t.group_by("ctx").agg(pl.len().alias("n"), pl.col("p").mean().round(3).alias("mean_p"), (pl.col("p") >= 0.865).mean().round(3).alias("accepted")).sort("ctx"))
