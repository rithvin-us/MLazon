"""One name word replaced (not a typo) + same address: true-match rate on labelled val by the replaced words' IDF,
vs model p; and how often this pattern is chosen in France test."""
import sys
import polars as pl
sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
import pipeline as pp
R = pp.RUNS_DIR
src = sorted(R.glob("*-v10src"))[-1]
cols = ["s1_idx", "cand_idx", "nc_tset", "ad_tset", "hn_eq", "n_rep", "n_rep_idf", "n_idf_jacc", "len_addr_c"]
f = pl.scan_parquet(sorted((src / "train_feats").glob("part-*.parquet"))).filter(pl.col("is_val")).select("label", *cols).collect()
f = f.join(pl.read_parquet(sorted(R.glob("*-v10s_ce"))[-1] / "val_scored_ce.parquet").select("s1_idx", "cand_idx", "p"), on=["s1_idx", "cand_idx"], how="left").with_columns(pl.col("p").fill_null(0.0))
sw = f.filter((pl.col("n_rep") == 1) & (pl.col("ad_tset") >= 95) & (pl.col("len_addr_c") > 0))
print("VAL one-word-replaced + same address:", sw.height, "match rate", round(sw["label"].mean(), 3), "mean p", round(sw["p"].mean(), 3))
print(sw.group_by(pl.col("n_rep_idf").cut([2, 4, 6, 8]).alias("rep_idf"), pl.col("hn_eq")).agg(pl.len(), pl.col("label").mean().round(3).alias("rate"), pl.col("p").mean().round(3).alias("mean_p")).sort("rep_idf", "hn_eq"))
t = pl.scan_parquet(sorted((sorted(R.glob("*-v10testsrc"))[-1] / "test_feats").glob("part-*.parquet"))).select(cols).collect()
cm = pp.scan_norm("test", "s1").select(pl.col("idx").alias("s1_idx"), "country_n").collect()
t = t.join(cm, on="s1_idx").join(pl.read_parquet(sorted(R.glob("*-v10s_ce_dense"))[-1] / "pred" / "scored-*.parquet").select("s1_idx", "cand_idx", "p"), on=["s1_idx", "cand_idx"], how="left").with_columns(pl.col("p").fill_null(0.0))
ts = t.filter((pl.col("n_rep") == 1) & (pl.col("ad_tset") >= 95) & (pl.col("len_addr_c") > 0))
n = cm.group_by("country_n").len()
print("TEST one-word-replaced + same address, per S1, and share with p>=0.885 (France threshold)")
print(ts.group_by("country_n").agg(pl.len().alias("pairs"), pl.col("n_rep_idf").mean().round(2).alias("mean_rep_idf"), pl.col("p").mean().round(3).alias("mean_p"),
      (pl.col("p") >= 0.885).sum().alias("p>=0.885")).join(n, on="country_n").with_columns((pl.col("pairs") / pl.col("len")).round(3).alias("per_S1"), (pl.col("p>=0.885") / pl.col("len")).round(4).alias("chosen_per_S1")).sort("country_n"))
