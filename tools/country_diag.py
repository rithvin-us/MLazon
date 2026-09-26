"""Per-country diagnostic of a scored test run: candidates/S1, uncertain-band share, max-p spread."""
import sys
import polars as pl
run = sys.argv[1]
sc = pl.scan_parquet(f"runs/{run}/pred/scored-*.parquet")
print(sc.collect_schema().names())
cols = sc.collect_schema().names()
p = "p_ce" if "p_ce" in cols else "p"
s1 = pl.scan_parquet("cache/norm_test_s1_v5/*.parquet").select(pl.col("idx").alias("s1_id"), pl.col("country_n").alias("country"))
key = "s1_idx"
df = sc.select(key, pl.col(p).alias("p")).rename({key: "s1_id"}).join(s1, on="s1_id").collect()
print(df.group_by("country").agg(
    pl.len().alias("pairs"),
    pl.col("s1_id").n_unique().alias("s1"),
    ((pl.col("p") >= 0.02) & (pl.col("p") < 0.995)).mean().round(4).alias("band"),
    ((pl.col("p") >= 0.2) & (pl.col("p") < 0.8)).mean().round(4).alias("mid"),
    (pl.col("p") >= 0.9).sum().alias("p90"),
).with_columns((pl.col("pairs") / pl.col("s1")).round(2).alias("c/S1"), (pl.col("p90") / pl.col("s1")).round(3).alias("p90/S1")).sort("country"))
per = df.group_by("country", "s1_id").agg(pl.col("p").max().alias("mx"))
print(per.group_by("country").agg(
    (pl.col("mx") < 0.5).mean().round(4).alias("maxp<0.5"),
    ((pl.col("mx") >= 0.5) & (pl.col("mx") < 0.9)).mean().round(4).alias("maxp.5-.9"),
).sort("country"))
