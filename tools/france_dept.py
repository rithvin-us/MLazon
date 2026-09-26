"""Does a department name in the candidate address (instead of the region code) lower its score?"""
import polars as pl
R = "runs"
s1 = pl.scan_parquet("cache/norm_test_s1_v5/*.parquet").filter(pl.col("country_n") == "france").select(pl.col("idx").alias("s1_idx"))
pool = pl.scan_parquet("cache/norm_test_pool_v5/*.parquet").filter(pl.col("country_n") == "france").select(pl.col("entity_id").alias("cid"), "addr")
sc = pl.scan_parquet(f"{R}/20260926-155941-v9_ce/pred/scored-*.parquet").select("s1_idx", "cand_idx", "p")
f = (pl.scan_parquet(f"{R}/20260926-151110-v9_test/test_feats/*.parquet").select("s1_idx", "cand_idx", "cid", "nc_tset", "hn_eq", "ad_tset", "ad_core_tset")
     .join(s1, on="s1_idx").join(sc, on=["s1_idx", "cand_idx"], how="left").with_columns(pl.col("p").fill_null(0.0))
     .join(pool, on="cid").with_columns(
         pl.when(pl.col("addr").str.contains(r"\b(nord|gironde|loire atlantique|pas de calais)\b")).then(pl.lit("dept"))
         .when(pl.col("addr").str.contains(r"\b(hdf|naq|pdl)\b")).then(pl.lit("region"))
         .when(pl.col("addr").fill_null("") == "").then(pl.lit("empty")).otherwise(pl.lit("neither")).alias("admin"))
     .collect())
print("all France pairs by candidate admin form")
print(f.group_by("admin").agg(pl.len(), pl.col("p").mean().round(3).alias("mean_p"), pl.col("ad_tset").mean().round(1), pl.col("ad_core_tset").mean().round(1)).sort("admin"))
print("\nsame name (nc_tset=100) AND same house number: strong-evidence pairs")
g = f.filter((pl.col("nc_tset") == 100) & (pl.col("hn_eq") == 1))
print(g.group_by("admin").agg(pl.len(), pl.col("p").mean().round(4).alias("mean_p"), (pl.col("p") < 0.895).mean().round(4).alias("below_thr"),
                              pl.col("ad_tset").mean().round(1), pl.col("ad_core_tset").mean().round(1)).sort("admin"))
