"""Same name + same house number but different street: are these other businesses (owned by another S1)?"""
import polars as pl
R = "runs"
s1 = pl.scan_parquet("cache/norm_test_s1_v5/*.parquet").select(pl.col("idx").alias("s1_idx"), pl.col("entity_id").alias("s1_id"), pl.col("country_n").alias("country"))
sc = pl.scan_parquet(f"{R}/20260926-155941-v9_ce/pred/scored-*.parquet").select("s1_idx", "cand_idx", "p")
m = (pl.read_csv(f"{R}/20260926-155941-v9_ce/output/matching_results.tsv", separator="\t", quote_char=None, infer_schema=False)
     .with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids")
     .filter(pl.col("matched_entity_ids") != "").select(pl.col("source1_entity_id").alias("owner"), pl.col("matched_entity_ids").alias("cid")))
f = (pl.scan_parquet(f"{R}/20260926-151110-v9_test/test_feats/*.parquet").select("s1_idx", "cand_idx", "cid", "nc_tset", "hn_eq", "ad_core_tset")
     .filter((pl.col("nc_tset") == 100) & (pl.col("hn_eq") == 1))
     .join(s1, on="s1_idx").join(sc, on=["s1_idx", "cand_idx"], how="left").with_columns(pl.col("p").fill_null(0.0)).collect()
     .join(m, on="cid", how="left")
     .with_columns(pl.when(pl.col("owner").is_null()).then(pl.lit("nobody")).when(pl.col("owner") == pl.col("s1_id")).then(pl.lit("this S1"))
                   .otherwise(pl.lit("other S1")).alias("given_to"),
                   pl.when(pl.col("ad_core_tset") >= 80).then(pl.lit("same street")).when(pl.col("ad_core_tset") < 50).then(pl.lit("diff street"))
                   .otherwise(pl.lit("partial")).alias("street")))
print("pairs with identical name AND identical house number, by country and street agreement")
t = f.group_by("country", "street").agg(pl.len().alias("pairs"), pl.col("p").mean().round(3).alias("mean_p"),
                                         (pl.col("given_to") == "this S1").mean().round(3).alias("to_this"),
                                         (pl.col("given_to") == "other S1").mean().round(3).alias("to_other"),
                                         (pl.col("given_to") == "nobody").mean().round(3).alias("to_nobody"))
n1 = s1.group_by("country").len().collect()
print(t.join(n1, on="country").with_columns((pl.col("pairs") / pl.col("len")).round(3).alias("per_S1")).drop("len").sort("country", "street"))
