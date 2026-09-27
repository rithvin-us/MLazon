"""Label-free check: near-certain true copies (same core name, same house number, same core address) should almost
always be chosen. Chosen rate by country and, for France, by candidate formatting. A format with a clearly lower rate
is a fixable country-specific failure."""
import sys
import polars as pl
sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
import pipeline as pp
R = pp.RUNS_DIR
cm = pp.scan_norm("test", "s1").select(pl.col("idx").alias("s1_idx"), "country_n", pl.col("entity_id").alias("sid")).collect()
f = (pl.scan_parquet(sorted((sorted(R.glob("*-v10testsrc"))[-1] / "test_feats").glob("part-*.parquet")))
     .select("s1_idx", "cand_idx", "cid", "nc_tset", "nf_tset", "hn_eq", "hn_both", "ad_core_tset", "ad_tset", "legal_conflict", "is_s3")
     .filter((pl.col("nc_tset") >= 100) & (pl.col("hn_eq") == 1) & (pl.col("ad_core_tset") >= 90)).collect())
sub = (pl.read_csv("output/SUBMIT_THIS/matching_results.tsv", separator="\t", quote_char=None, infer_schema=False)
       .with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids")
       .filter(pl.col("matched_entity_ids") != "").select(pl.col("source1_entity_id").alias("sid"), pl.col("matched_entity_ids").alias("cid"), pl.lit(1).alias("ch")))
owner = sub.select("cid", pl.col("sid").alias("owner"))
f = f.join(cm, on="s1_idx").join(sub, on=["sid", "cid"], how="left").join(owner, on="cid", how="left").with_columns(pl.col("ch").fill_null(0))
f = f.with_columns(pl.when(pl.col("ch") == 1).then(pl.lit("this S1")).when(pl.col("owner").is_not_null()).then(pl.lit("other S1")).otherwise(pl.lit("nobody")).alias("given"))
print("near-certain copies (core name = , house no = , core address >= 90): where do they go?")
print(f.group_by("country_n").agg(pl.len().alias("pairs"), (pl.col("given") == "this S1").mean().round(4).alias("to_this"),
                                  (pl.col("given") == "other S1").mean().round(4).alias("to_other"), (pl.col("given") == "nobody").mean().round(4).alias("to_nobody")).sort("country_n"))
raw = pl.concat([pl.scan_parquet(f"cache/raw_test_s{i}.parquet").select(pl.col("entity_id").alias("cid"), pl.col("business_address").alias("ra"), pl.col("business_name").alias("rn")) for i in (2, 3)])
fr = f.filter(pl.col("country_n") == "france").join(raw.collect(), on="cid", how="left").with_columns(pl.col("ra", "rn").fill_null(""))
ra = pl.col("ra").str.to_lowercase()
fr = fr.with_columns(
    ra.str.contains(r"\b(nord|gironde|loire-atlantique|loire atlantique|pas-de-calais|pas de calais)\b").alias("dept"),
    ra.str.contains(r"hauts-de-france|nouvelle-aquitaine|pays de la loire").alias("region"),
    ra.str.contains(r"^(no\.?|n°|#)\s").alias("no_prefix"), ra.str.contains(r"\b\d+\s*(bis|ter)\b").alias("bis_ter"),
    (pl.col("legal_conflict") == 1).alias("legal_swap"), (pl.col("is_s3") == 1).alias("src_s3"),
    (pl.col("rn") == pl.col("rn").str.to_uppercase()).alias("name_caps"), pl.col("rn").str.contains(r"[^\x00-\x7F]").alias("accents"))
print("\nFRANCE near-certain copies: share given to this S1 by candidate format (True vs False)")
for c in ("dept", "region", "no_prefix", "bis_ter", "legal_swap", "src_s3", "name_caps", "accents"):
    g = fr.group_by(c).agg(pl.len(), (pl.col("given") == "this S1").mean().round(4).alias("to_this"), (pl.col("given") == "nobody").mean().round(4).alias("to_nobody")).sort(c)
    print(f"  {c:11s}", g.rows())
