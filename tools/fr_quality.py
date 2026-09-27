"""Label-free quality profile of a matching file's France rows, next to the same file's US/India rows (whose quality we
know from validation, ~0.983-0.985 on test). The generator is the same in every country, so large gaps between the
two profiles point at France errors. Aggregates only.
  1. matches per S1: mean and distribution (0, 1, 2, 3, 4, 5+)
  2. evidence carried by accepted pairs: same house number, house numbers both present but different, identical core
     name, legal-form conflict, candidate name = S1 name + extra words, core address (no city/region) >= 90
  3. near-certain copies accepted (identical core name, core address 100, same house number)
  4. contested accepts: the record's second-best S1 also has p >= 0.5 (from the file's scored pairs, when given)
  python fr_quality.py "file1.tsv|label" "file2.tsv|label" ..."""
import sys

import polars as pl

sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
import pipeline as pp  # noqa: E402

R = pp.RUNS_DIR
s1 = pp.scan_norm("test", "s1").select(pl.col("idx").alias("s1_idx"), pl.col("entity_id").alias("source1_entity_id"), "country_n",
                                       pl.col("name_core").alias("n1")).collect()
s1 = s1.with_columns(pl.when(pl.col("country_n") == "france").then(pl.lit("france")).otherwise(pl.lit("us/india")).alias("grp"))
ref = s1.filter(pl.col("grp") == "us/india").sample(250_000, seed=7)  # US/India reference sample
keep = pl.concat([s1.filter(pl.col("grp") == "france"), ref]).select("s1_idx", "source1_entity_id", "grp", "n1")
pool = pp.scan_norm("test", "pool").select(pl.col("entity_id").alias("cid"), pl.col("name_core").alias("n2"))
feats = (pl.scan_parquet(sorted((sorted(R.glob("*-v10testsrc"))[-1] / "test_feats").glob("part-*.parquet")))
         .select("s1_idx", "cid", "hn_eq", "hn_both", "nc_tset", "legal_conflict", "ad_core_tset")
         .join(keep.lazy().select("s1_idx"), on="s1_idx").collect())
pn = pool.join(feats.select("cid").unique().lazy(), on="cid").collect()  # names of candidate records only


def profile(path, label):
    m = (pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False).join(keep, on="source1_entity_id")
         .with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",").list.eval(pl.element().filter(pl.element() != "")).alias("ids")))
    k = m.with_columns(pl.col("ids").list.len().alias("k"))
    dist = k.group_by("grp").agg(pl.col("k").mean().round(3).alias("mean_k"), *[(pl.col("k") == i).mean().round(4).alias(f"k={i}") for i in range(5)],
                                 (pl.col("k") >= 5).mean().round(4).alias("k>=5")).sort("grp")
    acc = (m.select("s1_idx", "grp", "n1", pl.col("ids")).explode("ids").drop_nulls("ids").rename({"ids": "cid"})
           .join(feats, on=["s1_idx", "cid"], how="left").join(pn, on="cid", how="left"))
    words = lambda c: pl.col(c).fill_null("").str.split(" ")  # noqa: E731
    acc = acc.with_columns(
        ((words("n2").list.set_difference(words("n1")).list.len() > 0) & (words("n1").list.set_difference(words("n2")).list.len() == 0)
         & (pl.col("n1") != pl.col("n2"))).alias("extra_word"))
    ev = acc.group_by("grp").agg(pl.len().alias("accepted"), (pl.col("hn_eq") == 1).mean().round(4).alias("same_hn"),
                                 ((pl.col("hn_both") == 1) & (pl.col("hn_eq") == 0)).mean().round(4).alias("hn_differs"),
                                 (pl.col("n1") == pl.col("n2")).mean().round(4).alias("same_core_name"),
                                 (pl.col("legal_conflict") == 1).mean().round(4).alias("legal_conflict"),
                                 pl.col("extra_word").mean().round(4).alias("extra_word"),
                                 (pl.col("ad_core_tset") >= 90).mean().round(4).alias("core_addr>=90")).sort("grp")
    nc = feats.join(keep.select("s1_idx", "grp", "n1"), on="s1_idx").join(pn, on="cid", how="left").filter(
        (pl.col("n1") == pl.col("n2")) & (pl.col("ad_core_tset") == 100) & (pl.col("hn_eq") == 1))
    nc = nc.join(acc.select("s1_idx", "cid", pl.lit(True).alias("acc")), on=["s1_idx", "cid"], how="left").group_by("grp").agg(
        pl.len().alias("near_certain"), pl.col("acc").fill_null(False).mean().round(4).alias("accepted_share")).sort("grp")
    print(f"=== {label}")
    print(dist)
    print(ev)
    print(nc)


for arg in sys.argv[1:]:
    path, _, label = arg.partition("|")
    profile(path, label or path)
