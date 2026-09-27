"""Label-free test of the house-number hypothesis. Copies per entity follow the same distribution in every country
(the match-count distributions agree). If true French copies with a changed house number are rejected, French S1
that HAVE such candidates must end up with fewer accepted matches than comparable US/India S1 (whose predictions are
~98.5% right). Final file, test set, per group:
  S1 with >= 1 "hd" candidate (numbers differ, same core name, core address >= 90) vs S1 without; mean accepted k,
  and how many hd candidates they have."""
import sys

import polars as pl

sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
import pipeline as pp  # noqa: E402

R = pp.RUNS_DIR
s1 = pp.scan_norm("test", "s1").select(pl.col("idx").alias("s1_idx"), pl.col("entity_id").alias("source1_entity_id"), "country_n").collect()
hd = (pl.scan_parquet(sorted((sorted(R.glob("*-v10testsrc"))[-1] / "test_feats").glob("part-*.parquet")))
      .filter((pl.col("hn_both") == 1) & (pl.col("hn_eq") == 0) & (pl.col("num_jacc") == 0) & (pl.col("nc_tset") == 100) & (pl.col("ad_core_tset") >= 90))
      .group_by("s1_idx").agg(pl.len().alias("n_hd")).collect())
m = pl.read_csv(pp.ROOT / "output" / "SUBMIT_THIS" / "matching_results.tsv", separator="\t", quote_char=None, infer_schema=False).select(
    "source1_entity_id", pl.col("matched_entity_ids").fill_null("").str.split(",").list.eval(pl.element().filter(pl.element() != "")).list.len().alias("k"))
d = s1.join(m, on="source1_entity_id").join(hd, on="s1_idx", how="left").with_columns(pl.col("n_hd").fill_null(0))
d = d.with_columns(pl.when(pl.col("n_hd") == 0).then(pl.lit("0 hd cand")).when(pl.col("n_hd") == 1).then(pl.lit("1 hd cand")).otherwise(pl.lit("2+ hd cand")).alias("g"))
with pl.Config(tbl_rows=20):
    print(d.group_by("country_n", "g").agg(pl.len().alias("S1"), pl.col("k").mean().round(3).alias("mean_k"), (pl.col("k") == 0).mean().round(4).alias("k=0"),
                                            pl.col("n_hd").mean().round(2).alias("mean_hd")).sort("country_n", "g"))
