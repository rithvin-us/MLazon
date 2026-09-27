"""Near-certain copies with a legal-form conflict: true-match rate on labelled val (US/India) vs model p; French examples."""
import sys
import polars as pl
sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
import pipeline as pp
R = pp.RUNS_DIR
src = sorted(R.glob("*-v10src"))[-1]
cols = ["s1_idx", "cand_idx", "nc_tset", "hn_eq", "ad_core_tset", "legal_conflict", "legal_eq"]
v = pl.scan_parquet(sorted((src / "train_feats").glob("part-*.parquet"))).filter(pl.col("is_val")).select("label", *cols).collect()
v = v.join(pl.read_parquet(sorted(R.glob("*-v10_ce"))[-1] / "val_scored_ce.parquet").select("s1_idx", "cand_idx", "p"), on=["s1_idx", "cand_idx"], how="left").with_columns(pl.col("p").fill_null(0.0))
nc = v.filter((pl.col("nc_tset") >= 100) & (pl.col("hn_eq") == 1) & (pl.col("ad_core_tset") >= 90))
print("VAL (US/India) near-certain copies by legal conflict:")
print(nc.group_by("legal_conflict").agg(pl.len(), pl.col("label").mean().round(4).alias("true_rate"), pl.col("p").mean().round(4).alias("mean_p")).sort("legal_conflict"))
# are there S1 duplicates differing only by legal form? France vs US/India (test S1)
s1 = pp.scan_norm("test", "s1").select("idx", "name_core", "name_full", "addr", "country_n").collect()
dup = s1.group_by("country_n", "name_core", "addr").agg(pl.len().alias("n"), pl.col("name_full").n_unique().alias("variants")).filter(pl.col("n") > 1)
print("\nTEST S1 sharing core name AND address with another S1 (per country, share of S1):")
print(dup.group_by("country_n").agg(pl.col("n").sum().alias("s1_in_groups")).join(s1.group_by("country_n").len(), on="country_n")
      .with_columns((pl.col("s1_in_groups") / pl.col("len")).round(4).alias("share")).sort("country_n"))
vs1 = pp.scan_norm("train", "s1").select("idx", "name_core", "addr", "country_n").collect()
vd = vs1.group_by("country_n", "name_core", "addr").agg(pl.len().alias("n")).filter(pl.col("n") > 1)
print("TRAIN S1 sharing core name AND address with another S1:")
print(vd.group_by("country_n").agg(pl.col("n").sum().alias("s1_in_groups")).join(vs1.group_by("country_n").len(), on="country_n")
      .with_columns((pl.col("s1_in_groups") / pl.col("len")).round(4).alias("share")).sort("country_n"))
