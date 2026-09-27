"""Empty-address exact-name candidates: match rate by whether the S1's candidate list holds a same-name record at a
DIFFERENT address (a twin business) and how many same-name empty records there are."""
import sys
import polars as pl
sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
import pipeline as pp
R = pp.RUNS_DIR
src = sorted(R.glob("*-v10src"))[-1]
ce = sorted(R.glob("*-v10_ce"))[-1]
f = (pl.scan_parquet(sorted((src / "train_feats").glob("part-*.parquet"))).filter(pl.col("is_val"))
     .select("s1_idx", "cand_idx", "label", "nc_tset", "ad_tset", "len_addr_c", "len_addr_s1", "hn_eq", "hn_both", "s1_same_name").collect())
f = f.join(pl.read_parquet(ce / "val_scored_ce.parquet").select("s1_idx", "cand_idx", "p"), on=["s1_idx", "cand_idx"], how="left").with_columns(pl.col("p").fill_null(0.0))
same = pl.col("nc_tset") >= 95
f = f.with_columns(
    (same & (pl.col("len_addr_c") > 0) & (pl.col("ad_tset") >= 85)).sum().over("s1_idx").alias("sn_match"),
    (same & (pl.col("len_addr_c") > 0) & (pl.col("ad_tset") < 70)).sum().over("s1_idx").alias("sn_other"),
    (same & (pl.col("len_addr_c") == 0)).sum().over("s1_idx").alias("sn_empty"))
e = f.filter((pl.col("nc_tset") >= 100) & (pl.col("len_addr_c") == 0) & (pl.col("len_addr_s1") > 0))
print("exact-name empty-address candidates:", e.height, "match rate", round(e["label"].mean(), 3))
print(e.group_by(pl.col("sn_other").clip(0, 2).alias("twins"), pl.col("sn_empty").clip(1, 4).alias("empties"))
      .agg(pl.len(), pl.col("label").mean().round(3).alias("match_rate"), pl.col("p").mean().round(3).alias("mean_p"),
           pl.col("sn_match").mean().round(2).alias("addr_matched_copies")).sort("twins", "empties"))

print("\n== twins=0, empties=1: by number of address-matched same-name copies")
g = e.filter((pl.col("sn_other") == 0) & (pl.col("sn_empty") == 1))
print(g.group_by(pl.col("sn_match").clip(0, 6).alias("confirmed")).agg(pl.len(), pl.col("label").mean().round(3).alias("rate"), pl.col("p").mean().round(3).alias("mean_p")).sort("confirmed"))
conf = f.filter(pl.col("p") >= 0.9).group_by("s1_idx").len().rename({"len": "n_conf"})
g2 = e.join(conf, on="s1_idx", how="left").with_columns(pl.col("n_conf").fill_null(0))
print("\n== all exact-name empty-address: by number of confident (p>=0.9) candidates of the S1")
print(g2.group_by(pl.col("n_conf").clip(0, 7).alias("n_conf")).agg(pl.len(), pl.col("label").mean().round(3).alias("rate"), pl.col("p").mean().round(3).alias("mean_p")).sort("n_conf"))
