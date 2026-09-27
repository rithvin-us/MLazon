"""Full-pool twin check. For val S1 with exact-name empty-address candidates: among ALL pool records of the country with
the same name_core, how many have an address and how many of those do NOT look like the S1's address (a different
business with the same name)? Match rate of the empty candidates by that count."""
import sys
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
import pipeline as pp
R = pp.RUNS_DIR
src = sorted(R.glob("*-v10src"))[-1]
ce = sorted(R.glob("*-v10_ce"))[-1]
f = (pl.scan_parquet(sorted((src / "train_feats").glob("part-*.parquet"))).filter(pl.col("is_val"))
     .select("s1_idx", "cand_idx", "label", "nc_tset", "len_addr_c", "len_addr_s1").collect())
f = f.join(pl.read_parquet(ce / "val_scored_ce.parquet").select("s1_idx", "cand_idx", "p"), on=["s1_idx", "cand_idx"], how="left").with_columns(pl.col("p").fill_null(0.0))
e = f.filter((pl.col("nc_tset") >= 100) & (pl.col("len_addr_c") == 0) & (pl.col("len_addr_s1") > 0))
s1 = pp.scan_norm("train", "s1").select(pl.col("idx").alias("s1_idx"), "name_core", "addr", "country_n").collect()
pool = pp.scan_norm("train", "pool").select(pl.col("idx").alias("pidx"), "name_core", "addr", "country_n").collect()
S = s1.join(e.select("s1_idx").unique(), on="s1_idx")
cnt = pool.group_by("country_n", "name_core").len()
S = S.join(cnt, on=["country_n", "name_core"], how="left").filter(pl.col("len") <= 300)
J = S.join(pool.filter(pl.col("addr").fill_null("") != ""), on=["country_n", "name_core"], suffix="_p")
J = J.with_columns(pl.Series("as", cpdist(J["addr"].fill_null("").to_list(), J["addr_p"].fill_null("").to_list(), scorer=fuzz.token_set_ratio, workers=-1)))
T = J.group_by("s1_idx").agg((pl.col("as") < 70).sum().alias("pool_twins"), (pl.col("as") >= 85).sum().alias("pool_same_addr"),
                             pl.len().alias("pool_addressed"))
E = pool.filter(pl.col("addr").fill_null("") == "").group_by("country_n", "name_core").len().rename({"len": "pool_empty"})
x = e.join(T, on="s1_idx", how="inner").join(S.select("s1_idx", "country_n", "name_core"), on="s1_idx").join(E, on=["country_n", "name_core"], how="left")
print("exact-name empty-address val candidates with pool stats:", x.height, "match rate", round(x["label"].mean(), 3))
print(x.group_by(pl.col("pool_twins").clip(0, 3).alias("pool_twins"), pl.col("pool_empty").clip(1, 4).alias("pool_empty"))
      .agg(pl.len(), pl.col("label").mean().round(3).alias("rate"), pl.col("p").mean().round(3).alias("mean_p"),
           pl.col("pool_same_addr").mean().round(2)).sort("pool_twins", "pool_empty"))
