"""Label-free blocking recall estimate per country: 'easy copies' = pool records with the S1's exact name_core and
address token-set >= 80. Share of them present in the S1's candidate list. Test (all countries) and val (labelled check)."""
import sys
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
import pipeline as pp
R = pp.RUNS_DIR


def easy(split, s1_ids, cands):
    s1 = pp.scan_norm(split, "s1").select(pl.col("idx").alias("s1_idx"), "name_core", "addr", "country_n").collect().join(s1_ids, on="s1_idx")
    pool = pp.scan_norm(split, "pool").select(pl.col("idx").alias("cand_idx"), "name_core", pl.col("addr").alias("addr_p"), "country_n").collect()
    j = s1.join(pool.filter(pl.col("addr_p").fill_null("") != ""), on=["country_n", "name_core"])
    j = j.with_columns(pl.Series("as", cpdist(j["addr"].fill_null("").to_list(), j["addr_p"].fill_null("").to_list(), scorer=fuzz.token_set_ratio, workers=-1)))
    j = j.filter(pl.col("as") >= 80).join(cands.with_columns(pl.lit(1).alias("in_c")), on=["s1_idx", "cand_idx"], how="left").with_columns(pl.col("in_c").fill_null(0))
    return j


# test: sample 30k S1 per country
cm = pp.scan_norm("test", "s1").select(pl.col("idx").alias("s1_idx"), "country_n").collect()
ids = pl.concat([cm.filter(pl.col("country_n") == c).sample(30000, seed=1) for c in ("france", "india", "us")]).select("s1_idx")
tc = pl.scan_parquet(sorted((sorted(R.glob("*-v10testsrc"))[-1] / "test_feats").glob("part-*.parquet"))).select("s1_idx", "cand_idx").collect().join(ids, on="s1_idx")
t = easy("test", ids, tc)
print("TEST easy-copy recall by country")
print(t.group_by("country_n").agg(pl.len().alias("easy_pairs"), pl.col("in_c").mean().round(4).alias("in_candidates")).sort("country_n"))
nc = tc.join(cm, on="s1_idx").group_by("s1_idx", "country_n").len()
print(nc.group_by("country_n").agg((pl.col("len") >= 40).mean().round(4).alias("at_cap40"), pl.col("len").mean().round(2).alias("mean_cands")).sort("country_n"))
# val (labelled): same estimate + true labels
vt = pl.read_parquet(R / "20260926-141000-v9" / "val_truth.parquet").select("s1_idx")
vc = pl.scan_parquet(sorted((sorted(R.glob("*-v10src"))[-1] / "train_feats").glob("part-*.parquet"))).filter(pl.col("is_val")).select("s1_idx", "cand_idx").collect()
v = easy("train", vt, vc)
print("VAL easy-copy recall by country")
print(v.group_by("country_n").agg(pl.len().alias("easy_pairs"), pl.col("in_c").mean().round(4).alias("in_candidates")).sort("country_n"))
