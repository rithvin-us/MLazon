"""Exact-ish name + empty candidate address: how often a true match, by name crowding; and what the model gives."""
import sys
import polars as pl
sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
import pipeline as pp
R = pp.RUNS_DIR
src = sorted(R.glob("*-v10src"))[-1]
ce = sorted(R.glob("*-v10_ce"))[-1]
f = (pl.scan_parquet(sorted((src / "train_feats").glob("part-*.parquet"))).filter(pl.col("is_val"))
     .select("s1_idx", "cand_idx", "label", "nc_tset", "nf_tset", "ncc_ratio", "len_addr_c", "len_addr_s1", "s1_same_name", "c_name_cnt",
             "s1_name_cnt", "n_cands", "rr_rank", "n_idf_jacc").collect())
p = pl.read_parquet(ce / "val_scored_ce.parquet").select("s1_idx", "cand_idx", "p")
f = f.join(p, on=["s1_idx", "cand_idx"], how="left").with_columns(pl.col("p").fill_null(0.0))
e = f.filter((pl.col("len_addr_c") == 0) & (pl.col("len_addr_s1") > 0))
print("val pairs with EMPTY candidate address:", e.height, "positives", e["label"].sum())
e = e.with_columns(pl.when(pl.col("nc_tset") >= 100).then(pl.lit("core=100")).when(pl.col("nc_tset") >= 90).then(pl.lit("core90-99"))
                   .when(pl.col("nc_tset") >= 75).then(pl.lit("core75-89")).otherwise(pl.lit("core<75")).alias("name"),
                   pl.when(pl.col("s1_same_name") <= 1).then(pl.lit("unique S1 name")).otherwise(pl.lit("shared S1 name")).alias("crowd"))
print(e.group_by("name", "crowd").agg(pl.len(), pl.col("label").mean().round(3).alias("match_rate"), pl.col("p").mean().round(3).alias("mean_p"),
                                      ((pl.col("p") >= 0.6) & (pl.col("label") == 1)).sum().alias("TP@0.6"), ((pl.col("p") < 0.6) & (pl.col("label") == 1)).sum().alias("FN@0.6"),
                                      ((pl.col("p") >= 0.6) & (pl.col("label") == 0)).sum().alias("FP@0.6")).sort("name", "crowd"))
# among exact-name empty-address: per S1, how many such candidates? (several copies of same name)
x = e.filter(pl.col("name") == "core=100")
x = x.with_columns(pl.len().over("s1_idx").alias("n_same_empty"))
print(x.group_by("crowd", pl.col("n_same_empty").clip(1, 4).alias("k")).agg(pl.len(), pl.col("label").mean().round(3).alias("match_rate"), pl.col("p").mean().round(3)).sort("crowd", "k"))
