"""Stage 2 on competition features (the other pipeline's u_margin idea), checked on labelled val (US/India) with the
final stacked scores (v10_bge2: L12 + bge). Per (S1, record): p, best competing S1's p for the record, margin,
competitors >= 0.2, rank among the record's S1s, the S1's best p, rank within the S1, the S1's sum of p, candidates
>= 0.2, gap to the next candidate. XGBoost depth 4, 5-fold out-of-fold by S1 (competitor rows scored by the fold
models' average). Decision family re-tuned on p and on p2 exactly as the pipeline does (tune_decision, exclusivity).
Also per crowding: S1 whose records are contested (the slice France has 3.5x more of)."""
import sys

import numpy as np
import polars as pl
import xgboost as xgb

sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
import pipeline as pp  # noqa: E402

R = pp.RUNS_DIR



FEAT = ["p", "max_other", "margin", "n_other", "rank_rec", "s1_max", "rank_s1", "s1_sum", "s1_n02", "gap_next"]


def feats(df):
    df = df.with_columns(
        pl.col("p").rank("ordinal", descending=True).over("cand_idx").alias("rank_rec"),
        pl.col("p").rank("ordinal", descending=True).over("s1_idx").alias("rank_s1"),
        pl.col("p").max().over("s1_idx").alias("s1_max"), pl.col("p").sum().over("s1_idx").alias("s1_sum"),
        (pl.col("p") >= 0.2).sum().over("s1_idx").alias("s1_n02"),
        ((pl.col("p") >= 0.2).sum().over("cand_idx") - (pl.col("p") >= 0.2).cast(pl.UInt32)).alias("n_other"))
    top2 = df.group_by("cand_idx").agg(pl.col("p").top_k(2).alias("t"))
    df = df.join(top2, on="cand_idx").with_columns(
        pl.when(pl.col("p") >= pl.col("t").list.first()).then(pl.col("t").list.get(1, null_on_oob=True)).otherwise(pl.col("t").list.first()).fill_null(0.0).alias("max_other")).drop("t")
    nxt = df.sort(["s1_idx", "p"], descending=[False, True]).with_columns(pl.col("p").shift(-1).over("s1_idx").fill_null(0.0).alias("p_next"))
    df = nxt.with_columns((pl.col("p") - pl.col("p_next")).alias("gap_next"), (pl.col("p") - pl.col("max_other")).alias("margin")).drop("p_next")
    return df


