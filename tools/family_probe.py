"""Do unmatched look-alikes come in families? For each (S1, candidate c): does another candidate d of the same S1
agree with c (name and address token-set >= 90) while c's address is clearly further from the S1 than from d?
Such a family = copies of a different business. Compared on labelled val (by label and p band) and on test (by band).
"""
import sys

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
import pipeline as pp  # noqa: E402

R = pp.RUNS_DIR


def family(pairs: pl.DataFrame, split: str) -> pl.DataFrame:
    s1 = pp.scan_norm(split, "s1").select(pl.col("idx").alias("s1_idx"), pl.col("name_full").alias("n1"), pl.col("addr").alias("a1"), "country_n")
    pool = pp.scan_norm(split, "pool").select(pl.col("idx").alias("cand_idx"), pl.col("name_full").alias("n2"), pl.col("addr").alias("a2"))
    p = (pairs.lazy().join(s1, on="s1_idx").join(pool, on="cand_idx").collect()
         .with_columns(pl.col("n1", "a1", "n2", "a2").fill_null("")))
    p = p.with_columns(pl.Series("as1", cpdist(p["a1"].to_list(), p["a2"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)))
    cc = p.select("s1_idx", "cand_idx", "n2", "a2", "as1")
    x = cc.join(cc.rename({"cand_idx": "d", "n2": "nd", "a2": "ad", "as1": "as1d"}), on="s1_idx").filter(pl.col("cand_idx") != pl.col("d"))
    x = x.with_columns(pl.Series("nn", cpdist(x["n2"].to_list(), x["nd"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)),
                       pl.Series("aa", cpdist(x["a2"].to_list(), x["ad"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)))
    fam = x.group_by("s1_idx", "cand_idx").agg(
        ((pl.col("nn") >= 90) & (pl.col("aa") >= 90) & (pl.col("a2") != "")).sum().alias("n_sib"),
        ((pl.col("nn") >= 90) & (pl.col("aa") >= 90) & (pl.col("a2") != "") & (pl.col("aa") >= pl.col("as1") + 10)
         & (pl.col("as1d") < 90)).sum().alias("n_fam_off"))
    return p.join(fam, on=["s1_idx", "cand_idx"], how="left").with_columns(pl.col("n_sib", "n_fam_off").fill_null(0))


band = pl.col("p").cut([0.1, 0.5, 0.9], labels=["<0.1", "0.1-0.5", "0.5-0.9", ">=0.9"])
va = pl.read_parquet(sorted(R.glob("*-v10_ce"))[-1] / "val_scored_ce.parquet")
vt = pl.read_parquet(R / "20260926-141000-v9" / "val_truth.parquet").select("s1_idx")
va = va.join(vt, on="s1_idx")  # val S1 only
fv = family(va, "train")
print("VAL: share of candidates with an off-S1 family (>=1 sibling agreeing with c but not with the S1)")
print(fv.group_by(band.alias("band"), "label").agg(pl.len(), (pl.col("n_fam_off") >= 1).mean().round(4).alias("fam_off"),
                                                   (pl.col("n_sib") >= 1).mean().round(4).alias("any_sib")).sort("band", "label"))
te = pl.read_parquet(sorted(R.glob("*-v10_ce"))[-1] / "pred" / "scored-*.parquet", columns=["s1_idx", "cand_idx", "p"])
ids = te["s1_idx"].unique().sample(60000, seed=1)
ft = family(te.filter(pl.col("s1_idx").is_in(ids.implode())), "test")
print("TEST (60k S1 sample): same share by band and country")
print(ft.group_by("country_n", band.alias("band")).agg(pl.len(), (pl.col("n_fam_off") >= 1).mean().round(4).alias("fam_off"),
                                                      (pl.col("n_sib") >= 1).mean().round(4).alias("any_sib")).sort("country_n", "band"))
print("VAL by country")
print(fv.join(pp.scan_norm("train", "s1").select(pl.col("idx").alias("s1_idx")).collect(), on="s1_idx").group_by("country_n", band.alias("band"))
      .agg(pl.len(), (pl.col("n_fam_off") >= 1).mean().round(4).alias("fam_off")).sort("country_n", "band"))
