"""House numbers that differ: format artefact or French recall hole? Labelled val per country (US, India separately):
share of true matches whose house numbers are both present but differ, the model's recall on them, and the match
rate of such candidates; then the same candidate slice on France test (score distribution, how far apart the numbers
are). Aggregates only."""
import sys

import polars as pl

sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
import pipeline as pp  # noqa: E402

R = pp.RUNS_DIR
cm = pp.scan_norm("train", "s1").select(pl.col("idx").alias("s1_idx"), "country_n").collect()
vs = pl.read_parquet(sorted(R.glob("*-v10"))[-1] / "val_scored.parquet").select("s1_idx", "cand_idx", "label", "p").filter(pl.col("cand_idx") >= 0)
ft = (pl.scan_parquet(sorted((sorted(R.glob("*-v10src"))[-1] / "train_feats").glob("part-*.parquet"))).filter(pl.col("is_val"))
      .select("s1_idx", "cand_idx", "hn_eq", "hn_both", "hn_sim", "nc_tset", "ad_core_tset").collect())
v = vs.join(ft, on=["s1_idx", "cand_idx"]).join(cm, on="s1_idx").with_columns(((pl.col("hn_both") == 1) & (pl.col("hn_eq") == 0)).alias("hd"))
print("[val] per country: true matches with differing house numbers, model recall on them (p>=0.7), candidate match rate")
print(v.group_by("country_n").agg(
    (pl.col("hd") & (pl.col("label") == 1)).sum().alias("true_hd"),
    ((pl.col("hd") & (pl.col("label") == 1)).sum() / (pl.col("label") == 1).sum()).round(4).alias("share_of_true"),
    ((pl.col("hd") & (pl.col("label") == 1) & (pl.col("p") >= 0.7)).sum() / (pl.col("hd") & (pl.col("label") == 1)).sum()).round(3).alias("recall_on_them"),
    (pl.col("label").filter(pl.col("hd")).mean()).round(3).alias("hd_cand_match_rate"),
    ((pl.col("p") >= 0.7) & pl.col("hd")).sum().alias("accepted_hd"),
    (((pl.col("p") >= 0.7) & pl.col("hd")).sum() / (pl.col("p") >= 0.7).sum()).round(4).alias("hd_share_of_accepted")).sort("country_n"))
print("[val] differing house numbers, same core name, core address >= 90, by house-number similarity: n, match rate, mean p")
w = v.filter(pl.col("hd") & (pl.col("nc_tset") == 100) & (pl.col("ad_core_tset") >= 90)).with_columns(pl.col("hn_sim").cut([50, 75, 90], left_closed=True).alias("hn_sim_b"))
print(w.group_by("country_n", "hn_sim_b").agg(pl.len().alias("n"), pl.col("label").mean().round(3).alias("match"), pl.col("p").mean().round(3).alias("mean_p")).sort("country_n", "hn_sim_b"))
del vs, ft, v
fr = pp.scan_norm("test", "s1").filter(pl.col("country_n") == "france").select(pl.col("idx").alias("s1_idx")).collect()
tf = (pl.scan_parquet(sorted((sorted(R.glob("*-v10testsrc"))[-1] / "test_feats").glob("part-*.parquet")))
      .select("s1_idx", "cand_idx", "hn_eq", "hn_both", "hn_sim", "nc_tset", "ad_core_tset").join(fr.lazy(), on="s1_idx").collect())
sc = pl.read_parquet(sorted(R.glob("*-v10sfr_bge2_dense"))[-1] / "pred" / "scored-*.parquet").select("s1_idx", "cand_idx", "p").join(fr, on="s1_idx")
t = tf.join(sc, on=["s1_idx", "cand_idx"], how="left").with_columns(pl.col("p").fill_null(0.0))
t = t.with_columns(((pl.col("hn_both") == 1) & (pl.col("hn_eq") == 0)).alias("hd"))
print(f"[France] candidate pairs {t.height:,}; differing house numbers {t['hd'].sum():,} ({t['hd'].mean():.4f}); accepted (p>=0.865) among them {(t.filter('hd')['p'] >= 0.865).mean():.4f}")
w = t.filter(pl.col("hd") & (pl.col("nc_tset") == 100) & (pl.col("ad_core_tset") >= 90)).with_columns(pl.col("hn_sim").cut([50, 75, 90], left_closed=True).alias("hn_sim_b"))
print("[France] differing house numbers, same core name, core address >= 90, by house-number similarity: n, mean p, accepted")
print(w.group_by("hn_sim_b").agg(pl.len().alias("n"), pl.col("p").mean().round(3).alias("mean_p"), (pl.col("p") >= 0.865).mean().round(3).alias("accepted")).sort("hn_sim_b"))
