"""Legal-form conflicts (both names carry a legal form, none shared: SARL vs SAS, LLC vs INC): labelled val match rate
vs the model's mean p by score band (calibration), and how France's conflicting pairs are scored / selected."""
import sys

import polars as pl

sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
import pipeline as pp  # noqa: E402

R = pp.RUNS_DIR
lab = pl.scan_parquet(sorted((sorted(R.glob("*-v10src"))[-1] / "train_feats").glob("part-*.parquet"))).filter(pl.col("is_val")).select("s1_idx", "cand_idx", "legal_conflict")
vs = pl.read_parquet(sorted(R.glob("*-v10"))[-1] / "val_scored.parquet").select("s1_idx", "cand_idx", "label", "p").filter(pl.col("cand_idx") >= 0)
vs = vs.join(lab.collect(), on=["s1_idx", "cand_idx"])
cut = lambda df: df.with_columns(pl.col("p").cut([0.2, 0.5, 0.8, 0.92, 0.98], left_closed=True).alias("band"))  # noqa: E731
c = cut(vs.filter(pl.col("legal_conflict") == 1))
print(f"[val] conflicting legal forms: {c.height:,} pairs, match rate {c['label'].mean():.3f}; by band (n, match rate, mean p):")
print(c.group_by("band").agg(pl.len().alias("n"), pl.col("label").mean().round(3).alias("match"), pl.col("p").mean().round(3).alias("mean_p")).sort("band"))
fr = pp.scan_norm("test", "s1").filter(pl.col("country_n") == "france").select(pl.col("idx").alias("s1_idx")).collect()
ft = (pl.scan_parquet(sorted((sorted(R.glob("*-v10testsrc"))[-1] / "test_feats").glob("part-*.parquet")))
      .select("s1_idx", "cand_idx", "legal_conflict").join(fr.lazy(), on="s1_idx").collect())
sc = pl.read_parquet(sorted(R.glob("*-v10sfr_dense"))[-1] / "pred" / "scored-*.parquet").select("s1_idx", "cand_idx", "p").join(fr, on="s1_idx")
ex = pp.exclusive(sc.filter(pl.col("p") >= 0.02)).join(ft, on=["s1_idx", "cand_idx"], how="left")
f = cut(ex.filter(pl.col("legal_conflict") == 1))
print(f"[France] conflicting legal forms: {f.height:,} pairs ({f.height / ex.height:.4f} of scored), selected at 0.92: {(f['p'] >= 0.92).sum():,}; by band:")
print(f.group_by("band").agg(pl.len().alias("n"), pl.col("p").mean().round(3).alias("mean_p")).sort("band"))
print(f"[France] share of all pairs by band that conflict: " + ", ".join(
    f"{b} {v:.3f}" for b, v in cut(ex).group_by("band").agg((pl.col("legal_conflict") == 1).mean().alias("v")).sort("band").iter_rows()))
