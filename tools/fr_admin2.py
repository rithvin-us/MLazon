"""Controlled check: French pairs whose core names are identical and whose addresses are identical once the admin
token (departement name or region code) is removed. Compare the final model's p when the candidate uses the same
admin form as the S1 (region) vs the departement of that region vs no admin part. Any gap = the cost of the
unnormalised departement token."""
import sys

import polars as pl

sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
import pipeline as pp  # noqa: E402

R = pp.RUNS_DIR
DEPT = {"gironde": "naq", "loire atlantique": "pdl", "nord": "hdf", "pas de calais": "hdf"}
ADMIN_RE = r"\b(gironde|loire atlantique|nord|pas de calais|naq|pdl|hdf)\b"


def code(col):
    e = pl.lit("none")
    for k, v in {**DEPT, "naq": "naq", "pdl": "pdl", "hdf": "hdf"}.items():
        e = pl.when(pl.col(col).str.contains(rf"\b{k}\b")).then(pl.lit(v)).otherwise(e)
    return e


def form(col):
    return (pl.when(pl.col(col).str.contains(r"\b(gironde|loire atlantique|nord|pas de calais)\b")).then(pl.lit("dept"))
            .when(pl.col(col).str.contains(r"\b(naq|pdl|hdf)\b")).then(pl.lit("region")).otherwise(pl.lit("none")))


strip = lambda col: pl.col(col).str.replace_all(ADMIN_RE, "").str.replace_all(r"\s+", " ").str.strip_chars()  # noqa: E731
fr = pp.scan_norm("test", "s1").filter(pl.col("country_n") == "france").select(pl.col("idx").alias("s1_idx"), pl.col("addr").alias("a1"), pl.col("name_core").alias("n1"))
po = pp.scan_norm("test", "pool").filter(pl.col("country_n") == "france").select(pl.col("idx").alias("cand_idx"), pl.col("addr").alias("a2"), pl.col("name_core").alias("n2"))
ft = (pl.scan_parquet(sorted((sorted(R.glob("*-v10testsrc"))[-1] / "test_feats").glob("part-*.parquet")))
      .select("s1_idx", "cand_idx", "ad_tset", "ad_core_tset", "baddr").join(fr, on="s1_idx").join(po, on="cand_idx"))
sc = pl.scan_parquet(sorted(R.glob("*-v10sfr_bge2_dense"))[-1] / "pred" / "scored-*.parquet").select("s1_idx", "cand_idx", "p")
f = (ft.filter(pl.col("n1") == pl.col("n2"))
     .with_columns(strip("a1").alias("x1"), strip("a2").alias("x2"), form("a1").alias("f1"), form("a2").alias("f2"), code("a1").alias("c1"), code("a2").alias("c2"))
     .filter((pl.col("x1") == pl.col("x2")) & (pl.col("x1") != ""))
     .join(sc, on=["s1_idx", "cand_idx"], how="left").with_columns(pl.col("p").fill_null(0.0)).collect())
f = f.with_columns(pl.when(pl.col("f2") == "none").then(pl.lit("cand: no admin part"))
                   .when(pl.col("c1") != pl.col("c2")).then(pl.lit("cand: other region"))
                   .when(pl.col("f1") == pl.col("f2")).then(pl.lit("cand: same form")).otherwise(pl.lit("cand: departement of S1's region")).alias("k"))
print(f"identical core name + identical address apart from the admin token: {f.height:,} pairs (S1 forms: "
      + ", ".join(f"{a} {b:,}" for a, b in f.group_by("f1").agg(pl.len()).sort("f1").iter_rows()) + ")")
print(f.filter(pl.col("f1") == "region").group_by("k").agg(
    pl.len().alias("n"), pl.col("p").mean().round(4).alias("mean_p"), (pl.col("p") < 0.865).mean().round(4).alias("rejected"),
    pl.col("ad_tset").mean().round(1).alias("ad_tset"), pl.col("ad_core_tset").mean().round(1).alias("ad_core"), pl.col("baddr").mean().round(3).alias("baddr")).sort("n", descending=True))
