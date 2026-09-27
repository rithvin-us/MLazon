"""Departement vs region interchange in French addresses (e.g. "Gironde" vs "Nouvelle-Aquitaine"). Aggregates only.
1. How French raw addresses end (last comma component), S1 and pool.
2. Every French candidate pair (all pairs, incl. p < 0.02): S1 admin form x candidate admin form, with the final
   model's p; strong-evidence pairs (same core name, same house number) where the forms differ vs agree."""
import sys

import polars as pl

sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
import pipeline as pp  # noqa: E402

R = pp.RUNS_DIR
C = pp.CACHE_DIR
DEPT = {"gironde": "naq", "loire atlantique": "pdl", "nord": "hdf", "pas de calais": "hdf"}
REG = {"naq": "naq", "pdl": "pdl", "hdf": "hdf"}


def last_comp(kind, raw):
    ids = pp.scan_norm("test", kind).filter(pl.col("country_n") == "france").select("entity_id")
    lf = pl.scan_parquet(raw).join(ids, on="entity_id").select(
        pl.col("business_address").fill_null("").str.split(",").list.last().str.strip_chars().str.to_lowercase().alias("last"),
        pl.col("business_address").fill_null("").str.split(",").list.len().alias("ncomp"))
    top = lf.group_by("last").agg(pl.len().alias("n")).sort("n", descending=True).head(14).collect()
    nc = lf.group_by("ncomp").agg(pl.len().alias("n")).sort("ncomp").collect()
    print(f"[{kind}] last address component, top: " + ", ".join(f"'{a}' {b:,}" for a, b in top.iter_rows()))
    print(f"[{kind}] comma components: " + ", ".join(f"{a}: {b:,}" for a, b in nc.iter_rows()))


last_comp("s1", C / "raw_test_s1.parquet")
last_comp("pool", [C / "raw_test_s2.parquet", C / "raw_test_s3.parquet"])


def admin(col):
    d = "|".join(DEPT)
    return (pl.when(pl.col(col).str.contains(rf"\b({d})\b")).then(pl.lit("dept"))
            .when(pl.col(col).str.contains(r"\b(naq|pdl|hdf)\b")).then(pl.lit("region"))
            .otherwise(pl.lit("none")))


def code(col):  # region code the address points to, from either form
    e = pl.lit(None, pl.String)
    for k, v in {**DEPT, **REG}.items():
        e = pl.when(pl.col(col).str.contains(rf"\b{k}\b")).then(pl.lit(v)).otherwise(e)
    return e


fr = pp.scan_norm("test", "s1").filter(pl.col("country_n") == "france").select(pl.col("idx").alias("s1_idx"), pl.col("addr").alias("a1"))
po = pp.scan_norm("test", "pool").filter(pl.col("country_n") == "france").select(pl.col("idx").alias("cand_idx"), pl.col("addr").alias("a2"))
ft = (pl.scan_parquet(sorted((sorted(R.glob("*-v10testsrc"))[-1] / "test_feats").glob("part-*.parquet")))
      .select("s1_idx", "cand_idx", "nc_tset", "hn_eq", "ad_tset").join(fr, on="s1_idx").join(po, on="cand_idx"))
sc = pl.scan_parquet(sorted(R.glob("*-v10sfr_bge2_dense"))[-1] / "pred" / "scored-*.parquet").select("s1_idx", "cand_idx", "p")
f = (ft.join(sc, on=["s1_idx", "cand_idx"], how="left").with_columns(pl.col("p").fill_null(0.0))
     .with_columns(admin("a1").alias("f1"), admin("a2").alias("f2"), code("a1").alias("c1"), code("a2").alias("c2")).collect())
print(f"[pairs] French candidate pairs {f.height:,}")
print(f.group_by("f1", "f2").agg(pl.len().alias("n"), pl.col("p").mean().round(3).alias("mean_p"), (pl.col("p") >= 0.865).mean().round(3).alias("sel"),
                                 pl.col("ad_tset").mean().round(1).alias("ad_tset")).sort("n", descending=True))
g = f.filter((pl.col("nc_tset") == 100) & (pl.col("hn_eq") == 1))
g = g.with_columns(pl.when(pl.col("f1") == pl.col("f2")).then(pl.lit("same form")).when((pl.col("f1") == "none") | (pl.col("f2") == "none")).then(pl.lit("one side none"))
                   .when(pl.col("c1") == pl.col("c2")).then(pl.lit("dept<->its region")).otherwise(pl.lit("different region")).alias("k"))
print("[strong evidence: same core name + same house number] by admin agreement")
print(g.group_by("k").agg(pl.len().alias("n"), pl.col("p").mean().round(4).alias("mean_p"), (pl.col("p") < 0.865).mean().round(4).alias("rejected"),
                          pl.col("ad_tset").mean().round(1).alias("ad_tset")).sort("n", descending=True))
