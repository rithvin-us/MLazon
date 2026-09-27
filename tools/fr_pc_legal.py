"""Two French-specific gaps suggested from outside, measured before building anything (aggregates only):
A. Postcode prefix (departement / ZIP3): the model only sees exact postcode equality (pc_eq). How many true matches
   have a different postcode with the same 2/3-digit prefix, and how many of those does the model miss? (labelled val)
   France test: pairs with a different postcode but the same departement, by score band.
B. French legal forms missing from normalize.LEGAL (snc, gie, selarl, scp, ...): token counts in French names."""
import sys

import polars as pl

sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
import pipeline as pp  # noqa: E402

R = pp.RUNS_DIR
pc = lambda split, kind, tag: pp.scan_norm(split, kind).select(pl.col("idx").alias(f"{tag}_idx" if tag == "s1" else "cand_idx"), pl.col("postcode").alias(f"pc_{tag}"))  # noqa: E731


def pc_cols(df):
    return df.with_columns(
        both=(pl.col("pc_s1") != "") & (pl.col("pc_c") != ""),
        eq=(pl.col("pc_s1") == pl.col("pc_c")) & (pl.col("pc_s1") != ""),
        pre2=pl.col("pc_s1").str.slice(0, 2) == pl.col("pc_c").str.slice(0, 2),
        pre3=pl.col("pc_s1").str.slice(0, 3) == pl.col("pc_c").str.slice(0, 3))


# A1 labelled validation (stage-1 v10 scores)
vs = pl.read_parquet(sorted(R.glob("*-v10"))[-1] / "val_scored.parquet").select("s1_idx", "cand_idx", "label", "p").filter(pl.col("cand_idx") >= 0)
vs = pc_cols(vs.join(pc("train", "s1", "s1").collect(), on="s1_idx").join(pc("train", "pool", "c").collect(), on="cand_idx"))
t = vs.filter(pl.col("label") == 1)
diff = t.filter(pl.col("both") & ~pl.col("eq"))
print(f"[val] true pairs {t.height:,}: both postcodes {t['both'].mean():.3f}; postcode differs {diff.height / t.height:.4f} "
      f"(same 2-digit prefix {diff['pre2'].mean():.3f}, 3-digit {diff['pre3'].mean():.3f})")
for name, g in (("all true", t), ("pc differs", diff), ("pc differs, same pre3", diff.filter(pl.col("pre3")))):
    print(f"[val]   {name:24s} n {g.height:7,}  missed (p<0.5) {(g['p'] < 0.5).mean():.3f}")
neg = vs.filter((pl.col("label") == 0) & pl.col("both") & ~pl.col("eq") & (pl.col("p") >= 0.2))
print(f"[val] negatives p>=0.2 with differing postcode {neg.height:,}: same pre3 {neg['pre3'].mean():.3f}; "
      f"true pairs p<0.5 with differing postcode and same pre3: {diff.filter(pl.col('pre3') & (pl.col('p') < 0.5)).height:,} "
      f"of {t.height:,} true pairs ({diff.filter(pl.col('pre3') & (pl.col('p') < 0.5)).height / t.height:.5f})")

# A2 France test (round-1 France model scores)
fr = pp.scan_norm("test", "s1").filter(pl.col("country_n") == "france").select(pl.col("idx").alias("s1_idx")).collect()
sc = pl.read_parquet(sorted(R.glob("*-v10sfr_dense"))[-1] / "pred" / "scored-*.parquet").select("s1_idx", "cand_idx", "p").join(fr, on="s1_idx")
sc = pc_cols(sc.join(pc("test", "s1", "s1").collect(), on="s1_idx").join(pc("test", "pool", "c").collect(), on="cand_idx"))
bands = sc.with_columns(pl.col("p").cut([0.2, 0.5, 0.8, 0.92, 0.98], left_closed=True).alias("band"))
print("[France] per score band: pairs, both postcodes, postcode differs, differs but same departement (2 digits)")
print(bands.group_by("band").agg(pl.len().alias("pairs"), pl.col("both").mean().round(3).alias("both"),
                                 (pl.col("both") & ~pl.col("eq")).mean().round(4).alias("differs"),
                                 (pl.col("both") & ~pl.col("eq") & pl.col("pre2")).mean().round(4).alias("differs_same_dept")).sort("band"))

# B legal-form tokens in French names
FORMS = ["sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "gie", "selarl", "selas", "selurl", "scp", "scm", "sca",
         "scop", "scea", "earl", "gaec", "eirl", "ei", "sem", "spa", "srl", "sl", "gmbh", "ste", "co", "ets"]
for kind in ("s1", "pool"):
    nm = pp.scan_norm("test", kind).filter(pl.col("country_n") == "france").select("name_full")
    n = nm.select(pl.len()).collect().item()
    vc = (nm.select(pl.col("name_full").str.split(" ").alias("t")).explode("t").filter(pl.col("t").is_in(FORMS))
          .group_by("t").agg(pl.len().alias("n")).sort("n", descending=True).collect())
    print(f"[France {kind}] {n:,} names; legal-form tokens: " + ", ".join(f"{a} {b:,}" for a, b in vc.iter_rows()))
