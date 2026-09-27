"""Would teaching the normaliser SNC / EI (legal forms it does not know) recover French matches?
US/India analogue with labels: S1 without a legal form, candidate = same core name + an added legal form ("" -> ltd).
France: S1 without snc/ei, candidate core = S1 core + snc/ei only, by address agreement; score band and selection."""
import sys

import polars as pl

sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
import pipeline as pp  # noqa: E402
from normalize import LEGAL_TOKENS  # noqa: E402

R = pp.RUNS_DIR
NEW = ["snc", "ei"]
drop = lambda col, toks: pl.col(col).fill_null("").str.split(" ").list.eval(pl.element().filter(~pl.element().is_in(toks))).list.join(" ")  # noqa: E731
has = lambda col, toks: pl.col(col).fill_null("").str.split(" ").list.eval(pl.element().is_in(toks)).list.any()  # noqa: E731

# US/India analogue: S1 has no legal form, candidate adds one, cores identical
vs = pl.read_parquet(sorted(R.glob("*-v10"))[-1] / "val_scored.parquet").select("s1_idx", "cand_idx", "label", "p").filter(pl.col("cand_idx") >= 0)
ft = (pl.scan_parquet(sorted((sorted(R.glob("*-v10src"))[-1] / "train_feats").glob("part-*.parquet"))).filter(pl.col("is_val"))
      .select("s1_idx", "cand_idx", "ad_tset", "hn_eq").collect())
n1 = pp.scan_norm("train", "s1").select(pl.col("idx").alias("s1_idx"), pl.col("name_full").alias("f1"), pl.col("name_core").alias("c1")).collect()
n2 = pp.scan_norm("train", "pool").select(pl.col("idx").alias("cand_idx"), pl.col("name_full").alias("f2"), pl.col("name_core").alias("c2")).collect()
v = vs.join(ft, on=["s1_idx", "cand_idx"]).join(n1, on="s1_idx").join(n2, on="cand_idx")
L = sorted(LEGAL_TOKENS)
add = v.filter(~has("f1", L) & has("f2", L) & (pl.col("c1") == pl.col("c2")))
for name, g in (("added legal form, same core", add), ("  + address tset>=90 & same house no", add.filter((pl.col("ad_tset") >= 90) & (pl.col("hn_eq") == 1)))):
    print(f"[val] {name}: {g.height:,} pairs, match rate {g['label'].mean():.3f}, mean p {g['p'].mean():.3f}, p<0.5 {(g['p'] < 0.5).mean():.3f}")
del vs, ft, n1, n2, v

# France: candidate = S1 core + snc/ei
fr = pp.scan_norm("test", "s1").filter(pl.col("country_n") == "france").select(pl.col("idx").alias("s1_idx"), pl.col("name_full").alias("f1"), pl.col("name_core").alias("c1")).collect()
n2 = pp.scan_norm("test", "pool").filter(pl.col("country_n") == "france").select(pl.col("idx").alias("cand_idx"), pl.col("name_full").alias("f2"), pl.col("name_core").alias("c2")).collect()
tf = (pl.scan_parquet(sorted((sorted(R.glob("*-v10testsrc"))[-1] / "test_feats").glob("part-*.parquet"))).select("s1_idx", "cand_idx", "ad_tset", "hn_eq")
      .join(fr.select("s1_idx").lazy(), on="s1_idx").collect())
sc = pl.read_parquet(sorted(R.glob("*-v10sfr_dense"))[-1] / "pred" / "scored-*.parquet").select("s1_idx", "cand_idx", "p").join(fr, on="s1_idx").join(n2, on="cand_idx")
ex = pp.exclusive(sc.filter(pl.col("p") >= 0.02)).join(tf, on=["s1_idx", "cand_idx"], how="left")
g = ex.filter(has("f2", NEW) & ~has("f1", NEW)).with_columns(drop("c2", NEW).alias("c2x"), drop("c1", NEW).alias("c1x"))
same = g.filter(pl.col("c1x") == pl.col("c2x"))
s1_nolegal = same.filter(~has("f1", L))
print(f"[France] candidate adds snc/ei: {g.height:,} pairs; core identical once snc/ei dropped: {same.height:,} "
      f"(S1 without any legal form: {s1_nolegal.height:,}; with another legal form = conflict: {same.height - s1_nolegal.height:,})")
for name, h in (("S1 no legal form, same core", s1_nolegal),
                ("  + address tset>=90 & same house no", s1_nolegal.filter((pl.col("ad_tset") >= 90) & (pl.col("hn_eq") == 1)))):
    print(f"[France] {name}: {h.height:,} pairs, mean p {h['p'].mean() if h.height else 0:.3f}, selected (p>=0.92) {(h['p'] >= 0.92).sum():,}, "
          f"S1 without any selected match {h.join(ex.filter(pl.col('p') >= 0.92).select('s1_idx').unique(), on='s1_idx', how='anti').height:,}")
