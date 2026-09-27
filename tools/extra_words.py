"""Candidate name = S1 core name + exactly one extra word (the "name plus one word" distractor type, or noise on a
true copy). Per extra word: pairs, share accepted by the final file. France vs labelled US/India val (match rate and
the model's mean p for the same construction), to see whether French distractor words are accepted too often."""
import sys

import polars as pl

sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
import pipeline as pp  # noqa: E402

R = pp.RUNS_DIR
words = lambda c: pl.col(c).fill_null("").str.split(" ")  # noqa: E731


def one_extra(df):
    return (df.with_columns(words("n2").list.set_difference(words("n1")).alias("ex"), words("n1").list.set_difference(words("n2")).alias("miss"))
            .filter((pl.col("ex").list.len() == 1) & (pl.col("miss").list.len() == 0)).with_columns(pl.col("ex").list.first().alias("word")))


fr = pp.scan_norm("test", "s1").filter(pl.col("country_n") == "france").select(pl.col("idx").alias("s1_idx"), pl.col("name_core").alias("n1")).collect()
po = pp.scan_norm("test", "pool").filter(pl.col("country_n") == "france").select(pl.col("idx").alias("cand_idx"), pl.col("name_core").alias("n2")).collect()
tf = (pl.scan_parquet(sorted((sorted(R.glob("*-v10testsrc"))[-1] / "test_feats").glob("part-*.parquet"))).select("s1_idx", "cand_idx", "hn_eq")
      .join(fr.lazy().select("s1_idx"), on="s1_idx").collect())
sc = pl.read_parquet(sorted(R.glob("*-v10sfr_bge2_dense"))[-1] / "pred" / "scored-*.parquet").select("s1_idx", "cand_idx", "p").join(fr.select("s1_idx"), on="s1_idx")
t = one_extra(tf.join(fr, on="s1_idx").join(po, on="cand_idx").join(sc, on=["s1_idx", "cand_idx"], how="left").with_columns(pl.col("p").fill_null(0.0)))
print(f"[France] one-extra-word pairs {t.height:,}, accepted {(t['p'] >= 0.865).mean():.3f}; same house number {(t['hn_eq'] == 1).mean():.3f}")
with pl.Config(tbl_rows=25):
    print(t.group_by("word").agg(pl.len().alias("n"), (pl.col("p") >= 0.865).mean().round(3).alias("accepted"), pl.col("p").mean().round(3).alias("mean_p"),
                                 (pl.col("hn_eq") == 1).mean().round(3).alias("same_hn")).sort("n", descending=True).head(22))
del t, tf, sc, po
cm = pp.scan_norm("train", "s1").select(pl.col("idx").alias("s1_idx"), "country_n", pl.col("name_core").alias("n1")).collect()
pn = pp.scan_norm("train", "pool").select(pl.col("idx").alias("cand_idx"), pl.col("name_core").alias("n2")).collect()
vs = pl.read_parquet(sorted(R.glob("*-v10"))[-1] / "val_scored.parquet").select("s1_idx", "cand_idx", "label", "p").filter(pl.col("cand_idx") >= 0)
v = one_extra(vs.join(cm, on="s1_idx").join(pn, on="cand_idx"))
print(f"[val] one-extra-word pairs {v.height:,}: match rate {v['label'].mean():.3f}, mean p {v['p'].mean():.3f}")
with pl.Config(tbl_rows=25):
    print(v.group_by("word").agg(pl.len().alias("n"), pl.col("label").mean().round(3).alias("match"), pl.col("p").mean().round(3).alias("mean_p"))
          .sort("n", descending=True).head(16))
