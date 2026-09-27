"""Is the house-number signature a valid label-free tell for extra-word pairs? Labelled US/India val: per extra word,
match rate, same-house-number share, match rate when the house number is the same vs different; and whether the
same-hn share of a word predicts its match rate across words (correlation). Then France: the same per-word table on
the test pairs, with the final file's acceptance when the house number is the same."""
import sys

import polars as pl

sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
import pipeline as pp  # noqa: E402

R = pp.RUNS_DIR
words = lambda c: pl.col(c).fill_null("").str.split(" ")  # noqa: E731


def one_extra(df):
    return (df.with_columns(words("n2").list.set_difference(words("n1")).alias("ex"), words("n1").list.set_difference(words("n2")).alias("miss"))
            .filter((pl.col("ex").list.len() == 1) & (pl.col("miss").list.len() == 0)).with_columns(pl.col("ex").list.first().alias("word")))


cm = pp.scan_norm("train", "s1").select(pl.col("idx").alias("s1_idx"), "country_n", pl.col("name_core").alias("n1")).collect()
pn = pp.scan_norm("train", "pool").select(pl.col("idx").alias("cand_idx"), pl.col("name_core").alias("n2")).collect()
vs = pl.read_parquet(sorted(R.glob("*-v10"))[-1] / "val_scored.parquet").select("s1_idx", "cand_idx", "label", "p").filter(pl.col("cand_idx") >= 0)
vf = pl.scan_parquet(sorted((sorted(R.glob("*-v10src"))[-1] / "train_feats").glob("part-*.parquet"))).filter(pl.col("is_val")).select("s1_idx", "cand_idx", "hn_eq").collect()
v = one_extra(vs.join(vf, on=["s1_idx", "cand_idx"]).join(cm, on="s1_idx").join(pn, on="cand_idx"))
del vs, vf, pn
g = v.group_by("word").agg(pl.len().alias("n"), pl.col("label").mean().round(3).alias("match"), (pl.col("hn_eq") == 1).mean().round(3).alias("same_hn"),
                           pl.col("label").filter(pl.col("hn_eq") == 1).mean().round(3).alias("match|same_hn"),
                           pl.col("label").filter(pl.col("hn_eq") == 0).mean().round(3).alias("match|diff_hn"),
                           pl.col("p").filter(pl.col("hn_eq") == 1).mean().round(3).alias("p|same_hn")).filter(pl.col("n") >= 150).sort("n", descending=True)
with pl.Config(tbl_rows=30):
    print("[val] extra word: n, match, same_hn share, match given same / different house number, mean p given same hn")
    print(g.head(26))
print(f"[val] across words (n>=150): corr(same_hn share, match rate) = {g.select(pl.corr('same_hn', 'match')).item():.3f}; "
      f"overall match | same hn {v.filter(pl.col('hn_eq') == 1)['label'].mean():.3f}, | different hn {v.filter(pl.col('hn_eq') == 0)['label'].mean():.3f}")
