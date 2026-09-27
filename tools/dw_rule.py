"""Label-free "distractor word" rule. For every country, per extra word w (candidate core name = S1 core name + w only),
s(w) = share of such pairs with the same house number, computed WITHOUT labels. Labelled US/India: noise words have
s >= 0.32 (match | same hn ~0.97), distractor words s <= 0.22 (match 0 even with the same hn). Rule: words with >= MIN_N
pairs and s(w) < S_MAX are distractor words -> their one-extra-word pairs are rejected.
Checks: (1) labelled val (US, India): the rule's flagged words and, among pairs the model accepts (p >= 0.7), how many
the rule would flip and their true match rate (safety: flips must be ~all false);
(2) France: flagged words, pairs accepted in the final file that the rule would reject."""
import sys

import polars as pl

sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
import pipeline as pp  # noqa: E402

R = pp.RUNS_DIR
MIN_N, S_MAX = 200, 0.25
words = lambda c: pl.col(c).fill_null("").str.split(" ")  # noqa: E731


def one_extra(df):
    return (df.with_columns(words("n2").list.set_difference(words("n1")).alias("ex"), words("n1").list.set_difference(words("n2")).alias("miss"))
            .filter((pl.col("ex").list.len() == 1) & (pl.col("miss").list.len() == 0)).with_columns(pl.col("ex").list.first().alias("word")))


def flagged(pairs):
    s = pairs.group_by("word").agg(pl.len().alias("n"), (pl.col("hn_eq") == 1).mean().alias("s"))
    return s.filter((pl.col("n") >= MIN_N) & (pl.col("s") < S_MAX))


# (1) labelled val, per country, label-free flagging
cm = pp.scan_norm("train", "s1").select(pl.col("idx").alias("s1_idx"), "country_n", pl.col("name_core").alias("n1")).collect()
pn = pp.scan_norm("train", "pool").select(pl.col("idx").alias("cand_idx"), pl.col("name_core").alias("n2")).collect()
vs = pl.read_parquet(sorted(R.glob("*-v10"))[-1] / "val_scored.parquet").select("s1_idx", "cand_idx", "label", "p").filter(pl.col("cand_idx") >= 0)
vf = pl.scan_parquet(sorted((sorted(R.glob("*-v10src"))[-1] / "train_feats").glob("part-*.parquet"))).filter(pl.col("is_val")).select("s1_idx", "cand_idx", "hn_eq").collect()
v = one_extra(vs.join(vf, on=["s1_idx", "cand_idx"]).join(cm, on="s1_idx").join(pn, on="cand_idx"))
del vs, vf, pn
for c in ("us", "india"):
    vc = v.filter(pl.col("country_n") == c)
    fw = flagged(vc)
    hit = vc.join(fw.select("word"), on="word")
    acc = hit.filter(pl.col("p") >= 0.7)
    print(f"[val {c}] flagged words {fw.height} (e.g. {', '.join(fw.sort('n', descending=True)['word'].head(8).to_list())}); "
          f"their pairs {hit.height:,}, true {hit['label'].mean():.4f}; accepted by the model {acc.height:,} -> would flip, of which true {acc['label'].sum():,}")
    nf = vc.join(fw.select("word"), on="word", how="anti").group_by("word").agg(pl.len().alias("n"), (pl.col("hn_eq") == 1).mean().alias("s"), pl.col("label").mean().alias("m")).filter(pl.col("n") >= MIN_N)
    print(f"[val {c}] NOT flagged words {nf.height}: mean match rate {nf['m'].mean():.3f}, lowest s {nf['s'].min():.3f}; "
          f"flagged words' match rate max {vc.join(fw.select('word'), on='word').group_by('word').agg(pl.col('label').mean())['label'].max():.3f}")
del v, cm
# (2) France
fr = pp.scan_norm("test", "s1").filter(pl.col("country_n") == "france").select(pl.col("idx").alias("s1_idx"), pl.col("name_core").alias("n1"), pl.col("entity_id").alias("sid")).collect()
po = pp.scan_norm("test", "pool").filter(pl.col("country_n") == "france").select(pl.col("idx").alias("cand_idx"), pl.col("entity_id").alias("cid"), pl.col("name_core").alias("n2")).collect()
tf = (pl.scan_parquet(sorted((sorted(R.glob("*-v10testsrc"))[-1] / "test_feats").glob("part-*.parquet"))).select("s1_idx", "cand_idx", "hn_eq")
      .join(fr.lazy().select("s1_idx"), on="s1_idx").collect())
t = one_extra(tf.join(fr, on="s1_idx").join(po, on="cand_idx"))
fw = flagged(t)
owner = (pl.read_csv(pp.ROOT / "output" / "SUBMIT_THIS" / "matching_results.tsv", separator="\t", quote_char=None, infer_schema=False)
         .select(pl.col("source1_entity_id").alias("sid"), pl.col("matched_entity_ids").fill_null("").str.split(",").alias("cid")).explode("cid").filter(pl.col("cid") != "")
         .with_columns(pl.lit(True).alias("acc")))
hit = t.join(fw.select("word"), on="word").join(owner, on=["sid", "cid"], how="left").with_columns(pl.col("acc").fill_null(False))
with pl.Config(tbl_rows=40):
    print(f"[France] flagged words {fw.height}; their pairs {hit.height:,}; accepted in the final file {hit['acc'].sum():,} (rule would reject these)")
    print(hit.group_by("word").agg(pl.len().alias("n"), (pl.col("hn_eq") == 1).mean().round(3).alias("s"), pl.col("acc").sum().alias("accepted")).sort("accepted", descending=True).head(20))
    nf = t.join(fw.select("word"), on="word", how="anti").group_by("word").agg(pl.len().alias("n"), (pl.col("hn_eq") == 1).mean().round(3).alias("s")).filter(pl.col("n") >= MIN_N)
    print(f"[France] NOT flagged words {nf.height}: " + ", ".join(f"{a} {b}" for a, b in nf.sort("n", descending=True).head(15).select("word", "s").iter_rows()))
