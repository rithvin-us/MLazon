"""France one-extra-word pairs with the SAME house number: is the candidate's core name itself another French S1's core
name (the record then belongs to that S1, rejection is right) or not (then it can only be this S1's copy or an
unmatched distractor)? Per word group: pairs, share whose name is another S1, acceptance in each case, and where the
record went in the final file (to this S1 / to another S1 / to nobody). Same breakdown on labelled US/India val."""
import sys

import polars as pl

sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
import pipeline as pp  # noqa: E402

R = pp.RUNS_DIR
words = lambda c: pl.col(c).fill_null("").str.split(" ")  # noqa: E731
GROUPS = {"noise-like: services fils associes": ["services", "fils", "associes"],
          "club amicale comite ecole sportive amis union maison": ["club", "amicale", "comite", "ecole", "sportive", "amis", "union", "maison"],
          "distractor-like: france groupe developpement": ["france", "groupe", "developpement"],
          "holding distribution international participations": ["holding", "distribution", "international", "participations"]}
gexpr = pl.lit(None, pl.String)
for g, ws in GROUPS.items():
    gexpr = pl.when(pl.col("word").is_in(ws)).then(pl.lit(g)).otherwise(gexpr)


def one_extra(df):
    return (df.with_columns(words("n2").list.set_difference(words("n1")).alias("ex"), words("n1").list.set_difference(words("n2")).alias("miss"))
            .filter((pl.col("ex").list.len() == 1) & (pl.col("miss").list.len() == 0)).with_columns(pl.col("ex").list.first().alias("word")))


s1 = pp.scan_norm("test", "s1").filter(pl.col("country_n") == "france").select(pl.col("idx").alias("s1_idx"), pl.col("name_core").alias("n1"), pl.col("entity_id").alias("sid")).collect()
s1names = s1.select(pl.col("n1").alias("n2")).unique().with_columns(pl.lit(True).alias("is_s1_name"))
po = pp.scan_norm("test", "pool").filter(pl.col("country_n") == "france").select(pl.col("idx").alias("cand_idx"), pl.col("entity_id").alias("cid"), pl.col("name_core").alias("n2")).collect()
tf = (pl.scan_parquet(sorted((sorted(R.glob("*-v10testsrc"))[-1] / "test_feats").glob("part-*.parquet"))).filter(pl.col("hn_eq") == 1)
      .select("s1_idx", "cand_idx").join(s1.lazy().select("s1_idx"), on="s1_idx").collect())
t = one_extra(tf.join(s1, on="s1_idx").join(po, on="cand_idx")).with_columns(gexpr.alias("grp")).drop_nulls("grp").join(s1names, on="n2", how="left")
owner = (pl.read_csv(pp.ROOT / "output" / "SUBMIT_THIS" / "matching_results.tsv", separator="\t", quote_char=None, infer_schema=False)
         .select(pl.col("source1_entity_id").alias("owner"), pl.col("matched_entity_ids").fill_null("").str.split(",").alias("cid")).explode("cid").filter(pl.col("cid") != ""))
t = t.join(owner, on="cid", how="left").with_columns(
    pl.when(pl.col("owner") == pl.col("sid")).then(pl.lit("this S1")).when(pl.col("owner").is_null()).then(pl.lit("nobody")).otherwise(pl.lit("other S1")).alias("went_to"))
with pl.Config(tbl_rows=30, tbl_width_chars=200, fmt_str_lengths=60):
    print("[France] one extra word, SAME house number: by group x candidate-name-is-an-S1-name, where the record went")
    print(t.group_by("grp", pl.col("is_s1_name").fill_null(False)).agg(
        pl.len().alias("n"), (pl.col("went_to") == "this S1").mean().round(3).alias("to_this"),
        (pl.col("went_to") == "other S1").mean().round(3).alias("to_other"), (pl.col("went_to") == "nobody").mean().round(3).alias("to_nobody")).sort("grp", "is_s1_name"))
