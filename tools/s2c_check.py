"""Near-certain French copies (identical core name, core address 100, same house number) accepted in S2a (France =
previous final) but not in S2c (France stage 2): where did the record go in S2c (another S1 / nobody), how contested
was it (records scored >= 0.2 by 2+ S1), and the reverse flow (accepted in S2c, not in S2a)."""
import sys

import polars as pl

sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
import pipeline as pp  # noqa: E402

R = pp.RUNS_DIR
SUB = pp.ROOT / "output" / "submissions"
s1 = pp.scan_norm("test", "s1").filter(pl.col("country_n") == "france").select(pl.col("idx").alias("s1_idx"), pl.col("entity_id").alias("sid"), pl.col("name_core").alias("n1")).collect()
po = pp.scan_norm("test", "pool").filter(pl.col("country_n") == "france").select(pl.col("entity_id").alias("cid"), pl.col("name_core").alias("n2")).collect()
f = (pl.scan_parquet(sorted((sorted(R.glob("*-v10testsrc"))[-1] / "test_feats").glob("part-*.parquet"))).select("s1_idx", "cid", "hn_eq", "ad_core_tset")
     .join(s1.lazy(), on="s1_idx").collect().join(po, on="cid").filter((pl.col("n1") == pl.col("n2")) & (pl.col("ad_core_tset") == 100) & (pl.col("hn_eq") == 1)))
sc = pl.read_parquet(sorted(R.glob("*-v10sfr_bge2_dense"))[-1] / "pred" / "scored-*.parquet").select("s1_idx", "cid", "p").join(s1.select("s1_idx"), on="s1_idx")
contested = sc.filter(pl.col("p") >= 0.2).group_by("cid").agg(pl.len().alias("n_s1"))


def owners(name):
    return (pl.read_csv(SUB / f"PROBE_{name}_matching_results.tsv", separator="\t", quote_char=None, infer_schema=False)
            .join(s1.select(pl.col("sid").alias("source1_entity_id")), on="source1_entity_id")
            .select(pl.col("source1_entity_id").alias("owner"), pl.col("matched_entity_ids").fill_null("").str.split(",").alias("cid")).explode("cid").filter(pl.col("cid") != ""))


a, c = owners("stage2_usin"), owners("stage2_frdense")
g = (f.join(a.rename({"owner": "own_a"}), on="cid", how="left").join(c.rename({"owner": "own_c"}), on="cid", how="left")
     .join(contested, on="cid", how="left").with_columns(pl.col("n_s1").fill_null(0)))
lost = g.filter((pl.col("own_a") == pl.col("sid")) & (pl.col("own_c").is_null() | (pl.col("own_c") != pl.col("sid"))))
won = g.filter((pl.col("own_c") == pl.col("sid")) & (pl.col("own_a").is_null() | (pl.col("own_a") != pl.col("sid"))))
for name, d in (("accepted in S2a, not in S2c", lost), ("accepted in S2c, not in S2a", won)):
    other = "own_c" if name.startswith("accepted in S2a") else "own_a"
    print(f"[near-certain] {name}: {d.height:,} pairs; the record in the other file went to nobody {d[other].is_null().mean():.3f}, "
          f"to another S1 {(d[other].is_not_null()).mean():.3f}; contested (2+ S1 >= 0.2) {(d['n_s1'] >= 2).mean():.3f}")
