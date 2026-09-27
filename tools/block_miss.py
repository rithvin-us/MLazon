"""Blocking misses on val: true matches that never became candidates. Categories + examples."""
import sys
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
import pipeline as pp
R = pp.RUNS_DIR
src = sorted(R.glob("*-v10src"))[-1]
vt = pl.read_parquet(R / "20260926-141000-v9" / "val_truth.parquet").select("s1_idx", "s1_id", "country_n")
gt = (pl.scan_csv("student_resource/dataset/train/train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema=False)
      .join(vt.lazy(), left_on="source1_entity_id", right_on="s1_id").with_columns(pl.col("matched_entity_ids").str.split(","))
      .explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "").select("s1_idx", "country_n", pl.col("matched_entity_ids").alias("cid")).collect())
pool = pp.scan_norm("train", "pool").select(pl.col("idx").alias("cand_idx"), pl.col("entity_id").alias("cid"), pl.col("name_full").alias("n2"), pl.col("addr").alias("a2")).collect()
gt = gt.join(pool, on="cid", how="left")
cand = pl.scan_parquet(sorted((src / "train_feats").glob("part-*.parquet"))).filter(pl.col("is_val")).select("s1_idx", "cand_idx").collect()
miss = gt.join(cand, on=["s1_idx", "cand_idx"], how="anti")
print("val true pairs", gt.height, "missed by blocking", miss.height, f"({miss.height / gt.height:.4f})")
s1 = pp.scan_norm("train", "s1").select(pl.col("idx").alias("s1_idx"), pl.col("name_full").alias("n1"), pl.col("addr").alias("a1")).collect()
m = miss.join(s1, on="s1_idx").with_columns(pl.col("n1", "a1", "n2", "a2").fill_null(""))
m = m.with_columns(pl.Series("ns", cpdist(m["n1"].to_list(), m["n2"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)),
                   pl.Series("as", cpdist(m["a1"].to_list(), m["a2"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)))
cat = (pl.when(pl.col("a2") == "").then(pl.lit("cand addr EMPTY")).otherwise(pl.lit("cand addr present")) + pl.lit(" | ") +
       pl.when(pl.col("ns") >= 90).then(pl.lit("name>=90")).when(pl.col("ns") >= 60).then(pl.lit("name60-89")).otherwise(pl.lit("name<60")) + pl.lit(" | ") +
       pl.when(pl.col("as") >= 85).then(pl.lit("addr>=85")).when(pl.col("as") >= 50).then(pl.lit("addr50-84")).otherwise(pl.lit("addr<50")))
print(m.group_by(cat.alias("type")).len().sort("len", descending=True))
print(m.group_by("country_n").len())
raw = pl.concat([pl.scan_parquet(f"cache/raw_train_s{i}.parquet").select("entity_id", "business_name", "business_address") for i in (1, 2, 3)])
ex = m.filter(pl.col("a2") != "").sample(12, seed=5).join(pp.scan_norm("train", "s1").select(pl.col("idx").alias("s1_idx"), pl.col("entity_id").alias("sid")).collect(), on="s1_idx")
r = raw.filter(pl.col("entity_id").is_in(pl.concat([ex["sid"], ex["cid"]]).implode())).collect()
g = lambda i: r.filter(pl.col("entity_id") == i).row(0, named=True)
for x in ex.iter_rows(named=True):
    a, b = g(x["sid"]), g(x["cid"])
    print(f"ns={x['ns']:.0f} as={x['as']:.0f}\n  S1: {a['business_name']} | {a['business_address']}\n  C : {b['business_name']} | {b['business_address']}")

pl.Config.set_fmt_str_lengths(80)
print(m.group_by(cat.alias("type")).len().sort("len", descending=True))
pc = pp.scan_norm("train", "pool").select(pl.col("entity_id").alias("cid"), pl.col("country_n").alias("c_country")).collect()
m2 = m.join(pc, on="cid", how="left")
print("candidate country == S1 country:", (m2["c_country"] == m2["country_n"]).mean())
print(m2.group_by("c_country").len().sort("len", descending=True).head(8))
easy = m2.filter((pl.col("ns") >= 85) & (pl.col("as") >= 80))
print("easy misses (name>=85 & addr>=80):", easy.height)
