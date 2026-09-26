"""France error analysis on the v9_ce test predictions (no labels exist for France).
A. feature distributions by country, for confident / uncertain / top-ranked pairs
B. normalisation coverage by country (postcode, house numbers, empty fields)
C. small text samples: uncertain France pairs, over-matched France S1, US contrast
Writes runs/france_diag.txt
"""
import sys
from pathlib import Path

import polars as pl

pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(30)
pl.Config.set_fmt_str_lengths(70)
pl.Config.set_tbl_width_chars(250)

ROOT = Path(r"D:\amazon-ml")
FEATS = ROOT / "runs" / "20260926-151110-v9_test" / "test_feats"
CE = ROOT / "runs" / "20260926-155941-v9_ce"
OUT = ROOT / "runs" / "france_diag.txt"
out = open(OUT, "w", encoding="utf-8")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    out.write(s + "\n")


s1n = pl.scan_parquet(ROOT / "cache/norm_test_s1_v5/*.parquet")
pooln = pl.scan_parquet(ROOT / "cache/norm_test_pool_v5/*.parquet")
ctry = s1n.select(pl.col("idx").alias("s1_idx"), pl.col("country_n").alias("country"))

# chosen pairs from the submitted file
m = (pl.read_csv(CE / "output/matching_results.tsv", separator="\t", quote_char=None, infer_schema=False)
     .with_columns(pl.col("matched_entity_ids").fill_null("").str.split(","))
     .explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
     .select(pl.col("source1_entity_id").alias("s1_id"), pl.col("matched_entity_ids").alias("cid"), pl.lit(1).cast(pl.Int8).alias("chosen")))
s1ids = s1n.select(pl.col("idx").alias("s1_idx"), pl.col("entity_id").alias("s1_id")).collect()
m = m.join(s1ids, on="s1_id").drop("s1_id")

KEY = ["pc_eq", "pc_both", "hn_eq", "hn_both", "hn_sim", "num_both", "num_jacc", "ad_tset", "ad_core_tset", "ad_core_empty",
       "addr_empty_any", "nf_tset", "nc_tset", "ncc_ratio", "legal_eq", "legal_conflict", "c_oov_frac", "twin_better",
       "anc_ad_mean", "len_addr_s1", "len_addr_c", "n_cands", "rr", "is_s3"]
sc = pl.scan_parquet(CE / "pred/scored-*.parquet").select("s1_idx", "cand_idx", "p")
f = (pl.scan_parquet(FEATS / "*.parquet").select("s1_idx", "cand_idx", "cid", "rr_rank", *KEY)
     .join(sc, on=["s1_idx", "cand_idx"], how="left").with_columns(pl.col("p").fill_null(0.0))
     .join(ctry, on="s1_idx").collect())
f = f.join(m, on=["s1_idx", "cid"], how="left").with_columns(pl.col("chosen").fill_null(0))
say(f"pairs {f.height:,}")

say("\n=== A1. share of pairs by p-bucket, per country")
say(f.with_columns(pl.col("p").cut([0.02, 0.2, 0.5, 0.8, 0.9, 0.995]).alias("pb"))
    .group_by("country", "pb").len().with_columns((pl.col("len") / pl.col("len").sum().over("country")).round(4).alias("share"))
    .pivot(on="country", index="pb", values="share").sort("pb"))


def means(df, title):
    say(f"\n=== {title}")
    say(df.group_by("country").agg(pl.len().alias("n"), *[pl.col(k).mean().round(3) for k in KEY]).sort("country")
        .transpose(include_header=True, header_name="feat", column_names="country"))


means(f.filter(pl.col("chosen") == 1), "A2. feature means on CHOSEN pairs (what the model accepts)")
means(f.filter((pl.col("p") >= 0.2) & (pl.col("p") < 0.8)), "A3. feature means on UNCERTAIN pairs (0.2<=p<0.8)")
means(f.filter(pl.col("rr_rank") == 1), "A4. feature means on re-ranker top-1 pair of every S1")

say("\n=== A5. chosen pairs per S1 distribution by country (true generator: ~3.36 mean everywhere)")
k = (f.group_by("country", "s1_idx").agg(pl.col("chosen").sum().alias("k"))
     .with_columns(pl.col("k").clip(0, 8)))
say(k.group_by("country", "k").len().with_columns((pl.col("len") / pl.col("len").sum().over("country")).round(4).alias("share"))
    .pivot(on="country", index="k", values="share").sort("k"))

say("\n=== B. normalisation coverage (S1 and pool)")
for name, lf in (("S1", s1n), ("pool", pooln)):
    say(f"-- {name}")
    say(lf.group_by("country_n").agg(
        pl.len().alias("n"),
        (pl.col("postcode").fill_null("") != "").mean().round(4).alias("has_pc"),
        pl.col("postcode").fill_null("").str.len_chars().mean().round(2).alias("pc_len"),
        (pl.col("addr_nums").fill_null("") != "").mean().round(4).alias("has_nums"),
        (pl.col("addr").fill_null("") == "").mean().round(4).alias("addr_empty"),
        (pl.col("name_core").fill_null("") == "").mean().round(4).alias("core_empty"),
        pl.col("addr").fill_null("").str.split(" ").list.len().mean().round(2).alias("addr_tok"),
        pl.col("name_full").fill_null("").str.split(" ").list.len().mean().round(2).alias("name_tok"),
    ).sort("country_n").collect())

say("\n=== B2. most frequent address tokens, France S1 (top 60)")
say(s1n.filter(pl.col("country_n") == "france").select(pl.col("addr").fill_null("").str.split(" ").alias("t")).explode("t")
    .filter(pl.col("t") != "").group_by("t").len().sort("len", descending=True).head(60).collect()
    .select(pl.format("{}:{}", "t", "len").str.join("  ")).item())
say("\n=== B3. most frequent name tokens, France S1 (top 60)")
say(s1n.filter(pl.col("country_n") == "france").select(pl.col("name_full").fill_null("").str.split(" ").alias("t")).explode("t")
    .filter(pl.col("t") != "").group_by("t").len().sort("len", descending=True).head(60).collect()
    .select(pl.format("{}:{}", "t", "len").str.join("  ")).item())

# ---- C. text samples
raw = pl.concat([pl.scan_parquet(ROOT / f"cache/raw_test_s{i}.parquet").select("entity_id", "business_name", "business_address")
                 for i in (1, 2, 3)])
norm = pl.concat([s1n, pooln]).select("entity_id", "name_full", "addr", "postcode", "addr_nums")


def show(pairs, title):
    ids = pl.concat([pairs.select(pl.col("s1_id").alias("entity_id")), pairs.select(pl.col("cid").alias("entity_id"))]).unique()
    r = raw.join(ids.lazy(), on="entity_id").collect()
    n = norm.join(ids.lazy(), on="entity_id").collect()
    say(f"\n=== {title}")
    for row in pairs.iter_rows(named=True):
        a = r.filter(pl.col("entity_id") == row["s1_id"]).row(0, named=True)
        b = r.filter(pl.col("entity_id") == row["cid"]).row(0, named=True)
        na = n.filter(pl.col("entity_id") == row["s1_id"]).row(0, named=True)
        nb = n.filter(pl.col("entity_id") == row["cid"]).row(0, named=True)
        say(f"p={row['p']:.3f} chosen={row['chosen']} rr#{int(row['rr_rank'])} pc_eq={row['pc_eq']:.0f} hn_eq={row['hn_eq']:.0f} "
            f"ad_tset={row['ad_tset']:.2f} nc_tset={row['nc_tset']:.2f} legal_conf={row['legal_conflict']:.0f} twin={row['twin_better']:.0f}")
        say(f"   S1  raw: {a['business_name']} | {a['business_address']}")
        say(f"   S1 norm: {na['name_full']} | {na['addr']} | pc={na['postcode']} nums={na['addr_nums']}")
        say(f"   C   raw: {b['business_name']} | {b['business_address']}")
        say(f"   C  norm: {nb['name_full']} | {nb['addr']} | pc={nb['postcode']} nums={nb['addr_nums']}")


f = f.join(s1ids, on="s1_idx")
fr_mid = f.filter((pl.col("country") == "france") & (pl.col("p") >= 0.2) & (pl.col("p") < 0.8)).sample(25, seed=7)
show(fr_mid, "C1. 25 random UNCERTAIN France pairs")
over = k.filter((pl.col("country") == "france") & (pl.col("k") >= 6)).sample(3, seed=3).select("s1_idx")
show(f.join(over, on="s1_idx").filter(pl.col("p") >= 0.3).sort("s1_idx", "p", descending=[False, True]), "C2. 3 France S1 with >=6 chosen matches (over-match?)")
us_mid = f.filter((pl.col("country") == "us") & (pl.col("p") >= 0.2) & (pl.col("p") < 0.8)).sample(8, seed=7)
show(us_mid, "C3. 8 random UNCERTAIN US pairs (contrast)")
out.close()
print(f"\nwritten {OUT}")
