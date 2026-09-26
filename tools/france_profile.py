"""Dataset-level profile of the test split by country (aggregates only): volumes, name crowding,
vocabulary concentration, legal forms, noise types in Source 2/3, address admin forms.
Writes runs/france_profile.txt
"""
from pathlib import Path

import polars as pl

pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(250)
ROOT = Path(r"D:\amazon-ml")
OUT = ROOT / "runs" / "france_profile.txt"
out = open(OUT, "w", encoding="utf-8")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    out.write(s + "\n")


s1 = pl.scan_parquet(ROOT / "cache/norm_test_s1_v5/*.parquet").rename({"country_n": "country"})
pool = pl.scan_parquet(ROOT / "cache/norm_test_pool_v5/*.parquet").rename({"country_n": "country"})

say("=== 1. volumes")
v = (pool.group_by("country", "src").len().collect().pivot(on="src", index="country", values="len")
     .join(s1.group_by("country").len().collect().rename({"len": "S1"}), on="country"))
say(v.with_columns((pl.sum_horizontal(pl.exclude("country", "S1")) / pl.col("S1")).round(2).alias("pool_per_S1")).sort("country"))

say("\n=== 2. name crowding: how many other businesses share the same core name")
s1c = s1.group_by("country", "name_core").len().rename({"len": "n_s1_same"})
poolc = pool.group_by("country", "name_core").len().rename({"len": "n_pool_same"})
c = (s1.select("country", "name_core").join(s1c, on=["country", "name_core"]).join(poolc, on=["country", "name_core"], how="left")
     .with_columns(pl.col("n_pool_same").fill_null(0)).collect())
say(c.group_by("country").agg(
    (pl.col("n_s1_same") > 1).mean().round(4).alias("S1_name_shared"),
    pl.col("n_s1_same").mean().round(2).alias("S1_per_name"),
    pl.col("n_s1_same").max().alias("max_S1_one_name"),
    pl.col("n_pool_same").mean().round(2).alias("pool_same_name_mean"),
    (pl.col("n_pool_same") >= 10).mean().round(4).alias("pool_same>=10"),
    pl.col("name_core").n_unique().alias("distinct_names"),
).sort("country"))

say("\n=== 3. vocabulary concentration (S1 names and addresses)")
for col in ("name_full", "addr"):
    t = (s1.select("country", pl.col(col).fill_null("").str.split(" ").alias("t")).explode("t").filter(pl.col("t") != "")
         .group_by("country", "t").len().collect())
    say(f"-- {col}")
    say(t.sort("len", descending=True).group_by("country", maintain_order=True).agg(
        pl.len().alias("distinct_tokens"),
        (pl.col("len").head(20).sum() / pl.col("len").sum()).round(3).alias("top20_share"),
        (pl.col("len").head(100).sum() / pl.col("len").sum()).round(3).alias("top100_share"),
        (pl.col("len").head(1000).sum() / pl.col("len").sum()).round(3).alias("top1000_share"),
    ).sort("country"))

say("\n=== 4. legal forms in S1 names (share of S1)")
lf = {"fr": r"\b(sarl|sas|eurl|sa|sci|ets|ste|ei)\b", "us": r"\b(llc|inc|corp|co|ltd|lp|llp|pllc)\b", "in": r"\b(pvt|ltd|llp|opc)\b"}
say(s1.group_by("country").agg(*[pl.col("name_full").fill_null("").str.contains(r).mean().round(3).alias(k) for k, r in lf.items()],
                               pl.col("name_full").fill_null("").str.split(" ").list.len().mean().round(2).alias("name_tokens"),
                               pl.col("addr").fill_null("").str.split(" ").list.len().mean().round(2).alias("addr_tokens")).sort("country").collect())

say("\n=== 5. noise types in Source 2/3 records (raw text), share of pool records")
raw = pl.concat([pl.scan_parquet(ROOT / f"cache/raw_test_s{i}.parquet").select("entity_id", "business_name", "business_address") for i in (2, 3)])
r = raw.join(pool.select("entity_id", "country"), on="entity_id")
n, a = pl.col("business_name").fill_null(""), pl.col("business_address").fill_null("")
say(r.group_by("country").agg(
    pl.len().alias("n"),
    n.str.starts_with("@").mean().round(4).alias("handle_@"),
    n.str.contains(r"(?i)\.com|www\.").mean().round(4).alias("domain"),
    n.str.contains(r"(?i)\bt/a\b|\bdba\b|\bd/b/a\b").mean().round(4).alias("t/a_dba"),
    (n == n.str.to_uppercase()).mean().round(4).alias("ALLCAPS"),
    (n == n.str.to_lowercase()).mean().round(4).alias("lower"),
    n.str.contains(r"[a-zA-Z][0-9][a-zA-Z]|\b[0-9][a-zA-Z]{2,}").mean().round(4).alias("digit_in_word"),
    n.str.contains(r"[^\x00-\x7F]").mean().round(4).alias("non_ascii_name"),
    (a == "").mean().round(4).alias("addr_empty"),
    a.str.contains(r"(?i)^(no\.?|#)\s").mean().round(4).alias("addr_No/#"),
    a.str.contains(r"^[^0-9,]+,").mean().round(4).alias("addr_starts_non_number"),
    a.str.contains(r"(?i)\b\d+\s*(bis|ter)\b|\d+[a-z]\d+").mean().round(4).alias("bis/ter/36A63"),
).sort("country").collect())

say("\n=== 6. France address admin form (S1 vs pool)")
admin = (pl.when(pl.col("addr").str.contains(r"\b(nord|gironde|loire atlantique|pas de calais)\b")).then(pl.lit("department"))
         .when(pl.col("addr").str.contains(r"\b(hdf|naq|pdl)\b")).then(pl.lit("region"))
         .when(pl.col("addr").fill_null("") == "").then(pl.lit("empty")).otherwise(pl.lit("none")))
for name, lf_ in (("S1", s1), ("pool", pool)):
    say(f"-- {name}")
    say(lf_.filter(pl.col("country") == "france").group_by(admin.alias("form")).len()
        .with_columns((pl.col("len") / pl.col("len").sum()).round(3).alias("share")).sort("form").collect())

say("\n=== 7. France cities (S1 address tokens that are city names, top 25)")
say(s1.filter(pl.col("country") == "france").select(pl.col("addr").fill_null("").str.extract(r"([a-z ]+?)\s+(hdf|naq|pdl)\s*$", 1).alias("city"))
    .group_by("city").len().sort("len", descending=True).head(25).collect())
out.close()
print(f"\nwritten {OUT}")
