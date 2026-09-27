"""What makes France harder than US/India beyond vocabulary? Per country, aggregates only (final model's test scores):
  name genericity  share of S1 whose core name is shared with another S1 of the same country, and with >= 5
  city spread      distinct city-level tokens (last-but-one address component) covering 80% of S1
  crowding         candidate pairs per S1 (p >= 0.02), uncertain pairs (0.2 <= p < 0.8) per S1
  competition      pool records scored >= 0.2 for two or more S1 (share of such records among scored records)
  empty address    S1 / pool records with no address"""
import sys

import polars as pl

sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
import pipeline as pp  # noqa: E402

R = pp.RUNS_DIR
C = pp.CACHE_DIR
s1 = pp.scan_norm("test", "s1").select(pl.col("idx").alias("s1_idx"), "entity_id", "country_n", "name_core", "addr").collect()
rows = []
for c in ("us", "india", "france"):
    a = s1.filter(pl.col("country_n") == c)
    nc = a.group_by("name_core").agg(pl.len().alias("n"))
    a = a.join(nc, on="name_core")
    raw = pl.scan_parquet(C / "raw_test_s1.parquet").join(a.lazy().select("entity_id"), on="entity_id").select(
        pl.col("business_address").fill_null("").str.split(",").alias("parts")).collect()
    city = raw.select(pl.col("parts").list.get(-2, null_on_oob=True).str.strip_chars().str.to_lowercase().alias("city")).drop_nulls()
    cc = city.group_by("city").agg(pl.len().alias("n")).sort("n", descending=True).with_columns((pl.col("n").cum_sum() / pl.col("n").sum()).alias("cum"))
    rows.append({"country": c, "S1": a.height, "name_shared": round((a["n"] >= 2).mean(), 4), "name_shared>=5": round((a["n"] >= 5).mean(), 4),
                 "cities_for_80%": int((cc["cum"] < 0.8).sum() + 1), "s1_empty_addr": round((a["addr"].fill_null("") == "").mean(), 4)})
sc = pl.scan_parquet(sorted(R.glob("*-v10sfr_bge2_dense"))[-1] / "pred" / "scored-*.parquet").select("s1_idx", "cand_idx", "p").join(
    s1.lazy().select("s1_idx", "country_n"), on="s1_idx")
g = sc.group_by("country_n").agg(pl.len().alias("pairs"), ((pl.col("p") >= 0.2) & (pl.col("p") < 0.8)).sum().alias("uncertain")).collect()
comp = (sc.filter(pl.col("p") >= 0.2).group_by("country_n", "cand_idx").agg(pl.len().alias("n")).group_by("country_n")
        .agg((pl.col("n") >= 2).mean().round(4).alias("contested_records")).collect())
po = pp.scan_norm("test", "pool").group_by("country_n").agg(pl.len().alias("pool"), (pl.col("addr").fill_null("") == "").mean().round(4).alias("pool_empty_addr")).collect()
out = pl.DataFrame(rows).join(g.rename({"country_n": "country"}), on="country").join(comp.rename({"country_n": "country"}), on="country").join(
    po.rename({"country_n": "country"}), on="country")
out = out.with_columns((pl.col("pairs") / pl.col("S1")).round(2).alias("pairs/S1(p>=.02)"), (pl.col("uncertain") / pl.col("S1")).round(3).alias("uncertain/S1"),
                       (pl.col("pool") / pl.col("S1")).round(2).alias("pool/S1")).drop("pairs", "uncertain", "pool")
with pl.Config(tbl_cols=20, tbl_width_chars=250):
    print(out)
