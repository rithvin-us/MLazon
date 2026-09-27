"""Does raw formatting (erased by normalisation) separate true copies from decoys? Val pairs, labelled."""
import sys
import polars as pl
sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
import pipeline as pp
R = pp.RUNS_DIR
src = sorted(R.glob("*-v10src"))[-1]
f = (pl.scan_parquet(sorted((src / "train_feats").glob("part-*.parquet"))).filter(pl.col("is_val"))
     .select("s1_idx", "cand_idx", "label", "nc_tset", "ad_tset", "len_addr_c", "len_addr_s1").collect())
s1 = pp.scan_norm("train", "s1").select(pl.col("idx").alias("s1_idx"), pl.col("entity_id").alias("sid")).collect()
pool = pp.scan_norm("train", "pool").select(pl.col("idx").alias("cand_idx"), pl.col("entity_id").alias("cid")).collect()
raw = pl.concat([pl.scan_parquet(f"cache/raw_train_s{i}.parquet").select("entity_id", "business_name", "business_address") for i in (1, 2, 3)])
f = f.join(s1, on="s1_idx").join(pool, on="cand_idx")
ids = pl.concat([f["sid"], f["cid"]]).unique()
r = raw.filter(pl.col("entity_id").is_in(ids.implode())).collect()
f = (f.join(r.rename({"entity_id": "sid", "business_name": "rn1", "business_address": "ra1"}), on="sid")
      .join(r.rename({"entity_id": "cid", "business_name": "rn2", "business_address": "ra2"}), on="cid"))
n1, n2 = pl.col("rn1").fill_null(""), pl.col("rn2").fill_null("")
f = f.with_columns((n1 == n2).alias("raw_eq"), (n1.str.to_lowercase() == n2.str.to_lowercase()).alias("ci_eq"),
                   (n2 == n2.str.to_uppercase()).alias("c_upper"), (n2 == n2.str.to_lowercase()).alias("c_lower"),
                   n2.str.contains("  ").alias("c_dblspace"), (n2 != n2.str.strip_chars()).alias("c_pad"),
                   pl.col("cid").str.slice(0, 2).alias("src"))
def rep(df, title):
    print(f"\n== {title}: n={df.height} match rate {df['label'].mean():.3f}")
    for c in ["raw_eq", "ci_eq", "c_upper", "c_lower", "c_dblspace", "c_pad", "src"]:
        print(df.group_by(c).agg(pl.len(), pl.col("label").mean().round(3).alias("rate")).sort(c).rows())
rep(f.filter((pl.col("nc_tset") >= 100) & (pl.col("len_addr_c") == 0) & (pl.col("len_addr_s1") > 0)), "exact core name, EMPTY candidate address")
rep(f.filter((pl.col("nc_tset") >= 100) & (pl.col("ad_tset") >= 95)), "exact core name, same address")
rep(f.filter((pl.col("nc_tset") >= 100) & (pl.col("len_addr_c") > 0) & (pl.col("ad_tset") < 95) & (pl.col("ad_tset") >= 70)), "exact core name, similar address")

ce = sorted(R.glob("*-v10_ce"))[-1]
f = f.join(pl.read_parquet(ce / "val_scored_ce.parquet").select("s1_idx", "cand_idx", "p"), on=["s1_idx", "cand_idx"], how="left").with_columns(pl.col("p").fill_null(0.0))
a1, a2 = pl.col("ra1").fill_null(""), pl.col("ra2").fill_null("")
f = f.with_columns((a1.str.to_lowercase() == a2.str.to_lowercase()).alias("addr_ci_eq"))
print("\n== model p vs raw identity (all val pairs)")
print(f.group_by("ci_eq", "addr_ci_eq").agg(pl.len(), pl.col("label").mean().round(4).alias("rate"), pl.col("p").mean().round(4).alias("mean_p"),
      ((pl.col("p") < 0.5) & (pl.col("label") == 1)).sum().alias("pos_p<0.5"), ((pl.col("p") >= 0.5) & (pl.col("label") == 0)).sum().alias("neg_p>=0.5")).sort("ci_eq", "addr_ci_eq"))
print("\n== among label=1 with p<0.5 (current misses): raw identity shares")
m = f.filter((pl.col("label") == 1) & (pl.col("p") < 0.5))
print(m.select(pl.len(), pl.col("ci_eq").mean().round(3), pl.col("addr_ci_eq").mean().round(3), (pl.col("len_addr_c") == 0).mean().round(3).alias("empty_addr")))
print("\n== among label=0 with p>=0.5 (current false alarms)")
m = f.filter((pl.col("label") == 0) & (pl.col("p") >= 0.5))
print(m.select(pl.len(), pl.col("ci_eq").mean().round(3), pl.col("addr_ci_eq").mean().round(3), (pl.col("len_addr_c") == 0).mean().round(3).alias("empty_addr")))
