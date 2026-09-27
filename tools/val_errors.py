"""Examples of current errors on labelled validation (v10_ce, its decision): false positives and false negatives."""
import json, sys
import polars as pl
sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
import pipeline as pp
R = pp.RUNS_DIR
ce = sorted(R.glob("*-v10_ce"))[-1]
m = json.loads((ce / "metrics.json").read_text())
dec = {"mode": m["decision_mode"], "param": m["decision_param"], "excl": m["decision_excl"]}
va = pl.read_parquet(ce / "val_scored_ce.parquet")
vt = pl.read_parquet(R / "20260926-141000-v9" / "val_truth.parquet")
sel = pp.apply_decision(va, dec, 0.02).select("s1_idx", "cand_idx", pl.lit(1).alias("chosen"))
v = va.join(vt.select("s1_idx", "country_n"), on="s1_idx").join(sel, on=["s1_idx", "cand_idx"], how="left").with_columns(pl.col("chosen").fill_null(0))
fp = v.filter((pl.col("chosen") == 1) & (pl.col("label") == 0))
fn = v.filter((pl.col("chosen") == 0) & (pl.col("label") == 1))
print("val S1", vt.height, "FP", fp.height, "FN", fn.height, "FN p<0.02(not stored?)", fn.filter(pl.col("p") < 0.02).height)
print("FN p bands", fn.group_by(pl.col("p").cut([0.1, 0.3, 0.5, 0.7]).alias("b")).len().sort("b"))
s1 = pp.scan_norm("train", "s1").select(pl.col("idx").alias("s1_idx"), "entity_id")
pool = pp.scan_norm("train", "pool").select(pl.col("idx").alias("cand_idx"), pl.col("entity_id").alias("cid"))
raw = pl.concat([pl.scan_parquet(f"cache/raw_train_s{i}.parquet").select("entity_id", "business_name", "business_address") for i in (1, 2, 3)])
def show(df, title, n):
    d = df.sample(min(n, df.height), seed=3).lazy().join(s1, on="s1_idx").join(pool, on="cand_idx").collect()
    ids = pl.concat([d["entity_id"], d["cid"]]).unique()
    r = raw.filter(pl.col("entity_id").is_in(ids.implode())).collect()
    get = lambda i: r.filter(pl.col("entity_id") == i).row(0, named=True)
    print(f"\n=== {title}")
    for x in d.iter_rows(named=True):
        a, b = get(x["entity_id"]), get(x["cid"])
        print(f"p={x['p']:.3f} {x['country_n']}\n  S1: {a['business_name']} | {a['business_address']}\n  C : {b['business_name']} | {b['business_address']}")
show(fp, "FALSE POSITIVES (chosen, not a match)", 14)
show(fn.filter(pl.col("p") >= 0.1), "FALSE NEGATIVES with p>=0.1 (missed)", 14)
