"""France S1 with 0 or 1 predicted matches (current SUBMIT_THIS France rows = v10s_ce_dense): top candidates + p."""
import sys
import polars as pl
sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
import pipeline as pp
R = pp.RUNS_DIR
d = sorted(R.glob("*-v10s_ce_dense"))[-1]
sc = pl.read_parquet(d / "pred" / "scored-*.parquet")
cm = pp.scan_norm("test", "s1").select(pl.col("idx").alias("s1_idx"), "country_n", pl.col("entity_id").alias("sid")).collect()
sub = pl.read_csv("output/SUBMIT_THIS/matching_results.tsv", separator="\t", quote_char=None, infer_schema=False).with_columns(
    pl.col("matched_entity_ids").fill_null("").str.split(",").list.eval(pl.element().filter(pl.element() != "")).list.len().alias("k"))
fr = cm.filter(pl.col("country_n") == "france").join(sub.rename({"source1_entity_id": "sid"}), on="sid")
raw = pl.concat([pl.scan_parquet(f"cache/raw_test_s{i}.parquet").select("entity_id", "business_name", "business_address") for i in (1, 2, 3)])
for k in (0, 1):
    ids = fr.filter(pl.col("k") == k).sample(6, seed=k + 3)
    top = sc.join(ids.select("s1_idx", "sid"), on="s1_idx").sort(["s1_idx", "p"], descending=[False, True]).group_by("s1_idx", maintain_order=True).head(4)
    r = raw.filter(pl.col("entity_id").is_in(pl.concat([ids["sid"], top["cid"]]).implode())).collect()
    g = lambda i: r.filter(pl.col("entity_id") == i).row(0, named=True)
    print(f"\n######## France S1 with k={k} predicted")
    for s in ids.iter_rows(named=True):
        a = g(s["sid"])
        print(f"S1: {a['business_name']} | {a['business_address']}")
        for c in top.filter(pl.col("s1_idx") == s["s1_idx"]).iter_rows(named=True):
            b = g(c["cid"])
            print(f"   p={c['p']:.3f}  {b['business_name']} | {b['business_address']}")
