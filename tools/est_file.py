"""Estimate a submission's LB using our calibrated test probabilities (US/India: v10_ce_dense, France: v10s_ce_dense).
Per S1: expected true count = sum of our p over its candidates (+ blocking misses); empty prediction scores
P(singleton) ~ prod(1-p); otherwise F ~ 1.25 * sum(p of predicted) / (0.25 * E[n_true] + n_pred).
Calibrated by applying the same estimator to our own submitted file (known LB 0.975766)."""
import sys
import polars as pl
sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
import pipeline as pp
R = pp.RUNS_DIR
cm = pp.scan_norm("test", "s1").select(pl.col("idx").alias("s1_idx"), "country_n", pl.col("entity_id").alias("source1_entity_id")).collect()
a = pl.read_parquet(sorted(R.glob("*-v10_ce_dense"))[-1] / "pred" / "scored-*.parquet").join(cm.filter(pl.col("country_n") != "france").select("s1_idx"), on="s1_idx")
b = pl.read_parquet(sorted(R.glob("*-v10s_ce_dense"))[-1] / "pred" / "scored-*.parquet").join(cm.filter(pl.col("country_n") == "france").select("s1_idx"), on="s1_idx")
sc = pl.concat([a.select("s1_idx", "cid", "p"), b.select("s1_idx", "cid", "p")])
per = sc.group_by("s1_idx").agg((pl.col("p").sum() * 1.015).alias("ek"), (1 - pl.col("p")).clip(1e-6, 1).log().sum().exp().alias("p0"))
base = cm.join(per, on="s1_idx", how="left").with_columns(pl.col("ek").fill_null(0.05), pl.col("p0").fill_null(0.95))


def est(path):
    m = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)
    rows, uniq = m.height, m["source1_entity_id"].n_unique()
    e = m.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
    dup_ids = e.height - e["matched_entity_ids"].n_unique()
    e = e.join(cm.select("source1_entity_id", "s1_idx"), on="source1_entity_id").join(sc.rename({"cid": "matched_entity_ids"}), on=["s1_idx", "matched_entity_ids"], how="left").with_columns(pl.col("p").fill_null(0.005))
    g = e.group_by("s1_idx").agg(pl.len().alias("n_pred"), pl.col("p").sum().alias("etp"))
    x = base.join(g, on="s1_idx", how="left").with_columns(pl.col("n_pred", "etp").fill_null(0))
    x = x.with_columns(pl.when(pl.col("n_pred") == 0).then(pl.col("p0")).otherwise(1.25 * pl.col("etp") / (0.25 * pl.col("ek") + pl.col("n_pred"))).alias("f"))
    by = x.group_by("country_n").agg(pl.col("f").mean().round(4).alias("est_f"), (pl.col("n_pred") == 0).mean().round(3).alias("empty_share"),
                                     pl.col("n_pred").mean().round(2).alias("mean_pred")).sort("country_n")
    return rows, uniq, dup_ids, float(x["f"].mean()), by


for name, path in (("OURS (LB 0.975766)", "output/SUBMIT_THIS/matching_results.tsv"), ("zeba_V1", r"E:\Downloads\matching_results_zeba_V1.tsv")):
    rows, uniq, dup, f, by = est(path)
    print(f"\n== {name}: rows {rows:,} unique S1 {uniq:,} ids assigned to >1 S1: {dup:,}  raw estimate {f:.4f}")
    print(by)
