"""France-only variants of the mix: US/India rows fixed (v10_ce_dense); France rows from a France model at a chosen
matches-per-S1 target. Each file differs from SUBMIT_THIS only on France, so LB change / 0.15 = France F0.5 change."""
import sys
from pathlib import Path
import numpy as np
import polars as pl
sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
import pipeline as pp
R = pp.RUNS_DIR
OUTD = Path(r"D:\amazon-ml\output\submissions")
cm = pp.scan_norm("test", "s1").select(pl.col("idx").alias("s1_idx"), "country_n", pl.col("entity_id").alias("source1_entity_id")).collect()
fr = cm.filter(pl.col("country_n") == "france")
base = pl.read_csv("output/SUBMIT_THIS/matching_results.tsv", separator="\t", quote_char=None, infer_schema=False)
for model in ("v10s_ce_dense", "v10_ce_dense"):
    d = sorted(R.glob(f"*-{model}"))[-1]
    sc = pl.read_parquet(d / "pred" / "scored-*.parquet").join(fr.select("s1_idx"), on="s1_idx")
    cid = pl.read_parquet(d / "pred" / "cand-*.parquet") if False else None
    for k in (3.20, 3.35, 3.50):
        best = min(((abs(pp.apply_decision(sc, {"mode": "threshold", "param": float(t), "excl": True}, 0.02).height / fr.height - k), float(t))
                    for t in np.arange(0.30, 0.995, 0.005)), key=lambda x: x[0])
        sel = pp.apply_decision(sc, {"mode": "threshold", "param": best[1], "excl": True}, 0.02)
        ids = pp._join_ids(sel).join(fr.select("s1_idx", "source1_entity_id"), on="s1_idx").select("source1_entity_id", pl.col("ids").alias("fr_ids"))
        out = (base.join(ids, on="source1_entity_id", how="left", maintain_order="left")
               .with_columns(pl.when(pl.col("source1_entity_id").is_in(fr["source1_entity_id"].implode()))
                             .then(pl.col("fr_ids").fill_null("")).otherwise(pl.col("matched_entity_ids").fill_null("")).alias("matched_entity_ids"))
               .drop("fr_ids"))
        name = f"fr_{model.replace('_ce_dense', '')}_k{int(round(k * 100))}"
        out.write_csv(OUTD / f"{name}_matching_results.tsv", separator="\t", quote_style="never")
        print(name, "thr", round(best[1], 3), "France k", round(sel.height / fr.height, 3), "rows", out.height)
