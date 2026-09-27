"""Where is the remaining F0.5 loss (v10_ce, its decision, val)? Oracle what-ifs."""
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
tvc = vt.select("s1_idx", "n_true")
vv = va.join(vt.select("s1_idx"), on="s1_idx")
sel = pp.apply_decision(va, dec, 0.02).join(vt.select("s1_idx"), on="s1_idx")
f = lambda s: pp.eval_selection(s, tvc)["f05"]
cur = f(sel)
in_cand = vv.group_by("s1_idx").agg(pl.col("label").sum().alias("pos_c"))
tot = vt.select("s1_idx", "n_true").join(in_cand, on="s1_idx", how="left").fill_null(0)
print(f"current {cur:.5f}; blocking recall {tot['pos_c'].sum() / tot['n_true'].sum():.4f}")
perfect = vv.filter(pl.col("label") == 1)
print(f"perfect choice among candidates (blocking limit) {f(perfect):.5f}")
nofp = sel.filter(pl.col("label") == 1)
print(f"current minus all false positives {f(nofp):.5f}  (FP cost {f(nofp) - cur:.5f})")
allfn = pl.concat([sel, vv.filter(pl.col("label") == 1).join(sel.select("s1_idx", "cand_idx"), on=["s1_idx", "cand_idx"], how="anti").select(sel.columns)])
print(f"current plus all missed candidates (model FN) {f(allfn):.5f}  (model-FN cost {f(allfn) - cur:.5f})")
print(f"blocking cost {f(perfect) and (1 - f(perfect)):.5f} (1 - perfect)")
# empty-address share of model FN
src = sorted(R.glob("*-v10src"))[-1]
fe = (pl.scan_parquet(sorted((src / "train_feats").glob("part-*.parquet"))).filter(pl.col("is_val")).select("s1_idx", "cand_idx", "len_addr_c", "nc_tset").collect())
fn = vv.filter(pl.col("label") == 1).join(sel.select("s1_idx", "cand_idx"), on=["s1_idx", "cand_idx"], how="anti").join(fe, on=["s1_idx", "cand_idx"])
print("model FN:", fn.height, "empty cand addr", (fn["len_addr_c"] == 0).sum(), "name core<90", (fn["nc_tset"] < 90).sum())
fp = sel.filter(pl.col("label") == 0).join(fe, on=["s1_idx", "cand_idx"])
print("FP:", fp.height, "empty cand addr", (fp["len_addr_c"] == 0).sum())
