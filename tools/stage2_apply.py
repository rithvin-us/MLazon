"""Stage 2 on the stacked scores -> matching files (no preprocessing). Model: XGBoost depth 4 on competition +
S1-context features (no per-S1 count features), trained on density-augmented validation (US/India labels, copies
jittered); decision tuned on its out-of-fold scores (same family as the pipeline). Validated: plain val 0.98623 ->
0.98752, dense val 0.98487 -> 0.98683.
  S2a: US/India rows from stage 2, France rows from SUBMIT_THIS (current final)
  S2b: France rows also from stage 2 (count-matched threshold, as the pipeline does for an unlabelled country)"""
import subprocess
import sys

import numpy as np
import polars as pl
import xgboost as xgb

sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))  # stage2_margin_lib.py
import pipeline as pp  # noqa: E402
from stage2_margin_lib import feats, FEAT  # noqa: E402

R = pp.RUNS_DIR
ROOT = pp.ROOT
SUB = ROOT / "output" / "submissions"
COLS = [f for f in FEAT if f not in ("s1_sum", "s1_n02")]
MONO = "(" + ",".join("1" if f in ("p", "margin", "s1_max") else ("-1" if f in ("max_other", "rank_rec") else "0") for f in COLS) + ")"
PRM = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", device="cuda", eta=0.05, max_depth=4, min_child_weight=20,
           subsample=0.8, monotone_constraints=MONO, seed=0)


class _Q:
    def log(self, m):
        pass


# ---- training on dense val
v = pl.read_parquet(sorted(R.glob("*-v10_bge2"))[-1] / "val_scored_ce.parquet").filter(pl.col("p") >= 0.02)
vt = pl.read_parquet(sorted(R.glob("*-v10"))[-1] / "val_truth.parquet")
tvc = vt.select("s1_idx", "n_true")
cmap = pp.scan_norm("test", "s1").select(pl.col("idx").alias("s1_idx"), "country_n").collect()
us_in = pl.read_parquet(sorted(R.glob("*-v10_bge2_dense"))[-1] / "pred" / "scored-*.parquet").join(cmap, on="s1_idx").filter(pl.col("country_n") != "france")
fr = pl.read_parquet(sorted(R.glob("*-v10sfr_bge2_dense"))[-1] / "pred" / "scored-*.parquet").join(cmap, on="s1_idx").filter(pl.col("country_n") == "france")
aug, _ = pp.density_augment(v, vt, pl.concat([us_in.select("s1_idx", "cand_idx", "p"), fr.select("s1_idx", "cand_idx", "p")]), seed=42)
rng = np.random.default_rng(1)
aug = aug.with_columns(pl.when(pl.col("cand_idx") < 0).then((pl.col("p") + pl.Series(rng.uniform(-0.01, 0.01, aug.height).astype(np.float32))).clip(0.02, 1.0))
                       .otherwise(pl.col("p")).cast(pl.Float32).alias("p"))
da = feats(aug)
dv = da.join(vt.select("s1_idx"), on="s1_idx", how="semi").with_columns((pl.col("s1_idx") * 7919 % 5).alias("fold"))
dc = da.join(vt.select("s1_idx"), on="s1_idx", how="anti")
oof, cp = np.zeros(dv.height, np.float32), np.zeros(dc.height, np.float32)
for k in range(5):
    tr = dv.filter(pl.col("fold") != k)
    b = xgb.train(PRM, xgb.DMatrix(tr.select(COLS).to_numpy(), tr["label"].to_numpy(), feature_names=COLS), 400)
    m = (dv["fold"] == k).to_numpy()
    oof[m] = b.predict(xgb.DMatrix(dv.filter(pl.col("fold") == k).select(COLS).to_numpy(), feature_names=COLS))
    cp += b.predict(xgb.DMatrix(dc.select(COLS).to_numpy(), feature_names=COLS)) / 5
s2v = pl.concat([dv.drop("fold").with_columns(pl.Series("p2", oof)), dc.with_columns(pl.Series("p2", cp))])
dec_all = pp.tune_decision(s2v.select("s1_idx", "cand_idx", "label", pl.col("p2").alias("p")), tvc, 0.02, _Q())
dec = max(dec_all.values(), key=lambda z: z["f05"])
base_dec = max(pp.tune_decision(s2v.select("s1_idx", "cand_idx", "label", "p"), tvc, 0.02, _Q()).values(), key=lambda z: z["f05"])
print(f"[dense val] stacked p {base_dec['f05']:.5f} ({base_dec['mode']} {base_dec['param']}) -> stage 2 {dec['f05']:.5f} ({dec['mode']} {dec['param']} excl={dec['excl']})")
final = xgb.train(PRM, xgb.DMatrix(dv.select(COLS).to_numpy(), dv["label"].to_numpy(), feature_names=COLS), 400)
final.save_model(str(ROOT / "models" / "stage2_margin.json"))

# ---- test
t = feats(pl.concat([us_in.select("s1_idx", "cand_idx", "cid", "p", "country_n"), fr.select("s1_idx", "cand_idx", "cid", "p", "country_n")]))
t = t.with_columns(pl.Series("p2", final.predict(xgb.DMatrix(t.select(COLS).to_numpy(), feature_names=COLS)), dtype=pl.Float32))
d = {"mode": dec["mode"], "param": dec["param"], "excl": dec["excl"]}
sel_ui = pp.apply_decision(t.filter(pl.col("country_n") != "france").select("s1_idx", "cand_idx", "cid", pl.col("p2").alias("p")), d, 0.02)
n_ui = cmap.filter(pl.col("country_n") != "france").height
k_ui = sel_ui.height / n_ui
frp = t.filter(pl.col("country_n") == "france").select("s1_idx", "cand_idx", "cid", pl.col("p2").alias("p"))
n_fr = cmap.filter(pl.col("country_n") == "france").height
ex = pp.exclusive(frp.filter(pl.col("p") >= 0.02))
bt = None
for th in [x / 1000 for x in range(300, 996, 5)]:
    g = abs(ex.filter(pl.col("p") >= th).height / n_fr - k_ui)
    if bt is None or g < bt[0]:
        bt = (g, th)
sel_fr = ex.filter(pl.col("p") >= bt[1])
print(f"[test] US/India matches/S1 {k_ui:.3f} (final file 3.332); France threshold {bt[1]:.3f} -> {sel_fr.height / n_fr:.3f}")

ids = cmap.join(pp.scan_norm("test", "s1").select(pl.col("idx").alias("s1_idx"), pl.col("entity_id").alias("source1_entity_id")).collect(), on="s1_idx")
base = pl.read_csv(ROOT / "output" / "SUBMIT_THIS" / "matching_results.tsv", separator="\t", quote_char=None, infer_schema=False)


def write(sel, countries, name):
    j = pp._join_ids(sel).join(ids.select("s1_idx", "source1_entity_id"), on="s1_idx").select("source1_entity_id", pl.col("ids").alias("f"))
    rows = ids.filter(pl.col("country_n").is_in(countries))["source1_entity_id"]
    out = (base.join(j, on="source1_entity_id", how="left", maintain_order="left")
           .with_columns(pl.when(pl.col("source1_entity_id").is_in(rows.implode())).then(pl.col("f").fill_null(""))
                         .otherwise(pl.col("matched_entity_ids").fill_null("")).alias("matched_entity_ids")).drop("f"))
    dst = SUB / f"PROBE_{name}_matching_results.tsv"
    out.write_csv(dst, separator="\t", quote_style="never")
    r = subprocess.run([sys.executable, str(ROOT / "student_resource" / "utils" / "validate_submission.py"), "--matching", str(dst),
                        "--candidate", str(ROOT / "output" / "SUBMIT_THIS" / "candidate_pairs.tsv"), "--test-dir", str(ROOT / "student_resource" / "dataset" / "test"),
                        "--check-ids"], capture_output=True, text=True, encoding="utf-8", errors="replace")
    canon = lambda df, n: df.select("source1_entity_id", pl.col("matched_entity_ids").fill_null("").str.split(",").list.eval(pl.element().filter(pl.element() != "")).list.sort().list.join(",").alias(n))  # noqa: E731
    cmpj = canon(base, "a").join(canon(out, "b"), on="source1_entity_id").join(ids.select("source1_entity_id", "country_n"), on="source1_entity_id")
    ch = cmpj.group_by(pl.col("country_n") == "france").agg((pl.col("a") != pl.col("b")).mean().round(4).alias("changed")).rows()
    print(f"{dst.name}: validator {'PASS' if r.returncode == 0 else 'FAIL'}; lists changed vs final (is_france, share): {ch}")
    return out


write(sel_ui, ["us", "india"], "stage2_usin")
write(pl.concat([sel_ui, sel_fr]), ["us", "india", "france"], "stage2_all")
