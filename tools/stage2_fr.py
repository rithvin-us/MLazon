"""Stage 2 for the unlabelled country at ITS density: labelled US/India validation (France model's stacked scores,
v10sfr_bge2) is augmented until its pairs per S1 per p band match the FRENCH test scores (not US/India's), stage 2 is
trained there (OOF by S1 for the check) and applied to the French rows; France threshold count-matched to the US/India
rows of the stage-2 file. Check printed first: stacked p vs stage 2 on the France-density validation.
-> output/submissions/PROBE_stage2_frdense_matching_results.tsv (US/India rows from PROBE_stage2_usin)"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
sys.path.insert(0, str(Path(__file__).resolve().parent))
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


v = pl.read_parquet(sorted(R.glob("*-v10sfr_bge2"))[-1] / "val_scored_ce.parquet").filter(pl.col("p") >= 0.02)
vt = pl.read_parquet(sorted(R.glob("*-v10"))[-1] / "val_truth.parquet")
tvc = vt.select("s1_idx", "n_true")
cmap = pp.scan_norm("test", "s1").select(pl.col("idx").alias("s1_idx"), "country_n").collect()
fr = pl.read_parquet(sorted(R.glob("*-v10sfr_bge2_dense"))[-1] / "pred" / "scored-*.parquet").join(cmap, on="s1_idx").filter(pl.col("country_n") == "france")
n_fr = cmap.filter(pl.col("country_n") == "france").height
# France's pairs per S1 per band, used as the target for every labelled country (density_augment logic, other target)
edges = np.array(pp.DENSITY_BANDS)
tb = np.searchsorted(edges, fr.filter(pl.col("p") >= edges[0])["p"].to_numpy(), side="right") - 1
target = {b: float((tb == b).sum()) / n_fr for b in range(len(edges) - 1)}
vv = v.join(vt.select("s1_idx", "country_n"), on="s1_idx").filter(pl.col("p") >= edges[0])
vv = vv.with_columns(pl.Series("b", np.searchsorted(edges, vv["p"].to_numpy(), side="right") - 1))
n_val = dict(vt.group_by("country_n").len().iter_rows())
rng = np.random.default_rng(42)
extra, factors = [], {}
for c in sorted(n_val):
    for b in range(len(edges) - 1):
        vb = vv.filter((pl.col("country_n") == c) & (pl.col("b") == b))
        pos, negr = vb["label"].sum() / n_val[c], vb.filter(pl.col("label") == 0)
        neg = negr.height / n_val[c]
        f = 1.0 if neg <= 0 else float(np.clip((target[b] - pos) / neg, 1.0, 6.0))
        factors[f"{c}:{edges[b]:.2f}"] = round(f, 2)
        if f > 1.0 and negr.height:
            k = np.floor(f - 1.0) + (rng.random(negr.height) < (f - 1.0 - np.floor(f - 1.0)))
            extra.append(negr.select("s1_idx", "label", "p").with_columns(pl.Series("k", k.astype(np.int64))).filter(pl.col("k") > 0)
                         .with_columns(pl.int_ranges(0, pl.col("k")).alias("j")).explode("j").select("s1_idx", "label", "p"))
ex = pl.concat(extra).with_row_index("r")
ex = ex.select("s1_idx", (-(pl.col("r").cast(pl.Int64) + 1)).alias("cand_idx"), pl.col("label").cast(v["label"].dtype),
               (pl.col("p") + pl.Series(rng.uniform(-0.01, 0.01, ex.height).astype(np.float32))).clip(0.02, 1.0).cast(v["p"].dtype).alias("p"))
aug = pl.concat([v.select("s1_idx", "cand_idx", "label", "p"), ex])
print(f"France-density factors {factors}; val rows {v.height:,} -> {aug.height:,}")
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
b0 = max(pp.tune_decision(s2v.select("s1_idx", "cand_idx", "label", "p"), tvc, 0.02, _Q()).values(), key=lambda z: z["f05"])
b1 = max(pp.tune_decision(s2v.select("s1_idx", "cand_idx", "label", pl.col("p2").alias("p")), tvc, 0.02, _Q()).values(), key=lambda z: z["f05"])
print(f"[France-density val] stacked p {b0['f05']:.5f} ({b0['mode']} {b0['param']}) -> stage 2 {b1['f05']:.5f} ({b1['mode']} {b1['param']}) ({b1['f05'] - b0['f05']:+.5f})")
final = xgb.train(PRM, xgb.DMatrix(dv.select(COLS).to_numpy(), dv["label"].to_numpy(), feature_names=COLS), 400)
t = feats(fr.select("s1_idx", "cand_idx", "cid", "p"))
t = t.with_columns(pl.Series("p2", final.predict(xgb.DMatrix(t.select(COLS).to_numpy(), feature_names=COLS)), dtype=pl.Float32))
base = pl.read_csv(SUB / "PROBE_stage2_usin_matching_results.tsv", separator="\t", quote_char=None, infer_schema=False)
ids = pp.scan_norm("test", "s1").select(pl.col("idx").alias("s1_idx"), pl.col("entity_id").alias("source1_entity_id"), "country_n").collect()
n_ids = pl.col("matched_entity_ids").fill_null("").str.split(",").list.eval(pl.element().filter(pl.element() != "")).list.len()
k_ui = base.join(ids.filter(pl.col("country_n") != "france").select("source1_entity_id"), on="source1_entity_id").select(n_ids.mean()).item()
exf = pp.exclusive(t.select("s1_idx", "cand_idx", "cid", pl.col("p2").alias("p")).filter(pl.col("p") >= 0.02))
bt = None
for th in [x / 1000 for x in range(300, 996, 5)]:
    g = abs(exf.filter(pl.col("p") >= th).height / n_fr - k_ui)
    if bt is None or g < bt[0]:
        bt = (g, th)
sel = exf.filter(pl.col("p") >= bt[1])
j = pp._join_ids(sel).join(ids.select("s1_idx", "source1_entity_id"), on="s1_idx").select("source1_entity_id", pl.col("ids").alias("f"))
frows = ids.filter(pl.col("country_n") == "france")["source1_entity_id"]
out = (base.join(j, on="source1_entity_id", how="left", maintain_order="left")
       .with_columns(pl.when(pl.col("source1_entity_id").is_in(frows.implode())).then(pl.col("f").fill_null(""))
                     .otherwise(pl.col("matched_entity_ids").fill_null("")).alias("matched_entity_ids")).drop("f"))
dst = SUB / "PROBE_stage2_frdense_matching_results.tsv"
out.write_csv(dst, separator="\t", quote_style="never")
r = subprocess.run([sys.executable, str(ROOT / "student_resource" / "utils" / "validate_submission.py"), "--matching", str(dst),
                    "--candidate", str(ROOT / "output" / "SUBMIT_THIS" / "candidate_pairs.tsv"), "--test-dir", str(ROOT / "student_resource" / "dataset" / "test"),
                    "--check-ids"], capture_output=True, text=True, encoding="utf-8", errors="replace")
canon = lambda df, n: df.select("source1_entity_id", pl.col("matched_entity_ids").fill_null("").str.split(",").list.eval(pl.element().filter(pl.element() != "")).list.sort().list.join(",").alias(n))  # noqa: E731
cj = canon(base, "a").join(canon(out, "b"), on="source1_entity_id").join(ids.select("source1_entity_id", "country_n"), on="source1_entity_id")
print(f"France threshold {bt[1]:.3f} -> {sel.height / n_fr:.3f} matches/S1 (US/India {k_ui:.3f}); French lists changed vs final "
      f"{cj.filter(pl.col('country_n') == 'france').select((pl.col('a') != pl.col('b')).mean()).item():.4f}; validator {'PASS' if r.returncode == 0 else 'FAIL'} -> {dst.name}")
