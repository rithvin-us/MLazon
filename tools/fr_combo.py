"""Final France rows: scores of the pseudo-label-adapted model (v10sfr_dense: v10s + structure-based French
pseudo-labels + L12 CE, test-density run) with the rule-based fallback in its uncertain band (p in [0.2, 0.8): accept
iff core name token-set >= 95, address token-set >= 90, same house number, no legal-form conflict, else reject), then
exclusivity and the count-matched France threshold (matches/S1 = labelled countries', same grid as
pipeline.shape_matched_selection). US/India rows unchanged from the LB-verified base file.
  python fr_combo.py [SRC_RUN=v10sfr_dense] [RULE=1] [TAG=final]
SRC_RUN=v10sfr2_dense (second pseudo-label round) gives the "combo" file; the proxy (loco_combo.py) prefers round 1.
TAG=check writes nothing and compares the French rows with PROBE_france_v10sfr2 (RULE=0 must reproduce it exactly)."""
import os
import subprocess
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
import pipeline as pp  # noqa: E402

ROOT = Path(r"D:\amazon-ml")
R = pp.RUNS_DIR
SUB = ROOT / "output" / "submissions"
BEST = SUB / "BEST_LB0.975968_mix_v10sfr_matching_results.tsv"  # LB-verified
BASE = Path(os.environ.get("FR_BASE", BEST))  # US/India rows come from here (FR_BASE: another mix output)
SRC_RUN = sys.argv[1] if len(sys.argv) > 1 else "v10sfr_dense"
RULE = sys.argv[2] != "0" if len(sys.argv) > 2 else True
TAG = sys.argv[3] if len(sys.argv) > 3 else "final"

cm = pp.scan_norm("test", "s1").select(pl.col("idx").alias("s1_idx"), "country_n", pl.col("entity_id").alias("source1_entity_id")).collect()
fr = cm.filter(pl.col("country_n") == "france")
sc = (pl.read_parquet(sorted(R.glob(f"*-{SRC_RUN}"))[-1] / "pred" / "scored-*.parquet")
      .select("s1_idx", "cand_idx", "cid", "p").join(fr.select("s1_idx"), on="s1_idx"))
st = {}
if RULE:
    ft = (pl.scan_parquet(sorted((sorted(R.glob("*-v10testsrc"))[-1] / "test_feats").glob("part-*.parquet")))
          .select("s1_idx", "cand_idx", "nc_tset", "ad_tset", "hn_eq", "legal_conflict")
          .join(fr.select("s1_idx").lazy(), on="s1_idx").collect())
    sc = sc.join(ft, on=["s1_idx", "cand_idx"], how="left")
    band = (pl.col("p") >= 0.2) & (pl.col("p") < 0.8)
    strict = (pl.col("nc_tset") >= 95) & (pl.col("ad_tset") >= 90) & (pl.col("hn_eq") == 1) & (pl.col("legal_conflict") == 0)
    st = {"band": sc.filter(band).height, "accept": sc.filter(band & strict).height,
          "band_no_feats": sc.filter(band & pl.col("nc_tset").is_null()).height}
    sc = sc.with_columns(pl.when(band).then(pl.when(strict).then(0.99).otherwise(0.01)).otherwise(pl.col("p")).cast(pl.Float32).alias("p"))

base = pl.read_csv(BASE, separator="\t", quote_char=None, infer_schema=False)
n_ids = pl.col("matched_entity_ids").fill_null("").str.split(",").list.eval(pl.element().filter(pl.element() != "")).list.len()
k_lab = base.join(cm.filter(pl.col("country_n") != "france").select("source1_entity_id"), on="source1_entity_id").select(n_ids.mean()).item()
k_lab = float(os.environ.get("FR_K", k_lab))  # FR_K: force the pipeline's own target (sanity check)
ex = pp.exclusive(sc.filter(pl.col("p") >= 0.02))
best = None
for t in [x / 1000 for x in range(300, 996, 5)]:  # same grid and tie-break as pipeline.shape_matched_selection
    gap = abs(ex.filter(pl.col("p") >= t).height / fr.height - k_lab)
    if best is None or gap < best[0]:
        best = (gap, t)
th = best[1]
sel = ex.filter(pl.col("p") >= th)
ids = pp._join_ids(sel).join(fr.select("s1_idx", "source1_entity_id"), on="s1_idx").select("source1_entity_id", pl.col("ids").alias("f"))
out = (base.join(ids, on="source1_entity_id", how="left", maintain_order="left")
       .with_columns(pl.when(pl.col("source1_entity_id").is_in(fr["source1_entity_id"].implode())).then(pl.col("f").fill_null(""))
                     .otherwise(pl.col("matched_entity_ids").fill_null("")).alias("matched_entity_ids")).drop("f"))


def canon(df, n):  # order-insensitive list per S1
    return df.select("source1_entity_id", pl.col("matched_entity_ids").fill_null("").str.split(",")
                     .list.eval(pl.element().filter(pl.element() != "")).list.sort().list.join(",").alias(n))


def changed(other, only_fr=True):
    j = canon(out, "a").join(canon(other, "b"), on="source1_entity_id")
    j = j.join(fr.select("source1_entity_id"), on="source1_entity_id", how="semi" if only_fr else "anti")
    return float((j["a"] != j["b"]).mean())


rd = lambda p: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False)  # noqa: E731
print(f"{SRC_RUN} rule={RULE} {st} | France thr {th:.3f} -> {sel.height / fr.height:.3f} matches/S1 (labelled {k_lab:.3f})")
print(f"rows {out.height:,} (base {base.height:,}); US/India lists changed vs base {changed(base, False):.6f}, "
      f"vs LB-verified file {changed(rd(BEST), False):.4f}")
print(f"French lists changed: vs base {changed(base):.4f}, vs PROBE_v10sfr2 {changed(rd(SUB / 'PROBE_france_v10sfr2_matching_results.tsv')):.4f}, "
      f"vs PROBE_rule {changed(rd(SUB / 'PROBE_france_rule_matching_results.tsv')):.4f}")
if TAG != "check":
    dst = SUB / (f"PROBE_{TAG}_matching_results.tsv" if "FR_BASE" in os.environ else f"PROBE_france_{TAG}_matching_results.tsv")
    out.write_csv(dst, separator="\t", quote_style="never")
    r = subprocess.run([sys.executable, str(ROOT / "student_resource" / "utils" / "validate_submission.py"), "--matching", str(dst),
                        "--candidate", str(ROOT / "output" / "SUBMIT_THIS" / "candidate_pairs.tsv"),
                        "--test-dir", str(ROOT / "student_resource" / "dataset" / "test"), "--check-ids"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(f"validator {'PASS' if r.returncode == 0 else 'FAIL'} -> {dst.name}")
    print("\n".join(r.stdout.strip().splitlines()[-3:]))
