"""Rule-based fallback for the unlabelled country (proxy: +0.0023 US->India, +0.0015 India->US): France pairs in the
model's uncertain band [0.2, 0.8) are decided by a strict string rule (core name token-set >= 95, address token-set
>= 90, same house number, no legal-form conflict) instead of the score; then exclusivity and the usual France
threshold (matches per S1 = the labelled countries'). US/India rows unchanged from SUBMIT_THIS."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
import pipeline as pp  # noqa: E402

ROOT = Path(r"D:\amazon-ml")
R = pp.RUNS_DIR
cm = pp.scan_norm("test", "s1").select(pl.col("idx").alias("s1_idx"), "country_n", pl.col("entity_id").alias("source1_entity_id")).collect()
fr = cm.filter(pl.col("country_n") == "france")
sc = pl.read_parquet(sorted(R.glob("*-v10sfr_dense"))[-1] / "pred" / "scored-*.parquet").join(fr.select("s1_idx"), on="s1_idx")
ft = (pl.scan_parquet(sorted((sorted(R.glob("*-v10testsrc"))[-1] / "test_feats").glob("part-*.parquet")))
      .select("s1_idx", "cand_idx", "nc_tset", "ad_tset", "hn_eq", "legal_conflict").collect())
sc = sc.join(ft, on=["s1_idx", "cand_idx"], how="left")
band = (pl.col("p") >= 0.2) & (pl.col("p") < 0.8)
strict = (pl.col("nc_tset") >= 95) & (pl.col("ad_tset") >= 90) & (pl.col("hn_eq") == 1) & (pl.col("legal_conflict") == 0)
n_band = sc.filter(band).height
n_acc = sc.filter(band & strict).height
sc2 = sc.with_columns(pl.when(band).then(pl.when(strict).then(0.99).otherwise(0.01)).otherwise(pl.col("p")).cast(pl.Float32).alias("p"))
base = pl.read_csv(ROOT / "output" / "SUBMIT_THIS" / "matching_results.tsv", separator="\t", quote_char=None, infer_schema=False)
k_lab = (base.join(cm.filter(pl.col("country_n") != "france").select("source1_entity_id"), on="source1_entity_id")["matched_entity_ids"]
         .fill_null("").str.split(",").list.eval(pl.element().filter(pl.element() != "")).list.len().mean())
ex = pp.exclusive(sc2.filter(pl.col("p") >= 0.02))
ths = np.arange(0.3, 0.996, 0.005)
ks = np.array([ex.filter(pl.col("p") >= t).height / fr.height for t in ths])
th = float(ths[np.argmin(np.abs(ks - k_lab))])
sel = ex.filter(pl.col("p") >= th)
ids = pp._join_ids(sel).join(fr.select("s1_idx", "source1_entity_id"), on="s1_idx").select("source1_entity_id", pl.col("ids").alias("f"))
out = (base.join(ids, on="source1_entity_id", how="left", maintain_order="left")
       .with_columns(pl.when(pl.col("source1_entity_id").is_in(fr["source1_entity_id"].implode())).then(pl.col("f").fill_null(""))
                     .otherwise(pl.col("matched_entity_ids").fill_null("")).alias("matched_entity_ids")).drop("f"))
dst = ROOT / "output" / "submissions" / "PROBE_france_rule_matching_results.tsv"
out.write_csv(dst, separator="\t", quote_style="never")
rc = subprocess.run([sys.executable, str(ROOT / "student_resource" / "utils" / "validate_submission.py"), "--matching", str(dst),
                     "--candidate", str(ROOT / "output" / "SUBMIT_THIS" / "candidate_pairs.tsv"),
                     "--test-dir", str(ROOT / "student_resource" / "dataset" / "test"), "--check-ids"], capture_output=True, text=True,
                    encoding="utf-8", errors="replace").returncode
rd = lambda df, n: df.select("source1_entity_id", pl.col("matched_entity_ids").fill_null("").str.split(",").list.eval(pl.element().filter(pl.element() != "")).list.sort().list.join(",").alias(n))  # noqa: E731
chg = rd(base, "a").join(rd(out, "b"), on="source1_entity_id").join(fr.select("source1_entity_id"), on="source1_entity_id")
print(f"France band pairs {n_band:,}: rule accepts {n_acc:,} ({n_acc / max(n_band, 1):.1%}), rejects the rest")
print(f"France threshold {th:.3f} -> {sel.height / fr.height:.3f} matches/S1 (labelled {k_lab:.3f}); French lists changed "
      f"{(chg['a'] != chg['b']).mean():.4f}; validator {'PASS' if rc == 0 else 'FAIL'} -> {dst.name}")
