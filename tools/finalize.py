"""After bge_chain2: France 0.98x probe file on top of SUBMIT_THIS, zip rebuilt from SUBMIT_THIS, report. Log: runs/finalize.log"""
import hashlib
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(r"D:\amazon-ml")
RUNS = ROOT / "runs"
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution" / "src"))
LOG = RUNS / "finalize.log"


def log(m):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {m}\n")


while "bge chain finished" not in ((RUNS / "bge_chain2.log").read_text(encoding="utf-8") if (RUNS / "bge_chain2.log").exists() else ""):
    time.sleep(30)
import pipeline as pp  # noqa: E402

sub = ROOT / "output" / "SUBMIT_THIS"
what = (sub / "WHAT_IS_THIS.txt").read_text(encoding="utf-8")
log(f"SUBMIT_THIS: {what.splitlines()[0]}")
# France source of the current pick
fr_src = "v10s_bge2_dense" if what.startswith("mix_bge2") else "v10s_ce_dense"
d = sorted(RUNS.glob(f"*-{fr_src}"))[-1]
cm = pp.scan_norm("test", "s1").select(pl.col("idx").alias("s1_idx"), "country_n", pl.col("entity_id").alias("source1_entity_id")).collect()
fr = cm.filter(pl.col("country_n") == "france")
us_in = cm.filter(pl.col("country_n") != "france")
base = pl.read_csv(sub / "matching_results.tsv", separator="\t", quote_char=None, infer_schema=False)
k_lab = (base.join(us_in.select("source1_entity_id"), on="source1_entity_id")["matched_entity_ids"].fill_null("")
         .str.split(",").list.eval(pl.element().filter(pl.element() != "")).list.len().mean())
sc = pp.exclusive(pl.read_parquet(d / "pred" / "scored-*.parquet").join(fr.select("s1_idx"), on="s1_idx").filter(pl.col("p") >= 0.02))
ths = np.arange(0.3, 0.996, 0.005)
ks = np.array([sc.filter(pl.col("p") >= t).height / fr.height for t in ths])
for r in (0.98, 0.96):
    th = float(ths[np.argmin(np.abs(ks - r * k_lab))])
    sel = sc.filter(pl.col("p") >= th)
    ids = pp._join_ids(sel).join(fr.select("s1_idx", "source1_entity_id"), on="s1_idx").select("source1_entity_id", pl.col("ids").alias("f"))
    out = (base.join(ids, on="source1_entity_id", how="left", maintain_order="left")
           .with_columns(pl.when(pl.col("source1_entity_id").is_in(fr["source1_entity_id"].implode())).then(pl.col("f").fill_null(""))
                         .otherwise(pl.col("matched_entity_ids").fill_null("")).alias("matched_entity_ids")).drop("f"))
    p = ROOT / "output" / "submissions" / f"PROBE_france_k{int(r * 100)}_matching_results.tsv"
    out.write_csv(p, separator="\t", quote_style="never")
    rc = subprocess.run([sys.executable, str(ROOT / "student_resource" / "utils" / "validate_submission.py"), "--matching", str(p),
                         "--candidate", str(sub / "candidate_pairs.tsv"), "--test-dir", str(ROOT / "student_resource" / "dataset" / "test"),
                         "--check-ids"], capture_output=True, text=True, encoding="utf-8", errors="replace").returncode
    log(f"probe France {r}x: thr {th:.3f}, France k {sel.height / fr.height:.3f} (labelled k {k_lab:.3f}), validator {'PASS' if rc == 0 else 'FAIL'} -> {p.name}")
# zip from SUBMIT_THIS
zp = ROOT / "Techiva_submission.zip"
if zp.exists():
    shutil.copy(zp, ROOT / "Techiva_submission_prev.zip")
src_dir = ROOT / "code" / "business_entity_resolution"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for fn in ("matching_results.tsv", "candidate_pairs.tsv"):
        z.write(sub / fn, f"output/{fn}")
    for f in sorted((src_dir / "src").glob("*.py")):
        z.write(f, f"code/business_entity_resolution/src/{f.name}")
    for fn in ("README.md", "requirements.txt"):
        z.write(src_dir / fn, f"code/business_entity_resolution/{fn}")
    z.write(ROOT / "docs" / "Documentation_template.md", "Documentation_template.md")
zz = zipfile.ZipFile(zp)
same = hashlib.sha256(zz.read("output/matching_results.tsv")).hexdigest() == hashlib.sha256((sub / "matching_results.tsv").read_bytes()).hexdigest()
log(f"zip rebuilt: {len(zz.namelist())} files, matching == SUBMIT_THIS: {same}, {zp.stat().st_size / 1e6:.0f} MB")
log("finalize done")
