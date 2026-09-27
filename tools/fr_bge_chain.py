"""France rows with both cross-encoders, no preprocessing (saved test features): ce-apply (MiniLM-L12 + bge-reranker-base
stacked) on the round-1 pseudo-label model v10sfr -> redecide -> fr_combo (rule fallback; US/India rows from mix_bge2;
France matches/S1 pinned to the final's 3.343) -> output/submissions/PROBE_bge_all_matching_results.tsv
Log: runs/fr_bge_chain.log"""
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(r"D:\amazon-ml")
PY = str(ROOT / ".venv311" / "Scripts" / "python.exe")
PIPE = str(ROOT / "code" / "business_entity_resolution" / "src" / "pipeline.py")
RUNS = ROOT / "runs"
LOG = RUNS / "fr_bge_chain.log"
BLOCK = ["--max-df", "600", "--rr-tau", "0.002", "--rr-min", "3"]
CE = f"{ROOT / 'models' / 'ce_20260926-141000-v9'},{ROOT / 'models' / 'ce_bge_base'}"
MIX_BGE2 = RUNS / "20260927-124708-mix_bge2" / "output" / "matching_results.tsv"


def log(m):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {m}\n")


def newest(name):
    ds = sorted(d for d in RUNS.glob(f"*-{name}") if d.is_dir())
    return ds[-1] if ds else None


def run(tag, cmd, extra_env=None):
    t = time.time()
    log(f"start {tag}: {' '.join(cmd[1:])}")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", **(extra_env or {})}
    with open(RUNS / f"fb_{tag}.out", "w", encoding="utf-8") as out:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, env=env).returncode
    log(f"done {tag} rc={rc} in {time.time() - t:.0f}s")
    if rc != 0:
        log("fr bge chain FAILED")
        raise SystemExit(1)


run("v10sfr_bge2", [PY, PIPE, "ce-apply", "--run", newest("v10sfr").name, "--feats-run", newest("v10sfr_test").name,
                    "--ce-dir", CE, "--name", "v10sfr_bge2", *BLOCK])
log("ce metrics: " + (newest("v10sfr_bge2") / "metrics.json").read_text(encoding="utf-8").replace("\n", " ")[:600])
run("v10sfr_bge2_dense", [PY, PIPE, "redecide", "--run", newest("v10sfr_bge2").name, "--name", "v10sfr_bge2_dense", *BLOCK])
for line in (newest("v10sfr_bge2_dense") / "log.txt").read_text(encoding="utf-8").splitlines():
    if "dense" in line or "unlabelled country" in line:
        log("  " + line)
run("combo", [PY, str(ROOT / "tools" / "fr_combo.py"), "v10sfr_bge2_dense", "1", "bge_all"],
    {"FR_BASE": str(MIX_BGE2), "FR_K": "3.343"})
log("combo: " + " | ".join((RUNS / "fb_combo.out").read_text(encoding="utf-8").strip().splitlines()[:4]))
log("fr bge chain finished")
