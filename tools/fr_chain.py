"""France retrain with structure-based French pseudo-labels -> France-only probe file.
retrain (v10s features + French pseudo parts) -> rescore test -> L12 CE -> test-density decision -> mix (US/India rows
from v10_ce_dense, France rows from the new model) -> validator -> output/submissions/PROBE_france_pseudo_*.tsv
  python fr_chain.py <frpseudo run name> [tag]      Log: runs/fr_chain.log"""
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"D:\amazon-ml")
PY = str(ROOT / ".venv311" / "Scripts" / "python.exe")
PIPE = str(ROOT / "code" / "business_entity_resolution" / "src" / "pipeline.py")
RUNS = ROOT / "runs"
LOG = RUNS / "fr_chain.log"
BLOCK = ["--max-df", "600", "--rr-tau", "0.002", "--rr-min", "3"]
DROP = "n_rep_idf,n_miss_idf,a_idf_jacc,a_idf_miss1,a_idf_miss2,a_rep,a_rep_idf,a_miss_idf,a_idf_rank"
PS = sys.argv[1]
TAG = sys.argv[2] if len(sys.argv) > 2 else "v10sfr"


def log(m):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {m}\n")


def newest(name):
    ds = sorted(d for d in RUNS.glob(f"*-{name}") if d.is_dir())
    return ds[-1] if ds else None


def run(tag, args):
    t = time.time()
    log(f"start {tag}: {' '.join(args)}")
    with open(RUNS / f"f_{tag}.out", "w", encoding="utf-8") as out:
        rc = subprocess.run([PY, PIPE, *args], cwd=ROOT, stdout=out, stderr=subprocess.STDOUT,
                            env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"}).returncode
    log(f"done {tag} rc={rc} in {time.time() - t:.0f}s")
    if rc != 0:
        sys.exit(1)


run(TAG, ["retrain", "--run", newest("v10src").name, "--name", TAG, "--rounds", "7000", "--drop-feats", DROP,
          "--extra-parts", str(RUNS / PS), *BLOCK])
run(f"{TAG}_test", ["rescore", "--run", newest(TAG).name, "--feats-run", newest("v10testsrc").name, "--name", f"{TAG}_test", *BLOCK])
run(f"{TAG}_ce", ["ce-apply", "--run", newest(TAG).name, "--feats-run", newest(f"{TAG}_test").name,
                  "--ce-dir", str(ROOT / "models" / "ce_20260926-141000-v9"), "--name", f"{TAG}_ce", *BLOCK])
run(f"{TAG}_dense", ["redecide", "--run", newest(f"{TAG}_ce").name, "--name", f"{TAG}_dense", *BLOCK])
run(f"mix_{TAG}", ["mix", "--run", newest("v10_ce_dense").name, "--unlabelled-from", newest(f"{TAG}_dense").name, "--name", f"mix_{TAG}"])
d = newest(f"mix_{TAG}") / "output"
r = subprocess.run([sys.executable, str(ROOT / "student_resource" / "utils" / "validate_submission.py"), "--matching", str(d / "matching_results.tsv"),
                    "--candidate", str(d / "candidate_pairs.tsv"), "--test-dir", str(ROOT / "student_resource" / "dataset" / "test"), "--check-ids"],
                   capture_output=True, text=True, encoding="utf-8", errors="replace")
dst = ROOT / "output" / "submissions" / f"PROBE_france_{TAG}_matching_results.tsv"
shutil.copy(d / "matching_results.tsv", dst)
log(f"validator {'PASS' if r.returncode == 0 else 'FAIL'} -> {dst.name}")
fl = [l for l in (newest(f"{TAG}_dense") / "log.txt").read_text(encoding="utf-8").splitlines() if "unlabelled country" in l]
log(f"France decision: {fl[-1] if fl else '?'}")
log("fr chain finished")
