"""bge-reranker-base (MIT, 278M, multilingual) as the stage-3 cross-encoder.
ce-train on v10src train pairs -> ce-apply for v10 (US/India source) and v10s (France source) -> redecide -> mix ->
validate. SUBMIT_THIS replaced only if the new mix beats the current one on test-density validation (US/India) and
plain validation. Log: runs/bge_chain2.log"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"D:\amazon-ml")
PY = str(ROOT / ".venv311" / "Scripts" / "python.exe")
PIPE = str(ROOT / "code" / "business_entity_resolution" / "src" / "pipeline.py")
RUNS = ROOT / "runs"
LOG = RUNS / "bge_chain2.log"
BLOCK = ["--max-df", "600", "--rr-tau", "0.002", "--rr-min", "3"]
CE_BGE = str(ROOT / "models" / "ce_bge_base")
CE = str(ROOT / "models" / "ce_20260926-141000-v9") + "," + CE_BGE
N_PAIRS = sys.argv[1] if len(sys.argv) > 1 else "1000000"


def log(m):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {m}\n")


def newest(name):
    ds = sorted(d for d in RUNS.glob(f"*-{name}") if d.is_dir())
    return ds[-1] if ds else None


def metrics(name):
    d = newest(name)
    return json.loads((d / "metrics.json").read_text()) if d and (d / "metrics.json").exists() else {}


def run(tag, args):
    t = time.time()
    log(f"start {tag}: {' '.join(args)}")
    with open(RUNS / f"b_{tag}.out", "w", encoding="utf-8") as out:
        rc = subprocess.run([PY, PIPE, *args], cwd=ROOT, stdout=out, stderr=subprocess.STDOUT,
                            env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"}).returncode
    log(f"done {tag} rc={rc} in {time.time() - t:.0f}s")
    return rc == 0


v10src = newest("v10src").name
while not (Path(CE_BGE) / "model.safetensors").exists() or any("ce-train" in l for l in __import__("subprocess").run(["powershell", "-c", "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | % { $_.CommandLine }"], capture_output=True, text=True).stdout.splitlines()):
    time.sleep(30)
if False:
    ok = run("bge_train", ["ce-train", "--run", v10src, "--name", "bge_train", "--ce-base", "BAAI/bge-reranker-base",
                           "--ce-pairs", N_PAIRS, "--ce-epochs", "1", "--ce-dir", CE_BGE])
    if not ok:
        sys.exit(1)
for src, test, tag in (("v10", "v10_test", "v10_bge2"), ("v10s", "v10s_test", "v10s_bge2")):
    if run(tag, ["ce-apply", "--run", newest(src).name, "--feats-run", newest(test).name, "--ce-dir", CE, "--name", tag, *BLOCK]):
        log(f"{tag} val {metrics(tag).get('val_f05')} (L12: {metrics(src + '_ce').get('val_f05')})")
        run(f"{tag}_dense", ["redecide", "--run", newest(tag).name, "--name", f"{tag}_dense", *BLOCK])
        md = metrics(f"{tag}_dense")
        log(f"{tag}_dense dense before {md.get('val_f05_dense_before')} after {md.get('val_f05_dense')} "
            f"(L12 {metrics(src + '_ce_dense').get('val_f05_dense')})")
if newest("v10_bge2_dense") and newest("v10s_bge2_dense"):
    run("mix_bge2", ["mix", "--run", newest("v10_bge2_dense").name, "--unlabelled-from", newest("v10s_bge2_dense").name, "--name", "mix_bge2"])
    d = newest("mix_bge2") / "output"
    r = subprocess.run([sys.executable, str(ROOT / "student_resource" / "utils" / "validate_submission.py"), "--matching",
                        str(d / "matching_results.tsv"), "--candidate", str(d / "candidate_pairs.tsv"),
                        "--test-dir", str(ROOT / "student_resource" / "dataset" / "test"), "--check-ids"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    ok = r.returncode == 0
    new_dense, old_dense = metrics("v10_bge2_dense").get("val_f05_dense"), metrics("v10_ce_dense").get("val_f05_dense")
    new_val, old_val = metrics("v10_bge2").get("val_f05"), metrics("v10_ce").get("val_f05")
    log(f"mix_bge validator {'PASS' if ok else 'FAIL'}; US/India dense {new_dense} vs {old_dense}; plain val {new_val} vs {old_val}")
    shutil.copy(d / "matching_results.tsv", ROOT / "output" / "submissions" / "mix_bge2_matching_results.tsv")
    if ok and new_dense and old_dense and new_dense > old_dense + 0.0002 and new_val >= old_val:
        dst = ROOT / "output" / "SUBMIT_THIS"
        shutil.copy(d / "matching_results.tsv", dst / "matching_results.tsv")
        shutil.copy(d / "candidate_pairs.tsv", dst / "candidate_pairs.tsv")
        (dst / "WHAT_IS_THIS.txt").write_text(
            f"mix_bge2: US/India rows v10 + stacked MiniLM-L12 and bge-reranker-base cross-encoders + test-density decision, France rows v10s + same\n"
            f"US/India test-density val {new_dense} (L12 mix {old_dense}); plain val {new_val} (L12 {old_val})\n"
            f"estimated LB change vs previous SUBMIT_THIS {0.85 * (new_dense - old_dense):+.5f} (US/India part only)\n"
            f"validator --check-ids PASS\n{time.ctime()}\n", encoding="utf-8")
        log("SUBMIT_THIS <- mix_bge")
    else:
        log("SUBMIT_THIS unchanged")
log("bge chain finished")

