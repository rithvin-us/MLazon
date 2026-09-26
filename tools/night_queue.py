"""Unattended night queue (2026-09-26/27). France-aware v10 features (country-IDF token agreement) on the v9 candidates.

  1  augment test features (v9_test -> v10testsrc)                       CPU
  2  retrain v10 on augmented train features, val + competitor tuning     GPU
  3  rescore test, 4 cross-encoder + France shape rule  -> v10_ce         GPU
  5  unseen-country proxy (train US -> India, India -> US)                GPU
  6  crowd-weighted variant v10w (+ rescore + CE)                          GPU
  7  ensemble v9 + v10 (+ CE)                                              GPU
  8  validate every candidate (--check-ids), pick by estimated LB, SUBMIT_THIS, report
Every step is guarded; a failure is logged and the queue moves on. Log: runs/night_queue.log
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import polars as pl

ROOT = Path(r"D:\amazon-ml")
PY = str(ROOT / ".venv311" / "Scripts" / "python.exe")
PIPE = str(ROOT / "code" / "business_entity_resolution" / "src" / "pipeline.py")
RUNS = ROOT / "runs"
LOG = RUNS / "night_queue.log"
BLOCK = ["--max-df", "600", "--rr-tau", "0.002", "--rr-min", "3"]
V9, V9TEST, V9CE = "20260926-141000-v9", "20260926-151110-v9_test", "20260926-155941-v9_ce"
V10SRC = sorted(d.name for d in RUNS.glob("*-v10src"))[-1]
CE_DIR = str(ROOT / "models" / f"ce_{V9}")
today = time.strftime("%Y-%m-%d")
T = lambda hm, day=None: time.mktime(time.strptime(f"{day or today} {hm}", "%Y-%m-%d %H:%M"))  # noqa: E731
tomorrow = time.strftime("%Y-%m-%d", time.localtime(time.time() + 86400)) if time.localtime().tm_hour >= 12 else today
LAST_GPU_START = T("03:15", tomorrow)   # no new 40-minute GPU chain after this
FINAL_BY = T("04:15", tomorrow)


def log(m):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {m}\n")


def newest(name):
    ds = sorted(d for d in RUNS.glob(f"*-{name}") if d.is_dir())
    return ds[-1] if ds else None


def run(tag, args, script=None):
    t = time.time()
    log(f"start {tag}: {' '.join(args)}")
    with open(RUNS / f"n_{tag}.out", "w", encoding="utf-8") as out:
        rc = subprocess.run([PY, script or PIPE, *args], cwd=ROOT, stdout=out, stderr=subprocess.STDOUT,
                            env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"}).returncode
    log(f"done {tag} rc={rc} in {time.time() - t:.0f}s")
    return rc == 0


def val(name):
    d = newest(name)
    p = d / "metrics.json" if d else None
    return json.loads(p.read_text()).get("val_f05") if p and p.exists() else None


def chain(tag, train_args):
    """retrain -> rescore -> ce-apply; returns the ce run name or None."""
    if time.time() > LAST_GPU_START:
        log(f"skip {tag}: past last GPU start")
        return None
    if not run(tag, ["retrain", "--run", V10SRC, "--name", tag, "--rounds", "7000", *train_args, *BLOCK]):
        return None
    log(f"{tag} stage-1 val {val(tag)}")
    if not run(f"{tag}_test", ["rescore", "--run", newest(tag).name, "--feats-run", newest("v10testsrc").name,
                               "--name", f"{tag}_test", *BLOCK]):
        return None
    if not run(f"{tag}_ce", ["ce-apply", "--run", newest(tag).name, "--feats-run", newest(f"{tag}_test").name,
                             "--ce-dir", CE_DIR, "--name", f"{tag}_ce", *BLOCK]):
        return None
    log(f"{tag}_ce val {val(f'{tag}_ce')}")
    return f"{tag}_ce"


log(f"night queue start; source {V10SRC}")
# 1. test features + IDF
ok_test = run("aug_test", ["augment", "--run", V9TEST, "--split", "test", "--name", "v10testsrc"])
# 2-4. v10
v10 = chain("v10", []) if ok_test else None
# 5. unseen-country proxy
run("loco", [V10SRC], script=str(ROOT / "tmp" / "scratch" / "loco_idf.py"))
loco = json.loads((RUNS / "loco_idf.json").read_text()) if (RUNS / "loco_idf.json").exists() else {}


def lget(k, f="tgt_shape"):
    return loco.get(k, {}).get(f)


def delta(a, b):
    return (a - b) if a is not None and b is not None else None


d_ui = delta(lget("us->india idf"), lget("us->india base"))
d_iu = delta(lget("india->us idf"), lget("india->us base"))
d_idf = sum(x for x in (d_ui, d_iu) if x is not None) / max(1, sum(x is not None for x in (d_ui, d_iu))) if (d_ui is not None or d_iu is not None) else 0.0
d_crowd = delta(lget("us->india idf+crowd3"), lget("us->india base"))
log(f"loco deltas (France proxy, shape decision): idf us->india {d_ui} india->us {d_iu} mean {d_idf}; idf+crowd3 us->india {d_crowd}")
# 6. crowd-weighted variant
v10w = chain("v10w", ["--crowd-w", "3"]) if ok_test else None
# 7. ensemble v9 + v10
ens = None
if v10 and time.time() < LAST_GPU_START:
    if run("ens", ["ens_v9v10", V9, V9TEST, newest("v10").name, newest("v10_test").name],
           script=str(ROOT / "tmp" / "scratch" / "ens.py")):
        if run("ens_ce", ["ce-apply", "--run", newest("ens_v9v10").name, "--ce-dir", CE_DIR, "--name", "ens_v9v10_ce", *BLOCK]):
            ens = "ens_v9v10_ce"
            log(f"ens_v9v10_ce val {val(ens)}")

# 8. validate, estimate, pick
base_val = val("v9_ce") or 0.98585
cands = [("v9_ce", 0.0, "v9 + L12 CE + France shape rule (LB 0.975411)")]
if v10:
    cands.append(("v10_ce", d_idf, "v10: + country-IDF token features, L12 CE, France shape rule"))
if v10w:
    cands.append(("v10w_ce", d_crowd if d_crowd is not None else d_idf, "v10 + 3x weight on crowded S1 (>=15 candidates)"))
if ens:
    cands.append(("ens_v9v10_ce", 0.5 * d_idf, "average of v9 and v10 stage-1, L12 CE, France shape rule"))
rows = []
for name, dfr, note in cands:
    d = newest(name)
    f = d / "output" / "matching_results.tsv" if d else None
    if not f or not f.exists():
        rows.append({"file": name, "note": note, "status": "missing"})
        continue
    r = subprocess.run([sys.executable, str(ROOT / "student_resource" / "utils" / "validate_submission.py"),
                        "--matching", str(f), "--candidate", str(d / "output" / "candidate_pairs.tsv"),
                        "--test-dir", str(ROOT / "student_resource" / "dataset" / "test"), "--check-ids"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    m = pl.read_csv(f, separator="\t", quote_char=None, infer_schema=False).with_columns(
        pl.col("matched_entity_ids").fill_null("").str.split(",").list.eval(pl.element().filter(pl.element() != "")).list.len().alias("k"))
    s1 = pl.read_parquet(ROOT / "cache" / "raw_test_s1.parquet", columns=["entity_id", "country"])
    by = (m.join(s1.rename({"entity_id": "source1_entity_id"}), on="source1_entity_id").group_by("country")
          .agg(pl.col("k").mean().round(3).alias("k")).sort("country"))
    v = val(name)
    est = 0.85 * ((v or base_val) - base_val) + 0.15 * dfr
    rows.append({"file": name, "note": note, "status": "PASS" if r.returncode == 0 else "FAIL", "val": v, "fr_proxy": dfr,
                 "est": est, "shape": "; ".join(f"{c}: {k}/S1" for c, k in by.iter_rows()), "dir": d})
    log(f"validated {name}: {rows[-1]['status']} val {v} fr_proxy {dfr} est_dLB {est:+.5f} {rows[-1]['shape']}")
good = [r for r in rows if r.get("status") == "PASS"]
pick = max(good, key=lambda r: r["est"]) if good else None
if pick and pick["file"] != "v9_ce" and pick["est"] < 0.0003:
    pick = next((r for r in good if r["file"] == "v9_ce"), pick)
dst = ROOT / "output" / "SUBMIT_THIS"
for r in good:
    shutil.copy(r["dir"] / "output" / "matching_results.tsv", ROOT / "output" / "submissions" / f"{r['file']}_matching_results.tsv")
if pick:
    shutil.copy(pick["dir"] / "output" / "matching_results.tsv", dst / "matching_results.tsv")
    shutil.copy(pick["dir"] / "output" / "candidate_pairs.tsv", dst / "candidate_pairs.tsv")
    (dst / "WHAT_IS_THIS.txt").write_text(f"{pick['file']}: {pick['note']}\nval F0.5 {pick.get('val')}  France proxy delta "
                                          f"{pick.get('fr_proxy')}  estimated LB change vs v9_ce {pick['est']:+.5f}\n"
                                          f"{pick.get('shape')}\nvalidator --check-ids PASS\n{time.ctime()}\n", encoding="utf-8")
lines = ["# Night run report", f"generated {time.ctime()}", "",
         f"**Recommended file:** `output/SUBMIT_THIS/matching_results.tsv` = **{pick['file'] if pick else 'NONE'}**", "",
         "Estimated leaderboard change = 0.85 x (US/India val change) + 0.15 x (France proxy change). France proxy = "
         "train on one labelled country, score the other with the France decision rule (matches per S1 equal).", "",
         "| file | what | validator | val F0.5 (US/India) | France proxy change | est. LB change | matches per S1 |",
         "|---|---|---|---|---|---|---|"]
fmt = lambda x: "" if x is None else f"{x:+.5f}"  # noqa: E731
for r in rows:
    lines.append(f"| {r['file']} | {r['note']} | {r['status']} | {r.get('val', '')} | {r.get('fr_proxy', '')} | "
                 f"{fmt(r.get('est'))} | {r.get('shape', '')} |")
lines += ["", "## Unseen-country proxy (runs/loco_idf.json)", "", "```", json.dumps(loco, indent=1), "```", "",
          "Stable copies: output/submissions/<file>_matching_results.tsv. Queue log: runs/night_queue.log"]
(RUNS / "NIGHT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
log(f"report written, picked {pick['file'] if pick else None}")
