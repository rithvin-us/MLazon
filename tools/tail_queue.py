"""Tail queue: runs after night_queue. Distractor-density-matched decisions (`redecide`) for every candidate, then
the final pick. The test pool has ~2x the unmatched look-alikes per S1 of train, so each candidate is compared on
validation made equally dense (val_f05_dense), not on plain validation.

  1  ce-apply v9 again (saves val_scored_ce.parquet)  -> v9_ce2          GPU
  2  redecide v9_ce2, v10_ce, v10w_ce, ens_v9v10_ce    -> *_dense         CPU
  3  validate, estimate, pick, SUBMIT_THIS, runs/NIGHT_REPORT.md (appended)
Log: runs/tail_queue.log
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
LOG = RUNS / "tail_queue.log"
BLOCK = ["--max-df", "600", "--rr-tau", "0.002", "--rr-min", "3"]
V9, V9TEST = "20260926-141000-v9", "20260926-151110-v9_test"
CE_DIR = str(ROOT / "models" / f"ce_{V9}")
STOP = time.time() + 6.5 * 3600  # hard stop for waiting


def log(m):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {m}\n")


def newest(name):
    ds = sorted(d for d in RUNS.glob(f"*-{name}") if d.is_dir())
    return ds[-1] if ds else None


def metrics(name):
    d = newest(name)
    p = d / "metrics.json" if d else None
    return json.loads(p.read_text()) if p and p.exists() else {}


def run(tag, args):
    t = time.time()
    log(f"start {tag}: {' '.join(args)}")
    with open(RUNS / f"t_{tag}.out", "w", encoding="utf-8") as out:
        rc = subprocess.run([PY, PIPE, *args], cwd=ROOT, stdout=out, stderr=subprocess.STDOUT,
                            env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"}).returncode
    log(f"done {tag} rc={rc} in {time.time() - t:.0f}s")
    return rc == 0


log("tail queue waiting for night queue")
while "report written" not in ((RUNS / "night_queue.log").read_text(encoding="utf-8") if (RUNS / "night_queue.log").exists() else ""):
    if time.time() > STOP:
        log("night queue not finished; abort")
        sys.exit(0)
    time.sleep(30)
log("night queue finished; tail starts")

# 1. v9 CE again so its stacked val scores are saved
run("v9_ce2", ["ce-apply", "--run", V9, "--feats-run", V9TEST, "--ce-dir", CE_DIR, "--name", "v9_ce2", *BLOCK])
# 1b. ensemble of the two v10 models (same candidates, both with IDF features)
if newest("v10_test") and newest("v10w_test") and (newest("v10w_ce") is not None):
    t = time.time()
    log("start ens_v10v10w")
    rc = subprocess.run([PY, str(ROOT / "tmp" / "scratch" / "ens.py"), "ens_v10v10w", newest("v10").name, newest("v10_test").name,
                         newest("v10w").name, newest("v10w_test").name], cwd=ROOT, capture_output=True, text=True).returncode
    log(f"done ens_v10v10w rc={rc} in {time.time() - t:.0f}s")
    if rc == 0:
        run("ens_v10v10w_ce", ["ce-apply", "--run", newest("ens_v10v10w").name, "--ce-dir", CE_DIR, "--name", "ens_v10v10w_ce", *BLOCK])
# 2. density-matched decisions
plain = [n for n in ("v9_ce2", "v10_ce", "v10w_ce", "ens_v9v10_ce", "ens_v10v10w_ce") if newest(n) and (newest(n) / "val_scored_ce.parquet").exists()]
for n in plain:
    run(f"{n}_dense", ["redecide", "--run", newest(n).name, "--name", f"{n}_dense", *BLOCK])

# 3. estimates. dense_F(plain X) = its current decision on dense val (logged by its redecide run)
night = {}
for line in (RUNS / "night_queue.log").read_text(encoding="utf-8").splitlines():
    if "loco deltas" in line:
        night["loco"] = line
loco = json.loads((RUNS / "loco_idf.json").read_text()) if (RUNS / "loco_idf.json").exists() else {}


def lget(k):
    return loco.get(k, {}).get("tgt_shape")


def d(a, b):
    return a - b if a is not None and b is not None else None


d_idf = [x for x in (d(lget("us->india idf"), lget("us->india base")), d(lget("india->us idf"), lget("india->us base"))) if x is not None]
d_idf = sum(d_idf) / len(d_idf) if d_idf else 0.0
d_crowd = d(lget("us->india idf+crowd3"), lget("us->india base"))
fr = {"v9_ce2": 0.0, "v10_ce": d_idf, "v10w_ce": d_crowd if d_crowd is not None else d_idf, "ens_v9v10_ce": 0.5 * d_idf,
      "ens_v10v10w_ce": 0.5 * (d_idf + (d_crowd if d_crowd is not None else d_idf))}
rows = []
base_dense = metrics("v9_ce2_dense").get("val_f05_dense_before")
for n in plain:
    md = metrics(f"{n}_dense")
    if not md:
        continue
    for variant, dense_f, dirname, note in ((n, md.get("val_f05_dense_before"), newest(n), "val-tuned decision"),
                                            (f"{n}_dense", md.get("val_f05_dense"), newest(f"{n}_dense"), "density-matched decision")):
        rows.append({"file": variant, "note": note, "dense": dense_f, "val": metrics(variant).get("val_f05") if variant != n else metrics(n).get("val_f05"),
                     "fr": fr.get(n, 0.0), "dir": dirname})
for r in rows:
    r["est"] = 0.85 * ((r["dense"] or 0) - (base_dense or 0)) + 0.15 * (r["fr"] or 0.0) if base_dense and r["dense"] else None
# validate (plain CE runs were validated by the night queue; dense ones here)
for r in rows:
    f = r["dir"] / "output" / "matching_results.tsv" if r["dir"] else None
    if not f or not f.exists():
        r["status"] = "missing"
        continue
    res = subprocess.run([sys.executable, str(ROOT / "student_resource" / "utils" / "validate_submission.py"), "--matching", str(f),
                          "--candidate", str(r["dir"] / "output" / "candidate_pairs.tsv"),
                          "--test-dir", str(ROOT / "student_resource" / "dataset" / "test"), "--check-ids"],
                         capture_output=True, text=True, encoding="utf-8", errors="replace")
    r["status"] = "PASS" if res.returncode == 0 else "FAIL"
    m = pl.read_csv(f, separator="\t", quote_char=None, infer_schema=False).with_columns(
        pl.col("matched_entity_ids").fill_null("").str.split(",").list.eval(pl.element().filter(pl.element() != "")).list.len().alias("k"))
    s1 = pl.read_parquet(ROOT / "cache" / "raw_test_s1.parquet", columns=["entity_id", "country"])
    by = m.join(s1.rename({"entity_id": "source1_entity_id"}), on="source1_entity_id").group_by("country").agg(pl.col("k").mean().round(3)).sort("country")
    r["shape"] = "; ".join(f"{c}: {k}/S1" for c, k in by.iter_rows())
    log(f"{r['file']}: {r['status']} val {r['val']} dense {r['dense']} fr {r['fr']} est {r['est']} {r['shape']}")
good = [r for r in rows if r.get("status") == "PASS" and r.get("est") is not None]
pick = max(good, key=lambda r: r["est"]) if good else None
if pick and pick["est"] < 0.0005:  # not clearly better than the submitted v9_ce: keep it
    pick = None
dst = ROOT / "output" / "SUBMIT_THIS"
for r in good:
    shutil.copy(r["dir"] / "output" / "matching_results.tsv", ROOT / "output" / "submissions" / f"{r['file']}_matching_results.tsv")
if pick:
    shutil.copy(pick["dir"] / "output" / "matching_results.tsv", dst / "matching_results.tsv")
    shutil.copy(pick["dir"] / "output" / "candidate_pairs.tsv", dst / "candidate_pairs.tsv")
    (dst / "WHAT_IS_THIS.txt").write_text(
        f"{pick['file']} ({pick['note']})\nplain val F0.5 {pick['val']}  density-matched val F0.5 {pick['dense']} "
        f"(v9_ce as submitted: {base_dense})\nFrance proxy change {pick['fr']}  estimated LB change vs v9_ce {pick['est']:+.5f}\n"
        f"{pick['shape']}\nvalidator --check-ids PASS\n{time.ctime()}\n", encoding="utf-8")
else:  # authoritative: nothing clearly beats the submitted file under test-like density -> keep exactly v9_ce
    src = RUNS / "20260926-155941-v9_ce" / "output"
    shutil.copy(src / "matching_results.tsv", dst / "matching_results.tsv")
    shutil.copy(src / "candidate_pairs.tsv", dst / "candidate_pairs.tsv")
    (dst / "WHAT_IS_THIS.txt").write_text(f"v9_ce (LB 0.975411): no candidate beat it by >= 0.0005 estimated LB under "
                                          f"test-like distractor density\n{time.ctime()}\n", encoding="utf-8")
fmt = lambda x: "" if x is None else f"{x:+.5f}"  # noqa: E731
lines = ["", "## Density-matched decisions (tail queue)", f"generated {time.ctime()}", "",
         "Test has ~2x the unmatched look-alikes per S1 of train. dense = F0.5 on validation with val negatives copied "
         "per score band until the band counts match test. Baseline (v9_ce as submitted, LB 0.975411) dense: "
         f"{base_dense}.", "", f"**Final pick:** {pick['file'] if pick else 'v9_ce unchanged'}", "",
         "| file | decision | validator | plain val | dense val | France proxy | est. LB change | matches per S1 |", "|---|---|---|---|---|---|---|---|"]
for r in rows:
    lines.append(f"| {r['file']} | {r['note']} | {r.get('status')} | {r['val']} | {r['dense']} | {r['fr']} | {fmt(r.get('est'))} | {r.get('shape', '')} |")
with open(RUNS / "NIGHT_REPORT.md", "a", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
log(f"tail done, pick {pick['file'] if pick else 'v9_ce (unchanged)'}")
