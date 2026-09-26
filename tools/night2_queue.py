"""Night queue part 2 (replaces the v10w / ensemble plan after the unseen-country proxy showed the full v10 IDF set
hurts US->India transfer). Steps:
  1  wait for the running proxy job; 2  proxy on IDF feature subsets (both directions)
  3  retrain the best subset (if not base/full) -> rescore -> CE          GPU
  4  v9 CE again (saves stacked val scores)                               GPU
  5  density-matched redecide for v9_ce2, v10_ce, v10s_ce                  CPU
  6  validate, estimate LB change = 0.85 x dense-val change + 0.15 x proxy change, pick, SUBMIT_THIS, report
Log: runs/night2_queue.log
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
LOG = RUNS / "night2_queue.log"
BLOCK = ["--max-df", "600", "--rr-tau", "0.002", "--rr-min", "3"]
V9, V9TEST, V9CE = "20260926-141000-v9", "20260926-151110-v9_test", "20260926-155941-v9_ce"
V10SRC = sorted(d.name for d in RUNS.glob("*-v10src"))[-1]
V10TESTSRC = sorted(d.name for d in RUNS.glob("*-v10testsrc"))[-1]
CE_DIR = str(ROOT / "models" / f"ce_{V9}")
LOCO = str(ROOT / "tmp" / "scratch" / "loco_idf.py")
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution" / "src"))
from features import IDF_FEATURES  # noqa: E402

SUBSETS = {  # must mirror loco_idf.py
    "idf": IDF_FEATURES,
    "ratio": [f for f in IDF_FEATURES if f not in ("n_rep_idf", "n_miss_idf", "a_rep_idf", "a_miss_idf")],
    "name": [f for f in IDF_FEATURES if f.startswith("n_") and f not in ("n_rep_idf", "n_miss_idf")],
    "addr": [f for f in IDF_FEATURES if f.startswith("a_") and f not in ("a_rep_idf", "a_miss_idf")],
    "jacc": ["n_idf_jacc", "a_idf_jacc", "n_idf_rank", "a_idf_rank"],
}


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


def run(tag, args, script=None):
    t = time.time()
    log(f"start {tag}: {' '.join(args)}")
    with open(RUNS / f"n2_{tag}.out", "w", encoding="utf-8") as out:
        rc = subprocess.run([PY, script or PIPE, *args], cwd=ROOT, stdout=out, stderr=subprocess.STDOUT,
                            env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"}).returncode
    log(f"done {tag} rc={rc} in {time.time() - t:.0f}s")
    return rc == 0


def loco():
    return json.loads((RUNS / "loco_idf.json").read_text()) if (RUNS / "loco_idf.json").exists() else {}


def proxy_delta(name):
    """mean over directions of tgt_shape(name) - tgt_shape(base); None if missing."""
    L = loco()
    ds = [L[f"{s}->{t} {name}"]["tgt_shape"] - L[f"{s}->{t} base"]["tgt_shape"]
          for s, t in (("us", "india"), ("india", "us")) if f"{s}->{t} {name}" in L and f"{s}->{t} base" in L]
    return sum(ds) / len(ds) if ds else None


def src_delta(name):
    L = loco()
    ds = [L[f"{s}->{t} {name}"]["src_val"] - L[f"{s}->{t} base"]["src_val"]
          for s, t in (("us", "india"), ("india", "us")) if f"{s}->{t} {name}" in L and f"{s}->{t} base" in L]
    return sum(ds) / len(ds) if ds else None


log(f"night2 start; v10src {V10SRC}, v10testsrc {V10TESTSRC}")
# 1. wait for the first proxy job (india->us idf)
t0 = time.time()
while "india->us idf" not in loco() and time.time() - t0 < 1800:
    time.sleep(20)
time.sleep(20)  # let it exit
log(f"proxy so far: idf {proxy_delta('idf')} (src {src_delta('idf')})")
# 2. subsets
run("loco_subsets", [V10SRC, "ratio,jacc,name,addr"], script=LOCO)
scores = {n: (proxy_delta(n), src_delta(n)) for n in ("idf", "ratio", "jacc", "name", "addr")}
log(f"proxy deltas (tgt_shape, src_val) vs base: {scores}")
# pick subset by estimated LB effect of the feature change: 0.85 x src gain + 0.15 x unseen-country gain
est = {n: 0.85 * (v[1] or 0) + 0.15 * (v[0] or 0) for n, v in scores.items() if v[0] is not None}
best = max(est, key=est.get) if est else "idf"
log(f"subset estimates {est} -> best {best}")
# 3. retrain best subset if it is a new one
if best != "idf" and est.get(best, -1) > 0:
    drop = [f for f in IDF_FEATURES if f not in SUBSETS[best]]
    if run("v10s", ["retrain", "--run", V10SRC, "--name", "v10s", "--rounds", "7000", "--drop-feats", ",".join(drop), *BLOCK]):
        log(f"v10s ({best}) stage-1 val {metrics('v10s').get('val_f05')}")
        if run("v10s_test", ["rescore", "--run", newest("v10s").name, "--feats-run", V10TESTSRC, "--name", "v10s_test", *BLOCK]):
            if run("v10s_ce", ["ce-apply", "--run", newest("v10s").name, "--feats-run", newest("v10s_test").name,
                               "--ce-dir", CE_DIR, "--name", "v10s_ce", *BLOCK]):
                log(f"v10s_ce val {metrics('v10s_ce').get('val_f05')}")
# 4. v9 CE again (stacked val scores for density evaluation)
run("v9_ce2", ["ce-apply", "--run", V9, "--feats-run", V9TEST, "--ce-dir", CE_DIR, "--name", "v9_ce2", *BLOCK])
# 5. density-matched decisions
plain = [n for n in ("v9_ce2", "v10_ce", "v10s_ce") if newest(n) and (newest(n) / "val_scored_ce.parquet").exists()]
for n in plain:
    run(f"{n}_dense", ["redecide", "--run", newest(n).name, "--name", f"{n}_dense", *BLOCK])
# 6. estimates, validation, pick
fr = {"v9_ce2": 0.0, "v10_ce": proxy_delta("idf") or 0.0, "v10s_ce": proxy_delta(best) or 0.0}
base_dense = metrics("v9_ce2_dense").get("val_f05_dense_before")
rows = []
for n in plain:
    md = metrics(f"{n}_dense")
    if not md:
        continue
    rows.append({"file": n, "note": "val-tuned decision", "val": metrics(n).get("val_f05"), "dense": md.get("val_f05_dense_before"),
                 "fr": fr[n], "dir": newest(n)})
    rows.append({"file": f"{n}_dense", "note": "density-matched decision", "val": md.get("val_f05"), "dense": md.get("val_f05_dense"),
                 "fr": fr[n], "dir": newest(f"{n}_dense")})
for r in rows:
    r["est"] = 0.85 * (r["dense"] - base_dense) + 0.15 * r["fr"] if base_dense and r["dense"] else None
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
if pick and pick["est"] < 0.0005:
    pick = None
dst = ROOT / "output" / "SUBMIT_THIS"
for r in good:
    shutil.copy(r["dir"] / "output" / "matching_results.tsv", ROOT / "output" / "submissions" / f"{r['file']}_matching_results.tsv")
if pick:
    shutil.copy(pick["dir"] / "output" / "matching_results.tsv", dst / "matching_results.tsv")
    shutil.copy(pick["dir"] / "output" / "candidate_pairs.tsv", dst / "candidate_pairs.tsv")
    (dst / "WHAT_IS_THIS.txt").write_text(
        f"{pick['file']} ({pick['note']})\nplain val F0.5 {pick['val']}  test-density val F0.5 {pick['dense']} (v9_ce as submitted: {base_dense})\n"
        f"unseen-country proxy change {pick['fr']}  estimated LB change vs v9_ce {pick['est']:+.5f}\n{pick['shape']}\n"
        f"validator --check-ids PASS\n{time.ctime()}\n", encoding="utf-8")
else:
    src = RUNS / V9CE / "output"
    shutil.copy(src / "matching_results.tsv", dst / "matching_results.tsv")
    shutil.copy(src / "candidate_pairs.tsv", dst / "candidate_pairs.tsv")
    (dst / "WHAT_IS_THIS.txt").write_text(f"v9_ce (LB 0.975411): nothing beat it by >= 0.0005 estimated LB\n{time.ctime()}\n", encoding="utf-8")
fmt = lambda x: "" if x is None else f"{x:+.5f}"  # noqa: E731
lines = ["# Night report (2026-09-27)", f"generated {time.ctime()}", "",
         f"**Recommended file:** `output/SUBMIT_THIS/matching_results.tsv` = **{pick['file'] if pick else 'v9_ce (unchanged)'}**", "",
         "Estimated LB change = 0.85 x (change in test-density val F0.5) + 0.15 x (change in unseen-country proxy).",
         f"Test-density val F0.5 of v9_ce as submitted: {base_dense}.", "",
         "| file | decision | validator | plain val | test-density val | proxy change | est. LB change | matches per S1 |",
         "|---|---|---|---|---|---|---|---|"]
for r in rows:
    lines.append(f"| {r['file']} | {r['note']} | {r.get('status')} | {r['val']} | {r['dense']} | {fmt(r['fr'])} | {fmt(r.get('est'))} | {r.get('shape', '')} |")
lines += ["", "## Unseen-country proxy (train one country, score the other; France-style decision)", "",
          f"subset deltas vs base (tgt_shape, src_val): {scores}", f"subset estimates: {est}; retrained: {best}", "",
          "```", json.dumps(loco(), indent=1), "```"]
(RUNS / "NIGHT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
log(f"report written, pick {pick['file'] if pick else 'v9_ce'}")
