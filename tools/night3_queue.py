"""Night queue part 3: after night2. Transfer-robust training on the v9 features (monotone constraints / depth 6),
chosen on the unseen-country proxy, then the final pick over every candidate. Log: runs/night3_queue.log"""
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
LOG = RUNS / "night3_queue.log"
BLOCK = ["--max-df", "600", "--rr-tau", "0.002", "--rr-min", "3"]
V9CE = "20260926-155941-v9_ce"
V10SRC = sorted(d.name for d in RUNS.glob("*-v10src"))[-1]
V10TESTSRC = sorted(d.name for d in RUNS.glob("*-v10testsrc"))[-1]
CE_DIR = str(ROOT / "models" / "ce_20260926-141000-v9")
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution" / "src"))
from features import IDF_FEATURES  # noqa: E402

FLAGS = {"mono": ["--monotone"], "d6": ["--depth", "6"], "mono_d6": ["--monotone", "--depth", "6"]}
NAME = ["n_idf_jacc", "n_idf_miss1", "n_idf_miss2", "n_rep", "n_idf_rank"]
HARD_STOP = time.time() + 5.5 * 3600


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
    with open(RUNS / f"n3_{tag}.out", "w", encoding="utf-8") as out:
        rc = subprocess.run([PY, script or PIPE, *args], cwd=ROOT, stdout=out, stderr=subprocess.STDOUT,
                            env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"}).returncode
    log(f"done {tag} rc={rc} in {time.time() - t:.0f}s")
    return rc == 0


def js(name):
    p = RUNS / name
    return json.loads(p.read_text()) if p.exists() else {}


def delta(res, name, field):
    base = js("loco_idf.json")
    ds = [res[f"{s}->{t} {name}"][field] - base[f"{s}->{t} base"][field]
          for s, t in (("us", "india"), ("india", "us")) if f"{s}->{t} {name}" in res and f"{s}->{t} base" in base]
    return sum(ds) / len(ds) if ds else None


log("night3 waiting for night2")
while "report written" not in ((RUNS / "night2_queue.log").read_text(encoding="utf-8") if (RUNS / "night2_queue.log").exists() else ""):
    if time.time() > HARD_STOP:
        log("night2 not finished; abort")
        sys.exit(0)
    time.sleep(30)
log("night2 done; night3 starts")
run("loco_robust", [V10SRC], script=str(ROOT / "tmp" / "scratch" / "loco_robust.py"))
rob = js("loco_robust.json")
est = {n: 0.85 * (delta(rob, n, "src_val") or 0) + 0.15 * (delta(rob, n, "tgt_shape") or 0) for n in FLAGS if delta(rob, n, "tgt_shape") is not None}
log(f"robust proxy: { {n: (delta(rob, n, 'src_val'), delta(rob, n, 'tgt_shape')) for n in rob and FLAGS} } est {est}")
best = max(est, key=est.get) if est else None
idf0 = js("loco_idf.json")
est_name = 0.85 * (delta(idf0, "name", "src_val") or 0) + 0.15 * (delta(idf0, "name", "tgt_shape") or 0)
log(f"name-subset (v10s) estimate {est_name}")
if best and est[best] > est_name + 0.0003 and time.time() < HARD_STOP - 3600:
    drop = ",".join(f for f in IDF_FEATURES if f not in NAME)
    if run("v10r", ["retrain", "--run", V10SRC, "--name", "v10r", "--rounds", "7000", "--drop-feats", drop, *FLAGS[best], *BLOCK]):
        log(f"v10r ({best}) stage-1 val {metrics('v10r').get('val_f05')}")
        if run("v10r_test", ["rescore", "--run", newest("v10r").name, "--feats-run", V10TESTSRC, "--name", "v10r_test", *BLOCK]) and            run("v10r_ce", ["ce-apply", "--run", newest("v10r").name, "--feats-run", newest("v10r_test").name, "--ce-dir", CE_DIR, "--name", "v10r_ce", *BLOCK]):
            log(f"v10r_ce val {metrics('v10r_ce').get('val_f05')}")
            run("v10r_ce_dense", ["redecide", "--run", newest("v10r_ce").name, "--name", "v10r_ce_dense", *BLOCK])
else:
    log(f"no robust variant worth retraining (best {best})")

# final pick over every candidate
idf = js("loco_idf.json")
sub_line = next((l for l in (RUNS / "night2_queue.log").read_text(encoding="utf-8").splitlines() if "-> best" in l), "")
sub_best = sub_line.rsplit("-> best ", 1)[-1].strip() if sub_line else "idf"
fr = {"v9_ce2": 0.0, "v10_ce": delta(idf, "idf", "tgt_shape") or 0.0, "v10s_ce": delta(idf, sub_best, "tgt_shape") or 0.0,
      "v10r_ce": (delta(rob, best, "tgt_shape") or 0.0) if best else 0.0}
base_dense = metrics("v9_ce2_dense").get("val_f05_dense_before")
rows = []
for n in fr:
    md = metrics(f"{n}_dense")
    if not md or not newest(n):
        continue
    rows.append({"file": n, "note": "val-tuned decision", "val": metrics(n).get("val_f05"), "dense": md.get("val_f05_dense_before"), "fr": fr[n], "dir": newest(n)})
    rows.append({"file": f"{n}_dense", "note": "density-matched decision", "val": md.get("val_f05"), "dense": md.get("val_f05_dense"), "fr": fr[n], "dir": newest(f"{n}_dense")})
s1 = pl.read_parquet(ROOT / "cache" / "raw_test_s1.parquet", columns=["entity_id", "country"])
for r in rows:
    r["est"] = 0.85 * (r["dense"] - base_dense) + 0.15 * r["fr"] if base_dense and r["dense"] else None
    f = r["dir"] / "output" / "matching_results.tsv"
    if not f.exists():
        r["status"] = "missing"
        continue
    res = subprocess.run([sys.executable, str(ROOT / "student_resource" / "utils" / "validate_submission.py"), "--matching", str(f),
                          "--candidate", str(r["dir"] / "output" / "candidate_pairs.tsv"),
                          "--test-dir", str(ROOT / "student_resource" / "dataset" / "test"), "--check-ids"],
                         capture_output=True, text=True, encoding="utf-8", errors="replace")
    r["status"] = "PASS" if res.returncode == 0 else "FAIL"
    m = pl.read_csv(f, separator="\t", quote_char=None, infer_schema=False).with_columns(
        pl.col("matched_entity_ids").fill_null("").str.split(",").list.eval(pl.element().filter(pl.element() != "")).list.len().alias("k"))
    by = m.join(s1.rename({"entity_id": "source1_entity_id"}), on="source1_entity_id").group_by("country").agg(pl.col("k").mean().round(3)).sort("country")
    r["shape"] = "; ".join(f"{c}: {k}/S1" for c, k in by.iter_rows())
    log(f"{r['file']}: {r['status']} val {r['val']} dense {r['dense']} fr {r['fr']} est {r['est']} {r['shape']}")
good = sorted([r for r in rows if r.get("status") == "PASS" and r.get("est") is not None], key=lambda r: -r["est"])
pick = good[0] if good and good[0]["est"] >= 0.0005 else None
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
lines = ["", "## Final pick (night3, after transfer-robust training)", f"generated {time.ctime()}", "",
         f"**SUBMIT_THIS = {pick['file'] if pick else 'v9_ce (unchanged)'}**", "",
         f"robust proxy results: {json.dumps(rob)}", f"robust estimates: {est}; retrained: {best if newest('v10r') else 'none'}", "",
         "| rank | file | decision | validator | plain val | test-density val | proxy change | est. LB change | matches per S1 |",
         "|---|---|---|---|---|---|---|---|---|"]
for i, r in enumerate(sorted(rows, key=lambda r: -(r.get("est") or -9))):
    lines.append(f"| {i + 1} | {r['file']} | {r['note']} | {r.get('status')} | {r['val']} | {r['dense']} | {fmt(r['fr'])} | {fmt(r.get('est'))} | {r.get('shape', '')} |")
with open(RUNS / "NIGHT_REPORT.md", "a", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
log(f"night3 report written, pick {pick['file'] if pick else 'v9_ce'}")
