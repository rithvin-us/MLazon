"""Night part 5: stage-1 ensembles for the US/India side (v9+v10, v10+v10s), CE + density-matched decision, then the
per-country mix again: US/India rows from the best test-density validation, France rows from the best unseen-country
proxy (v10s, name-IDF). Re-picks SUBMIT_THIS only if the new mix beats the current pick's estimate by >= 0.0002.
Log: runs/night5_ens.log"""
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import polars as pl

ROOT = Path(r"D:\amazon-ml")
PY = str(ROOT / ".venv311" / "Scripts" / "python.exe")
PIPE = str(ROOT / "code" / "business_entity_resolution" / "src" / "pipeline.py")
ENS = str(ROOT / "tmp" / "scratch" / "ens.py")
RUNS = ROOT / "runs"
LOG = RUNS / "night5_ens.log"
BLOCK = ["--max-df", "600", "--rr-tau", "0.002", "--rr-min", "3"]
CE_DIR = str(ROOT / "models" / "ce_20260926-141000-v9")
V9, V9TEST = "20260926-141000-v9", "20260926-151110-v9_test"
DEADLINE = time.time() + 3.2 * 3600


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
    with open(RUNS / f"n5_{tag}.out", "w", encoding="utf-8") as out:
        rc = subprocess.run([PY, script or PIPE, *args], cwd=ROOT, stdout=out, stderr=subprocess.STDOUT,
                            env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"}).returncode
    log(f"done {tag} rc={rc} in {time.time() - t:.0f}s")
    return rc == 0


def wait(logname, marker):
    while marker not in ((RUNS / logname).read_text(encoding="utf-8") if (RUNS / logname).exists() else ""):
        if time.time() > DEADLINE:
            log(f"{logname} never showed '{marker}'; abort")
            sys.exit(0)
        time.sleep(30)


log("waiting for night3")
wait("night3_queue.log", "night3 report written")
for tag, pairs in (("ens_v9v10", [V9, V9TEST, newest("v10").name, newest("v10_test").name]),
                   ("ens_v10v10s", [newest("v10").name, newest("v10_test").name, newest("v10s").name, newest("v10s_test").name])):
    if time.time() > DEADLINE - 1800:
        log(f"skip {tag}: late")
        continue
    if run(tag, [tag, *pairs], script=ENS) and run(f"{tag}_ce", ["ce-apply", "--run", newest(tag).name, "--ce-dir", CE_DIR, "--name", f"{tag}_ce", *BLOCK]):
        log(f"{tag}_ce val {metrics(f'{tag}_ce').get('val_f05')}")
        run(f"{tag}_ce_dense", ["redecide", "--run", newest(f"{tag}_ce").name, "--name", f"{tag}_ce_dense", *BLOCK])
        md = metrics(f"{tag}_ce_dense")
        log(f"{tag}: dense before {md.get('val_f05_dense_before')} after {md.get('val_f05_dense')}")

wait("night4_mix.log", "mix done")
base_dense = metrics("v9_ce2_dense").get("val_f05_dense_before")
cands = {}  # name -> dense (US/India) for every file whose US/India rows could be used
for n in ("v9_ce2", "v10_ce", "v10s_ce", "ens_v9v10_ce", "ens_v10v10s_ce"):
    md = metrics(f"{n}_dense")
    if md:
        cands[n] = md.get("val_f05_dense_before")
        cands[f"{n}_dense"] = md.get("val_f05_dense")
cands = {k: v for k, v in cands.items() if v is not None and newest(k)}
us_src = max(cands, key=cands.get)
fr_src = "v10s_ce_dense" if newest("v10s_ce_dense") else "v10s_ce"
fr_proxy = 0.0
idf = json.loads((RUNS / "loco_idf.json").read_text())
ds = [idf[f"{s}->{t} name"]["tgt_shape"] - idf[f"{s}->{t} base"]["tgt_shape"] for s, t in (("us", "india"), ("india", "us"))]
fr_proxy = sum(ds) / len(ds)
est = 0.85 * (cands[us_src] - base_dense) + 0.15 * fr_proxy
log(f"US/India candidates (dense) {cands}; pick US/India {us_src}, France {fr_src}; est {est:+.5f}")
# current pick's estimate (night4 mix or night3 single)
cur_est = None
m4 = re.search(r"-> est ([+-][0-9.]+)", (RUNS / "night4_mix.log").read_text(encoding="utf-8"))
picked4 = "picked=True" in (RUNS / "night4_mix.log").read_text(encoding="utf-8")
if picked4 and m4:
    cur_est = float(m4.group(1))
else:
    w = (ROOT / "output" / "SUBMIT_THIS" / "WHAT_IS_THIS.txt").read_text(encoding="utf-8")
    m3 = re.search(r"estimated LB change vs v9_ce ([+-][0-9.]+)", w)
    cur_est = float(m3.group(1)) if m3 else 0.0
log(f"current SUBMIT_THIS estimate {cur_est:+.5f}")
out = RUNS / f"{time.strftime('%Y%m%d-%H%M%S')}-mix2"
(out / "output").mkdir(parents=True)
s1 = pl.read_parquet(ROOT / "cache" / "raw_test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "source1_entity_id"})
fr_ids = s1.filter(pl.col("country").str.to_lowercase().str.strip_chars() == "france")["source1_entity_id"]
rd = lambda n: pl.read_csv(newest(n) / "output" / "matching_results.tsv", separator="\t", quote_char=None, infer_schema=False)  # noqa: E731
a, b = rd(us_src), rd(fr_src).rename({"matched_entity_ids": "fr"})
mixed = (a.join(b, on="source1_entity_id", how="left", maintain_order="left")
         .with_columns(pl.when(pl.col("source1_entity_id").is_in(fr_ids.implode())).then(pl.col("fr"))
                       .otherwise(pl.col("matched_entity_ids")).fill_null("").alias("matched_entity_ids")).drop("fr"))
assert mixed.height == a.height
mixed.write_csv(out / "output" / "matching_results.tsv", separator="\t", quote_style="never")
shutil.copy(newest(us_src) / "output" / "candidate_pairs.tsv", out / "output" / "candidate_pairs.tsv")
res = subprocess.run([sys.executable, str(ROOT / "student_resource" / "utils" / "validate_submission.py"), "--matching",
                      str(out / "output" / "matching_results.tsv"), "--candidate", str(out / "output" / "candidate_pairs.tsv"),
                      "--test-dir", str(ROOT / "student_resource" / "dataset" / "test"), "--check-ids"],
                     capture_output=True, text=True, encoding="utf-8", errors="replace")
ok = res.returncode == 0
(out / "metrics.json").write_text(json.dumps({"mix_us_india": us_src, "mix_france": fr_src, "est": est, "validator_pass": ok}))
log(f"mix2 validator {'PASS' if ok else 'FAIL'} ({out.name})")
picked = False
if ok and est >= max(cur_est, 0.0005) + (0.0002 if cur_est > 0 else 0.0):
    dst = ROOT / "output" / "SUBMIT_THIS"
    shutil.copy(out / "output" / "matching_results.tsv", dst / "matching_results.tsv")
    shutil.copy(out / "output" / "candidate_pairs.tsv", dst / "candidate_pairs.tsv")
    (dst / "WHAT_IS_THIS.txt").write_text(f"mix2: US/India rows from {us_src} (test-density val {cands[us_src]}), France rows from "
                                          f"{fr_src} (name-IDF model, best unseen-country proxy)\nestimated LB change vs v9_ce {est:+.5f}\n"
                                          f"validator --check-ids PASS\n{time.ctime()}\n", encoding="utf-8")
    picked = True
if ok:
    shutil.copy(out / "output" / "matching_results.tsv", ROOT / "output" / "submissions" / "mix2_matching_results.tsv")
with open(RUNS / "NIGHT_REPORT.md", "a", encoding="utf-8") as f:
    f.write(f"\n## Ensembles + mix2\nUS/India test-density val by source: {json.dumps(cands)}\nmix2 = US/India from {us_src}, France from "
            f"{fr_src}: est {est:+.5f} vs current {cur_est:+.5f}. {'**mix2 is now SUBMIT_THIS.**' if picked else 'SUBMIT_THIS unchanged.'}\n")
log(f"night5 done, picked={picked}")
