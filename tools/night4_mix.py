"""Final step after night3: per-country mix. Candidates never cross countries (blocking is per country), so a file
can take US/India rows from the model with the best test-density validation and France rows from the model with
the best unseen-country proxy. est = 0.85 x dense-val change (US/India source) + 0.15 x proxy change (France source).
Validates, re-picks SUBMIT_THIS if the mix wins, appends to runs/NIGHT_REPORT.md. Log: runs/night4_mix.log"""
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import polars as pl

ROOT = Path(r"D:\amazon-ml")
RUNS = ROOT / "runs"
LOG = RUNS / "night4_mix.log"
HARD_STOP = time.time() + 5 * 3600


def log(m):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {m}\n")


def newest(name):
    ds = sorted(d for d in RUNS.glob(f"*-{name}") if d.is_dir())
    return ds[-1] if ds else None


log("waiting for night3")
while "night3 report written" not in ((RUNS / "night3_queue.log").read_text(encoding="utf-8") if (RUNS / "night3_queue.log").exists() else ""):
    if time.time() > HARD_STOP:
        log("night3 not finished; abort")
        sys.exit(0)
    time.sleep(30)
# rows logged by night3: "<file>: PASS val .. dense .. fr .. est .. shape"
rows = {}
for line in (RUNS / "night3_queue.log").read_text(encoding="utf-8").splitlines():
    m = re.search(r"\] (\S+): (PASS|FAIL) val (\S+) dense (\S+) fr (\S+) est (\S+) ", line)
    if m and m.group(2) == "PASS" and m.group(6) != "None":
        rows[m.group(1)] = {"dense": float(m.group(4)), "fr": float(m.group(5)), "est": float(m.group(6))}
log(f"candidates {rows}")
if not rows:
    sys.exit(0)
# US/India source: best dense val; France source: best proxy (tie -> better dense)
us_src = max(rows, key=lambda k: rows[k]["dense"])
fr_src = max(rows, key=lambda k: (rows[k]["fr"], rows[k]["dense"]))
base_line = next(l for l in (RUNS / "night3_queue.log").read_text(encoding="utf-8").splitlines() if "v9_ce2:" in l) if any(
    "v9_ce2:" in l for l in (RUNS / "night3_queue.log").read_text(encoding="utf-8").splitlines()) else None
best_single = max(rows, key=lambda k: rows[k]["est"])
if us_src == fr_src:
    log(f"mix = single file {us_src}; nothing to do")
    sys.exit(0)
# est of the mix: US/India part from us_src, France part from fr_src
est_mix = rows[us_src]["est"] - 0.15 * rows[us_src]["fr"] + 0.15 * rows[fr_src]["fr"]
log(f"mix: US/India from {us_src} (est {rows[us_src]['est']:+.5f}), France from {fr_src} (proxy {rows[fr_src]['fr']:+.5f}) -> est {est_mix:+.5f}; best single {best_single} {rows[best_single]['est']:+.5f}")
out = RUNS / f"{time.strftime('%Y%m%d-%H%M%S')}-mix"
(out / "output").mkdir(parents=True)
s1 = pl.read_parquet(ROOT / "cache" / "raw_test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "source1_entity_id"})
fr_ids = s1.filter(pl.col("country").str.to_lowercase().str.strip_chars() == "france").select("source1_entity_id")
rd = lambda n: pl.read_csv(newest(n) / "output" / "matching_results.tsv", separator="\t", quote_char=None, infer_schema=False)  # noqa: E731
a, b = rd(us_src), rd(fr_src)
mixed = (a.join(b.join(fr_ids, on="source1_entity_id").rename({"matched_entity_ids": "fr"}), on="source1_entity_id", how="left",
                maintain_order="left")
         .with_columns(pl.when(pl.col("source1_entity_id").is_in(fr_ids["source1_entity_id"].implode()))
                       .then(pl.col("fr")).otherwise(pl.col("matched_entity_ids")).alias("matched_entity_ids")).drop("fr"))
assert mixed.height == a.height
mixed.with_columns(pl.col("matched_entity_ids").fill_null("")).write_csv(out / "output" / "matching_results.tsv", separator="\t", quote_style="never")
shutil.copy(newest(us_src) / "output" / "candidate_pairs.tsv", out / "output" / "candidate_pairs.tsv")
res = subprocess.run([sys.executable, str(ROOT / "student_resource" / "utils" / "validate_submission.py"), "--matching",
                      str(out / "output" / "matching_results.tsv"), "--candidate", str(out / "output" / "candidate_pairs.tsv"),
                      "--test-dir", str(ROOT / "student_resource" / "dataset" / "test"), "--check-ids"],
                     capture_output=True, text=True, encoding="utf-8", errors="replace")
ok = res.returncode == 0
(out / "metrics.json").write_text(json.dumps({"mix_us_india": us_src, "mix_france": fr_src, "est": est_mix, "validator_pass": ok}))
log(f"mix validator {'PASS' if ok else 'FAIL'}; {out.name}")
picked = False
if ok and est_mix >= max(rows[best_single]["est"], 0.0) + 0.0002 and est_mix >= 0.0005:
    dst = ROOT / "output" / "SUBMIT_THIS"
    shutil.copy(out / "output" / "matching_results.tsv", dst / "matching_results.tsv")
    shutil.copy(out / "output" / "candidate_pairs.tsv", dst / "candidate_pairs.tsv")
    (dst / "WHAT_IS_THIS.txt").write_text(f"mix: US/India rows from {us_src}, France rows from {fr_src}\nestimated LB change vs v9_ce "
                                          f"{est_mix:+.5f} (best single file {best_single} {rows[best_single]['est']:+.5f})\n"
                                          f"validator --check-ids PASS\n{time.ctime()}\n", encoding="utf-8")
    picked = True
if ok:
    shutil.copy(out / "output" / "matching_results.tsv", ROOT / "output" / "submissions" / "mix_matching_results.tsv")
with open(RUNS / "NIGHT_REPORT.md", "a", encoding="utf-8") as f:
    f.write(f"\n## Per-country mix\nUS/India rows from {us_src}, France rows from {fr_src}: est LB change {est_mix:+.5f}, "
            f"validator {'PASS' if ok else 'FAIL'}; best single file {best_single} {rows[best_single]['est']:+.5f}. "
            f"{'**Mix is now SUBMIT_THIS.**' if picked else 'SUBMIT_THIS unchanged.'} Copy: output/submissions/mix_matching_results.tsv\n")
log(f"mix done, picked={picked}")
