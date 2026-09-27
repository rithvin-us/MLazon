"""Final package: output/ from SUBMIT_THIS, pipeline src + README + requirements, the France adaptation scripts
(france/), documentation. Checks the packed output files byte-for-byte against SUBMIT_THIS.
  python make_final_zip.py [Techiva_final.zip]"""
import hashlib
import sys
import zipfile
from pathlib import Path

ROOT = Path(r"D:\amazon-ml")
SUB = ROOT / "output" / "SUBMIT_THIS"
PROJ = ROOT / "code" / "business_entity_resolution"
OUTS = ("matching_results.tsv", "candidate_pairs.tsv")
zp = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "Techiva_final.zip")


def digest(fh):
    h = hashlib.sha256()
    for chunk in iter(lambda: fh.read(1 << 20), b""):
        h.update(chunk)
    return h.hexdigest()


with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for fn in OUTS:
        z.write(SUB / fn, f"output/{fn}")
    for f in sorted((PROJ / "src").glob("*.py")):
        z.write(f, f"code/business_entity_resolution/src/{f.name}")
    for fn in ("README.md", "requirements.txt"):
        z.write(PROJ / fn, f"code/business_entity_resolution/{fn}")
    for fn in ("build_fr_pseudo.py", "fr_chain.py", "fr_bge_chain.py", "fr_combo.py"):
        z.write(ROOT / "tools" / fn, f"code/business_entity_resolution/france/{fn}")
    z.write(ROOT / "docs" / "Documentation_template.md", "Documentation_template.md")
with zipfile.ZipFile(zp) as z:
    bad = z.testzip()
    same = {}
    for fn in OUTS:
        with z.open(f"output/{fn}") as a, open(SUB / fn, "rb") as b:
            same[fn] = digest(a) == digest(b)
    names = z.namelist()
print(f"{zp.name}: {len(names)} files, {zp.stat().st_size / 1e6:.0f} MB, CRC {'OK' if bad is None else 'BAD ' + bad}, "
      f"identical to SUBMIT_THIS: {same}")
print("\n".join(names))
