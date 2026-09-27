"""Stage 2 (competition + S1-context features on the stacked p) under the test split's distractor density:
validation augmented as in `redecide` (density_augment: copies of val negatives per p band until pairs per S1 match
the test scores). Compares, on the dense validation, the pipeline's dense-tuned decision on p vs stage 2 trained
(a) on dense val (OOF by S1) and (b) on plain val then applied to dense val (robustness to the density shift);
plus feature sets: all / without the per-S1 count features (s1_sum, s1_n02) / competition only."""
import sys

import numpy as np
import polars as pl
import xgboost as xgb

sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
import pipeline as pp  # noqa: E402
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from stage2_margin_lib import feats, FEAT  # noqa: E402

R = pp.RUNS_DIR
SRC = sys.argv[1] if len(sys.argv) > 1 else "v10_bge2"
v = pl.read_parquet(sorted(R.glob(f"*-{SRC}"))[-1] / "val_scored_ce.parquet")
vt = pl.read_parquet(sorted(R.glob("*-v10"))[-1] / "val_truth.parquet")
test_sc = pl.read_parquet(sorted(R.glob(f"*-{SRC}_dense"))[-1] / "pred" / "scored-*.parquet", columns=["s1_idx", "cand_idx", "p"])
aug, factors = pp.density_augment(v, vt, test_sc, seed=42)
tvc = vt.select("s1_idx", "n_true")
print(f"[{SRC}] val rows {v.height:,} -> dense {aug.height:,}")


class _Q:
    def log(self, m):
        pass


def fit_oof(d, cols, train_on=None):
    """OOF p2 for val S1 rows of d (5 folds by S1); competitor rows by the fold models' mean. train_on: another frame
    (plain val) to fit on instead, same folds."""
    mono = "(" + ",".join("1" if f in ("p", "margin", "s1_max") else ("-1" if f in ("max_other", "rank_rec") else "0") for f in cols) + ")"
    prm = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", device="cuda", eta=0.05, max_depth=4, min_child_weight=20,
               subsample=0.8, monotone_constraints=mono)
    fold = (pl.col("s1_idx") * 7919 % 5).alias("fold")
    dv = d.join(vt.select("s1_idx"), on="s1_idx", how="semi").with_columns(fold)
    dc = d.join(vt.select("s1_idx"), on="s1_idx", how="anti")
    src = (train_on if train_on is not None else d).join(vt.select("s1_idx"), on="s1_idx", how="semi").with_columns(fold)
    oof, cp = np.zeros(dv.height, np.float32), np.zeros(dc.height, np.float32)
    for k in range(5):
        tr = src.filter(pl.col("fold") != k)
        b = xgb.train(prm, xgb.DMatrix(tr.select(cols).to_numpy(), tr["label"].to_numpy(), feature_names=cols), 400)
        m = (dv["fold"] == k).to_numpy()
        oof[m] = b.predict(xgb.DMatrix(dv.filter(pl.col("fold") == k).select(cols).to_numpy(), feature_names=cols))
        cp += b.predict(xgb.DMatrix(dc.select(cols).to_numpy(), feature_names=cols)) / 5
    return pl.concat([dv.drop("fold").with_columns(pl.Series("p2", oof)), dc.with_columns(pl.Series("p2", cp))])


def best(df, col):
    return max(pp.tune_decision(df.select("s1_idx", "cand_idx", "label", pl.col(col).alias("p")), tvc, 0.02, _Q()).values(), key=lambda z: z["f05"])


da = feats(aug)
dp = feats(v)
b0 = best(da, "p")
print(f"[dense] stacked p, dense-tuned decision: {b0['mode']} {b0['param']} F0.5 {b0['f05']:.5f} P {b0['precision']:.4f} R {b0['recall']:.4f}")
for name, cols in (("all features", FEAT), ("no per-S1 counts", [f for f in FEAT if f not in ("s1_sum", "s1_n02")]),
                   ("competition only", ["p", "max_other", "margin", "n_other", "rank_rec"])):
    r_a = best(fit_oof(da, cols), "p2")
    r_b = best(fit_oof(da, cols, train_on=dp), "p2")
    print(f"[dense] stage 2 {name:17s}: trained dense F0.5 {r_a['f05']:.5f} ({r_a['f05'] - b0['f05']:+.5f}) {r_a['mode']} {r_a['param']} | "
          f"trained plain -> dense F0.5 {r_b['f05']:.5f} ({r_b['f05'] - b0['f05']:+.5f})")
