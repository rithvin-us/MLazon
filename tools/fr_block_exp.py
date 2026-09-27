"""France blocking experiment (label-free): recall of 'easy copies' (exact name_core, address token-set >= 90, same or
missing house number) under wider retrieval / higher candidate cap / higher key df cutoff. Test split, France sample.
Also India/US samples for reference.   python fr_block_exp.py [country] [n]"""
import sys
import time

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
import pipeline as pp  # noqa: E402
from config import Config  # noqa: E402

C = sys.argv[1] if len(sys.argv) > 1 else "france"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
cfg = Config()
cfg.extra.update(max_df=600, rr_tau=0.002, rr_min=3, rr_keep=40, rr_wide=(300, 60, 60), n_jobs=12)
cfg.n_jobs = 12
reranker = pp.load_reranker(pp.RUNS_DIR / "20260926-141000-v9", "cuda")
s1 = pp.scan_norm("test", "s1").filter(pl.col("country_n") == C).select(pp.NORM_COLS).collect().sample(N, seed=11)
pool = pp.scan_norm("test", "pool").filter(pl.col("country_n") == C).select(pp.NORM_COLS).collect()
# easy copies
j = s1.select("idx", "name_core", "addr").join(pool.select(pl.col("idx").alias("cand_idx"), "name_core", pl.col("addr").alias("ap")), on="name_core")
j = j.filter(pl.col("ap").fill_null("") != "")
j = j.with_columns(pl.Series("as", cpdist(j["addr"].fill_null("").to_list(), j["ap"].fill_null("").to_list(), scorer=fuzz.token_set_ratio, workers=-1)),
                   pl.col("addr").str.extract(r"\b(\d+)", 1).alias("h1"), pl.col("ap").str.extract(r"\b(\d+)", 1).alias("h2"))
easy = j.filter((pl.col("as") >= 90) & (pl.col("h1").is_null() | pl.col("h2").is_null() | (pl.col("h1") == pl.col("h2"))))
easy = easy.select(pl.col("idx").alias("s1_idx"), "cand_idx")
print(f"[{C}] sample S1 {N}, easy copies {easy.height} ({easy.height / N:.2f}/S1)", flush=True)
import xgboost as xgb  # noqa: E402

for max_df in (600, 3000):
    cfg.extra["max_df"] = max_df
    t = time.time()
    index, hit = pp.country_index("test", C, pool, cfg)
    print(f"max_df {max_df}: index {'cached' if hit else 'built'} in {time.time() - t:.0f}s, keys kept {len(index.idf):,}", flush=True)
    for wide in ((300, 60, 60), (600, 120, 120), (1200, 200, 200)):
        cfg.extra["rr_wide"] = wide
        t = time.time()
        outs = []
        for k in range(0, s1.height, 2000):
            w = pp.wide_query(index, s1.slice(k, 2000), pool, cfg)
            names = reranker.feature_names or pp.RR_FEATURES
            w = w.with_columns(pl.Series("rr", reranker.predict(xgb.DMatrix(pp._rr_matrix(w, names), feature_names=names)), dtype=pl.Float32))
            outs.append(w.select("s1_idx", "cand_idx", "rr").with_columns(pl.col("rr").rank("ordinal", descending=True).over("s1_idx").alias("rr_rank")))
        w = pl.concat(outs)
        e = easy.join(w, on=["s1_idx", "cand_idx"], how="left")
        res = {"wide_recall": round(e["rr"].is_not_null().mean(), 4), "wide_per_S1": round(w.height / N, 1)}
        for cap in (40, 80, 150, 400):
            kept = (e["rr_rank"] <= cap) & ((e["rr_rank"] <= 3) | (e["rr"] >= 0.002))
            n_c = w.filter((pl.col("rr_rank") <= cap) & ((pl.col("rr_rank") <= 3) | (pl.col("rr") >= 0.002))).height / N
            res[f"cap{cap}"] = (round(float(kept.fill_null(False).mean()), 4), round(n_c, 1))
        print(f"  wide {wide}: {res}  [{time.time() - t:.0f}s]", flush=True)
    del index
