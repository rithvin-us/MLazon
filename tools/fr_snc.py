"""SNC / EI: legal forms the normaliser does not know (so legal_conflict cannot fire). Aggregates only.
1. Train: which legal tokens occur (almost) only in the pool, and how often a pool record whose legal form differs from
   its S1's is a true match (labelled val, per S1-token / cand-token combination).
2. France test: candidates carrying snc / ei, by S1 legal form and score band, and how many the final rule selects."""
import sys

import polars as pl

sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
import pipeline as pp  # noqa: E402
from normalize import LEGAL_TOKENS  # noqa: E402

R = pp.RUNS_DIR
EXTRA = ["snc", "ei", "eirl", "gie", "selarl", "selurl", "scp", "scm", "sl", "spa", "pllc", "pc", "pa", "opc", "huf"]
TOK = sorted(LEGAL_TOKENS | set(EXTRA))


def legal_of(col):
    return pl.col(col).fill_null("").str.split(" ").list.eval(pl.element().filter(pl.element().is_in(TOK))).list.unique().list.sort().list.join("+")


# 1a token frequency S1 vs pool per country
for split in ("train", "test"):
    for c in ("us", "india", "france"):
        s1 = pp.scan_norm(split, "s1").filter(pl.col("country_n") == c).select("name_full")
        po = pp.scan_norm(split, "pool").filter(pl.col("country_n") == c).select("name_full")
        n1, n2 = s1.select(pl.len()).collect().item(), po.select(pl.len()).collect().item()
        if not n1:
            continue
        cnt = lambda lf: (lf.select(pl.col("name_full").str.split(" ").alias("t")).explode("t").filter(pl.col("t").is_in(TOK))  # noqa: E731
                          .group_by("t").agg(pl.len().alias("n")).collect())
        j = cnt(s1).rename({"n": "s1"}).join(cnt(po).rename({"n": "pool"}), on="t", how="full", coalesce=True).fill_null(0)
        j = j.with_columns((pl.col("s1") / n1 * 100).round(2).alias("s1%"), (pl.col("pool") / n2 * 100).round(2).alias("pool%")).sort("pool", descending=True)
        print(f"[{split} {c}] S1 {n1:,} pool {n2:,}: " + ", ".join(f"{t} {a}%/{b}%" for t, a, b in j.select("t", "s1%", "pool%").iter_rows() if b >= 0.05 or a >= 0.05))

# 1b labelled val: match rate by (S1 legal, cand legal) when they differ
vs = pl.read_parquet(sorted(R.glob("*-v10"))[-1] / "val_scored.parquet").select("s1_idx", "cand_idx", "label", "p").filter(pl.col("cand_idx") >= 0)
n1 = pp.scan_norm("train", "s1").select(pl.col("idx").alias("s1_idx"), pl.col("name_full").alias("n1")).collect()
n2 = pp.scan_norm("train", "pool").select(pl.col("idx").alias("cand_idx"), pl.col("name_full").alias("n2")).collect()
vs = vs.join(n1, on="s1_idx").join(n2, on="cand_idx").with_columns(l1=legal_of("n1"), l2=legal_of("n2")).drop("n1", "n2")
d = vs.filter((pl.col("l1") != pl.col("l2")) & (pl.col("p") >= 0.2))
print("[val] pairs p>=0.2 whose legal forms differ, top combinations: n, true-match rate, mean p")
print(d.group_by("l1", "l2").agg(pl.len().alias("n"), pl.col("label").mean().round(3).alias("match_rate"), pl.col("p").mean().round(3).alias("mean_p"))
      .sort("n", descending=True).head(14))
print(f"[val] all pairs p>=0.2: n {vs.filter(pl.col('p') >= 0.2).height:,}, match rate {vs.filter(pl.col('p') >= 0.2)['label'].mean():.3f}")
del vs, n1, n2

# 2 France test with the round-1 model's scores
fr = pp.scan_norm("test", "s1").filter(pl.col("country_n") == "france").select(pl.col("idx").alias("s1_idx"), pl.col("name_full").alias("n1")).collect()
n2 = pp.scan_norm("test", "pool").filter(pl.col("country_n") == "france").select(pl.col("idx").alias("cand_idx"), pl.col("name_full").alias("n2")).collect()
sc = pl.read_parquet(sorted(R.glob("*-v10sfr_dense"))[-1] / "pred" / "scored-*.parquet").select("s1_idx", "cand_idx", "p").join(fr, on="s1_idx").join(n2, on="cand_idx")
sc = sc.with_columns(l1=legal_of("n1"), l2=legal_of("n2")).drop("n1", "n2")
ex = pp.exclusive(sc.filter(pl.col("p") >= 0.02)).with_columns((pl.col("p") >= 0.92).alias("sel"))
print(f"[France] scored pairs {sc.height:,}; selected (excl, p>=0.92) {ex['sel'].sum():,}")
for tok in ("snc", "ei"):
    g = ex.filter(pl.col("l2").str.contains(rf"(^|\+){tok}($|\+)") & ~pl.col("l1").str.contains(rf"(^|\+){tok}($|\+)"))
    print(f"[France] candidate has {tok}, S1 does not: pairs {g.height:,}, selected {g['sel'].sum():,}; by S1 legal form:")
    print(g.group_by("l1").agg(pl.len().alias("n"), pl.col("sel").sum().alias("selected"), pl.col("p").mean().round(3).alias("mean_p"),
                               (pl.col("p") >= 0.98).mean().round(3).alias("p>=0.98")).sort("n", descending=True).head(8))
kn = ex.filter((pl.col("l1") != "") & (pl.col("l2") != "") & (pl.col("l1") != pl.col("l2")) & ~pl.col("l2").str.contains("snc|ei"))
print(f"[France] known legal forms that differ (legal_conflict-like): pairs {kn.height:,}, selected {kn['sel'].sum():,}, mean p {kn['p'].mean():.3f}")
