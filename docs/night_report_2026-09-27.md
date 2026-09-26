# Night report (2026-09-27)
generated Sun Sep 27 00:57:10 2026

**Recommended file:** `output/SUBMIT_THIS/matching_results.tsv` = **v9_ce (unchanged)**

Estimated LB change = 0.85 x (change in test-density val F0.5) + 0.15 x (change in unseen-country proxy).
Test-density val F0.5 of v9_ce as submitted: 0.98406.

| file | decision | validator | plain val | test-density val | proxy change | est. LB change | matches per S1 |
|---|---|---|---|---|---|---|---|
| v9_ce2 | val-tuned decision | PASS | 0.98585 | 0.98406 | +0.00000 | +0.00000 | France: 3.355/S1; India: 3.336/S1; US: 3.379/S1 |
| v9_ce2_dense | density-matched decision | PASS | 0.98583 | 0.9842 | +0.00000 | +0.00012 | France: 3.347/S1; India: 3.331/S1; US: 3.371/S1 |
| v10_ce | val-tuned decision | PASS | 0.98612 | 0.98461 | -0.00924 | -0.00092 | France: 3.348/S1; India: 3.326/S1; US: 3.37/S1 |
| v10_ce_dense | density-matched decision | PASS | 0.98596 | 0.9848 | -0.00924 | -0.00076 | France: 3.308/S1; India: 3.324/S1; US: 3.365/S1 |
| v10s_ce | val-tuned decision | PASS | 0.98591 | 0.98397 | +0.00244 | +0.00029 | France: 3.349/S1; India: 3.329/S1; US: 3.371/S1 |
| v10s_ce_dense | density-matched decision | PASS | 0.98582 | 0.98409 | +0.00244 | +0.00039 | France: 3.349/S1; India: 3.331/S1; US: 3.369/S1 |

## Unseen-country proxy (train one country, score the other; France-style decision)

subset deltas vs base (tgt_shape, src_val): {'idf': (-0.009240000000000026, 0.0019950000000000245), 'ratio': (0.0011200000000000099, 0.0012650000000000161), 'jacc': (-0.0009199999999999764, 0.0008900000000000019), 'name': (0.0024350000000000205, 0.0010350000000000081), 'addr': (-0.003334999999999977, 0.0004250000000000087)}
subset estimates: {'idf': 0.00030975000000001704, 'ratio': 0.0012432500000000152, 'jacc': 0.0006185000000000052, 'name': 0.00124500000000001, 'addr': -0.00013899999999998907}; retrained: name

```
{
 "us->india base": {
  "src_val": 0.98275,
  "tgt_val": 0.94429,
  "tgt_shape": 0.94331,
  "shape_thr": 0.69,
  "tgt_oracle": 0.94716,
  "iters": 1856,
  "secs": 211
 },
 "us->india idf": {
  "src_val": 0.98459,
  "tgt_val": 0.94329,
  "tgt_shape": 0.93846,
  "shape_thr": 0.54,
  "tgt_oracle": 0.94364,
  "iters": 1671,
  "secs": 208
 },
 "us->india idf+crowd3": {
  "src_val": 0.98421,
  "tgt_val": 0.9428,
  "tgt_shape": 0.93889,
  "shape_thr": 0.58,
  "tgt_oracle": 0.94402,
  "iters": 1430,
  "secs": 182
 },
 "india->us base": {
  "src_val": 0.98147,
  "tgt_val": 0.9623,
  "tgt_shape": 0.96651,
  "shape_thr": 0.81,
  "tgt_oracle": 0.9664,
  "iters": 1819,
  "secs": 176
 },
 "india->us idf": {
  "src_val": 0.98362,
  "tgt_val": 0.95123,
  "tgt_shape": 0.95288,
  "shape_thr": 0.69,
  "tgt_oracle": 0.95478,
  "iters": 1441,
  "secs": 153
 },
 "us->india ratio": {
  "src_val": 0.98369,
  "tgt_val": 0.94761,
  "tgt_shape": 0.94309,
  "shape_thr": 0.56,
  "tgt_oracle": 0.9477,
  "iters": 1619,
  "secs": 204
 },
 "us->india jacc": {
  "src_val": 0.98356,
  "tgt_val": 0.94251,
  "tgt_shape": 0.94148,
  "shape_thr": 0.63,
  "tgt_oracle": 0.94618,
  "iters": 1605,
  "secs": 188
 },
 "us->india name": {
  "src_val": 0.98365,
  "tgt_val": 0.94886,
  "tgt_shape": 0.94626,
  "shape_thr": 0.6,
  "tgt_oracle": 0.95068,
  "iters": 1780,
  "secs": 205
 },
 "us->india addr": {
  "src_val": 0.98298,
  "tgt_val": 0.93895,
  "tgt_shape": 0.937,
  "shape_thr": 0.61,
  "tgt_oracle": 0.9409,
  "iters": 1584,
  "secs": 186
 },
 "india->us ratio": {
  "src_val": 0.98306,
  "tgt_val": 0.96656,
  "tgt_shape": 0.96897,
  "shape_thr": 0.76,
  "tgt_oracle": 0.96923,
  "iters": 1722,
  "secs": 185
 },
 "india->us jacc": {
  "src_val": 0.98244,
  "tgt_val": 0.96395,
  "tgt_shape": 0.9665,
  "shape_thr": 0.76,
  "tgt_oracle": 0.96699,
  "iters": 1785,
  "secs": 169
 },
 "india->us name": {
  "src_val": 0.98264,
  "tgt_val": 0.96597,
  "tgt_shape": 0.96843,
  "shape_thr": 0.77,
  "tgt_oracle": 0.9684,
  "iters": 1560,
  "secs": 156
 },
 "india->us addr": {
  "src_val": 0.98209,
  "tgt_val": 0.96292,
  "tgt_shape": 0.96615,
  "shape_thr": 0.79,
  "tgt_oracle": 0.96629,
  "iters": 1693,
  "secs": 165
 }
}
```
## Final pick (night3, after transfer-robust training)
generated Sun Sep 27 01:29:36 2026

**SUBMIT_THIS = v9_ce (unchanged)**

robust proxy results: {"us->india mono": {"src_val": 0.97748, "tgt_val": 0.93088, "tgt_shape": 0.93457, "iters": 1299, "secs": 134}, "us->india d6": {"src_val": 0.98356, "tgt_val": 0.94826, "tgt_shape": 0.94593, "iters": 3210, "secs": 260}, "us->india mono_d6": {"src_val": 0.97719, "tgt_val": 0.93076, "tgt_shape": 0.93323, "iters": 2583, "secs": 203}, "us->india reg": {"src_val": 0.98357, "tgt_val": 0.94857, "tgt_shape": 0.94501, "iters": 2966, "secs": 286}, "india->us mono": {"src_val": 0.97261, "tgt_val": 0.9562, "tgt_shape": 0.9614, "iters": 1693, "secs": 139}, "india->us d6": {"src_val": 0.98244, "tgt_val": 0.96648, "tgt_shape": 0.96765, "iters": 3593, "secs": 232}, "india->us mono_d6": {"src_val": 0.97223, "tgt_val": 0.95912, "tgt_shape": 0.96159, "iters": 2770, "secs": 180}, "india->us reg": {"src_val": 0.98241, "tgt_val": 0.96551, "tgt_shape": 0.96691, "iters": 2359, "secs": 192}}
robust estimates: {'mono': -0.0070439999999999834, 'd6': 0.0010385000000000088, 'mono_d6': -0.0074149999999999685}; retrained: none

| rank | file | decision | validator | plain val | test-density val | proxy change | est. LB change | matches per S1 |
|---|---|---|---|---|---|---|---|---|
| 1 | v10s_ce_dense | density-matched decision | PASS | 0.98582 | 0.98409 | +0.00244 | +0.00039 | France: 3.349/S1; India: 3.331/S1; US: 3.369/S1 |
| 2 | v10s_ce | val-tuned decision | PASS | 0.98591 | 0.98397 | +0.00244 | +0.00029 | France: 3.349/S1; India: 3.329/S1; US: 3.371/S1 |
| 3 | v9_ce2_dense | density-matched decision | PASS | 0.98583 | 0.9842 | +0.00000 | +0.00012 | France: 3.347/S1; India: 3.331/S1; US: 3.371/S1 |
| 4 | v10_ce_dense | density-matched decision | PASS | 0.98596 | 0.9848 | -0.00924 | -0.00076 | France: 3.308/S1; India: 3.324/S1; US: 3.365/S1 |
| 5 | v10_ce | val-tuned decision | PASS | 0.98612 | 0.98461 | -0.00924 | -0.00092 | France: 3.348/S1; India: 3.326/S1; US: 3.37/S1 |
| 6 | v9_ce2 | val-tuned decision | PASS | 0.98585 | 0.98406 | +0.00000 | +0.00000 | France: 3.355/S1; India: 3.336/S1; US: 3.379/S1 |

## Per-country mix
US/India rows from v10_ce_dense, France rows from v10s_ce_dense: est LB change +0.00099, validator PASS; best single file v10s_ce_dense +0.00039. **Mix is now SUBMIT_THIS.** Copy: output/submissions/mix_matching_results.tsv

## Ensembles + mix2
US/India test-density val by source: {"v9_ce2": 0.98406, "v9_ce2_dense": 0.9842, "v10_ce": 0.98461, "v10_ce_dense": 0.9848, "v10s_ce": 0.98397, "v10s_ce_dense": 0.98409, "ens_v9v10_ce": 0.98444, "ens_v9v10_ce_dense": 0.98444, "ens_v10v10s_ce": 0.98443, "ens_v10v10s_ce_dense": 0.98465}
mix2 = US/India from v10_ce_dense, France from v10s_ce_dense: est +0.00099 vs current +0.00099. SUBMIT_THIS unchanged.
