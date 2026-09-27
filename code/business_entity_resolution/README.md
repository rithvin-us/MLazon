# Business Entity Resolution — reproduction guide

Matches every Source 1 record to its Source 2 / Source 3 records. Pure Python pipeline: learned blocking
(inverted index + XGBoost re-ranker) → pairwise + group-consensus features → XGBoost matcher (GPU if
available) → cross-encoder re-scoring of uncertain pairs (MiniLM, Apache-2.0, 22M params) →
exclusivity-aware decision. Uses only the provided training/test data; no external lookups of any kind.
Methodology: see `Documentation_template.md` in the submission root.

## Environment
```
python -m venv .venv311
.venv311\Scripts\python.exe -m pip install -r code/business_entity_resolution/requirements.txt
```
Tested on Python 3.11.9, Windows 11, 16 GB RAM, RTX 4050 6 GB (CUDA). Without a GPU add `--device cpu`
(slower, same results up to floating point).

## Data layout (repo root)
```
student_resource/dataset/train/train_source{1,2,3}.tsv, train_ground_truth.tsv
student_resource/dataset/test/test_source{1,2,3}.tsv
```
Override the root with env `ER_ROOT`, the dataset dir with `ER_DATA`. Caches go to `cache/`, run
artefacts to `runs/<run_id>/`, final files to `output/`. All temp files stay under the repo root.

## End-to-end (data → blocking → matching → output)
Run from the repo root. `SRC=code/business_entity_resolution/src`.
```
# 1. normalise all sources (streamed, multiprocess) -> cache/norm_*
python $SRC/pipeline.py prep --prep-workers 4

# 2. train: fit blocking re-ranker, block + featurise train S1, train matcher, tune decision on validation
python $SRC/pipeline.py train --name final --train-max-s1 1600000 --max-df 600 --rounds 6000 \
       --rr-fit-s1 15000 --rr-tau 0.002 --rr-min 3

# 3. predict test with that run's model + re-ranker + tuned decision; writes and validates
#    output/matching_results.tsv and output/candidate_pairs.tsv
python $SRC/pipeline.py predict --run <train_run_id> --name final_test --max-df 600 \
       --rr-tau 0.002 --rr-min 3
```
```
# 4. stage 3: cross-encoder on the uncertain band (one-time download of the Apache-2.0 base model, 90 MB;
#    after that everything runs offline)
python -c "from huggingface_hub import snapshot_download as s; s('cross-encoder/ms-marco-MiniLM-L12-v2')"
python $SRC/pipeline.py ce-train --run <train_run_id> --name final_cetrain \
       --ce-base cross-encoder/ms-marco-MiniLM-L12-v2 --ce-pairs 1000000 --ce-epochs 2   # -> models/ce_<train_run_id>
# ce-apply reads val scores from <train_run_id> and test scores from --feats-run (the predict run's pred/)
python $SRC/pipeline.py ce-apply --run <train_run_id> --feats-run <predict_run_id> --name final_ce
```
Step 4 re-writes and re-validates `output/matching_results.tsv`; `output/candidate_pairs.tsv` is unchanged
(the cross-encoder only re-scores pairs already in the candidate set).

`<train_run_id>` is printed by step 2 and stored in `runs/LATEST` (format `YYYYMMDD-HHMMSS-final`).
Blocking and re-ranker settings (`--max-df`, `--rr-*`) must match between train and predict.
Step 3 runs the official validator (`student_resource/utils/validate_submission.py`) on the matching
file and a streamed format check on the candidate file (the official validator's candidate check needs
more RAM than 16 GB machines have for this candidate volume).

Stage 3 can stack several fine-tuned cross-encoders: pass a comma list to `--ce-dir` and the stacker uses all of
them. Second cross-encoder: `BAAI/bge-reranker-base` (MIT, 278M, multilingual; input embeddings frozen, batch 32
so it trains in 6 GB):
```
python $SRC/pipeline.py ce-train --run <train or augmented run> --ce-base BAAI/bge-reranker-base --ce-pairs 800000 \
       --ce-epochs 1 --ce-dir models/ce_bge_base
python $SRC/pipeline.py ce-apply --run <train run> --feats-run <predict run> \
       --ce-dir models/ce_<train_run_id>,models/ce_bge_base --name final_ce2
```
```
# 5. decision matched to the test split's distractor density
python $SRC/pipeline.py redecide --run <final_ce_run_id> --name final_dense
# 6. per-country model choice: labelled countries from the best-validated model, countries without labels (France)
#    from the model that transfers best to an unseen country (v10s: `retrain --drop-feats <address IDF features>`)
python $SRC/pipeline.py mix --run <v10 dense run> --unlabelled-from <v10s dense run> --name final_mix
```
Final file of 2026-09-27 (reproduced byte-for-byte by step 6): US/India rows from v10 + CE + `redecide`, France rows
from v10s + CE + `redecide`. v10s = `retrain --drop-feats n_rep_idf,n_miss_idf,a_idf_jacc,a_idf_miss1,a_idf_miss2,a_rep,a_rep_idf,a_miss_idf,a_idf_rank`.
The test pool has ~5.8 Source 2/3 records per S1 against 4.7 in train with the same ~3.46 true matches, i.e.
about twice the unmatched look-alikes. `redecide` copies validation negatives per score band until the band
counts per S1 match the test scores, re-tunes the decision on that, and writes/validates the output again.

Features: `train`/`predict` compute all `FEATURES` in `features.py`, including the v10 country-IDF token
agreement features. Fast path used during development, identical values: `augment --run <run> --split train`
(or `--split test` on a `--save-test-feats` predict run) adds the v10 columns to saved features, then
`retrain --run <augmented run>` and `rescore --run <retrain run> --feats-run <augmented test run>`.

Other commands: `decide --run <id>` (re-apply a decision to saved test scores), `rescore --run <model_run>
--feats-run <run with --save-test-feats>` (score saved test features with another model),
`retrain --crowd-w 3` (3x weight on training rows of S1 with >= 15 candidates), `--drop-feats a,b` (ablation),
`submit --run <test_run_id> --team <name>` (build the submission zip).

## Outputs
- `output/matching_results.tsv` — final matches (one row per test S1, empty list for singletons)
- `output/candidate_pairs.tsv` — exactly the candidate pairs the matcher scores (post re-ranker); every
  matched id is a candidate

## Source map (`src/`)
| file | role |
|---|---|
| `pipeline.py` | CLI and stages: prep, re-ranker fit, blocking + features, training, decision tuning, test prediction, validation, zip |
| `prep.py` | streamed normalisation into parquet caches (multiprocess) |
| `normalize.py`, `indic.py` | name/address normalisation; Devanagari → Latin transliteration dictionary learned from train ground truth |
| `blocking.py` | IDF-weighted inverted index over hashed name/address keys (memory-mapped CSR) |
| `features.py` | re-ranker sims, pairwise features, group-consensus features |
| `cross_encoder.py` | stage 3: MiniLM cross-encoder fine-tuning, band re-scoring, monotone stacking, re-decision |
| `stage2.py` | optional post-hoc re-scoring with group context (not used for the final file) |
| `config.py`, `io_utils.py`, `tracking.py`, `hwmon.py` | paths/knobs, loading, run registry, hardware log |
