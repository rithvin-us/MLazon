# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Techiva  
**Team Members:** Rithvin U S, Preethika Kumaravel, Shreenithi N  
**Submission Date:** 2026-09-27

---

## 1. Executive Summary
A two-stage *learned blocking* pipeline feeds a gradient-boosted matcher. Blocking retrieves a wide candidate
pool from an IDF-weighted inverted index over hashed name and address keys, then a small XGBoost re-ranker
keeps only a handful of candidates per Source 1 record (11.2 on average on test, down from ~350 retrieved)
while keeping ~98.5% of true matches. The matcher combines ~70 pairwise and **group-consensus** features (does
this candidate agree with the S1's other strongest candidates?); a fine-tuned MiniLM cross-encoder re-scores
only the uncertain pairs, and a decision rule is tuned for macro F0.5 under the one-owner-per-record structure
of the ground truth.

---

## 2. Methodology

### 2.1 Problem Analysis
- **Scale:** 2.2M train / 1.73M test S1; ~10M Source 2+3 records per split. 16 GB RAM machine, so every stage
  is streamed per country and per S1 chunk, with parquet spills on disk.
- **Ground truth is exclusive:** each S2/S3 record matches at most one S1 (7.64M train pairs, none shared).
  An S1 has 3.46 matches on average; ~5% of S1 are singletons.
- **Noise in true matches:** typos (`sager`/`5ager`), word transpositions, legal-suffix swaps, added words
  (`services`, `center`), handle/domain forms (`lcprivate` for *Lex Communication Pvt*, `creativeinternational`),
  names replaced by pseudo-words with an intact address (`zephtavo`), house-number typos or truncation
  (`8250→8252`, `27724→2772`), dropped/reordered address parts, Devanagari names, empty addresses.
- **Hard negatives:** unmatched records are mostly *copies of a real entity with one field nudged*: house number
  `1030→1031`, one name word `medical→media`. At the pair level they look like typo-positives; they only stand
  out against the entity's other copies.
- **France** appears only in test: nothing country-specific is learned; the index, vocabularies and stop tokens
  are built per country label from the (unlabelled) data of the split itself.

### 2.2 Solution Strategy
**Approach Type:** Learned blocking + gradient-boosted classifier + exclusivity-aware decision (hybrid)  
**Core Innovation:** (1) a learned re-ranker inside blocking, which lifts candidate recall from 0.965 to 0.985
while cutting candidates per S1 from 40 to ~10; (2) group-consensus features that expose "one field nudged"
distractors; (3) a cross-encoder applied only to the uncertain band (~1.5 pairs per S1); (4) decision tuning
with *competitor* S1s so validation sees the same record-ownership competition as the full test set.

---

## 3. Candidate Generation (Blocking)
1. **Normalisation** (`normalize.py`): look-alike digits inside name words are mapped back to letters
   (`5atguru`→`satguru`, `8ashaw`→`bashaw`, `cu1tural`→`cultural`; real numbers such as `shop15` stay), then
   Unicode NFKC + transliteration (Devanagari via a dictionary learned from
   train ground truth, accents stripped), legal-form canonicalisation (`private limited→pvt ltd`,
   `sasu→sas`), address abbreviation maps (`road→rd`, `r./rue`, `allée→all`), state/region codes, postcode and
   house-number extraction. Output: `name_full`, `name_core` (no legal/stop words), `name_skel` (consonant
   skeleton), `addr`, `postcode`, `addr_nums`.
2. **Inverted index per country** (`blocking.py`): 10 hashed key types — name tokens, name bigrams, 4-char
   prefixes, consonant-skeleton tokens, compact-name prefix and exact compact name, address tokens, adjacent
   address token pairs, postcode, first-name-token × house number. Keys with document frequency > 600 are
   dropped; score = Σ key weight × IDF. Stored as memory-mapped CSR arrays.
3. **Wide retrieval:** top 300 by total score + top 60 by name-only and 60 by address-only score (~350 per S1).
4. **Learned re-ranker:** XGBoost on the key scores/ranks plus five cheap fuzzy similarities (compact-name
   ratio, full-name partial ratio, name/address token-set ratio, house-number equality), fit on train S1 that
   are in neither the matcher's training nor its validation set.
5. **Final candidate set** (`candidate_pairs.tsv`): top-3 by re-ranker score always, plus any candidate with
   re-ranker probability ≥ 0.002, capped at 40. This is exactly the set the matcher scores.

- **Blocking keys used:** name tokens / bigrams / prefixes / skeleton, compact name, address tokens and token
  pairs, postcode, name-token × house number; learned re-ranking on top.
- **Candidate pairs generated:** 19,481,044 on test = 11.24 per S1 (was 69.2M / 39.9 per S1 with a fixed
  top-40); 9.0 per S1 on validation.
- **How true matches were not lost:** two independent channels (name-only and address-only extras) so records
  with an empty address or a renamed business still enter the wide pool; the re-ranker is judged on recall of
  the wide pool; validation recall is tracked for every run (0.9645 plain IDF top-40 → 0.9875 re-ranked top-40
  → 0.9855 with the adaptive candidate set at 9.0 per S1).

---

## 4. Matching Model

**Features used (~70):**
- Name: token-set / token-sort / partial / plain ratio and Jaro-Winkler on full and core name; compact-name
  ratio and partial; consonant-skeleton ratios; first-token equality; initials-as-handle prefix; share of
  candidate name tokens never seen in the country's S1 vocabulary (renamed copies); legal-form agreement and
  conflict; name frequencies in pool and S1 (chains / generic names).
- Address: token-set / partial / plain ratio; the same after removing city/state-level tokens (tokens in
  ≥0.2% of the country's S1 addresses); postcode equality; number-set Jaccard; house-number equality, fuzzy
  similarity and prefix (truncation) relation; empty-address flags; lengths.
- Blocking: key scores (total / name / address, raw and normalised), ranks, re-ranker score and rank.
- **Group consensus:** similarity of the candidate to the S1's other top-3 re-ranked candidates (name and
  address mean/max, house-number vote), `twin_better` (a near-identical sibling has the exact house number
  and this one does not), candidates per S1, gaps to the S1's best, within-S1 ranks of key similarities.

- **Country-IDF token agreement (v10):** name and address tokens weighted by their inverse document frequency within
  the record's own country (S1 plus Source 2/3 of the split, unlabelled). Near-identical leftovers (typos,
  truncations) count as shared; features are the weighted Jaccard, the weighted share of each side left unmatched,
  "exactly one word replaced", the IDF of the most distinctive differing word, and within-S1 ranks. Where names and
  streets come from a small vocabulary (France: generic words, ~15 cities, 15k address words) plain fuzzy ratios stay
  high for different businesses; IDF weighting lets the distinctive words decide.

**Model type:** XGBoost (`hist`, CUDA), depth 8, learning rate 0.05, early stopping on a 5% holdout of the
training S1 by log-loss (the decision relies on calibrated probabilities), trained on 1.6M S1 (14.3M pairs),
up to 6,000 rounds (best iteration 5,437).
Training data is streamed from per-chunk parquet files into a `QuantileDMatrix`.

**Stage 3 — cross-encoder on the uncertain band:** `cross-encoder/ms-marco-MiniLM-L6-v2` (Apache-2.0, 22M
parameters; the L12 variant, 33M parameters, in the final run) fine-tuned for 2 epochs on 1M training pairs ("name | address" of S1 vs candidate, balanced
positives / hard negatives). It re-scores only pairs with stage-1 probability in [0.02, 0.995) — 2.7M of the
20.1M test pairs — and a monotone depth-3 XGBoost stacks [logit p, cross-encoder logit] (fit on the validation
band with out-of-fold estimates by block). Band AUC: stage-1 0.952, cross-encoder 0.944, stacked 0.969.

**Threshold selection method:** grid search on validation macro F0.5 over three rules — global threshold,
threshold plus "top-1 rescue" for S1 with nothing above threshold, and per-S1 expected-F0.5 optimisation — each
with and without **exclusivity** (every pool record goes only to the S1 that scores it highest, matching the
ground truth structure). Validation S1 are sampled as whole blocks (country | city | name prefix) and the
ground-truth owners of their candidates are scored as *competitors* (never counted in the metric), so
exclusivity acts as it does on the full test set.

**Test-density matching:** the test split has ~5.8 Source 2/3 records per S1 against 4.7 in train, with the same
~3.46 true matches per S1 (the high-confidence pairs per S1 agree), i.e. about twice the unmatched look-alikes;
uncertain pairs per S1 are ~2x validation's. `redecide` copies validation negatives per score band until each band's
pairs per S1 match the test scores, and re-tunes the decision on that denser validation (a stricter threshold).

**Unseen-country proxy:** France has no labels, so every change aimed at it is checked by training on one labelled
country and scoring the other (US -> India, India -> US) with the France decision rule (threshold whose matches per
S1 equal the training country's). Changes are kept by an estimated leaderboard effect of 0.85 x (validation change) +
0.15 x (proxy change), the countries' test shares.

---

## 5. Results & Error Analysis

| version | change | val F0.5 | public LB |
|---|---|---|---|
| v2 | IDF blocking + pairwise XGBoost | 0.9498 | 0.9357 |
| v4 | Indic transliteration, two-channel blocking, exclusivity | 0.9708 | – |
| v5 | learned re-ranker in blocking | 0.9764 | 0.9685 |
| v6 | group-consensus + sharper pair features, competitor-aware tuning | 0.9812 | – |
| v6 + CE | cross-encoder on the uncertain band | 0.9839 | – |
| v7 | adaptive candidates (11.6/S1 on test), 700k training S1 | 0.9822 | – |
| v7 + CE | cross-encoder MiniLM-L6 | 0.9846 | 0.9739 |
| v8 + CE | 1.3M training S1, MiniLM-L12 | 0.9852 | – |
| v9 + CE | look-alike digit normalisation, 1.6M S1, L12 on 1M pairs, unlabelled-country decision | 0.9859 | 0.9754 |
| v10 + CE | + country-IDF name and address agreement (stage 1 0.9832 -> 0.9850) | 0.9861 | – |
| v10s + CE | + country-IDF name agreement only (the subset that transfers to an unseen country) | 0.9859 | – |
| mix | US/India rows: v10 + CE, test-density decision; France rows: v10s + CE | 0.9860 | 0.9758 |
| mix + French pseudo-labels | France rows from v10s retrained with structure-based French pseudo-labels (confident exclusive owners as positives, records confidently owned by another French S1 as hard negatives); France F0.5 +0.0013 on the leaderboard | 0.9860 | 0.9760 |
| **final** | **France rows: a second pseudo-label round (positives from the round-1 model, hard negatives from both rounds) and a rule-based fallback for the pairs the model is unsure of (0.2 <= p < 0.8); US/India rows unchanged** | **0.9860** | **pending** |

Leaderboard decomposition (one diagnostic submission with the French rows emptied): US/India 0.983, France 0.932 for
v9 + CE. Validation made as distractor-dense as test predicted US/India 0.984, so validation tracks the test closely;
the gap to the top of the leaderboard is mostly France.

- **F_0.5 Score (macro):** 0.9859 on validation (40k block-sampled train S1 never used for training, decision
  tuned with competitor S1); v6 onwards numbers include competitor-aware tuning.
- **Common false positives (wrong merges):** "nudged copies" — same name with the house number moved by a few
  units (`1305` vs `1318 pacific ave`) or one name word swapped (`medical`/`media`), usually in the same city.
- **Common false negatives (missed matches):** renamed records (pseudo-word or handle names) with a partial or
  empty address; heavy address truncation combined with a name typo; S1 with a single true match.

---

## 6. Conclusion
Most of the gain came from treating blocking as a learning problem: a small re-ranker recovers true matches
that pure IDF ranking buried and lets the final candidate set shrink to single digits per S1. Group features
that compare a candidate with the entity's other copies are what separate genuine typos from deliberately
nudged distractors. Everything runs on a 16 GB laptop by streaming per country and per chunk.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/` — `README.md` (exact commands), `requirements.txt` (pinned), `src/`:
`pipeline.py` (entry point: `prep`, `train`, `predict`, `decide`, `rescore`, `submit`), `prep.py`,
`normalize.py`, `indic.py`, `blocking.py`, `features.py`, `stage2.py`, `config.py`, `io_utils.py`,
`tracking.py`, `hwmon.py`. `prep` → `train` → `predict --run <train_run_id>` regenerates both output files;
`ce-train` / `ce-apply` add stage 3, `redecide` the test-density decision, `mix` the per-country model choice, and
`france/` (`build_fr_pseudo.py`, `fr_chain.py`, `fr_combo.py`) the France adaptation of the final file (see README
for the exact commands).

### B. Additional Results
- Final candidate set on test: 19,481,044 pairs (11.24 per S1; 2 S1 with no candidates).
- Unseen country (France): with the validation-tuned threshold the model predicts 3.44 matches per France S1
  against 3.36 for US/India (the generator's match-count distribution is identical across training countries),
  i.e. it over-matches. For any country without validation labels the decision picks the threshold at which its
  matches per S1 equal the labelled countries' (France: 0.895 after the cross-encoder).
- Validation blocking recall by stage: IDF top-40 0.9645 · re-ranked top-40 0.9875 · adaptive (~9) ~0.984.
- Test vs train density: Source 2/3 records per S1 are 5.76 (US), 5.82 (India), 5.53 (France) on test against 4.67 in
  train; pairs with 0.2 <= p < 0.8 per S1 on test are 0.32 (US) / 0.23 (India) against 0.16 / 0.15 on validation.
  Validation made equally dense (`redecide`): v9 + CE 0.9859 -> 0.9841, v10 + CE 0.9861 -> 0.9848 (0.9846 before
  re-tuning), which is closer to what the leaderboard sees.
- Unseen-country proxy (F0.5 on the held-out country, France-style decision; train US -> India / India -> US):
  v9 features 0.9433 / 0.9665 · + all country-IDF features 0.9385 / 0.9529 · + name country-IDF only 0.9463 / 0.9684.
  Address IDF weights do not transfer between countries (their vocabularies differ too much); name IDF weights do.
  Monotone constraints and depth 6 did not help.
- Stage 3 with a stronger reranker: `BAAI/bge-reranker-base` (MIT, 278M, multilingual) fine-tuned on 800k pairs and
  stacked with the MiniLM-L12 cross-encoder: validation 0.98612 -> 0.98623. The ambiguity left on US/India is
  not a model-capacity problem.
- Where the remaining validation loss is (v10 + CE): true matches the model rejects 0.0074 (75% of them are copies
  with an empty address whose exact name also appears on unowned copies: 39% match rate even when unique), blocking
  misses 0.0046, false positives 0.0019. The model is calibrated in every slice checked (empty-address copies by
  name crowding, same-name records across the whole pool, raw spelling identity, copies already confirmed), so
  these are genuinely ambiguous given name and address.
- Unseen-country match count: on the proxy the best matches per S1 for the held-out country is 0.96x (US -> India)
  and 1.00x (India -> US) of the training country's; the France rule keeps 1.00x.
- France adaptation without labels (final file). (1) Structure-based pseudo-labels: French pairs with p >= 0.98 whose
  record has no better-scoring S1 become positives, records owned by another French S1 with p >= 0.98 hard negatives,
  a sample of p < 0.02 pairs easy negatives; the model is retrained on the training pairs plus these (weight 0.5),
  then a second round takes positives from the adapted model and hard negatives from both rounds (weight 1.0). On
  the proxy one round gives +0.0027 (US -> India; pseudo-positive precision 0.991, hard negatives 0.9996 true
  negatives); on the leaderboard France +0.0013. (2) Rule-based fallback: French pairs the model is unsure of
  (0.2 <= p < 0.8) are accepted only when the core-name token-set similarity is >= 95, the address token-set
  similarity >= 90, the house numbers agree and the legal forms do not conflict, and rejected otherwise; proxy
  +0.0023 (US -> India) / +0.0015 (India -> US). On France the band holds 63,206 pairs, the rule accepts 793, and the
  count-matched threshold becomes 0.920 (3.343 matches per S1, as US/India). Together they change 3.1% of the French
  lists against the 0.9760 file.
- Checked on France and not the cause of its gap: blocking (99.2% of near-certain French copies are retrieved),
  tokenisation (IDF tables are built per country from the scored data itself, unseen tokens get the maximum weight;
  character-level fuzzy similarities), formatting variants (departement names, "No.", bis/ter, accents, case: kept
  95.6% of the time), and the threshold (France scores are not deflated: the model over-matched France before the
  country-IDF features, and the count-matched threshold is 0.92, not lower).
- Loss breakdown (v6 validation, F0.5 points lost): model misses 0.0077, S1 with zero correct matches
  0.0044, false positives 0.0039, blocking misses 0.0034, singleton false positives 0.0009.
