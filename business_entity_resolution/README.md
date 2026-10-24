# Business Entity Resolution — two-stage blocking + LightGBM

Finds, for every Source-1 entity, the matching Source-2 / Source-3 records.
Stage 1 (blocking) produces `output/candidate_pairs.tsv`; stage 2 (LightGBM + threshold)
produces `output/matching_results.tsv`. Metric: macro F0.5 (singletons count).

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# data: folder that contains train/ and test/  (auto-detected: ./DB, ../DB,
# ../Dataset/student_resource/dataset, or pass --data-dir / set ER_DATA_DIR)
python main.py --mode all                     # train -> models/lgbm_er.pkl, then test -> ../output/*.tsv
python main.py --mode train                   # only train
python main.py --mode test                    # only inference (needs models/lgbm_er.pkl)
python main.py --mode validate                # run utils/validate_submission.py on the outputs
```

Smoke test (minutes, not hours): `python main.py --mode all --dev-frac 0.01`
(`--dev-full-pool` samples only S1 but indexes the whole S2/S3 pool -> realistic difficulty).

Useful flags: `--n-jobs`, `--top-k` (candidates per source per S1, default 15 -> <=30 total),
`--chunk-size` (blocking task size; lower it if RAM is tight), `--n-train-s1` (S1 entities used to fit the
classifier, default 300k, `0` = all), `--optuna-trials N`, `--tau-min/--tau-max`, `--no-one-to-one`.
Normalised data and candidate tables are cached in `work/` (delete it after changing `src/utils.py`
normalisation or `src/blocking.py`).

## Layout

| file | role |
| --- | --- |
| `src/utils.py` | TSV I/O, normalisation (accents, legal suffixes, `dba`, `domain.com`, address abbreviations, Indic transliteration from Unicode names), token-sort key, exact macro F0.5 |
| `src/blocking.py` | per-country sparse TF-IDF blocking, top-K per source, blocking-context features |
| `src/features.py` | pairwise features (RapidFuzz, token/char-n-gram Jaccard, digit overlap, length ratios, script flags, S2/S3 indicator, per-S1 context) |
| `src/pipeline.py` | data prep + caching, chunked scoring, decision rule, scoring |
| `src/train.py` | grouped 80/20 split, LightGBM, tau sweep, artifact `models/lgbm_er.pkl` |
| `src/inference.py` | test-time blocking -> candidates -> scoring -> matches |
| `main.py` | CLI |

## Method

**Normalisation** (country-agnostic): NFKC + accent folding, `&`->and, dotted acronyms (`S.A.S`->`sas`),
`X dba Y`->`Y`, `brand.com - <phone>`->`brand`, honorific/wrapper tokens and phone numbers dropped, legal
suffixes (Ltd/Pvt/Inc/LLC/Corp/Sarl/SAS/...) removed from the *core* name but kept as a signature feature,
duplicated tokens removed. Addresses: abbreviation expansion (`Rd`->`road`, `R.`->`rue`), ordinals, leading
zeros, PO-box/unit markers dropped. Indic scripts are transliterated to Latin using the Unicode character
names themselves (no dictionary), and a consonant "skeleton" feature bridges spelling/transliteration variants.

**Blocking**: exact partition on the (lower-cased) country string — an open set, nothing hard-coded. Inside a
partition every S1 record is scored against the S2+S3 pool with three sparse TF-IDF channels, stacked into one
index so a single sparse product gives the weighted sum:
`0.3*cos(name char-3-grams of sorted/squashed tokens) + 0.6*cos(address word uni+bigrams) + 0.1*cos(phonetic
consonant-skeleton char-4-grams)`. Very frequent n-grams are pruned (`max_df`) to bound posting lists; chunks run in
parallel over memory-mapped matrices. Top-K per (S1, source) are kept (K=15 -> <=30 candidates).
Why these weights: on this data names are often garbled/aliased while addresses are reliable, so the address channel
must dominate; address bigrams keep truncated addresses selective; the skeleton channel rescues phonetic/transliterated
names and records with no address. Measured on real train data with full-size pools (recall of true pairs at K=15
per source): original name-heavy design 0.75 (India) / 0.88 (US) -> 0.91 / 0.96.

**Matching**: LightGBM (`objective=binary`, `scale_pos_weight=1.0`) on ~75 relative-similarity features plus
blocking-context features (rank, gap to best candidate, and "rival" features: how strongly *other* S1 entities
claim the same S2/S3 record). No country- or vocabulary-specific input, so it transfers to unseen countries.
Threshold `tau` is picked in `[0.70, 0.95]` on a validation split grouped by S1 id, maximising macro F0.5 with
singletons included. At prediction time an S2/S3 record is assigned only to the S1 entity that scores it highest
(verified on train: every S2/S3 id appears in exactly one S1 ground-truth row).

## Verified so far / not yet verified

Verified on real data (train S1 sampled at 0.5%, **full-size S2/S3 pools**, `--dev-frac 0.005 --dev-full-pool`):
candidate recall 0.946 at 30 candidates/S1, blocking-ceiling F0.5 0.978, validation macro F0.5 0.947 (small validation
set of ~2.2k S1 entities, so treat as +/- ~0.01). Peak RAM 5.8 GB, ~1.5 minutes end to end. Inference was run on a 0.5%
sample of the test S1 (France included) and passes `validate_submission.py --check-ids` except for the expected
"missing rows" caused by sampling.

**Not yet run: the full-size train / test jobs.** Expect (extrapolated, 8 cores): blocking ~1 h for the 1.7M test S1
entities (about the same for the 2.2M train S1), feature extraction + scoring ~20-30 min. The full candidate tables
(50-66M pairs) plus the pooled S2/S3 frame need roughly 10-15 GB of RAM: use a machine with >= 32 GB, and lower
`--chunk-size` / `--n-jobs` if memory is tight. `--n-train-s1` bounds the feature matrix used to fit LightGBM.
Note the dev-sample F0.5 above is not the expected leaderboard score.

## Compliance

No external APIs, web requests, geocoders, gazetteers or pretrained language models are used; the only model
is LightGBM (MIT), trained solely on the provided training data.
