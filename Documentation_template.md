# ML Challenge 2026: Business Entity Resolution

**Team Name:** Trough  
**Team Members:** R Jeyanth & Amber Chordia  
**Package preparation date:** 2026-09-27

## 1. Executive Summary
A same-country blocker retrieves at most 100 targets per reference business. A boosted-tree classifier scores 23 name/address features and returns zero, one, or multiple matches. Only the supplied challenge data is used.

## 2. Methodology
Training has 2,206,821 reference businesses and 10,320,219 targets; test has 1,732,544 reference businesses including 259,452 in France. Names and addresses contain typos, abbreviations, reordered tokens and missing components. Country is an open string label. Accent normalization preserves non-Latin script marks; address aliases and common legal suffix removal improve retrieval. No external business lookup, geocoding or augmentation is used.

Approach: blocking plus classifier. Contentless SQLite FTS postings and packed source-file byte locators avoid duplicating target text and support a bounded-memory, resumable pipeline.

## 3. Candidate Generation
Keys cover normalized names, legal-stripped compact names, informative name tokens and pairs, name character grams, address token pairs, numeric/name and numeric/address combinations, and exact address signatures. All keys include country. Buckets exceeding 100 records are dropped. Candidates are deduplicated, cheaply reduced to 500, and ranked to 100 using name/address character and token similarity with numeric overlap. Exported candidates are exactly the records scored by the classifier.

Full test candidate pairs: **149,648,464**. Fresh labeled evaluation blocking recall: **95.63%**. Retrieval does lose true links; no claim of perfect recall is made.

## 4. Matching Model
The 23 features cover name character similarity, token overlap/containment, compact equality, character grams and lengths; address token/character similarity, numbers and postcode-like tokens; missing addresses, country agreement and script indicators. Raw IDs and categorical countries are not inputs.

HistGradientBoostingClassifier: 300 boosting iterations, at most 31 leaves per tree, minimum leaf size 40, L2 regularization 2. Training uses 6,000 S1 groups, 494,572 retrieved pairs and 19,961 positives. Original trained artifact and code are MIT licensed; dependencies retain upstream licenses. This tree model is far below the 8-billion-parameter ceiling.

A seeded, earlier-sample-excluded split reserves 1,000 tuning and 1,000 evaluation groups. Three predeclared configurations and a threshold grid are selected using tuning macro F0.5 subject to precision at least max(96%, original baseline tuning precision). Selected threshold: **0.775**. Selection is frozen before evaluation; no refit uses evaluation labels.

## 5. Results and Error Analysis
Fresh paired evaluation macro F0.5: **0.8766** versus baseline **0.8528**. Precision: **96.18%**; recall: **77.02%**. India macro F0.5: **0.8288**; US: **0.9105**. Five of 54 singleton businesses receive false matches. Common-name/address similarity can produce false merges; truncated names, script/format variation and capped/common buckets cause missed links. Of 3,429 true links, blocking misses 150 and the matcher rejects another 638 retrieved true links.

France has no supplied training labels; France accuracy is unknown. These validation results are not leaderboard scores. Full test output contains **1,732,544** rows, **5,062,956** predicted links, and **146,341** empty match lists. Full output validation: **PASS**, including target-ID existence and match/candidate consistency.

## 6. Conclusion
Expanded training improves both precision and recall on a fresh paired sample. The pipeline preserves singleton and multi-match behavior and covers France through country-generic retrieval. Remaining accuracy limitations include lost blocking links and unmeasured France generalization.

## Appendix A: Code Artefacts
All Python source is under `code/business_entity_resolution/src/`, with pinned requirements, MIT license, a frozen model and exact reproduction instructions. Build a test index with `persistent_index.py`, then regenerate both output files with `run_inference.py`. `finalize_submission.py` validates and packages results. Supplied data is not redistributed.

## Appendix B: Validation
The unmodified supplied validator's `validate()` function checks both files in 5,000-row chunks to keep RAM bounded. An additional streaming pass verifies complete source-order coverage, every candidate target ID against the full test target pool, list uniqueness and strict match-subset constraints. Validation reports and SHA256 hashes are included under `code/business_entity_resolution/reports/`.
