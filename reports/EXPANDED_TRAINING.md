# Expanded classifier: frozen selection and fresh paired evaluation

## What changed

Training increased from 670 to 6,000 S1 groups. The new sample contains 8,000 uniformly sampled training businesses (seed 20270927), excluding all 2,499 unique S1 IDs used in earlier experiments and the production pilot. Disjoint groups are allocated to training (6,000), threshold tuning (1,000), and evaluation (1,000). Full dataset integrity checks previously established that positive targets are not shared across S1 groups.

The blocker and 23 features remain unchanged. Retrieved hard negatives and positives produce 494,572 training pairs, including 19,961 positives. Across all splits there are 662,132 candidate pairs. Feature extraction uses the complete 10,320,219-target training index and four workers, with restartable local feature stores. The experiment took approximately 160 seconds on this machine.

## Selection before evaluation

Three model configurations were declared in `reports/expanded/plan.json`. Selection maximized tuning macro F0.5 while requiring precision at least the greater of 96% and the original baseline's tuning precision (96.963%). The selected configuration uses 300 boosting iterations, at most 31 leaves per tree, minimum leaf size 40, L2 regularization 2, and threshold 0.775. Tuning macro F0.5 is 0.8754 and precision is 97.007%.

The model and threshold were saved before computing evaluation predictions. Evaluation labels were not used to select the configuration or threshold. The model was not refitted on tuning or evaluation groups. `models/expanded/frozen_model.json` records the configuration, hashes, and selection metrics. Original code/model license is MIT; dependencies retain upstream licenses.

## Fresh evaluation: both models on the same 1,000 businesses

- Macro F0.5: baseline **0.8528**, expanded **0.8766**; gain **0.0238**.
- Pair precision: baseline **95.67%**, expanded **96.18%**.
- Pair recall: baseline **73.43%**, expanded **77.02%**.
- True positives increase from **2,518 to 2,641**; false positives decrease from **114 to 105**.
- India macro F0.5 improves from **0.7984 to 0.8288** (415 groups).
- US macro F0.5 improves from **0.8913 to 0.9105** (585 groups).
- Both models falsely match **5 of 54** singleton businesses.
- Blocking retains **3,279 of 3,429** true links (**95.63%**). Expanded matcher misses another **638** retained links; **150** links are absent from candidates.

An independent verification recomputes both sets of saved predictions, including singleton scores; checks all split exclusions, 8,000 unique candidate rows, cap of 100, duplicate-free predictions, matches within candidates, and both model hashes. Result: **PASS**. A paired bootstrap of 5,000 seeded S1 resamples gives a 95% interval of **+0.0142 to +0.0337** for the macro F0.5 difference. This measures sample uncertainty, not generalization to France or the leaderboard.

The earlier 500-record baseline result is a different sample; use the paired comparison above to judge this change. The expanded model is selected for the planned test run. Preserve the original baseline for reproducibility.

## Verification and artifacts

- `src/expanded_training.py`: sample generation, indexed feature extraction, tuning, freeze, and paired evaluation.
- `src/verify_expanded.py`: independent saved-output checks and paired bootstrap.
- `models/expanded/matcher.joblib`: selected inference artifact.
- `models/expanded/expanded_report.json`: full metrics and tuning search.
- `reports/expanded/verification.json`: independently recomputed scores and checks.
- `reports/expanded-inference-smoke/`: 30-row partial training inference using the selected artifact; serial and parallel results are compared exactly. These files are not submissions.

## Remaining work

Build the separate test index, run resumable inference over all 1,732,544 test S1 records, validate both complete output files with the supplied validator, fill the competition documentation, and package the reproducible code and submission ZIP. Existing four-worker throughput projects approximately 16.3 hours for full test inference, excluding index construction and pauses; this was measured with the original model and training workload, so expanded-model/test runtime may differ.

About 6.4 GiB free disk remains. Check the actual test index size and expected output requirements before full inference, retaining the 1.5 GiB free-space stop. France has no supplied labels, so France accuracy is unknown. Training remains sampled rather than using every labeled business. Full test inference has not started.
