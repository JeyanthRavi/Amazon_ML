# Amazon ML Challenge 2026

**Amazon ML Challenge 2026 project by R Jeyanth and Amber Chordia.** This is a portfolio copy of the team project. The challenge round has ended; the team was not selected for the next round.

Match business records across datasets despite differences in spelling, abbreviations, names, and addresses. This repository contains our Amazon ML Challenge 2026 submission: a resumable candidate-retrieval pipeline, a trained matching model, and evaluation reports.

## Approach

1. **Retrieve candidates:** country-specific name and address keys select up to 100 targets per reference business. A SQLite index keeps retrieval bounded in memory.
2. **Score pairs:** a boosted-tree classifier evaluates 23 name and address similarity features.
3. **Return matches:** a frozen threshold of 0.775 produces zero, one, or multiple matches per business.

The frozen trained model is included; retraining is unnecessary to run inference. The original challenge dataset must be supplied separately.

## Reported results

The submitted `matching_results.tsv` received a **public leaderboard macro F0.5 score of 0.857** on 27 September 2026. This is the public score, not a private leaderboard result or a claim of final ranking. The results below are from a separate, held-out evaluation of 1,000 reference businesses after model and threshold selection.

| Metric | Baseline | Submitted model |
| --- | ---: | ---: |
| Macro F0.5 | 0.8528 | **0.8766** |
| Precision | 95.67% | **96.18%** |
| Recall | 73.43% | **77.02%** |

See the [verification report](reports/verification.json) and [methodology](docs/methodology.md) for evaluation details. France has no supplied training labels, so accuracy on France is unknown.

## Run inference

Requires Python 3.9+ with SQLite FTS5. The submission was prepared using Python 3.9.6 on macOS arm64; dependencies are pinned in [requirements.txt](requirements.txt).

Run these commands from the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

# Original challenge dataset directory containing train/ and test/.
DATA=/absolute/path/to/student_resource/dataset
mkdir -p output

.venv/bin/python src/persistent_index.py \
  --data "$DATA" --split test --index cache/test-index --budget-gib 3

.venv/bin/python src/run_inference.py \
  --data "$DATA" --split test --index cache/test-index \
  --model models/expanded/matcher.joblib --output output \
  --workers 4 --batch-size 100
```

Allow approximately 3 GiB for the index, 2 GiB for outputs, and at least 1.5 GiB of spare disk space. Full inference may take many hours. See [reproduction instructions](docs/reproduction.md) for checkpoint resumption and retraining details.

## Repository layout

```text
src/                 Retrieval, training, inference, audits, and tests
models/expanded/     Frozen model, selection metadata, and training report
reports/             Evaluation and full-output validation reports
docs/                Methodology and reproduction instructions
requirements.txt     Pinned Python dependencies
LICENSE              MIT license
```

## Generated outputs

Inference writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`. The original submission contains about 84 MB of matching results and 1.8 GB of candidate pairs. These generated files and the original dataset are excluded from Git; regenerate the outputs using the commands above. This repository contains no Colab notebook; the pipeline is implemented in Python scripts and can be run in a terminal or a notebook environment.

The archived [full-output validation report](reports/final_validation.json) records validation of the original submission. This repository reorganization does not constitute a new full-dataset evaluation.

`src/finalize_submission.py` creates the challenge’s original nested ZIP submission layout, which differs from this repository’s layout. Packaging requires the original dataset, validator, inference outputs, and test index; see [reproduction instructions](docs/reproduction.md). The published repository does not itself contain the generated TSVs or the 818 MiB submission ZIP.

## Contributors and license

**Contributors:** R Jeyanth & Amber Chordia.

Original source and trained model are covered by the [MIT license](LICENSE). Third-party packages and supplied challenge data retain their own licenses.
