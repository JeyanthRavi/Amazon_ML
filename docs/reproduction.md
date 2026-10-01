# Reproduce the submission

Requires Python 3.9+ with SQLite FTS5. The run was prepared on macOS arm64 with Python 3.9.6. Supply the original challenge dataset; do not move source files while an index or inference run is active.

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
# Set this to the original supplied dataset directory (contains train/ and test/).
DATA=/absolute/path/to/student_resource/dataset
mkdir -p output
.venv/bin/python src/persistent_index.py --data "$DATA" --split test --index cache/test-index --budget-gib 3
.venv/bin/python src/run_inference.py --data "$DATA" --split test --index cache/test-index --model models/expanded/matcher.joblib --output output --workers 4 --batch-size 100
```

The frozen model is included, so retraining is unnecessary to reproduce outputs. Checkpoints support restarting the same inference command with `--resume`; code, model, input paths/metadata and index must remain unchanged. Index construction automatically resumes compatible checkpoints. Allow approximately 3 GiB for the index, 2 GiB for outputs and at least 1.5 GiB spare disk. Inference may take many hours and varies with hardware and workload.

Source also includes audits, blocker/model experiments, and verification. Original experiment samples and all supplied data are excluded from the archive; exact experiment metrics and model selection hashes are included in reports. To retrain, generate samples with the audit/blocking scripts and reproduce the documented exclusions before running expanded_training.py. The included trained artifact is the authoritative inference input.

MIT license covers original source and trained model. Third-party packages and supplied challenge data retain their own licenses. Team: Trough. Members: R Jeyanth & Amber Chordia.

## Challenge packaging

`src/finalize_submission.py` preserves the original challenge ZIP layout (`code/business_entity_resolution/`, `output/`, and `Documentation_template.md`) and the team name from `submission/team.json`. It requires the supplied dataset and validator, complete inference outputs and checkpoints, the test index, and `reports/expanded/verification.json` (included here). The generated TSVs and original dataset are not included in Git. Run full inference first; use `--package-only` only with outputs whose hashes match the existing full validation report. The complete original experiment workspace is not included.
