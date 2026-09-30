# Trough — Business Entity Resolution

Python pipeline for matching business records using country-based candidate retrieval and a boosted-tree classifier.

## Contents

- `code/business_entity_resolution/`: source, pinned dependencies, trained model, and validation reports.
- `Documentation_template.md`: submission methodology and reported evaluation results.

## Run the pipeline

Follow [the reproduction instructions](code/business_entity_resolution/README.md). The original challenge dataset must be supplied separately. The frozen model is included.

## Generated results

The original submission includes `output/matching_results.tsv` (about 84 MB) and `output/candidate_pairs.tsv` (about 1.8 GB). These generated files are excluded from this Git repository. They remain in the original submission directory and can be regenerated using the documented inference command. Create the root `output/` directory before running inference if needed.

## License

Original source and trained model are covered by the [MIT license](code/business_entity_resolution/LICENSE). Third-party dependencies and challenge data retain their own licenses.
