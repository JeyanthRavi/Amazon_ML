"""Independently check saved development candidate files and aggregate metrics."""
import argparse
import csv
import json
import math
from pathlib import Path


def load(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report-dir', type=Path, required=True)
    parser.add_argument('--original-sample', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.report_dir/'blocking_v2.json').read_text())
    sample = load(args.report_dir/'validation_sample.tsv')
    output = load(args.report_dir/'sample_candidate_pairs.tsv')
    original = load(args.original_sample)
    byid = {row['source1_entity_id']: row for row in output}
    sample_ids = {row['source1_entity_id'] for row in sample}
    if len(sample) != len(sample_ids) or len(output) != len(byid) or set(byid) != sample_ids:
        raise ValueError('Duplicate or missing S1 rows')
    if len(sample) != report['sample_size']:
        raise ValueError('Sample count inconsistent with report')
    original_ids = {row['source1_entity_id'] for row in original}
    if report['mode'] == 'fresh_evaluation' and sample_ids & original_ids:
        raise ValueError('Fresh sample overlaps initial sample')
    if report['mode'] == 'development':
        dev_ids = {row['source1_entity_id'] for row in original if row['split'] == 'development'}
        if sample_ids != dev_ids:
            raise ValueError('Development set changed or includes old holdout')
    for row in output:
        ids = row['candidate_entity_ids'].split(',') if row['candidate_entity_ids'] else []
        if len(ids) != len(set(ids)) or len(ids) > 100 or any(not i.startswith(('S2-', 'S3-')) for i in ids):
            raise ValueError('Invalid candidate list')
    for k in report['config']['candidate_limits']:
        for country, metrics in report['metrics'][str(k)].items():
            subset = [r for r in sample if country == 'all' or r['country'] == country]
            truth_count = hits = candidate_count = positive = complete = singletons = 0
            entity_recall = 0.0
            for row in subset:
                truth = set(row['matched_entity_ids'].split(',')) if row['matched_entity_ids'] else set()
                text = byid[row['source1_entity_id']]['candidate_entity_ids']
                candidates = set(text.split(',')[:k]) if text else set()
                found = len(truth & candidates)
                truth_count += len(truth)
                hits += found
                candidate_count += len(candidates)
                if truth:
                    positive += 1
                    complete += truth <= candidates
                    entity_recall += found/len(truth)
                else:
                    singletons += 1
            if (len(subset), truth_count, hits, candidate_count, complete, singletons) != (
                    metrics['entities'], metrics['truth_pairs'], metrics['recovered_pairs'], metrics['candidate_total'],
                    metrics['all_matches_recovered'], metrics['singleton_entities']):
                raise ValueError(f'Counts mismatch for {country}, k={k}')
            if not math.isclose(hits/truth_count, metrics['pair_recall']) or not math.isclose(entity_recall/positive, metrics['mean_entity_recall_positive_only']):
                raise ValueError('Recall calculation mismatch')
    stage = report['stage_metrics']
    if not stage['raw_pair_recall'] >= stage['prefilter_pair_recall'] >= report['metrics']['100']['all']['pair_recall']:
        raise ValueError('Stage recall order is impossible')
    if stage['missing_labeled_target_ids']:
        raise ValueError('Some sampled labels refer to absent training targets')
    verification = {'status': 'PASS', 'sample_rows': len(sample), 'mode': report['mode'],
                    'checks': ['Sample coverage and uniqueness', 'Candidate deduplication, prefixes and caps',
                               'Development/fresh sample isolation', 'Independent recall/count recomputation by country and limit',
                               'Stage recall ordering', 'Sampled true targets present in full training pool']}
    (args.report_dir/'verification.json').write_text(json.dumps(verification, indent=2)+'\n')
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    main()
