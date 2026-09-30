"""Streaming audit and label-independent blocking evaluation; Python stdlib only."""
import argparse
import csv
import hashlib
import heapq
import itertools
import json
import random
import re
import time
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

STOP = set('the and of for company co corporation corp incorporated inc limited ltd private pvt llc llp'.split())
ALIASES = {'rd': 'road', 'st': 'street', 'ave': 'avenue', 'blvd': 'boulevard', 'dr': 'drive'}


def normalize(value):
    value = unicodedata.normalize('NFKD', value.casefold())
    value = ''.join(c for c in value if not unicodedata.combining(c))
    return ' '.join(re.findall(r'[^\W_]+', value.replace('&', ' and ')))


def prepare(row):
    eid, name, address, country = row
    name = normalize(name)
    nt = tuple(sorted(set(name.split()) - STOP))
    at = frozenset(ALIASES.get(t, t) for t in normalize(address).split())
    return eid, name, nt, at, normalize(country)


def keys(record):
    _, name, nt, at, country = record
    result = {('exact', country, name)} if name else set()
    # Fixed limits keep work bounded even for unusually long strings.
    tokens = sorted(nt, key=lambda t: (-len(t), t))[:8]
    for token in tokens:
        if len(token) >= 4:
            result.add(('token', country, token))
    for a, b in itertools.combinations(tokens[:6], 2):
        result.add(('pair', country, *sorted((a, b))))
    numbers = sorted(t for t in at if t.isdigit())[:4]
    for number in numbers:
        for token in tokens[:4]:
            result.add(('number_name', country, number, token))
    # Character prefixes offer a modest fallback for misspelled long names.
    for token in tokens[:3]:
        if len(token) >= 6:
            result.add(('prefix', country, token[:5]))
    return result


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a or b else 0.0


def cheap_score(a, b):
    return 3 * jaccard(a[2], b[2]) + jaccard(a[3], b[3]) + 2 * (a[1] == b[1])


def score(a, b):
    return cheap_score(a, b) + SequenceMatcher(None, a[1], b[1], autojunk=False).ratio()


def rows(path):
    with path.open(encoding='utf-8', newline='') as handle:
        reader = csv.reader(handle, delimiter='\t')
        header = next(reader)
        for row in reader:
            if len(row) != len(header):
                raise ValueError(f'{path}: malformed row with {len(row)} fields')
            yield row


def audit_source(path, hook=None):
    countries, missing = Counter(), Counter()
    count = 0
    for count, row in enumerate(rows(path), 1):
        countries[row[3]] += 1
        for field, value in zip(('entity_id', 'business_name', 'business_address', 'country'), row):
            if not value.strip():
                missing[field] += 1
        if hook:
            hook(count, row)
    result = {'rows': count, 'bytes': path.stat().st_size, 'countries': dict(countries), 'missing': dict(missing)}
    print(path.name, result, flush=True)
    return result


def evaluate(sample_ids, truth, candidates, countries, limits, source_sizes):
    result = {}
    for k in limits:
        groups = defaultdict(lambda: {'entities': 0, 'positive_entities': 0, 'truth_pairs': 0, 'recovered_pairs': 0,
                                     'all_matches_recovered': 0, 'entity_recall_sum': 0.0, 'candidate_total': 0,
                                     'no_candidate_entities': 0, 'singleton_entities': 0, 'singleton_with_candidates': 0})
        counts = []
        for eid in sample_ids:
            expected = truth[eid]
            actual = set(candidates[eid][:k])
            counts.append(len(actual))
            for group in ('all', countries[eid]):
                d = groups[group]
                d['entities'] += 1
                d['candidate_total'] += len(actual)
                d['no_candidate_entities'] += not actual
                d['truth_pairs'] += len(expected)
                d['recovered_pairs'] += len(expected & actual)
                if expected:
                    d['positive_entities'] += 1
                    d['entity_recall_sum'] += len(expected & actual) / len(expected)
                    d['all_matches_recovered'] += expected <= actual
                else:
                    d['singleton_entities'] += 1
                    d['singleton_with_candidates'] += bool(actual)
        for group, d in groups.items():
            d['pair_recall'] = d['recovered_pairs'] / d['truth_pairs'] if d['truth_pairs'] else None
            d['mean_entity_recall_positive_only'] = d.pop('entity_recall_sum') / d['positive_entities'] if d['positive_entities'] else None
            d['average_candidates'] = d['candidate_total'] / d['entities']
            d['all_matches_recovered_fraction_positive_only'] = d['all_matches_recovered'] / d['positive_entities'] if d['positive_entities'] else None
        counts.sort()
        groups['all']['p95_candidates'] = counts[min(len(counts)-1, int(.95*len(counts)))]
        groups['all']['max_candidates'] = max(counts)
        denominator = sum(source_sizes[countries[eid]] for eid in sample_ids)
        groups['all']['reduction_ratio_vs_same_country_all_pairs'] = 1 - groups['all']['candidate_total'] / denominator
        result[str(k)] = dict(groups)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sample-size', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--bucket-cap', type=int, default=100)
    args = parser.parse_args()
    if args.sample_size < 10 or args.bucket_cap < 1:
        parser.error('sample-size must be >=10 and bucket-cap >=1')
    started = time.time()
    args.output.mkdir(parents=True, exist_ok=True)
    rng, sample, audit = random.Random(args.seed), [], {}

    def reservoir(n, row):
        if len(sample) < args.sample_size:
            sample.append(row)
        else:
            i = rng.randrange(n)
            if i < args.sample_size:
                sample[i] = row

    audit['train_source1.tsv'] = audit_source(args.data / 'train/train_source1.tsv', reservoir)
    selected = {r[0]: prepare(r) for r in sample}
    truth, histogram = {}, Counter()
    gt_count = 0
    for gt_count, row in enumerate(rows(args.data / 'train/train_ground_truth.tsv'), 1):
        ids = row[1].split(',') if row[1] else []
        histogram[len(ids)] += 1
        if row[0] in selected:
            truth[row[0]] = set(ids)
    if set(truth) != set(selected):
        raise ValueError('Sampled S1 IDs missing from ground truth')
    audit['train_ground_truth.tsv'] = {'rows': gt_count, 'match_count_histogram': dict(sorted(histogram.items())),
                                     'total_labeled_pairs': sum(k*v for k,v in histogram.items())}
    print('Ground truth:', audit['train_ground_truth.tsv'], flush=True)
    keysets = {eid: keys(r) for eid, r in selected.items()}
    postings = {key: [] for ks in keysets.values() for key in ks}
    frequencies = Counter()
    # Ground truth is used only to collect diagnostic examples, never to add candidates.
    wanted_examples = set().union(*(truth[eid] for eid in selected))
    example_targets = {}

    def index_target(n, row):
        record = prepare(row)
        if row[0] in wanted_examples:
            example_targets[row[0]] = row
        for key in keys(record):
            if key not in postings:
                continue
            frequencies[key] += 1
            bucket = postings[key]
            if bucket is None:
                continue
            if len(bucket) == args.bucket_cap:
                # Drop common buckets completely; do not bias toward early file rows.
                postings[key] = None
            else:
                bucket.append(record)
        if n % 1000000 == 0:
            print(f'Indexed {n:,} target records in current source', flush=True)

    for source in (2, 3):
        name = f'train_source{source}.tsv'
        audit[name] = audit_source(args.data / 'train' / name, index_target)
    print('Ranking sampled candidates', flush=True)
    candidates, pool_sizes = {}, []
    for eid, record in selected.items():
        pool = {r[0]: r for key in keysets[eid] for r in (postings[key] or [])}
        pool_sizes.append(len(pool))
        # Label-independent cheap prefilter bounds expensive string comparisons.
        shortlist = heapq.nlargest(500, pool.values(), key=lambda r: (cheap_score(record, r), r[0]))
        ranked = sorted(shortlist, key=lambda r: (-score(record, r), r[0]))[:100]
        candidates[eid] = [r[0] for r in ranked]
    for source in (1, 2, 3):
        name = f'test_source{source}.tsv'
        audit[name] = audit_source(args.data / 'test' / name)
    source_sizes = Counter()
    for source in (2, 3):
        source_sizes.update(audit[f'train_source{source}.tsv']['countries'])
    # Stable 80/20 split at S1 level; no learned model in this baseline.
    split = {eid: ('holdout' if int(hashlib.sha256(f'{args.seed}:{eid}'.encode()).hexdigest()[:8], 16) % 5 == 0 else 'development') for eid in selected}
    country_map = {r[0]: r[3] for r in sample}
    metrics = {part: evaluate([eid for eid in selected if split[eid] == part], truth, candidates, country_map,
                             (10, 25, 50, 100), source_sizes) for part in ('development', 'holdout')}
    report = {'seed': args.seed, 'sample_size': len(sample), 'bucket_cap': args.bucket_cap, 'audit': audit,
              'metrics': metrics, 'blocking': {'queried_keys': len(postings), 'dropped_common_keys': sum(v is None for v in postings.values()),
              'average_raw_pool': sum(pool_sizes)/len(pool_sizes), 'max_raw_pool': max(pool_sizes), 'scoring_prefilter_cap': 500},
              'elapsed_seconds': time.time()-started,
              'limitations': ['Uniform sample, not a full-training recall measurement.', 'No France labels: test-country generalization cannot be measured.',
                             'Query-specific postings scan the full target pool; production requires a persistent index.',
                             'No matching classifier or final F0.5 score yet.', 'Full ID uniqueness and cross-file integrity are not checked by this audit.']}
    (args.output / 'audit_and_blocking.json').write_text(json.dumps(report, indent=2)+'\n')
    with (args.output / 'validation_sample.tsv').open('w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['source1_entity_id', 'split', 'country', 'business_name', 'business_address', 'matched_entity_ids'])
        for row in sample:
            writer.writerow([row[0], split[row[0]], row[3], row[1], row[2], ','.join(sorted(truth[row[0]]))])
    with (args.output / 'sample_candidate_pairs.tsv').open('w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['source1_entity_id', 'candidate_entity_ids'])
        for eid in selected:
            writer.writerow([eid, ','.join(candidates[eid])])
    examples = []
    for row in sample:
        expected = truth[row[0]]
        recovered = set(candidates[row[0]]) & expected
        missed = expected - set(candidates[row[0]])
        if len(examples) < 20 or missed and sum(bool(e['missed_ids_at_100']) for e in examples) < 20:
            examples.append({'source1': dict(zip(('entity_id','business_name','business_address','country'),row)),
                             'recovered_ids_at_100': sorted(recovered), 'missed_ids_at_100': sorted(missed),
                             'true_targets': [dict(zip(('entity_id','business_name','business_address','country'),example_targets[t])) for t in sorted(expected) if t in example_targets]})
    (args.output / 'matched_examples.json').write_text(json.dumps(examples, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({'metrics': metrics, 'blocking': report['blocking'], 'elapsed_seconds': report['elapsed_seconds']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
