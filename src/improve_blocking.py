"""Second blocker: address routes and typo-tolerant names; no external data."""
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
from collections import Counter
from pathlib import Path

from audit_blocking import ALIASES, STOP, evaluate, jaccard, rows
from difflib import SequenceMatcher

ADDRESS_STOP = set('no number near opposite road street avenue boulevard drive lane floor flt flat plot house h block sector village city district state india us united states'.split())
CONFIG = {'version': 2, 'bucket_cap': 100, 'scoring_prefilter_cap': 500,
          'candidate_limits': [10, 25, 50, 100], 'address_word_limit': 8, 'address_pair_word_limit': 6,
          'name_gram_length': 4, 'name_gram_token_limit': 3}


def normalize(value):
    # Remove Latin diacritics while preserving vowel signs/viramas in other scripts.
    if value.isascii():
        return ' '.join(re.findall(r'[^\W_]+', value.casefold().replace('&', ' and ')))
    output = []
    latin_base = False
    for char in unicodedata.normalize('NFKD', value.casefold().replace('&', ' and ')):
        category = unicodedata.category(char)
        if category.startswith('M'):
            if not latin_base:
                output.append(char)
        elif category.startswith(('L', 'N')):
            output.append(char)
            latin_base = 'LATIN' in unicodedata.name(char, '')
        else:
            output.append(' ')
            latin_base = False
    return ' '.join(''.join(output).split())


def prepare(row):
    eid, name, address, country = row
    name = re.sub(r'\.(?:com|net|org|in|co|fr)(?:/.*)?$', '', name.strip(), flags=re.I)
    name = normalize(name)
    nt = tuple(sorted(set(name.split()) - STOP))
    at = frozenset(ALIASES.get(t, t) for t in normalize(address).split())
    return eid, name, nt, at, normalize(country)


def compact_name(record):
    return ''.join(t for t in record[1].split() if t not in STOP)


def keys(record):
    _, name, nt, at, country = record
    result = set()
    if name:
        result.add(('exact', country, name))
        result.add(('compact', country, compact_name(record)))
    tokens = sorted(nt, key=lambda t: (-len(t), t))[:8]
    for token in tokens:
        if len(token) >= 4:
            result.add(('token', country, token))
    for a, b in itertools.combinations(tokens[:6], 2):
        result.add(('name_pair', country, *sorted((a, b))))
    numbers = sorted(t for t in at if t.isdigit())[:4]
    for number in numbers:
        for token in tokens[:4]:
            result.add(('number_name', country, number, token))
    for token in tokens[:CONFIG['name_gram_token_limit']]:
        if len(token) >= 6:
            # All grams for a bounded number of tokens, including middle spelling edits.
            for start in range(min(len(token)-3, 24)):
                result.add(('name_gram', country, token[start:start+4]))
    address_words = sorted((t for t in at if len(t) >= 3 and not t.isdigit() and t not in ADDRESS_STOP),
                           key=lambda t: (-len(t), t))[:CONFIG['address_word_limit']]
    for a, b in itertools.combinations(address_words[:CONFIG['address_pair_word_limit']], 2):
        result.add(('address_pair', country, *sorted((a, b))))
    for number in numbers:
        for word in address_words[:4]:
            result.add(('number_address', country, number, word))
    if at:
        result.add(('address_exact', country, tuple(sorted(at))))
    return result


def cheap_score(a, b):
    name = max(jaccard(a[2], b[2]), float(compact_name(a) == compact_name(b) and bool(compact_name(a))))
    address = jaccard(a[3], b[3])
    return 3 * max(name, address) + min(name, address)


def score(a, b):
    name = max(jaccard(a[2], b[2]), SequenceMatcher(None, compact_name(a), compact_name(b), autojunk=False).ratio()
               if compact_name(a) and compact_name(b) else 0)
    address = jaccard(a[3], b[3])
    numbers_a = {t for t in a[3] if t.isdigit()}
    numbers_b = {t for t in b[3] if t.isdigit()}
    return 3 * max(name, address) + min(name, address) + .3 * jaccard(numbers_a, numbers_b)


def load_sample(args):
    if args.sample:
        with args.sample.open() as handle:
            samples = list(csv.DictReader(handle, delimiter='\t'))
        return [[r['source1_entity_id'], r['business_name'], r['business_address'], r['country']]
                for r in samples if r['split'] == 'development']
    excluded = set()
    with args.exclude_sample.open() as handle:
        excluded = {r['source1_entity_id'] for r in csv.DictReader(handle, delimiter='\t')}
    rng, sample, count = random.Random(args.seed), [], 0
    for row in rows(args.data / 'train/train_source1.tsv'):
        if row[0] in excluded:
            continue
        count += 1
        if len(sample) < args.sample_size:
            sample.append(row)
        else:
            index = rng.randrange(count)
            if index < args.sample_size:
                sample[index] = row
    return sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--sample', type=Path)
    parser.add_argument('--exclude-sample', type=Path)
    parser.add_argument('--sample-size', type=int, default=500)
    parser.add_argument('--seed', type=int, default=9027)
    args = parser.parse_args()
    if bool(args.sample) == bool(args.exclude_sample):
        parser.error('Choose --sample for development or --exclude-sample for fresh evaluation')
    started = time.time()
    args.output.mkdir(parents=True, exist_ok=True)
    sample = load_sample(args)
    if not sample:
        raise ValueError('Empty sample')
    selected = {r[0]: prepare(r) for r in sample}
    truth = {}
    for eid, targets in rows(args.data / 'train/train_ground_truth.tsv'):
        if eid in selected:
            truth[eid] = set(targets.split(',')) if targets else set()
    if set(truth) != set(selected):
        raise ValueError('Missing sampled labels')
    keysets = {eid: keys(r) for eid, r in selected.items()}
    postings = {k: [] for ks in keysets.values() for k in ks}
    print(f'Selected {len(sample)} businesses; querying {len(postings)} keys', flush=True)
    wanted = set().union(*truth.values())
    true_records, row_counts, target_countries = {}, {}, Counter()
    for source in (2, 3):
        count = 0
        for count, row in enumerate(rows(args.data / f'train/train_source{source}.tsv'), 1):
            record = prepare(row)
            target_countries[row[3]] += 1
            if record[0] in wanted:
                true_records[record[0]] = row
            for key in keys(record):
                if key not in postings:
                    continue
                bucket = postings[key]
                if bucket is None:
                    continue
                if len(bucket) == CONFIG['bucket_cap']:
                    postings[key] = None
                else:
                    bucket.append(record)
            if count % 1000000 == 0:
                print(f'S{source}: {count:,} records indexed', flush=True)
        row_counts[f'S{source}'] = count
    candidates, raw_hits, prefilter_hits, pool_sizes, prefilter_sizes = {}, 0, 0, [], []
    examples = []
    for row in sample:
        eid, record = row[0], selected[row[0]]
        pool = {r[0]: r for key in keysets[eid] for r in (postings[key] or [])}
        raw_hits += len(truth[eid] & set(pool))
        pool_sizes.append(len(pool))
        shortlist = heapq.nlargest(CONFIG['scoring_prefilter_cap'], pool.values(), key=lambda r: (cheap_score(record, r), r[0]))
        prefilter_hits += len(truth[eid] & {r[0] for r in shortlist})
        prefilter_sizes.append(len(shortlist))
        ranked = sorted(shortlist, key=lambda r: (-score(record, r), r[0]))[:100]
        candidates[eid] = [r[0] for r in ranked]
        missed = truth[eid] - set(candidates[eid])
        # Fresh evaluation saves metrics only; do not inspect its labeled examples.
        if args.sample and missed:
            examples.append({'source1': row, 'missed_at_100': [true_records[t] for t in sorted(missed) if t in true_records],
                             'missing_before_ranking': sorted(truth[eid] - set(pool)),
                             'missing_after_prefilter': sorted(truth[eid] - {r[0] for r in shortlist})})
    audit = json.loads(args.audit.read_text())['audit']
    expected_counts = {f'S{s}': audit[f'train_source{s}.tsv']['rows'] for s in (2, 3)}
    if row_counts != expected_counts:
        raise ValueError('Target row counts changed since original audit')
    country_map = {r[0]: r[3] for r in sample}
    metrics = evaluate(list(selected), truth, candidates, country_map, CONFIG['candidate_limits'], target_countries)
    total_truth = sum(map(len, truth.values()))
    report = {'mode': 'development' if args.sample else 'fresh_evaluation', 'sample_size': len(sample), 'seed': args.seed,
              'config': CONFIG, 'config_sha256': hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest(),
              'target_rows': row_counts, 'metrics': metrics,
              'stage_metrics': {'raw_pair_recall': raw_hits/total_truth, 'prefilter_pair_recall': prefilter_hits/total_truth,
                                'mean_raw_candidates': sum(pool_sizes)/len(pool_sizes), 'max_raw_candidates': max(pool_sizes),
                                'mean_scored_candidates': sum(prefilter_sizes)/len(prefilter_sizes),
                                'queried_keys': len(postings), 'dropped_common_keys': sum(v is None for v in postings.values()),
                                'missing_labeled_target_ids': sorted(wanted - set(true_records))},
              'elapsed_seconds': time.time()-started,
              'limitations': ['Sampled training evaluation, not a leaderboard score.', 'No France labels.',
                             'Production still needs a persistent index.', 'No trained matching classifier.']}
    (args.output / 'blocking_v2.json').write_text(json.dumps(report, indent=2)+'\n')
    with (args.output / 'validation_sample.tsv').open('w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['source1_entity_id','split','country','business_name','business_address','matched_entity_ids'])
        for row in sample:
            writer.writerow([row[0],report['mode'],row[3],row[1],row[2],','.join(sorted(truth[row[0]]))])
    with (args.output / 'sample_candidate_pairs.tsv').open('w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['source1_entity_id','candidate_entity_ids'])
        for eid in selected:
            writer.writerow([eid, ','.join(candidates[eid])])
    if args.sample:
        (args.output / 'development_misses.json').write_text(json.dumps(examples, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
