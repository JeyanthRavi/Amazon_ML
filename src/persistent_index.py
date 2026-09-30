"""Resumable contentless FTS index preserving the version 2 blocking rules."""
import argparse
import csv
import functools
import hashlib
import heapq
import json
import os
import resource
import shutil
import sqlite3
import struct
import time
from pathlib import Path

from improve_blocking import CONFIG, cheap_score, keys, prepare, score

LOCATOR = struct.Struct('<BQ')


def raw_atoms(key):
    kind, country = key[:2]
    if kind == 'name_pair':
        return [('np', country, word) for word in key[2:]]
    if kind == 'address_pair':
        return [('ap', country, word) for word in key[2:]]
    if kind == 'number_name':
        return [('num', country, key[2]), ('nn', country, key[3])]
    if kind == 'number_address':
        return [('num', country, key[2]), ('na', country, key[3])]
    return [('single', *key)]


def atom_token(atom):
    data = json.dumps(atom, ensure_ascii=False, separators=(',', ':')).encode()
    return 'x' + hashlib.blake2b(data, digest_size=16).hexdigest()


def index_tokens(record):
    atoms = {atom for key in keys(record) for atom in raw_atoms(key)}
    return ' '.join(sorted(atom_token(atom) for atom in atoms))


def source_metadata(paths):
    return [{'path': str(p.resolve()), 'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns} for p in paths]


def parse(raw):
    row = next(csv.reader([raw.decode('utf-8')], delimiter='\t'))
    if len(row) != 4:
        raise ValueError('Index requires four fields in one physical TSV record')
    return row


def disk_bytes(directory):
    return sum(p.stat().st_size for p in directory.iterdir() if p.is_file())


def connect(path):
    connection = sqlite3.connect(path)
    connection.execute('PRAGMA cache_size=-65536')
    connection.execute('PRAGMA temp_store=MEMORY')
    return connection


def build(data, split, directory, limit=0, budget_gib=3):
    directory.mkdir(parents=True, exist_ok=True)
    paths = [data / split / f'{split}_source{s}.tsv' for s in (2, 3)]
    connection = connect(directory/'tokens.sqlite')
    connection.execute('PRAGMA journal_mode=WAL')
    connection.execute('CREATE VIRTUAL TABLE IF NOT EXISTS docs USING fts5(tokens, content="", detail=none)')
    connection.execute('CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
    saved = connection.execute("SELECT value FROM meta WHERE key='state'").fetchone()
    fingerprint = hashlib.sha256(Path(__file__).read_bytes() + Path(__file__).with_name('improve_blocking.py').read_bytes()).hexdigest()
    state = json.loads(saved[0]) if saved else {'sources': source_metadata(paths), 'source_index': 0, 'position': 0,
             'count': 0, 'source_counts': [0, 0], 'complete': False, 'schema': 1, 'code_sha256': fingerprint,
             'config': CONFIG, 'build_seconds': 0}
    if state['sources'] != source_metadata(paths) or state['code_sha256'] != fingerprint:
        raise ValueError('Sources or indexing code changed; build a separate index directory')
    if state['complete']:
        print('Index already complete', flush=True)
        connection.close()
        return state
    locator_path = directory/'locators.bin'
    if not locator_path.exists():
        locator_path.touch()
    started, batch_started = time.time(), time.time()
    with locator_path.open('r+b') as locators:
        if locator_path.stat().st_size < state['count']*LOCATOR.size:
            raise ValueError('Locator file is shorter than committed checkpoint')
        locators.truncate(state['count']*LOCATOR.size)
        locators.seek(0, os.SEEK_END)
        stop = False
        for source in range(state['source_index'], 2):
            with paths[source].open('rb') as handle:
                header = parse(handle.readline())
                if header != ['entity_id','business_name','business_address','country']:
                    raise ValueError('Unexpected source header')
                if source == state['source_index'] and state['position']:
                    handle.seek(state['position'])
                while not stop:
                    batch, offset_bytes = [], bytearray()
                    batch_limit = min(20000, limit-state['count']) if limit else 20000
                    if batch_limit <= 0:
                        stop = True
                        break
                    for _ in range(batch_limit):
                        offset = handle.tell()
                        raw = handle.readline()
                        if not raw:
                            break
                        row = parse(raw)
                        rowid = state['count'] + len(batch) + 1
                        batch.append((rowid, index_tokens(prepare(row))))
                        offset_bytes.extend(LOCATOR.pack(source, offset))
                    if batch:
                        if disk_bytes(directory) > budget_gib*1024**3 or shutil.disk_usage(directory).free < 1.5*1024**3:
                            raise RuntimeError('Index disk budget reached; last committed checkpoint is resumable')
                        connection.executemany('INSERT INTO docs(rowid,tokens) VALUES(?,?)', batch)
                        locators.write(offset_bytes)
                        locators.flush()
                        os.fsync(locators.fileno())
                        state['count'] += len(batch)
                        state['source_counts'][source] += len(batch)
                        state['source_index'], state['position'] = source, handle.tell()
                        state['build_seconds'] += time.time()-batch_started
                        connection.execute('INSERT OR REPLACE INTO meta VALUES(?,?)', ('state', json.dumps(state)))
                        connection.commit()
                        connection.execute('PRAGMA wal_checkpoint(TRUNCATE)')
                        batch_started = time.time()
                        if state['count'] % 100000 == 0:
                            print(f"Indexed {state['count']:,}; {disk_bytes(directory)/1024**2:.1f} MiB; cumulative {state['build_seconds']:.1f}s", flush=True)
                        if limit and state['count'] >= limit:
                            stop = True
                    if len(batch) < batch_limit:
                        state['source_index'], state['position'] = source+1, 0
                        break
            if stop:
                break
        if not stop and state['source_index'] == 2:
            state['complete'] = True
        connection.execute('INSERT OR REPLACE INTO meta VALUES(?,?)', ('state', json.dumps(state)))
        connection.commit()
        connection.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    state['disk_bytes'] = disk_bytes(directory)
    state['last_run_seconds'] = time.time()-started
    state['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    (directory/'build_report.json').write_text(json.dumps(state, indent=2)+'\n')
    connection.close()
    return state


class TargetIndex:
    def __init__(self, directory, allow_partial=False):
        self.connection = connect(directory/'tokens.sqlite')
        self.state = json.loads(self.connection.execute("SELECT value FROM meta WHERE key='state'").fetchone()[0])
        if not self.state['complete'] and not allow_partial:
            raise ValueError('Index incomplete; resume build before evaluating recall')
        paths = [Path(s['path']) for s in self.state['sources']]
        if source_metadata(paths) != self.state['sources']:
            raise ValueError('Source file metadata changed after indexing')
        self.locators = (directory/'locators.bin').open('rb')
        self.sources = [p.open('rb') for p in paths]
        self.get = functools.lru_cache(maxsize=20000)(self._get)

    def _get(self, rowid):
        self.locators.seek((rowid-1)*LOCATOR.size)
        source, offset = LOCATOR.unpack(self.locators.read(LOCATOR.size))
        handle = self.sources[source]
        handle.seek(offset)
        return prepare(parse(handle.readline()))

    def candidates(self, row, return_records=False):
        record, pool = prepare(row), set()
        for key in sorted(keys(record)):
            query = ' AND '.join(atom_token(atom) for atom in raw_atoms(key))
            bucket = [r[0] for r in self.connection.execute('SELECT rowid FROM docs WHERE docs MATCH ? LIMIT ?',
                      (query, CONFIG['bucket_cap']+1))]
            if len(bucket) <= CONFIG['bucket_cap']:
                pool.update(bucket)
        records = [self.get(i) for i in pool]
        shortlist = heapq.nlargest(CONFIG['scoring_prefilter_cap'], records, key=lambda r: (cheap_score(record, r), r[0]))
        ranked = sorted(shortlist, key=lambda r: (-score(record, r), r[0]))[:100]
        return ranked if return_records else [r[0] for r in ranked]

    def close(self):
        self.get.cache_clear()
        self.connection.close()
        self.locators.close()
        for source in self.sources:
            source.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--split', choices=['train','test'], default='train')
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--budget-gib', type=float, default=3)
    args = parser.parse_args()
    state = build(args.data, args.split, args.index, args.limit, args.budget_gib)
    print(json.dumps(state, indent=2))


if __name__ == '__main__':
    main()
