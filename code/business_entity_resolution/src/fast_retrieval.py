"""Cache immutable full-index buckets without changing candidate semantics."""
import functools
import heapq

from improve_blocking import CONFIG, cheap_score, keys, prepare, score
from persistent_index import TargetIndex, atom_token, raw_atoms


class CachedTargetIndex(TargetIndex):
    def __init__(self,directory):
        super().__init__(directory)
        self.bucket=functools.lru_cache(maxsize=50000)(self._bucket)

    def _bucket(self,key):
        query=' AND '.join(atom_token(atom) for atom in raw_atoms(key))
        rows=tuple(r[0] for r in self.connection.execute('SELECT rowid FROM docs WHERE docs MATCH ? LIMIT ?',
                   (query,CONFIG['bucket_cap']+1)))
        return rows if len(rows)<=CONFIG['bucket_cap'] else ()

    def candidates(self,row,return_records=False):
        record=prepare(row);pool=set()
        for key in sorted(keys(record)):pool.update(self.bucket(key))
        shortlist=heapq.nlargest(CONFIG['scoring_prefilter_cap'],(self.get(i) for i in pool),key=lambda r:(cheap_score(record,r),r[0]))
        ranked=sorted(shortlist,key=lambda r:(-score(record,r),r[0]))[:100]
        return ranked if return_records else [r[0] for r in ranked]

    def close(self):
        self.bucket.cache_clear()
        super().close()
