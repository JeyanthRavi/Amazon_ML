import csv
import tempfile
import unittest
from pathlib import Path

from improve_blocking import CONFIG, cheap_score, keys, prepare, score
from persistent_index import TargetIndex, build
from fast_retrieval import CachedTargetIndex
import heapq


class PersistentIndexTests(unittest.TestCase):
    def test_resumption_and_candidate_equivalence_including_common_buckets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'train').mkdir()
            targets = [['S2-a','rootinstitute.com','12 Balaji Nagar Nellore','India'],
                       ['S2-b','Ectocalovio','E-122 Patel Nagar Gwalior','India']]
            targets += [[f'S2-common{i}','Common Business','Same Address Road','US'] for i in range(105)]
            targets += [['S2-unique','Common Business','22 Uniquelane Address Road','US']]
            targets3 = [['S3-c','शिवम सिस्टम्स','W/O DR RC PANDEY VARANASI','India'],
                        ['S3-d','Société Française','12 rue Paris','France']]
            for s, data in [(2,targets),(3,targets3)]:
                with (root/f'train/train_source{s}.tsv').open('w',newline='') as f:
                    writer=csv.writer(f,delimiter='\t');writer.writerow(['entity_id','business_name','business_address','country']);writer.writerows(data)
            first=build(root,'train',root/'index',limit=50)
            self.assertFalse(first['complete'])
            with self.assertRaises(ValueError):
                TargetIndex(root/'index')
            final=build(root,'train',root/'index')
            self.assertTrue(final['complete'])
            self.assertEqual(final['count'],len(targets)+len(targets3))
            index=TargetIndex(root/'index')
            cached=CachedTargetIndex(root/'index')
            records=[prepare(r) for r in targets+targets3]
            query_rows=[['S1-q','Root Institute Ltd','12 Balaji Nagar Nellore','India'],
                        ['S1-r','AC Hardware','E-122 Patel Nagar Gwalior','India'],
                        ['S1-s','Shivam Systems','W/O Dr Rc Pandey Varanasi','India'],
                        ['S1-t','Common Business','Same Address Road','US'],
                        ['S1-rare','Common Business','22 Uniquelane Address Road','US'],
                        ['S1-u','Societe Francaise','12 rue Paris','France']]
            for row in query_rows:
                q=prepare(row); pool={}
                for key in keys(q):
                    bucket=[r for r in records if key in keys(r)]
                    if len(bucket)<=CONFIG['bucket_cap']:
                        pool.update({r[0]:r for r in bucket})
                short=heapq.nlargest(500,pool.values(),key=lambda r:(cheap_score(q,r),r[0]))
                expected=[r[0] for r in sorted(short,key=lambda r:(-score(q,r),r[0]))[:100]]
                self.assertEqual(index.candidates(row),expected)
                self.assertEqual(cached.candidates(row),expected)
            index.close()
            cached.close()


if __name__ == '__main__':
    unittest.main()
