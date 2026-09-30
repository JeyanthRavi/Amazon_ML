import csv
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import joblib
import numpy as np
from persistent_index import build
from run_inference import run
from train_matcher import FEATURE_NAMES


class ExactNameEstimator:
    def predict_proba(self,matrix):
        probability=matrix[:,3]
        return np.column_stack([1-probability,probability])


class InferenceTests(unittest.TestCase):
    def test_resumption_keeps_empty_rows_and_exact_model_candidates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'test').mkdir()
            datasets={1:[['S1-a','Alpha Unique','12 Main Road','US'],['S1-b','Sans Candidat','99 rue Libre','France'],['S1-c','Beta Unique','7 Canal Street','India']],
                      2:[['S2-a','Alpha Unique','12 Main Road','US']],3:[['S3-c','Beta Unique','7 Canal Street','India']]}
            for source,records in datasets.items():
                with (root/f'test/test_source{source}.tsv').open('w',newline='') as f:
                    writer=csv.writer(f,delimiter='\t');writer.writerow(['entity_id','business_name','business_address','country']);writer.writerows(records)
            build(root,'test',root/'index')
            joblib.dump({'model':ExactNameEstimator(),'feature_names':FEATURE_NAMES,'threshold':.5},root/'model.joblib')
            args=Namespace(data=root,split='test',index=root/'index',model=root/'model.joblib',output=root/'output',resume=False,batch_size=1,max_entities=1,min_free_gib=0)
            first=run(args);self.assertFalse(first['complete']);self.assertEqual(first['rows'],1)
            # Simulate uncommitted output bytes after an interrupted batch.
            with (root/'output/candidate_pairs.tsv').open('ab') as f:f.write(b'UNCOMMITTED\tS2-a\n')
            args.resume=True;args.max_entities=0;args.workers=2;final=run(args);self.assertTrue(final['complete']);self.assertEqual(final['rows'],3)
            with (root/'output/candidate_pairs.tsv').open() as f:candidates=list(csv.DictReader(f,delimiter='\t'))
            with (root/'output/matching_results.tsv').open() as f:matches=list(csv.DictReader(f,delimiter='\t'))
            self.assertEqual([r['source1_entity_id'] for r in matches],['S1-a','S1-b','S1-c'])
            self.assertEqual([r['matched_entity_ids'] for r in matches],['S2-a','','S3-c'])
            for a,b in zip(matches,candidates):
                self.assertEqual(a['source1_entity_id'],b['source1_entity_id'])
                self.assertTrue(set(filter(None,a['matched_entity_ids'].split(','))) <= set(filter(None,b['candidate_entity_ids'].split(','))))


if __name__=='__main__':unittest.main()
