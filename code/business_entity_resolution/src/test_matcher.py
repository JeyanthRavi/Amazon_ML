import unittest

from train_matcher import evaluate, features
from improve_blocking import prepare


class MatcherTests(unittest.TestCase):
    def test_macro_f05_includes_singletons_and_missed_blocking_matches(self):
        sample=[{'source1_entity_id':'a','matched_entity_ids':'x,y','country':'US'},
                {'source1_entity_id':'b','matched_entity_ids':'','country':'US'},
                {'source1_entity_id':'c','matched_entity_ids':'','country':'India'},
                {'source1_entity_id':'d','matched_entity_ids':'q','country':'India'}]
        candidates={'a':['x','y','z'],'b':[],'c':['z'],'d':[]}
        predicted={'a':['x','y','z'],'c':['z']}
        metric=evaluate(sample,candidates,predicted)['all']
        self.assertAlmostEqual(metric['macro_f05'],(5/7+1+0+0)/4)
        self.assertEqual((metric['tp'],metric['fp'],metric['fn']),(2,2,1))
        self.assertEqual(metric['singleton_false_positive'],1)
        self.assertAlmostEqual(metric['blocking_recall'],2/3)

    def test_empty_fields_do_not_create_name_or_address_agreement(self):
        row=prepare(['S1-a','','','France'])
        values=features(row,prepare(['S2-b','','','France']))
        self.assertEqual(values[0],0)
        self.assertEqual(values[3],0)
        self.assertEqual(values[7],0)
        self.assertEqual(values[9],0)
        self.assertEqual(values[14:17],[1,1,1])


if __name__=='__main__':unittest.main()
