"""Regression checks based on observed failure modes in supplied training data."""
import unittest

import improve_blocking as blocker
from audit_blocking import evaluate


class BlockingRegressionTests(unittest.TestCase):
    def test_non_latin_script_marks_survive(self):
        for name in ('शिवम सिस्टम्स', 'ଗୁରୁ ସଲ୍ୟୁସନ୍ସ୍'):
            self.assertEqual(blocker.normalize(name), name)
        self.assertEqual(blocker.normalize('Société Française'), 'societe francaise')

    def test_domain_style_name_retrieves_without_address(self):
        a = blocker.prepare(['S1-a', 'Root Institute Ltd', '', 'India'])
        b = blocker.prepare(['S2-b', 'rootinstitute.com', '', 'India'])
        self.assertIn(('compact', 'india', 'rootinstitute'), blocker.keys(a) & blocker.keys(b))

    def test_address_route_survives_completely_changed_name(self):
        a = blocker.prepare(['S1-a', 'AC Hardware Private Limited', 'E-122, Site No.2 Patel Nagar, City Centre, Gwalior', 'India'])
        b = blocker.prepare(['S2-b', 'Ectocalovio', 'E-122, SITE NO.2 PATEL NAGAR, CITY CENTRE, GWALIOR', 'India'])
        self.assertTrue(any(k[0] == 'address_pair' for k in blocker.keys(a) & blocker.keys(b)))

    def test_address_route_survives_name_script_change(self):
        a = blocker.prepare(['S1-a', 'Shivam Systems', 'W/O Dr Rc Pandey, Harbo Nagarvyasbagth Tarn, Varanasi', 'India'])
        b = blocker.prepare(['S3-b', 'शिवम सिस्टम्स', 'W/O DR RC PANDEY, VARANASI', 'India'])
        self.assertTrue(any(k[0] == 'address_pair' for k in blocker.keys(a) & blocker.keys(b)))

    def test_country_is_an_open_label_and_isolates_keys(self):
        a = blocker.prepare(['S1-a', 'Root Institute', '12 Main Road', 'France'])
        b = blocker.prepare(['S2-b', 'Root Institute', '12 Main Road', 'France'])
        c = blocker.prepare(['S2-c', 'Root Institute', '12 Main Road', 'US'])
        self.assertTrue(blocker.keys(a) & blocker.keys(b))
        self.assertFalse(blocker.keys(a) & blocker.keys(c))

    def test_recall_and_reduction_have_correct_denominators(self):
        result = evaluate(['a', 'b'], {'a': {'x', 'y'}, 'b': set()}, {'a': ['x', 'z'], 'b': []},
                          {'a': 'US', 'b': 'US'}, [10], {'US': 100})['10']['all']
        self.assertEqual(result['pair_recall'], .5)
        self.assertEqual(result['mean_entity_recall_positive_only'], .5)
        self.assertEqual(result['singleton_entities'], 1)
        self.assertEqual(result['average_candidates'], 1)
        self.assertEqual(result['reduction_ratio_vs_same_country_all_pairs'], .99)


if __name__ == '__main__':
    unittest.main()
