"""B1 compiler-census admission controls; host-only, repository LICENSE."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools/sycl'))
import check_split_kv as candidate


class CoherentSplitCensus(unittest.TestCase):
    def setUp(self):
        self.row = dict(actual_inputs=dict(source_commit='owned', translation_units=137,
            uncovered=0), native_source_commit='owned', project=dict(revision='owned'),
            planned_compilation=[None] * 142,
            binaries={name: {} for name in candidate.build.TARGETS})

    def test_fresh_B1_shape(self):
        candidate.check_census(self.row)

    def test_reused_A_census_is_not_B1(self):
        self.row['actual_inputs']['translation_units'] = 136
        self.row['planned_compilation'] = [None] * 140
        with self.assertRaises(RuntimeError): candidate.check_census(self.row)

    def test_missing_same_build_KV_fixture(self):
        del self.row['binaries']['sycl_kv_copy_groups_bench']
        with self.assertRaises(RuntimeError): candidate.check_census(self.row)

    def test_uncovered_compiler_input(self):
        self.row['actual_inputs']['uncovered'] = 1
        with self.assertRaises(RuntimeError): candidate.check_census(self.row)

    def test_source_receipt_mismatch(self):
        self.row['project']['revision'] = 'another'
        with self.assertRaises(RuntimeError): candidate.check_census(self.row)

    def test_partial_link_does_not_qualify(self):
        self.row['planned_compilation'].pop()
        with self.assertRaises(RuntimeError): candidate.check_census(self.row)


if __name__ == '__main__': unittest.main()
