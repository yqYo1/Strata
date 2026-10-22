#!/usr/bin/env python3
"""Admission guards for a same-or-smaller-byte cache comparison; no GPU launch."""
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/sycl'))
import cache_placement as cache
import cpu_work_census as census


class CachePlacementTests(unittest.TestCase):
    def setUp(self):
        self.profile = Path(__file__).resolve().parents[2] / 'data/expert-profile.bin'
        self.pairs = cache.read_profile(self.profile)

    def test_real_profile_per_layer_matches_existing_census_admission(self):
        pairs = cache.per_layer(self.pairs, 1256)
        self.assertEqual(cache.membership(pairs), census.expected_membership(self.profile))
        self.assertEqual([l for l, _ in pairs[:128]], [0]*26+[1]*26+[2]*26+[3]*26+[4]*24)

    def test_remainder_and_expert_limit(self):
        pairs = cache.per_layer(self.pairs, 1256)
        self.assertEqual([m.bit_count() for m in cache.membership(pairs)], [26]*47+[34])
        for slots in (0, 128, 24577, 24575, True):
            with self.subTest(slots=slots), self.assertRaises(ValueError):
                cache.per_layer(self.pairs, slots)

    def test_prefix_stops_without_skipping_larger_expert(self):
        self.assertEqual(cache.ranked([(0, 0), (1, 1), (0, 2)], [1, 513], 512), [(0, 0)])
        self.assertEqual(cache.ranked([(0, 0), (1, 1)], [1, 513], 1024), [(0, 0), (1, 1)])

    def test_padding_and_prefix_use_slot_order(self):
        pairs = [(0, e) for e in range(128)] + [(1, 0)]
        row = cache.summary(pairs, [257, 513])
        self.assertEqual(row['prefix128_bytes'], 128*512)
        self.assertEqual(row['maximum_loan_bytes'], 768)
        self.assertEqual(row['allocated_bytes'], 128*512+768)
        self.assertEqual(row['payload_bytes'], 128*257+513)
        with self.assertRaises(ValueError): cache.summary(pairs[:128], [257, 513])
        with self.assertRaises(ValueError): cache.membership([(0, 0), (0, 0)])

    def test_refuses_invalid_sizes(self):
        for size in (0, -1, True, 1.0, 1 << 63):
            with self.subTest(size=size), self.assertRaises(ValueError): cache.aligned(size)

    def test_profile_header_pairs_reverse_and_length_are_bound(self):
        raw = self.profile.read_bytes()
        header = bytearray(raw); struct.pack_into('<I', header, 16, 1)
        duplicate = bytearray(raw); duplicate[28:32] = duplicate[24:28]
        out_of_range = bytearray(raw); struct.pack_into('<H', out_of_range, 24, 48)
        reverse = bytearray(raw); reverse[-1] ^= 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'profile'
            for changed in (raw[:-1], raw+b'x', header, duplicate, out_of_range, reverse):
                path.write_bytes(changed)
                with self.assertRaises(ValueError): cache.read_profile(path)

    def test_manifest_and_equal_budget_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'native_experts.txt'
            lines = [f'{l} 18 20 0 {257+l*16} 0 0 0' for l in range(48)]
            path.write_text('\n'.join(lines)+'\n')
            result = cache.plan(self.profile, path)
            a, b = result['baseline'], result['candidate']
            self.assertLessEqual(b['allocated_bytes'], b['nominal_cap_bytes'])
            self.assertLessEqual(b['nominal_cap_bytes'], a['allocated_bytes'])
            self.assertEqual(b['nominal_cap_bytes'], b['expert_cache_argument']*(257+47*16))
            self.assertEqual(result['common_experts']+result['removed_experts'], 1256)
            self.assertEqual(result['common_experts']+result['added_experts'], b['slots'])
            self.assertEqual(a['prefix128_bytes']+a['maximum_loan_bytes'], a['allocated_bytes'])
            self.assertEqual(b['prefix128_bytes']+b['maximum_loan_bytes'], b['allocated_bytes'])
            for changed in (lines[:-1], [lines[0]]+lines, lines[:-1]+['47 18 20 0 0 0 0 0']):
                path.write_text('\n'.join(changed)+'\n')
                with self.assertRaises(ValueError): cache.read_manifest(path)


if __name__ == '__main__':
    unittest.main()
