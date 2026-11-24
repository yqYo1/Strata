#!/usr/bin/env python3
"""Exercise routing framing, CPU accounting and protected expert slots."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'tools/sycl'))
import dflash2_expert_routing as routing


class ExpertRoutingTests(unittest.TestCase):
    def test_routing_suffix_reconstructs_window_multiplicity_and_checks_cpu_census(self):
        formats = [(5, 6, 100)] * 48
        membership = [3] * 48
        census_sample = dict(windows=2, rows=3, cpu_jobs=21 * 48, cpu_entries=26 * 48,
                            logical_job_blob_bytes=21 * 48 * 100,
                            format_bins=[dict(gu=5, down=6, blob_bytes=100,
                                              jobs_NT=[16 * 48, 5 * 48, 0, 0, 0, 0, 0, 0])])
        pack = routing.ROUTE_ROW.pack
        zero = bytes(40)
        prefix = pack(47, 10, *range(10), zero)
        one_row_window = b''.join(pack(l, 10, *range(10), zero) for l in range(48))
        two_row_window = b''.join(pack(l, 10, *range(start, start + 10), zero)
                                  for l in range(48) for start in (0, 5))
        capture = prefix + one_row_window + two_row_window
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'routing.dat'
            path.write_bytes(capture)
            result = routing.verifier_routing_usage(path, (1, 2), census_sample, formats, membership,
                                                  max_bytes=len(capture))
            self.assertEqual(len(result), 15 * 48)
            self.assertEqual(result[5], (0, 5, [1, 1, 0, 0]))
            self.assertEqual(result[14], (0, 14, [1, 0, 0, 0]))
            with self.assertRaises(RuntimeError):
                routing.verifier_routing_usage(path, (1, 2), dict(census_sample, cpu_jobs=1),
                                              formats, membership, max_bytes=len(capture))
            invalid_row = pack(1, 10, *range(10), zero)
            for bad in (capture[:-1], prefix + invalid_row + capture[2 * len(prefix):]):
                path.write_bytes(bad)
                with self.assertRaises(RuntimeError):
                    routing.verifier_routing_usage(path, (1, 2), census_sample, formats, membership,
                                                  max_bytes=len(capture))

    def test_job_rank_prefers_two_width_one_windows_before_one_width_four_window(self):
        reference = [(l, e) for e in range(512) for l in range(48)]
        groups = [(0, 0, [0, 0, 0, 1, 0, 0, 0, 0]), (0, 1, [2, 0, 0, 0, 0, 0, 0, 0])]
        ranked = routing.rank_by_verifier_jobs(reference, [groups], cache_slots=1256, protected_prefix=0)
        self.assertEqual(ranked[0], (0, 1))
        self.assertEqual(ranked[48], (0, 0))
        self.assertEqual(ranked[1], (1, 0))
        self.assertEqual(set(ranked), set(reference))

    def test_job_rank_keeps_protected_prefill_slots_and_scores_held_out_work(self):
        reference = [(layer, expert) for expert in range(512) for layer in range(48)]
        training = [(0, 511, [5]), (4, 511, [5]), (5, 511, [5]), (6, 511, [5])]
        ranked = routing.rank_by_verifier_jobs(reference, [training], cache_slots=1256, protected_prefix=132)
        for layer in range(5):
            protected = [expert for actual, expert in ranked if actual == layer][:26]
            self.assertEqual(protected, list(range(26)))
        before = [(1 << 26) - 1] * 47 + [(1 << 34) - 1]
        after = before.copy()
        for layer in (5, 6):
            after[layer] = (before[layer] ^ (1 << 25)) | (1 << 511)
        self.assertEqual([ranked[i * 48 + 5] for i in range(2)], [(5, 0), (5, 1)])
        self.assertEqual(ranked[2 * 48 + 5], (5, 511))
        self.assertEqual(ranked[6], (6, 511))
        admitted = [expert for layer, expert in ranked if layer == 5][:26]
        self.assertEqual(sum(1 << expert for expert in admitted), after[5])
        admitted = [expert for layer, expert in ranked if layer == 6][:26]
        self.assertEqual(sum(1 << expert for expert in admitted), after[6])
        held_out = [(0, 511, [3, 0, 0, 0]), (5, 511, [0, 2, 0, 0]),
                    (5, 25, [1, 0, 1, 0]), (6, 511, [0, 0, 0, 4])]
        formats = [(5, 6, 100)] * 48
        self.assertEqual(routing.projected_cpu_expert_work(held_out, before, formats),
                         dict(cpu_jobs=9, cpu_entries=23, logical_job_blob_bytes=900))
        self.assertEqual(routing.projected_cpu_expert_work(held_out, after, formats),
                         dict(cpu_jobs=5, cpu_entries=7, logical_job_blob_bytes=500))
        with self.assertRaises(RuntimeError):
            routing.rank_by_verifier_jobs(reference, [training], cache_slots=1256, protected_prefix=1257)


if __name__ == '__main__':
    unittest.main()
