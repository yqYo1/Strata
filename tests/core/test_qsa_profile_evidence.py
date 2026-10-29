"""CPU-only rejection controls for the finite QSA attribution observer.

These cover log/coverage admission, not native arithmetic or performance.
Repository LICENSE; own-fork fixtures, with the pinned native log format.
"""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/sycl'))
from qsa_pair_model_profile import d, profile


def window():
    n, chunks = 117488, 15
    coarse = []
    qsa = []
    for layer in range(48):
        active = layer % 4 == 3
        coarse.append(dict(layer=layer, wall_ms=[1.] * 12,
            queue_ms=[0. if layer == 0 else 2., 2. if active else 0., 0. if active else 2., 2., 2., 0.],
            samples=[0 if layer == 0 else chunks, chunks if active else 0,
                     0 if active else chunks, 2, 1, 0], upload_bytes=0,
            input_bytes_host_device=[0, 0 if layer == 0 else n * 10240 * 2],
            store_bytes_host_device=[0, 0], input_queue_host_device_ms=[0., 0. if layer == 0 else 2.],
            store_queue_host_device_ms=[0., 0.]))
        qsa.append(dict(layer=layer, expected_chunks=chunks if active else 0,
            queue_ms=[3.] * 7 if active else [0.] * 7,
            host_wall_ms=[.1] * 7 if active else [0.] * 7,
            samples=[chunks] * 7 if active else [0] * 7,
            cap=2051 if active else 0, expected_queries=n if active else 0,
            dispatch_chunks=chunks if active else 0, query_total=n if active else 0,
            matrix_chunks=chunks if active else 0, fallback_chunks=0,
            matrix_queries=n if active else 0, fallback_queries=0, staged_chunks=0))
    def total(rows):
        result = dict(layer=-1)
        for key in rows[0].keys() - {'layer'}:
            if key == 'cap': result[key] = 2051
            elif isinstance(rows[0][key], list):
                result[key] = [sum(row[key][i] for row in rows) for i in range(len(rows[0][key]))]
            else: result[key] = sum(row[key] for row in rows)
        return result
    def line(prefix, row):
        return prefix + ' '.join(k + '=' + (','.join(map(str, v)) if isinstance(v, list) else str(v))
                                 for k, v in row.items())
    common = (' complete=1 errors=0 first_status=0 wrapper_ms=120000 '
              'denominator=started_through_checked_lease_retirement teardown_unmeasured=1 '
              'queue_intervals=NONEXCLUSIVE_including_host_submission_gaps ')
    text = ['strata lm profile: v=2' + common + 'bytes=REQUESTED_not_wire '
        'wall_order=source,upload_submit,upload_drain,activation,input_submit_host,input_submit_device,'
        'store_wall_host,store_wall_device,store_drain_host,store_drain_device,chunk_drain,wrapper_drain '
        'queue_order=input,QSA,GDN,MoE,upload,store']
    text += [line('strata lm profile: ', row) for row in [*coarse, total(coarse)]]
    text += ['strata lm qsa profile: v=1' + common + 'granularity=QSA_LAYER_CHUNK '
        'sections=projections,kv_stage,kv_append_query,indexer_append,selection,attention,output_projection '
        'host_wall=NONEXCLUSIVE_submission_and_helper_waits_not_CPU_or_GPU_service markers_included=1 '
        'kv_stage_noop_sample=1 dispatch=ACTUAL_matrix_used_result queries=SUM_T_not_request_tokens']
    text += [line('strata lm qsa profile: ', row) for row in [*qsa, total(qsa)]]
    return '\n'.join(text)


class ProfileEvidence(unittest.TestCase):
    def test_valid_mode10_counts_windows_not_chunks(self):
        result = profile(window(), 117488)
        self.assertEqual(result['coarse']['aggregate']['samples'], [705, 180, 540, 96, 48, 0])
        self.assertEqual(result['qsa']['aggregate']['query_total'], 1409856)

    def test_missing_or_duplicate_layer(self):
        lines = window().splitlines()
        for bad in ('\n'.join(lines[:-1]), '\n'.join([*lines, lines[-1]]),
                    window().replace('strata lm qsa profile: layer=3 ', 'strata lm qsa profile: layer=7 ')):
            with self.assertRaises(Exception): profile(bad, 117488)

    def test_failed_header_or_changed_semantics(self):
        for old, new in (('complete=1', 'complete=0'), ('errors=0', 'errors=1'),
                         ('first_status=0', 'first_status=1'),
                         ('NONEXCLUSIVE_including_host_submission_gaps', 'GPU_service')):
            with self.assertRaises(Exception): profile(window().replace(old, new, 1), 117488)

    def test_partial_query_or_wrong_staging(self):
        for old, new in (('query_total=117488', 'query_total=117487'),
                         ('matrix_chunks=15', 'matrix_chunks=14'), ('staged_chunks=0', 'staged_chunks=15')):
            with self.assertRaises(Exception): profile(window().replace(old, new, 1), 117488)

    def test_wrong_extent_or_window_census(self):
        with self.assertRaises(Exception): profile(window(), 117487)
        with self.assertRaises(Exception):
            profile(window().replace('samples=0,0,15,2,1,0', 'samples=0,0,15,15,1,0'), 117488)

    def test_nan_and_aggregate_mismatch(self):
        with self.assertRaises(Exception): profile(window().replace('queue_ms=2.0', 'queue_ms=nan', 1), 117488)
        with self.assertRaises(Exception): profile(window().replace('query_total=1409856', 'query_total=1409855'), 117488)

    def test_duplicate_key_and_denominator(self):
        with self.assertRaises(Exception): profile(window().replace('v=2 ', 'v=2 v=2 ', 1), 117488)
        with self.assertRaises(Exception):
            profile(window().replace('wrapper_ms=120000', 'wrapper_ms=130000', 1), 117488)

    def test_missing_complete_window(self):
        for prefix in ('strata lm profile: ', 'strata lm qsa profile: '):
            text = '\n'.join(line for line in window().splitlines() if not line.startswith(prefix))
            with self.assertRaises(Exception): profile(text, 117488)

    def test_missing_required_key(self):
        with self.assertRaises(Exception):
            profile(window().replace(' matrix_queries=117488', '', 1), 117488)

    def test_wrong_active_cap_despite_coherent_aggregate(self):
        with self.assertRaises(Exception):
            profile(window().replace('cap=2051', 'cap=2050'), 117488)

    def test_completion_does_not_replace_host_acceptance_protocol(self):
        validation = d.m.runtime.validation
        with patch.object(validation.subprocess, 'check_output', return_value=''), \
             patch.object(validation.guards, 'driver_guard', return_value={}), \
             patch.object(validation.Path, 'read_text', return_value='mock-boot'):
            for accepted in (None, False, True):
                record = dict(complete=True)
                if accepted is not None: record['host_probe_accepted'] = accepted
                validation.finish(record, SimpleNamespace(active=None, commands=[]), 'mock-cursor', 'mock-boot')
                self.assertEqual(record['passed'], accepted is True)

    def test_host_acceptance_cannot_override_fresh_fault(self):
        validation = d.m.runtime.validation
        with patch.object(validation.subprocess, 'check_output', return_value='xe GPU reset'), \
             patch.object(validation.guards, 'driver_guard', return_value={}), \
             patch.object(validation.Path, 'read_text', return_value='mock-boot'):
            record = dict(complete=True, host_probe_accepted=True)
            validation.finish(record, SimpleNamespace(active=None, commands=[]), 'mock-cursor', 'mock-boot')
            self.assertFalse(record['passed'])
            self.assertEqual(record['fresh_faults'], ['xe GPU reset'])


if __name__ == '__main__':
    unittest.main()
