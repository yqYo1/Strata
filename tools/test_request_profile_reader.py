"""Host controls for request/clock binding and disjoint trace accounting."""
import copy
from decimal import Decimal
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).parent / 'sycl'))
import request_profile_reader as reader


class RequestProfileReaderTests(unittest.TestCase):
    def setUp(self):
        self.receipt = dict(schema=reader.SCHEMA, clock_domain=reader.CLOCK,
            pti_revision=reader.PTI_REVISION, unitrace_system_time='1', level_zero_only=True,
            native_exit_code=0, ownership_closed=True, fresh_faults=[], boot_unchanged=True,
            source=dict(native_source_commit='1'*40, binary_sha256='2'*64),
            native_lifetime=dict(begin_raw_ns=1000, end_raw_ns=20000), requests=[dict(
                request_id='first', request_sha256='3'*64, actual_read_tokens=32769,
                expected_read_tokens=32769, actual_reused_tokens=0, generated_tokens=32,
                thinking=True, sample_scope='first_long', begin_raw_ns=2000, end_raw_ns=12000)])
        self.trace = dict(traceEvents=[
            dict(cat='gpu_op', ph='X', name='a', ts=3, dur=4),
            dict(cat='gpu_op', ph='X', name='b', ts=5, dur=4),
            dict(cat='gpu_op', ph='X', name='late', ts=11, dur=3),
            dict(cat='gpu_op', ph='X', name='startup', ts=1, dur=Decimal('.5')),
            dict(cat='cpu_op', ph='X', name='submit', ts=0, dur=50), dict(ph='M', ts=0)])

    def test_overlap_service_union_and_crossing_are_distinct(self):
        r = reader.summarize(self.trace, self.receipt); q = r['requests'][0]
        self.assertEqual(q['contained_GPU_service_ns'], 8000)
        self.assertEqual(q['observed_GPU_interval_union_ns'], 7000)
        self.assertEqual(q['boundary_crossing_GPU_events'], 1)
        self.assertEqual(q['contained_GPU_events'], 2)
        self.assertEqual(r['outside_request_GPU_events'], 1)
        self.assertFalse(r['phase_attribution_qualified']); self.assertFalse(r['hardware_saturation_qualified'])

    def test_host_and_epoch_clock_mismatches_are_refused(self):
        for field, value in [('clock_domain', 'CLOCK_BOOTTIME'), ('unitrace_system_time', '0'),
                             ('pti_revision', 'f'*40), ('level_zero_only', False)]:
            r = dict(self.receipt, **{field:value})
            with self.assertRaises(ValueError): reader.summarize(self.trace, r)
        r = copy.deepcopy(self.trace); r['traceEvents'][0]['ts'] = 1790000000000000
        with self.assertRaises(ValueError): reader.summarize(r, self.receipt)

    def test_native_failure_and_changed_source_are_refused(self):
        for field, value in [('native_exit_code', 1), ('ownership_closed', False),
                             ('fresh_faults', ['fault']), ('boot_unchanged', False), ('source', {})]:
            with self.assertRaises(ValueError): reader.summarize(self.trace, dict(self.receipt, **{field:value}))

    def test_wrong_token_scope_thinking_and_wire_identity_are_refused(self):
        for field, value in [('actual_read_tokens', 82), ('expected_read_tokens', 65537),
                             ('actual_reused_tokens', 1), ('generated_tokens', 0), ('thinking', False),
                             ('sample_scope', 'unknown'), ('request_sha256', 'not-a-hash')]:
            r = copy.deepcopy(self.receipt); r['requests'][0][field] = value
            with self.assertRaises(ValueError): reader.summarize(self.trace, r)

    def test_request_order_and_identity_are_checked(self):
        r = copy.deepcopy(self.receipt); r['requests'].append(dict(r['requests'][0]))
        with self.assertRaises(ValueError): reader.summarize(self.trace, r)
        r['requests'][1]['request_id'] = 'next'
        with self.assertRaises(ValueError): reader.summarize(self.trace, r)
        r['requests'][1].update(begin_raw_ns=12000, end_raw_ns=19000, sample_scope='warmed')
        out = reader.summarize(self.trace, r)
        self.assertEqual(out['requests'][1]['contained_GPU_events'], 0)
        self.assertEqual(out['requests'][1]['boundary_crossing_GPU_events'], 1)

    def test_partial_multitile_nonfinite_and_negative_events_are_refused(self):
        for field, value in [('ph', 'B'), ('name', 'Tile #0: a'), ('dur', -1), ('ts', True),
                             ('dur', float('nan')), ('ts', Decimal('Infinity')), ('dur', Decimal('1e30'))]:
            t = copy.deepcopy(self.trace); t['traceEvents'][0][field] = value
            with self.assertRaises(ValueError): reader.summarize(t, self.receipt)
        with self.assertRaises(ValueError): reader.summarize(dict(traceEvents=[]), self.receipt)

    def test_emitted_double_rounding_and_empty_request_coverage(self):
        t = copy.deepcopy(self.trace); t['traceEvents'][0]['ts'] = Decimal('3.000488')
        self.assertEqual(reader.summarize(t, self.receipt)['requests'][0]['contained_GPU_service_ns'], 8000)
        r = copy.deepcopy(self.receipt); r['requests'][0].update(begin_raw_ns=15000, end_raw_ns=19000)
        with self.assertRaises(ValueError): reader.summarize(self.trace, r)

    def test_long_tail_totals_are_conserved_without_full_event_output(self):
        t = dict(traceEvents=[dict(cat='gpu_op', ph='X', name=str(i), ts=3, dur=1) for i in range(25)])
        q = reader.summarize(t, self.receipt)['requests'][0]
        self.assertEqual(len(q['top_operations']), 20)
        self.assertEqual(q['other_operation_calls'], 5)
        self.assertEqual(sum(v['service_ns'] for v in q['top_operations']) + q['other_operation_service_ns'],
                         q['contained_GPU_service_ns'])

    def test_duplicate_json_and_nonfinite_constants_are_refused(self):
        with tempfile.TemporaryDirectory() as root:
            p = Path(root) / 'input.json'
            for text in ('{"ts":1,"ts":2}', '{"ts":NaN}', '{"ts":Infinity}'):
                p.write_text(text)
                with self.assertRaises(ValueError): reader.load_json(p)
            p.write_text('{"ts":1.001}')
            self.assertEqual(reader.load_json(p)['ts'], Decimal('1.001'))


if __name__ == '__main__': unittest.main()
