#!/usr/bin/env python3
"""Reject partial or mismatched attribution before selecting CPU optimization."""
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/sycl'))
import cpu_service_evidence as service
from test_cpu_work_census import fixture as census_fixture


def fixture(request=1):
    text, binding, formats, membership = census_fixture(request)
    lines = [f'strata CPU_SERVICE request={request} ' + ' '.join(f'{k}={v}' for k, v in service.HEADER.items())]
    calls = wall = cpu = 0
    for gu in (18, 22):
        for nt, jobs in ((1, 312), (4, 192)):
            for op, width in (('GU', 640), ('HQ', 640), ('DOWN', 2560)):
                count = jobs if op == 'HQ' else 2 * jobs
                rows = jobs * width
                lines.append(f'strata CPU_SERVICE_BIN request={request} gu={gu} down=20 nt={nt} op={op} '
                    f'calls={count} rows={rows} token_rows={rows*nt} wall_ns=20000 cpu_ns=10000')
                calls += count
                wall += 20000
                cpu += 10000
    lines.append(f'strata CPU_SERVICE_THREAD request={request} slot=64 tid=9123 calls={calls} wall_ns={wall} cpu_ns={cpu}')
    marker = f'strata cpu work: bin request={request} bin=0 '
    text = text.replace(marker, '\n'.join(lines) + '\n' + marker)
    return text, binding, formats, membership


class ServiceEvidenceTests(unittest.TestCase):
    def extract(self, text=None):
        original, binding, formats, membership = fixture()
        return service.extract(original if text is None else text, [binding], formats, membership)

    def test_same_request_leaf_and_independent_census_conserve(self):
        row, = self.extract()
        self.assertEqual(row['cpu_jobs'], 1008)
        self.assertEqual(row['cpu_entries'], 2160)
        self.assertEqual(len(row['format_bins']), 12)
        self.assertEqual(row['thread_count'], 1)
        self.assertEqual(row['calls'], 5040)
        self.assertEqual((row['wall_ns'], row['cpu_ns']), (240000, 120000))
        self.assertTrue(row['same_request_census'])

    def test_requests_stay_separate_and_pool_owner_stays_fixed(self):
        one, binding, formats, membership = fixture()
        two, second, _, _ = fixture(2)
        rows = service.extract(one + two, [binding, second], formats, membership)
        self.assertEqual([r['request'] for r in rows], [1, 2])
        with self.assertRaises(RuntimeError):
            service.extract(one + two.replace('tid=9123', 'tid=9124'), [binding, second], formats, membership)

    def test_worker_threads_conserve_and_do_not_move_between_slots(self):
        one, binding, formats, membership = fixture()
        two, second, _, _ = fixture(2)
        def split(text, request, slot):
            old = f'strata CPU_SERVICE_THREAD request={request} slot=64 tid=9123 calls=5040 wall_ns=240000 cpu_ns=120000'
            new = f'strata CPU_SERVICE_THREAD request={request} slot={slot} tid=9100 calls=2520 wall_ns=120000 cpu_ns=60000\n'
            new += f'strata CPU_SERVICE_THREAD request={request} slot=64 tid=9123 calls=2520 wall_ns=120000 cpu_ns=60000'
            return text.replace(old, new)
        first = split(one, 1, 0)
        rows = service.extract(first + split(two, 2, 0), [binding, second], formats, membership)
        self.assertEqual([r['thread_count'] for r in rows], [2, 2])
        with self.assertRaises(RuntimeError):
            service.extract(first + split(two, 2, 1), [binding, second], formats, membership)

    def test_refuses_invalid_header_rows_clocks_and_thread_conservation(self):
        text, _, _, _ = fixture()
        variants = [text.replace('valid=1 error_mask=0', 'valid=0 error_mask=0'),
            text.replace('scope=decode_pool_leaves', 'scope=prefill'),
            text.replace('op=GU calls=624', 'op=GU calls=623'),
            text.replace('op=HQ calls=312', 'op=HQ calls=311'),
            text.replace('rows=199680 token_rows=199680', 'rows=199679 token_rows=199680', 1),
            text.replace('token_rows=199680', 'token_rows=199679', 1),
            text.replace('cpu_ns=10000', 'cpu_ns=20001', 1),
            text.replace('wall_ns=20000', 'wall_ns=18446744073709551616', 1),
            text.replace('slot=64', 'slot=5'), text.replace('tid=9123', 'tid=0'),
            text.replace('calls=5040', 'calls=5041'), text.replace('cpu_ns=120000', 'cpu_ns=120001'),
            text.replace('request=1 gu=18', 'request=2 gu=18', 1),
            text.replace('gu=18 down=20 nt=1 op=GU', 'gu=18 down=20 nt=2 op=GU', 1),
            text.replace('CPU_SERVICE_BIN', 'CPU_SERVICE_OTHER', 1),
            text.replace('op=GU', 'op=OTHER', 1), text.replace('op=GU', 'op=GU op=GU', 1),
            text + 'strata CPU_SERVICE request=2\n']
        for changed in variants:
            with self.subTest(changed=changed[-80:]), self.assertRaises(RuntimeError):
                self.extract(changed)

    def test_refuses_missing_duplicate_reordered_or_outside_request_observers(self):
        text, _, _, _ = fixture()
        lines = text.splitlines()
        leaf = next(i for i, line in enumerate(lines) if line.startswith('strata CPU_SERVICE_BIN'))
        for changed in (lines[:leaf] + lines[leaf+1:], lines[:leaf] + [lines[leaf]] + lines[leaf:],
                lines[:leaf] + [lines[leaf+1], lines[leaf]] + lines[leaf+2:]):
            with self.assertRaises(RuntimeError):
                self.extract('\n'.join(changed) + '\n')
        observer = [line for line in lines if line.startswith('strata CPU_SERVICE')]
        rest = [line for line in lines if not line.startswith('strata CPU_SERVICE')]
        for changed in (observer + rest, rest + observer):
            with self.assertRaises(RuntimeError):
                self.extract('\n'.join(changed) + '\n')

    def test_independent_census_failure_is_not_hidden_by_valid_leaf_timing(self):
        text, _, _, _ = fixture()
        with self.assertRaises(RuntimeError):
            self.extract(text.replace('jobs_NT=312,0,0,192,0,0,0,0', 'jobs_NT=311,0,0,192,0,0,0,0', 1))


if __name__ == '__main__':
    unittest.main()
