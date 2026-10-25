"""Host-only negative controls for complete UR subscriber call records."""
from pathlib import Path
import copy
import json
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/sycl'))
from ur_call_subscriber import parse_calls


def records():
    return [dict(kind='load', pid=71, raw_ns=1),
            dict(kind='subscribe', stream='ur.call', pid=71, raw_ns=2),
            dict(kind='begin', pid=71, tid=72, instance=100, function_id=8,
                 name='urPlatformGetApiVersion', raw_ns=3, status=None),
            dict(kind='end', pid=71, tid=72, instance=100, function_id=8,
                 name='urPlatformGetApiVersion', raw_ns=4, status=46),
            dict(kind='finish', stream='ur.call', pid=71, raw_ns=5)]


def parse(rows):
    return parse_calls(''.join(json.dumps(row) + '\n' for row in rows), 71)


class SubscriberRecordTests(unittest.TestCase):
    def test_records_rejected_call_and_actual_clock_identity_without_GPU_claim(self):
        value = parse(records())
        self.assertEqual(value['call_pairs'], 1)
        self.assertEqual(value['outcomes'], [dict(name='urPlatformGetApiVersion', status=46, calls=1)])
        self.assertFalse(value['device_work_qualified'])
        self.assertFalse(value['ordinary_runtime_teardown_qualified'])

    def test_missing_load_subscription_begin_or_end_fails(self):
        for index in range(4):
            rows = records()
            del rows[index]
            with self.subTest(index=index), self.assertRaises((RuntimeError, ValueError)):
                parse(rows)

    def test_duplicate_or_out_of_order_records_fail(self):
        for index in range(5):
            rows = records()
            rows.insert(index, copy.deepcopy(rows[index]))
            with self.subTest(index=index), self.assertRaises(RuntimeError):
                parse(rows)

    def test_wrong_owner_thread_function_clock_or_status_fails(self):
        for change in (dict(pid=73), dict(tid=73), dict(function_id=9), dict(instance=101),
                       dict(name='urOther'), dict(raw_ns=2), dict(raw_ns=float('nan')),
                       dict(status=None), dict(status=True), dict(status=1 << 32)):
            rows = records()
            rows[3].update(change)
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                parse(rows)

    def test_nested_calls_need_matching_per_thread_stack(self):
        rows = records()
        nested = dict(rows[2], instance=101, function_id=9, name='urEventRelease', raw_ns=3)
        rows[3:3] = [nested, dict(nested, kind='end', status=0)]
        self.assertEqual(parse(rows)['call_pairs'], 2)
        rows[3], rows[4], rows[5] = rows[3], rows[5], rows[4]
        with self.assertRaises(RuntimeError):
            parse(rows)

    def test_function_ID_name_contract_cannot_change_between_calls(self):
        rows = records()
        rows[4:4] = [dict(rows[2], instance=101, name='urEventRelease'),
                     dict(rows[3], instance=101, name='urEventRelease')]
        with self.assertRaisesRegex(RuntimeError, 'function identity'):
            parse(rows)

    def test_finish_may_be_absent_with_retained_runtime_reference(self):
        rows = records()[:-1]
        self.assertEqual(parse(rows)['call_pairs'], 1)
        rows.append(dict(rows[2], instance=100))
        with self.assertRaises(RuntimeError):
            parse(rows)


if __name__ == '__main__':
    unittest.main()
