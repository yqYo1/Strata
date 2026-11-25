import unittest

import dflash2_primary_prefill_cpu_wait as cold


def captured_shape():
    wait = dict(version=4, available=1, complete=1, clock_or_accounting_error=0,
                open_waits=0, wait_throws=0, wall_s=12, main_cpu_s=7, wait_cpu_s=3,
                wait_wall_s=6, outside_wait_cpu_s=4, wait_count=49)
    for name in cold.WAIT_KINDS:
        wait.update({f'{name}_{k}': 0 for k in ('count', 'cpu_s', 'wall_s')})
    wait.update(compute_count=1, compute_cpu_s=2, compute_wall_s=2,
                compat_count=48, compat_cpu_s=1, compat_wall_s=4)

    host = dict(version=4, available=1, open_regions=0, unclassified_cpu_s=3,
                unclassified_wall_s=5)
    for name in cold.HOST_KINDS:
        host.update({f'{name}_{k}': 0 for k in ('count', 'throws', 'cpu_s', 'wall_s',
                                               'outside_wait_cpu_s', 'outside_wait_wall_s')})
    host.update(moe_status_count=48, moe_status_cpu_s=1.7, moe_status_wall_s=4.8,
                moe_status_outside_wait_cpu_s=1, moe_status_outside_wait_wall_s=1)

    status = dict(version=1, available=1, complete=1, scopes=48, completed=48,
                  open_scopes=0, open_parts=0, cpu_s=2, wall_s=5,
                  unclassified_cpu_s=0.5, unclassified_wall_s=0.5)
    for name in cold.observer.CPU_STATUS_NAMES:
        count = 96 if name in ('precheck', 'queue_context') else 48
        status[f'{name}_count'] = count
        status.update({f'{name}_{k}': 0 for k in ('throws', 'cpu_s', 'wall_s', 'wait_cpu_s', 'wait_wall_s')})
    status.update(enqueue_cpu_s=0.5, enqueue_wall_s=0.5, drain_cpu_s=1, drain_wall_s=4,
                  drain_wait_cpu_s=1, drain_wait_wall_s=4)

    return '\n'.join(prefix + ' ' + ' '.join(f'{k}={v}' for k, v in fields.items())
                     for prefix, fields in zip(cold.OBSERVER_PREFIXES, (wait, host, status), strict=True))


class ColdCpuWaitAccounting(unittest.TestCase):
    def test_status_is_nested_in_the_request_partition(self):
        result = cold.cpu_partition(captured_shape(), 48)
        self.assertEqual(result['cpu_s'], 7)
        self.assertEqual(result['wait_cpu_s'], 3)
        self.assertEqual(result['outside_wait']['moe_status'], [1, 1])
        self.assertEqual(result['nested_status']['phases']['drain']['wait_wall_s'], 4)

    def test_aggregate_mismatch_is_rejected(self):
        log = captured_shape().replace('main_cpu_s=7', 'main_cpu_s=8')
        with self.assertRaisesRegex(RuntimeError, 'whole-scope CPU intervals'):
            cold.cpu_partition(log, 48)

    def test_incomplete_or_changed_status_wrapper_is_rejected(self):
        log = captured_shape()
        for text, expected in ((log.rsplit('\n', 1)[0], 'missing or duplicate status observer'),
                               (log.replace('precheck_count=96', 'precheck_count=48'), 'status call/exception count'),
                               (log.replace('moe_status_wall_s=4.8', 'moe_status_wall_s=6'), 'nested status wrapper')):
            with self.subTest(expected=expected), self.assertRaisesRegex(RuntimeError, expected):
                cold.cpu_partition(text, 48)
