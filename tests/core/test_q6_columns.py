"""Reject invalid evidence before advancing the complete Q6 component."""
import copy
import json
import sys
from pathlib import Path
import unittest
from unittest import mock
from tempfile import TemporaryDirectory
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/sycl'))
import measure_q6_columns as comparison
import run_q6_columns as runner


def samples():
    lines = ['Q6_TIMER pairs=2048 median_seconds=0.00000002']
    for ni, no, calls in comparison.SHAPES:
        for nc in (1, 2, 3, 4, 8):
            for batch in range(5):
                for position in (0, 1):
                    arm = (batch + nc + position) % 2
                    lines.append(f'Q6_SAMPLE ni={ni} no={no} nc={nc} arm={arm} '
                        f'batch={batch} position={position} calls={calls} '
                        f'seconds={.125 if arm else .25}')
    return lines


class ClosedAudit(unittest.TestCase):
    def valid(self):
        return dict(stage='closed', passed=True, complete=True, active=False,
            ownership_closed=True, processes=[dict(index=i, passed=True, complete=True,
                ownership_closed=True, native_exit_code=0, fresh_faults=[]) for i in range(6)])

    def test_complete_six_process_audit(self):
        row = self.valid()
        with mock.patch.object(comparison, 'committed', return_value=row):
            self.assertIs(comparison.cohort(), row)

    def test_incomplete_failed_or_unclosed_audit(self):
        changes = [('stage', 'running'), ('passed', False), ('complete', False),
                   ('active', True), ('ownership_closed', False)]
        for key, value in changes:
            row = self.valid(); row[key] = value
            with self.subTest(key=key), mock.patch.object(comparison, 'committed', return_value=row):
                with self.assertRaises(RuntimeError): comparison.cohort()
        for key, value in [('passed', False), ('complete', False), ('ownership_closed', False),
                           ('native_exit_code', 1), ('fresh_faults', ['new fault'])]:
            row = self.valid(); row['processes'][2][key] = value
            with self.subTest(process_key=key), mock.patch.object(comparison, 'committed', return_value=row):
                with self.assertRaises(RuntimeError): comparison.cohort()

    def test_wrong_count_or_duplicate_process(self):
        short = self.valid(); short['processes'].pop()
        duplicate = self.valid(); duplicate['processes'][-1] = copy.deepcopy(duplicate['processes'][0])
        for row in (short, duplicate):
            with mock.patch.object(comparison, 'committed', return_value=row):
                with self.assertRaises(RuntimeError): comparison.cohort()


class DurableAdmission(unittest.TestCase):
    def test_changed_generated_command_refused_before_inputs_or_build(self):
        row = dict(sources={}, runtime_pins={}, commands=[['compile', ['icpx', '-O3']]])
        frozen = comparison.compact(row)
        row['commands'][0][1].append('-ffast-math')
        with TemporaryDirectory() as directory:
            root = Path(directory); (root/'record.json').write_text(json.dumps(row))
            with mock.patch.object(comparison, 'ROOT', root), \
                 mock.patch.object(comparison, 'committed', return_value=frozen), \
                 mock.patch.object(comparison, 'inputs_current') as check:
                with self.assertRaises(RuntimeError): comparison.current()
                check.assert_not_called()

    def test_active_owner_identity_is_durable(self):
        identity = dict(pid=17, start_ticks=29, sid=17, pgid=17)
        supervisor = SimpleNamespace(active=SimpleNamespace(pid=17),
            commands=[dict(label='native', direct_identity=identity, owners=[identity],
                           session_empty=False, direct_child_reaped=False)])
        with TemporaryDirectory() as directory:
            path = Path(directory)/'owner.json'; comparison.persist_owner(supervisor, path)
            row = json.loads(path.read_text())
            self.assertEqual(row['active_pid'], 17)
            self.assertEqual(row['commands'][0]['direct_identity'], identity)
            supervisor.active = None
            supervisor.commands[0].update(session_empty=True, direct_child_reaped=True, exit_code=0)
            comparison.persist_owner(supervisor, path)
            self.assertIsNone(json.loads(path.read_text())['active_pid'])

    def test_fault_input_and_owner_errors_are_all_retained(self):
        with mock.patch.object(comparison, 'fresh_faults', return_value=['xe fresh fault']), \
             mock.patch.object(comparison, 'inputs_current', side_effect=RuntimeError('driver drift')), \
             mock.patch.object(comparison.guards, 'no_model_owners', side_effect=RuntimeError('owner present')):
            result = comparison.native_postchecks({}, 'cursor')
        self.assertFalse(result['passed'])
        self.assertEqual(result['faults']['evidence'], ['xe fresh fault'])
        self.assertIn('driver drift', result['inputs_boot_driver']['error'])
        self.assertIn('owner present', result['model_owners']['error'])

    def test_final_owner_failure_cannot_leave_success(self):
        row = dict(stage='closed', passed=True, complete=True)
        with mock.patch.object(comparison.guards, 'no_model_owners', side_effect=RuntimeError('owner present')), \
             mock.patch.object(comparison, 'save') as save:
            self.assertFalse(comparison.final_owner_guard(row))
        self.assertEqual(row['stage'], 'failed-final-owner')
        self.assertFalse(row['passed'])
        self.assertFalse(row['complete'])
        save.assert_called_once_with(row)


class Evidence(unittest.TestCase):
    def test_complete_production_shapes(self):
        result = comparison.parse_speed('\n'.join(samples()))
        self.assertEqual(len(result), 35)
        self.assertTrue(all(s['median_time_ratio'] == .5 for s in result))

    def test_missing_duplicate_and_wrong_shape(self):
        lines = samples()
        for changed in (lines[:-1], lines + [lines[1]],
                        [lines[0], lines[1].replace('ni=2560', 'ni=1280')] + lines[2:]):
            with self.subTest(first=changed[0], count=len(changed)), self.assertRaises(RuntimeError):
                comparison.parse_speed('\n'.join(changed))

    def test_wrong_order_and_call_count(self):
        lines = samples()
        for wrong in ('position=7', 'calls=1'):
            field = 'position=0' if wrong.startswith('position') else 'calls=256'
            with self.subTest(wrong=wrong), self.assertRaises(RuntimeError):
                comparison.parse_speed('\n'.join([lines[0], lines[1].replace(field, wrong)] + lines[2:]))

    def test_short_and_nonfinite_batch(self):
        lines = samples()
        for value in ('0.009', 'nan', 'inf'):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                comparison.parse_speed('\n'.join([lines[0], lines[1].replace('seconds=0.125',
                    'seconds=' + value)] + lines[2:]))

    def test_missing_duplicate_nonfinite_and_expensive_clock_screen(self):
        lines = samples()
        cases = [lines[1:], lines + [lines[0]]]
        cases += [[lines[0].replace('0.00000002', value)] + lines[1:]
                  for value in ('0', 'nan', 'inf', '0.001')]
        cases.append([lines[0].replace('pairs=2048', 'pairs=1')] + lines[1:])
        for changed in cases:
            with self.subTest(timer=changed[0]), self.assertRaises(RuntimeError):
                comparison.parse_speed('\n'.join(changed))


class ValidationTrace(unittest.TestCase):
    begin = 'strata UR validation: pid=17 phase=begin loader=/qualified/loader\n'
    negative = ('   <--- urPlatformGetApiVersion(.hPlatform = nullptr, '
        '.pVersion = 0xabcdef (-1.65535)) -> UR_RESULT_ERROR_INVALID_NULL_HANDLE;\n')
    proof = ('strata UR validation: pid=17 phase=before-main threads_before=1 '
        'preinit=37 config_create=0 layers_query=0 enable=0 init=0 config_release=0 '
        'checked=46 untouched=1 retained_ref=1\n')

    def test_exact_pre_main_probe_and_quiet_mode(self):
        self.assertEqual(runner.validation_log(self.begin+self.negative+self.proof,
            17, '/qualified/loader', True)['dispatch_result'], 46)
        runner.validation_log(self.begin+self.proof, 17, '/qualified/loader', False)

    def test_wrong_location_duplicate_api_and_fresh_error(self):
        cases = [self.begin+self.proof+self.negative,
            self.begin+self.negative*2+self.proof,
            self.begin+self.negative.replace('urPlatformGetApiVersion', 'urUSMDeviceAlloc')+self.proof,
            self.begin+self.negative+self.proof+'unexpected UR_RESULT_ERROR_DEVICE_LOST\n',
            self.begin+self.negative+self.proof+'[WARNING] bad parameter\n']
        for text in cases:
            with self.subTest(text=text), self.assertRaises(RuntimeError):
                runner.validation_log(text, 17, '/qualified/loader', True)


if __name__ == '__main__':
    unittest.main()
