"""CPU controls for the separate active-from-exec collector mode."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import profile_compat_exec as mode


class ExecModeTest(unittest.TestCase):
    def test_actual_launcher_and_child_selectors(self):
        proof = json.loads(mode.RESULT.read_text())[mode.adapter.KEY]
        profile = dict(mode.adapter.profile.profiling(mode.ROOT, proof['assets']), mode='aggregate')
        command = mode.adapter.engine_once.profile_command(profile, ['/fixture'])
        self.assertEqual(command, [profile['launcher']['path'], '--session', profile['session'],
            '--device-timing', '--verbose', '/fixture'])
        row = dict(profile=profile, native_validation_probe=None, environment={
            'UNITRACE_HostTiming': '1', 'UNITRACE_ChromeCallLogging': '1',
            'UNITRACE_ChromeKernelLogging': '1', 'UNITRACE_ChromeEventBufferSize': '128'})
        # Only environment composition is under test; use a minimal immutable
        # native-probe stub as required by the existing composition helper.
        row['native_validation_probe'] = {'library': {'path': '/probe'}}
        from unittest.mock import patch
        with patch.object(mode.adapter.engine_once, 'native_environment',
                return_value=dict(row['environment'], LD_PRELOAD='/probe')):
            expected = mode.expected_environment(row)
        self.assertNotIn('UNITRACE_StartPaused', expected)
        for key in ('UNITRACE_HostTiming', 'UNITRACE_ChromeCallLogging', 'UNITRACE_ChromeKernelLogging',
                    'UNITRACE_DeviceTiming', 'UNITRACE_Verbose'):
            self.assertEqual(expected[key], '1')

    def test_fresh_environment_refuses_inherited_pause(self):
        from unittest.mock import patch
        with patch.object(mode.adapter, 'environment', return_value={'UNITRACE_StartPaused': '1'}):
            with self.assertRaisesRegex(RuntimeError, 'fresh active-from-exec'):
                mode.environment({})

    def test_select_only_the_original_passing_active_arm(self):
        original = json.loads(mode.RESULT.read_text())[mode.adapter.KEY]
        selected = mode.selection(original)
        self.assertTrue(selected['direct_and_active_passed'])
        self.assertFalse(selected['paused_construction_passed'])
        corruptions = [lambda p: p.update(passed=True),
            lambda p: p['cases'][1]['trace'].update(inner_kernels=5),
            lambda p: p['cases'][1].update(ownership_closed=False),
            lambda p: p['cases'][1].update(fresh_faults=['fault']),
            lambda p: p['cases'][2].update(passed=True),
            lambda p: p['cases'][2].update(native_exit_code=-9)]
        for corrupt in corruptions:
            bad = copy.deepcopy(original); corrupt(bad)
            with self.assertRaises(RuntimeError): mode.selection(bad)


if __name__ == '__main__': unittest.main()
