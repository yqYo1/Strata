"""Exercise recovery guards/order/timeouts without a GPU, sudo, or sysfs writes."""
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import time
import types
import unittest
from unittest.mock import patch

source = (Path(__file__).with_name('recover-xe.sh')).read_text().split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0]
recovery = types.ModuleType('xe_recovery')
exec(compile(source, 'recover-xe.sh:python', 'exec'), recovery.__dict__)


class FakeDevice:
    def __init__(self, path):
        self.path, self.driver = path / '0000:05:00.0', path / 'xe'
        self.path.mkdir(); self.driver.mkdir()
        (self.path / 'driver').symlink_to(self.driver)
        (self.path / 'reset').touch()
        self.bdf = '0000:05:00.0'
        self.methods = 'flr bus'
        self.active, self.hidden, self.display, self.survivability = [], [], False, '0'

    def read(self, name):
        return self.methods

    def describe(self):
        return dict(boot_vga='0', connectors={'DP-1': 'connected' if self.display else 'disconnected'},
                    survivability_mode=self.survivability)

    def clients(self):
        return self.active, self.hidden


class FakeRunner:
    def __init__(self, device, fail=None, stranded=False):
        self.device, self.fail, self.stranded = device, fail, False
        self.leave_stranded, self.calls = stranded, []

    def write(self, label, path, value):
        self.calls.append((label, path.name, value))
        if label == self.fail:
            self.stranded = self.leave_stranded
            raise recovery.RecoveryError(label)
        if label == 'unbind':
            (self.device.path / 'driver').unlink()
        if label.startswith('bind'):
            (self.device.path / 'driver').symlink_to(self.device.driver)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.device = FakeDevice(self.directory)

    def tearDown(self):
        self.temp.cleanup()

    def test_rebind_order(self):
        runner = FakeRunner(self.device)
        recovery.reset(self.device, runner, 'rebind')
        self.assertEqual([c[0] for c in runner.calls], ['unbind', 'bind'])

    def test_flr_restores_exact_method_order(self):
        runner = FakeRunner(self.device)
        recovery.reset(self.device, runner, 'flr')
        self.assertEqual([c[0] for c in runner.calls],
                         ['unbind', 'select-flr', 'function-reset', 'restore-reset-methods', 'bind'])
        self.assertEqual(runner.calls[3][2], 'flr bus')

    def test_no_flr_support_no_unbind(self):
        self.device.methods = 'bus'
        runner = FakeRunner(self.device)
        with self.assertRaises(recovery.RecoveryError):
            recovery.reset(self.device, runner, 'flr')
        self.assertFalse(runner.calls)

    def test_bus_isolation(self):
        runner = FakeRunner(self.device)
        recovery.reset(self.device, runner, 'bus')
        self.assertEqual(runner.calls[1][2], 'bus')
        (self.directory / '0000:05:00.1').mkdir()
        runner = FakeRunner(self.device)
        with self.assertRaisesRegex(recovery.RecoveryError, 'sole function'):
            recovery.reset(self.device, runner, 'bus')
        self.assertFalse(runner.calls)

    def test_guards_prevent_all_writes(self):
        for key, value in [('active', [{'pid': 123}]), ('hidden', [124]),
                           ('display', True), ('survivability', '1')]:
            with self.subTest(key=key):
                setattr(self.device, key, value)
                runner = FakeRunner(self.device)
                with self.assertRaises(recovery.RecoveryError):
                    recovery.reset(self.device, runner, 'flr')
                self.assertFalse(runner.calls)
                setattr(self.device, key, False if key == 'display' else '0' if key == 'survivability' else [])

    def test_failure_restores_methods_and_binding(self):
        runner = FakeRunner(self.device, 'function-reset')
        with self.assertRaises(recovery.RecoveryError):
            recovery.reset(self.device, runner, 'flr')
        self.assertEqual([c[0] for c in runner.calls][-2:], ['restore-after-failure', 'bind-after-failure'])
        self.assertTrue((self.device.path / 'driver').exists())

    def test_stranded_writer_prevents_cleanup_write(self):
        runner = FakeRunner(self.device, 'function-reset', stranded=True)
        with self.assertRaises(recovery.RecoveryError):
            recovery.reset(self.device, runner, 'flr')
        self.assertEqual([c[0] for c in runner.calls], ['unbind', 'select-flr', 'function-reset'])

    def test_runner_timeout_and_nonzero(self):
        runner = recovery.Runner(self.directory)
        with self.assertRaises(recovery.RecoveryError):
            runner.run('nonzero', [sys.executable, '-c', 'raise SystemExit(7)'])
        self.assertEqual(runner.calls[-1]['exit_code'], 7)
        started = time.monotonic()
        with self.assertRaises(recovery.RecoveryError):
            runner.run('timeout', [sys.executable, '-c', 'import time; time.sleep(60)'], seconds=0.1)
        self.assertLess(time.monotonic() - started, 3)
        self.assertTrue(runner.calls[-1]['timed_out'])
        self.assertFalse(runner.calls[-1]['still_alive'])

    def test_health_rejects_faults_even_after_exact_pass(self):
        binary = self.directory / 'probe'; binary.write_text(''); binary.chmod(0o755)
        runner = types.SimpleNamespace(output=self.directory, stranded=False)
        def run(label, argv, **kwargs):
            return {'health-runtime': '', 'before-health-cursor': '-- cursor: cursor-1\n',
                    'health': 'PASS 0000:05:00.0: 3 rounds, 16384 exact words each\n',
                    'health-kernel': 'xe 0000:05:00.0: GT0 Engine reset\n'}[label]
        runner.run = run
        with patch.object(recovery.os, 'geteuid', return_value=1000):
            with self.assertRaisesRegex(recovery.RecoveryError, 'new xe'):
                recovery.check_health(self.device, runner, binary)
        self.assertTrue((self.directory / 'health-xe.txt').exists())

    def test_missing_adapter_dependency_stops_before_gpu_probe(self):
        binary = self.directory / 'probe'; binary.write_text(''); binary.chmod(0o755)
        calls = []
        def run(label, argv, **kwargs):
            calls.append(label)
            self.assertIn('/opt/intel/oneapi/umf/1.1/lib', kwargs['env']['LD_LIBRARY_PATH'].split(':'))
            raise recovery.RecoveryError('missing libumf.so.1')
        runner = types.SimpleNamespace(output=self.directory, run=run)
        with self.assertRaisesRegex(recovery.RecoveryError, 'libumf'):
            recovery.check_health(self.device, runner, binary)
        self.assertEqual(calls, ['health-runtime'])

    def test_health_rejects_wrong_device_or_partial_output(self):
        binary = self.directory / 'probe'; binary.write_text(''); binary.chmod(0o755)
        for result in ['PASS 0000:06:00.0: 3 rounds, 16384 exact words each', 'round 0 complete']:
            runner = types.SimpleNamespace(output=self.directory, stranded=False)
            runner.run = lambda label, argv, **kwargs: '-- cursor: c\n' if 'cursor' in label else result if label == 'health' else ''
            with patch.object(recovery.os, 'geteuid', return_value=1000):
                with self.assertRaises(recovery.RecoveryError):
                    recovery.check_health(self.device, runner, binary)

    def test_health_uses_provided_cursor_and_includes_display_faults(self):
        binary = self.directory / 'probe'; binary.write_text(''); binary.chmod(0o755)
        runner = types.SimpleNamespace(output=self.directory, stranded=False)
        calls = []
        def run(label, argv, **kwargs):
            calls.append((label, argv))
            self.assertNotIn('before-health-cursor', label)
            return ('PASS 0000:05:00.0: 3 rounds, 16384 exact words each\n'
                    if label == 'health' else 'xe 0000:05:00.0: GT0 Engine reset\n')
        runner.run = run
        with patch.object(recovery.os, 'geteuid', return_value=1000):
            with self.assertRaisesRegex(recovery.RecoveryError, 'new xe'):
                recovery.check_health(self.device, runner, binary, cursor='before-display-start')
        self.assertIn('before-display-start', calls[-1][1])


class RecoveryEntryTests(unittest.TestCase):
    """Run the actual shell entry after escalation, with harmless fake helpers."""
    def run_entry(self, flr_result, bus_result=0, arguments=(), display_result=None, resume_result=3):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = Path(__file__).with_name('recover-gpu.sh').read_text()
            escalation = ('if (( EUID != 0 )); then\n'
                          '    exec /usr/bin/sudo -- /bin/bash "$(readlink -f -- "$0")"\n'
                          'fi\n')
            self.assertEqual(entry.count(escalation), 1)
            # Only the unavailable privilege escalation is omitted. The actual
            # argument checks, helper lookup and reset control flow are run.
            script = root / 'strata-gpu-recover'
            script.write_text(entry.replace(escalation, '# CPU test: already past escalation\n'))
            core = root / 'strata-xe-recover-core'
            core.write_text('#!/bin/bash\necho "$*" >> "$(dirname "$0")/calls"\n'
                            'case "$3" in\n'
                            f'flr) exit {flr_result};;\n'
                            f'bus) exit {bus_result};;\n'
                            '*) exit 99;;\nesac\n')
            probe = root / 'strata-xe-health'
            probe.write_text('#!/bin/bash\nexit 99\n'); probe.chmod(0o755)
            if display_result is not None:
                (root / 'strata-xe-display-recover').write_text(
                    '#!/bin/bash\n'
                    'if [[ "$*" == --resume ]]; then\n'
                    'echo "--apply --method resume" >> "$(dirname "$0")/calls"\n'
                    f'exit {resume_result}\nfi\n'
                    '(( $# == 0 )) || exit 99\n'
                    'echo "--apply --method display" >> "$(dirname "$0")/calls"\n'
                    f'exit {display_result}\n')
            result = subprocess.run(['/bin/bash', str(script), *arguments],
                                    capture_output=True, text=True, timeout=5)
            calls = (root / 'calls').read_text().splitlines() if (root / 'calls').exists() else []
            return result.returncode, [line.split()[2] for line in calls]

    def test_first_success_stops(self):
        self.assertEqual(self.run_entry(0), (0, ['flr']))

    def test_failure_tries_bus_once(self):
        self.assertEqual(self.run_entry(1), (0, ['flr', 'bus']))
        self.assertEqual(self.run_entry(1, 1), (1, ['flr', 'bus']))

    def test_refusal_or_interruption_never_retries(self):
        for code in (4, 130, 137, 143):
            with self.subTest(code=code):
                self.assertEqual(self.run_entry(code), (code, ['flr']))

    def test_options_rejected_before_helpers(self):
        self.assertEqual(self.run_entry(0, arguments=('--apply',)), (2, []))

    def test_refusal_never_delegates_to_gui_shutdown(self):
        self.assertEqual(self.run_entry(4, display_result=3), (4, ['resume', 'flr']))
        self.assertEqual(self.run_entry(4, display_result=4), (4, ['resume', 'flr']))
        self.assertEqual(self.run_entry(0, display_result=3), (0, ['resume', 'flr']))
        self.assertEqual(self.run_entry(143, display_result=3), (143, ['resume', 'flr']))

    def test_resumed_restoration_stops_before_any_new_reset(self):
        for code in [0, 1, 4]:
            with self.subTest(code=code):
                self.assertEqual(self.run_entry(0, display_result=3, resume_result=code), (code, ['resume']))


if __name__ == '__main__':
    unittest.main()
