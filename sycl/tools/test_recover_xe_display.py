"""Exercise GUI recovery approval/guards/order with harmless fake backends."""
import io
import json
import os
from pathlib import Path
import pwd
import tempfile
import time
import types
import unittest
from unittest.mock import patch


def embedded(name):
    source = Path(__file__).with_name(name).read_text().split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0]
    module = types.ModuleType(name)
    exec(compile(source, name, 'exec'), module.__dict__)
    return module


core = embedded('recover-xe.sh')
display = embedded('recover-xe-display.sh')


class FakeRunner:
    def __init__(self, output):
        self.output, self.calls, self.fail = output, [], None

    def run(self, label, argv, **kwargs):
        self.calls.append({'label': label, 'argv': argv})
        if label == self.fail:
            raise core.RecoveryError(label)
        return 'active\n' if label == 'display-before' else ''


class DisplayRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.umask = os.umask(0o077)
        self.output = self.base / 'display-test'; self.output.mkdir()
        self.path = self.output / 'plan.json'
        self.script = self.base / 'strata-xe-display-recover'
        self.script.write_text('')
        (self.base / 'strata-xe-health').write_text(''); (self.base / 'strata-xe-health').chmod(0o755)
        self.user = pwd.getpwuid(os.getuid())
        self.plan = dict(approved=True, service='sddm.service', bdf=display.BDF,
                         user=self.user.pw_name, uid=self.user.pw_uid,
                         clients=[(4330, 3692)], boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                         approved_monotonic=time.monotonic(), worker_started=False,
                         display_stop_attempted=False, display_restore_requested=False)
        self.patch = patch.object(display, 'BASE', self.base); self.patch.start()
        display.save_plan(self.path, self.plan)
        self.runner = FakeRunner(self.output)
        self.fake = types.SimpleNamespace(RecoveryError=core.RecoveryError, RecoveryRefused=core.RecoveryRefused,
                    Runner=lambda output: self.runner, acquire_recovery_lock=core.acquire_recovery_lock,
                    main=lambda *args, **kwargs: 0, journal_cursor=lambda *args: 'before-GUI-start',
                    check_health=lambda *args, **kwargs: None, Device=lambda bdf: object(),
                    publish_summary=lambda *args: None)

    def tearDown(self):
        self.patch.stop(); os.umask(self.umask); self.temp.cleanup()

    def worker(self):
        with patch.object(display, 'admissible', return_value=[(4330, 3692)]):
            return display.worker(self.fake, self.script, self.path)

    def labels(self):
        return [row['label'] for row in self.runner.calls]

    def test_declined_confirmation_does_not_launch_or_stop(self):
        with patch.dict(os.environ, SUDO_USER=self.user.pw_name), patch.object(display, 'admissible'), patch.object(display, 'confirm', return_value=False):
            self.assertEqual(display.offer(self.fake, self.script), 4)
        self.assertEqual(self.runner.calls, [])

    def test_confirmation_launches_detached_job_only(self):
        with patch.dict(os.environ, SUDO_USER=self.user.pw_name), patch.object(display, 'admissible', return_value=[(4330, 3692)]), patch.object(display, 'confirm', return_value=True):
            self.assertEqual(display.offer(self.fake, self.script), 3)
        self.assertEqual(self.labels(), ['launch-service'])
        command = self.runner.calls[0]['argv']
        self.assertIn('--service-type=exec', command)
        self.assertIn('--setenv=SUDO_USER=' + self.user.pw_name, command)
        self.assertTrue(any(a.startswith('--property=ExecStopPost=') for a in command))
        self.assertNotIn('--scope', command); self.assertNotIn('--user', command)

    def test_fresh_success_order_and_post_gui_health_cursor(self):
        calls = []
        def attempt(argv, held_lock, publish):
            self.assertIn('display-stop', self.labels())
            self.assertNotIn('display-start', self.labels())
            calls.append(argv[argv.index('--method') + 1])
            self.assertIsNotNone(held_lock)
            self.assertFalse(publish)
            return 0
        self.fake.main = attempt
        self.fake.check_health = lambda *args, **kwargs: self.assertEqual(kwargs['cursor'], 'before-GUI-start')
        self.assertEqual(self.worker(), 0)
        self.assertEqual(calls, ['flr'])
        self.assertEqual(self.labels(), ['display-stop', 'display-start'])
        self.assertTrue(json.loads((self.output / 'report.json').read_text())['healthy'])

    def test_failed_flr_tries_bus_before_restart(self):
        methods = []
        def attempt(argv, held_lock, publish):
            methods.append(argv[argv.index('--method') + 1]); return 1 if len(methods) == 1 else 0
        self.fake.main = attempt
        self.assertEqual(self.worker(), 0)
        self.assertEqual(methods, ['flr', 'bus'])

    def test_refusal_does_not_try_bus_but_restores_display(self):
        methods = []
        self.fake.main = lambda argv, held_lock, publish: methods.append(argv) or 4
        self.assertEqual(self.worker(), 4)
        self.assertEqual(len(methods), 1)
        self.assertEqual(self.labels(), ['display-stop', 'display-start'])

    def test_service_stop_failure_never_resets_and_attempts_restore(self):
        self.runner.fail = 'display-stop'
        self.fake.main = lambda *args, **kwargs: self.fail('reset called after failed stop')
        self.assertEqual(self.worker(), 1)
        self.assertEqual(self.labels(), ['display-stop', 'display-start'])

    def test_new_gui_fault_invalidates_recovered_probe(self):
        def fail(*args, **kwargs): raise core.RecoveryError('new xe fault during GUI startup')
        self.fake.check_health = fail
        self.assertEqual(self.worker(), 1)
        self.assertFalse(json.loads((self.output / 'report.json').read_text())['healthy'])

    def test_stale_or_replayed_approval_does_not_touch_display(self):
        for field, value in [('worker_started', True), ('approved_monotonic', time.monotonic() - 61)]:
            with self.subTest(field=field):
                plan = dict(self.plan); plan[field] = value; display.save_plan(self.path, plan)
                self.assertEqual(self.worker(), 4)
                self.assertEqual(self.runner.calls, [])

    def test_changed_clients_reject_before_stop(self):
        with patch.object(display, 'admissible', side_effect=core.RecoveryRefused('owner changed')):
            self.assertEqual(display.worker(self.fake, self.script, self.path), 4)
        self.assertEqual(self.runner.calls, [])

    def test_stop_post_does_not_start_untouched_display(self):
        display.restore(self.fake, self.path, runner=self.runner)
        self.assertEqual(self.runner.calls, [])

    def test_stop_post_restores_once_after_attempted_stop(self):
        self.plan['display_stop_attempted'] = True; display.save_plan(self.path, self.plan)
        display.restore(self.fake, self.path, runner=self.runner)
        display.restore(self.fake, self.path, runner=self.runner)
        self.assertEqual(self.labels(), ['display-start'])

    def test_backup_never_starts_display_during_another_recovery(self):
        self.plan['display_stop_attempted'] = True; display.save_plan(self.path, self.plan)
        with core.acquire_recovery_lock(self.base):
            with self.assertRaises(BlockingIOError): display.restore(self.fake, self.path)
        self.assertEqual(self.runner.calls, [])

    def test_backup_never_starts_display_with_a_stalled_writer(self):
        self.plan['display_stop_attempted'] = True; display.save_plan(self.path, self.plan)
        raw = Path(f'/proc/{os.getpid()}/stat').read_text()
        (self.base / 'stalled-writer.json').write_text(json.dumps(
            dict(pid=os.getpid(), start_ticks=int(raw[raw.rfind(')') + 2:].split()[19]))))
        with self.assertRaises(core.RecoveryRefused): display.restore(self.fake, self.path)
        self.assertEqual(self.runner.calls, [])

    def test_stop_post_reports_interrupted_worker_without_health_claim(self):
        self.plan['display_stop_attempted'] = True; display.save_plan(self.path, self.plan)
        (self.output / 'report.json').write_text(json.dumps(dict(status='queued', healthy=False)))
        published = []
        self.fake.publish_summary = lambda report, *args: published.append(dict(report))
        display.restore_after_exit(self.fake, self.path)
        self.assertEqual(self.labels(), ['display-start'])
        self.assertEqual(published[0]['status'], 'interrupted')
        self.assertFalse(published[0]['healthy'])
        self.assertFalse(published[0]['recovery_completed'])

    def test_stop_post_preserves_completed_worker_receipt(self):
        self.plan['display_stop_attempted'] = True
        self.plan['display_restore_requested'] = True; display.save_plan(self.path, self.plan)
        receipt = dict(status='finished', healthy=True, finished_utc='recorded')
        (self.output / 'report.json').write_text(json.dumps(receipt))
        self.fake.publish_summary = lambda *args: self.fail('completed receipt replaced')
        display.restore_after_exit(self.fake, self.path)
        self.assertEqual(json.loads((self.output / 'report.json').read_text()), receipt)
        self.assertEqual(self.runner.calls, [])

    def test_confirmation_requires_literal_yes_and_a_terminal(self):
        for answer in ['\n', 'no\n', 'y\n', 'YES\n', 'yes\n']:
            with self.subTest(answer=answer):
                # Writes to a real terminal do not overwrite its input stream.
                terminal = io.StringIO(answer)
                terminal.write = lambda text: len(text)
                with patch('builtins.open', return_value=terminal):
                    self.assertEqual(display.confirm(), answer == 'yes\n')
        with patch('builtins.open', side_effect=OSError('no terminal')):
            self.assertFalse(display.confirm())

    def test_invalid_plan_and_permissions_never_stop(self):
        for field, value in [('approved', False), ('service', 'other.service'), ('bdf', '0000:06:00.0'), ('boot_id', 'old-boot')]:
            with self.subTest(field=field):
                plan = dict(self.plan); plan[field] = value; display.save_plan(self.path, plan)
                with self.assertRaises(RuntimeError): display.read_plan(self.path)
        display.save_plan(self.path, self.plan); self.path.chmod(0o644)
        with self.assertRaises(RuntimeError): display.read_plan(self.path)
        self.assertEqual(self.runner.calls, [])

    def test_service_paths_reject_command_parser_characters(self):
        with self.assertRaises(RuntimeError): display.unit_arguments(Path('/tmp/unsafe$path'), self.path, self.user.pw_name)

    def test_borrowed_lock_remains_held_through_inner_close(self):
        with core.acquire_recovery_lock(self.base) as outer:
            with core.acquire_recovery_lock(self.base, outer):
                with self.assertRaises(BlockingIOError): core.acquire_recovery_lock(self.base)
            with self.assertRaises(BlockingIOError): core.acquire_recovery_lock(self.base)
        with core.acquire_recovery_lock(self.base): pass

    def test_unrelated_lock_descriptor_is_rejected(self):
        with core.acquire_recovery_lock(self.base), (self.base / 'wrong.lock').open('a') as wrong:
            with self.assertRaises(core.RecoveryRefused): core.acquire_recovery_lock(self.base, wrong)

    def test_readiness_guards_reject_other_clients_and_display(self):
        description = dict(driver='xe', boot_vga='0', connectors={'DP': 'disconnected'}, survivability_mode='0')
        client = dict(pid=4330, start_ticks=3692, comm='Xorg', state='S')
        device = types.SimpleNamespace(describe=lambda: description, clients=lambda: ([client], []))
        fake = types.SimpleNamespace(Device=lambda bdf: device, RecoveryRefused=core.RecoveryRefused)
        with patch.object(display, 'independent_display', return_value=True), patch.object(display, 'owned_by_display_service', return_value=True):
            self.assertEqual(display.admissible(fake, self.runner), [(4330, 3692)])
            for field, value in [('comm', 'llama-server'), ('state', 'D')]:
                original = client[field]; client[field] = value
                with self.assertRaises(core.RecoveryRefused): display.admissible(fake, self.runner)
                client[field] = original
            description['connectors']['DP'] = 'connected'
            with self.assertRaises(core.RecoveryRefused): display.admissible(fake, self.runner)

    def test_readiness_rejects_hidden_clients_wrong_service_or_primary_gpu(self):
        description = dict(driver='xe', boot_vga='0', connectors={'DP': 'disconnected'}, survivability_mode='0')
        client = dict(pid=4330, start_ticks=3692, comm='Xorg', state='S')
        device = types.SimpleNamespace(describe=lambda: description, clients=lambda: ([client], []))
        fake = types.SimpleNamespace(Device=lambda bdf: device, RecoveryRefused=core.RecoveryRefused)
        with patch.object(display, 'independent_display', return_value=True), patch.object(display, 'owned_by_display_service', return_value=True):
            for field, value in [('boot_vga', '1'), ('driver', 'other'), ('survivability_mode', '1')]:
                original = description[field]; description[field] = value
                with self.assertRaises(core.RecoveryRefused): display.admissible(fake, self.runner)
                description[field] = original
            device.clients = lambda: ([client], [1234])
            with self.assertRaises(core.RecoveryRefused): display.admissible(fake, self.runner)
            device.clients = lambda: ([client], [])
            with patch.object(display, 'independent_display', return_value=False):
                with self.assertRaises(core.RecoveryRefused): display.admissible(fake, self.runner)
            with patch.object(display, 'owned_by_display_service', return_value=False):
                with self.assertRaises(core.RecoveryRefused): display.admissible(fake, self.runner)


if __name__ == '__main__':
    unittest.main()
