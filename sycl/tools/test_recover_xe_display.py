"""Exercise GUI recovery approval/guards/order with harmless fake backends."""
import errno
import fcntl
import io
import json
import os
from pathlib import Path
import pwd
import select
import subprocess
import tempfile
import time
import termios
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
        self.umask = os.umask(0o022)
        self.output = self.base / 'display-test'; self.output.mkdir(mode=0o700)
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

    def test_cli_cannot_offer_or_launch_any_gui_shutdown(self):
        with patch.object(display.sys, 'argv', ['python3', str(self.script)]), patch.object(display.os, 'geteuid', return_value=0), patch.object(display, 'secure_base'), patch.object(display, 'load_core', return_value=self.fake), patch.object(display, 'offer', side_effect=AssertionError('GUI offered')), patch.object(display, 'worker', side_effect=AssertionError('GUI worker called')):
            self.assertEqual(display.main([]), 4)
            self.assertEqual(display.main(['--worker', str(self.path)]), 4)
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

    def test_plan_replacements_are_private_under_manager_umask(self):
        os.umask(0o022)
        for started in [False, True, False]:
            self.plan['worker_started'] = started
            display.save_plan(self.path, self.plan)
            self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(display.read_plan(self.path)['worker_started'], started)
        self.assertEqual(list(self.output.glob('.plan-*')), [])

    def test_legacy_repair_requires_private_directory_and_unwritable_file(self):
        self.path.chmod(0o644)
        self.assertEqual(display.read_plan(self.path, repair_legacy=True), json.loads(self.path.read_text()))
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)
        for mode in [0o664, 0o666, 0o640]:
            self.path.chmod(mode)
            with self.assertRaises(RuntimeError): display.read_plan(self.path, repair_legacy=True)
        self.path.chmod(0o644); self.output.chmod(0o755)
        with self.assertRaises(RuntimeError): display.read_plan(self.path, repair_legacy=True)
        self.output.chmod(0o700)
        link = self.output / 'hardlink'; os.link(self.path, link)
        with self.assertRaises(RuntimeError): display.read_plan(self.path, repair_legacy=True)
        link.unlink()
        self.path.unlink(); self.path.symlink_to(self.output / 'missing')
        with self.assertRaises(RuntimeError): display.read_plan(self.path, repair_legacy=True)

    def mark_interrupted(self):
        self.plan.update(worker_started=True, display_stop_attempted=True)
        display.save_plan(self.path, self.plan)
        self.path.chmod(0o644)  # Actual previous manager-worker mode.

    def resume(self):
        with patch.dict(os.environ, SUDO_USER=self.user.pw_name):
            return display.resume(self.fake, self.script)

    def test_resume_legacy_interruption_restores_and_checks_without_reset(self):
        self.mark_interrupted()
        self.fake.main = lambda *args, **kwargs: self.fail('resume performed a reset')
        self.fake.check_health = lambda *args, **kwargs: self.assertEqual(kwargs['cursor'], 'before-GUI-start')
        published = []; self.fake.publish_summary = lambda report, *args: published.append(dict(report))
        self.assertEqual(self.resume(), 0)
        self.assertEqual(self.labels(), ['display-start'])
        self.assertTrue(published[0]['healthy'])
        self.assertFalse(published[0]['recovery_completed'])
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.resume(), 3)
        self.assertEqual(self.labels(), ['display-start'])

    def test_resume_pending_writer_or_live_worker_never_starts_display(self):
        self.mark_interrupted()
        with core.acquire_recovery_lock(self.base):
            with self.assertRaises(BlockingIOError): self.resume()
        raw = Path(f'/proc/{os.getpid()}/stat').read_text()
        (self.base / 'stalled-writer.json').write_text(json.dumps(
            dict(pid=os.getpid(), start_ticks=int(raw[raw.rfind(')') + 2:].split()[19]))))
        self.assertEqual(self.resume(), 4)
        self.assertEqual(self.runner.calls, [])
        self.assertFalse(json.loads((self.output / 'report.json').read_text())['healthy'])

    def test_resume_restore_or_health_failure_is_not_success(self):
        for failure in ['display-start', 'health']:
            with self.subTest(failure=failure):
                self.mark_interrupted(); self.runner.calls.clear(); self.runner.fail = failure
                def probe(*args, **kwargs):
                    if failure == 'health': raise core.RecoveryError('new GPU fault')
                self.fake.check_health = probe
                self.assertEqual(self.resume(), 1)
                self.assertFalse(json.loads((self.output / 'report.json').read_text())['healthy'])
                self.assertEqual(self.labels(), ['display-start'])

    def test_resume_skips_old_boot_and_refuses_other_caller(self):
        self.mark_interrupted()
        self.plan['boot_id'] = 'old-boot'; display.save_plan(self.path, self.plan)
        self.assertEqual(self.resume(), 3)
        self.plan['boot_id'] = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        display.save_plan(self.path, self.plan)
        with patch.dict(os.environ, SUDO_USER='another-caller'):
            with self.assertRaises(core.RecoveryRefused): display.resume(self.fake, self.script)
        self.assertEqual(self.runner.calls, [])

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
                with patch('builtins.open', side_effect=[io.StringIO(answer), io.StringIO()]):
                    self.assertEqual(display.confirm(), answer == 'yes\n')
        with patch('builtins.open', side_effect=OSError('no terminal')):
            self.assertFalse(display.confirm())

    def test_real_controlling_terminal_confirmation_and_no_terminal(self):
        # Execute only the actual confirm() definition, never main/offer/worker.
        script = ("from pathlib import Path; import sys,types; "
                  "source=Path(sys.argv[1]).read_text().split(\"<<'PY'\\n\",1)[1].rsplit('\\nPY',1)[0]; "
                  "module=types.ModuleType('tty_confirmation_test'); exec(compile(source,sys.argv[1],'exec'),module.__dict__); "
                  "print('CONFIRM_RESULT='+str(module.confirm()),flush=True)")
        command = ['/usr/bin/python3', '-c', script, str(Path(__file__).with_name('recover-xe-display.sh'))]
        prompt = 'yes と入力してください'.encode()
        for answer in [b'yes\n', b'no\n', b'\n']:
            with self.subTest(answer=answer):
                master, slave = os.openpty()
                def controlling_terminal():
                    os.setsid(); fcntl.ioctl(0, termios.TIOCSCTTY, 0)
                child = subprocess.Popen(command, stdin=slave, stdout=slave, stderr=slave,
                                         preexec_fn=controlling_terminal)
                os.close(slave)
                text, supplied = b'', False
                deadline = time.monotonic() + 5
                try:
                    while time.monotonic() < deadline:
                        ready, _, _ = select.select([master], [], [], 0.05)
                        if ready:
                            try:
                                part = os.read(master, 4096)
                            except OSError as error:
                                if error.errno == errno.EIO: break
                                raise
                            if not part: break
                            text += part
                        if not supplied and prompt in text:
                            os.write(master, answer); supplied = True
                    self.assertTrue(supplied, text.decode(errors='replace'))
                    self.assertEqual(child.wait(timeout=1), 0)
                    result = b'True' if answer == b'yes\n' else b'False'
                    self.assertIn(b'CONFIRM_RESULT=' + result, text)
                finally:
                    os.close(master)
                    if child.poll() is None: child.kill(); child.wait(timeout=1)
        detached = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True,
                                  start_new_session=True, timeout=5)
        self.assertEqual(detached.returncode, 0)
        self.assertIn(b'CONFIRM_RESULT=False', detached.stdout)

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
