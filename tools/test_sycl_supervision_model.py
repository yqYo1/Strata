#!/usr/bin/env python3
"""CPU-only admission corruption controls; repository LICENSE."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import copy
import json

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent / 'sycl'))
import supervision_model as control


class PlacementInput(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='profile-admission-',
            dir=os.environ.get('STRATA_SUPERVISION_TEST_ROOT'))
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name); self.path = self.root / 'profile.bin'
        self.original = b'committed placement bytes'; self.path.write_bytes(self.original)
        self.config = dict(args=['--expert-profile', str(self.path)])
        self.addCleanup(patch.stopall)
        patch.object(control, 'P', self.root).start()
        patch.object(control.gen, 'git', return_value=self.original).start()

    def pin(self, expected=None):
        return control.profile_pin(self.config, 'source-revision', expected)

    def test_unchanged_committed_profile(self):
        pin = self.pin(); self.assertEqual(self.pin(pin), pin)
        self.assertEqual(pin['identity']['bytes'], len(self.original))

    def test_rewrite_with_restored_mtime_is_rejected(self):
        pin = self.pin(); old = self.path.stat()
        self.path.write_bytes(b'X' * len(self.original))
        os.utime(self.path, ns=(old.st_atime_ns, old.st_mtime_ns))
        with self.assertRaises(RuntimeError): self.pin(pin)

    def test_equal_byte_path_replacement_is_rejected(self):
        pin = self.pin(); other = self.root / 'other.bin'; other.write_bytes(self.original)
        self.config['args'][-1] = str(other)
        with self.assertRaises(RuntimeError): self.pin(pin)

    def test_missing_or_duplicate_profile_flag_is_rejected(self):
        for flags in ([], ['--expert-profile'], ['--expert-profile', str(self.path)] * 2):
            with self.subTest(flags=flags), self.assertRaises(RuntimeError):
                control.profile_pin(dict(args=flags), 'source-revision')

    def test_process_admission_stops_before_native_qualification(self):
        row = {key: {} for key in ('binary', 'qualification', 'model_assets', 'probe')}
        row.update(configuration=self.config, environment_template={}, template_sha256='template',
            runtime_sha256='runtime', NEO='provider', boot='boot', request_hashes=[], profile=self.pin(),
            benchmark_runtime={}, benchmark_runtime_host='host')
        row['plan'] = dict(source_commit='source-revision', qualified_input_identity=control.input_identity(row))
        self.path.write_bytes(b'changed profile')
        with patch.object(control.runtime.validation, 'committed', return_value=row['plan']), \
                patch.object(control.gen, 'committed', return_value=dict(environment_template={})), \
                patch.object(control, 'qualification') as native:
            with self.assertRaises(RuntimeError): control.current(row)
            native.assert_not_called()


class RuntimeInputs(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='runtime-admission-', dir=str(control.B))
        self.addCleanup(self.directory.cleanup); self.root = Path(self.directory.name)
        self.runtime = control.python_runtime

    def test_changed_runtime_file_is_rejected(self):
        source = self.root / 'runtime.py'; source.write_text('original')
        manifest = dict(inventory=self.runtime.inventory([str(self.root)], []), explicit=[])
        source.write_text('modified')
        with self.assertRaises(RuntimeError): self.runtime.validate_manifest(manifest)

    def test_new_runtime_search_input_is_rejected(self):
        manifest = dict(inventory=self.runtime.inventory([str(self.root)], []), explicit=[])
        (self.root / 'new_namespace').mkdir()
        with self.assertRaises(RuntimeError): self.runtime.validate_manifest(manifest)

    def test_live_interpreter_script_environment_and_maps(self):
        source = self.root / 'fixture.py'
        source.write_text("import sys\nprint('ready', flush=True)\nsys.stdin.readline()\n")
        env = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8',
            PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', PYTHONPYCACHEPREFIX=str(self.root / 'cache'))
        owner = None
        def persist():
            (self.root / 'execution.json').write_text(json.dumps(dict(active=bool(owner and owner.active),
                commands=owner.commands if owner else [])))
        with patch.object(self.runtime, 'P', self.root):
            chosen = self.runtime.selection(env)
            def interact(io, entry):
                self.assertEqual(io.line(3), 'ready')
                allowed = self.runtime.code_maps(entry['pid'])
                checked = self.runtime.check_running(entry['pid'], source, chosen, env, allowed)
                self.assertEqual(checked['sid'], entry['pid'])
                with self.assertRaises(RuntimeError):
                    self.runtime.check_running(entry['pid'], self.root / 'other.py', chosen, env, allowed)
                wrong = copy.deepcopy(chosen); wrong['file']['sha256'] = '0' * 64
                with self.assertRaises(RuntimeError): self.runtime.check_running(entry['pid'], source, wrong, env, allowed)
                with self.assertRaises(RuntimeError):
                    self.runtime.check_running(entry['pid'], source, chosen, dict(env, PYTHONNOUSERSITE='0'), allowed)
                with self.assertRaises(RuntimeError): self.runtime.check_running(entry['pid'], source, chosen, env, [])
                io.proc.stdin.close(); io.exit(3)
            owner = control.baseline.Owner(self.root / 'owned', persist)
            entry, _, _ = owner.run('python', [chosen['path'], str(source)], env, self.root, interaction=interact,
                wall=8, cpu=3, rss_cap=64 << 20, text_cap=1 << 20, file_cap=1 << 20, total_cap=4 << 20)
            self.assertTrue(control.baseline.closed(entry), entry.get('error'))
            self.assertEqual(entry['exit_code'], 0)


if __name__ == '__main__': unittest.main(failfast=True)
