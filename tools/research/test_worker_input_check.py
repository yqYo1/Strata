"""Real mount-namespace controls; no GPU, network, Orca task or product write."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).with_name('worker_input_check.py').resolve()
STATE = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')


class WorkerFilesystemTest(unittest.TestCase):
    def run_case(self, change=None, writable=False):
        bwrap = shutil.which('bwrap')
        self.assertIsNotNone(bwrap, 'real bwrap namespace required;no skipped qualification')
        with tempfile.TemporaryDirectory(prefix='worker-fs-control-', dir=STATE) as tmp:
            root = Path(tmp); sources = root / 'sources'; report = root / 'report'
            sources.mkdir(); report.mkdir()
            rows = []
            for i in range(8):
                p = sources / str(i); p.write_text('source' + str(i))
                rows.append(dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
            if change: change(rows)
            manifest = sources / 'manifest.json'
            manifest.write_text(json.dumps(dict(report_directory=str(report), protected_inputs=rows)))
            manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
            command = [bwrap, '--die-with-parent', '--unshare-user', '--unshare-pid', '--unshare-net',
                '--ro-bind', '/', '/', '--proc', '/proc', '--dev', '/dev', '--tmpfs', '/tmp',
                '--bind', str(report), str(report)]
            if writable: command += ['--bind', str(sources), str(sources)]
            command += ['--chdir', str(report), sys.executable, '-B', str(SCRIPT), str(manifest), manifest_sha]
            result = subprocess.run(command, capture_output=True, text=True, timeout=20)
            proof = report / 'worker-filesystem-proof.json'
            self.assertTrue(proof.is_file(), result.stderr)
            data = json.loads(proof.read_text())
            # The nontruncating write-open check must not alter any source.
            for i in range(8): self.assertEqual((sources / str(i)).read_text(), 'source' + str(i))
            return result.returncode, data

    def test_all_inputs_on_readonly_mount(self):
        code, data = self.run_case()
        self.assertEqual(code, 0); self.assertTrue(data['passed'])
        self.assertEqual(data['checked_inputs'], 9); self.assertEqual(data['passed_inputs'], 9)
        self.assertTrue(data['no_dev_dri']); self.assertFalse(data['worker_side_orca_status_qualified'])

    def test_missing_input_fails_instead_of_counting_read_denial(self):
        code, data = self.run_case(lambda r: r[3].update(path=r[3]['path'] + '-missing'))
        self.assertEqual(code, 1); self.assertFalse(data['passed'])
        self.assertEqual(data['checked_inputs'], 9); self.assertEqual(data['passed_inputs'], 8)

    def test_wrong_source_hash_fails(self):
        code, data = self.run_case(lambda r: r[5].update(sha256='0' * 64))
        self.assertEqual(code, 1); self.assertFalse(data['passed'])
        self.assertEqual(data['passed_inputs'], 8)

    def test_writable_source_mount_fails_without_modifying_bytes(self):
        code, data = self.run_case(writable=True)
        self.assertEqual(code, 1); self.assertFalse(data['passed'])
        self.assertEqual(data['passed_inputs'], 0)
        self.assertTrue(all(not r['readonly_mount'] and r['write_open_errno'] is None for r in data['inputs']))


if __name__ == '__main__': unittest.main()
