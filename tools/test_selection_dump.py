"""Host-only checked-readback qualification. No GPU/runtime initialization."""
import hashlib
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest

P = Path(__file__).resolve().parents[1]


class SelectionDumpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory(prefix='strata-selection-host-')
        cls.root = Path(cls.folder.name)
        cls.binary = cls.root / 'check'
        compiler = Path(shutil.which('c++')).resolve()
        print('Host compiler:', compiler, hashlib.sha256(compiler.read_bytes()).hexdigest(), flush=True)
        subprocess.run([str(compiler), '-std=c++20', '-O0', '-Wall', '-Wextra', '-Werror', '-pedantic',
            '-I' + str(P / 'include'), str(P / 'tests/core/test_selection_dump.cpp'),
            '-o', str(cls.binary)], check=True, timeout=45)

    @classmethod
    def tearDownClass(cls): cls.folder.cleanup()

    def call(self, mode, path=None, expected=0):
        output = path or self.root / (mode + '.bin')
        result = subprocess.run([str(self.binary), mode, str(output)], capture_output=True, timeout=5)
        self.assertEqual(result.returncode, expected, result.stderr)
        return output

    def test_success(self):
        output = self.call('success')
        self.call('success', output)  # separate calls append whole records
        record = struct.pack('=8i', 0, 32, 2, 2, 1, 2, 3, 4)
        self.assertEqual(output.read_bytes(), record * 2)

    def test_submission_failure_drains(self):
        self.assertFalse(self.call('submit-fail').exists())

    def test_exception_drains(self):
        self.assertFalse(self.call('throw').exists())

    def test_invalid_geometry_never_submits(self):
        self.assertFalse(self.call('geometry').exists())

    def test_uncertain_completion_never_returns(self):
        self.assertFalse(self.call('drain-fail', expected=73).exists())

    def test_open_error(self): self.call('open-fail', self.root / 'absent' / 'out')

    @unittest.skipUnless(Path('/dev/full').exists(), 'requires Linux full device')
    def test_write_or_close_error(self): self.call('file-fail', Path('/dev/full'))


if __name__ == '__main__': unittest.main()
