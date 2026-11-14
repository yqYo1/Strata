"""Host-only regression checks for the diagnostic's prelaunch/input contracts."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[2] / 'tools/sycl'
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location('fp64_probe_controls', TOOLS / 'device_fp64_probe.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class ToolchainImage(unittest.TestCase):
    def test_new_member_and_package_changes_are_observed(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            sdk = root / 'sdk'; sdk.mkdir()
            (sdk / 'compiler').write_text('compiler-v1')
            package = root / 'host-library'; package.write_text('library-v1')
            first = probe.file_image([sdk], [package])
            (sdk / 'subtool').write_text('new subtool')
            self.assertNotEqual(first, probe.file_image([sdk], [package]))
            second = probe.file_image([sdk], [package])
            package.write_text('library-v2')
            self.assertNotEqual(second, probe.file_image([sdk], [package]))
            package.unlink()
            self.assertEqual(probe.file_image([sdk], [package])['files'][str(package)]['kind'], 'missing')

    def test_symlink_retarget_is_observed_even_with_equal_bytes(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            for n in ('one', 'two'): (root / n).write_text('same bytes')
            link = root / 'tool'; link.symlink_to(root / 'one')
            first = probe.file_image([], [link])
            link.unlink(); link.symlink_to(root / 'two')
            self.assertNotEqual(first, probe.file_image([], [link]))

    def test_actual_unpinned_include_or_link_is_rejected(self):
        with tempfile.TemporaryDirectory() as name, patch.object(probe, 'ROOT', Path(name)):
            root = Path(name)
            cpp, header = root / 'probe.cpp', root / 'include.hpp'
            libc, cxx = root / 'libc.so', root / 'libstdc++.so'
            for f in (cpp, header, libc, cxx): f.write_text(f.name)
            row = dict(toolchain=probe.file_image([], [header, libc, cxx]), runtime_pins={},
                executable=str(root / 'probe'), sources={probe.SOURCE: probe.sha(cpp)})
            dependencies = root / 'headers.d'
            dependencies.write_text(f'{row["executable"]}: {cpp} {header}\n')
            trace = f'{libc}\n{cxx}\n'
            self.assertEqual(probe.build_inputs(row, trace)['host_link_count'], 2)
            foreign = root / 'foreign'; foreign.write_text('not pinned')
            with self.assertRaisesRegex(RuntimeError, 'unqualified compiler'):
                probe.build_inputs(row, trace + str(foreign) + '\n')
            dependencies.write_text(f'{row["executable"]}: {cpp} {foreign}\n')
            with self.assertRaisesRegex(RuntimeError, 'unqualified compiler'):
                probe.build_inputs(row, trace)


class PrelaunchReceipt(unittest.TestCase):
    def check_failure(self, cursor_fails):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); result = root / 'result.json'; result.write_text('{}\n')
            row = dict(stage='prepared', active=False, ownership_closed=True, complete=False,
                passed=False, results=[], command=['must-not-run'], environment={}, boot='fixture',
                driver={}, executable=str(root / 'probe'))
            initial = copy.deepcopy(row)
            cursor = subprocess.CalledProcessError(1, 'journalctl', stderr='cursor fixture') if cursor_fails else None
            with patch.object(probe, 'ROOT', root), patch.object(probe, 'RESULT', result), \
                    patch.object(probe, 'current', return_value=row), \
                    patch.object(probe.subprocess, 'check_output', **(
                        {'side_effect': cursor} if cursor_fails else {'return_value': '-- cursor: fixture\n'})), \
                    patch.object(probe.owner, 'Owner', side_effect=OSError('Owner fixture')) as owner:
                with self.assertRaises(RuntimeError): probe.run('build')
                owner.assert_not_called() if cursor_fails else owner.assert_called_once()
                receipt = json.loads((root / 'build-execution.json').read_text())
                saved = json.loads((root / 'record.json').read_text())
                self.assertEqual(saved['stage'], 'failed')
                self.assertTrue(receipt['prelaunch_failure'])
                self.assertFalse(receipt['native_started'])
                self.assertFalse(receipt['passed'])
                self.assertIsNone(receipt['fresh_faults'])
                self.assertTrue(receipt['ownership_closed'])
                # Even if a caller accidentally presents the old prepared row,
                # its existing receipt prevents a replacement attempt.
                row.clear(); row.update(initial)
                with self.assertRaisesRegex(RuntimeError, 'existing operation receipt'):
                    probe.run('build')

    def test_cursor_failure_has_terminal_receipt(self): self.check_failure(True)

    def test_owner_setup_failure_has_terminal_receipt(self): self.check_failure(False)


if __name__ == '__main__':
    unittest.main()
