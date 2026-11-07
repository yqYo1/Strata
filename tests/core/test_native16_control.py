"""Protect compact historical evidence when updating one calibration scope."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/sycl'))
import build_native16_control as control


class ScopedResult(unittest.TestCase):
    def test_insert_and_replace_preserve_unrelated_raw_tables(self):
        raw = ('{\n  "samples": [[1,2,3], [4,5,6]],\n'
               '  "event_reconciliation": {\n    "fused_command_profile": {\n'
               '      "existing": {"passed": false, "rows": [[7,8],[9,10]]}\n'
               '    }\n  }\n}\n')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); result = root / 'result.json'; result.write_text(raw)
            row = dict(stage='prepared', project=dict(root='source', revision='r', count=2, aggregate_sha256='a'),
                dependencies=[], external_inputs={'private': 'hash'})
            with patch.object(control, 'ROOT', root), patch.object(control, 'RESULT', result):
                control.save(row)
                first = result.read_text()
                self.assertIn('"samples": [[1,2,3], [4,5,6]]', first)
                self.assertIn('"existing": {"passed": false, "rows": [[7,8],[9,10]]}', first)
                row['stage'] = 'failed'
                control.save(row)
                second = result.read_text()
                self.assertEqual(first.replace('"stage": "prepared"', '"stage": "failed"'), second)
                self.assertEqual(control.scope(json.loads(second))[control.KEY], control.compact(row))
                self.assertNotIn('external_inputs', control.scope(json.loads(second))[control.KEY])

    def test_ambiguous_key_refuses_without_overwriting_evidence(self):
        raw = json.dumps(dict(event_reconciliation=dict(fused_command_profile={control.KEY: {}}),
            elsewhere={control.KEY: {}}))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); result = root / 'result.json'; result.write_text(raw)
            row = dict(project=dict(root='s', revision='r', count=1, aggregate_sha256='a'), dependencies=[])
            with patch.object(control, 'ROOT', root), patch.object(control, 'RESULT', result):
                with self.assertRaisesRegex(RuntimeError, 'ambiguous calibration result key'):
                    control.save(row)
            self.assertEqual(result.read_text(), raw)
            self.assertFalse((root / 'record.json').exists())


if __name__ == '__main__': unittest.main()
