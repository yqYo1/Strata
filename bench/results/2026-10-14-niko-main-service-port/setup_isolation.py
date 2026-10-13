#!/usr/bin/env python3
"""Replay two existing, mocked setup fixtures in a fresh import namespace."""
import importlib
import json
import sys
import unittest
from pathlib import Path

root, output = map(Path, sys.argv[1:])
sys.path[:0] = [str(root), str(root / 'tools')]
S = importlib.import_module('setup')
G = importlib.import_module('tools.test_setup_golden')
ram, cards = G.PROFILES['64GB-1x32GB']
code, _, actual, _ = G.install(ram, cards, G.argv_for('qwen', None))
expected = json.loads(G.GOLDEN.read_text())['64GB-1x32GB qwen recommended']['config']
differences = {k: {'actual': actual.get(k), 'expected': expected.get(k)}
               for k in sorted(actual.keys() | expected.keys()) if actual.get(k) != expected.get(k)}
suite = unittest.defaultTestLoader.loadTestsFromName('tools.test_setup_amd.WindowsDetection.test_prebuilt_hip_zip')
result = unittest.TextTestRunner(verbosity=0).run(suite)
record = {'root': str(root), 'WIN': S.WIN, 'EXE': S.EXE, 'golden_exit': code,
          'golden_differences': differences, 'hip_fixture_passed': result.wasSuccessful(),
          'hip_fixture_failures': [{'test': t.id(), 'detail': s} for t, s in result.failures],
          'GPU_model_installer_executed': False}
output.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({k: v for k, v in record.items() if k != 'hip_fixture_failures'}), flush=True)
