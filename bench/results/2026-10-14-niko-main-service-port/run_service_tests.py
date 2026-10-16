#!/usr/bin/env python3
"""No-model service tests with individual results, not an inference benchmark."""
import json
import sys
import time
import unittest
from pathlib import Path

ROOT = Path('/home/yayoi/orca/workspaces/Strata/sycl')
OUTPUT = Path(sys.argv[1])
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]

class Result(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.samples = []
        self.started = {}

    def startTest(self, test):
        self.started[test.id()] = time.monotonic()
        super().startTest(test)

    def record(self, test, status, detail=None):
        self.samples.append({'test': test.id(), 'status': status,
            'seconds': time.monotonic() - self.started.get(test.id(), time.monotonic()), 'detail': detail})
        if len(self.samples) % 50 == 0:
            print(json.dumps({'completed_cases': len(self.samples)}), flush=True)

    def addSuccess(self, test):
        super().addSuccess(test)
        self.record(test, 'passed')

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.record(test, 'failed', self._exc_info_to_string(err, test))

    def addError(self, test, err):
        super().addError(test, err)
        self.record(test, 'error', self._exc_info_to_string(err, test))

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.record(test, 'skipped', reason)

    def addExpectedFailure(self, test, err):
        super().addExpectedFailure(test, err)
        self.record(test, 'expected_failure', self._exc_info_to_string(err, test))

    def addUnexpectedSuccess(self, test):
        super().addUnexpectedSuccess(test)
        self.record(test, 'unexpected_success')

suite = unittest.defaultTestLoader.discover(str(ROOT / 'serve'), 'test_*.py', top_level_dir=str(ROOT))
runner = unittest.TextTestRunner(verbosity=0, resultclass=Result)
result = runner.run(suite)
OUTPUT.write_text(json.dumps({'passed': result.wasSuccessful(), 'tests_run': result.testsRun,
    'failures': len(result.failures), 'errors': len(result.errors), 'skips': len(result.skipped),
    'model_GPU_executed': False, 'engine_BGEN_VRAM_qualified': False, 'samples': result.samples}, indent=2) + '\n')
print(json.dumps({'passed': result.wasSuccessful(), 'tests': result.testsRun, 'skips': len(result.skipped)}), flush=True)
sys.exit(0 if result.wasSuccessful() else 1)
