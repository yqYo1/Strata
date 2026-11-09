"""CPU controls for the actual benchmark configuration/server launch boundary.

Main-authored; repository LICENSE applies. No native process or GPU launch.
"""
import copy
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import q6_head_generation_speed as cohort
from serve import server


class LaunchEnvironment(unittest.TestCase):
    def reference(self):
        return {'configuration': {'args': ['--adapt-swaps', '32',
            '--max-context', '131072', '--kv-resident', '131072',
            '--prefill', '8192', '--expert-cache', '1256'], 'parallel': 1}}

    def test_real_server_preserves_sycl_environment(self):
        reference = self.reference()
        original = copy.deepcopy(reference)
        config = cohort.configuration(reference)
        parent = {'PATH': '/usr/bin:/bin', 'ONEAPI_DEVICE_SELECTOR': 'level_zero:gpu',
                  'STRATA_SYCL_GDN_FP16': '1', 'EnableDirectSubmission': '0'}
        with patch.dict(os.environ, parent, clear=True):
            actual = server.child_env(config)
        self.assertEqual(actual, parent)
        self.assertEqual(config.get('backend'), 'sycl')
        self.assertEqual(reference, original)

    def test_missing_backend_reproduces_foreign_defaults(self):
        config = cohort.configuration(self.reference())
        config.pop('backend', None)
        parent = {'PATH': '/usr/bin:/bin'}
        with patch.dict(os.environ, parent, clear=True):
            actual = server.child_env(config)
        self.assertEqual(set(actual) - set(parent), set(server.GDN_STACK_CUDA))
        for key, value in server.GDN_STACK_CUDA.items():
            self.assertEqual(actual[key], value)


if __name__ == '__main__':
    unittest.main()
