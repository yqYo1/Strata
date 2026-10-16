"""Host admission regression: frozen bare/full pins agree only in serialization."""
import importlib.util
from pathlib import Path
import unittest

p = Path(__file__).resolve().parent / 'own-metric-query-tiny-v4/main_controller.py'
spec = importlib.util.spec_from_file_location('metric_controller_pin_test', p)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class PinIdentity(unittest.TestCase):
    def test_full_and_bare_same_identity(self):
        full = dict(path='/example', realpath='/actual', bytes=7, sha256='a'*64)
        bare = {k: v for k, v in full.items() if k != 'path'}
        self.assertTrue(m.same_pin(full, full, '/example'))
        self.assertTrue(m.same_pin(bare, full, '/example'))
        self.assertEqual(set(bare), {'realpath', 'bytes', 'sha256'})

    def test_reject_every_changed_identity(self):
        full = dict(path='/example', realpath='/actual', bytes=7, sha256='a'*64)
        for key, value in dict(path='/different', realpath='/different', bytes=8, sha256='b'*64).items():
            with self.subTest(key=key):
                self.assertFalse(m.same_pin({**full, key:value}, full, '/example'))
                self.assertFalse(m.same_pin(full, {**full, key:value}, '/example'))

    def test_unknown_missing_keys_and_wrong_map_key_hold(self):
        full = dict(path='/example', realpath='/actual', bytes=7, sha256='a'*64)
        self.assertFalse(m.same_pin({**full, 'unreviewed':True}, full, '/example'))
        self.assertFalse(m.same_pin(full, {**full, 'unreviewed':True}, '/example'))
        for key in ('realpath', 'bytes', 'sha256'):
            self.assertFalse(m.same_pin({k:v for k,v in full.items() if k != key}, full, '/example'))
        self.assertFalse(m.same_pin(full, full, '/different'))


if __name__ == '__main__':
    unittest.main()
