#!/usr/bin/env python3
"""Keep data mappings separate from executable provider identity checks."""
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/sycl'))
import runtime_probe as probe


def mapping(path, permissions='r-xp'):
    return '1000-2000 ' + permissions + ' 00000000 00:01 1 ' + path + '\n'


class ExecutableMappingTests(unittest.TestCase):
    def test_loader_cache_and_other_read_only_data_are_not_code(self):
        self.assertEqual(probe.executable_mappings(mapping('/etc/ld.so.cache', 'r--p') +
            mapping('/tmp/model.weights', 'r--s') + mapping('/tmp/data.so', 'rw-p')), set())

    def test_libraries_and_main_binary_are_checked_by_mapping_permissions(self):
        self.assertEqual(probe.executable_mappings(mapping('/usr/lib/libdevice.so.1.2') +
            mapping('/tmp/sycl_submission_test')), {'/usr/lib/libdevice.so.1.2', '/tmp/sycl_submission_test'})

    def test_executable_code_is_not_hidden_by_its_filename(self):
        self.assertEqual(probe.executable_mappings(mapping('/tmp/arbitrary-name')),
            {'/tmp/arbitrary-name'})

    def test_deleted_executable_mapping_is_refused(self):
        with self.assertRaises(RuntimeError):
            probe.executable_mappings(mapping('/tmp/provider.so (deleted)'))

    def test_kernel_and_anonymous_maps_are_not_file_providers(self):
        self.assertEqual(probe.executable_mappings(mapping('[vdso]') +
            '1000-2000 r-xp 00000000 00:00 0\n'), set())

    def test_duplicate_mapping_is_one_provider(self):
        self.assertEqual(probe.executable_mappings(mapping('/tmp/provider.so') * 2), {'/tmp/provider.so'})


if __name__ == '__main__':
    unittest.main()
