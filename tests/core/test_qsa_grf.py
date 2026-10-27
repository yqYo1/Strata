"""Main-authored negative controls for compile-only metadata admission."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/sycl'))
from compile_qsa_grf import metadata


class MetadataAdmission(unittest.TestCase):
    def parse(self, request='128', subgroup='16', kernels=1, expected=128):
        lines = [f'define spir_kernel void @kernel{i}() #0 !intel_reqd_sub_group_size !2 {{'
                 for i in range(kernels)]
        attr = '' if request is None else f'"sycl-grf-size"="{request}"'
        lines += ['attributes #0 = { ' + attr + ' }', '!2 = !{i32 ' + subgroup + '}']
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.ll'
            path.write_text('\n'.join(lines) + '\n')
            return metadata(path, expected)['metadata_pattern_confirmed']

    def test_matching_request_and_resolved_subgroup(self):
        self.assertTrue(self.parse())
        self.assertTrue(self.parse('256', expected=256))

    def test_wrong_or_missing_request(self):
        self.assertFalse(self.parse('256'))
        self.assertFalse(self.parse(None))

    def test_wrong_or_missing_subgroup_value(self):
        self.assertFalse(self.parse(subgroup='8'))
        self.assertFalse(self.parse(subgroup='undef'))

    def test_request_must_belong_to_exactly_one_kernel(self):
        self.assertFalse(self.parse(kernels=0))
        self.assertFalse(self.parse(kernels=2))

    def test_original_must_not_request_grf(self):
        self.assertTrue(self.parse(None, expected=None))
        self.assertFalse(self.parse(expected=None))


if __name__ == '__main__':
    unittest.main()
