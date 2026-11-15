"""CPU reporting on ARM/x86; no GPU, downloads or native instruction probes.

Independent tests for the Niko1221/Strata CPU-name slice through46c152a.
Repository LICENSE. Run: python -m unittest tools.test_setup_cpu_info
"""
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import setup


class CpuInfo(unittest.TestCase):
    def read_cpu(self, cpuinfo, lscpu='', processor=''):
        with mock.patch.object(setup, 'WIN', False), \
             mock.patch.object(setup.platform, 'processor', return_value=processor), \
             mock.patch('builtins.open', mock.mock_open(read_data=cpuinfo)), \
             mock.patch.object(setup, 'out', return_value=lscpu) as command:
            result = setup.cpu_info()
        return result, command

    def test_arm_heterogeneous_names_are_deduplicated_without_avx(self):
        result, command = self.read_cpu('processor : 0\nFeatures : fp asimd sve\n',
            '  Model name: Cortex-X925\nModel name: Cortex-A725\nModel name: Cortex-X925\n')
        self.assertEqual(result, ('Cortex-X925 + Cortex-A725', False, False))
        command.assert_called_once_with(['lscpu'])

    def test_missing_lscpu_name_uses_processor_then_plain_fallback(self):
        for processor, expected in [('AArch64 core', 'AArch64 core'), ('', 'arm64 CPU')]:
            with self.subTest(processor=processor):
                result, _ = self.read_cpu('Features : fp asimd\n', 'Architecture: aarch64\n', processor)
                self.assertEqual(result, (expected, False, False))

    def test_x86_keeps_the_complete_avx512_requirement(self):
        for flags, avx512 in [('avx2 avx512f', False),
                             ('avx2 avx512f avx512bw avx512vl avx512_vnni avx512vbmi', True)]:
            with self.subTest(flags=flags):
                result, command = self.read_cpu('model name : AMD Ryzen 5 5600X\nflags : ' + flags + '\n')
                self.assertEqual(result, ('AMD Ryzen 5 5600X', True, avx512))
                command.assert_not_called()

    def test_unreadable_cpuinfo_preserves_safe_processor_fallback(self):
        with mock.patch.object(setup, 'WIN', False), \
             mock.patch.object(setup.platform, 'processor', return_value='Known CPU'), \
             mock.patch('builtins.open', side_effect=OSError('unavailable')), \
             mock.patch.object(setup, 'out') as command:
            self.assertEqual(setup.cpu_info(), ('Known CPU', False, False))
            command.assert_not_called()


if __name__ == '__main__':
    unittest.main()
