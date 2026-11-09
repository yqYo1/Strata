"""CPU regressions for actual selected targets and shared SPIR-V images.

Main-authored; repository LICENSE. No native executable or GPU access.
"""
from pathlib import Path
import struct
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_compat_profile as build
import q6_dpas_component as component
import tools.sycl.test_q6_dpas_component as fixtures


class SelectedCommands(unittest.TestCase):
    @staticmethod
    def command(source, target):
        return dict(file=source, command=f'icpx -O3 -o CMakeFiles/{target}.dir/unit.o -c {source}')

    def test_same_file_in_unbuilt_production_is_excluded(self):
        actual = self.command('/repo/kernel.cpp', 'measured')
        runtime = self.command('/repo/runtime.cpp', 'strata_sycl_runtime')
        unbuilt = self.command('/repo/kernel.cpp', 'strata_kernels')
        selected = build.selected_compile_commands([unbuilt, actual, runtime],
            {'/repo/kernel.cpp', '/repo/runtime.cpp'}, 'measured')
        self.assertEqual(selected, [actual, runtime])

    def test_duplicate_within_actual_target_is_refused(self):
        actual = self.command('/repo/kernel.cpp', 'measured')
        with self.assertRaisesRegex(RuntimeError, 'one production compile command'):
            build.selected_compile_commands([actual, dict(actual)], {'/repo/kernel.cpp'}, 'measured')

    def test_missing_actual_unit_is_refused(self):
        actual = self.command('/repo/kernel.cpp', 'measured')
        with self.assertRaisesRegex(RuntimeError, 'one production compile command'):
            build.selected_compile_commands([actual], {'/repo/kernel.cpp', '/repo/missing.cpp'}, 'measured')

    def test_similar_target_name_does_not_match(self):
        other = self.command('/repo/kernel.cpp', 'measured_extra')
        with self.assertRaisesRegex(RuntimeError, 'one production compile command'):
            build.selected_compile_commands([other], {'/repo/kernel.cpp'}, 'measured')


class SharedImage(unittest.TestCase):
    @staticmethod
    def image(*, unrelated_only=False, duplicate_name=False):
        blob = fixtures.LinkedSpirv.image(calls=0 if unrelated_only else 2)
        words = list(struct.unpack('<' + 'I' * (len(blob) // 4), blob))
        at = 5
        while words[at] & 65535 != 54:
            at += words[at] >> 16
        name = component.KERNEL_ENTRY if duplicate_name else 'unrelated_kernel'
        raw = name.encode() + b'\0'; raw += b'\0' * (-len(raw) % 4)
        text = list(struct.unpack('<' + 'I' * (len(raw) // 4), raw))
        entry = [((3 + len(text)) << 16) | 15, 6, 8, *text]
        function = [(5 << 16) | 54, 1, 8, 0, 2]
        if unrelated_only:
            for i in range(2): function += [(4 << 16) | 57, 1, 40 + i, 9]
        function += [(1 << 16) | 56]
        words[at:at] = entry; words += function
        return struct.pack('<' + 'I' * len(words), *words)

    def test_exact_entry_callgraph_in_shared_image(self):
        result = component.embedded_dpas(fixtures.LinkedSpirv.elf([self.image()]), allow_shared_image=True)
        self.assertEqual(result['entrypoint'], component.KERNEL_ENTRY)
        self.assertEqual(result['dpas_static_calls'], 2)

    def test_original_single_entry_contract_stays_strict(self):
        with self.assertRaisesRegex(RuntimeError, 'one exact C4 kernel entrypoint'):
            component.embedded_dpas(fixtures.LinkedSpirv.elf([self.image()]))

    def test_unrelated_kernel_dpas_does_not_qualify_target(self):
        with self.assertRaisesRegex(RuntimeError, 'at least two reachable'):
            component.embedded_dpas(fixtures.LinkedSpirv.elf([self.image(unrelated_only=True)]), allow_shared_image=True)

    def test_duplicate_named_entry_is_refused(self):
        with self.assertRaises(RuntimeError):
            component.embedded_dpas(fixtures.LinkedSpirv.elf([self.image(duplicate_name=True)]), allow_shared_image=True)


if __name__ == '__main__':
    unittest.main()
