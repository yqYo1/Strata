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
import build_q6_head_model as model_build
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


class PlannedCommands(unittest.TestCase):
    def test_root_compile_target(self):
        value = model_build.planned_compilation('icpx -o CMakeFiles/strata_engine.dir/a.o -c /repo/a.cpp')
        self.assertEqual(value, [dict(file='/repo/a.cpp', output='CMakeFiles/strata_engine.dir/a.o',
                                    target='strata_engine')])

    def test_nested_dependency_target(self):
        value = model_build.planned_compilation('cc -o ggml/src/CMakeFiles/ggml-base.dir/ggml.c.o -c /repo/ggml.c')
        self.assertEqual(value, [dict(file='/repo/ggml.c', output='ggml/src/CMakeFiles/ggml-base.dir/ggml.c.o',
                                    target='ggml-base')])

    def test_unowned_or_unrecognized_output_refused(self):
        for output in ('../escaped.o', '/tmp/escaped.o', 'other/a.o', 'CMakeFiles/fake/a.o'):
            with self.assertRaises(RuntimeError):
                model_build.planned_compilation('icpx -o ' + output + ' -c /repo/a.cpp')

    def test_duplicate_actual_source_refused(self):
        with self.assertRaisesRegex(RuntimeError, 'duplicate selected source compilation'):
            model_build.planned_compilation('icpx -o CMakeFiles/one.dir/a.o -c /repo/a.cpp\n'
                                           'icpx -o nested/CMakeFiles/two.dir/a.o -c /repo/a.cpp')


class SpirvOrder(unittest.TestCase):
    @staticmethod
    def module(offset=0, reverse=False, literal=4, wrap=False, rewire=False, body_swap=False, extra=None):
        def op(code, *a): return [((len(a) + 1) << 16) | code, *a]
        def ID(value): return value + offset
        def name(value):
            raw = value.encode() + b'\0'; raw += b'\0' * (-len(raw) % 4)
            return struct.unpack('<' + 'I' * (len(raw) // 4), raw)
        entries = [op(15, 6, ID(20), *name('kernel')), op(15, 6, ID(40), *name('other'))]
        types = [op(19, ID(1)), op(21, ID(3), 32, 0)]
        constants = [op(43, ID(3), ID(4), literal), op(43, ID(3), ID(5), 5)]
        body = [op(128, ID(3), ID(22), ID(4), ID(5 if rewire else 4)),
                op(130, ID(3), ID(23), ID(5), ID(4))]
        if body_swap: body.reverse()
        first = op(54, ID(1), ID(20), 0, ID(2)) + op(248, ID(21)) + sum(body, []) + op(253) + op(56)
        second = op(54, ID(1), ID(40), 0, ID(2)) + op(248, ID(41)) + op(253) + op(56)
        if reverse: entries.reverse(); types.reverse(); constants.reverse()
        words = [0x07230203, 0x10600, 0, 1000 + offset, 0]
        words += op(17, 6) + op(14, 2, 2) + sum(entries, []) + op(16, ID(20), 35, 1)
        if wrap: words += op(71, ID(22), 4469)
        words += sum(types, []) + op(33, ID(2), ID(1)) + sum(constants, [])
        if extra == 'alias': words += op(43, ID(3), ID(6), literal)
        if extra == 'unknown': words += op(65000)
        if extra == 'undefined': words += op(71, ID(999), 4469)
        words += second + first if reverse else first + second
        return struct.pack('<' + 'I' * len(words), *words)

    def test_ID_and_declaration_order_only(self):
        a = component.spirv_structure(self.module())
        self.assertEqual(a, component.spirv_structure(self.module(offset=50, reverse=True)))

    def test_literal_equal_to_an_ID_is_preserved(self):
        a = component.spirv_structure(self.module())
        self.assertNotEqual(a['sha256'], component.spirv_structure(self.module(offset=50, literal=54))['sha256'])

    def test_wrap_annotation_is_preserved(self):
        self.assertNotEqual(component.spirv_structure(self.module())['sha256'],
                            component.spirv_structure(self.module(wrap=True))['sha256'])

    def test_operand_edges_are_preserved(self):
        self.assertNotEqual(component.spirv_structure(self.module())['sha256'],
                            component.spirv_structure(self.module(rewire=True))['sha256'])

    def test_function_body_order_is_preserved(self):
        self.assertNotEqual(component.spirv_structure(self.module())['sha256'],
                            component.spirv_structure(self.module(body_swap=True))['sha256'])

    def test_unknown_instructions_and_undefined_IDs_refused(self):
        for extra in ('unknown', 'undefined'):
            with self.assertRaises(RuntimeError): component.spirv_structure(self.module(extra=extra))

    def test_ambiguous_global_IDs_refused(self):
        with self.assertRaisesRegex(RuntimeError, 'ambiguous SPIR-V'):
            component.spirv_structure(self.module(extra='alias'))

    def test_truncated_instruction_refused(self):
        blob = bytearray(self.module()); blob[-4:] = struct.pack('<I', (2 << 16) | 56)
        with self.assertRaisesRegex(RuntimeError, 'instruction extent'): component.spirv_structure(blob)


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
