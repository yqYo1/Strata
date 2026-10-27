"""Reject incomplete/misclassified selected-QSA aggregate diagnostics; no GPU."""
import sys
from pathlib import Path
import unittest
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/sycl'))
from qsa_service import summarize, verify_native_environment

PREFIX = 'strata::kernels::prefix_batch_impl<33, false, false, true, true, true, true, false>'


def table():
    names = [f'{PREFIX}::attention[SIMD16 {{16384; 1; 1}} {{128; 1; 1}}]',
        f'{PREFIX}::attention[SIMD16 {{12982; 1; 1}} {{128; 1; 1}}]',
        f'{PREFIX}::pack[SIMD32 {{8192; 1; 1}} {{64; 1; 1}}]']
    counts = (48, 12, 60)
    rows = [f'{n}, {c}, {c * 10}, 0.0, 10, 10, 10' for n, c in zip(names, counts)]
    return ('=== Device Timing Summary ===\nTotal Device Time for L0 backend (ns): 1200\n'
        'Kernel, Calls, Time (ns), Time (%), Average (ns), Min (ns), Max (ns)\n' +
        '\n'.join(rows) + '\n=== Kernel Properties ===\n')


class QsaServiceTests(unittest.TestCase):
    def test_preload_alias_is_the_same_qualified_object(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); collector = root/'collector.so'; probe = root/'probe.so'
            collector.touch(); probe.touch(); alias = root/'launcher-collector.so'; alias.symlink_to(collector)
            expected = {'EnableDirectSubmission': '0', 'LD_PRELOAD': str(probe)}
            actual = {b'EnableDirectSubmission': b'0', b'LD_PRELOAD': f'{alias}:{probe}'.encode()}
            verify_native_environment(actual, expected, collector, probe)
            for variant in (actual | {b'EnableDirectSubmission': b'1'},
                    actual | {b'LD_PRELOAD': f'{probe}:{alias}'.encode()},
                    actual | {b'LD_PRELOAD': f'{alias}:{probe}:'.encode()},
                    actual | {b'LD_PRELOAD': f'{alias}:{probe}:/unexpected.so'.encode()},
                    actual | {b'UNITRACE_KernelMetrics': b'1'}):
                with self.subTest(variant=variant), self.assertRaises(RuntimeError):
                    verify_native_environment(variant, expected, collector, probe)

    def test_independent_query_and_call_conservation(self):
        result = summarize(table(), 39260)
        self.assertEqual(result['attention']['calls'], 60)
        self.assertEqual(result['attention']['service_ns'], 600)
        self.assertEqual(result['pack']['service_ns'], 600)

    def test_refuse_missing_extra_wrong_route_shape_and_clock(self):
        original = table()
        corruptions = [original.replace('48, 480', '47, 480'),
            original.replace('false, false, true', 'true, false, true'),
            original.replace('{12982; 1; 1}', '{12984; 1; 1}'),
            original.replace('SIMD16 {16384', 'SIMD32 {16384'),
            original.replace('{64; 1; 1}', '{32; 1; 1}'),
            original.replace('1200', '1201'),
            original.replace('60, 600, 0.0, 10, 10, 10', '60, 600, 0.0, 10, 0, 10'),
            original.replace('=== Kernel Properties ===', ''), original + original]
        for text in corruptions:
            with self.subTest(text=text), self.assertRaises((RuntimeError, ValueError)):
                summarize(text, 39260)

    def test_refuse_other_workload(self):
        with self.assertRaises(RuntimeError): summarize(table(), 33214)


if __name__ == '__main__': unittest.main()
