"""Reject ambiguous or incomplete closed QSA attribution; repository LICENSE."""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'tools/sycl'))
import analyze_qsa_prefix_service as subject


def samples():
    result = []
    for phase, groups, calls in [('pack', x, 12) for x in [16384, 32768, 49152, 65536, 66432]] + [
            ('attention', 16384, 48), ('attention', 890, 12)]:
        suffix = '::{lambda(sycl::_V1::nd_item<1>)#1}'
        if phase == 'attention':
            suffix = '::{lambda(sycl::_V1::handler&)#1}::operator()(sycl::_V1::handler&) const' + suffix
        name = '"' + subject.FAMILY + '(fixture)' + suffix + (
            f'[SIMD{32 if phase == "pack" else 16} {{{groups}; 1; 1}} '
            f'{{{64 if phase == "pack" else 128}; 1; 1}}]"')
        result.append(dict(name=name, calls=calls, total_ns=200*calls, average_ns=200, min_ns=100, max_ns=300))
    return result


class AttributionTests(unittest.TestCase):
    def test_inventory(self):
        result = subject.extract(samples(), 33213)
        self.assertEqual(result['calls_per_phase'], 60)
        self.assertEqual(result['packed_logical_rows'], 2763264)
        self.assertEqual(result['totals_ns'], dict(pack=12000, attention=12000))

    def test_missing_duplicate_and_call_tamper(self):
        for kind in ['missing', 'duplicate', 'count']:
            rows = samples()
            if kind == 'missing': rows.pop()
            elif kind == 'duplicate': rows.append(rows[0].copy())
            else: rows[0]['calls'] = 13
            with self.subTest(kind=kind), self.assertRaises(RuntimeError): subject.extract(rows, 33213)

    def test_wrong_geometry_and_ambiguous_kernel(self):
        for before, after in [('SIMD32', 'SIMD16'), ('{64; 1; 1}', '{128; 1; 1}'),
                              ('nd_item<1>)#1}', 'nd_item<1>)#2}'), ('16384', '16385')]:
            rows = samples(); rows[0]['name'] = rows[0]['name'].replace(before, after)
            with self.subTest(before=before), self.assertRaises(RuntimeError): subject.extract(rows, 33213)

    def test_nonpositive_and_inconsistent_statistics(self):
        for key, value in [('min_ns', 0), ('total_ns', 0), ('average_ns', 2010), ('calls', True)]:
            rows = samples(); rows[0][key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError): subject.extract(rows, 33213)


if __name__ == '__main__':
    unittest.main()
