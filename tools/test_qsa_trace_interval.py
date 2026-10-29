"""Behavior tests of the prospective checker, without GPU or task claims."""
import copy
from fractions import Fraction as F
import struct
import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parent/'sycl'))
import qsa_trace_interval as c
q = c.q


def bits(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def records():
    p = [bits(1.)] + [bits(2.**-25)]*31
    total = sum(map(q.exact_fp32, p), F(0))
    serial = F(0)
    for v in p:
        serial = q.rn32(serial + q.exact_fp32(v))
    a = 0x3f808001  # Opposite sides of a BF16 midpoint after normalization.
    result = []
    for arm, norm in [('original', serial), ('candidate', q.rn32(total))]:
        y = bits(float(q.rn32(q.exact_fp32(a)/norm)))
        result.append(dict(call=0, arm=arm, cap=32, width=32, tiles=1, queries=1,
            query=0, kv_head=0, head=0, live_cells=32, page=4,
            normal_parity=True, guards=True,
            state=[0, 0, bits(float(norm)), bits(float(norm))],
            details=p+[c.bf16_rne(v) for v in p]+[a]*256+[y]*256,
            output=[c.bf16_rne(y)]*256))
    return result


def check(a, b, **kwargs):
    return c.pair(a, b, arithmetic_mode='RNE_gradual', division_ulps=F(5,2),
                  bf16_mode='RNE', **kwargs)


class TraceTests(unittest.TestCase):
    def test_reassociation_and_bf16_midpoint_change_are_derived_not_rejected(self):
        a,b = records()
        result = check(a,b)
        self.assertEqual(result['changed_sums'], 1)
        self.assertEqual(result['changed_norms'], 1)
        self.assertEqual(result['changed_output_bits'], 256)
        self.assertEqual(result['checked_ratios'], 512)

    def test_unchanged_operand_and_accumulator_are_independent_invariants(self):
        for at in (1, 33, 65):
            a,b = records();b['details'][at] ^= 1
            with self.assertRaises(ValueError):
                check(a,b)

    def test_wrong_serial_sum_and_wrong_fma_are_refused(self):
        a,b = records();a['state'][3] += 1
        with self.assertRaisesRegex(ValueError, 'ascending'):
            check(a,b)
        a,b = records();b['state'][2] += 1
        with self.assertRaisesRegex(ValueError, 'FMA'):
            check(a,b)

    def test_division_and_bf16_cast_have_separate_gates(self):
        a,b = records();b['details'][-1] += 16
        with self.assertRaisesRegex(ValueError, 'division'):
            check(a,b)
        a,b = records();b['output'][-1] ^= 1
        with self.assertRaisesRegex(ValueError, 'BF16'):
            check(a,b)

    def test_zero_masked_diagnostic_and_padded_branches(self):
        a,b = records()
        for row in (a,b):
            row.update(width=-1, tiles=0, state=[], output=[0]*256)
            row['details'] = [c.MARK]*64+[0]*256+[c.MARK]*256
        self.assertEqual(check(a,b)['zero_heads'], 2)
        b['output'][1] = 1
        with self.assertRaisesRegex(ValueError, 'no-division'):
            check(a,b)
        for row in (a,b):
            row['output'] = [0x7fc0]*256
        self.assertEqual(check(a,b)['diagnostic_heads'], 2)
        for row in (a,b):
            row.update(head=15, output=[])
        self.assertEqual(check(a,b)['padded_heads'], 1)

    def test_zero_sign_and_nonfinite_cannot_hide_in_tolerance(self):
        a,b = records()
        for row in (a,b):
            row['details'][64] = 0x80000000
            row['details'][320] = 0x80000000
            row['output'][0] = 0x8000
        check(a,b)
        b['details'][320] = 0;b['output'][0] = 0
        with self.assertRaisesRegex(ValueError, 'zero sign'):
            check(a,b)
        a,b = records();b['details'][320] = 0x7f800000
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            check(a,b)

    def test_fp32_subnormal_to_bf16_rounding_and_ulp(self):
        self.assertEqual(c.bf16_rne(0x8000), 0)
        self.assertEqual(c.bf16_rne(0x18000), 2)
        self.assertEqual(c.bf16_rne(0x80008000), 0x8000)
        self.assertEqual(c.ulp(q.MIN_NORMAL), q.MIN_SUBNORMAL)
        self.assertEqual(c.ulp(1-q.power2(-24)), q.power2(-24))
        self.assertEqual(c.ulp(1), q.power2(-23))

    def test_incomplete_duplicate_or_unqualified_capture_is_refused(self):
        a,b = records()
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            c.check('',mode='tiny',page=4,arithmetic_mode='RNE_gradual',
                    division_ulps=F(5,2),bf16_mode='RNE')
        with self.assertRaisesRegex(ValueError, 'independently'):
            c.pair(a,b,arithmetic_mode='FTZ',division_ulps=F(5,2),bf16_mode='RNE')
        with self.assertRaises(ValueError):
            c.pair(a,b,arithmetic_mode='RNE_gradual',division_ulps=None,bf16_mode='RNE')


if __name__ == '__main__':
    unittest.main()
