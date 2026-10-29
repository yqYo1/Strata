"""Host tests of exact rounding/error gates, not GPU or task qualification."""
from fractions import Fraction as F
import struct
import unittest
from sycl import qsa_sum_interval as q


def bits(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


class IntervalTests(unittest.TestCase):
    def test_full_exponent_spread_is_not_lost_to_binary64(self):
        total, interval = q.tile_interval([bits(1.0), bits(2.0**-100)] + [0]*30,
            depth=16, arithmetic_mode='RNE_gradual')
        self.assertEqual(total, 1+q.power2(-100))
        self.assertNotEqual(total, F(float(total)))
        self.assertLessEqual(interval[0], total)
        self.assertGreaterEqual(interval[1], total)

    def test_subnormal_positive_sum_is_exact(self):
        total, interval = q.tile_interval([1]*32, depth=16, arithmetic_mode='RNE_gradual')
        self.assertEqual(interval, (32*q.MIN_SUBNORMAL,)*2)
        self.assertEqual(total, interval[0])

    def test_known_binary32_midpoints_and_exponent_boundary(self):
        self.assertEqual(q.rn32(1+q.power2(-24)), 1)
        self.assertEqual(q.rn32(1+3*q.power2(-24)), 1+q.power2(-22))
        self.assertEqual(q.rn32(-1-q.power2(-24)), -1)
        self.assertEqual(q.rn32(q.MIN_SUBNORMAL/2), 0)
        self.assertEqual(q.rn32(q.MIN_SUBNORMAL*F(3,2)), 2*q.MIN_SUBNORMAL)
        self.assertEqual(q.fp32_bounds(q.MIN_NORMAL-q.MIN_SUBNORMAL/2),
                         (q.MIN_NORMAL-q.MIN_SUBNORMAL, q.MIN_NORMAL))

    def test_fma_endpoint_uses_exact_product_and_handles_underflow(self):
        prior = (q.MIN_SUBNORMAL,)*2
        bounds = q.normalizer_step(prior, bits(0.5), (F(0),F(0)), arithmetic_mode='RNE_gradual')
        self.assertEqual(bounds, (F(0),q.MIN_SUBNORMAL))
        x = 1+q.power2(-23)
        result = q.normalizer_step((x,x), bits(1-2.0**-23),
            (q.power2(-100),)*2, arithmetic_mode='RNE_gradual')
        exact = x*q.exact_fp32(bits(1-2.0**-23))+q.power2(-100)
        self.assertEqual(result, (1-q.power2(-24), F(1)))
        self.assertLessEqual(result[0], q.rn32(exact))
        self.assertGreaterEqual(result[1], q.rn32(exact))
        result = q.normalizer_step((F(1),F(1)), bits(1.0),
            (q.power2(-100),)*2, arithmetic_mode='RNE_gradual')
        self.assertEqual(result, (F(1),1+q.power2(-23)))

    def test_65_tile_constant_sequence_encloses_both_sum_orders(self):
        pattern = [bits(1.0)]+[bits(2.0**-25)]*31
        total, serial = q.tile_interval(pattern, depth=31, arithmetic_mode='RNE_gradual')
        _, parallel = q.tile_interval(pattern, depth=16, arithmetic_mode='RNE_gradual')
        self.assertLessEqual(serial[0], parallel[0])
        self.assertGreaterEqual(serial[1], parallel[1])
        control = F(0)
        for value in pattern:control = q.rn32(control+q.exact_fp32(value))
        self.assertEqual(control, 1)
        interval, value = (F(0),F(0)), F(0)
        for _ in range(65):
            interval = q.normalizer_step(interval,bits(1.0),serial,arithmetic_mode='RNE_gradual')
            value = q.rn32(value+control)
            self.assertLessEqual(interval[0],value)
            self.assertGreaterEqual(interval[1],value)

    def test_signed_numerator_and_zero_denominator_are_separate(self):
        self.assertEqual(q.exact_ratio_interval(bits(-2.0),(F(1),F(2))), (F(-2),F(-1)))
        with self.assertRaises(ValueError):q.exact_ratio_interval(bits(0.0),(F(0),F(1)))

    def test_unqualified_modes_nonfinite_negative_and_overflow_rejected(self):
        with self.assertRaises(ValueError):q.tile_interval([0]*32,depth=16,arithmetic_mode='FTZ')
        with self.assertRaises(ValueError):q.tile_interval([bits(-1.)]+[0]*31,depth=16,arithmetic_mode='RNE_gradual')
        with self.assertRaises(ValueError):q.exact_fp32(0x7f800000)
        with self.assertRaises(ValueError):q.fp32_bounds(q.MAX_FINITE+1)


if __name__=='__main__':unittest.main()
