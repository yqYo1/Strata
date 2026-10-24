"""CPU corruption controls for clock/provider/timing eligibility."""
import unittest
from unittest.mock import patch
import measurement_clock as clock


class Clock(unittest.TestCase):
    def setUp(self):
        self.provider=dict(path='/tmp/pinned-libstdc++.so.6',sha256='a'*64)
        self.text=('clock_provider=/tmp/pinned-libstdc++.so.6\n'
            'clock_samples=65536 steady=1 period_num=1 period_den=1000000000 '
            'resolution_ns=1 min_ns=20 p99_ns=31 max_ns=6823\n')

    def test_valid_screen_and_recorded_fraction(self):
        with patch.object(clock.build,'digest',return_value='a'*64):
            row=clock.evidence(self.text,self.provider)
        self.assertEqual(row['samples'],65536)
        self.assertAlmostEqual(row['endpoint_rounding_fraction_at_floor'],.0000063646)

    def test_provider_change_and_unbounded_cost_are_refused(self):
        with patch.object(clock.build,'digest',return_value='b'*64),self.assertRaises(RuntimeError):
            clock.evidence(self.text,self.provider)
        with patch.object(clock.build,'digest',return_value='a'*64):
            for text in [self.text.replace('max_ns=6823','max_ns=1000000'),
                self.text.replace('65536','65535'),self.text.replace('p99_ns=31','p99_ns=10'),
                self.text+'extra\n',self.text.replace('/tmp/pinned','/tmp/other')]:
                with self.subTest(text=text),self.assertRaises(RuntimeError):clock.evidence(text,self.provider)


if __name__ == '__main__':unittest.main()
