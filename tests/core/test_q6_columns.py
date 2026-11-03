"""Reject invalid evidence before advancing a column-specialization candidate."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/sycl'))
import measure_q6_columns as comparison
import run_q6_columns as runner


def samples():
    lines = []
    for ni, no, calls in comparison.SHAPES:
        for nc in (1, 2, 3, 4, 8):
            for batch in range(5):
                for position in (0, 1):
                    arm = (batch + nc + position) % 2
                    lines.append(f'Q6_SAMPLE ni={ni} no={no} nc={nc} arm={arm} '
                        f'batch={batch} position={position} calls={calls} '
                        f'seconds={.125 if arm else .25}')
    return lines


class Evidence(unittest.TestCase):
    def test_complete_production_shapes(self):
        result = comparison.parse_speed('\n'.join(samples()))
        self.assertEqual(len(result), 35)
        self.assertTrue(all(s['median_time_ratio'] == .5 for s in result))

    def test_missing_duplicate_and_wrong_shape(self):
        lines = samples()
        for changed in (lines[:-1], lines + [lines[0]],
                        [lines[0].replace('ni=2560', 'ni=1280')] + lines[1:]):
            with self.subTest(first=changed[0], count=len(changed)), self.assertRaises(RuntimeError):
                comparison.parse_speed('\n'.join(changed))

    def test_wrong_order_and_call_count(self):
        lines = samples()
        for wrong in ('position=7', 'calls=1'):
            field = 'position=0' if wrong.startswith('position') else 'calls=256'
            with self.subTest(wrong=wrong), self.assertRaises(RuntimeError):
                comparison.parse_speed('\n'.join([lines[0].replace(field, wrong)] + lines[1:]))

    def test_short_and_nonfinite_batch(self):
        lines = samples()
        for value in ('0.009', 'nan', 'inf'):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                comparison.parse_speed('\n'.join([lines[0].replace('seconds=0.125',
                    'seconds=' + value)] + lines[1:]))


class ValidationTrace(unittest.TestCase):
    begin = 'strata UR validation: pid=17 phase=begin loader=/qualified/loader\n'
    negative = ('   <--- urPlatformGetApiVersion(.hPlatform = nullptr, '
        '.pVersion = 0xabcdef (-1.65535)) -> UR_RESULT_ERROR_INVALID_NULL_HANDLE;\n')
    proof = ('strata UR validation: pid=17 phase=before-main threads_before=1 '
        'preinit=37 config_create=0 layers_query=0 enable=0 init=0 config_release=0 '
        'checked=46 untouched=1 retained_ref=1\n')

    def test_exact_pre_main_probe_and_quiet_mode(self):
        self.assertEqual(runner.validation_log(self.begin+self.negative+self.proof,
            17, '/qualified/loader', True)['dispatch_result'], 46)
        runner.validation_log(self.begin+self.proof, 17, '/qualified/loader', False)

    def test_wrong_location_duplicate_api_and_fresh_error(self):
        cases = [self.begin+self.proof+self.negative,
            self.begin+self.negative*2+self.proof,
            self.begin+self.negative.replace('urPlatformGetApiVersion', 'urUSMDeviceAlloc')+self.proof,
            self.begin+self.negative+self.proof+'unexpected UR_RESULT_ERROR_DEVICE_LOST\n',
            self.begin+self.negative+self.proof+'[WARNING] bad parameter\n']
        for text in cases:
            with self.subTest(text=text), self.assertRaises(RuntimeError):
                runner.validation_log(text, 17, '/qualified/loader', True)


if __name__ == '__main__':
    unittest.main()
