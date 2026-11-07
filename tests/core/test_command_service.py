"""Host rejection controls for sparse command profiles; no GPU required."""
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/sycl'))
from command_service import parse_profiles


def fixture(version=3):
    second, bin_, cap = ('gu_swiglu', 8, 224) if version == 3 else ('down', 7, 192)
    return (
        'strata command sample: window=0 kind=gu bin=7 expert=0 m=2333 n=1280 k=2560 format=21 row=0 submit=10 start=20 end=40\n'
        f'strata command sample: window=0 kind={second} bin={bin_} expert=1 m=2334 n=1280 k=2560 format=21 row=2333 submit=90 start=70 end=100\n'
        'strata command group: window=0 layer=0 first_row=0 kind=gu bin=7 eligible=8 selected=1 service_samples=1 queue_delay_samples=1 sample_service_ns=20 sample_queue_delay_ns=10\n'
        f'strata command group: window=0 layer=0 first_row=0 kind={second} bin={bin_} eligible=8 selected=1 service_samples=1 queue_delay_samples=0 sample_service_ns=30 sample_queue_delay_ns=0\n'
        f'strata command profile: version={version} service_complete=1 submit_clock_qualified=0 submit_clock_inversions=1 windows=1 expected=1 records=2 peak_events=2 cap={cap} timer_resolution_ns=1 max_query_wall_ns=100 error=none invalid=0 invalid_saved=0 scope=selected_first_windows hardware_busy=UNMEASURED\n')


class CommandServiceTests(unittest.TestCase):
    def parse(self, text):
        return parse_profiles(text, expected_windows=1)

    def test_both_versions_preserve_service_and_signed_submit_diagnostic(self):
        for version in (2, 3):
            value, = self.parse(fixture(version))
            self.assertEqual(value['version'], version)
            self.assertEqual([s['service_ns'] for s in value['samples']], [20, 30])
            self.assertEqual([s['start_minus_submit_ns'] for s in value['samples']], [10, -20])

    def test_boundary_and_maximum_job_have_separate_fused_stratum(self):
        text = fixture().replace('m=2334', 'm=8192')
        self.assertEqual(self.parse(text)[0]['samples'][1]['m'], '8192')
        for old, new in [('m=2333', 'm=2334'), ('m=2334', 'm=2333'), ('m=2334', 'm=8193')]:
            with self.subTest(new=new), self.assertRaises(RuntimeError):
                self.parse(fixture().replace(old, new))

    def test_expert_transfer_is_separate_and_unsampled_strata_fail(self):
        text = fixture().replace('kind=gu bin=7', 'kind=expert_upload bin=0')
        self.assertEqual(self.parse(text)[0]['samples'][0]['kind'], 'expert_upload')
        with self.assertRaises(RuntimeError):
            self.parse(text.replace('selected=1', 'selected=0'))

    def test_missing_duplicate_or_trailing_records_are_rejected(self):
        lines = fixture().splitlines(True)
        for text in (''.join(lines[1:]), ''.join(lines[:2]+lines[3:]), ''.join(lines[:-1]),
                     ''.join(lines[:4]+[lines[3]]+lines[4:]), fixture()+lines[0]):
            with self.subTest(text=text), self.assertRaises(RuntimeError):
                self.parse(text)

    def test_count_service_and_delay_conservation_is_mandatory(self):
        for old, new in [('eligible=8', 'eligible=0'), ('selected=1', 'selected=2'),
                         ('service_samples=1', 'service_samples=2'), ('sample_service_ns=20', 'sample_service_ns=19'),
                         ('sample_queue_delay_ns=10', 'sample_queue_delay_ns=9'), ('records=2', 'records=3'),
                         ('submit_clock_inversions=1', 'submit_clock_inversions=0')]:
            with self.subTest(new=new), self.assertRaises(RuntimeError):
                self.parse(fixture().replace(old, new))

    def test_bad_gpu_clock_or_unsigned_underflow_is_rejected(self):
        for old, new in [('end=40', 'end=19'), ('start=70', 'start=39'), ('submit=90', 'submit=-1'),
                         ('timer_resolution_ns=1', 'timer_resolution_ns=21'),
                         ('max_query_wall_ns=100', 'max_query_wall_ns=1000000000')]:
            with self.subTest(new=new), self.assertRaises(RuntimeError):
                self.parse(fixture().replace(old, new))

    def test_error_capacity_version_and_scope_cannot_be_relaxed(self):
        for old, new in [('service_complete=1', 'service_complete=0'), ('error=none', 'error=event_capacity'),
                         ('invalid=0', 'invalid=1'), ('cap=224', 'cap=192'), ('version=3', 'version=4'),
                         ('kind=gu_swiglu', 'kind=unknown'), ('layer=0', 'layer=15'),
                         ('hardware_busy=UNMEASURED', 'hardware_busy=SATURATED')]:
            with self.subTest(new=new), self.assertRaises(RuntimeError):
                self.parse(fixture().replace(old, new))

    def test_closed_profiles_split_at_footer(self):
        self.assertEqual(len(self.parse('unrelated log\n'+fixture()+fixture())), 2)
        with self.assertRaises(RuntimeError):
            self.parse(fixture().replace('kind=gu bin=7', 'kind=gu bin=7 bin=7'))


if __name__ == '__main__':
    unittest.main()
