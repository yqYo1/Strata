"""Check requested-copy accounting, launch normalization and conservation."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).parent/'sycl'))
import request_resource_summary as resource
from tools import test_request_profile_reader as controls


class RequestResourceSummaryTests(unittest.TestCase):
    def test_copy_bytes_overlap_and_kernel_shapes(self):
        case=controls.RequestProfileReaderTests();case.setUp()
        trace=dict(traceEvents=[
            dict(cat='gpu_op',ph='X',name='zeCommandListAppendMemoryCopy(M2D)[128]',ts=3,dur=4),
            dict(cat='gpu_op',ph='X',name='kernel<4>[SIMD16 {1,1,1} {128,1,1}]',ts=5,dur=3),
            dict(cat='gpu_op',ph='X',name='kernel<4>[SIMD16 {2,1,1} {128,1,1}]',ts=8,dur=1),
            dict(cat='gpu_op',ph='X',name='crossing',ts=11,dur=3)])
        original=resource.reader.summarize(trace,case.receipt)
        unseeded=resource.summarize(trace,case.receipt)
        self.assertEqual(unseeded,resource.summarize(trace,case.receipt,original))
        q=unseeded['requests'][0]
        self.assertEqual(q['copy_and_other_commands']['M2D']['requested_API_bytes'],128)
        self.assertEqual(q['copy_kernel_overlap_ns'],2000)
        self.assertEqual(q['top_kernel_families'],[dict(name='kernel<4>',calls=2,service_ns=4000)])
        self.assertEqual(q['boundary_crossing_GPU_events'],1)
        self.assertEqual(q['contained_GPU_service_ns'],8000)

    def test_changed_original_summary_is_rejected(self):
        case=controls.RequestProfileReaderTests();case.setUp()
        original=resource.reader.summarize(case.trace,case.receipt)
        original['requests'][0]['contained_GPU_service_ns']+=1
        with self.assertRaises(ValueError):resource.summarize(case.trace,case.receipt,original)


if __name__=='__main__':unittest.main()
