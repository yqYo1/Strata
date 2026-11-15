"""Regression for the actual clean-model end-gate parser; repository LICENSE.

The always-on headers below are from the failed64K GDN16 campaign. STRATA_TRACE
is absent in that experiment,so the policy line must not be required to prove
the actual native16 route. No model,server or GPU is loaded by these tests.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/sycl'))
import compare_prefill_strategies as campaign


MAIN=('strata prefill: layer-major candidate, tokens=65536 chunk=8192 '
      'residual_dtype=bf16 residual_element_bytes=2 residual_total=1342177280 '
      'residual_gpu_rows=65536 residual_gpu=1342177280 residual_host=0 '
      'residual_staging=0 layer_cache=1022361600; extra allocations, '
      'GPU headroom=536870912, RAM headroom=8589934592\n')
NATIVE=('strata prefill: native16 BF16 HC residual/projection IO and MoE outputs; '
        'FP32 local accumulations/router/recurrent state; PLE and final callbacks '
        'bridge FP32; QSA tensor IO=BF16, indexer/GDN internals FP32; '
        'reserved scratch arena unchanged; unqualified\n')
TRACE='strata prefill policy: requested=throughput route=layer-major tokens=65536 chunk=8192\n'


class Native16CleanRoute(unittest.TestCase):
    def test_actual_clean_headers_without_trace_policy(self):
        campaign.native16_route(MAIN+NATIVE,65537)

    def test_optional_trace_must_agree(self):
        campaign.native16_route(TRACE+MAIN+NATIVE,65537)
        with self.assertRaises(RuntimeError):
            campaign.native16_route(TRACE.replace('throughput','ttft')+MAIN+NATIVE,65537)

    def test_trace_and_native_marker_without_actual_geometry_rejected(self):
        with self.assertRaises(RuntimeError):
            campaign.native16_route(TRACE+NATIVE,65537)

    def test_trace_cannot_hide_wrong_actual_geometry(self):
        for wrong in (MAIN.replace('tokens=65536','tokens=65535'),
                      MAIN.replace('chunk=8192','chunk=0'),
                      MAIN.replace('residual_dtype=bf16','residual_dtype=fp32')):
            with self.subTest(wrong=wrong),self.assertRaises(RuntimeError):
                campaign.native16_route(TRACE+wrong+NATIVE,65537)

    def test_missing_and_duplicate_actual_headers_rejected(self):
        for wrong in ('',TRACE+MAIN,TRACE+MAIN+MAIN+NATIVE,TRACE+MAIN+NATIVE+NATIVE):
            with self.subTest(wrong=wrong),self.assertRaises(RuntimeError):
                campaign.native16_route(wrong,65537)


if __name__=='__main__':
    unittest.main()
