"""Maintained prefill route-reader regression; Main, repository LICENSE.

Actual native14508/controller469 V3telemetry exercises split0's missing marker.
This narrow fixture is reader evidence, not model performance/normal exit.
The closed full-wrapper experiment and nine controls restore from1ab5f0465.
"""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent/'sycl'))
import compare_official_prefill as comparison

READ,STRIDE=98305,2662400
ACTUAL_SPLIT_DISABLED = """strata prefill heads: target detached bytes=521472000 source=owned-RAM D2H=521472000 H2D-on-restore=521472000
strata prefill heads: draft detached bytes=0 retained-map-bytes=425196
strata MoE mixed banks: storage=scratch primary_bytes=335544320 tail_bytes=83886080 tail_offset=1059131904 extra_allocation_bytes=0 phase=FFN_only_after_attention_drain
strata MoE window configuration: requested_mode=10 effective_mode=10 maximum_rows=81920 mixed_fp16_bytes=0
strata MoE expert bank: requested_slots=512 effective_slots=512 staged=1 host_slab_bytes=1363148800 device_bytes=1363148800 residual_gpu_rows=98304 allocation_limit_bytes=10132807680 general_map=private_scratch_never_published
strata prefill: layer-major candidate, tokens=98304 chunk=8192 residual_dtype=bf16 residual_element_bytes=2 residual_total=2013265920 residual_gpu_rows=98304 residual_gpu=2013265920 residual_host=0 residual_staging=0 layer_cache=1363148800; extra allocations, GPU headroom=536870912, RAM headroom=8589934592
strata MoE expert bank complete: uploads=18665 requested_H2D_bytes=38107750400 permanent_hits=13563 permanent_slots=259 total_slots=512 final_drain=checked
strata MoE Down fusion: fused_jobs=6239 fused_rows=40890567 separate_jobs=29400 min_rows=2334
strata MoE accumulator window: windows=96 dequants=32228 jobs=35639 source_H2D_bytes=188743680 GPU_resident_windows_only=1 operator_max=8192 maximum_rows=81920
"""

def fixture(slots=384,n=READ-1,gpu=None,split_requested=False):
    gpu=n if gpu is None else gpu
    maximum=min(81920,gpu);windows=48*((n+81919)//81920);size=slots*STRIDE
    text=(f'strata prefill: layer-major candidate, tokens={n} chunk=8192 residual_dtype=bf16 '
        f'residual_element_bytes=2 residual_total={n*20480} residual_gpu_rows={gpu} '
        f'residual_gpu={gpu*20480} residual_host={(n-gpu)*20480} residual_staging=0 layer_cache={size};\n'
        f'strata MoE expert bank: requested_slots={slots} effective_slots={slots} staged=1 '
        f'host_slab_bytes=1363148800 device_bytes={size} residual_gpu_rows={gpu} '
        'allocation_limit_bytes=10132807680 general_map=private_scratch_never_published\n'
        f'strata MoE window configuration: requested_mode=10 effective_mode=10 maximum_rows={maximum} mixed_fp16_bytes=0\n'
        f'strata MoE mixed banks: storage=scratch primary_bytes={min(maximum,65536)*5120} '
        f'tail_bytes={max(0,maximum-65536)*5120} tail_offset=1180698624 extra_allocation_bytes=0 phase=FFN_only_after_attention_drain\n'
        f'strata MoE accumulator window: windows={windows} dequants=49152 jobs=60000 source_H2D_bytes=100000 '
        f'GPU_resident_windows_only=1 operator_max=8192 maximum_rows={maximum}\n'
        f'strata MoE split admission: active=0 rows={n} primary_rows=0 tail_payload_bytes=0 '
        f'extra_allocation_bytes=0 prefill_slots={slots} max_single_allocation_bytes=10132807680 decode_changed=0\n'
        f'strata MoE expert bank complete: uploads=30720 requested_H2D_bytes=30720000 '
        f'permanent_hits=18432 permanent_slots={slots} total_slots={slots} final_drain=checked\n'
        'strata MoE GU fusion: fused_jobs=10000 fused_rows=30000000 separate_jobs=50000 min_rows=2334\n'
        'strata MoE Down fusion: fused_jobs=10000 fused_rows=30000000 separate_jobs=50000 min_rows=2334\n'
        'strata prefill heads: target detached bytes=521472000 source=owned-RAM D2H=521472000 H2D-on-restore=521472000\n'
        'strata prefill heads: draft detached bytes=0 retained-map-bytes=425196\n')
    return text if split_requested else '\n'.join(line for line in text.splitlines()
        if not line.startswith('strata MoE split admission: '))+'\n'

def ordered_fixture(*args,**kwargs):
    # Production emits head retirement before allocating/admitting the bank.
    lines=fixture(*args,**kwargs).splitlines()
    return '\n'.join(lines[-2:]+[lines[1],lines[0]]+lines[2:-2])+'\n'


class PrefillBankRouteControls(unittest.TestCase):
    def route(self,text,slots=384):
        return comparison.route(text,READ,1,expected_rows=READ-1,
            requested_slots=slots,draft_bytes=0,multi_window=True,split_requested=False)

    def test_two_window_route_and_copy_counts(self):
        for slots in (384,512):
            p=self.route(ordered_fixture(slots),slots)
            self.assertEqual((p['windows'],p['maximum_window_rows']),(96,81920))
            self.assertEqual(p['permanent_hits'],18432);self.assertEqual(p['uploads'],30720)
            self.assertIn('requested_source_map_H2D_bytes',p)
            self.assertTrue(p['split_startup_disabled'])

    def test_legacy_route_contract_remains(self):
        p=comparison.route(ordered_fixture(n=65536,split_requested=True),65537,1,expected_rows=65536,draft_bytes=0)
        self.assertEqual((p['windows'],p['maximum_window_rows']),(48,65536))
        self.assertNotIn('permanent_hits',p);self.assertIn('requested_dequant_H2D_bytes',p)
        with self.assertRaises(RuntimeError):
            comparison.route(ordered_fixture(),READ,1,expected_rows=READ-1,draft_bytes=0)

    def test_refuse_wrong_extent_or_window_and_storage(self):
        for old,new in [('windows=96','windows=48'),('maximum_rows=81920','maximum_rows=131071'),
            ('permanent_hits=18432','permanent_hits=-1'),
            ('tail_bytes=83886080','tail_bytes=83886081'),('mixed_fp16_bytes=0','mixed_fp16_bytes=1'),
            (f'residual_total={(READ-1)*20480}',f'residual_total={(READ-1)*20480+1}'),
            ('requested_slots=384','requested_slots=512')]:
            with self.subTest(old=old):
                text=ordered_fixture();self.assertIn(old,text)
                with self.assertRaises(RuntimeError):self.route(text.replace(old,new))
        with self.assertRaises(RuntimeError):self.route(ordered_fixture(gpu=READ-8193))

    def test_disabled_startup_and_legacy_marker_contracts(self):
        for active in ('0','1'):
            with self.assertRaises(RuntimeError):
                self.route(ordered_fixture(split_requested=True).replace('active=0 rows=','active='+active+' rows='))
        with self.assertRaises(RuntimeError):
            comparison.route(ordered_fixture(),READ,1,expected_rows=READ-1,
                draft_bytes=0,multi_window=True)  # enabled default still requires its marker
        with self.assertRaises(RuntimeError):
            comparison.route(ordered_fixture(n=65536),65537,1,expected_rows=65536,
                draft_bytes=0,split_requested=False)

    def test_refuse_unplanned_multi_window_scope(self):
        for read,rows,heads in ((READ-1,READ-2,1),(READ,READ-1,0),(65537,65536,1)):
            with self.assertRaises(RuntimeError):comparison.route(ordered_fixture(),read,heads,
                expected_rows=rows,draft_bytes=0,multi_window=True)

    def test_actual_disabled_startup_excerpt(self):
        p=self.route(ACTUAL_SPLIT_DISABLED,512)
        self.assertEqual((p['rows'],p['residual_gpu_rows'],p['windows'],p['maximum_window_rows']),
            (98304,98304,96,81920))
        self.assertTrue(p['split_startup_disabled']);self.assertFalse(p['split'])
        self.assertEqual(p['permanent_hits'],13563)
        self.assertEqual(p['requested_bank_H2D_bytes'],38107750400)


if __name__=='__main__':unittest.main()
