"""Host-only multi-window, phase-isolation and independent-gate controls.

Synthetic reader/decision inputs below are not model measurements. Main-authored;
repository LICENSE, no XeStrata source or GPU operation.
"""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parent/'sycl'))
import prefill_bank_reuse as b


def fixture(slots=384,n=b.READ-1,gpu=None):
    gpu=n if gpu is None else gpu
    maximum=min(81920,gpu);windows=48*((n+81919)//81920);size=slots*b.STRIDE
    return (f'strata prefill: layer-major candidate, tokens={n} chunk=8192 residual_dtype=bf16 '
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


def ordered_fixture(*args,**kwargs):
    # Production emits head retirement before allocating/admitting the bank.
    lines=fixture(*args,**kwargs).splitlines()
    return '\n'.join(lines[-2:]+[lines[1],lines[0]]+lines[2:-2])+'\n'


def processes():
    values=[]
    for i,slots in enumerate(b.SLOTS):
        ms=100.0 if slots==384 else 98.0
        bank=dict(requested_bank_H2D_bytes=200 if slots==384 else 100,permanent_hits=10 if slots==384 else 20)
        routes=[dict(actual_GU_dispatch=dict(jobs=60000,windows=96),bank=copy.deepcopy(bank)) for _ in range(3)]
        if i==0:
            for r in routes:r['actual_full_fit_layers']=48
        samples=[dict(prompt_ms=ms,prompt_tps=b.READ*1000/ms,ttft_s=ms/1000+.1) for _ in range(3)]
        values.append(dict(index=i,read=b.READ,passed=True,complete=True,ownership_closed=True,
            native_exit_code=0,fresh_faults=[],samples=samples,routes=routes,subsequent_median_ms=ms))
    return values


class ReuseControls(unittest.TestCase):
    def route(self,text,slots=384):
        return b.comparison.route(text,b.READ,1,expected_rows=b.READ-1,
            requested_slots=slots,draft_bytes=0,multi_window=True)

    def test_two_window_route_and_copy_counts(self):
        for slots in (384,512):
            p=self.route(ordered_fixture(slots),slots)
            self.assertEqual((p['windows'],p['maximum_window_rows']),(96,81920))
            self.assertEqual(p['permanent_hits'],18432);self.assertEqual(p['uploads'],30720)
            self.assertIn('requested_source_map_H2D_bytes',p)

    def test_legacy_route_contract_remains(self):
        p=b.comparison.route(ordered_fixture(n=65536),65537,1,expected_rows=65536,draft_bytes=0)
        self.assertEqual((p['windows'],p['maximum_window_rows']),(48,65536))
        self.assertNotIn('permanent_hits',p);self.assertIn('requested_dequant_H2D_bytes',p)
        with self.assertRaises(RuntimeError):
            b.comparison.route(ordered_fixture(),b.READ,1,expected_rows=b.READ-1,draft_bytes=0)

    def test_refuse_wrong_extent_or_window_and_storage(self):
        for old,new in [('windows=96','windows=48'),('maximum_rows=81920','maximum_rows=131071'),
            ('active=0 rows=','active=1 rows='),('permanent_hits=18432','permanent_hits=-1'),
            ('tail_bytes=83886080','tail_bytes=83886081'),('mixed_fp16_bytes=0','mixed_fp16_bytes=1'),
            (f'residual_total={(b.READ-1)*20480}',f'residual_total={(b.READ-1)*20480+1}'),
            ('requested_slots=384','requested_slots=512')]:
            with self.subTest(old=old):
                text=ordered_fixture();self.assertIn(old,text)
                with self.assertRaises(RuntimeError):self.route(text.replace(old,new))
        with self.assertRaises(RuntimeError):self.route(ordered_fixture(gpu=b.READ-8193))

    def test_refuse_unplanned_multi_window_scope(self):
        for read,rows,heads in ((b.READ-1,b.READ-2,1),(b.READ,b.READ-1,0),(65537,65536,1)):
            with self.assertRaises(RuntimeError):b.comparison.route(ordered_fixture(),read,heads,
                expected_rows=rows,draft_bytes=0,multi_window=True)

    def test_environment_changes_only_capacity(self):
        template=dict(STRATA_MEASUREMENT_ROOT='/old',EnableDirectSubmission='0',
            UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1',STRATA_SYCL_PREFILL_SAME_QUEUE='1',
            STRATA_SYCL_PREFILL_EXPERT_UPLOAD_STAGING='0',STRATA_SYCL_PREFILL_EXPERT_BANK_PACKED='1',
            STRATA_SYCL_CACHE_SCRATCH_LEASE='1',STRATA_SYCL_PREFILL_MIXED_SCRATCH='1',
            STRATA_SYCL_PREFILL_MOE_WINDOW='10',STRATA_SYCL_PREFILL_MOE_SPLIT_ACCUMULATOR='1')
        row=dict(configuration=dict(args=['--prefill','8192']),environment_template=copy.deepcopy(template))
        plans=[dict(root='/new',slots=s,heads=1,trace_progress=False) for s in (384,512)]
        a,c=[b.environment(row,p) for p in plans]
        self.assertEqual([k for k in a if a[k]!=c[k]],['STRATA_SYCL_PREFILL_EXPERT_BANK_SLOTS'])
        self.assertEqual(a['STRATA_SYCL_PREFILL_MOE_SPLIT_ACCUMULATOR'],'0')
        self.assertEqual(b.invariant(row,plans[0]),b.invariant(row,plans[1]));self.assertEqual(row['environment_template'],template)

    def test_time_and_copy_mechanism_are_independent(self):
        values=processes();p=b.decision(values);self.assertTrue(p['eligible_for_further_model_gates'])
        self.assertFalse(p['inference_adopted'])
        values[6]['subsequent_median_ms']=101;p=b.decision(values)
        self.assertTrue(p['requested_copy_reuse_mechanism']);self.assertFalse(p['repeated_prefill_gain'])
        values=processes();values[2]['routes'][1]['bank']['requested_bank_H2D_bytes']=200;p=b.decision(values)
        self.assertTrue(p['repeated_prefill_gain']);self.assertFalse(p['eligible_for_further_model_gates'])

    def test_decision_rejects_incomplete_or_mismatched_math(self):
        for which in ('order','math','fault'):
            values=processes()
            if which=='order':values[6]['index']=5
            if which=='math':values[2]['routes'][0]['actual_GU_dispatch']['jobs']+=1
            if which=='fault':values[5]['fresh_faults']=['unit-test-fault']
            with self.assertRaises(RuntimeError):b.decision(values)


if __name__=='__main__':unittest.main()
