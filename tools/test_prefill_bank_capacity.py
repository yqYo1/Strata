"""Negative controls for the bank-only experiment; repository LICENSE.

Synthetic admission/route telemetry is not a GPU or model qualification.
"""
import copy
from pathlib import Path
import re
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parent/'sycl'))
import prefill_bank_capacity_bench as campaign


def fixture(read=131072,slots=1,trace=True):
    n=read-1;cache=slots*2662400;maximum=min(n,81920)
    prefix=((n+15)//16)*16*2056
    fixed=n*20480+cache+prefix+256+536870912
    text=(
        'strata prefill heads: target detached bytes=1\n'
        'strata prefill heads: draft detached bytes=1\n'
        f'strata MoE expert bank: requested_slots={slots} effective_slots={slots} staged=1 host_slab_bytes={512*2662400} device_bytes={cache} residual_gpu_rows={n} allocation_limit_bytes=10132807680 general_map=private_scratch_never_published\n'
        f'strata prefill: layer-major candidate, tokens={n} chunk=8192 residual_dtype=bf16 residual_element_bytes=2 residual_total={n*20480} residual_gpu_rows={n} residual_gpu={n*20480} residual_host=0 residual_staging=0 layer_cache={cache}\n'
        f'strata MoE accumulator admission: requested_mode=10 rows={maximum} gpu_prefix_rows={n} shared_arena_bytes=1180698624 free_gpu_bytes=4245643264 fixed_including_headroom_bytes={fixed} mixed_fp16_bytes={maximum*5120}\n'
        f'strata MoE split admission: active=0 rows={n} primary_rows=0 tail_payload_bytes=0 extra_allocation_bytes=0 prefill_slots={slots} max_single_allocation_bytes=10132807680 decode_changed=0\n'
        f'strata MoE window configuration: requested_mode=10 effective_mode=10 maximum_rows={maximum} mixed_fp16_bytes=0\n'
        f'strata MoE accumulator window: windows={48*((n+81919)//81920)} dequants=2000 jobs=10000 source_H2D_bytes=100000 GPU_resident_windows_only=1 operator_max=8192 maximum_rows={maximum}\n'
        f'strata MoE mixed banks: storage=scratch primary_bytes={min(maximum,65536)*5120} tail_bytes={max(0,maximum-65536)*5120} tail_offset=900000000 extra_allocation_bytes=0 phase=FFN_only_after_attention_drain\n'
        'strata prefill expert source: requested_view=1 effective_view=1 layers=48 owned_slab_bytes=0 borrowed_layer_span_bytes=50000000000 host_copy=eliminated source_pin=unspecified source_owner=caller_through_checked_retirement\n'
        f'strata MoE expert bank complete: total_slots={slots} permanent_slots=0 requested_H2D_bytes={100000000000 if slots==1 else 44000000000} permanent_hits=0 uploads=2000 final_drain=checked\n'
        'strata prefill: softmax=parallel-exp-ascending-sum\n')
    for phase in ('before-detach','after-detach','after-prompt-before-restore','after-restore'):
        text+=f'strata mtp: prompt expert phase={phase} bytes=707788800\n'
    text+='strata mtp: experts restored bytes=707788800, unchanged=1\n'
    if trace:
        for layer in range(48):
            text+=f'strata packed expert bank: layer={layer} slots={slots} stride=2662400 used_bytes={cache} backing_bytes={cache} full_fit=0\n'
    return text,dict(read=read,slots=slots,heads=1,trace_progress=trace)


def processes(ratio=1.1):
    result=[]
    for i,(read,slots) in enumerate(zip(campaign.READS,campaign.SLOTS)):
        text,plan=fixture(read,slots,i<3);bank=campaign.placement_route(text,plan)
        duration=100000*(ratio if slots==1 else 1)
        sample=dict(prompt_ms=duration,prompt_tps=read*1000/duration,ttft_s=duration/1000+.2)
        route=dict(bank=bank,actual_GU_dispatch=[dict(jobs=10000,fused=500)])
        result.append(dict(index=i,read=read,passed=True,complete=True,ownership_closed=True,
            native_exit_code=0,fresh_faults=[],samples=[copy.deepcopy(sample) for _ in range(3)],
            routes=[copy.deepcopy(route) for _ in range(3)],subsequent_median_ms=duration))
    return result


class BankControls(unittest.TestCase):
    def test_declared_small_and_two_long_arms(self):
        for read,slots in ((32769,1),(131072,256),(131072,1)):
            text,plan=fixture(read,slots);route=campaign.placement_route(text,plan)
            self.assertEqual(route['residual_host_rows'],0)
            self.assertEqual(route['windows'],48 if read==32769 else 96)
            self.assertTrue(route['budget_reproduced'])

    def test_budget_and_single_allocation_refusals(self):
        text,plan=fixture()
        for changed in (re.sub(r'free_gpu_bytes=\d+','free_gpu_bytes=0',text),
            re.sub(r'fixed_including_headroom_bytes=\d+','fixed_including_headroom_bytes=1',text),
            text.replace('allocation_limit_bytes=10132807680','allocation_limit_bytes=2000000000')):
            with self.assertRaises(RuntimeError):campaign.placement_route(changed,plan)

    def test_workload_and_split_drift_refused(self):
        text,plan=fixture()
        for a,b in (('windows=96','windows=48'),('active=0 rows=','active=1 rows='),
            ('maximum_rows=81920','maximum_rows=131071'),('residual_host=0','residual_host=1'),
            ('requested_slots=1 ','requested_slots=2 '),('effective_mode=10','effective_mode=16')):
            with self.subTest(a=a):
                with self.assertRaises(RuntimeError):campaign.placement_route(text.replace(a,b),plan)

    def test_source_packed_layout_and_drain_refusals(self):
        text,plan=fixture()
        for a,b in (('source_owner=caller_through_checked_retirement','source_owner=unknown'),
            ('backing_bytes=2662400','backing_bytes=2662399'),('final_drain=checked','final_drain=unchecked'),
            ('full_fit=0','full_fit=1'),('storage=scratch','storage=owned')):
            with self.subTest(a=a):
                with self.assertRaises(RuntimeError):campaign.placement_route(text.replace(a,b),plan)

    def test_loss_retained_and_window_eligibility_independent(self):
        value=campaign.decision(processes())
        self.assertFalse(value['repeated_bank_only_gain'])
        self.assertFalse(value['bank_only_performance_eligible'])
        self.assertTrue(value['eligible_for_separately_planned_window_question'])
        self.assertGreater(value['geometric_mean_time_ratio'],1)
        self.assertTrue(campaign.decision(processes(.99))['bank_only_performance_eligible'])
        bad=processes();bad[4]['routes'][0]['bank']['jobs']+=1
        with self.assertRaises(RuntimeError):campaign.decision(bad)

    def test_math_and_decode_handoff_refusals(self):
        text,plan=fixture()
        with patch.object(campaign.core,'request_route',return_value={}),patch.object(campaign.dispatch_reader,'dispatch',return_value=[]):
            self.assertTrue(campaign.request_route(text,plan)['bank']['budget_reproduced'])
            for a,b in (('softmax=parallel-exp-ascending-sum','softmax=subgroup'),
                ('experts restored bytes=707788800','experts restored bytes=1')):
                with self.assertRaises(RuntimeError):campaign.request_route(text.replace(a,b),plan)


if __name__=='__main__':unittest.main()
