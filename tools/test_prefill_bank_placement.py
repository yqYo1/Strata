"""Host refusal controls for the prospective placement screen; repo LICENSE.

Synthetic telemetry exercises distinct route/admission/decision failure modes;
it establishes no GPU allocation, timing or model qualification.
"""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent/'sycl'))
import prefill_bank_placement as campaign
from tools.test_prefill_bank_route import ordered_fixture


def fixture(read=131072,slots=256,trace=True):
    n=read-1;staged=slots==256;gpu=n if staged else 98304;effective=slots if staged else 512
    text=ordered_fixture(effective,n=n,gpu=gpu,split_requested=True)
    text=text.replace(f'requested_slots={effective}',f'requested_slots={slots}')
    if not staged:
        text=text.replace('staged=1','staged=0')
        text='\n'.join(x for x in text.splitlines() if not x.startswith('strata MoE expert bank complete: '))+'\n'
    prefix=((n+15)//16)*16*2056
    free=98304*20480+512*2662400+prefix+256+536870912+(64<<20)
    fixed=gpu*20480+effective*2662400+prefix+256+536870912
    maximum=min(81920,gpu)
    text+=(f'strata MoE accumulator admission: requested_mode=10 rows={maximum} gpu_prefix_rows={gpu} '
        f'shared_arena_bytes=3000000000 free_gpu_bytes={free} fixed_including_headroom_bytes={fixed} '
        f'mixed_fp16_bytes={maximum*5120}\n'
        f'strata prefill expert source: requested_view=1 effective_view={int(staged)} layers={48 if staged else 0} '
        f'owned_slab_bytes={0 if staged else 512*2662400} borrowed_layer_span_bytes={50000000000 if staged else 0} '
        f'host_copy={"eliminated" if staged else "synchronous"} source_pin=unspecified source_owner=caller_through_checked_retirement\n'
        'strata prefill: native16 BF16 HC; QSA tensor IO=BF16, indexer/GDN internals FP32;\n'
        'strata softmax: softmax=parallel-exp-ascending-sum\n')
    for phase in ('before-detach','after-detach','after-prompt-before-restore','after-restore'):
        text+=f'strata mtp: prompt expert phase={phase} bytes=707788800\n'
    text+='strata mtp: experts restored bytes=707788800, unchanged=1\n'
    if trace and staged:
        for layer in range(48):
            text+=(f'strata packed expert bank: layer={layer} slots=256 stride=2662400 '
                'used_bytes=681574400 backing_bytes=681574400 full_fit=0\n')
    return text,dict(read=read,slots=slots,heads=1,trace_progress=trace)


def processes(ratio=.99,mechanism=True):
    result=[]
    for i,(read,slots) in enumerate(zip(campaign.READS,campaign.SLOTS)):
        text,plan=fixture(read,slots,trace=i<3)
        bank=campaign.placement_route(text,plan)
        if not mechanism and slots==256:bank['requested_bank_H2D_bytes']=bank['immutable_all_expert_layer_bytes']+1
        duration=100000*(ratio if slots==256 else 1)
        sample=dict(prompt_ms=duration,prompt_tps=read*1000/duration,ttft_s=duration/1000+.2)
        result.append(dict(index=i,read=read,passed=True,complete=True,ownership_closed=True,
            native_exit_code=0,fresh_faults=[],samples=[copy.deepcopy(sample) for _ in range(3)],
            routes=[dict(bank=copy.deepcopy(bank)) for _ in range(3)],subsequent_median_ms=duration))
    return result


class PlacementControls(unittest.TestCase):
    def test_small_full_and_partial_placements(self):
        for read,slots in ((32769,256),(131072,384),(131072,256)):
            text,plan=fixture(read,slots)
            value=campaign.request_route(text,plan)
            self.assertEqual(value['bank']['residual_host_rows'],32767 if slots==384 else 0)
            self.assertEqual(value['bank']['windows'],48 if read==32769 else 96)
            self.assertTrue(value['bank']['budget_reproduced'])

    def test_budget_and_single_allocation_refusals(self):
        text,plan=fixture()
        for changed in (text.replace('free_gpu_bytes=','free_gpu_bytes=9'),
            text.replace('fixed_including_headroom_bytes=','fixed_including_headroom_bytes=9'),
            text.replace('allocation_limit_bytes=10132807680','allocation_limit_bytes=2000000000')):
            with self.assertRaises(RuntimeError):campaign.placement_route(changed,plan)

    def test_unplanned_partial_or_work_extent_refused(self):
        text,plan=fixture()
        for old,new in [('residual_gpu_rows=131071','residual_gpu_rows=98304'),
            ('windows=96','windows=48'),('maximum_rows=81920','maximum_rows=131071'),
            ('requested_slots=256','requested_slots=128')]:
            with self.subTest(old=old):
                with self.assertRaises(RuntimeError):campaign.placement_route(text.replace(old,new),plan)

    def test_inactive_split_requires_its_actual_marker(self):
        text,plan=fixture()
        for changed in (text.replace('active=0 rows=','active=1 rows='),
            '\n'.join(x for x in text.splitlines() if not x.startswith('strata MoE split admission: ')),
            text+'strata MoE split complete: rows=131071\n'):
            with self.assertRaises(RuntimeError):campaign.placement_route(changed,plan)

    def test_source_geometry_and_checked_retirement_refused(self):
        text,plan=fixture()
        for old,new in [('source_owner=caller_through_checked_retirement','source_owner=unknown'),
            ('full_fit=0','full_fit=1'),('backing_bytes=681574400','backing_bytes=681574399'),
            ('final_drain=checked','final_drain=unchecked'),('storage=scratch','storage=owned')]:
            with self.subTest(old=old):
                with self.assertRaises(RuntimeError):campaign.placement_route(text.replace(old,new),plan)

    def test_wrapper_math_job_conservation_and_handoff_refusals(self):
        text,plan=fixture()
        for old,new in [('softmax=parallel-exp-ascending-sum','softmax=subgroup'),
            ('QSA tensor IO=BF16','QSA tensor IO=FP32'),('experts restored bytes=707788800','experts restored bytes=1'),
            ('GU fusion: fused_jobs=10000','GU fusion: fused_jobs=9999')]:
            with self.subTest(old=old):
                with self.assertRaises(RuntimeError):campaign.request_route(text.replace(old,new),plan)

    def test_time_failure_does_not_erase_mechanism_success(self):
        result=campaign.decision(processes(ratio=1.01))
        self.assertTrue(result['actual_residency_and_requested_transfer_mechanism'])
        self.assertFalse(result['eligible_for_further_model_gates'])
        self.assertFalse(result['repeated_prefill_gain'])

    def test_mechanism_failure_does_not_erase_time_success(self):
        result=campaign.decision(processes(mechanism=False))
        self.assertTrue(result['repeated_prefill_gain'])
        self.assertFalse(result['actual_residency_and_requested_transfer_mechanism'])
        self.assertFalse(result['eligible_for_further_model_gates'])
        self.assertTrue(campaign.decision(processes())['eligible_for_further_model_gates'])


if __name__=='__main__':unittest.main()
