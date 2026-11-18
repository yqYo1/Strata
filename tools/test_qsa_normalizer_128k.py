"""Negative controls for the QSA denominator comparison; repository LICENSE.

Synthetic admission/route telemetry is not a GPU or model qualification.
"""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent/'sycl'))
import qsa_normalizer_128k as campaign


def fixture(arm, read=131072, trace=True):
    slots = 256
    n=read-1;cache=slots*2662400;maximum=min(n,81920)
    prefix=((n+15)//16)*16*2056
    fixed=n*20480+cache+prefix+256+536870912
    text=(
        'strata prefill heads: target detached bytes=521472000 source=owned-RAM D2H=521472000 H2D-on-restore=521472000\n'
        'strata prefill heads: draft detached bytes=0 retained-map-bytes=425196\n'
        f'strata MoE expert bank: requested_slots={slots} effective_slots={slots} staged=1 host_slab_bytes={512*2662400} device_bytes={cache} residual_gpu_rows={n} allocation_limit_bytes=10132807680 general_map=private_scratch_never_published\n'
        f'strata prefill: layer-major candidate, tokens={n} chunk=8192 residual_dtype=bf16 residual_element_bytes=2 residual_total={n*20480} residual_gpu_rows={n} residual_gpu={n*20480} residual_host=0 residual_staging=0 layer_cache={cache}\n'
        f'strata MoE accumulator admission: requested_mode=10 rows={maximum} gpu_prefix_rows={n} shared_arena_bytes=1180698624 free_gpu_bytes=4245643264 fixed_including_headroom_bytes={fixed} mixed_fp16_bytes={maximum*5120}\n'
        f'strata MoE split admission: active=0 rows={n} primary_rows=0 tail_payload_bytes=0 extra_allocation_bytes=0 prefill_slots={slots} max_single_allocation_bytes=10132807680 decode_changed=0\n'
        f'strata MoE window configuration: requested_mode=10 effective_mode=10 maximum_rows={maximum} mixed_fp16_bytes=0\n'
        f'strata MoE accumulator window: windows={48*((n+81919)//81920)} dequants=2000 jobs=10000 source_H2D_bytes={48*n*40} GPU_resident_windows_only=1 operator_max=8192 maximum_rows={maximum}\n'
        f'strata MoE mixed banks: storage=scratch primary_bytes={min(maximum,65536)*5120} tail_bytes={max(0,maximum-65536)*5120} tail_offset=900000000 extra_allocation_bytes=0 phase=FFN_only_after_attention_drain\n'
        'strata prefill expert source: requested_view=1 effective_view=1 layers=48 owned_slab_bytes=0 borrowed_layer_span_bytes=50292326400 host_copy=eliminated source_pin=unspecified source_owner=caller_through_checked_retirement\n'
        f'strata MoE expert bank complete: total_slots={slots} permanent_slots=0 requested_H2D_bytes=44000000000 permanent_hits=0 uploads=2000 final_drain=checked\n'
        'strata prefill: QSA tensor IO=BF16, indexer/GDN internals FP32;\n'
        'strata prefill: QSA BF16 operands, FP32 online state/accumulation; '
        f'KV=active-logical-prefix transient_bytes={prefix+256}; source=resident-Mode0 score_pitch=33 operand_layout=row-major softmax={campaign.SOFTMAX[arm]}\n'
        'strata MoE GU fusion: fused_jobs=2 fused_rows=5200 separate_jobs=9998 min_rows=2334\n'
        'strata MoE Down fusion: fused_jobs=2 fused_rows=5200 separate_jobs=9998 min_rows=2334\n')
    for phase in ('before-detach','after-detach','after-prompt-before-restore','after-restore'):
        text+=f'strata mtp: prompt expert phase={phase} bytes=707788800\n'
    text+='strata mtp: experts restored bytes=707788800, unchanged=1\n'
    if trace:
        for layer in range(48):
            text+=f'strata packed expert bank: layer={layer} slots={slots} stride=2662400 used_bytes={cache} backing_bytes={cache} full_fit=0\n'
    return text,dict(read=read,arm=arm,heads=1,trace_progress=trace)


class ComparisonChecks(unittest.TestCase):
    def test_actual_math_and_fixed_workspace(self):
        for arm in (0, 1):
            for read, traced in ((32769, True), (131072, False)):
                text, plan = fixture(arm, read, traced)
                result = campaign.request_route(text, plan)
                self.assertEqual(result['actual_attention_math'], campaign.SELECTORS[arm])
                self.assertEqual(result['bank']['residual_host_rows'], 0)
                self.assertEqual(result['bank']['windows'], 48 if read == 32769 else 96)

    def test_refuses_math_drift_and_workspace_fallback(self):
        text, plan = fixture(1)
        for before, after in (
            ('softmax=subgroup', 'softmax=parallel-exp-ascending-sum'),
            ('score_pitch=33', 'score_pitch=32'),
            ('effective_slots=256', 'effective_slots=128'),
            ('effective_mode=10', 'effective_mode=16'),
            ('free_gpu_bytes=4245643264', 'free_gpu_bytes=0'),
            ('allocation_limit_bytes=10132807680', 'allocation_limit_bytes=1'),
            ('uploads=2000', 'uploads=2001'),
            ('residual_host=0', 'residual_host=1'),
            ('experts restored bytes=707788800', 'experts restored bytes=1'),
        ):
            with self.subTest(before=before), self.assertRaises(RuntimeError):
                campaign.request_route(text.replace(before, after), plan)

    def test_closure_and_attribution(self):
        processes = []
        for i, (read, arm) in enumerate(zip(campaign.READS, campaign.ARMS)):
            text, plan = fixture(arm, read, i < 3)
            route = campaign.request_route(text, plan)
            duration = 99000 if arm == 1 else 100000
            sample = dict(prompt_ms=duration, ttft_s=duration/1000 + .2)
            processes.append(dict(index=i, read=read, clean_speed=i >= 3, passed=True,
                complete=True, ownership_closed=True, native_exit_code=0, fresh_faults=[],
                native_identity=dict(pid=i+5, start_ticks=200+i), samples=[sample]*3,
                routes=[route]*3, subsequent_median_ms=duration))
        self.assertTrue(campaign.decision(processes)['isolated_reduction_attribution_eligible'])
        processes[4]['routes'][0]['bank']['jobs'] += 1
        different_work = campaign.decision(processes)
        self.assertFalse(different_work['isolated_reduction_attribution_eligible'])
        self.assertTrue(different_work['repeated_warmed_gain'])
        processes[4]['subsequent_median_ms'] = 100100
        self.assertFalse(campaign.decision(processes)['eligible_for_next_model_gates'])
        processes[4]['native_identity'] = processes[3]['native_identity']
        with self.assertRaises(RuntimeError):
            campaign.decision(processes)


if __name__ == '__main__':
    unittest.main()
