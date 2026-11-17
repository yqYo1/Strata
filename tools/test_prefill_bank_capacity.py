"""Reject unplanned bank/placement/state changes before real model timing."""
import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent/'sycl'))
import prefill_bank_capacity as bank


class BankCapacityTests(unittest.TestCase):
    def text(self,slots,traced=False):
        rows=65536;size=slots*bank.STRIDE
        lines=[
            'strata prefill heads: target detached bytes=521472000 source=owned-RAM D2H=521472000 H2D-on-restore=521472000',
            'strata prefill heads: draft detached bytes=0 retained-map-bytes=425196',
            f'strata MoE expert bank: requested_slots={slots} effective_slots={slots} residual_gpu_rows={rows} device_bytes={size} allocation_limit_bytes=2147483648 general_map=private_scratch_never_published staged=1',
            f'strata prefill: layer-major candidate, tokens={rows} chunk=8192 residual_dtype=bf16 residual_element_bytes=2 residual_gpu_rows={rows} residual_total={rows*20480} residual_gpu={rows*20480} residual_host=0 residual_staging=0 layer_cache={size}',
            'QSA tensor IO=BF16, indexer/GDN internals FP32;',
            'softmax=parallel-exp-ascending-sum',
            f'strata MoE split admission: active=0 rows={rows} primary_rows=0 tail_payload_bytes=0 extra_allocation_bytes=0 prefill_slots={slots} decode_changed=0',
            f'strata MoE window configuration: requested_mode=10 effective_mode=10 maximum_rows={rows}',
            f'strata MoE accumulator window: windows=48 dequants=64 jobs=96 source_H2D_bytes=1024 GPU_resident_windows_only=1 operator_max=8192 maximum_rows={rows}',
            f'strata MoE expert bank complete: uploads=64 requested_H2D_bytes=1024 permanent_slots={slots} total_slots={slots} final_drain=checked',
            'strata MoE GU fusion: fused_jobs=4 fused_rows=10000 separate_jobs=92 min_rows=2334',
            'strata MoE Down fusion: fused_jobs=4 fused_rows=10000 separate_jobs=92 min_rows=2334',
            *['strata mtp: prompt expert phase='+p+' checked=1' for p in
                ['before-detach','after-detach','after-prompt-before-restore','after-restore']],
            'strata mtp: experts restored bytes=707788800, checked=1']
        if traced:
            lines += [f'strata packed expert bank: layer={i} slots=512 stride=1971200 used_bytes=1009254400 backing_bytes={size} full_fit=1' for i in range(48)]
        return '\n'.join(lines)+'\n'

    def test_actual_route_requires_selected_slots_and_preserved_state(self):
        for slots in (384,512):
            plan=dict(read=65537,slots=slots,heads=1,trace_progress=False)
            text=self.text(slots)
            self.assertEqual(bank.request_route(text,plan)['bank']['prefill_slots'],slots)
            for old,new in [('requested_slots='+str(slots),'requested_slots=128'),
                ('residual_host=0','residual_host=20480'),
                ('decode_changed=0','decode_changed=1'),
                ('restored bytes=707788800','restored bytes=0'),
                ('final_drain=checked','final_drain=unchecked')]:
                with self.subTest(slots=slots,field=old),self.assertRaises(RuntimeError):
                    bank.request_route(text.replace(old,new),plan)

    def test_traced_admission_needs_every_layer_and_actual_allocation(self):
        plan=dict(read=65537,slots=512,heads=1,trace_progress=True)
        text=self.text(512,True)
        self.assertEqual(bank.request_route(text,plan)['actual_full_fit_layers'],48)
        for changed in [text.replace('layer=47 slots=512','layer=46 slots=512'),
            text.replace('full_fit=1','full_fit=0',1),
            text.replace('layer_cache=1363148800','layer_cache=1022361600'),
            text.replace('backing_bytes=1363148800','backing_bytes=1022361600',1)]:
            with self.assertRaises(RuntimeError):bank.request_route(changed,plan)

    def test_comparison_changes_only_capacity_in_preexec_environment(self):
        env=dict(STRATA_MEASUREMENT_ROOT='/original',EnableDirectSubmission='0',
            UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1',STRATA_SYCL_PREFILL_SAME_QUEUE='1',
            STRATA_SYCL_PREFILL_EXPERT_BANK_PACKED='1',STRATA_SYCL_PREFILL_EXPERT_UPLOAD_STAGING='0')
        row=dict(configuration=dict(exe='engine_once.py',args=['--expert-cache','1256']),environment_template=env)
        before=copy.deepcopy(row)
        plans=[dict(root='/process/'+str(s),slots=s,heads=1,trace_progress=False) for s in (384,512)]
        a,b=[bank.environment(row,p) for p in plans]
        for p,e in zip(plans,(a,b)):e['STRATA_MEASUREMENT_ROOT']='<root>'
        self.assertEqual({k for k in a if a[k]!=b[k]},{'STRATA_SYCL_PREFILL_EXPERT_BANK_SLOTS'})
        self.assertEqual(bank.invariant(row,plans[0]),bank.invariant(row,plans[1]))
        changed=copy.deepcopy(row);changed['environment_template']['STRATA_SYCL_PREFILL_SAME_QUEUE']='0'
        with self.assertRaises(RuntimeError):bank.environment(changed,plans[0])
        changed=copy.deepcopy(row);changed['configuration']['args']+=['--prefill','4096']
        self.assertNotEqual(bank.invariant(changed,plans[0]),bank.invariant(row,plans[0]))
        self.assertEqual(row,before)

    def test_decision_rejects_incomplete_or_failed_processes(self):
        for values in [[],[dict(passed=True)]*13,[dict(passed=False,complete=False)]*13]:
            with self.assertRaises((RuntimeError,KeyError)):bank.decision(values)


if __name__=='__main__':unittest.main()
