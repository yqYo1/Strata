"""Host-only admission-reader controls; Main, repository LICENSE."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parent/'sycl'))
import coherent_context_model as c


class ContextReader(unittest.TestCase):
    def fixture(self, full=False):
        n,gpu=(262143,81920) if full else (16384,16384)
        slots=512; bank=slots*2662400; limit=2<<30
        row=dict(vocab_size=8192,native_validation_probe=dict(loader=dict(path='/pinned/loader')))
        case=dict(name='full' if full else 'short',tokens=n+1,outputs=0 if full else 4)
        err=('strata UR validation: pid=7 phase=before-main checked=46\n'
            f'strata prefill: layer-major candidate, tokens={n} chunk=8192 residual_dtype=bf16 '
            f'residual_element_bytes=2 residual_total={n*20480} residual_gpu_rows={gpu} '
            f'residual_gpu={gpu*20480} residual_host={(n-gpu)*20480} residual_staging=0 layer_cache={bank};\n'
            f'strata MoE expert bank: requested_slots=384 effective_slots={slots} staged=0 '
            f'host_slab_bytes=1363148800 device_bytes={bank} residual_gpu_rows={gpu} allocation_limit_bytes={limit}\n'
            'strata MoE window configuration: requested_mode=10 effective_mode=2 maximum_rows=16384 mixed_fp16_bytes=0\n'
            'strata prefill: QSA BF16 operands, FP32 online state/accumulation; '
            f'KV=active-logical-prefix transient_bytes={((n+15)//16)*16*2056}; source=resident-Mode0 score_pitch=33 '
            'operand_layout=row-major softmax=parallel-exp-ascending-sum\n'
            'strata prefill: native16 BF16 HC residual/projection IO and MoE outputs;\n'
            'strata trace: 1500 MiB free after releasing the standalone prompt buffers\n')
        out=f'prefill                  {n} tokens in 123.4 ms\noutput                  :'+('' if full else ' 1 2 3 4')+'\n'
        if full:out+=c.MARKER+'\n'
        return out,err,row,case,7

    def read(self,args):
        with patch.object(c.validation,'preload_evidence',return_value=dict(loader='/pinned/loader')):
            return c.evidence(*args)

    def test_full_host_tail_is_capacity_not_residency(self):
        proof=self.read(self.fixture(True))
        self.assertTrue(proof['final_slot_consumed']);self.assertGreater(proof['allocation']['host_bytes'],0)
        self.assertFalse(proof['speed_eligible']);self.assertEqual(proof['actual_batched_rows'],262143)

    def test_short_does_not_claim_full_fill(self):
        proof=self.read(self.fixture());self.assertFalse(proof['final_slot_consumed']);self.assertEqual(proof['generated_ids'],4)

    def test_reject_missing_or_inexact_final_commit(self):
        for marker in ('',c.MARKER.replace('consumed_position=262144','consumed_position=262143'),c.MARKER+'\n'+c.MARKER):
            args=list(self.fixture(True));args[0]=args[0].replace(c.MARKER,marker)
            with self.assertRaises(RuntimeError):self.read(args)

    def test_reject_corrupt_or_duplicate_allocation_evidence(self):
        for old,new in [('residual_host=0','residual_host=1'),('chunk=8192','chunk=4096'),
            ('transient_bytes=33685504','transient_bytes=0'),('allocation_limit_bytes=2147483648','allocation_limit_bytes=1024')]:
            args=list(self.fixture());self.assertIn(old,args[1]);args[1]=args[1].replace(old,new)
            with self.assertRaises(RuntimeError):self.read(args)
        args=list(self.fixture());args[1]+=next(s+'\n' for s in args[1].splitlines() if 'layer-major candidate' in s)
        with self.assertRaises(RuntimeError):self.read(args)

    def test_reject_runtime_error_and_wrong_outputs(self):
        args=list(self.fixture());args[1]+='UR_RESULT_ERROR_OUT_OF_RESOURCES\n'
        with self.assertRaises(RuntimeError):self.read(args)
        args=list(self.fixture());args[0]=args[0].replace('1 2 3 4','1 2 3 9000')
        with self.assertRaises(RuntimeError):self.read(args)

    def test_window_storage_and_numeric_release_are_mandatory(self):
        for old,new in [('maximum_rows=16384','maximum_rows=8192'),
            ('effective_mode=2','effective_mode=10'),('mixed_fp16_bytes=0','mixed_fp16_bytes=1'),
            ('strata trace: 1500 MiB free after releasing','strata trace: unknown MiB free after releasing')]:
            args=list(self.fixture());args[1]=args[1].replace(old,new)
            with self.assertRaises(RuntimeError):self.read(args)
        args=list(self.fixture());args[1]=args[1].replace('effective_mode=2 maximum_rows=16384 mixed_fp16_bytes=0',
            'effective_mode=10 maximum_rows=16384 mixed_fp16_bytes=83886080')
        self.assertEqual(self.read(args)['window']['effective_mode'],10)

    def test_workload_generator_and_extent_admission(self):
        raw=c.token_bytes(262144).splitlines();self.assertEqual(len(raw),262144);self.assertEqual(raw[-1],b'4095')
        for invalid in (True,262143,262145,0):
            with self.assertRaises(RuntimeError):c.token_bytes(invalid)

    def test_command_preserves_decode_layout_and_exact_full_entry(self):
        args=['--max-context','262144','--kv-resident','20480','--expert-cache','1256',
            '--spec','4','--suffix-draft','0','--prefill','8192']
        row=dict(configuration=dict(args=args),binary=dict(path='/exact/diagnostic'))
        case=dict(name='full',outputs=0,fixture=dict(path='/exact/tokens'))
        actual=c.command(row,case);self.assertEqual(actual[-1],'--consume-prompt');self.assertEqual(actual[1:1+len(args)],args)
        bad=copy.deepcopy(row);bad['configuration']['args'][3]='131072'
        with self.assertRaises(RuntimeError):c.command(bad,case)

    def test_standalone_allocation_dependencies_and_safety_settings(self):
        template=dict(STRATA_MEASUREMENT_ROOT='/old',EnableDirectSubmission='0',
            UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1',STRATA_SYCL_PREFILL_SAME_QUEUE='1',
            STRATA_SYCL_PREFILL_EXPERT_BANK_SLOTS='384',
            STRATA_SYCL_PREFILL_QSA_MATH='bf16-prefix-ordered-exp',
            STRATA_SYCL_CACHE_SCRATCH_LEASE='1',STRATA_SYCL_PREFILL_MIXED_SCRATCH='1',
            STRATA_SYCL_PREFILL_MOE_SPLIT_ACCUMULATOR='1',STRATA_SYCL_PREFILL_MOE_WINDOW='10')
        before=copy.deepcopy(template)
        row=dict(environment_template=template,native_validation_probe=dict(
            library=dict(path='/exact/preload'),loader=dict(path='/exact/loader')))
        env=c.environment(row,dict(root='/new'))
        for name in ('STRATA_SYCL_CACHE_SCRATCH_LEASE','STRATA_SYCL_PREFILL_MIXED_SCRATCH',
            'STRATA_SYCL_PREFILL_MOE_SPLIT_ACCUMULATOR'):self.assertEqual(env[name],'0')
        for name in ('EnableDirectSubmission','UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD',
            'STRATA_SYCL_PREFILL_SAME_QUEUE','STRATA_SYCL_PREFILL_MOE_WINDOW'):
            self.assertEqual(env[name],template[name])
        self.assertEqual(template,before)


if __name__=='__main__':unittest.main()
