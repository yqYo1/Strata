"""CPU-only controls for current dense-DPAS leaf cost attribution."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parent/'sycl'))
import profile_q6_cpu_leaves as profile
from tools.test_adaptive_cpu_leaf_evidence import fixture


def routes(prompt=False):
    text=[];expected={}
    phases=[('decode',1),('decode',4)]+([('prompt',4)] if prompt else [])
    for phase,tokens in phases:
        identity=f'request=5 device=0 phase={phase} tokens={tokens} capture_id=9'
        text += [f'strata Q6_ROUTE {identity} scope=live_capture_cumulative native_half_enabled=0 complete=1 shapes=1 replay_started=7 replay_completed=7',
            f'strata Q6_PACKED_HEAD {identity} layout=sole-packed extra_weight_bytes=0 calls=1 complete=1',
            f'strata Q6_SHAPE {identity} ni=2560 no=512 nc={tokens} reference_calls=21 columns_calls=0 centered_calls=0 native_half_calls=0']
        expected[f'{phase}/{tokens}']=profile.signature({f'2560x512x{tokens}':[21,0,0,0]})
        for (ni,no),calls in zip(profile.dense.model.SHAPES,(40,14,22,5)):
            count=calls if (phase,tokens)==('decode',4) else 0
            text.append(f'strata Q6_DPAS_DENSE {identity} ni={ni} no={no} nc=4 layout=canonical extra_weight_bytes=0 canonical_table_excludes_dpas=1 calls={count} complete=1')
    return '\n'.join(text),expected


class Q6LeafControls(unittest.TestCase):
    def test_actual_dense_join_accepts_optional_verifier_prompt(self):
        for prompt in (False,True):
            text,expected=routes(prompt)
            row=profile.dense_route(text,expected)
            self.assertEqual(row['decode_C4'],dict(capture_id=9,replay_completed=7))
            self.assertEqual(row['selected_C4_calls'],dict(zip(('6144x2560','2560x6144','2560x10240','2560x12288'),(40,14,22,5))))

    def test_changed_or_incomplete_actual_dense_routes_refused(self):
        text,expected=routes()
        marker=next(v for v in text.splitlines() if v.startswith('strata Q6_DPAS_DENSE '))
        bad=[text.replace(marker,'',1),text+'\n'+marker,
            text.replace('canonical_table_excludes_dpas=1','canonical_table_excludes_dpas=0',1),
            text.replace('calls=40','calls=39',1),text.replace('calls=0','calls=1',1),
            text.replace('reference_calls=21','reference_calls=20',1),
            text.replace('request=5','request=6'),
            text.replace('replay_started=7 replay_completed=7','replay_started=0 replay_completed=0'),
            text.replace('extra_weight_bytes=0','extra_weight_bytes=1',1),
            text.replace('shapes=1','shapes=2',1),
            text+'\n'+marker.replace('capture_id=9','capture_id=10')]
        for value in bad:
            with self.subTest(value=value[-160:]),self.assertRaises(RuntimeError):
                profile.dense_route(value,expected)
        with self.assertRaises(RuntimeError):profile.dense_route(text,dict(expected,**{'decode/1':'changed'}))

    def test_focused_contract_keeps_full_thinking_and_answer_eligibility(self):
        generation=dict(workload_version=5,thinking=True,reasoning_budget_tokens=0,max_tokens=16384,
            diagnostic_subset=True,selected_workloads=['regular-code'])
        rows=[dict(workload='regular-code',run=0,order=0,controller_eligible=True)]
        profile.contract(generation,rows)
        for k,v in (('thinking',False),('max_tokens',1024),('reasoning_budget_tokens',1),
                    ('diagnostic_subset',False),('workload_version',4),('selected_workloads',['varied-prose'])):
            with self.subTest(field=k),self.assertRaises(RuntimeError):profile.contract(dict(generation,**{k:v}),rows)
        for k,v in (('workload','varied-prose'),('run',1),('order',1),('controller_eligible',False)):
            with self.subTest(field=k),self.assertRaises(RuntimeError):profile.contract(generation,[dict(rows[0],**{k:v})])

    def test_only_leaf_startup_flag_differs(self):
        base=dict(STRATA_SYCL_DECODE_Q6_DPAS_HEAD='2',STRATA_PREFILL_HEAD_HOST_BACKING='0',
            UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1',EnableDirectSubmission='0',ZES_ENABLE_SYSMAN='0')
        with patch.object(profile.dense,'environment',side_effect=lambda *_:dict(base)):
            off,on=(profile.environment(Path('/private/0'),enabled) for enabled in (False,True))
        self.assertEqual({k for k in off if off[k]!=on[k]},{'STRATA_CPU_SERVICE'})
        self.assertEqual(on['STRATA_SYCL_DECODE_Q6_DPAS_DENSE'],'1')
        self.assertEqual(on['STRATA_CPU_WORK_CENSUS'],'0')
        self.assertEqual(base,dict(STRATA_SYCL_DECODE_Q6_DPAS_HEAD='2',STRATA_PREFILL_HEAD_HOST_BACKING='0',
            UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1',EnableDirectSubmission='0',ZES_ENABLE_SYSMAN='0'))

    def test_invariant_normalizes_only_private_roots_and_leaf_flag(self):
        a=dict(log='/private/0/engine.stderr',args=['--adapt-swaps','96'])
        b=dict(log='/private/1/engine.stderr',args=['--adapt-swaps','96'])
        x=profile.invariant(a,dict(STRATA_CPU_SERVICE='0'),'/private/0')
        self.assertEqual(x,profile.invariant(b,dict(STRATA_CPU_SERVICE='1'),'/private/1'))
        self.assertNotEqual(x,profile.invariant(dict(b,args=['--adapt-swaps','32']),dict(STRATA_CPU_SERVICE='1'),'/private/1'))

    def test_leaf_reader_presence_and_operation_conservation(self):
        text,sample=fixture();quiet='\n'.join(v for v in text.splitlines() if not v.startswith('strata CPU_SERVICE'))
        a,parsed=profile.leaf.evidence(text,[sample],4,True)
        b,off=profile.leaf.evidence(quiet,[sample],4,False)
        self.assertEqual(a,b);self.assertEqual(off,[])
        summary=profile.leaf.summarize(parsed[0])
        self.assertEqual(sum(v['cpu_ns'] for v in summary['operations'].values()),summary['concurrent_cpu_ns'])
        for body,enabled in ((text,False),(quiet,True)):
            with self.assertRaises(RuntimeError):profile.leaf.evidence(body,[sample],4,enabled)

    def test_observer_cost_keeps_failed_pairs_and_work_mismatch(self):
        rows=[dict(kernel=arm,adapt_swaps=96,passed=True,complete=True,ownership_closed=True,native_exit_code=0,
            fresh_faults=[],native_identity=dict(pid=i+1,start_ticks=i+1),
            samples=[dict(generated_tokens=2048,decode_ms=1040 if arm=='candidate' else 1000)],
            resources=dict(passed=True,completed_requests=5,warmup_prefix=4,leaf_enabled=arm=='candidate',work_signature_sha256='same'))
            for i,arm in enumerate(profile.ORDER)]
        self.assertTrue(profile.leaf.decision(rows)['leaf_observer_cost_qualified'])
        for change in ('slow','work','owner'):
            bad=copy.deepcopy(rows)
            if change=='owner':
                bad[1]['native_identity']=bad[0]['native_identity']
                with self.assertRaises(RuntimeError):profile.leaf.decision(bad)
            else:
                if change=='slow':bad[1]['samples'][0]['decode_ms']=1051
                else:bad[1]['resources']['work_signature_sha256']='changed'
                self.assertFalse(profile.leaf.decision(bad)['leaf_observer_cost_qualified'])


if __name__=='__main__':unittest.main()
