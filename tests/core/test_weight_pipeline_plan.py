"""Host-only admission tests for the independently controlled prefill fixture.

Main-authored; repository LICENSE. No GPU/runtime/process inspection.
"""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/sycl'))
import prefill_weight_pipeline as t


def case(family='math',arm='both64',mode='batch',width=1):
    value=dict(family=family,arm=arm,mode=mode,rows=128,trace=False,pair=0,
        environment=dict(t.BACKEND|t.SAFETY))
    if family=='lifetime':
        value.update(arm='lifetime',rows=0,trace=True,pair=None,width=width,placement='separate',
            argv=[str(t.ROOT/'lifetime'),mode,'separate',str(width)])
    else:value['argv']=[str(t.ROOT/'benchmark'),mode,arm,'128','weights']
    return value


def batch(arm='both64',rows=128):
    adapter,_,_,_,ready,readers=t.POLICIES[arm]
    return dict(kind='batch',arm=arm,rows=rows,epochs=512,requested_H2D_bytes=1009254400,
        copies=512,gu_fused=512 if rows>=2334 else 0,down_fused=512 if rows>=2334 else 0,
        split_primary_rows=rows//2,actual_source_experts=512,source_footprint_bytes=1009254400,
        production_gemm_dispatch=True,split_tail_checked=True,full_bitwise_reference_and_guards=True,
        copy_compat_calls=512 if arm=='direct' else 0,
        copy_shortcut_calls=512 if arm in ['shortcut','layout'] else 0,
        copy_adapter_calls=512 if adapter else 0,ready_barrier_calls=512//ready if adapter else 0,
        reader_barrier_calls=512//readers if adapter else 0,wall_s=1.,timer_pair_max_s=1e-7)


class Admission(unittest.TestCase):
    def test_policy_and_independent_order(self):
        self.assertEqual(len(t.ARMS),9)
        for block in t.BLOCKS:self.assertEqual(set(block),set(t.ARMS));self.assertEqual(len(block),9)
        self.assertEqual(t.ROWS,(128,1024,8192))

    def test_minimum_unit_deltas(self):
        for before,after,indices in [('shortcut','layout',{2}),('layout','bridge',{0}),
            ('bridge','queue',{1}),('queue','window',{3}),('window','ready64',{4}),
            ('window','readers64',{5}),('ready64','both64',{5}),('readers64','both64',{4})]:
            with self.subTest(before=before,after=after):
                self.assertEqual({i for i,(a,b) in enumerate(zip(t.POLICIES[before],t.POLICIES[after])) if a!=b},indices)

    def test_fixed_layout_factorial(self):
        policies=[t.POLICIES[k] for k in ['window','ready64','readers64','both64']]
        self.assertEqual({v[:4] for v in policies},{(True,True,128,128)})
        self.assertEqual({v[4:] for v in policies},{(1,1),(64,1),(1,64),(64,64)})

    def test_native_arguments_are_admitted_exactly(self):
        for arm in t.ARMS:t.case_gate(case(arm=arm),'weights')
        wrong=case();wrong['argv'][-1]='other-weights'
        with self.assertRaises(RuntimeError):t.case_gate(wrong,'weights')
        wrong=case();wrong['rows']=4096
        with self.assertRaises(RuntimeError):t.case_gate(wrong,'weights')

    def test_safety_and_backend_cannot_drift(self):
        for name in (t.SAFETY|t.BACKEND):
            wrong=case();wrong['environment'][name]='unqualified'
            with self.subTest(name=name),self.assertRaises(RuntimeError):t.case_gate(wrong,'weights')

    def test_finite_lifetime_arguments(self):
        for mode in ['metadata','ordered','throw-copy','throw-reader','grain1','ready64','readers64','both64']:
            t.case_gate(case('lifetime',mode=mode),'weights')
        wrong=case('lifetime',mode='both64',width=64)
        with self.assertRaises(RuntimeError):t.case_gate(wrong,'weights')

    def test_actual_source_proof_not_just_the_policy_label(self):
        for arm in t.ARMS:
            adapter,separate,slots,window,ready,readers=t.POLICIES[arm]
            proof=dict(kind='source-proof',ordinary_readonly_mapping=True,host_slots=0,policy=arm,
                dependency_bridge=adapter,separate_queue=separate,physical_slots=slots,prefetch_window=window,
                ready_width=ready,readers_width=readers,reader_marker='after-math' if adapter else 'none',
                runtime_check_and_observer_retained=True)
            t.source_proof_gate(proof,arm);proof['physical_slots']=2
            with self.assertRaises(RuntimeError):t.source_proof_gate(proof,arm)

    def test_measured_dispatch_counts_are_required(self):
        for arm in t.ARMS:
            t.batch_gate(batch(arm),arm,128)
        for field in ['copy_compat_calls','copy_shortcut_calls','copy_adapter_calls',
                      'ready_barrier_calls','reader_barrier_calls']:
            value=batch();value[field]+=1
            with self.subTest(field=field),self.assertRaises(RuntimeError):t.batch_gate(value,'both64',128)

    def test_timer_and_full_math_eligibility(self):
        for field,value in [('wall_s',.099),('timer_pair_max_s',.002),('full_bitwise_reference_and_guards',False),
            ('split_tail_checked',False),('gu_fused',512),('copies',511),('actual_source_experts',2)]:
            wrong=batch();wrong[field]=value
            with self.subTest(field=field),self.assertRaises(RuntimeError):t.batch_gate(wrong,'both64',128)

    def test_lifetime_census_includes_matching_sized_readbacks(self):
        self.assertEqual(t.lifetime_counts('ordered',1),(15,5))
        self.assertEqual(t.lifetime_counts('ordered',64),(582,194))
        self.assertEqual(t.lifetime_counts('throw-copy',1),(0,2))
        self.assertEqual(t.lifetime_counts('throw-reader',64),(192,65))
        self.assertEqual(t.lifetime_counts('both64',1),(768,256))
        self.assertEqual(t.lifetime_done('both64',1)['readers_width'],64)

    def test_separately_pinned_fixture_provider(self):
        row=dict(binary={'path':'math','sha256':'math-hash'},lifetime_binary={'path':'life','sha256':'life-hash'})
        with patch.object(t.base.event,'mappings',return_value={'ok':True}) as mappings:
            t.case_mappings(123,row,dict(family='lifetime',argv=['life']))
            self.assertEqual(mappings.call_args.args[1]['binary'],row['lifetime_binary'])
            with self.assertRaises(RuntimeError):t.case_mappings(123,row,dict(family='lifetime',argv=['math']))

    def test_faster_variant_survives_failed_combination(self):
        samples=[]
        for rows in t.ROWS:
            for pair in range(3):
                for arm in t.ARMS:
                    time=.8 if arm=='ready64' else 1.2 if arm=='both64' else 1.
                    samples.append(dict(family='math',clean=True,rows=rows,pair=pair,arm=arm,
                        batch={'wall_s':time},lifecycle_plus_batch_s=time+.01))
        a=t.analyze(samples)
        for rows in t.ROWS:
            self.assertTrue(a[str(rows)]['netReady']['repeated_component_gain'])
            self.assertFalse(a[str(rows)]['netBoth']['repeated_component_gain'])
            self.assertEqual(len(a[str(rows)]['interaction']['paired_multiplicative_time_ratios']),3)

    def test_exact_historical_seed_and_metadata(self):
        seed=dict(stage='closed',passed=True,complete=True,active=False,ownership_closed=True,
            native_source_commit=t.SEED_SOURCE,project={'revision':t.SEED_SOURCE},
            original_math_runtime_source=t.base.original.SOURCE)
        for key in ['native_validation_probe','runtime_manifest','validation_provider','driver','boot','weights']:seed[key]={'present':True}
        with patch.object(t,'git',return_value=json.dumps({'prefill_weight_pipeline_v2':seed})):
            self.assertEqual(t.seed_recipe(),seed)
        seed['native_source_commit']='wrong'
        with patch.object(t,'git',return_value=json.dumps({'prefill_weight_pipeline_v2':seed})),self.assertRaises(RuntimeError):t.seed_recipe()


if __name__=='__main__':unittest.main()
