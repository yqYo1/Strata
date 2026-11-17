"""Negative host admission and independent-unit controls; no native execution.

Main-authored; repository LICENSE. Old13controls restore from05eceacd.
"""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/sycl'))
import prefill_weight_pipeline as t


def cases():
    return [dict(label=label,root=str(t.ROOT/label),family='math',mode=mode,arm=arm,
        rows=rows,trace=trace,pair=pair,environment=dict(t.SAFETY|t.BACKEND),
        argv=[str(t.ROOT/'benchmark'),mode,arm,str(rows),'weights'])
        for label,mode,arm,rows,trace,pair in t.expected_cases()]


def row():
    value={k:{} for k in ['foundation','source_view','recipe_tools','runtime_manifest',
        'native_validation_probe','validation_provider','build_environment','driver',
        'phase_references','historical_safety','plan']}
    value.update(project={'source_sha256':{'source':'pinned'},'revision':'original'},
        native_source_commit='original',allowed={'source':'pinned'},recipe=[],
        cases=cases(),weights={'path':'weights'},stage='prepared',passed=False,complete=False,
        active=False,ownership_closed=True,commands=[],results=[],model_trial_allowed=False,
        performance_qualified=False,boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    return value


def batch(arm='unique512',rows=128):
    return dict(kind='batch',arm=arm,rows=rows,epochs=512,requested_H2D_bytes=1009254400,
        copies=512,gu_fused=512 if rows>=2334 else 0,down_fused=512 if rows>=2334 else 0,
        split_primary_rows=rows//2,actual_source_experts=512,source_footprint_bytes=1009254400,
        copy_adapter_calls=512,ready_barrier_calls=8,reader_barrier_calls=0 if arm=='unique512' else 8,
        single_write_copy_calls=512 if arm=='unique512' else 0,timed_initial_slot=0,
        production_gemm_dispatch=True,split_tail_checked=True,full_bitwise_reference_and_guards=True,
        warm_owner_retired=True,wall_s=1.,timer_pair_max_s=1e-7)


class Admission(unittest.TestCase):
    def test_minimum_unit_deltas_and_combined_contrast(self):
        self.assertEqual(t.POLICIES['base128'],(False,128))
        self.assertEqual(t.POLICIES['base512'],(False,512))
        self.assertEqual(t.POLICIES['unique512'],(True,512))
        self.assertEqual(t.UNITS,{'S':('base128','base512'),'U':('base512','unique512'),
                                 'netUnique':('base128','unique512')})

    def test_complete_three_block_plan_and_frozen_cases(self):
        r=row();t.plan_gate(r);self.assertEqual(len(r['cases']),39)
        for block in t.BLOCKS:self.assertEqual(set(block),set(t.ARMS))
        for changed in [r['cases'][:-1],list(reversed(r['cases'])),r['cases']+r['cases'][:1]]:
            with self.assertRaises(RuntimeError):t.plan_gate(dict(r,cases=changed))
        wrong=copy.deepcopy(cases()[0]);wrong['argv'][-1]='different'
        with self.assertRaises(RuntimeError):t.case_gate(wrong,'weights')
        for key in t.SAFETY|t.BACKEND:
            wrong=copy.deepcopy(cases()[0]);wrong['environment'][key]='changed'
            with self.subTest(key=key),self.assertRaises(RuntimeError):t.case_gate(wrong,'weights')

    def test_real_source_allocation_and_readiness_contract(self):
        for arm in t.ARMS:
            unique,slots=t.POLICIES[arm]
            p=dict(kind='source-proof',ordinary_readonly_mapping=True,host_slots=0,policy=arm,
                dependency_bridge=True,separate_queue=True,single_write_destinations=unique,
                physical_slots=slots,prefetch_window=128,ready_width=64,refill_width=64,
                reader_marker='none-single-write' if unique else 'after-math',
                admitted_bank_allocation_bytes=slots*t.PAYLOAD+512,runtime_check_and_observer_retained=True)
            t.source_proof_gate(p,arm)
            for key,value in [('admitted_bank_allocation_bytes',0),('physical_slots',1),
                ('separate_queue',False),('ready_width',1),('single_write_destinations',not unique)]:
                with self.subTest(arm=arm,key=key),self.assertRaises(RuntimeError):t.source_proof_gate(dict(p,**{key:value}),arm)

    def test_actual_counts_math_and_warm_owner_reset(self):
        for m in t.ROWS:
            for arm in t.ARMS:t.batch_gate(batch(arm,m),arm,m)
        for key,value in [('copies',511),('requested_H2D_bytes',1),('single_write_copy_calls',0),
            ('reader_barrier_calls',8),('timed_initial_slot',2),('warm_owner_retired',False),
            ('gu_fused',512),('split_tail_checked',False),('copies',True)]:
            with self.subTest(key=key),self.assertRaises(RuntimeError):t.batch_gate(dict(batch(),**{key:value}),'unique512',128)

    def test_timing_eligibility_and_no_model_promotion(self):
        for key,value in [('wall_s',.099),('timer_pair_max_s',.002),('wall_s',float('nan'))]:
            with self.subTest(key=key),self.assertRaises(RuntimeError):t.batch_gate(dict(batch(),**{key:value}),'unique512',128)
        for key in ['model_trial_allowed','performance_qualified']:
            with self.assertRaises(RuntimeError):t.plan_gate(dict(row(),**{key:True}))

    def test_individual_unit_not_rejected_by_losing_combination(self):
        samples=[]
        for m in t.ROWS:
            for pair in range(3):
                for arm,time in [('base128',1.),('base512',2.),('unique512',1.5)]:
                    samples.append(dict(clean=True,rows=m,pair=pair,arm=arm,
                        batch={'wall_s':time},lifecycle_plus_batch_s=time+.01))
        for value in t.analyze(samples).values():
            self.assertTrue(value['U']['repeated_component_gain'])
            self.assertFalse(value['S']['repeated_component_gain'])
            self.assertFalse(value['netUnique']['repeated_component_gain'])
        samples[-1]['batch']['wall_s']=3.
        self.assertFalse(t.analyze(samples)['8192']['U']['repeated_component_gain'])

    def test_only_outputs_can_change_without_admission_commit(self):
        r=row();original=t.admission(r)
        with patch.object(t.base.ur_validation,'committed',return_value=original):
            t.admission_gate(r)
            output=copy.deepcopy(r);output.update(stage='testing',results=[{'case':'outcome'}])
            t.admission_gate(output)
            for key,value in [('cases',[]),('recipe',[{'new':'recipe'}]),('boot','other'),
                ('allowed',{'source':'different'}),('runtime_manifest',{'provider':'different'}),
                ('weights',{'path':'changed-corpus'})]:
                with self.subTest(key=key),self.assertRaises(RuntimeError):t.admission_gate(dict(r,**{key:value}))
            built=copy.deepcopy(r);built.update(binary={'path':'fixture','sha256':'new'},actual_inputs={'compiled':{'header':'sha'}})
            with self.assertRaises(RuntimeError):t.admission_gate(built)
        with patch.object(t.base.ur_validation,'committed',return_value=t.admission(built)):
            t.admission_gate(built);built['actual_inputs']['compiled']['header']='changed'
            with self.assertRaises(RuntimeError):t.admission_gate(built)

    def test_half_written_output_cannot_advance(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/'public.json';r=row();p.write_text(json.dumps({t.KEY:t.compact(r)}))
            with patch.object(t,'RESULT',p):
                t.output_gate(r);r['results'].append({'unrecorded':'sample'})
                with self.assertRaises(RuntimeError):t.output_gate(r)

    def test_prelaunch_or_owner_setup_failure_is_terminal(self):
        for failed_check in [True,False]:
            with self.subTest(prelaunch=failed_check),tempfile.TemporaryDirectory() as root:
                root=Path(root);record=root/'record.json';public=root/'public.json';r=row()
                record.write_text(json.dumps(r));public.write_text(json.dumps({t.KEY:t.compact(r)}))
                with patch.object(t,'ROOT',root),patch.object(t,'RECORD',record),patch.object(t,'RESULT',public),\
                     patch.object(t,'check',side_effect=RuntimeError('input mismatch') if failed_check else None),\
                     patch.object(t.base.guards,'driver_guard',return_value={}),\
                     patch.object(t.base.owner_api,'Owner',side_effect=RuntimeError('setup failed')) as owner:
                    with self.assertRaises(RuntimeError):t.execute('build')
                    result=json.loads(record.read_text());self.assertEqual(result['stage'],'failed')
                    self.assertFalse(result['active']);self.assertTrue(result['ownership_closed'])
                    self.assertEqual((root/'build.started').exists(),not failed_check)
                    if failed_check:owner.assert_not_called()
                    with self.assertRaisesRegex(RuntimeError,'terminal/wrong-stage'):t.execute('build')


if __name__=='__main__':unittest.main()
