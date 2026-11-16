"""Fail-closed actual-weight controller admission; repository LICENSE."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/sycl'))
import prefill_weight_pipeline as t


class ReachedSource(Exception):pass


class WeightPipelineAdmission(unittest.TestCase):
    def setUp(self):
        self.row=dict(project=dict(revision='a'*40,source_sha256={}),foundation={},allowed={},
            native_validation_probe={'library':'pin'},runtime_manifest={'sha256':'b'*64},
            validation_provider={'sha256':'c'*64},driver={'driver':'xe'},boot='original',
            weights=dict(path='/exact/weights',sha256='d'*64),
            cases=[dict(family='math',arm='direct',mode='batch',rows=16,
                argv=[str(t.ROOT/'benchmark'),'batch','direct','16','/exact/weights'],
                environment={'EnableDirectSubmission':'0',**t.BACKEND})])
        self.public=t.compact(self.row)

    def test_exact_metadata_and_plan_reaches_source_check(self):
        with patch.object(t.base.ur_validation,'committed',return_value=self.public), \
             patch.object(t.base.original.coherent.build.common.source,'pin',side_effect=ReachedSource):
            with self.assertRaises(ReachedSource):t.check(self.row)

    def test_changed_weights_startup_or_typed_state_are_rejected_first(self):
        for which in ['weights','environment','argv','missing_provider']:
            row=copy.deepcopy(self.row)
            if which=='weights':row['weights']['sha256']='e'*64
            elif which=='environment':row['cases'][0]['environment']['EnableDirectSubmission']='1'
            elif which=='argv':row['cases'][0]['argv'][1]='lookahead'
            else:del row['validation_provider']
            with self.subTest(which=which),patch.object(t.base.ur_validation,'committed',return_value=self.public), \
                 patch.object(t.base.original.coherent.build.common.source,'pin') as source:
                with self.assertRaises(RuntimeError):t.check(row)
                source.assert_not_called()

    def test_matching_receipt_cannot_omit_transitive_provider(self):
        row=copy.deepcopy(self.row);del row['validation_provider']
        with patch.object(t.base.ur_validation,'committed',return_value=t.compact(row)), \
             patch.object(t.base.original.coherent.build.common.source,'pin') as source:
            with self.assertRaises(RuntimeError):t.check(row)
            source.assert_not_called()

    def test_terminal_cohort_refuses_before_check_or_receipt_write(self):
        for stage in ['closed','failed']:
            for action in ['build','next']:
                with self.subTest(stage=stage,action=action), \
                     patch.object(t.base.ur_validation,'committed',return_value=dict(stage=stage)), \
                     patch.object(t.Path,'read_text',side_effect=AssertionError('retired checkpoint read')) as read, \
                     patch.object(t,'check',side_effect=RuntimeError('must not requalify')) as check, \
                     patch.object(t,'save') as save:
                    with self.assertRaises(RuntimeError):t.execute(action)
                    read.assert_not_called();check.assert_not_called();save.assert_not_called()


    def test_seed_recipe_recovers_from_git_without_private_checkpoint(self):
        with patch.object(t.Path,'read_text',side_effect=AssertionError('external receipt read')):
            seed=t.seed_recipe()
        self.assertEqual(seed['native_source_commit'],t.SEED_SOURCE)
        self.assertTrue(seed['recipe'])

    def test_seed_recipe_rejects_wrong_source_open_or_missing_provider(self):
        original=t.seed_recipe()
        for field in ['native_source_commit','ownership_closed','active','provider']:
            value=copy.deepcopy(original)
            if field=='native_source_commit':value[field]='f'*40
            elif field=='ownership_closed':value[field]=1
            elif field=='active':value[field]=True
            else:del value['native_validation_probe']
            with self.subTest(field=field),patch.object(t,'git',return_value=json.dumps({'prefill_weight_pipeline_v2':value})):
                with self.assertRaises(RuntimeError):t.seed_recipe()

    def test_matching_production_shapes_and_fused_dispatch_are_admitted(self):
        for rows in t.ROWS:
            for arm in t.ARMS:
                with self.subTest(arm=arm,rows=rows):t.batch_gate(self.batch(arm,rows),arm,rows)

    def test_changed_source_dispatch_work_or_timer_are_rejected(self):
        for rows in [16,4096]:
            for arm in t.ARMS:
                changes=[('copies',1),('epochs',128),('actual_source_experts',2),
                    ('source_footprint_bytes',3942400),('split_primary_rows',rows),
                    ('gu_fused',512 if rows==16 else 0),('down_fused',512 if rows==16 else 0),
                    ('production_gemm_dispatch',False),('split_tail_checked',False),
                    ('full_bitwise_reference_and_guards',False),('rows',True),
                    ('wall_s',.099),('wall_s',float('nan')),('timer_pair_max_s',.002)]
                for field,value in changes:
                    batch=self.batch(arm,rows);batch[field]=value
                    with self.subTest(arm=arm,rows=rows,field=field):
                        with self.assertRaises(RuntimeError):t.batch_gate(batch,arm,rows)
                with self.assertRaises(RuntimeError):
                    t.batch_gate(self.batch(arm,rows),'shortcut' if arm=='direct' else 'direct',rows)

    def test_exact_dependency_queue_and_reader_policy_are_required(self):
        for arm in t.ARMS:
            bridge=arm in ['bridge','queue','lookahead']
            proof=dict(kind='source-proof',ordinary_readonly_mapping=True,host_slots=0,policy=arm,
                dependency_bridge=bridge,separate_queue=arm in ['queue','lookahead'],
                reader_marker=('after-both-IQ' if arm=='lookahead' else 'after-math') if bridge else 'none',
                runtime_check_and_observer_retained=True)
            t.source_proof_gate(proof,arm)
            for key,value in [('ordinary_readonly_mapping',False),('host_slots',4),
                ('runtime_check_and_observer_retained',False),('policy','staged'),
                ('dependency_bridge',int(bridge)),('separate_queue',not proof['separate_queue']),
                ('reader_marker','after-one-IQ'),('unexpected_field',True)]:
                with self.subTest(arm=arm,key=key),self.assertRaises(RuntimeError):
                    t.source_proof_gate(dict(proof,**{key:value}),arm)
            with self.assertRaises(RuntimeError):t.source_proof_gate(proof,'staged')

    def test_changed_math_protocol_is_rejected_before_source_access(self):
        for field,value in [('family','other'),('mode','ordered'),('arm','staged'),
                ('rows',True),('rows',640),('argv',['wrong']),
                ('environment',dict(t.BACKEND,STRATA_SYCL_PARALLEL_IQ_DEQUANT='0'))]:
            row=copy.deepcopy(self.row);row['cases'][0][field]=value
            with self.subTest(field=field,value=value), \
                 patch.object(t.base.ur_validation,'committed',return_value=t.compact(row)), \
                 patch.object(t.base.original.coherent.build.common.source,'pin') as source:
                with self.assertRaises(RuntimeError):t.check(row)
                source.assert_not_called()

    def test_explicit_lifetime_mode_placement_and_marker_are_required(self):
        for mode in ['metadata','ordered','throw-copy','throw-reader']:
            for placement in ['compute','separate']:
                for retirement in ['math','iq']:
                    case=dict(family='lifetime',mode=mode,arm='lifetime',rows=0,trace=True,
                        placement=placement,retirement=retirement,environment=t.BACKEND,
                        argv=[str(t.ROOT/'lifetime'),mode,placement,retirement])
                    t.case_gate(case,'/exact/weights')
                    for field,value in [('mode','batch'),('placement','transfer'),
                            ('retirement','after-one-IQ'),('trace',False),('arm','direct'),
                            ('rows',16),('argv',[str(t.ROOT/'benchmark'),mode,placement,retirement])]:
                        with self.subTest(mode=mode,placement=placement,field=field), \
                             self.assertRaises(RuntimeError):
                            t.case_gate(dict(case,**{field:value}),'/exact/weights')

    def test_unit_losses_survive_a_faster_combination_and_missing_arms_refuse(self):
        results=[]
        for rows in t.ROWS:
            for pair in range(3):
                # R0 loses in every block; O loses once although net always wins.
                times=dict(direct=10,shortcut=8,bridge=9,queue=6,
                    lookahead=7 if pair==2 else 5)
                for arm,value in times.items():
                    value*=1+pair*.1
                    results.append(dict(family='math',clean=True,rows=rows,pair=pair,arm=arm,
                        batch=dict(wall_s=value),lifecycle_plus_batch_s=value+2))
        analysis=t.analyze(results)
        for value in analysis.values():
            self.assertFalse(value['R0']['repeated_component_gain'])
            self.assertTrue(value['Q1']['repeated_component_gain'])
            self.assertFalse(value['O']['repeated_component_gain'])
            self.assertTrue(value['net']['repeated_component_gain'])
            self.assertEqual(len(value['O']['paired_time_ratios']),3)
            self.assertGreater(value['O']['paired_time_ratios'][2],1)
            self.assertAlmostEqual(value['R0']['geometric_mean_ratio'],9/8)
            self.assertTrue(all(v['model_adoption_qualified'] is False for v in value.values()))
        with self.assertRaises(RuntimeError):t.analyze(results[:-1])
        duplicated=copy.deepcopy(results);duplicated[0]['arm']='shortcut'
        with self.assertRaises(RuntimeError):t.analyze(duplicated)

    @staticmethod
    def batch(arm,rows):
        return dict(kind='batch',arm=arm,rows=rows,epochs=512,
            requested_H2D_bytes=512*1971200,copies=512,gu_fused=512 if rows>=2334 else 0,
            down_fused=512 if rows>=2334 else 0,production_gemm_dispatch=True,
            split_primary_rows=rows//2,actual_source_experts=512,source_footprint_bytes=512*1971200,
            split_tail_checked=True,full_bitwise_reference_and_guards=True,
            wall_s=.5,timer_pair_max_s=.000001)


if __name__=='__main__':unittest.main()
