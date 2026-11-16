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
            cases=[dict(arm='direct',mode='batch',rows=16,argv=['/exact/exe','batch','direct','16','/exact/weights'],environment={'EnableDirectSubmission':'0',**t.BACKEND})])
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
            for arm in ['direct','staged']:
                with self.subTest(arm=arm,rows=rows):t.batch_gate(self.batch(arm,rows),arm,rows)

    def test_changed_source_dispatch_work_or_timer_are_rejected(self):
        for rows in [16,4096]:
            for arm in ['direct','staged']:
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
                    t.batch_gate(self.batch(arm,rows),'direct' if arm=='staged' else 'staged',rows)

    @staticmethod
    def batch(arm,rows):
        return dict(kind='batch',arm=arm,rows=rows,epochs=512,
            requested_H2D_bytes=512*1971200,copies=512,gu_fused=512 if rows>=2334 else 0,
            down_fused=512 if rows>=2334 else 0,production_gemm_dispatch=True,
            split_primary_rows=rows//2,actual_source_experts=512,source_footprint_bytes=512*1971200,
            split_tail_checked=True,full_bitwise_reference_and_guards=True,
            wall_s=.5,timer_pair_max_s=.000001)


if __name__=='__main__':unittest.main()
