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
            cases=[dict(source='usm',argv=['/exact/exe','serial','/exact/weights'],environment={'EnableDirectSubmission':'0'})])
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
                     patch.object(t.Path,'read_text',return_value=json.dumps(dict(stage=stage))), \
                     patch.object(t,'check',side_effect=RuntimeError('must not requalify')) as check, \
                     patch.object(t,'save') as save:
                    with self.assertRaises(RuntimeError):t.execute(action)
                    check.assert_not_called();save.assert_not_called()

    def test_matching_serial_and_lookahead_work_are_admitted(self):
        for arm in ['serial','lookahead']:
            with self.subTest(arm=arm):t.batch_gate(self.batch(arm),arm)

    def test_changed_dependency_counts_work_or_timer_are_rejected(self):
        for arm in ['serial','lookahead']:
            for field,value in [('copies_after_iq',2),('copies_after_math',2),
                                ('epochs',128),('source','mapped-neo'),('wall_s',.099),
                                ('timer_pair_max_s',.002)]:
                batch=self.batch(arm);batch[field]=value
                with self.subTest(arm=arm,field=field):
                    with self.assertRaises(RuntimeError):t.batch_gate(batch,arm)
            with self.assertRaises(RuntimeError):
                t.batch_gate(self.batch(arm),'serial' if arm=='lookahead' else 'lookahead')

    @staticmethod
    def batch(arm):
        return dict(kind='batch',arm=arm,source='usm',rows=1024,epochs=512,
            requested_H2D_bytes=512*1971200,copies_after_iq=511 if arm=='lookahead' else 0,
            copies_after_math=1 if arm=='lookahead' else 512,actual_weight_reader=True,
            full_bitwise_reference_and_guards=True,wall_s=.5,timer_pair_max_s=.000001)


if __name__=='__main__':unittest.main()
