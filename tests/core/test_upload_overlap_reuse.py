"""Host-only controls for historical E2 admission; repository LICENSE."""
import copy
from pathlib import Path
import sys
import unittest
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/sycl'))
import prefill_upload_overlap_reuse as target


class HistoricalE2Gate(unittest.TestCase):
    def setUp(self):
        modes=['metadata','forward','reverse','permanent','cancel','cancel-throw','refuse']
        results=[dict(label='runtime-'+m,passed=True,native_exit_code=0,ownership_closed=True) for m in modes]
        results += [dict(label='component-'+str(i),passed=True,native_exit_code=0,ownership_closed=True) for i in range(16)]
        command=dict(observation_complete=True,direct_child_reaped=True,session_empty=True,exit_code=0)
        self.row=dict(stage='closed',passed=True,complete=True,ownership_closed=True,active=False,
            model_trial_allowed=False,native_source_commit=target.NATIVE,
            project=dict(revision=target.NATIVE,count=1,source_sha256={'source.cpp':'a'*64}),
            foundation={},allowed={},results=results,recipe=[dict(dep='source') for _ in range(5)],
            actual_inputs={k:{str(i):'b'*64 for i in range(n)} for k,n in [('compiled',925),('links',25),('objects',7)]},
            commands=[copy.deepcopy(command) for _ in range(33)],analysis={
                '1024':dict(repeated_component_wall_gain=False),
                '4096':dict(repeated_component_wall_gain=True)})
        self.public=target.component.compact(self.row)

    def test_exact_closed_component_is_accepted(self):
        target.original_gate(self.public,self.row)

    def test_partial_or_broader_claims_fail_even_with_matching_receipt(self):
        for change in ['source','active','model','missing_case','missing_input','missing_owner','failed_owner','small_gate']:
            with self.subTest(change=change):
                row=copy.deepcopy(self.row)
                if change=='source':row['native_source_commit']='c'*40
                elif change=='active':row['active']=True
                elif change=='model':row['model_trial_allowed']=True
                elif change=='missing_case':row['results'].pop(0)
                elif change=='missing_input':row['actual_inputs']['compiled'].pop('0')
                elif change=='missing_owner':row['commands'].pop()
                elif change=='failed_owner':row['commands'][0]['exit_code']=1
                else:row['analysis']['1024']['repeated_component_wall_gain']=True
                with self.assertRaises(RuntimeError):target.original_gate(target.component.compact(row),row)

    def test_changed_recipe_is_rejected_by_original_receipt(self):
        row=copy.deepcopy(self.row);row['recipe'][0]['dep']='other-source'
        with self.assertRaises(RuntimeError):target.original_gate(self.public,row)

    def test_boolean_integer_confusion_is_rejected(self):
        row=copy.deepcopy(self.row);row['complete']=1
        with self.assertRaises(RuntimeError):target.original_gate(self.public,row)


if __name__=='__main__':unittest.main()
