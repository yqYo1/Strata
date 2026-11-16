"""Fail-closed historical source-cost qualification; no device execution.

Main-authored; repository LICENSE. A successful cohort must not erase failed
shape eligibility or substitute typed plan/build evidence before native reuse.
"""
import copy
from pathlib import Path
import sys
import unittest
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/sycl'))
import prefill_mapped_source_reuse as target


class MappedHistoricalAdmission(unittest.TestCase):
    def setUp(self):
        labels=['runtime-metadata-usm','mapped-import']+[s+'-'+m
            for s in ['usm','mapped-neo'] for m in
            ['forward','reverse','permanent','cancel','cancel-throw','refuse']]+['mapped-retire']
        labels += [f'batch-{i}' for i in range(16)]
        result=dict(passed=True,native_exit_code=0,ownership_closed=True)
        command=dict(exit_code=0,observation_complete=True,direct_child_reaped=True,session_empty=True)
        self.row=dict(project=dict(revision=target.NATIVE,source_sha256={}),foundation={},allowed={},
            native_source_commit=target.NATIVE,stage='closed',complete=True,passed=True,
            ownership_closed=True,active=False,model_trial_allowed=False,
            results=[dict(result,label=label) for label in labels],
            analysis={'1024':dict(source_cost_eligible=True),'4096':dict(source_cost_eligible=False)},
            host_admission_controls=dict(methods=14,passed=True,device_executed=False),
            recipe=[dict(dep=str(i)) for i in range(5)],commands=[dict(command) for _ in range(41)],
            actual_inputs={k:{str(i):'a'*64 for i in range(n)}
                           for k,n in [('compiled',928),('links',25),('objects',7)]})
        self.public=target.component.compact(self.row)

    def test_exact_closed_cohort_keeps_shape_specific_decision(self):
        target.original_gate(self.public,self.row)

    def test_changed_private_plan_or_typed_outcome_is_rejected(self):
        for which in ['recipe','shape','owner','partial','boolean']:
            row=copy.deepcopy(self.row)
            if which=='recipe':row['recipe'][0]['dep']='changed'
            elif which=='shape':row['analysis']['4096']['source_cost_eligible']=True
            elif which=='owner':row['ownership_closed']=1
            elif which=='partial':row['results'].pop()
            else:row['results'][0]['native_exit_code']=False
            with self.subTest(which=which),self.assertRaises(RuntimeError):
                target.original_gate(self.public,row)

    def test_matching_encodings_do_not_waive_qualification(self):
        for which in ['1024gate','4096gate','open','process','census','controls','model']:
            row=copy.deepcopy(self.row)
            if which=='1024gate':row['analysis']['1024']['source_cost_eligible']=False
            elif which=='4096gate':row['analysis']['4096']['source_cost_eligible']=True
            elif which=='open':row['active']=True
            elif which=='process':row['commands'][0]['session_empty']=False
            elif which=='census':row['actual_inputs']['compiled'].pop('0')
            elif which=='controls':row['host_admission_controls']['methods']=13
            else:row['model_trial_allowed']=True
            with self.subTest(which=which),self.assertRaises(RuntimeError):
                target.original_gate(target.component.compact(row),row)


if __name__=='__main__':unittest.main()
