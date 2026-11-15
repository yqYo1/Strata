"""Fail-closed controls for the event fixture's actual local-plan admission.

Repository LICENSE. No runtime/driver,process,filesystem mutation or GPU work.
The source-pin sentinel proves rejection happens before subsequent admission.
"""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/sycl'))
import upload_event_lane as target


class ReachedSourcePin(Exception):
    pass


class UploadPlanAdmission(unittest.TestCase):
    def setUp(self):
        self.row=dict(stage='built',passed=True,complete=False,active=False,ownership_closed=True,
            project=dict(count=1,revision='a'*40,source_sha256={'unit.cpp':'b'*64}),foundation={},
            binary=dict(path='/guarded/probe',sha256='c'*64),plan=dict(gates='frozen'),
            cases=[dict(mode='forward',argv=['/guarded/probe','forward'],
                environment=dict(EnableDirectSubmission='0'))])
        self.public=target.compact(self.row)

    def invoke(self,row,exception):
        with patch.object(target.ur_validation,'committed',return_value=copy.deepcopy(self.public)), \
             patch.object(target.original.coherent.build.common.source,'pin',side_effect=ReachedSourcePin) as pin:
            with self.assertRaises(exception):target.check(row)
            if exception is RuntimeError:pin.assert_not_called()
            else:pin.assert_called_once()

    def test_exact_plan_reaches_source_check(self):
        self.invoke(copy.deepcopy(self.row),ReachedSourcePin)

    def test_changed_launch_and_safety_are_rejected_first(self):
        for mutation in ['argv','environment','binary','missing_gate','active']:
            with self.subTest(mutation=mutation):
                row=copy.deepcopy(self.row)
                if mutation=='argv':row['cases'][0]['argv'][0]='/other/probe'
                elif mutation=='environment':row['cases'][0]['environment']['EnableDirectSubmission']='1'
                elif mutation=='binary':row['binary']['sha256']='d'*64
                elif mutation=='missing_gate':del row['plan']['gates']
                else:row['active']=True
                self.invoke(row,RuntimeError)

    def test_json_boolean_and_integer_types_are_distinct(self):
        for mutation in ['ownership','source_count']:
            with self.subTest(mutation=mutation):
                row=copy.deepcopy(self.row)
                if mutation=='ownership':row['ownership_closed']=1
                else:row['project']['count']=True
                self.invoke(row,RuntimeError)


if __name__=='__main__':unittest.main()
