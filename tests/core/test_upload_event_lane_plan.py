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
import prefill_upload_overlap as overlap


class ReachedSourcePin(Exception):
    pass


class UploadPlanAdmission(unittest.TestCase):
    target=target
    def setUp(self):
        self.row=dict(stage='built',passed=True,complete=False,active=False,ownership_closed=True,
            project=dict(count=1,revision='a'*40,source_sha256={'unit.cpp':'b'*64}),foundation={},allowed={},
            binary=dict(path='/guarded/probe',sha256='c'*64),plan=dict(gates='frozen'),
            cases=[dict(mode='forward',argv=['/guarded/probe','forward'],
                environment=dict(EnableDirectSubmission='0'))])
        self.public=self.target.compact(self.row)

    def invoke(self,row,exception):
        with patch.object(self.target.ur_validation,'committed',return_value=copy.deepcopy(self.public)), \
             patch.object(self.target.original.coherent.build.common.source,'pin',side_effect=ReachedSourcePin) as pin:
            with self.assertRaises(exception):self.target.check(row)
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


class UploadOverlapPlanAdmission(UploadPlanAdmission):
    target=overlap
    def setUp(self):
        super().setUp()
        # Both controllers reject changed inputs before source admission.
        self.row['allowed']={}
        self.public=self.target.compact(self.row)


class MappedSourceCapabilityAdmission(unittest.TestCase):
    def setUp(self):
        self.row=dict(stage='closed',passed=True,complete=True,ownership_closed=True,active=False,
            capabilities=dict(api_version=65553,external_sysmem_mapping_version=65536,
                read_only_mapping_reported_eligible=True,mapping_or_transfer_attempted=False))

    def test_exact_closed_discovery_reaches_source_unit(self):
        self.assertEqual(overlap.capability_gate(self.row,copy.deepcopy(self.row)),self.row['capabilities'])

    def test_changed_discovery_or_open_owner_is_rejected(self):
        original=copy.deepcopy(self.row)
        for which in ['history','open','partial','type']:
            row=copy.deepcopy(self.row)
            if which=='history':row['capabilities']['api_version']=65554
            elif which=='open':row['active']=True
            elif which=='partial':row['complete']=False
            else:row['ownership_closed']=1
            with self.subTest(which=which),self.assertRaises(RuntimeError):
                overlap.capability_gate(row,original)

    def test_external_map_and_read_only_floor_are_both_required(self):
        for name,value in [('api_version',65551),('api_version',True),
                           ('external_sysmem_mapping_version',0),
                           ('external_sysmem_mapping_version',True),
                           ('read_only_mapping_reported_eligible',False),
                           ('mapping_or_transfer_attempted',True)]:
            row=copy.deepcopy(self.row);row['capabilities'][name]=value
            with self.subTest(name=name,value=value),self.assertRaises(RuntimeError):
                overlap.capability_gate(row,copy.deepcopy(row))


if __name__=='__main__':unittest.main()
