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

    def test_original_runtime_cases_survive_e1_receipt_retirement(self):
        proof=overlap.parent_evidence()
        self.assertEqual(proof['evidence_commit'],'3ee710401f9cb8323757161dd53a17dce6689094')
        self.assertEqual(proof['native_source_commit'],'d4e45762a56b169036a67eb9535cd3ce9b238a46')
        self.assertIs(proof['execution_input'],False)
        self.assertIs(proof['new_source_or_performance_qualification'],False)

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


class MappedSourcePropertyAdmission(unittest.TestCase):
    def setUp(self):
        self.row=dict(stage='closed',passed=True,complete=True,ownership_closed=True,active=False,
            properties=dict(kind='mapping-properties',bytes=4096,read_only=True,same_address=True,
                before=dict(type=0,allocation_id=0,page_bytes=0),
                during=dict(type=1,allocation_id=1,page_bytes=4096),
                after=dict(type=0,allocation_id=0,page_bytes=0),
                address_range=dict(result=0,same_base=True,bytes=4096),
                original_heap_unchanged=True,explicit_native_context=True,
                production_runtime_context_qualified=False,allocation_and_unimport_executed=True,
                application_commands=0,source_admission_or_transfer_qualified=False))

    def test_exact_closed_property_observation(self):
        self.assertEqual(overlap.property_gate(self.row,copy.deepcopy(self.row)),self.row['properties'])

    def test_changed_history_open_owner_and_false_qualification(self):
        for which in ['history','owner','active','qualification']:
            row=copy.deepcopy(self.row)
            if which=='history':row['properties']['during']['allocation_id']=2
            elif which=='owner':row['ownership_closed']=1
            elif which=='active':row['active']=True
            else:row['properties']['source_admission_or_transfer_qualified']=True
            with self.subTest(which=which),self.assertRaises(RuntimeError):
                overlap.property_gate(row,self.row)

    def test_host_identity_range_and_removal_are_required(self):
        for key,field,value in [('during','type',4),('during','type',True),
                               ('during','allocation_id',0),('during','allocation_id',True),
                               ('address_range','bytes',8192),('address_range','same_base',False),
                               ('after','type',1),('before','allocation_id',True)]:
            row=copy.deepcopy(self.row);row['properties'][key][field]=value
            with self.subTest(key=key,field=field,value=value),self.assertRaises(RuntimeError):
                overlap.property_gate(row,copy.deepcopy(row))

    def test_original_property_survives_observer_retirement(self):
        proof=overlap.property_evidence()
        self.assertEqual(proof['properties']['during']['type'],1)
        self.assertEqual(proof['native_source_commit'],'b777f52f8ddfc60b71166cce9a68afb952c71dde')
        self.assertIs(proof['new_source_admission_qualified'],False)


if __name__=='__main__':unittest.main()
