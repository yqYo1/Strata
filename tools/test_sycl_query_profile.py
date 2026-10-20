#!/usr/bin/env python3
"""Offline tests for fail-closed query diagnostic evidence/runtime admission."""
import copy
import sys
from pathlib import Path
import unittest
from unittest import mock

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent/'sycl'))
import query_profile as q


def rows():
    return [dict(calls=3, ready=2, pending=1, errors=0, clock_errors=0,
                 wall_ns=900, max_wall_ns=600),
            dict(calls=3, ready=3, pending=0, errors=0, clock_errors=0,
                 wall_ns=600, max_wall_ns=300),
            dict.fromkeys(q.QUERY_KEYS, 0), dict.fromkeys(q.QUERY_KEYS, 0)]


def log(values, kinds=q.QUERY_KINDS):
    return '\n'.join('strata serve: SYCL_EVENT_QUERY scope=decode wrapper_wall=1 kind='+kind+' '+
        ' '.join(key+'='+str(row[key]) for key in q.QUERY_KEYS)
        for kind,row in zip(kinds,values))+'\n'


class QueryEvidence(unittest.TestCase):
    def test_multiple_requests(self):
        result=q.query_evidence(log(rows())*2,2)
        self.assertEqual(len(result),2)
        self.assertEqual(result[1]['resident_launch']['pending'],1)

    def test_reject_missing_duplicate_and_unparsed_records(self):
        for text in (log(rows())[:-1].rsplit('\n',1)[0]+'\n',
                     log(rows(),('resident_launch',)*4),
                     log(rows())+'SYCL_EVENT_QUERY malformed\n'):
            with self.subTest(text=text), self.assertRaises(RuntimeError):
                q.query_evidence(text,1)

    def test_reject_invalid_counters(self):
        for index,key,value in ((0,'calls',4),(0,'errors',1),(0,'clock_errors',1),
                                (0,'max_wall_ns',901),(0,'wall_ns',0),(0,'calls',-1),
                                (2,'wall_ns',1),(3,'calls',1)):
            values=rows();values[index][key]=value
            with self.subTest(index=index,key=key), self.assertRaises(RuntimeError):
                q.query_evidence(log(values),1)


class RuntimeAdmission(unittest.TestCase):
    def setUp(self):
        self.delta={n:dict(qualified_sha256=v[0],candidate_sha256=v[1]) for n,v in q.DELTA.items()}
        self.component=dict(passed=True,complete=True,ownership_closed=True,active=False,
                            fresh_faults=[],runtime_asset_delta=self.delta)
        self.settings=dict(runtime_elfs={n:dict(sha256=v[0]) for n,v in q.DELTA.items()})
        self.settings['runtime_elfs']['unchanged-runtime']=dict(sha256='unchanged')
        self.actual={n:v[1] for n,v in q.DELTA.items()};self.actual['unchanged-runtime']='unchanged'

    def admit(self,record):
        with mock.patch.object(q.gen,'committed',side_effect=lambda p:
                self.component if p==q.COMPONENT else self.settings), \
             mock.patch.object(q.gen.common,'digest',side_effect=self.actual.__getitem__):
            q.runtime_assets_unchanged(record)

    def test_exact_component_assets_and_both_diagnostic_modes(self):
        for mode in ('default-route-query-profile-check','default-route-query-profile-long','cache-borrow-check','cache-capacity-check'):
            self.admit(dict(mode=mode,runtime_asset_delta=self.delta))
        labeled=copy.deepcopy(self.delta)
        for item in labeled.values():item.update(package='label',candidate_version='version',upgrade_utc='time')
        self.admit(dict(mode='default-route-query-profile-check',runtime_asset_delta=labeled))

    def test_unknown_mode_and_changed_delta_refused(self):
        with self.assertRaises(RuntimeError):
            self.admit(dict(mode='early-pcie-check',runtime_asset_delta=self.delta))
        changed=copy.deepcopy(self.delta);next(iter(changed.values()))['candidate_sha256']='changed'
        with self.assertRaises(RuntimeError):
            self.admit(dict(mode='default-route-query-profile-check',runtime_asset_delta=changed))

    def test_failed_component_or_changed_runtime_refused(self):
        record=dict(mode='default-route-query-profile-check',runtime_asset_delta=self.delta)
        self.component['passed']=False
        with self.assertRaises(RuntimeError):self.admit(record)
        self.component['passed']=True
        self.actual['unchanged-runtime']='changed'
        with self.assertRaises(RuntimeError):self.admit(record)


if __name__=='__main__':unittest.main()
