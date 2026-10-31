#!/usr/bin/env python3
"""Refuse changed graph fixtures/recipes before reused model-observer admission."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent/'sycl'))
import mtp_model_observer as m


class GraphSupportReuseTests(unittest.TestCase):
    def setUp(self):
        fixture=dict(path='/same/fixture',sha256='f'*64)
        self.support=dict(passed=True,complete=True,ownership_closed=True,active=False,
            boot_unchanged=True,fresh_faults=[],fixture=fixture,source_commit=m.GRAPH_REUSE_SOURCES[0],
            build_record_sha256='original-record',checks=[dict(profile=i,passed=True,exit_code=0) for i in (0,1)])
        self.old=dict(source_commit=self.support['source_commit'],fixtures=dict(sycl_graph_envelope_test=fixture),
            compiler='compiler',configure_argv=['recipe'],recipe_reference='source-recipe',
            fixture_external_headers={'header':'hash'},inputs=[{}])
        self.full=dict(copy.deepcopy(self.old),source_commit=m.GRAPH_REUSE_SOURCES[1],dependencies=[])
        self.raw=json.dumps(self.support).encode();self.build_raw=json.dumps(self.old).encode()

    def evidence(self,full=None,support=None,delta=b''):
        def git(*args):
            if args[0]=='diff':return delta
            return self.build_raw if args[-1].endswith('native-gdn-engine-build.json') else self.raw
        with patch.object(m.gen,'git',side_effect=git), \
             patch.object(m,'GRAPH_ORIGIN_SHA256',hashlib.sha256(self.raw).hexdigest()), \
             patch.object(m,'GRAPH_BUILD_SHA256',hashlib.sha256(self.build_raw).hexdigest()):
            return m.graph_support_evidence(full or self.full,support or self.support)

    def test_identical_fixture_reuses_only_the_reviewed_source_pair(self):
        proof=self.evidence()
        self.assertTrue(proof['graph_implementation_unchanged'])
        self.assertFalse(proof['new_GPU_validation'])
        self.assertFalse(proof['model_profile_qualified'])

    def test_missing_or_changed_qualification_is_rejected(self):
        for variant in ('failed','incomplete','owner','active','boot','fault','fixture',
                        'missing_check','failed_check','other_source','compiler','recipe','header','dependency','native'):
            full,support=copy.deepcopy((self.full,self.support));delta=b''
            if variant=='failed':support['passed']=False
            if variant=='incomplete':support['complete']=False
            if variant=='owner':support['ownership_closed']=False
            if variant=='active':support['active']=True
            if variant=='boot':support['boot_unchanged']=False
            if variant=='fault':support['fresh_faults']=['fault']
            if variant=='fixture':full['fixtures']['sycl_graph_envelope_test']['sha256']='other'
            if variant=='missing_check':support['checks'].pop()
            if variant=='failed_check':support['checks'][1]['exit_code']=77
            if variant=='other_source':full['source_commit']='unreviewed-native'
            if variant=='compiler':full['compiler']='other'
            if variant=='recipe':full['configure_argv']=['other']
            if variant=='header':full['fixture_external_headers']={'other':'hash'}
            if variant=='dependency':full['dependencies']=[dict(root='other',revision='r',count=1,aggregate_sha256='h')]
            if variant=='native':delta=b'src/core/mtp.cpp\n'
            with self.subTest(variant=variant),self.assertRaises(RuntimeError):self.evidence(full,support,delta)

    def test_same_source_still_requires_its_exact_build_receipt(self):
        full=dict(self.full,source_commit=self.support['source_commit'])
        with patch.object(m.gen.common,'digest',return_value='original-record'):
            self.assertEqual(m.graph_support_evidence(full,self.support)['kind'],'same-build')
        with patch.object(m.gen.common,'digest',return_value='other'),self.assertRaises(RuntimeError):
            m.graph_support_evidence(full,self.support)

    def test_changed_archived_evidence_cannot_be_relabelled(self):
        self.raw=b'changed archive'
        with patch.object(m.gen,'git',return_value=self.raw),self.assertRaises(RuntimeError):
            m.graph_support_evidence(self.full,self.support)


class RestoredDiagnosticPlanTests(unittest.TestCase):
    def test_graph_diagnostic_preserves_decode_and_runtime_safety(self):
        root=Path('/new-profile')
        safety=dict(UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1',NEOReadDebugKeys='1',
                    EnableDirectSubmission='0',STRATA_SYCL_PREFILL_SAME_QUEUE='1')
        setup=dict(environment=dict(safety,STRATA_MEASUREMENT_ROOT='/closed',
            TMPDIR='/closed/tmp',STRATA_IQ2_PAIR='1',UR_LOG_TRACING='retired'))
        with patch.object(m.gen,'committed',return_value={'environment':safety}):
            env=m.environment(setup,root,'graph')
        self.assertEqual({k:env[k] for k in safety},safety)
        self.assertEqual(env['TMPDIR'],'/new-profile/tmp')
        self.assertEqual(env['STRATA_IQ2_PAIR'],'1')
        self.assertEqual(env['STRATA_SYCL_MTP_OBSERVER_REQUEST'],'5')
        self.assertEqual(env['STRATA_SYCL_MTP_OBSERVER_WINDOWS'],'64')
        self.assertEqual(env['UR_ENABLE_LAYERS'],'UR_LAYER_PARAMETER_VALIDATION')
        self.assertNotIn('UR_LOG_TRACING',env)
        self.assertNotIn('LD_PRELOAD',env)
        self.assertEqual(m.plan()['modes'],['graph'])
        self.assertIn('no observer-overhead bound',m.plan()['interpretation'])

    def test_old_config_reconstruction_is_recursive_without_mutating_its_source(self):
        config=dict(args=['--expert-profile','${RUN_ROOT}/expert-profile.bin'],
                    log='${RUN_ROOT}/engine.stderr',parallel=1,env={})
        original=copy.deepcopy(config)
        changed=m.expand_root(config,Path('/restored'))
        self.assertEqual(config,original)
        self.assertEqual(changed['args'][1],'/restored/expert-profile.bin')
        self.assertEqual(changed['log'],'/restored/engine.stderr')
        self.assertEqual(changed['parallel'],1)


if __name__=='__main__':unittest.main()
