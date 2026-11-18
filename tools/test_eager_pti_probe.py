"""Owned negative controls for same-image collector/safety/RAW admission."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parent/'sycl'))
import eager_pti_probe as probe


class EagerPTIProbeTests(unittest.TestCase):
    def setUp(self):
        self.row=dict(environment=dict(UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1',NEOReadDebugKeys='1',
            EnableDirectSubmission='0',STRATA_SYCL_PREFILL_SAME_QUEUE='1',ZES_ENABLE_SYSMAN='0'),
            native_validation_probe=dict(loader=dict(path='/qualified/loader'),library=dict(path='/qualified/shim')),
            fixture=dict(path='/qualified/eager'),tools=dict(artifacts=dict(unitrace=dict(path='/qualified/unitrace'),
                **{'libunitrace_tool.so':dict(path='/qualified/collector',sha256='collector')})))

    def test_target_only_injection_preserves_fixture_safety_and_input(self):
        original=copy.deepcopy(self.row);control=probe.environment(self.row,'control');trace=probe.environment(self.row,'trace')
        self.assertEqual(control['LD_PRELOAD'],'/qualified/shim');self.assertNotIn('LD_PRELOAD',trace)
        self.assertEqual(trace['STRATA_UNITRACE_TARGET_PRELOAD'],'/qualified/shim')
        self.assertEqual(trace['UNITRACE_SystemTime'],'1')
        for env in (control,trace):
            for k,v in original['environment'].items():self.assertEqual(env[k],v)
            self.assertEqual(env['STRATA_SYCL_PREFILL_FULLFIT_UPLOAD'],'eager')
        self.assertEqual(probe.command(self.row,'control'),['/qualified/eager','--fullfit-eager-smoke'])
        self.assertEqual(probe.command(self.row,'trace')[-2:],probe.command(self.row,'control'))
        self.assertEqual(self.row,original)
        for key,value in [('ZES_ENABLE_SYSMAN','1'),('EnableDirectSubmission','1'),
            ('UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD','0'),('STRATA_SYCL_PREFILL_SAME_QUEUE','0'),
            ('LD_PRELOAD','/unexpected'),('ZET_ENABLE_METRICS','1'),('UNITRACE_DeviceTiming','1')]:
            bad=copy.deepcopy(self.row);bad['environment'][key]=value
            with self.assertRaises((RuntimeError,ValueError)):probe.environment(bad,'trace')
        with self.assertRaises((RuntimeError,ValueError)):probe.command(self.row,'unknown')

    def test_only_positive_complete_gpu_intervals_inside_raw_lifetime_pass(self):
        def trace(**changes):return dict(traceEvents=[dict(cat='gpu_op',ph='X',ts=2.,dur=1.,**changes)])
        result=probe.trace_summary(trace(),1000,4000)
        self.assertEqual(result['GPU_events'],1);self.assertEqual(result['observed_GPU_union_ns'],1000)
        self.assertFalse(result['complete_service_count_coverage_qualified'])
        for value in [dict(traceEvents=[]),dict(traceEvents=[dict(cat='gpu_op',ph='B',ts=2.,dur=1.)]),
            dict(traceEvents=[dict(cat='gpu_op',ph='X',ts=2.,dur=0.)]),
            dict(traceEvents=[dict(cat='gpu_op',ph='X',ts=.5,dur=1.)]),
            dict(traceEvents=[dict(cat='gpu_op',ph='X',ts=3.,dur=2.)]),
            dict(traceEvents=[dict(cat='gpu_op',ph='X',ts=float('nan'),dur=1.)])]:
            with self.assertRaises((RuntimeError,ValueError)):probe.trace_summary(value,1000,4000)

    def test_actual_provider_hashes_same_pid_and_collector_are_required(self):
        row=copy.deepcopy(self.row);paths=['/qualified/loader','/qualified/shim','/qualified/adapter',
            '/qualified/libigc.so','/qualified/libsycl.so','/qualified/libze_intel_gpu.so']
        with tempfile.TemporaryDirectory() as root:
            manifest=Path(root)/'runtime.json';manifest.write_text(json.dumps({p:'expected' for p in paths}))
            row['runtime_manifest']=dict(path=str(manifest));row['environment']['UR_ADAPTERS_FORCE_LOAD']=paths[2]
            row['native_validation_probe']['loader']['sha256']=row['native_validation_probe']['library']['sha256']='expected'
            def text(extra=(),pid=123):return ''.join(f' {pid}: calling init: {p}\n' for p in [*paths,*extra])
            with patch.object(probe,'sha',side_effect=lambda p:'collector' if str(p)=='/qualified/collector' else 'expected'):
                self.assertEqual(probe.providers(text(),row,'control',123)['initialized_ELFs'],6)
                self.assertEqual(probe.providers(text(['/qualified/collector']),row,'trace',123)['initialized_ELFs'],7)
                for errors,label in [(text(), 'trace'),(text(pid=124),'control'),(text(['/unapproved/lib.so']),'control')]:
                    with self.assertRaises((RuntimeError,ValueError)):probe.providers(errors,row,label,123)
            with patch.object(probe,'sha',return_value='changed'):
                with self.assertRaises((RuntimeError,ValueError)):probe.providers(text(),row,'control',123)


if __name__=='__main__':unittest.main()
