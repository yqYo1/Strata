"""Owned arm,source and provider guards for the bounded PTI fixture probe."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).parent/'sycl'))
import current_pti_probe as probe


class CurrentPTIProbeTests(unittest.TestCase):
    def setUp(self):
        self.row=dict(base_environment=dict(ZES_ENABLE_SYSMAN='0',EnableDirectSubmission='0',
            UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1',NEOReadDebugKeys='1'),
            preload=dict(loader=dict(path='/qualified/loader'),library=dict(path='/qualified/shim')),
            fixture=dict(path='/qualified/fixture'),
            admission=dict(tools=dict(artifacts=dict(unitrace=dict(path='/qualified/unitrace')))))

    def test_arms_keep_safety_and_apply_preload_only_to_target(self):
        original=copy.deepcopy(self.row)
        a=probe.environment(self.row,'control');b=probe.environment(self.row,'trace')
        self.assertEqual(a['LD_PRELOAD'],'/qualified/shim');self.assertNotIn('LD_PRELOAD',b)
        self.assertEqual(b['STRATA_UNITRACE_TARGET_PRELOAD'],'/qualified/shim')
        self.assertEqual(b['UNITRACE_SystemTime'],'1')
        for env in (a,b):
            self.assertEqual(env['ZES_ENABLE_SYSMAN'],'0')
            self.assertEqual(env['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD'],'1')
            self.assertEqual(env['STRATA_TEST_MAIN_GATES'],'1')
        self.assertEqual(probe.command(self.row,'control'),['/qualified/fixture','--submission'])
        argv=probe.command(self.row,'trace');self.assertEqual(argv[0],'/qualified/unitrace')
        self.assertEqual(argv[-2:],['/qualified/fixture','--submission'])
        self.assertIn('--chrome-device-logging',argv);self.assertEqual(self.row,original)

    def test_unsafe_flags_launcher_preload_and_unplanned_arm_are_refused(self):
        for key,value in [('ZES_ENABLE_SYSMAN','1'),('EnableDirectSubmission','1'),
                          ('UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD','0'),('LD_PRELOAD','/unsafe'),
                          ('ZET_ENABLE_METRICS','1')]:
            row=copy.deepcopy(self.row);row['base_environment'][key]=value
            with self.assertRaises(ValueError):probe.environment(row,'trace')
        with self.assertRaises(ValueError):probe.environment(self.row,'unknown')
        with self.assertRaises(ValueError):probe.command(self.row,'unknown')

    def test_only_observer_commit_may_differ_from_closed_host_admission(self):
        original=dict(controller_commit='old',native_source_commit='native',tools={'sha':'exact'})
        gate=dict(passed=True,complete=True,ownership_closed=True,unit_tests=4,
            source_sha256={'source.py':'expected'},admission=original)
        native=dict(binaries={'sycl_submission_test':{'path':'fixture'}},native_validation_probe={'loader':{},'library':{}})
        with patch.object(probe.admission,'committed_json',return_value={'unitrace_current_admission_host_v3':gate}), \
             patch.object(probe,'digest',return_value='expected'), \
             patch.object(probe.admission.native,'verify_original',return_value=(native,{})), \
             patch.object(probe.admission,'admit',return_value=dict(original,controller_commit='new')) as run:
            self.assertEqual(probe.basis()[1],{'path':'fixture'})
            run.return_value=dict(original,tools={'sha':'changed'})
            with self.assertRaises(ValueError):probe.basis()
            gate['passed']=False
            with self.assertRaises(ValueError):probe.basis()

    def test_untracked_probe_or_missing_controls_cannot_be_pinned(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root);(p/'tools/sycl').mkdir(parents=True);(p/'tools/sycl/other.py').write_bytes(b'old')
            def git(repo,*args):return b'revision\n' if args[0]=='rev-parse' else b'tools/sycl/other.py\n'
            with patch.object(probe,'P',p),patch.object(probe.admission.pti,'git',side_effect=git), \
                 patch.object(probe.admission,'blob',return_value=b'old'):
                with self.assertRaises(ValueError):probe.source_pin()


if __name__=='__main__':unittest.main()
