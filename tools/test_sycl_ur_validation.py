#!/usr/bin/env python3
"""Regress actual UR enum parsing and reject mixed/ambiguous child evidence."""
from pathlib import Path
import hashlib
import sys
import tempfile
import unittest

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent/'sycl'))
import ur_validation as probe
import engine_once as wrapper


class Proof(unittest.TestCase):
    begin='strata UR validation: pid=17 phase=begin loader=/pinned/libur_loader.so.0.12.0\n'
    proof='strata UR validation: pid=17 phase=before-main threads_before=1 preinit=37 config_create=0 layers_query=0 enable=0 init=0 config_release=0 checked=46 untouched=1 retained_ref=1\n'

    def test_installed_enum_and_process_lifetime_limit(self):
        result=probe.preload_evidence(self.begin+self.proof,17)
        self.assertEqual(result['preinit'],37)
        self.assertTrue(result['first_init'])
        self.assertFalse(result['runtime_teardown_qualified'])

    def test_reject_wrong_enum_pid_order_duplicate_and_failure(self):
        for text,pid in ((self.begin+self.proof.replace('preinit=37','preinit=1'),17),
            (self.begin+self.proof,18),(self.proof+self.begin,17),
            (self.begin+self.proof+self.proof,17),
            (self.begin+self.proof+'strata UR validation failure: injected\n',17)):
            with self.subTest(text=text,pid=pid),self.assertRaises(RuntimeError):
                probe.preload_evidence(text,pid)


class EngineEnvironment(unittest.TestCase):
    def test_matched_control_keeps_safety_and_rejects_environment_layers(self):
        with tempfile.TemporaryDirectory() as directory:
            library=Path(directory)/'control';library.write_bytes(b'pinned')
            artifact=dict(path=str(library),sha256=hashlib.sha256(library.read_bytes()).hexdigest())
            parent=dict(CUDA_MODULE_LOADING='EAGER',UR_ENABLE_LAYERS='',
                UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1',NEOReadDebugKeys='1',
                EnableDirectSubmission='0',STRATA_SYCL_PREFILL_SAME_QUEUE='1')
            for mode in ('on','off'):
                probe=dict(library=artifact,loader=artifact,control='held-validation',mode=mode)
                plan=dict(native_validation_probe=probe)
                child=wrapper.native_environment(plan,parent)
                self.assertEqual(child['STRATA_UR_HOLD_VALIDATION'],mode)
                self.assertNotIn('STRATA_UR_HOLD_VALIDATION',parent)
                for changed in (dict(UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION'),
                    dict(EnableDirectSubmission='1'),dict(UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='0'),
                    dict(STRATA_SYCL_PREFILL_SAME_QUEUE='0'),dict(STRATA_UR_HOLD_VALIDATION=mode)):
                    with self.assertRaises(RuntimeError):wrapper.native_environment(plan,parent|changed)
            for bad in (dict(mode='invalid'),dict(control='unknown')):
                with self.assertRaises(RuntimeError):wrapper.native_environment(
                    dict(native_validation_probe=probe|bad),parent)

    def test_probe_is_native_only_and_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            library=Path(directory)/'library';library.write_bytes(b'pinned library bytes')
            artifact=dict(path=str(library),sha256=hashlib.sha256(library.read_bytes()).hexdigest())
            plan=dict(native_validation_probe=dict(library=artifact,loader=artifact))
            parent=dict(CUDA_MODULE_LOADING='EAGER',STRATA_MEASUREMENT_ROOT=directory)
            child=wrapper.native_environment(plan,parent)
            self.assertNotIn('LD_PRELOAD',parent)
            self.assertEqual(child['LD_PRELOAD'],str(library))
            for environment in ({},parent|dict(LD_PRELOAD='/another-probe')):
                with self.assertRaises(RuntimeError):wrapper.native_environment(plan,environment)
            library.write_bytes(b'changed')
            with self.assertRaises(RuntimeError):wrapper.native_environment(plan,parent)


if __name__=='__main__':unittest.main()
