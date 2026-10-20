#!/usr/bin/env python3
"""Regress actual UR enum parsing and reject mixed/ambiguous child evidence."""
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent/'sycl'))
import ur_validation as probe


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


if __name__=='__main__':unittest.main()
