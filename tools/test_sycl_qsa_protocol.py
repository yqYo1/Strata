#!/usr/bin/env python3
"""Exercise real owned pipe/EOF handling without a GPU; repository LICENSE."""
import fcntl
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent/'sycl'))
import check_qsa_subgroup_rows as q

FOOTER='fixture_status=PASS model_quality=UNRUN model_PP=UNRUN physical262144_model=UNRUN faults=EXTERNAL_REQUIRED'


class QsaProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lock=(q.B/'owned-v0141-measurement.lock').open('a+')
        cls.addClassCleanup(cls.lock.close)
        fcntl.flock(cls.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        q.base.guards.no_model_owners([])

    def owned(self,mode):
        child=("import sys\n"
            "for phase in ('before-runtime','runtime-ready','after-cleanup'):\n"
            " print('QSA_BF16_READY '+phase,flush=True)\n"
            " assert sys.stdin.buffer.read(1)==b'X'\n"
            f"if {mode!r}!='missing': print({FOOTER!r},flush=True)\n"
            f"if {mode!r}=='extra': print('UNPLANNED_TRAILING_LINE',flush=True)\n")
        with tempfile.TemporaryDirectory(prefix='qsa-protocol-',dir=q.B) as name:
            owner=q.base.supervision.Owner(Path(name)/'owned')
            def interact(io,command):
                for phase in ('before-runtime','runtime-ready','after-cleanup'):
                    self.assertEqual(io.line(5),'QSA_BF16_READY '+phase);io.send(b'X')
                if hasattr(q,'finish_protocol'):q.finish_protocol(io)
                else:io.proc.stdin.close();io.exit(5)
            command,out,err=owner.run('protocol',[sys.executable,'-I','-c',child],
                dict(PATH='/usr/bin:/bin',LANG='C.UTF-8',LC_ALL='C.UTF-8',
                     PYTHONDONTWRITEBYTECODE='1',TMPDIR=name),q.P,interaction=interact,
                wall=15,cpu=10,rss_cap=256<<20,text_cap=64<<10,file_cap=64<<10,total_cap=1<<20)
            self.assertTrue(command['direct_child_reaped']);self.assertTrue(command['session_empty'])
            self.assertFalse(command['survivors']);self.assertEqual(err.stat().st_size,0)
            return command

    def test_expected_footer_allows_closed_session(self):
        result=self.owned('normal')
        self.assertTrue(q.base.supervision.closed(result),result.get('error'))
        self.assertEqual(result['exit_code'],0)

    def test_unexpected_trailing_output_rejected(self):
        result=self.owned('extra')
        self.assertFalse(q.base.supervision.closed(result))
        self.assertIn('unexpected trailing protocol',result['error'])

    def test_missing_footer_rejected(self):
        result=self.owned('missing')
        self.assertFalse(q.base.supervision.closed(result))
        self.assertIn('error',result)


if __name__=='__main__':unittest.main(verbosity=2)
