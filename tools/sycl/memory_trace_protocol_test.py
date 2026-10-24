#!/usr/bin/env python3
"""CPU-only regression controls for known driver logs after native QUIT.

Main-authored Strata source, repository LICENSE; no native process or GPU call.
"""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.dont_write_bytecode=True
import memory_trace_model as trace

CONTAINER=json.loads(trace.RESULT.read_text())
CONTRACT=CONTAINER['owner_trace_obligations']['neo_logging_source_review']['output_contract']


class FakeIO:
    def __init__(self,raw,eof=True,dead=True,error=None):
        self.pending=bytearray(raw);self.eof=eof;self.calls=0;self.error=error
        self.exit_watch=SimpleNamespace(poll=lambda _: [1] if dead else [])
    def exit(self,seconds):
        self.calls+=1
        if self.error:raise RuntimeError(self.error)
        if self.pending.strip():raise RuntimeError('unexpected trailing protocol')


def wrapper(raw,**kwargs):
    io=FakeIO(raw,**kwargs);record={'source_review':{'output_contract':CONTRACT}}
    adapter=trace.AtBoundary(io,{'direct_identity':dict(pid=1,start_ticks=1,sid=1)},record)
    return adapter,io,record


class Protocol(unittest.TestCase):
    def test_no_extra_output(self):
        a,io,r=wrapper(b'');a.exit(1);self.assertEqual(io.calls,1)
        self.assertNotIn('trailing_neo_records',r)
    def test_recognized_complete_records(self):
        raw=(b'unbind BO-38 from VM 1, vmHandleId = 0, range: ffffd5592e800000 - ffffd55936810000, size: 134283264, result: 0\n'
             b'Calling gem close on handle: BO-38\n')
        a,io,r=wrapper(raw);a.exit(1);self.assertEqual(r['trailing_neo_records']['records'],2)
        self.assertFalse(io.pending);self.assertEqual(io.calls,2)
    def test_unexpected_protocol_or_partial_match(self):
        for raw in (b'DONE 0 0\n',b'unknown driver line\n',b'Calling gem close on handle: BO-1 junk\n',
                    b'junk ThreadID: 1 Type: BUFFER Pool: LocalMemory Root index: 0 Size: 1 CPU VA: NULL GPU VA: 0 - 1 Total sys mem allocated: 0 Total lmem allocated: 1\n'):
            with self.subTest(raw=raw):
                a,io,r=wrapper(raw)
                with self.assertRaises(RuntimeError):a.exit(1)
                self.assertEqual(bytes(io.pending),raw)
    def test_truncated_record(self):
        a,io,r=wrapper(b'Calling gem close on handle: BO-38')
        with self.assertRaises(RuntimeError):a.exit(1)
    def test_nonzero_bind(self):
        a,io,r=wrapper(b'unbind BO-38 from VM 1, vmHandleId = 0, range: 1000 - 2000, size: 4096, result: -1\n')
        with self.assertRaises(RuntimeError):a.exit(1)
        self.assertTrue(io.pending)
    def test_byte_budget(self):
        a,io,r=wrapper(b'Calling gem close on handle: BO-1\n'*20000)
        with self.assertRaises(RuntimeError):a.exit(1)
    def test_exit_and_eof_required(self):
        for kwargs in ({'eof':False},{'dead':False}):
            with self.subTest(kwargs=kwargs):
                a,io,r=wrapper(b'Calling gem close on handle: BO-1\n',**kwargs)
                with self.assertRaises(RuntimeError):a.exit(1)
                self.assertTrue(io.pending)
    def test_other_runtime_error_preserved(self):
        a,io,r=wrapper(b'Calling gem close on handle: BO-1\n',error='protocol send deadline')
        with self.assertRaisesRegex(RuntimeError,'^protocol send deadline$'):a.exit(1)
    def test_actual_failed_capture_remains_failed(self):
        failed=CONTAINER['neo_memory_model_v1'];e=failed['observer_protocol_failure']['stdout']
        self.assertFalse(failed['passed']);self.assertFalse(failed['complete'])
        raw=Path(e['path']).read_bytes();self.assertEqual(len(raw),e['bytes'])
        self.assertEqual(hashlib.sha256(raw).hexdigest(),e['sha256'])
        marker=raw.rfind(b'\nDONE ');self.assertGreater(marker,0)
        tail=raw[marker:].split(b'\n',2)[2];a,io,r=wrapper(tail);a.exit(1)
        self.assertEqual(r['trailing_neo_records']['records'],2172)
        self.assertEqual(r['trailing_neo_records']['bytes'],154602)
        self.assertFalse(failed['passed'])


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(Protocol))
    print(json.dumps(dict(tests=result.testsRun,passed=result.wasSuccessful(),GPU_executed=False)))
    sys.exit(0 if result.wasSuccessful() else 1)
