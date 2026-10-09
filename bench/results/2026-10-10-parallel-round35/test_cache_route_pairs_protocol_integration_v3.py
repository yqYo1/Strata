"""SOURCE ONLY fake-PTY/CPU qualification; never imports controller top-level."""
import ast
import errno
import io
import json
from pathlib import Path
import re
import tempfile
import time
import types
import unittest
from bounded_engine_protocol_v1 import BoundedProtocolLines, ProtocolFramingError

CONTROLLER=Path(__file__).with_name('run_owned_cache_route_pairs_v0141_code32k_v3.py')


def functions(directory):
    # Execute only named source functions on root's later CPU run, with fake
    # dependencies. No CLI/admission/OwnedGdb/GPU/top-level controller execution.
    tree=ast.parse(CONTROLLER.read_text())
    names={'text_bytes','reserve_write','protocol_line_kind','receive_stdout','available_stdout',
           'boundary','event','save','failure_save'}
    selected=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    if {node.name for node in selected}!=names: raise AssertionError('expected source functions changed')
    reads=[]
    def fake_read(fd,maximum):
        if not reads: raise BlockingIOError()
        value=reads.pop(0)
        if isinstance(value,BaseException): raise value
        assert len(value)<=maximum
        return value
    scope=dict(Path=Path,json=json,re=re,time=time,errno=errno,FAILURE_RESERVE=1024*1024,
               os=types.SimpleNamespace(read=fake_read),out=directory,master=1,reads=reads,
               raw=io.BytesIO(),events=io.StringIO(),started=time.monotonic(),current=None,g=None,
               r=types.SimpleNamespace(calls=[]),protocol=BoundedProtocolLines(cr_policy='reject'),
               record={'log_limit_bytes':64*1024**2,'active':False,'completed':False,'healthy':False,
                       'math_gate_passed':False,'cleanup':{},'error':'fixture'})
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(CONTROLLER),'exec'),scope)
    return scope


class SourceIntegrationFixtures(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.directory=Path(self.temp.name);self.s=functions(self.directory)
    def test_done_burst_tail_not_attributed_to_next_request(self):
        self.s['reads'][:]=[b'DONE 64\nT 999\n']
        self.s['available_stdout']()
        self.assertEqual(self.s['protocol'].pop_line(),b'DONE 64\n')
        with self.assertRaisesRegex(ValueError,'unexpected stdout tail at before-GEN'):
            self.s['boundary']('before-GEN')
        self.assertEqual(self.s['protocol'].queued_tail(),(b'T 999\n',))
        self.assertEqual(self.s['raw'].getvalue(),b'DONE 64\nT 999\n')
    def test_done_partial_tail_and_eof_are_not_clean(self):
        self.s['reads'][:]=[b'DONE\nT ']
        self.s['available_stdout']();self.s['protocol'].pop_line()
        with self.assertRaisesRegex(ValueError,'unexpected stdout tail'): self.s['boundary']('after-DONE')
        self.s['reads'][:]=[OSError(errno.EIO,'PTY closed')]
        with self.assertRaisesRegex(ProtocolFramingError,'EOF with unterminated protocol line'):
            self.s['boundary']('QUIT',allow_eof=True)
        self.assertEqual(self.s['protocol'].pending_tail(),b'T ')
    def test_quit_tail_rejected_normal_eof_allowed(self):
        self.s['reads'][:]=[b'UNEXPECTED\n']
        with self.assertRaisesRegex(ValueError,'unexpected stdout tail'):
            self.s['boundary']('QUIT',allow_eof=True)
        self.s['protocol'].pop_line();self.s['reads'][:]=[b'']
        self.s['boundary']('QUIT-exit',allow_eof=True)
        self.assertTrue(self.s['protocol'].state().eof)
    def test_save_grammar_is_separate_from_gen(self):
        kinds=self.s['protocol_line_kind']
        for line,kind in [('SESSION 0 4096','progress'),('SWAIT commit 1','progress'),
                          ('SAVED 66 4096 1.0','terminal'),('SERR io 0 failed','error'),('ERR commit failed','error')]:
            self.assertEqual(kinds(line,'save'),kind)
        for line in ('SESSION 0 4096','SWAIT commit 1','SAVED 66 4096 1.0','SERR io 0 failed'):
            with self.assertRaisesRegex(ValueError,'unexpected gen protocol response'): kinds(line,'gen')
        with self.assertRaisesRegex(ValueError,'unexpected save protocol response'): kinds('T 99','save')
    def test_line_bound_burst_and_raw_error_evidence(self):
        self.s['reads'][:]=[b'x'*65535,b'\nGOOD\n']
        self.s['available_stdout']()
        self.assertEqual(len(self.s['protocol'].pop_line()),65536)
        self.assertEqual(self.s['protocol'].pop_line(),b'GOOD\n')
        self.s['reads'][:]=[b'x'*65536]
        with self.assertRaisesRegex(ProtocolFramingError,'unterminated line cannot fit LF'):
            self.s['available_stdout']()
        self.assertTrue(self.s['raw'].getvalue().endswith(b'x'*65536))
    def test_final_json_reservation_failure_retains_failed_receipt(self):
        # Sparse file: bounded logical budget stimulus, no 63MiB memory string.
        (self.directory/'record.json').write_text('{}\n')
        with (self.directory/'diagnostic.log').open('wb') as stream: stream.truncate(63*1024**2)
        self.s['record']['payload']='x'*4096
        with self.assertRaisesRegex(RuntimeError,'aggregate text budget reservation exceeded'):
            self.s['save']()
        self.s['failure_save'](RuntimeError('final record reservation exceeded'))
        record=json.loads((self.directory/'record.json').read_text())
        self.assertFalse(record['healthy']);self.assertFalse(record['text_budget_gate_passed'])
        self.assertFalse(record['math_gate_passed']);self.assertTrue(record['final_receipt_compacted'])
        self.assertLessEqual(self.s['text_bytes'](self.directory),64*1024**2)
        self.assertFalse((self.directory/'record.json.tmp').exists())
    def test_external_overflow_cannot_be_relabelled_success(self):
        with (self.directory/'diagnostic.log').open('wb') as stream: stream.truncate(64*1024**2+1)
        self.s['failure_save'](RuntimeError('external writer exceeded cap'))
        record=json.loads((self.directory/'record.json').read_text())
        self.assertFalse(record['healthy']);self.assertFalse(record['text_budget_gate_passed'])
        self.assertGreater(self.s['text_bytes'](self.directory),64*1024**2)
    def test_normal_exit_and_cleanup_failure_remain_distinct(self):
        tree=ast.parse(CONTROLLER.read_text())
        assignments=[node for node in ast.walk(tree) if isinstance(node,ast.Assign)
                     and any(isinstance(target,ast.Subscript) and isinstance(target.slice,ast.Constant)
                             and target.slice.value=='healthy' for target in node.targets)]
        final=next(node.value for node in assignments if isinstance(node.value,ast.BoolOp))
        expression=compile(ast.Expression(final),str(CONTROLLER),'eval')
        record=dict(completed=True,exit_code=0,exit_signal=None,histogram_gate_passed=True,
                    pair_gate_passed=True,protocol_boundary_gate_passed=True,new_fault_messages=[],cleanup={},error=None)
        self.assertTrue(eval(expression,{'record':record}))
        for change in ({'cleanup':{'forced':True}},{'exit_code':1},{'exit_signal':'SIGABRT'},
                       {'protocol_boundary_gate_passed':False}):
            with self.subTest(change=change):
                self.assertFalse(eval(expression,{'record':dict(record,**change)}))
    def test_required_source_guards_present_and_old_gap_absent(self):
        source=CONTROLLER.read_text()
        self.assertIn("cli.add_argument('--protocol-helper-sha256', required=True)",source)
        self.assertIn("protocol.finish() # Unterminated EOF",source)
        self.assertIn("boundary('after-DONE-'",source)
        self.assertIn("boundary('after-SAVE-terminal')",source)
        self.assertIn("boundary('after-QUIT-exit'",source)
        self.assertIn("final_total=text_bytes(out)",source)
        self.assertNotIn('pending = bytearray()',source)
        self.assertNotIn('return value.decode().strip()',source)
        self.assertIn("'deferred_R42_pair_NT_and_per_type_join':True",source)


if __name__=='__main__': unittest.main()
