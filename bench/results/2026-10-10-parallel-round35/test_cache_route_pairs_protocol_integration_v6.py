"""SOURCE ONLY fake-PTY/CPU qualification; never imports controller top-level."""
import ast
import errno
import hashlib
import io
import json
from pathlib import Path
import re
import tempfile
import time
import types
import unittest
from bounded_engine_protocol_v1 import BoundedProtocolLines, ProtocolFramingError

CONTROLLER=Path(__file__).with_name('run_owned_cache_route_pairs_v0141_code32k_v6.py')


def functions(directory):
    # Execute only named source functions on root's later CPU run, with fake
    # dependencies. No CLI/admission/OwnedGdb/GPU/top-level controller execution.
    tree=ast.parse(CONTROLLER.read_text())
    names={'text_bytes','reserve_write','protocol_line_kind','receive_stdout','available_stdout',
           'boundary','event','save','failure_save','encoded_failure','startup_step','transport_small_evidence'}
    selected=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    if {node.name for node in selected}!=names: raise AssertionError('expected source functions changed')
    reads=[]
    def fake_read(fd,maximum):
        if not reads: raise BlockingIOError()
        value=reads.pop(0)
        if isinstance(value,BaseException): raise value
        assert len(value)<=maximum
        return value
    scope=dict(Path=Path,json=json,re=re,time=time,errno=errno,hashlib=hashlib,FAILURE_RESERVE=1024*1024,
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
        self.s['record']['quit_sent']=True
        self.s['g']=types.SimpleNamespace(exit_code=0,exit_signal=None,stops=['exited'])
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
                    pair_gate_passed=True,pair_shape_gate_passed=True,protocol_boundary_gate_passed=True,new_fault_messages=[],cleanup={},error=None)
        self.assertTrue(eval(expression,{'record':record}))
        for change in ({'cleanup':{'forced':True}},{'exit_code':1},{'exit_signal':'SIGABRT'},
                       {'protocol_boundary_gate_passed':False},{'pair_shape_gate_passed':False}):
            with self.subTest(change=change):
                self.assertFalse(eval(expression,{'record':dict(record,**change)}))
    def test_eagain_never_fabricates_eof_even_after_normal_exit(self):
        self.s['g']=types.SimpleNamespace(exit_code=0,exit_signal=None,stops=['exited'])
        self.s['record']['quit_sent']=True
        self.assertFalse(self.s['available_stdout']())
        self.s['boundary']('after-QUIT-exit',allow_eof=True)
        self.assertFalse(self.s['protocol'].state().eof)
        self.assertNotIn('transport_eof',self.s['record'])
    def test_eio_reason_offset_and_exit_order_are_recorded(self):
        self.s['receive_stdout'](b'DONE\n');self.s['protocol'].pop_line()
        self.s['g']=types.SimpleNamespace(exit_code=None,exit_signal=None,stops=['resumed'])
        self.s['record']['quit_sent']=True
        self.s['reads'][:]=[OSError(errno.EIO,'closed')]
        self.assertTrue(self.s['available_stdout']())
        evidence=self.s['record']['transport_eof']
        self.assertEqual(evidence['reason'],'pty_eio');self.assertEqual(evidence['raw_byte_offset'],5)
        self.assertEqual(evidence['received_bytes'],5)
        self.assertIsNone(evidence['inferior_exit_code_at_observation'])
        self.assertFalse(evidence['inferior_poll_state_at_observation']['exit_known'])
        self.assertEqual(evidence['inferior_poll_state_at_observation']['last_stop'],'resumed')
        # A later normal exit cannot erase the original early-EOF observation.
        self.s['g'].exit_code=0
        self.assertIsNone(evidence['inferior_exit_code_at_observation'])
    def test_observed_zero_read_after_exit_is_distinct_from_eio(self):
        self.s['g']=types.SimpleNamespace(exit_code=0,exit_signal=None,stops=['exited'])
        self.s['record']['quit_sent']=True;self.s['reads'][:]=[b'']
        self.assertTrue(self.s['available_stdout']())
        self.assertEqual(self.s['record']['transport_eof']['reason'],'zero_read')
        self.assertEqual(self.s['record']['transport_eof']['inferior_exit_code_at_observation'],0)
    def test_non_eio_read_error_remains_failure_evidence(self):
        self.s['reads'][:]=[OSError(errno.EBADF,'bad descriptor')]
        with self.assertRaises(OSError): self.s['available_stdout']()
        self.assertEqual(self.s['record']['transport_read_error']['errno'],errno.EBADF)
        self.assertFalse(self.s['protocol'].state().eof)
    def test_exact_final_payload_size_near_reserve_and_bounded_fallback(self):
        encode=self.s['encoded_failure'];reserve=1024*1024
        compact={'padding':'','math_gate_passed':True,'healthy':False,'text_budget_gate_passed':False}
        # Adjust independent filler to EXACTLY the reserve; encoded size fields
        # themselves are included. No constant monkeypatch or unchecked write.
        for unused in range(8):
            payload=encode(compact,0,64*reserve)
            gap=reserve-len(payload)
            if gap==0: break
            compact['padding'] += 'x'*gap if gap>0 else ''
            if gap<0: compact['padding']=compact['padding'][:gap]
        payload=encode(compact,0,64*reserve)
        self.assertEqual(len(payload),reserve)
        self.assertEqual(json.loads(payload)['serialized_failure_receipt_bytes'],len(payload))
        self.s['record'].update(math_gate_passed=True,error='x'*reserve,cleanup={'forced':True})
        self.s['failure_save'](RuntimeError('budget failed after mathematical PASS'))
        stored=(self.directory/'record.json').read_bytes();record=json.loads(stored)
        self.assertLessEqual(len(stored),reserve)
        self.assertTrue(record['failure_metadata_reserve_exceeded'])
        self.assertTrue(record['math_gate_passed']);self.assertFalse(record['text_budget_gate_passed'])
        self.assertFalse(record['healthy']);self.assertTrue(record['cleanup']['forced'])
        self.assertEqual(record['serialized_failure_receipt_bytes'],len(stored))
    def test_math_pass_is_preserved_in_ordinary_budget_failure(self):
        self.s['record']['math_gate_passed']=True
        self.s['failure_save'](RuntimeError('ordinary budget failure'))
        stored=(self.directory/'record.json').read_bytes();record=json.loads(stored)
        self.assertTrue(record['math_gate_passed']);self.assertFalse(record['healthy'])
        self.assertFalse(record['text_budget_gate_passed'])
        self.assertEqual(record['serialized_failure_receipt_bytes'],len(stored))

    def test_eof_before_quit_still_fails_and_preserves_origin(self):
        self.s['g']=types.SimpleNamespace(exit_code=0,exit_signal=None,stops=['exited'])
        self.s['reads'][:]=[OSError(errno.EIO,'closed')]
        with self.assertRaisesRegex(RuntimeError,'early transport EOF before QUIT'):
            self.s['available_stdout']()
        self.assertEqual(self.s['record']['transport_eof']['reason'],'pty_eio')
        self.assertFalse(self.s['record']['transport_eof']['quit_sent'])

    def test_join_provenance_and_compact_replay_guards(self):
        source=CONTROLLER.read_text()
        self.assertIn("cli.add_argument('--shape-helper-sha256', required=True)",source)
        self.assertIn("assert len(resident_pairs)==128",source)
        self.assertIn("layer_formats=formats_from_metadata((pack/'native_experts.txt').read_bytes(),\n    'd9ac2dfa3ee63c55c9c6a6db26f72da0cec0ee41733f007e5cbbbba71617aa5d')",source)
        self.assertIn("shape=reconcile_shapes(parsed,resident_pairs,layer_formats,request['native_dispatch_histogram'])",source)
        self.assertIn("sum(q['cache_route_pairs']['pair_count']+3",source)
        self.assertIn("latest_D_path=base/'owned-repeat-capture-v0141-two-full-diagnostic-r3/record.json'",source)
        self.assertNotIn("two-full-diagnostic-r4/record.json",source)

    def test_actual_info_literal_then_ready_is_accepted(self):
        # Independent literal from failed r5 receipt; not producer serialization.
        info='INFO context=262144 kv=int8 kv_resident=32768 expert_slots=128 expert_cache_mib=325 expert_slots_primary=128 expert_cache_primary_mib=325 spec=4 mtp_max=0 lookup=0 vram_free_mib=1565 cvec=0 arena_mib=47962 pool_workers=5 pcie_frac=0.00 spec_min_p=0.00 conversation_cache_mib=0 conversation_cache_slots=4 conversation_cache_min_free_mib=2560 tail_role_token=-1 vram_elastic=0 engine=0.1.41'
        seen=[]
        self.assertEqual(self.s['startup_step'](info,seen),'progress')
        self.assertEqual(self.s['startup_step']('READY 262144 stop',seen),'terminal')
        self.assertEqual(seen,['progress','terminal'])
        for free in ('0','9999'):
            self.assertEqual(self.s['protocol_line_kind'](info.replace('vram_free_mib=1565','vram_free_mib='+free),'startup'),'progress')
    def test_startup_missing_duplicate_order_and_config_rejected(self):
        info='INFO context=262144 kv=int8 kv_resident=32768 expert_slots=128 expert_cache_mib=325 expert_slots_primary=128 expert_cache_primary_mib=325 spec=4 mtp_max=0 lookup=0 vram_free_mib=1565 cvec=0 arena_mib=47962 pool_workers=5 pcie_frac=0.00 spec_min_p=0.00 conversation_cache_mib=0 conversation_cache_slots=4 conversation_cache_min_free_mib=2560 tail_role_token=-1 vram_elastic=0 engine=0.1.41'
        step=self.s['startup_step']
        with self.assertRaisesRegex(ValueError,'startup requires exactly INFO then READY'): step('READY 262144 stop',[])
        seen=[];step(info,seen)
        with self.assertRaisesRegex(ValueError,'startup requires exactly INFO then READY'): step(info,seen)
        step('READY 262144 stop',seen)
        with self.assertRaisesRegex(ValueError,'startup requires exactly INFO then READY'): step('READY 262144 stop',seen)
        for old,new in [('arena_mib=47962','arena_mib=47963'),('pool_workers=5','pool_workers=6'),
                        ('kv=int8','kv=fp16'),('spec=4','spec=3'),('pcie_frac=0.00','pcie_frac=0.01'),
                        ('engine=0.1.41','engine=0.1.42'),('vram_free_mib=1565','vram_free_mib=-1'),
                        ('vram_free_mib=1565','vram_free_mib=01565')]:
            with self.subTest(new=new),self.assertRaisesRegex(ValueError,'unexpected startup protocol response'):
                step(info.replace(old,new),[])
        with self.assertRaisesRegex(ValueError,'unexpected startup protocol response'): step('READY 32768 stop',[])
        with self.assertRaisesRegex(RuntimeError,'ERR startup failure'): step('ERR startup failure',[])
    def test_missing_ready_eof_before_requests_fails(self):
        info='INFO context=262144 kv=int8 kv_resident=32768 expert_slots=128 expert_cache_mib=325 expert_slots_primary=128 expert_cache_primary_mib=325 spec=4 mtp_max=0 lookup=0 vram_free_mib=1565 cvec=0 arena_mib=47962 pool_workers=5 pcie_frac=0.00 spec_min_p=0.00 conversation_cache_mib=0 conversation_cache_slots=4 conversation_cache_min_free_mib=2560 tail_role_token=-1 vram_elastic=0 engine=0.1.41'
        seen=[];self.s['startup_step'](info,seen)
        self.s['reads'][:]=[b'']
        with self.assertRaisesRegex(RuntimeError,'early transport EOF'): self.s['available_stdout']()
        self.assertEqual(seen,['progress'])

    def test_oversized_fallback_preserves_fixed_transport_tuple(self):
        self.s['record'].update(math_gate_passed=True,error='x'*(1024*1024),
            transport_eof={'reason':'pty_eio','raw_byte_offset':12345,'inferior_exit_code_at_observation':None,
                'inferior_exit_signal_at_observation':None,'inferior_poll_state_at_observation':{'exit_known':False},'quit_sent':True},
            transport_read_error={'errno':errno.EBADF,'error':'large error'*1000})
        self.s['failure_save'](RuntimeError('oversized metadata'))
        payload=(self.directory/'record.json').read_bytes();stored=json.loads(payload)
        self.assertEqual(stored['transport_small_evidence'],['pty_eio',12345,False,True,errno.EBADF,None,None])
        self.assertEqual(stored['serialized_failure_receipt_bytes'],len(payload))
        self.assertLessEqual(len(payload),1024*1024);self.assertTrue(stored['math_gate_passed'])
        self.assertFalse(stored['healthy']);self.assertFalse(stored['text_budget_gate_passed'])

    def test_oversized_read_error_fallback_preserves_errno_and_offset(self):
        self.s['record'].update(error='x'*(1024*1024),transport_read_error={'errno':errno.EBADF,'raw_byte_offset':999})
        self.s['failure_save'](RuntimeError('oversized read error'))
        payload=(self.directory/'record.json').read_bytes();stored=json.loads(payload)
        self.assertEqual(stored['transport_small_evidence'],['read_error',999,None,False,errno.EBADF,None,None])
        self.assertEqual(stored['serialized_failure_receipt_bytes'],len(payload))
        self.assertLessEqual(len(payload),1024*1024)

    def test_required_source_guards_present_and_old_gap_absent(self):
        source=CONTROLLER.read_text()
        self.assertIn("cli.add_argument('--protocol-helper-sha256', required=True)",source)
        self.assertIn("QUIT transport EOF not observed within deadline",source)
        self.assertNotIn("protocol.finish() # Unterminated EOF",source)
        self.assertIn("boundary('after-DONE-'",source)
        self.assertIn("boundary('after-SAVE-terminal')",source)
        self.assertIn("boundary('after-QUIT-exit'",source)
        self.assertIn("final_total=text_bytes(out)",source)
        self.assertNotIn('pending = bytearray()',source)
        self.assertNotIn('return value.decode().strip()',source)
        self.assertIn("'deferred_R42_pair_NT_and_per_type_join':False",source)


if __name__=='__main__': unittest.main()
