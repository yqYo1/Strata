"""Owned qualified869 long-prompt numerical reference, diagnostic only.

Source-only until root review/execution. Four fresh whole-input requests in one
process; original baseline argv/default task shape, never performance evidence.
"""
from pathlib import Path
import argparse
import array
import datetime
import errno
import fcntl
import hashlib
import json
import math
import os
import re
import selectors
import shutil
import stat
import struct
import subprocess
import sys
import time
import tty
import types

BASE=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
OBSERVER=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
ROOT=OBSERVER.parent/'sync-upstream-v0.1.41-20261009'
COMMIT='1eb89482a4afd20277ae0405780ed4f8eb98eb20'
CURRENT_HEAD='23268953314d12588fd3f496414a46bd426a7306'
ADMISSION_SHA='d98044784022fa105f025c8ce394d81daeb601051f6feebf28f05a9fa0b5af4c'
GGML_ROOT=Path('/home/yayoi/orca/workspaces/Strata/ggml-sycl-pinned')
BINARY_SHA='86972697ecb1903750d202f8be29a7a1da351eb3415678c3e3b6af790804d1c8'
FULL_SHA='168d7896cbe3a60dfc604a08b6fe310f16d823a17d13d8a013260bfc7facb0a4'
BUILD_SHA='20589f1486dedaf825052e95560155acd30e6017b7967e2df476b5dd41989c43'
INPUT_OWNER_SHA='974dbdfd6e59a2b61c65a06f4555750edaaa095d62f1054ad33898d88f52986b'
MANIFEST_SHA='4881239ef30f643a55ad13c1bff17c49bcc04abb921139bc97975b5eacbf8e40'
CANDIDATE_SHA='dc43ba8f0ddcd6e6ef22ae8e2e10457c70b7fb0b1e6b7868ec684be12a544b93'
TRANSPORT_SOURCE_SHA='9304f84dfce23b099c02d0090ea3b8f9b158100d157b0175ab3e4cfda1a9f423'
FRAMER_SHA='aa72bd7f1a9babf7c6e325b10634f1a046fe78fc2b2738e691d90624197131c1'
OWNED_SHA='61e1d206bf57bb320051cdd65943d23722bc523a602d80c54b7e7a2ad0d3f0d1'
HEALTH_SHA='2f90005f79d2309747d917cff51dc53d8ee6192d68cb1feca074f84af8b0796c'
RULES_SHA='f6c6e4fe4765b51563bef3914446213fd6fab9e1c0832d2b895f1f4ea4079229'
CPU_GUARD_SHA='e9925535b6a1753c89f71c32fe4b67b52c20ae7d423df269cccabe14d3b64d21'
COMPARE_SHA='9a4f55a4760d316045c1ec5179a946de9a9024f7e42fe09032012ceae2475a72'
PACK_PINS={'native_experts.txt':'d9ac2dfa3ee63c55c9c6a6db26f72da0cec0ee41733f007e5cbbbba71617aa5d','index.txt':'b2a014878c3b61d6a6f915e194055a5e68ede34f9465dd1cb13249d902f008ac','conversions.json':'51df3cd6ffd0d38c2da8e2d3a95a604b4a96c20b06fc639a9bf477282a42b86a'}
PROFILE_SHA='8f59b4aa8873209dff11c11e37bcda9529a1335b724a1afeea37bf6388975baf'


def require(ok,message):
    if not ok: raise ValueError(message)


def digest(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def verified(path,expected,cap=16*1024**2):
    require(re.fullmatch('[0-9a-f]{64}',expected),'exact SHA256 required')
    pre=os.lstat(path);require(stat.S_ISREG(pre.st_mode),'regular file precheck')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        before=os.fstat(fd)
        require((before.st_dev,before.st_ino)==(pre.st_dev,pre.st_ino) and stat.S_ISREG(before.st_mode) and before.st_uid==os.getuid() and before.st_nlink==1 and 0<=before.st_size<=cap,'bounded owned regular singlelink')
        data=bytearray()
        while len(data)<before.st_size:
            chunk=os.read(fd,min(65536,before.st_size-len(data)));require(chunk,'short read');data.extend(chunk)
        require(not os.read(fd,1),'file grew')
        after=os.fstat(fd)
        require(all(getattr(before,k)==getattr(after,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns','st_uid','st_nlink')),'file changed')
        require(hashlib.sha256(data).hexdigest()==expected,'SHA256 mismatch')
        return bytes(data)
    finally: os.close(fd)


def pinned_module(name,path,expected):
    source=verified(path,expected,262144)
    module=types.ModuleType(name);module.__file__=str(path);sys.modules[name]=module
    exec(compile(source,str(path),'exec'),module.__dict__)
    return module


def git(*args): return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()


def binary_artifact_bytes(directory):
    return sum(p.stat().st_size for p in directory.rglob('*.bin') if p.is_file())


FAILURE_RESERVE = 1024*1024


def text_bytes(directory):
    # Count actual text artifacts, INCLUDING record and temporary serializations.
    return sum(p.stat().st_size for p in directory.rglob('*') if p.is_file() and p.suffix != '.bin')


def reserve_write(directory, additional, limit, reserve=FAILURE_RESERVE):
    used=text_bytes(directory)
    if used+additional > limit-reserve:
        raise RuntimeError('aggregate text budget reservation exceeded')
    return used


def save():
    record['elapsed_seconds']=time.monotonic()-started
    record['steps']=r.calls if r is not None else [];record['active_request']=current
    if g: record.update(inferior=g.inferior,debugger=g.debugger_identity)
    payload=(json.dumps(record,indent=2)+'\n').encode('utf8')
    # Atomic replacement temporarily has BOTH old record and new .tmp on disk.
    reserve_write(out,len(payload),record['log_limit_bytes'])
    temporary=out/'record.json.tmp'
    with temporary.open('xb') as stream: stream.write(payload)
    temporary.replace(out/'record.json')


def event(kind, **fields):
    value=json.dumps({'kind':kind,'elapsed_seconds':time.monotonic()-started,**fields})+'\n'
    reserve_write(out,len(value.encode('utf8')),record['log_limit_bytes'])
    events.write(value)


def protocol_line_kind(value, phase):
    number=r'[0-9]+(?:\.[0-9]+)?'
    if phase=='startup':
        prefix='INFO context=262144 kv=int8 kv_resident=32768 expert_slots=128 expert_cache_mib=325 expert_slots_primary=128 expert_cache_primary_mib=325 spec=4 mtp_max=0 lookup=0 vram_free_mib='
        suffix=' cvec=0 arena_mib=47962 pool_workers=5 pcie_frac=0.00 spec_min_p=0.00 conversation_cache_mib=0 conversation_cache_slots=4 conversation_cache_min_free_mib=2560 tail_role_token=-1 vram_elastic=0 engine=0.1.41'
        match=re.fullmatch(re.escape(prefix)+r'(0|[1-9][0-9]{0,19})'+re.escape(suffix),value)
        if match and int(match[1])<(1<<64): return 'progress'
        if value=='READY 262144 stop': return 'terminal'
    if phase=='gen':
        patterns=(r'(?:RESUME|REUSED|T) [0-9]+',r'PP [0-9]+ [0-9]+ '+number+' '+number,
                  r'LP -?[0-9]+\.[0-9]+(?: [0-9]+:-?[0-9]+\.[0-9]+)*')
        if any(re.fullmatch(pattern,value) for pattern in patterns): return 'progress'
        if value.startswith('DONE ') and len(value.split())==16: return 'terminal'
    if phase=='save':
        if re.fullmatch(r'SESSION [0-9]+ [0-9]+',value) or re.fullmatch(r'SWAIT [A-Za-z0-9_-]+ [0-9]+',value): return 'progress'
        if re.fullmatch(r'SAVED [0-9]+ [0-9]+ '+number,value): return 'terminal'
        if value.startswith('SERR ') and len(value.split())>=4: return 'error'
    if value.startswith('ERR ') and len(value)>4: return 'error'
    raise ValueError('unexpected '+phase+' protocol response: '+repr(value))


def startup_step(value, seen):
    kind=protocol_line_kind(value,'startup')
    if kind=='error': raise RuntimeError(value)
    expected='progress' if not seen else 'terminal' if seen==['progress'] else None
    if kind!=expected: raise ValueError('startup requires exactly INFO then READY')
    seen.append(kind)
    return kind


def transport_small_evidence():
    eof=record.get('transport_eof') or {};read_error=record.get('transport_read_error') or {}
    def small_int(value):
        return value if type(value) is int and -(1<<63)<=value<(1<<64) else None
    # Fixed bounded schema: reason,offset,exit_known,quit_sent,non-EIO errno,
    # exit_code,exit_signal. No pending/raw tails or unbounded transport repr.
    reason=eof.get('reason')
    if reason not in ('pty_eio','zero_read'): reason='read_error' if read_error else None
    signal=eof.get('inferior_exit_signal_at_observation')
    if type(signal) is not str or len(signal)>32: signal=None
    known=eof.get('inferior_poll_state_at_observation',{}).get('exit_known')
    if type(known) is not bool: known=None
    return [reason,small_int(eof.get('raw_byte_offset',read_error.get('raw_byte_offset'))),known,
            bool(eof.get('quit_sent',record.get('quit_sent',False))),small_int(read_error.get('errno')),
            small_int(eof.get('inferior_exit_code_at_observation')),signal]


def receive_stdout(data):
    # Retain every read before framing. If the disk reservation itself fails,
    # keep that bounded chunk in failure metadata instead of losing evidence.
    try: reserve_write(out,len(data),record['log_limit_bytes'])
    except BaseException:
        record['unwritten_stdout_chunk_hex']=data.hex();raise
    raw.write(data);raw.flush()
    protocol.feed(data)
    event('stdout',bytes=len(data),raw_offset=raw.tell())


def available_stdout():
    if protocol.state().eof: return True
    while True:
        reason=None
        try: data=os.read(master,65536)
        except BlockingIOError: return False # EAGAIN is not transport EOF.
        except OSError as error:
            if error.errno!=errno.EIO:
                record['transport_read_error']={'errno':error.errno,'error':repr(error),'raw_byte_offset':raw.tell()}
                raise
            data=b'';reason='pty_eio'
        if not data:
            reason=reason or 'zero_read'
            record['transport_eof']={'reason':reason,'raw_byte_offset':raw.tell(),
                'received_bytes':protocol.state().received_bytes,
                'inferior_exit_code_at_observation':g.exit_code if g else None,
                'inferior_exit_signal_at_observation':g.exit_signal if g else None,
                'inferior_poll_state_at_observation':{
                    'exit_known':bool(g and (g.exit_code is not None or g.exit_signal is not None)),
                    'last_stop':g.stops[-1] if g and g.stops else None},
                'quit_sent':record.get('quit_sent',False)}
            # Preserve transport reason/order even if finish rejects a partial tail.
            protocol.finish()
            if not record.get('quit_sent') or not g:
                raise RuntimeError('early transport EOF before QUIT')
            # PTY and GDB/MI are independent streams. Retain unknown exit at
            # this observation; require actual MI normal exit before final pass.
            return True
        receive_stdout(data)


def boundary(label, allow_eof=False):
    eof=available_stdout()
    state=protocol.state()
    entry={'label':label,'received_bytes':state.received_bytes,'consumed_bytes':state.consumed_bytes,
           'queued_lines':state.queued_lines,'queued_bytes':state.queued_bytes,'pending_bytes':state.pending_bytes,
           'eof':state.eof,'error':state.error}
    entries=record.setdefault('protocol_boundaries',[])
    if not entries or entries[-1]!=entry: entries.append(entry)
    if state.has_unconsumed:
        entry['queued_hex']=[line.hex() for line in protocol.queued_tail()]
        entry['pending_hex']=protocol.pending_tail().hex()
        raise ValueError('unexpected stdout tail at '+label)
    if eof and not allow_eof: raise RuntimeError('unexpected engine EOF at '+label)


def encoded_failure(compact, other_bytes, limit):
    # All added fields are part of the bytes checked and written. Iterate the
    # self-describing byte count to stability; never write an unchecked re-encode.
    for attempt in range(16):
        payload=(json.dumps(compact,indent=2)+'\n').encode('utf8')
        size=len(payload);projected=other_bytes+size
        updates={'serialized_failure_receipt_bytes':size,'projected_final_text_bytes':projected,
                 'external_or_prior_text_overflow':projected>limit}
        if all(compact.get(key)==value for key,value in updates.items()): return payload
        compact.update(updates)
    raise RuntimeError('failure receipt size did not stabilize')


def failure_save(reason):
    record['healthy']=False;record['reference_established']=False;record['text_budget_gate_passed']=False
    compact={k:record.get(k) for k in ('active','completed','healthy','math_gate_passed','reference_established','pair_gate_passed',
        'histogram_gate_passed','pair_shape_gate_passed','protocol_boundary_gate_passed','text_budget_gate_passed','exit_code','exit_signal',
        'cleanup','new_fault_messages','error','boot_id','inferior','debugger','controller_sha256','protocol_boundaries',
        'unwritten_stdout_chunk_hex','protocol_failure_state','transport_eof','transport_read_error',
        'C_original_math_gate_passed','D_original_healthy','D_original_full_math_passed','prior_r5_original_math_passed','prior_r5_original_healthy')}
    compact.update(final_receipt_compacted=True,budget_failure=repr(reason),
        text_limit_bytes=record['log_limit_bytes'],text_bytes_before_failure_receipt=text_bytes(out),
        numerical_gate_not_upgraded=True)
    path=out/'record.json';old_size=path.stat().st_size if path.exists() else 0
    other_bytes=text_bytes(out)-old_size
    payload=encoded_failure(compact,other_bytes,record['log_limit_bytes'])
    if len(payload)>FAILURE_RESERVE:
        # Bounded fallback preserves actual math/exit/cleanup/C/D values rather
        # than truncating them or inventing a pass. Large details remain in prior
        # raw artifacts; identify omitted metadata by complete payload SHA/bytes.
        compact={k:record.get(k) for k in ('active','completed','math_gate_passed','reference_established','pair_gate_passed',
            'histogram_gate_passed','pair_shape_gate_passed','protocol_boundary_gate_passed','exit_code','exit_signal',
            'C_original_math_gate_passed','D_original_healthy','D_original_full_math_passed','prior_r5_original_math_passed','prior_r5_original_healthy')}
        compact.update(healthy=False,text_budget_gate_passed=False,final_receipt_compacted=True,
            failure_metadata_reserve_exceeded=True,transport_small_evidence=transport_small_evidence(),
            omitted_failure_metadata_bytes=len(payload),
            omitted_failure_metadata_sha256=hashlib.sha256(payload).hexdigest(),
            cleanup={key:bool(record.get('cleanup',{}).get(key,False)) for key in ('forced','inferior_survived','gdb_survived')},
            cleanup_observed=bool(record.get('cleanup')),
            cleanup_failure=any(record.get('cleanup',{}).values()),
            fault_message_count=len(record.get('new_fault_messages',[])),
            error_excerpt=str(record.get('error'))[:1024],budget_failure_excerpt=repr(reason)[:1024],
            numerical_gate_not_upgraded=True,text_limit_bytes=record['log_limit_bytes'])
        payload=encoded_failure(compact,other_bytes,record['log_limit_bytes'])
    if len(payload)>FAILURE_RESERVE: raise RuntimeError('bounded failure fallback exceeds reserve')
    with path.open('wb') as stream: stream.write(payload)


def poll():
    global next_update
    g.poll(.01)
    if time.monotonic() - started >= record['deadline_seconds']:
        raise TimeoutError('bounded model job deadline')
    if g.stops and g.stops[-1] != 'resumed' and g.exit_code is None and g.exit_signal is None:
        raise RuntimeError('inferior stopped: ' + g.stops[-1])
    failure_session=out/'numerical-failure.session.bin'
    if failure_session.exists() and failure_session.stat().st_size>1024**3:
        raise RuntimeError('failure SAVE byte budget exceeded')
    if binary_artifact_bytes(out) > 4*1024**3:
        raise RuntimeError('aggregate binary artifact budget exceeded')
    if text_bytes(out) >= record['log_limit_bytes']-FAILURE_RESERVE:
        raise RuntimeError('total diagnostic text budget exceeded')
    if time.monotonic() >= next_update:
        save()
        next_update = time.monotonic() + 5


def line(seconds=None):
    end=time.monotonic()+(seconds or record['protocol_timeout_seconds'])
    with selectors.DefaultSelector() as ready:
        ready.register(master,selectors.EVENT_READ)
        while time.monotonic()<end:
            value=protocol.pop_line()
            if value is not None:
                return value[:-1].decode('ascii') # LF only; preserve/reject CR, never strip content.
            if protocol.state().eof: raise RuntimeError('protocol EOF before response')
            poll()
            if not ready.select(.02): continue
            available_stdout()
    raise TimeoutError('engine protocol deadline')


def send(data):
    end = time.monotonic() + 20
    view = memoryview(data)
    boundary('immediately-before-command-write')
    with selectors.DefaultSelector() as ready:
        ready.register(master, selectors.EVENT_WRITE)
        while view:
            poll()
            if time.monotonic() >= end:
                raise TimeoutError('protocol input deadline')
            if not ready.select(.02):
                continue
            try:
                count = os.write(master, view[:8192])
            except BlockingIOError:
                continue
            view = view[count:]


def captures(name,n):
    head=out/'first-head.bin';state=out/'prefill-state.bin'
    require(head.is_file() and state.is_file(),'fresh captures required')
    require(head.stat().st_size==248320*4,'firsthead exact size')
    kept=out/(name+'.head.bin');require(not kept.exists(),'capture collision');head.rename(kept)
    blob=kept.read_bytes();rules.finite_head(blob)
    result={'first_head':{'file':str(kept),'sha256':hashlib.sha256(blob).hexdigest(),'floats':248320,'finite':True}}
    kept=out/(name+'.state.bin');require(not kept.exists(),'capture collision');state.rename(kept)
    expected=rules.state_sizes(n-1)
    require(kept.stat().st_size==sum(expected)+66*8,'exact66 state file bytes')
    parts=[]
    with kept.open('rb') as stream:
        for index,size in enumerate(expected):
            header=stream.read(8);require(len(header)==8 and struct.unpack('=Q',header)[0]==size,'state header/geometry')
            offset=stream.tell();left=size;hasher=hashlib.sha256()
            while left:
                chunk=stream.read(min(left,1048576));require(chunk,'state short read');hasher.update(chunk);left-=len(chunk)
            parts.append(dict(index=index,offset=offset,bytes=size,sha256=hasher.hexdigest()))
        require(not stream.read(1),'state trailing bytes')
    result['prefill_state']={'file':str(kept),'bytes':kept.stat().st_size,'parts':parts,'sha256':digest(kept)}
    require(binary_artifact_bytes(out)<=3*1024**3,'four capture budget')
    return result


def identity_closed(data,allow_failed=False):
    require(data.get('active') is False and data.get('exit_code')==0 and data.get('exit_signal') is None and not data.get('new_fault_messages'),'closed fault cursor exit')
    require(data.get('boot_id')==boot,'same boot required')
    for key in ('forced','inferior_survived','gdb_survived'):
        require(data.get('cleanup',{}).get(key) is False,'explicit clean ownership closure')
    if not allow_failed: require(data.get('healthy') is True and data.get('completed') is True and data.get('math_gate_passed') is True,'qualified terminal math')
    for role in ('inferior','debugger'):
        old=data[role];now=process_identity(old['pid'])
        require(not now or now['start_ticks']!=old['start_ticks'],'prior owned process remains')


def source_closure(build):
    compile_path=Path(build['build'])/'compile_commands.json'
    verified(compile_path,admission.COMPILE_SHA)
    verified(Path(build['build'])/'build.ninja',admission.NINJA_SHA)
    target_commands=verified(BASE/'baseline-869-source-equivalence-provenance-v1/target-commands.stdout','fadfe6129713809443e877df18a9d7ccc8b90c310d15c9ff34b0bb2888996779',262144).decode('utf8')
    target=admission.target_commands_paths(target_commands,str(ROOT),str(GGML_ROOT))
    require(target['compile_commands']==115 and target['unique_sources']==114,'frozen115 compile rules/114 target sources')
    compiled=target['sources']
    deps_sha='e1585f1b8c79cc5e1c900b5edb6e96183032e290d38316c5825c268b4088021f'
    deps_text=verified(BASE/'baseline-869-source-equivalence-provenance-v1/target-deps.stdout',deps_sha,4*1024**2).decode('utf8')
    dependencies=admission.recorded_dependency_paths(deps_text,str(ROOT),str(GGML_ROOT),build['build'])
    deps=dependencies['sources']
    require(len(deps['project'])==241 and len(deps['ggml'])==52 and deps['generated']==['ggml/src/ggml-version.h'],'frozen dependency ownership geometry')
    for key in ('project','ggml'): require(set(compiled[key])<=set(deps[key]),'reachable translation unit missing dependency coverage')
    tracked=git('ls-tree','-r','--name-only',COMMIT).splitlines()
    paths=admission.conservative_inputs(tracked,sorted(set(compiled['project'])|set(deps['project'])))
    expected={p:hashlib.sha256(subprocess.check_output(['git','show',COMMIT+':'+p],cwd=ROOT)).hexdigest() for p in paths}
    actual={p:digest(ROOT/p) for p in paths}
    changed=git('diff','--name-only',COMMIT,CURRENT_HEAD).splitlines()
    summary=admission.validate_closure(COMMIT,git('rev-parse','HEAD'),git('status','--porcelain'),expected,actual,changed)
    ggml_commit=build['ggml_commit']
    ggml_git=lambda *args:subprocess.check_output(['git',*args],cwd=GGML_ROOT,text=True).strip()
    require(ggml_git('rev-parse','HEAD')==ggml_commit and not ggml_git('status','--porcelain'),'clean pinned ggml build HEAD')
    ggml_paths=admission.conservative_inputs(ggml_git('ls-tree','-r','--name-only',ggml_commit).splitlines(),sorted(set(compiled['ggml'])|set(deps['ggml'])))
    external={p:hashlib.sha256(subprocess.check_output(['git','show',ggml_commit+':'+p],cwd=GGML_ROOT)).hexdigest() for p in ggml_paths}
    for p,pin in external.items(): require(digest(GGML_ROOT/p)==pin,'changed ggml compiled-input source')
    generated_path=Path(build['build'])/'ggml/src/ggml-version.h'
    generated_sha='9e45eee8391971371922f51b66925184609125d467c444744599f680b78edc92'
    verified(generated_path,generated_sha,65536)
    summary.update(reachable_target_commands_sha256='fadfe6129713809443e877df18a9d7ccc8b90c310d15c9ff34b0bb2888996779',reachable_compile_commands=target['compile_commands'],reachable_unique_sources=target['unique_sources'],compile_commands_sha256=admission.COMPILE_SHA,build_ninja_sha256=admission.NINJA_SHA,ggml_build_commit=ggml_commit,ggml_closure_count=len(external),ggml_closure_sha256=hashlib.sha256((json.dumps(external,sort_keys=True,separators=(',',':'))+'\n').encode('ascii')).hexdigest())
    summary.update(recorded_object_dependency_superset_sha256=deps_sha,recorded_dependency_object_blocks=dependencies['object_blocks'],recorded_project_dependency_paths=len(deps['project']),recorded_ggml_dependency_paths=len(deps['ggml']),external_dependency_unique_paths_outside_equivalence_claim=dependencies['external_unique_paths'],generated_input_provenance='Current generated header pinned before run and checked at exit; no historical generated-header/compiler/system-header hash claim')
    return {'project':expected,'ggml':external,'generated':{str(generated_path):generated_sha},'provenance':summary}


def admit(options):
    global admission,rules,base,boot,binary,root,process_identity,OwnedGdb,BoundedProtocolLines,compare_states
    require(sys.flags.optimize==0,'no python -O')
    require(not any(k.startswith('STRATA_') for k in os.environ),'inherited Strata settings forbidden')
    base=BASE;root=ROOT;boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    verified(__file__,options.self_sha256,262144)
    require(options.admission_sha256==ADMISSION_SHA,'frozen admission SHA')
    admission=pinned_module('long_reference_admission_v3',Path(__file__).with_name('long_prompt_reference_admission_v3.py'),options.admission_sha256)
    require(options.rules_sha256==RULES_SHA,'frozen numerical rules SHA')
    cpu_guard=json.loads(verified(BASE/'owned-main-thread-cpu/record.json',CPU_GUARD_SHA))
    require(cpu_guard['passed'] is True and len(cpu_guard['cases'])==4,'owned CPU guard')
    for case in cpu_guard['cases']:
        if case['name'].startswith('current-'): require(case['helper_sha256']==OWNED_SHA,'owned guard helper source')
    rules=pinned_module('long_reference_rules',Path(__file__).with_name('long_prompt_reference_rules_v1.py'),options.rules_sha256)
    require(options.protocol_helper_sha256==FRAMER_SHA and options.owned_helper_sha256==OWNED_SHA and options.comparator_sha256==COMPARE_SHA and options.health_helper_sha256==HEALTH_SHA,'frozen helper SHA arguments')
    verified(BASE/'run_owned_cache_route_pairs_v0141_code32k_v6.py',TRANSPORT_SOURCE_SHA,262144)
    framer=pinned_module('reference_protocol_framer',BASE/'bounded_engine_protocol_v1.py',FRAMER_SHA)
    owned=pinned_module('reference_owned_gdb',OBSERVER/'sycl/tools/owned_gdb.py',OWNED_SHA)
    comparator=pinned_module('reference_live_state_comparator',BASE/'compare_live_prefill_state_v01402_v2.py',COMPARE_SHA)
    verified(OBSERVER/'sycl/tools/recover-xe.sh',HEALTH_SHA,262144)
    OwnedGdb=owned.OwnedGdb;process_identity=owned.process_identity;BoundedProtocolLines=framer.BoundedProtocolLines;compare_states=comparator.compare_states
    full=json.loads(verified(BASE/'owned-upstream-v0141-integrated-full256k-diagnostic-r2/record.json',FULL_SHA))
    require(full.get('active') is False and full.get('healthy') is True and full.get('completed') is True and full.get('math_gate_passed') is True and full.get('physical256k_sequence_completed') is True and full.get('capacity_sequence_completed') is True,'independent full256K qualified baseline')
    require(full['binary_sha256']==BINARY_SHA and full['build_receipt_sha256']==BUILD_SHA,'qualified binary/build')
    build=json.loads(verified(BASE/'integrated-upstream-v0.1.41-20261009-v2/build-record.json',BUILD_SHA))
    require(build['active'] is False and build['passed'] is True and build['commit']==COMMIT and build['source_status_before']==build['source_status_after']=='' and build['ggml_status_before']==build['ggml_status_after']=='','baseline build provenance')
    require(build['root']==str(ROOT) and build['binary_sha256']==BINARY_SHA,'baseline build path')
    binary=Path(build['binary']);require(str(binary)==full['argv'][0] and digest(binary)==BINARY_SHA,'actual baseline binary')
    require(git('rev-parse','HEAD')==CURRENT_HEAD and not git('status','--porcelain'),'baseline reviewed current clean HEAD')
    source_pins=source_closure(build)
    flags_pins={}
    for name in ('CMakeCache.txt','build.ninja','compile_commands.json'):
        path=Path(build['build'])/name
        require(path.is_file() and path.stat().st_size<=16*1024**2,'bounded existing build flag artifact')
        flags_pins[str(path)]=digest(path)
    source_pins['existing_build_flags']=flags_pins
    require('--pool-tasks' not in full['argv'],'869 original argv has no pool-tasks')
    require('--pool-tasks' not in (ROOT/'sycl/src/program/generate.cpp').read_text(),'baseline source no pool-tasks support')
    pack=Path(full['argv'][full['argv'].index('--pack')+1])
    for name,sha in PACK_PINS.items(): verified(pack/name,sha,4*1024**2)
    profile=ROOT/'data/expert-profile.bin';verified(profile,PROFILE_SHA,196632)
    conversions=json.loads(verified(pack/'conversions.json',PACK_PINS['conversions.json']))
    native=Path(full['argv'][full['argv'].index('--native')+1]);model_identity=[]
    for shard in conversions['source_shards']:
        path=native.parent/shard['name'];info=path.stat()
        require(path.is_file() and info.st_size==shard['size'],'model shard metadata')
        model_identity.append(dict(path=str(path),bytes=info.st_size,dev=info.st_dev,ino=info.st_ino,mtime_ns=info.st_mtime_ns,ctime_ns=info.st_ctime_ns))
    owner=json.loads(verified(BASE/'independent-prompt-actual-pack-cpu-validation-v3/record.json',INPUT_OWNER_SHA))
    require(owner.get('active') is False and owner.get('complete') is True and owner.get('passed') is True and owner.get('owner_admission_passed') is True and owner.get('exit_code')==0 and owner.get('cleanup')==[] and owner.get('survivors')==[] and owner.get('actual_pack_tokenized') is True and owner.get('inference_run') is False,'owner-validated actual inputs')
    bundle=BASE/'independent-prompt-actual-pack-cpu-validation-v3/bundle'
    candidate=json.loads(verified(bundle/'record.json',CANDIDATE_SHA))
    require(candidate['completed'] is False and candidate['ready_for_owner_validation'] is True and candidate['manifest_sha256']==MANIFEST_SHA,'input candidate owner distinction')
    fixtures={};input_pins={}
    for name,entry in owner['output_files'].items():
        require(name in ('record.json','manifest.json','train.tokens.txt','validation.tokens.txt','train.rendered.txt','validation.rendered.txt'),'known bundle file')
        data=verified(bundle/name,entry['sha256'],1024**2);require(len(data)==entry['bytes'],'bundle exact bytes');input_pins[name]=entry['sha256']
        require((bundle/name).stat().st_mode & 0o777==0o600,'private input file mode')
    require(input_pins['manifest.json']==MANIFEST_SHA and input_pins['record.json']==CANDIDATE_SHA,'manifest/candidate pins')
    require(bundle.stat().st_mode & 0o777==0o700,'private input bundle')
    for key,(n,sha) in rules.INPUTS.items():
        data=verified(bundle/(key+'.tokens.txt'),sha,1024**2)
        tokens=list(map(int,data.decode('ascii').split()))
        require(all(0<=i<248320 for i in tokens) and data==(' '.join(map(str,tokens))+'\n').encode('ascii'),'canonical complete IDs')
        rules.input_identity(key,len(tokens),sha);fixtures[key]=tokens
        require(owner['prompt_summaries'][key]['token_count']==n and owner['prompt_summaries'][key]['token_ids_sha256']==sha,'owner token summary')
    fault=json.loads(verified(options.fault_cursor_receipt,options.fault_cursor_sha256))
    require(fault.get('active') is False,'closed supplied fault cursor')
    cursor_steps=[x for x in fault['steps'] if x['label']=='kernel-after']
    require(len(cursor_steps)==1 and cursor_steps[0]['exit_code']==0 and not cursor_steps[0]['timed_out'] and not cursor_steps[0]['still_alive'],'terminal cursor command')
    return full,build,source_pins,model_identity,fixtures,input_pins,fault


def main():
    global record,g,master,slave,cursor,protocol,current,raw,events,started,next_update,r,out
    cli=argparse.ArgumentParser(description=__doc__)
    for arg in ('self-sha256','rules-sha256','protocol-helper-sha256','owned-helper-sha256','comparator-sha256','health-helper-sha256','fault-cursor-sha256','admission-sha256'): cli.add_argument('--'+arg,required=True)
    cli.add_argument('--fault-cursor-receipt',type=Path,required=True)
    cli.add_argument('--cpu-preflight',action='store_true')
    options=cli.parse_args()
    full,build,source_pins,model_identity,fixtures,input_pins,fault=admit(options)
    out=BASE/'owned-long-prompt-baseline-v0141-reference-r3'
    require(not out.exists(),'new output path must not exist')
    if options.cpu_preflight:
        print(json.dumps(dict(cpu_preflight_passed=True,GPU_executed=False,runtime_qualification_claimed=False,binary_sha256=BINARY_SHA,baseline_full_receipt_sha256=FULL_SHA,input_owner_sha256=INPUT_OWNER_SHA,inputs=input_pins,source_pins=source_pins,argv=full['argv'],cwd=str(ROOT),pool_tasks='qualified869 default; no unsupported pool-tasks flag',output_not_created=True)))
        return
    identity_closed(full);identity_closed(fault,allow_failed=True)
    cpu_guard=json.loads(verified(BASE/'owned-main-thread-cpu/record.json',CPU_GUARD_SHA))
    for case in cpu_guard['cases']:
        for role in ('inferior','debugger'):
            old=case[role];now=process_identity(old['pid'])
            require(not now or now['start_ticks']!=old['start_ticks'] or now['state']=='Z','CPU guard process remains')
    lock=(BASE/'owned-v0141-measurement.lock').open('a')
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BaseException:
        lock.close();raise
    record=dict(active=True,completed=False,healthy=False,math_gate_passed=False,reference_established=False,protocol_boundary_gate_passed=False,text_budget_gate_passed=False,performance_eligible=False,adopted=False,full_lifecycle_passed=False,C_original_math_gate_passed=False,D_original_healthy=False,D_original_full_math_passed=False,prior_r5_original_math_passed=False,prior_r5_original_healthy=False,scope='Independent qualified869 long-prompt numerical reference; original default CPU task shape; logged timings never matched-performance',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),boot_id=boot,deadline_seconds=1800,protocol_timeout_seconds=450,log_limit_bytes=64*1024**2,binary_artifact_limit_bytes=4*1024**3,capture_limit_bytes=3*1024**3,failure_SAVE_limit_bytes=1024**3,requests=[],snapshots=[],steps=[],controller_sha256=options.self_sha256,admission_sha256=options.admission_sha256,build_commit=COMMIT,current_clean_HEAD=CURRENT_HEAD,rules_sha256=options.rules_sha256,cpu_debugger_guard_sha256=CPU_GUARD_SHA,helper_pins=dict(protocol=FRAMER_SHA,owned=OWNED_SHA,health=HEALTH_SHA,comparator=COMPARE_SHA,transport_source=TRANSPORT_SOURCE_SHA),argv=full['argv'],cwd=str(ROOT),commit=COMMIT,source_pins=source_pins,build_receipt_sha256=BUILD_SHA,binary_sha256=BINARY_SHA,baseline_full_receipt_sha256=FULL_SHA,input_owner_sha256=INPUT_OWNER_SHA,manifest_sha256=MANIFEST_SHA,input_pins=input_pins,model_identity=model_identity,weight_payload_hashed=False,profile_sha256=PROFILE_SHA,pack_metadata_sha256=PACK_PINS,fault_cursor_receipt=str(options.fault_cursor_receipt),fault_cursor_sha256=options.fault_cursor_sha256,numerical_reference_independence='Qualified869 before new candidate; no old32K head/state oracle reused',input_independence_limit='Distinct files/components in same frozen codebase; not population-generalization evidence')
    g=None;master=slave=None;cursor=None;current=None;protocol=BoundedProtocolLines(cr_policy='reject')
    raw=events=None;r=None
    started=time.monotonic();next_update=started
    try:
        require(shutil.disk_usage(BASE).free>8*1024**3,'finite capture/save free-space reserve8GiB')
        out.mkdir(mode=0o700);(out/'probes').mkdir(mode=0o700)
        source=OBSERVER/'sycl/tools/recover-xe.sh';source_bytes=verified(source,HEALTH_SHA,262144)
        health=types.ModuleType('long_reference_health')
        exec(compile(source_bytes.decode().split("<<'PY'\n",1)[1].rsplit('\nPY',1)[0],str(source),'exec'),health.__dict__)
        r=health.Runner(out/'probes');os.chdir(ROOT)
        raw=(out/'protocol.stdout.raw').open('xb');events=(out/'events.jsonl').open('x',buffering=1)
        save()
        old_cursor=next(x['argv'][x['argv'].index('--after-cursor')+1] for x in fault['steps'] if x['label']=='kernel-after')
        gap=r.run('kernel-gap',['/usr/bin/journalctl','-k','--after-cursor',old_cursor,'--no-pager','-o','json'],seconds=5)
        rows=[json.loads(s) for s in gap.splitlines() if s.startswith('{')]
        record['preflight_fault_messages']=[x['MESSAGE'] for x in rows if ('0000:05:00.0' in x.get('MESSAGE','') or re.search(r'\bxe\b',x.get('MESSAGE',''))) and health.FAULT.search(x.get('MESSAGE',''))]
        require(not record['preflight_fault_messages'],'kernel fault gap')
        require(r.run('embedding-state',['/usr/bin/systemctl','--user','show','llama-server-qwen3embed.service','-p','ActiveState'],seconds=5).strip()=='ActiveState=inactive','embedding service inactive required')
        os.environ.update(NEOReadDebugKeys='1',EnableDirectSubmission='0')
        health.check_health(types.SimpleNamespace(bdf='0000:05:00.0'),r,Path('/home/yayoi/.local/bin/strata-xe-health'))
        env=health.health_environment();env['LD_LIBRARY_PATH']+=':/opt/intel/oneapi/mkl/2026.1/lib'
        for key in list(env):
            if key.startswith(('STRATA_','UNITRACE_','XPTI_','UR_LOG_','ZE_ENABLE_','ZEL_')) or key in ('LD_PRELOAD','ZET_ENABLE_METRICS','UR_ENABLE_LAYERS','UR_LOG_TRACING'): env.pop(key)
        for key,value in full['environment'].items():
            if key.startswith('STRATA_') and key not in ('STRATA_DUMP_FIRST_LOGITS','STRATA_PREFILL_DUMP_STATE'): env[key]=value
        env.update(STRATA_DUMP_FIRST_LOGITS=str(out/'first-head.bin'),STRATA_PREFILL_DUMP_STATE=str(out/'prefill-state.bin'),ZEL_ENABLE_LOADER_LOGGING='1',ZEL_LOADER_LOG_CONSOLE='1',ZEL_LOADER_LOGGING_LEVEL='warn',ZEL_LOADER_LOGGING_ENABLE_SUCCESS_PRINT='0',ZE_ENABLE_VALIDATION_LAYER='1',ZE_ENABLE_PARAMETER_VALIDATION='1',UR_LOG_LOADER='level:warning;flush:warning;output:stderr',UR_LOG_LEVEL_ZERO='level:warning;flush:warning;output:stderr',STRATA_TRACE='1',NEOReadDebugKeys='1',EnableDirectSubmission='0')
        require(not any(k in env for k in ('STRATA_CACHE_ROUTE_PAIRS','STRATA_NATIVE_DISPATCH_HISTOGRAM','STRATA_DECODE_TIMING')),'unsupported baseline counters disabled')
        relevant=lambda values:{k:v for k,v in values.items() if k.startswith(('STRATA_','SYCL_','UR_','ZE_','ZEL_','ONEAPI_')) or k in ('LD_LIBRARY_PATH','NEOReadDebugKeys','EnableDirectSubmission')}
        record['environment']=relevant(env)
        functional=lambda values:{k:v for k,v in relevant(values).items() if not k.startswith(('UR_LOG_','ZE_ENABLE_','ZEL_')) and k not in ('UR_ENABLE_LAYERS','STRATA_DUMP_FIRST_LOGITS','STRATA_PREFILL_DUMP_STATE')}
        require(functional(env)==functional(full['environment']),'qualified functional environment unchanged')
        record['diagnostic_logging']='flushed warning UR/ZE/ZEL validation + Strata progress; no full API tracing; never performance samples'
        cursor=health.journal_cursor(r,'kernel-before');save()
        master,slave=os.openpty();tty.setraw(slave);os.set_blocking(master,False)
        g=OwnedGdb(full['argv'],out/'debugger',env,inferior_tty_fd=slave)
        g.command('-gdb-set may-call-functions off');g.command('-gdb-set debug-file-directory /usr/lib/debug:/home/yayoi/.local/state/strata-sycl/orderly-serve-shutdown-20261006/runtime-symbols/extracted/usr/lib/debug');g.run()
        record['startup']=[];startup_seen=[]
        while True:
            value=line();record['startup'].append(value);startup_step(value,startup_seen)
            if value.startswith('READY '): break
        os.close(slave);slave=None
        ident=process_identity(g.inferior['pid']);require(ident and ident['start_ticks']==g.inferior['start_ticks'],'owned inferior identity')
        proc=Path('/proc',str(ident['pid']))
        require((proc/'exe').resolve()==binary.resolve() and digest(proc/'exe')==BINARY_SHA,'actual executable')
        actual=dict(v.decode().split('=',1) for v in (proc/'environ').read_bytes().split(b'\0') if b'=' in v)
        require(relevant(actual)==record['environment'],'actual environment')
        require(not any(k.startswith(('UNITRACE_','XPTI_')) or k in ('LD_PRELOAD','ZET_ENABLE_METRICS') for k in actual),'no profiler preload')
        actual_argv=[v.decode() for v in (proc/'cmdline').read_bytes().split(b'\0') if v]
        require(actual_argv==full['argv'] and (proc/'cwd').resolve()==ROOT.resolve(),'actual original argv/cwd')
        record.update(actual_target_argv=actual_argv,actual_target_environment=relevant(actual),actual_target_cwd=str(ROOT),actual_executable_identity=ident)
        seen={};math_ok=True
        for read_index,key in enumerate(('train','validation','train','validation')):
            n=len(fixtures[key]);current=key+'-read'+str(read_index)
            result=dict(name=current,fixture=key,input_tokens=n,read_index=read_index,ids=[],logprobs=[],protocol=[])
            record['requests'].append(result);save()
            require(not (out/'first-head.bin').exists() and not (out/'prefill-state.bin').exists(),'no stale capture')
            boundary('before-GEN-'+str(read_index));request_start=time.monotonic()
            send(('GEN 64 ckpt=1 logprobs=5 '+','.join(map(str,fixtures[key]))+'\n').encode('ascii'))
            event('request',name=current)
            while True:
                value=line();result['protocol'].append(value);kind=protocol_line_kind(value,'gen')
                if kind=='error': raise RuntimeError(value)
                if value.startswith('T '): result['ids'].append(int(value[2:]))
                if value.startswith('LP '): result['logprobs'].append(value)
                if value.startswith('DONE '): break
            boundary('after-DONE-'+str(read_index))
            admitted=rules.validate_request(result,n)
            result.update(finish_reason=admitted['finish_reason'],mtp_counts=admitted['mtp_counts'],validation=admitted)
            result['measurement']=dict(prompt_ms=admitted['prompt_ms'],decode_ms=admitted['decode_ms'],wall_seconds=time.monotonic()-request_start,purpose='logged correctness only; original869 task shape; NEVER matched-performance')
            result.update(captures(current,n))
            if key in seen:
                state_cmp=compare_states(result['prefill_state'],seen[key]['prefill_state'],n-1)
                result['live_prefill_comparison']=state_cmp
                result['repeat_comparison']=rules.repeat_checks(result,seen[key],state_cmp)
                result['math_gate_passed']=all(result['repeat_comparison'].values())
            else:
                result['math_gate_passed']=True;result['reference_role']='first capture awaiting independent same-prompt repeat; not established alone';seen[key]=result
            math_ok &= result['math_gate_passed'];save()
            if not result['math_gate_passed']:
                failed=out/'numerical-failure.session.bin';require(not failed.exists(),'failureSAVE collision')
                boundary('before-SAVE');send(('SAVE '+str(failed)+'\n').encode('ascii'));responses=[]
                while True:
                    response=line();responses.append(response);kind=protocol_line_kind(response,'save')
                    if failed.exists(): require(failed.stat().st_size<=1024**3,'failureSAVE budget')
                    if kind in ('terminal','error'): break
                boundary('after-SAVE-terminal')
                record['numerical_failure_preservation']=dict(request=current,file=str(failed),protocol=responses,exists=failed.exists())
                if failed.exists(): record['numerical_failure_preservation'].update(bytes=failed.stat().st_size,sha256=digest(failed))
                break
        current=None;boundary('before-QUIT');send(b'QUIT\n');event('quit');record['quit_sent']=True
        end=time.monotonic()+30
        with selectors.DefaultSelector() as ready:
            ready.register(master,selectors.EVENT_READ)
            while time.monotonic()<end:
                poll()
                if ready.select(.01): boundary('QUIT-drain',allow_eof=True)
                if (g.exit_code is not None or g.exit_signal is not None) and protocol.state().eof: break
        boundary('after-QUIT-exit',allow_eof=True)
        require(protocol.state().eof and record.get('transport_eof') and g.exit_code==0 and g.exit_signal is None,'observed transportEOF AND normal inferior exit')
        record['protocol_boundary_gate_passed']=True
        require(source_closure(build)=={k:v for k,v in source_pins.items() if k!='existing_build_flags'},'unchanged reviewed build closure and current HEAD')
        for name,sha in PACK_PINS.items(): require(digest(Path(full['argv'][full['argv'].index('--pack')+1])/name)==sha,'pack metadata changed')
        for path,sha in source_pins['existing_build_flags'].items(): require(digest(path)==sha,'build flag artifact changed')
        require(digest(ROOT/'data/expert-profile.bin')==PROFILE_SHA and digest(binary)==BINARY_SHA,'profile/binary changed')
        for item in model_identity:
            info=Path(item['path']).stat()
            require((info.st_size,info.st_dev,info.st_ino,info.st_mtime_ns,info.st_ctime_ns)==(item['bytes'],item['dev'],item['ino'],item['mtime_ns'],item['ctime_ns']),'model shard identity changed')
        record['completed']=True
        record['math_gate_passed']=math_ok and len(record['requests'])==4 and all(all(record['requests'][i]['repeat_comparison'].values()) for i in (2,3))
        record['reference_established']=record['math_gate_passed']
        record['references']={key:dict(request_name=seen[key]['name'],repeat_request_name=record['requests'][i+2]['name'],input_sha256=rules.INPUTS[key][1],first_head=seen[key]['first_head'],prefill_state=seen[key]['prefill_state'],ids=seen[key]['ids'],logprobs=seen[key]['logprobs'],mtp_counts=seen[key]['mtp_counts'],finish_reason=seen[key]['finish_reason']) for i,key in enumerate(('train','validation'))} if record['reference_established'] else {}
        engine_log=out/'debugger/inferior.stderr';record['engine_log_bytes']=engine_log.stat().st_size;record['engine_log_sha256']=digest(engine_log)
    except BaseException as error:
        record['error']=repr(error);record['reference_established']=False
        if g is None: record.update(admission.failed_setup(error))
        state=protocol.state();record['protocol_failure_state']=dict(eof=state.eof,error=state.error,queued_hex=[v.hex() for v in protocol.queued_tail()],pending_hex=protocol.pending_tail().hex(),received_bytes=state.received_bytes,consumed_bytes=state.consumed_bytes)
        if g:
            try: record['snapshots'].append(g.snapshot('failure',resume=False))
            except BaseException as inspect: record['snapshot_error']=repr(inspect)
    finally:
        record['exit_code']=g.exit_code if g else None;record['exit_signal']=g.exit_signal if g else None
        try:
            record['cleanup']=g.close() if g else {'forced':False,'inferior_survived':False,'gdb_survived':False}
        except BaseException as error:
            record['cleanup']={'forced':None,'inferior_survived':None,'gdb_survived':None}
            record['cleanup_exception']=repr(error);record['error']=record.get('error') or repr(error)
        for fd in (master,slave):
            if fd is not None:
                try: os.close(fd)
                except BaseException as error: record['error']=record.get('error') or repr(error)
        for error in admission.close_setup_resources((raw,events)):
            record['error']=record.get('error') or error
        if cursor:
            try:
                text=r.run('kernel-after',['/usr/bin/journalctl','-k','--after-cursor',cursor,'--no-pager','-o','json'],seconds=5)
                rows=[json.loads(s) for s in text.splitlines() if s.startswith('{')]
                record['new_fault_messages']=[x['MESSAGE'] for x in rows if (('0000:05:00.0' in x.get('MESSAGE','') or re.search(r'\bxe\b',x.get('MESSAGE',''))) and health.FAULT.search(x.get('MESSAGE',''))) or ('strata' in x.get('MESSAGE','') and 'segfault' in x.get('MESSAGE',''))]
            except BaseException as error: record['error']=record.get('error') or repr(error)
        try:
            require(source_closure(build)=={k:v for k,v in source_pins.items() if k!='existing_build_flags'},'final unchanged compiled-input closure/current HEAD')
            for path,pin in source_pins['existing_build_flags'].items(): require(digest(path)==pin,'final build flag artifact changed')
            record['source_closure_exit_gate_passed']=True
        except BaseException as source_error:
            record['source_closure_exit_gate_passed']=False;record['error']=record.get('error') or repr(source_error)
        record['healthy']=record['source_closure_exit_gate_passed'] and record['completed'] and record['math_gate_passed'] and record['protocol_boundary_gate_passed'] and record['exit_code']==0 and record['exit_signal'] is None and not record.get('error') and not record.get('new_fault_messages') and all(record['cleanup'].get(k) is False for k in ('forced','inferior_survived','gdb_survived'))
        record['reference_established']=record['reference_established'] and record['healthy']
        if not record['reference_established']: record['references']={}
        record['active']=False;record['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        try:
            record['text_budget_gate_passed']=True
            if out.exists(): save()
            else: raise RuntimeError('output directory not created; no file receipt possible')
            require(text_bytes(out)<=record['log_limit_bytes'] and binary_artifact_bytes(out)<=4*1024**3,'final aggregate budgets')
        except BaseException as budget_error:
            record['reference_established']=False;record['error']=record.get('error') or repr(budget_error)
            if out.exists():
                try: failure_save(budget_error)
                except BaseException as final_error: record['terminal_receipt_write_error']=repr(final_error);record['healthy']=False;record['text_budget_gate_passed']=False
        finally:
            try: fcntl.flock(lock,fcntl.LOCK_UN)
            finally: lock.close()
    print(json.dumps({k:record.get(k) for k in ('healthy','math_gate_passed','reference_established','protocol_boundary_gate_passed','text_budget_gate_passed','exit_code','exit_signal','cleanup','error')},indent=2))
    if not all(record.get(k) is True for k in ('healthy','math_gate_passed','reference_established','protocol_boundary_gate_passed','text_budget_gate_passed')): raise SystemExit(1)


if __name__=='__main__': main()
