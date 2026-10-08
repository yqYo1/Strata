"""One owned and bounded, fully logged exact-helper GPU comparison."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import re
import selectors
import sys
import time
import tty
import types

base=Path(__file__).parent
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
sys.path.insert(0,str(root/'sycl/tools'))
from owned_gdb import OwnedGdb, process_identity
out=base/'qsa-vector-load-gpu/run';out.mkdir(mode=0o700)
probes=out/'probes';probes.mkdir()
helper=root/'sycl/tools/recover-xe.sh'
module=types.ModuleType('health_readonly')
exec(compile(helper.read_text().split("<<'PY'\n",1)[1].rsplit('\nPY',1)[0],str(helper),'exec'),module.__dict__)
runner=module.Runner(probes)
record={'scope':'Owned B570 exact original/candidate load8 helper GPU comparison with independent host IEEE reference; full Level Zero/UR logs and parameter checks; no model throughput, full-capacity or hang-cause proof',
        'active':True,'passed':False,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'deadline_seconds':90,'controller_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
g=None;master=slave=None;cursor=None
started=time.monotonic();payload=bytearray()

def save():
    record['elapsed_seconds']=time.monotonic()-started
    if g:record.update(inferior=g.inferior,debugger=g.debugger_identity)
    (out/'record.json').write_text(json.dumps(record,indent=2)+'\n')

try:
    previous=json.loads((base/'full-context-phase-sync-serve/record.json').read_text())
    health=json.loads((base/'post-phase-full-failure-health/record.json').read_text())
    assert not previous['active'] and health['healthy'] and health['boot_id']==record['boot_id']
    for key in ['inferior','debugger']:
        now=process_identity(previous[key]['pid'])
        assert not now or now['start_ticks']!=previous[key]['start_ticks'] or now['state']=='Z'
    build=json.loads((base/'qsa-vector-load-gpu/build.json').read_text());assert build['passed']
    binary=base/'qsa-vector-load-gpu/fixture'
    assert hashlib.sha256(binary.read_bytes()).hexdigest()==build['binary_sha256']
    record['binary_sha256']=build['binary_sha256']
    env=module.diagnostic_environment(module.health_environment())
    env.update(NEOReadDebugKeys='1',EnableDirectSubmission='0')
    record['environment']={k:v for k,v in env.items() if k.startswith(('STRATA_','SYCL_','UR_','ZE_','ZEL_','ONEAPI_')) or k in ('LD_LIBRARY_PATH','NEOReadDebugKeys','EnableDirectSubmission')}
    cursor=module.journal_cursor(runner,'kernel-before')
    master,slave=os.openpty();tty.setraw(slave);os.set_blocking(master,False)
    g=OwnedGdb([str(binary)],out/'debugger',env,inferior_tty_fd=slave)
    g.command('-gdb-set may-call-functions off');g.run();save()
    os.close(slave);slave=None
    with selectors.DefaultSelector() as ready:
        ready.register(master,selectors.EVENT_READ)
        while g.exit_code is None and g.exit_signal is None:
            g.poll(.02)
            if time.monotonic()-started>record['deadline_seconds']:raise TimeoutError('GPU fixture deadline')
            if g.stops and g.stops[-1]!='resumed' and g.exit_code is None and g.exit_signal is None:
                raise RuntimeError('GPU fixture stopped: '+g.stops[-1])
            if ready.select(.01):
                try:data=os.read(master,65536)
                except (BlockingIOError,OSError):continue
                payload.extend(data)
        while ready.select(.01):
            try:data=os.read(master,65536)
            except (BlockingIOError,OSError):break
            if not data:break
            payload.extend(data)
    text=payload.decode();(out/'fixture.stdout').write_bytes(payload)
    assert g.exit_code==0 and g.exit_signal is None
    record['result']=json.loads(next(v for v in reversed(text.splitlines()) if v.startswith('{')))
    assert record['result']=={'patterns':65536,'comparisons':6291456,'non_nan_bit_checks':2031676,'nan_class_checks':65476,'mismatches':0}
    assert text.count('patterns completed:')==32 and 'device 0000:05:00.0:' in text
    record['passed']=True
except BaseException as error:
    record['error']=repr(error)
    if g:
        try:record['snapshot']=g.snapshot('failure',resume=False)
        except BaseException as inspect:record['snapshot_error']=repr(inspect)
    raise
finally:
    if g:record.update(exit_code=g.exit_code,exit_signal=g.exit_signal,cleanup=g.close())
    for fd in (master,slave):
        if fd is not None:os.close(fd)
    if cursor:
        text=runner.run('kernel-after',['/usr/bin/journalctl','-k','--after-cursor',cursor,'--no-pager','-o','json'],seconds=5)
        rows=[json.loads(v) for v in text.splitlines() if v.startswith('{')]
        record['new_fault_messages']=[v.get('MESSAGE','') for v in rows if ('0000:05:00.0' in v.get('MESSAGE','') or re.search(r'\bxe\b',v.get('MESSAGE',''))) and module.FAULT.search(v.get('MESSAGE',''))]
    record['passed']=record['passed'] and not record.get('new_fault_messages') and not record.get('error') and not any(record.get('cleanup',{}).values())
    record.update(active=False,steps=runner.calls,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
print(json.dumps(record,indent=2))
assert record['passed']
