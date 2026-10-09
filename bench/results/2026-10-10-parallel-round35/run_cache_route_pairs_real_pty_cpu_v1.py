"""Root-owned real OwnedGdb/PTY transport check with a Python echo child only."""
from pathlib import Path
import ast, datetime, errno, fcntl, hashlib, json, os, re, selectors, sys, time, tty

B=Path(__file__).parent
O=B/'cache-route-pairs-real-pty-cpu-validation-v1'
H=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05/sycl/tools')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
P=B/'run_owned_cache_route_pairs_v0141_code32k_v5.py'
assert sha(P)=='c495ffa7c7785bc82012b5d057ecaa68420ffb4b47e0139c31548e25ee6897dc'
assert sha(H/'owned_gdb.py')=='61e1d206bf57bb320051cdd65943d23722bc523a602d80c54b7e7a2ad0d3f0d1'
assert sha(B/'bounded_engine_protocol_v1.py')=='aa72bd7f1a9babf7c6e325b10634f1a046fe78fc2b2738e691d90624197131c1'
prior=B/'cache-route-pairs-v5-cpu-validation-v1/record.json'
assert sha(prior)=='cf46ea97d85cad2dc9f982fa4f46082794e2acad5b28e6972548eed2c052582f'
assert json.loads(prior.read_text())['passed']
from bounded_engine_protocol_v1 import BoundedProtocolLines
sys.path.insert(0,str(H))
from owned_gdb import OwnedGdb, process_identity

lock=(B/'owned-v0141-measurement.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not O.exists();O.mkdir(mode=0o700)
start=time.monotonic()
record=dict(active=True,complete=False,passed=False,gpu_work_submitted=False,model_opened=False,
    adopted=False,performance_eligible=False,full_lifecycle_passed=False,
    scope='Real GDB/MI and Linux PTY with bounded extracted v5 transport functions; Python echo child only; no engine/controller admission/inference/GPU helper/model payload',
    boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
    started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    controller_sha256=sha(P),wrapper_sha256=sha(__file__),owned_gdb_sha256=sha(H/'owned_gdb.py'),
    preceding_cpu_receipt_sha256=sha(prior),log_limit_bytes=2*1024**2,deadline_seconds=40,
    cleanup={},survivors=[])
def save():
    record['elapsed_seconds']=time.monotonic()-start
    temporary=O/'record.json.tmp';temporary.write_text(json.dumps(record,indent=2)+'\n');temporary.replace(O/'record.json')
child=O/'echo.py'
child.write_text("import sys\nprint('READY 0 stop',flush=True)\nfor row in sys.stdin:\n if row=='QUIT\\n': sys.exit(0)\n sys.exit(2)\nsys.exit(3)\n")
names={'text_bytes','reserve_write','event','receive_stdout','available_stdout','boundary'}
nodes=[n for n in ast.parse(P.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
assert {n.name for n in nodes}==names
raw=(O/'protocol.stdout.raw').open('wb');events=(O/'events.jsonl').open('w',buffering=1)
scope=dict(Path=Path,json=json,os=os,errno=errno,time=time,FAILURE_RESERVE=1024**2,
    out=O,raw=raw,events=events,record=record,started=start,
    protocol=BoundedProtocolLines(cr_policy='reject'),g=None,master=None)
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(P),'exec'),scope)
g=None;master=slave=None;save()
try:
    master,slave=os.openpty();tty.setraw(slave);os.set_blocking(master,False);scope['master']=master
    env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
    g=OwnedGdb(['/usr/bin/python3',str(child)],O/'debugger',env,inferior_tty_fd=slave);scope['g']=g
    g.command('-gdb-set may-call-functions off');g.run()
    record.update(inferior=g.inferior,debugger=g.debugger_identity)
    with selectors.DefaultSelector() as ready:
        ready.register(master,selectors.EVENT_READ)
        end=time.monotonic()+10
        while True:
            assert time.monotonic()<end,'READY deadline'
            g.poll(.01)
            if ready.select(.01):scope['available_stdout']()
            line=scope['protocol'].pop_line()
            if line is not None:break
        assert line==b'READY 0 stop\n',repr(line)
        record['ready_hex']=line.hex();os.close(slave);slave=None
        scope['boundary']('before-QUIT')
        assert os.write(master,b'QUIT\n')==5
        record['quit_sent']=True;end=time.monotonic()+10
        while time.monotonic()<end:
            assert scope['text_bytes'](O)<=record['log_limit_bytes'],'text budget'
            g.poll(.01)
            if ready.select(.01):scope['boundary']('QUIT-drain',allow_eof=True)
            if (g.exit_code is not None or g.exit_signal is not None) and scope['protocol'].state().eof:break
    scope['boundary']('after-QUIT-exit',allow_eof=True)
    assert scope['protocol'].state().eof and record.get('transport_eof')
    assert g.exit_code==0 and g.exit_signal is None
    record.update(complete=True,passed=True)
except BaseException as error:
    record['error']=repr(error)
finally:
    if g:
        record.update(exit_code=g.exit_code,exit_signal=g.exit_signal)
        record['cleanup']=g.close()
        for role,owner in [('inferior',g.inferior),('debugger',g.debugger_identity)]:
            now=process_identity(owner['pid']) if owner else None
            if now and now['start_ticks']==owner['start_ticks'] and now['state']!='Z':record['survivors'].append(role)
    for fd in (master,slave):
        if fd is not None:os.close(fd)
    raw.close();events.close()
    record.update(active=False,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    record['passed']=record['passed'] and not any(record['cleanup'].values()) and not record['survivors']
    record['files']={str(p.relative_to(O)):dict(bytes=p.stat().st_size,sha256=sha(p))
        for p in sorted(O.rglob('*')) if p.is_file() and p.name!='record.json'}
    save()
    assert scope['text_bytes'](O)<=record['log_limit_bytes']
    fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
print(json.dumps({k:record.get(k) for k in ['passed','complete','error','elapsed_seconds','transport_eof','exit_code','cleanup','survivors']}))
if not record['passed']:raise SystemExit(1)
