"""Root-owned real OwnedGdb/PTY transport check with a scripted INFO/READY/GEN/QUIT Python peer only."""
from pathlib import Path
import ast, datetime, errno, fcntl, hashlib, json, os, re, selectors, sys, time, tty

B=Path(__file__).parent
O=B/'cache-route-pairs-real-pty-cpu-validation-v3'
H=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05/sycl/tools')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
P=B/'run_owned_cache_route_pairs_v0141_code32k_v6.py'
assert sha(P)=='9304f84dfce23b099c02d0090ea3b8f9b158100d157b0175ab3e4cfda1a9f423'
assert sha(H/'owned_gdb.py')=='61e1d206bf57bb320051cdd65943d23722bc523a602d80c54b7e7a2ad0d3f0d1'
assert sha(B/'bounded_engine_protocol_v1.py')=='aa72bd7f1a9babf7c6e325b10634f1a046fe78fc2b2738e691d90624197131c1'
prior=B/'cache-route-pairs-v6-cpu-validation-v1/record.json'
assert sha(prior)=='39aef99dd369f87454ff90f67f7a458537e0ec49ab083fe7f1716bf715067578'
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
    scope='Real GDB/MI Linux PTY; actual v6 startup AST block and transport/send/line/classifier functions with scripted INFO/READY/GEN/QUIT peer; no engine/controller full admission/inference/GPU helper/model payload; model math not evaluated',
    boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
    started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    controller_sha256=sha(P),wrapper_sha256=sha(__file__),owned_gdb_sha256=sha(H/'owned_gdb.py'),
    preceding_cpu_receipt_sha256=sha(prior),log_limit_bytes=2*1024**2,deadline_seconds=40,
    cleanup={},survivors=[],protocol_timeout_seconds=10,model_math_evaluated=False)
def save():
    record['elapsed_seconds']=time.monotonic()-start
    temporary=O/'record.json.tmp';temporary.write_text(json.dumps(record,indent=2)+'\n');temporary.replace(O/'record.json')
child=O/'echo.py'
child.write_text("import sys\ninfo='INFO context=262144 kv=int8 kv_resident=32768 expert_slots=128 expert_cache_mib=325 expert_slots_primary=128 expert_cache_primary_mib=325 spec=4 mtp_max=0 lookup=0 vram_free_mib=1565 cvec=0 arena_mib=47962 pool_workers=5 pcie_frac=0.00 spec_min_p=0.00 conversation_cache_mib=0 conversation_cache_slots=4 conversation_cache_min_free_mib=2560 tail_role_token=-1 vram_elastic=0 engine=0.1.41'\nprint(info,flush=True);print('READY 262144 stop',flush=True)\ngen=0\nfor row in sys.stdin:\n if row.startswith('GEN 64 ckpt=1 logprobs=5 '):\n  if gen:sys.exit(4)\n  gen+=1;print('RESUME 0',flush=True)\n  for n in (8192,16384,24576,32767):print(f'PP {n} 32767 1 1.0',flush=True)\n  print('REUSED 0',flush=True)\n  for n in range(64):print('T 42',flush=True);print('LP -1.000000 42:-1.000000',flush=True)\n  print('DONE 64 32768 1.0 1.0 length 0 0 0 0 0 0 0 0.0 32768 0',flush=True)\n elif row=='QUIT\\n':sys.exit(0 if gen==1 else 5)\n else:sys.exit(2)\nsys.exit(3)\n")
names={'text_bytes','reserve_write','event','receive_stdout','available_stdout','boundary','protocol_line_kind','startup_step','poll','line','send'}
nodes=[n for n in ast.parse(P.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
assert {n.name for n in nodes}==names
raw=(O/'protocol.stdout.raw').open('wb');events=(O/'events.jsonl').open('w',buffering=1)
scope=dict(Path=Path,json=json,os=os,errno=errno,time=time,FAILURE_RESERVE=1024**2,
    out=O,raw=raw,events=events,record=record,started=start,
    protocol=BoundedProtocolLines(cr_policy='reject'),g=None,master=None,selectors=selectors,save=save,next_update=start+5)
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(P),'exec'),scope)
g=None;master=slave=None;save()
try:
    master,slave=os.openpty();tty.setraw(slave);os.set_blocking(master,False);scope['master']=master
    env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
    g=OwnedGdb(['/usr/bin/python3',str(child)],O/'debugger',env,inferior_tty_fd=slave);scope['g']=g
    g.command('-gdb-set may-call-functions off');g.run()
    record.update(inferior=g.inferior,debugger=g.debugger_identity)
    tree=ast.parse(P.read_text())
    starts=[]
    for node in ast.walk(tree):
        if not isinstance(node,ast.Try):continue
        for index,statement in enumerate(node.body):
            if isinstance(statement,ast.Assign) and any(isinstance(target,ast.Subscript) and
                isinstance(target.value,ast.Name) and target.value.id=='record' and
                isinstance(target.slice,ast.Constant) and target.slice.value=='startup' for target in statement.targets):
                starts.append(node.body[index:index+3])
    assert len(starts)==1 and isinstance(starts[0][-1],ast.While)
    # Execute the actual startup loop, including INFO state and READY check.
    exec(compile(ast.Module(body=starts[0],type_ignores=[]),str(P),'exec'),scope)
    assert scope['startup_seen']==['progress','terminal'] and len(record['startup'])==2
    os.close(slave);slave=None
    scope['send'](b'GEN 64 ckpt=1 logprobs=5 1\n')
    response=[]
    while True:
        value=scope['line']();kind=scope['protocol_line_kind'](value,'gen');response.append(value)
        assert kind!='error'
        if kind=='terminal':break
    assert sum(row.startswith('T ') for row in response)==64
    assert sum(row.startswith('LP ') for row in response)==64
    assert [int(row.split()[1]) for row in response if row.startswith('PP ')]==[8192,16384,24576,32767]
    record['scripted_GEN_protocol']=response
    scope['boundary']('after-DONE')
    scope['send'](b'QUIT\n')
    with selectors.DefaultSelector() as ready:
        ready.register(master,selectors.EVENT_READ)
        record['quit_sent']=True;end=time.monotonic()+10
        while time.monotonic()<end:
            assert scope['text_bytes'](O)<=record['log_limit_bytes'],'text budget'
            scope['poll']()
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
