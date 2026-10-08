from pathlib import Path
import datetime,hashlib,json,os,signal,subprocess,sys,time,types

root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
base=Path(__file__).parent
stage=sys.argv[1];assert stage in ('cli','serve')
capacity=base/'full-context-copy-off'
out=capacity/(stage+'-supervisor');out.mkdir();out.chmod(0o700)
frozen=base/'strata-residency-candidate'
expected='79a4b363d33f66f41d910be6274e609b4eb73f62afb0dc49bca72bf2a538d223'
assert hashlib.sha256(frozen.read_bytes()).hexdigest()==expected
for name in ('model-retained-copy-off','model-lease-copy-off'):
 assert json.loads((base/name/'record.json').read_text())['healthy']
src=root/'sycl/tools/recover-xe.sh';mod=types.ModuleType('readonly')
exec(compile(src.read_text().split("<<'PY'\n",1)[1].rsplit('\nPY',1)[0],str(src),'exec'),mod.__dict__)
r=mod.Runner(out)
env=json.loads((capacity/'environment.json').read_text())
controller=root/'bench/results/2026-10-05-sycl-prefill-scaling/residual-inplace/check_full_context.py'
(out/'controller.py').write_bytes(controller.read_bytes())
record=dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),stage=stage,
 scope='Actual 262144-cell capacity; DS and copy offload disabled, main-cache VMM release and MTP immutable-weight release enabled; no GDB, no prevention/default-settings claim',
 healthy=False,binary_sha256=expected,controller_sha256=hashlib.sha256(controller.read_bytes()).hexdigest(),
 supervisor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 environment_sha256=hashlib.sha256((capacity/'environment.json').read_bytes()).hexdigest(),steps=[])
def save():
 record['steps']=r.calls;(out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
save();cursor=mod.journal_cursor(r,'kernel-before')
summary_path=None
try:
 for current in (('boundary','cli') if stage=='cli' else ('serve',)):
  seconds=30 if current=='boundary' else (7200 if current=='cli' else 10800)
  argv=['/usr/bin/python3',str(controller),'--stage',current,'--context','262144',
        '--recovery',str(capacity),'--executable',str(frozen),'--environment-file',str(capacity/'environment.json'),
        '--release-draft','--verify-draft','--job-timeout',str(seconds),'--protocol-timeout','5400','--shutdown-timeout','30']
  summary_path=capacity/(current+'-262144-'+frozen.stem+'-draft-lease-verified')/'summary.json'
  record['active_stage']=current;record['summary_path']=str(summary_path);save()
  r.run(current,argv,seconds=seconds+120,env=env)
  summary=json.loads(summary_path.read_text());assert summary['completed']
  assert all(p['exit_code']==0 or current=='boundary' and p['exit_code']==2 for p in summary['processes'])
  assert all(not p['still_alive'] for p in summary['processes'])
  record.setdefault('completed_stages',[]).append(current);save()
except BaseException as exc:
 record['error']=repr(exc)
finally:
 # The controller owns separate engine process groups. If it itself crashes or
 # hits its outer deadline, do not leave an engine behind or trust a reused PID.
 if summary_path is not None and summary_path.exists():
  summary=json.loads(summary_path.read_text())
  for proc in summary.get('processes',[]):
   path=Path(f"/proc/{proc['pid']}/stat")
   if not path.exists():continue
   stat=path.read_text();fields=stat[stat.rfind(')')+2:].split()
   if int(fields[19])!=proc['start_ticks'] or fields[0]=='Z':continue
   actions=[]
   for sig in (signal.SIGTERM,signal.SIGKILL):
    try:os.killpg(proc['pid'],sig);actions.append(sig.name)
    except ProcessLookupError:break
    end=time.monotonic()+2
    while path.exists() and time.monotonic()<end:
     stat=path.read_text();fields=stat[stat.rfind(')')+2:].split()
     if int(fields[19])!=proc['start_ticks'] or fields[0]=='Z':break
     time.sleep(0.05)
   record.setdefault('orphan_cleanup',[]).append(dict(pid=proc['pid'],actions=actions))
   record['error']=record.get('error','controller exited while its engine was still alive')
   if path.exists():
    stat=path.read_text();fields=stat[stat.rfind(')')+2:].split()
    if int(fields[19])==proc['start_ticks'] and fields[0]!='Z':
     (base/'stalled-writer.json').write_text(json.dumps(proc,indent=2)+'\n')
     record['surviving_engine_pid']=proc['pid']
 save()
 log=r.run('kernel-after',['/usr/bin/journalctl','-k','--after-cursor',cursor,'--no-pager','-o','json'])
 rows=[json.loads(line) for line in log.splitlines() if line.startswith('{')]
 (out/'kernel.json').write_text(json.dumps(rows,indent=2)+'\n')
 record['new_fault_messages']=[x.get('MESSAGE','') for x in rows if
  (('xe' in x.get('MESSAGE','') or '0000:05:00.0' in x.get('MESSAGE','')) and mod.FAULT.search(x.get('MESSAGE',''))) or
  ('segfault' in x.get('MESSAGE','') and 'strata' in x.get('MESSAGE',''))]
record.update(active_stage=None,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
record['healthy']='error' not in record and not record['new_fault_messages']
save();print(json.dumps(record,indent=2))
if not record['healthy']:raise SystemExit(1)
