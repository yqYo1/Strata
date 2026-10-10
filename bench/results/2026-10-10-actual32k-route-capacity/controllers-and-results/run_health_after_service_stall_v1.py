from pathlib import Path
import datetime,fcntl,hashlib,json,os,types
B=Path(__file__).parent;O=B/'health-after-service-stall-root-v1';S=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05/sycl/tools/recover-xe.sh')
with (B/'owned-v0141-measurement.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 old=json.loads((B/'owned-prefill-route-service-v3-code32k-diagnostic-r1/record.json').read_text());assert not old['active'] and not old['new_fault_messages'] and not old['cleanup']['inferior_survived'] and not old['cleanup']['gdb_survived']
 assert not Path('/sys/bus/pci/devices/0000:05:00.0/devcoredump').exists()
 assert not O.exists();O.mkdir(mode=0o700)
 m=types.ModuleType('read_only_health');exec(compile(S.read_text().split("<<'PY'\n",1)[1].rsplit('\nPY',1)[0],str(S),'exec'),m.__dict__)
 r=m.Runner(O);os.environ.update(NEOReadDebugKeys='1',EnableDirectSubmission='0')
 j=dict(active=True,passed=False,scope='One bounded existing normal integer health probe after combined profiling Stager status stall; no inference/retry/reset/services/driver/config change.',source_sha256=hashlib.sha256(S.read_bytes()).hexdigest(),controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),calls=r.calls)
 try:
  cursor=m.journal_cursor(r,'kernel-before');m.check_health(types.SimpleNamespace(bdf='0000:05:00.0'),r,Path('/home/yayoi/.local/bin/strata-xe-health'),cursor=cursor);j['passed']=True
 except BaseException as e:j['error']=repr(e)
 finally:
  j.update(active=False,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());(O/'record.json').write_text(json.dumps(j,indent=2)+'\n');print(json.dumps({k:j.get(k) for k in ['active','passed','error']}))
 if not j['passed']:raise SystemExit(1)
