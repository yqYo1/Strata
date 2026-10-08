"""A logged exact-word check before profiling the real model."""
from pathlib import Path
import hashlib
import json
import types
from profile_supervisor import run

b=Path(__file__).parent
r=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
out=b/'unitrace-smoke';out.mkdir(mode=0o700)
unitrace=b/'unitrace-build-system-cc/build/unitrace'
build=json.loads((b/'unitrace-build-system-cc/record.json').read_text())
assert build['passed'] and hashlib.sha256(unitrace.read_bytes()).hexdigest()==build['binaries'][str(unitrace)]
src=r/'sycl/tools/recover-xe.sh'
m=types.ModuleType('probe_env')
exec(compile(src.read_text().split("<<'PY'\n",1)[1].rsplit('\nPY',1)[0],str(src),'exec'),m.__dict__)
runner=m.Runner(out)
cursor=m.journal_cursor(runner,'kernel-before')
record=dict(scope='First unitrace Level Zero/SYCL timeline instrumentation smoke with exact integer health check; diagnostic logging and validation; no metrics or throughput claim',passed=False,
            source_commit=build['source_commit'],unitrace_sha256=build['binaries'][str(unitrace)])
env=m.health_environment();env.update(NEOReadDebugKeys='1',EnableDirectSubmission='0')
env=m.diagnostic_environment(env)
try:
    args=[str(unitrace),'-h','-d','-s','--chrome-kernel-logging','--chrome-call-logging',
          '--chrome-sycl-logging','--output-dir-path',str(out),'-o',str(out/'summary.csv'),
          '/home/yayoi/.local/bin/strata-xe-health','0000:05:00.0']
    job=run(args,out/'collect',env,seconds=90);record['job']=job
    assert job['exit_code']==0 and not job['survivors']
    text=(out/'collect/stdout').read_text(errors='replace')
    assert 'PASS 0000:05:00.0: 3 rounds, 16384 exact words each' in text
    traces=[p for p in out.glob('*.json') if p.name!='record.json']
    record['traces']=[]
    for p in traces:
        data=json.loads(p.read_text());events=data['traceEvents']
        record['traces'].append(dict(path=str(p),bytes=p.stat().st_size,
              sha256=hashlib.sha256(p.read_bytes()).hexdigest(),events=len(events),
              categories=sorted({e.get('cat','') for e in events}),
              first_complete_events=[e for e in events if e.get('ph')=='X'][:8]))
    assert record['traces']
    record['passed']=True
finally:
    text=runner.run('kernel-after',['/usr/bin/journalctl','-k','--after-cursor',cursor,'--no-pager','-o','short-iso-precise'],seconds=5)
    record['new_fault_messages']=[x for x in text.splitlines() if ('0000:05:00.0' in x or 'xe ' in x) and m.FAULT.search(x)]
    record['passed']=record['passed'] and not record['new_fault_messages']
    record['steps']=runner.calls
    (out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items() if k!='job'},indent=2))
