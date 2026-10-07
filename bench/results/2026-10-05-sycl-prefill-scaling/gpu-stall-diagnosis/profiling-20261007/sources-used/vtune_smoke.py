"""CPU software sampling and one bounded logged XPU health collection."""
from pathlib import Path
import hashlib
import json
import os
import types
from profile_supervisor import run

b=Path(__file__).parent
r=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
out=b/'vtune-smoke';out.mkdir(mode=0o700)
vtune='/opt/intel/oneapi/vtune/2026.4/bin64/vtune'
src=r/'sycl/tools/recover-xe.sh'
m=types.ModuleType('probe_env')
exec(compile(src.read_text().split("<<'PY'\n",1)[1].rsplit('\nPY',1)[0],str(src),'exec'),m.__dict__)
health=json.loads((b/'post-workspace-full-health/record.json').read_text())
assert health['healthy']
prev=json.loads((b/'full-context-workspace-reclaim-serve/record.json').read_text())
assert not prev['active'] and not prev['new_fault_messages']
for key in ['inferior','debugger']: assert not Path('/proc',str(prev[key]['pid'])).exists()
runner=m.Runner(out)
cursor=m.journal_cursor(runner,'kernel-before')
record=dict(scope='VTune CPU software sampling and small exact-word GPU API/timeline smoke; hardware counters disabled; not model tuning or transfer throughput',
            vtune=vtune, helper_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), jobs=[])
try:
    args=[vtune,'-collect','hotspots','-knob','sampling-mode=sw','-knob',
          'enable-characterization-insights=false','-knob','enable-stack-collection=true',
          '-data-limit','128','-no-summary','-result-dir',str(out/'cpu-result'),'--',
          '/usr/bin/python3','-c','import time; end=time.monotonic()+2; n=1\nwhile time.monotonic()<end: n=(n*1664525+1013904223)&0xffffffff\nprint(n)']
    cpu=run(args,out/'cpu-collect',dict(os.environ));record['jobs'].append(cpu)
    env=m.health_environment();env.update(NEOReadDebugKeys='1',EnableDirectSubmission='0')
    env=m.diagnostic_environment(env)
    args=[vtune,'-collect','xpu-offload','-knob','enable-characterization-insights=false',
          '-knob','collect-cpu-sampling=false','-knob','collect-programming-api=true',
          '-data-limit','128','-no-summary','-result-dir',str(out/'gpu-result'),'--',
          '/home/yayoi/.local/bin/strata-xe-health','0000:05:00.0']
    gpu=run(args,out/'gpu-collect',env);record['jobs'].append(gpu)
    stdout=(out/'gpu-collect/stdout').read_text(errors='replace')
    record['gpu_exact_words_pass']='PASS 0000:05:00.0: 3 rounds, 16384 exact words each' in stdout
finally:
    text=runner.run('kernel-after',['/usr/bin/journalctl','-k','--after-cursor',cursor,'--no-pager','-o','short-iso-precise'],seconds=5)
    record['new_fault_messages']=[x for x in text.splitlines() if ('0000:05:00.0' in x or 'xe ' in x) and m.FAULT.search(x)]
    record['steps']=runner.calls
    (out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items() if k!='jobs'},indent=2))
print('exit_codes',[x['exit_code'] for x in record['jobs']])
