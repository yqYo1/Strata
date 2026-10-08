"""Bounded four-head real-model profile, or the same run without a profiler."""
from pathlib import Path
import array
import datetime
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
import types
from profile_supervisor import OwnedSession

b=Path(__file__).parent
r=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
os.chdir(r)
mode=int(sys.argv[1])
assert mode in [0,128,256]
out=b/('expanded-fp16-refresh-2048-cache'+str(mode));out.mkdir(mode=0o700)
binary=b/'strata-expanded-fp16-cache-refresh-candidate'
expected_sha='291130f1d9aae6161ff578def487cac9ad12480a6d2fe590da92a5225b00da68'
assert hashlib.sha256(binary.read_bytes()).hexdigest()==expected_sha
sys.path.insert(0,str(r/'sycl/tools'))
from owned_gdb import process_identity
first=json.loads((b/'owned-expanded-fp16-refresh-128-2048/record.json').read_text())
assert first['healthy'] and not first['active'] and not first['new_fault_messages']
assert first['binary_sha256']==expected_sha
for name in ['owned-expanded-fp16-refresh-zero-short','owned-expanded-fp16-refresh-128-2048','owned-dequant-no-root-refresh-short','owned-upstream-refresh-short','owned-upstream-refresh-2048']:
 previous=json.loads((b/name/'record.json').read_text())
 assert previous['healthy'] and not previous['active']
 for key in ['inferior','debugger']:
  old=previous[key];now=process_identity(old['pid']);assert not now or now['start_ticks']!=old['start_ticks']
reference=json.loads((b/'model-lease-copy-off/normal-mtp.json').read_text())
for flag,value in [('--max-context','4096'),('--prefill','1024'),('--expert-cache','128'),('--pcie-frac','0')]:
    reference['args'][reference['args'].index(flag)+1]=value
reference['args'] += ['--kv','int8','--ple-io','direct','--prompt-cache','0']
tokens=(r/'bench/results/2026-10-05-sycl-prefill-scaling/coding-context-tokens.txt').read_text().split()
assistant_suffix=['248046','198','248045','74455','198','248068','198','248069','271']
request='GEN 4 logprobs=5 '+','.join(tokens[:2039]+assistant_suffix)
reference['requests']=[dict(name='2048',request=request)]
control=json.loads((b/'dequant-refresh-2048-control/record.json').read_text())
assert control['passed'] and not control['active'] and not control['new_fault_messages'] and not control['job']['survivors']
from profile_supervisor import identity
for count in [0,128,256]:
 path=b/('expanded-fp16-refresh-2048-cache'+str(count))/'record.json'
 if not path.exists():continue
 previous=json.loads(path.read_text());assert previous['passed'] and not previous['active'] and not previous['new_fault_messages']
 for old in previous['job']['owned']:
  now=identity(old['pid']);assert not now or now['start_ticks']!=old['start_ticks']
actual=control['requests'][0]
reference['requests']=[dict(name='2048-r'+str(i),request=request,ids=actual['ids'],logprobs=actual['logprobs'],
 first_head={'file':str(b/'dequant-refresh-2048-control/2048.head.bin')}) for i in range(1,5)]
source=r/'sycl/tools/recover-xe.sh';m=types.ModuleType('env_and_journal')
exec(compile(source.read_text().split("<<'PY'\n",1)[1].rsplit('\nPY',1)[0],str(source),'exec'),m.__dict__)
runner=m.Runner(out)
env=m.health_environment();env['LD_LIBRARY_PATH']+=':/opt/intel/oneapi/mkl/2026.1/lib'
env.update(NEOReadDebugKeys='1',EnableDirectSubmission='0',STRATA_PREFILL_COMPACT='2',
           STRATA_PREFILL_LAYER_MAJOR='1',STRATA_PREFILL_RELEASE_DRAFT='1',
           STRATA_PREFILL_DRAFT_VERIFY='1',STRATA_PREFILL_EXPERT_WAIT_BATCH='0',STRATA_PREFILL_SYNC='1',
           STRATA_PREFILL_RELEASE_CACHE='1',STRATA_PREFILL_CACHE_ALLOC='vmm',STRATA_PREFILL_CACHE_RESTORE='ram',
           STRATA_PREFILL_LAYER_MAJOR_R_INPLACE='1',STRATA_PREFILL_LAYER_MAJOR_R_GPU='0',
           STRATA_PREFILL_ATTN_BATCH='128',STRATA_PREFILL_ATTN_LAYOUT='4',STRATA_PREFILL_TOPK_TUNED='1',
           STRATA_GDN_KEYHEAD='0',STRATA_GDN_KEYHEAD_TUNED='1')
env['STRATA_PREFILL_FP16_CACHE_EXPERTS']=str(mode)
head=out/'first-head.bin';env['STRATA_DUMP_FIRST_LOGITS']=str(head)
argv=[str(binary)]+reference['args']
record=dict(scope='Four fresh normal-MTP requests,2048 input tokens,prefill chunks256/1024/767 plus finalinputdecode; same updated-cache binary,original phase waits/progress,API logs/validationdisabled; warmupfirstexcluded; wholehead/IDs/LP and repeatedcache/main/MTP restores; not full256Kcapacity or unsynchronized throughput',
            mode=mode,warmup_requests=1,active=True,passed=False,requests=[],protocol=[],deadline_seconds=600,
            binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
            started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
job=None;offset=0;pending=b'';next_scan=0;cursor=None

def save():
    (out/'record.json').write_text(json.dumps(record,indent=2)+'\n')

def poll():
    global next_scan
    if time.monotonic()-job.started>record['deadline_seconds']: raise TimeoutError('model profile deadline')
    if job.child.poll() is not None: raise RuntimeError('target/profiler exited before reply '+str(job.child.returncode))
    if (out/'collect/stderr').stat().st_size>512*1024**2:raise RuntimeError('stderr limit')
    if time.monotonic()>next_scan:
        job.scan();save();next_scan=time.monotonic()+1

def line():
    global offset,pending
    until=time.monotonic()+180
    while time.monotonic()<until:
        if b'\n' in pending:
            item,_,pending=pending.partition(b'\n')
            text=item.decode(errors='replace').strip()
            record['protocol'].append(dict(text=text,epoch_us=time.time_ns()/1000))
            return text
        poll()
        with (out/'collect/stdout').open('rb') as f:
            f.seek(offset);chunk=f.read();offset+=len(chunk);pending+=chunk
        if not chunk:time.sleep(.005)
    raise TimeoutError('protocol reply deadline')

def send(text):
    poll();job.child.stdin.write((text+'\n').encode());job.child.stdin.flush()

try:
    cursor=m.journal_cursor(runner,'kernel-before')
    job=OwnedSession(argv,out/'collect',env,stdin=subprocess.PIPE)
    record['startup']=[];save()
    while True:
        value=line();record['startup'].append(value)
        if value.startswith('ERR'):raise RuntimeError(value)
        if value.startswith('READY '):break
    # Verify the profiler's actual injected environment only for our owned PID.
    try:
        raw=Path(f'/proc/{job.child.pid}/environ').read_bytes().split(b'\0')
        record['target_environment']={k:v for item in raw if b'=' in item
            for k,v in [item.decode().split('=',1)] if k.startswith(('UNITRACE_','XPTI_','SYCL_','UR_','ZE_','ZEL_','STRATA_')) or k in ['LD_PRELOAD','LD_LIBRARY_PATH','NEOReadDebugKeys','EnableDirectSubmission']}
    except FileNotFoundError:raise RuntimeError('owned target exited at READY')
    for expected in reference['requests']:
        head.unlink(missing_ok=True)
        req=dict(name=expected['name'],start_epoch_us=time.time_ns()/1000,ids=[],logprobs=[],protocol=[])
        record['active_request']=req['name'];send(expected['request']);save()
        while True:
            value=line();req['protocol'].append(dict(text=value,epoch_us=time.time_ns()/1000))
            if value.startswith('ERR'):raise RuntimeError(value)
            if value.startswith('T '):req['ids'].append(int(value.split()[1]))
            if value.startswith('LP '):req['logprobs'].append(value)
            if value.startswith('DONE '):req['done']=value;break
        req['end_epoch_us']=time.time_ns()/1000
        data=head.read_bytes();values=array.array('f');values.frombytes(data)
        assert len(values)==248320 and all(map(math.isfinite,values))
        assert len(req['ids'])==4 and len(req['logprobs'])==4
        req['equality']=dict(ids=req['ids']==expected['ids'],logprobs=req['logprobs']==expected['logprobs'],head=data==Path(expected['first_head']['file']).read_bytes())
        req['head_sha256']=hashlib.sha256(data).hexdigest()
        (out/(req['name']+'.head.bin')).write_bytes(data)
        record['requests'].append(req);save();assert all(req['equality'].values())
    record['active_request']=None
    send('QUIT');job.child.stdin.close()
    until=time.monotonic()+60
    while job.child.poll() is None and time.monotonic()<until:
        job.scan();time.sleep(.05)
    assert job.child.poll()==0
    text=(out/'collect/stderr').read_text(errors='replace')
    record['cache_allocations']=re.findall(r'strata prefill FP16 cache allocation: requested (\d+), capacity (\d+), bytes (\d+)',text)
    record['cache_reports']=re.findall(r'strata prefill FP16 cache: capacity (\d+), bytes (\d+), admitted (\d+), hits (\d+), uncached (\d+), skipped dequant launches (\d+)',text)
    record['release_pairs']={'release':text.count('strata mtp decode release:'),'restore':text.count('strata mtp decode restore:')}
    assert record['release_pairs']['release']==record['release_pairs']['restore']>=4
    assert all('REUSED 0' in [x['text'] for x in req['protocol']] for req in record['requests'])
    assert not any(k.startswith(('ZEL_','ZE_ENABLE_','UR_LOG_')) for k in record['target_environment'])
    if mode:
        assert len(record['cache_reports'])==len(record['cache_allocations'])==4
        assert all(0<int(x[0])<=mode and int(x[3])>0 and int(x[4])>0 and int(x[5])==2*int(x[3]) for x in record['cache_reports'])
    else:assert not record['cache_allocations'] and not record['cache_reports']
    record['passed']=True
except BaseException as error:
    record['error']=repr(error)
finally:
    if job:
        record['job']=job.close(force=job.child.poll() is None)
        record['elapsed_seconds']=time.monotonic()-job.started
        record['passed']=record['passed'] and not record['job']['survivors'] and record['job']['exit_code']==0
    if cursor:
        text=runner.run('kernel-after',['/usr/bin/journalctl','-k','--after-cursor',cursor,'--no-pager','-o','short-iso-precise'],seconds=5)
        record['new_fault_messages']=[x for x in text.splitlines() if ('0000:05:00.0' in x or 'xe ' in x) and m.FAULT.search(x)]
        record['passed']=record['passed'] and not record['new_fault_messages']
    record['steps']=runner.calls;record['active']=False
    record['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
print(json.dumps({k:v for k,v in record.items() if k not in ['job','target_environment','protocol','steps','requests']},indent=2))
print('requests',[(x['name'],x['done'],x['equality']) for x in record['requests']])
if not record['passed']:raise SystemExit(1)
