"""Build the integrated v0.1.41 candidate with transferred qualified fixes."""
from pathlib import Path
import datetime, hashlib, json, os, subprocess, time

base=Path(__file__).parent
preceding=json.loads((base/'owned-dd5-phase-v2-v01402-code32k-diagnostic-r1/record.json').read_text())
assert not preceding['active'] and preceding['healthy'] and preceding['math_gate_passed'] and preceding['profiling_gate_passed']
assert preceding['exit_code']==0 and not preceding['exit_signal'] and not any(preceding['cleanup'].values())
for role in ['inferior','debugger']:
    assert not (Path('/proc')/str(preceding[role]['pid'])).exists()
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-v0.1.41-20261009')
commit='1eb89482a4afd20277ae0405780ed4f8eb98eb20'
ggml=Path('/home/yayoi/orca/workspaces/Strata/ggml-sycl-pinned')
def git(where,*args):return subprocess.check_output(['git',*args],cwd=where,text=True).strip()
assert git(root,'rev-parse','HEAD')==commit and not git(root,'status','--porcelain')
assert git(ggml,'rev-parse','HEAD')=='3cf03257f219afbe7334045ff7c6a06ac68c627d'
assert not git(ggml,'status','--porcelain')
tests=json.loads((base/'upstream-v0.1.41-server-tests-20261009-v2/record.json').read_text())
assert not tests['active'] and tests['passed'] and tests['exit_code']==0
out=base/'integrated-upstream-v0.1.41-20261009-v2';assert not out.exists();out.mkdir(mode=0o700)
build=root/'build-sycl-integrated-20261009';assert build.is_dir()
env_bytes=subprocess.check_output(['/bin/bash','-c','source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 && env -0'])
env=dict(x.decode().split('=',1) for x in env_bytes.split(b'\0') if b'=' in x)
record={'active':True,'passed':False,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope':'Integrated v0.1.41 with unchanged upstream server features and transferred qualified SYCL completion/state/KV corrections. Same project-default Release options and pinned ggml/toolchain as the pure build. No QSA or GEMM scalar speed experiment, no GPU invocation.',
        'commit':commit,'root':str(root),'build':str(build),'ggml_commit':git(ggml,'rev-parse','HEAD'),
        'source_status_before':'','ggml_status_before':'','steps':[],
        'environment':{k:env[k] for k in ['PATH','LD_LIBRARY_PATH','MKLROOT','CMPLR_ROOT','ONEAPI_ROOT'] if k in env},
        'retained_failed_build_receipt_sha256':hashlib.sha256((base/'integrated-upstream-v0.1.41-20261009/build-record.json').read_bytes()).hexdigest(),
        'predecessor_logged_phase_receipt_sha256':hashlib.sha256((base/'owned-dd5-phase-v2-v01402-code32k-diagnostic-r1/record.json').read_bytes()).hexdigest()}
start=time.monotonic();p=out/'build-record.json'
def save():record['elapsed_seconds']=time.monotonic()-start;p.write_text(json.dumps(record,indent=2)+'\n')
try:
    for label,args in [
        ('configure',['/usr/bin/cmake','-S',str(root/'sycl'),'-B',str(build),'-G','Ninja','-DCMAKE_CXX_COMPILER=/opt/intel/oneapi/compiler/2026.1/bin/icpx','-DCMAKE_C_COMPILER=/opt/intel/oneapi/compiler/2026.1/bin/icx','-DCMAKE_BUILD_TYPE=Release','-DSTRATA_GGML_DIR='+str(ggml)]),
        ('build',['/usr/bin/cmake','--build',str(build),'--parallel','2','--target','strata','--','-k','0'])]:
        step={'label':label,'argv':args,'cwd':str(root)};record['steps'].append(step);save();t=time.monotonic()
        with (out/(label+'.stdout')).open('w') as a,(out/(label+'.stderr')).open('w') as c:
            child=subprocess.Popen(args,cwd=root,env=env,stdout=a,stderr=c,start_new_session=True);step['pid']=child.pid;save()
            try:step['exit_code']=child.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                import signal
                os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=15);raise
        step['elapsed_seconds']=time.monotonic()-t;save()
        assert step['exit_code']==0,label
    binary=build/'strata'
    with binary.open('rb') as f:record['binary_sha256']=hashlib.file_digest(f,'sha256').hexdigest()
    record['binary']=str(binary)
    record['source_status_after']=git(root,'status','--porcelain');assert not record['source_status_after']
    record['ggml_status_after']=git(ggml,'status','--porcelain');assert not record['ggml_status_after']
    assert git(root,'rev-parse','HEAD')==commit
    record['passed']=True
except BaseException as error:record['error']=repr(error)
finally:
    record.update(active=False,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
print(json.dumps({k:record.get(k) for k in ['active','passed','elapsed_seconds','commit','binary_sha256','error']}))
if not record['passed']:raise SystemExit(1)
