"""Compile an isolated KV-stream candidate against accepted e82 inputs."""
from pathlib import Path
import datetime,hashlib,json,os,shlex,shutil,signal,subprocess,time
base=Path(__file__).parent
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
baseline=root/'build-sycl-event-ack-registered-copy-v3-20261008/build'
original=root/'build-sycl-event-ack-registered-copy-v3-20261008/source/sycl/src/kernels/cuda/kv_stream.dp.cpp'
candidate=root/'build-sycl-kv-stream-safe-v3-20261008'
source=candidate/'source/kv_stream.dp.cpp'
out=base/'kv-stream-safe-v01402-build-v3';out.mkdir(mode=0o700)
def digest(p):
    with p.open('rb') as s:return hashlib.file_digest(s,'sha256').hexdigest()
env=dict(os.environ,PATH='/usr/bin:/bin:/opt/intel/oneapi/compiler/2026.1/bin',MKLROOT='/opt/intel/oneapi/mkl/2026.1',LIBRARY_PATH='/opt/intel/oneapi/mkl/2026.1/lib:/opt/intel/oneapi/compiler/2026.1/lib:/usr/lib/x86_64-linux-gnu')
env.pop('LD_LIBRARY_PATH',None);env.pop('LD_PRELOAD',None)
record={'active':True,'passed':False,'gpu_tested':False,'adopted':False,'steps':[],
        'scope':'Private KV translation-unit v3: atomic hit metadata/page-table read, defined host/device counters and copy representations, supported-launch guard, four unused root properties removed. All other accepted e82 archive/object inputs, flags and subgroup/ranges remain pinned. No GPU/throughput/full-context proof or runtime/global change.',
        'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'controller_sha256':digest(Path(__file__))}
started=time.monotonic()
def save():
    record['elapsed_seconds']=time.monotonic()-started;(out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
def run(label,argv,timeout=600):
    step={'label':label,'argv':list(map(str,argv))};record['steps'].append(step);save();begin=time.monotonic()
    with (out/(label+'.stdout')).open('wb') as a,(out/(label+'.stderr')).open('wb') as b:
        p=subprocess.Popen(step['argv'],cwd=baseline,env=env,stdout=a,stderr=b,start_new_session=True);step['pid']=p.pid;save()
        try:rc=p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait();raise
    step.update(exit_code=rc,elapsed_seconds=time.monotonic()-begin);save();assert rc==0,label+' failed'
save()
try:
    nr_path=base/'event-ack-no-root-prefill-v01402-build/record.json';nr=json.loads(nr_path.read_text());assert nr['passed'] and not nr['active']
    nr_binary=Path(nr['candidate_binary']);assert digest(nr_binary)==nr['candidate_binary_sha256']=='e82fc5de5480b72255b601759497d5810e86bf691350417ed0dc11ed6c429323'
    nr_archive=nr_binary.parent/'libstrata_prefill.a';assert digest(nr_archive)==nr['candidate_archive_sha256']
    prepare=base/'kv-stream-safe-v01402-source-candidate-v3/record.json';s=json.loads(prepare.read_text());assert s['source_prepared'] and not s['compiled'] and not s['gpu_tested']
    assert digest(source)==s['candidate_sha256'] and digest(original)==s['original_sha256']
    record.update(source_candidate_receipt_sha256=digest(prepare),baseline_build_receipt_sha256=digest(nr_path),baseline_binary_sha256=digest(nr_binary),original_source_sha256=digest(original),candidate_source_sha256=digest(source))
    commands=subprocess.check_output(['/usr/bin/ninja','-t','commands','strata'],cwd=baseline,env=env,text=True,timeout=30).splitlines()
    compiles=[shlex.split(x) for x in commands if x.endswith(' -c '+str(original))];assert len(compiles)==1
    argv=compiles[0];record['original_compile_argv']=argv.copy();obj=candidate/'kv_stream.dp.cpp.o';assert not obj.exists()
    for flag,value in [('-c',source),('-o',obj),('-MT',obj),('-MF',candidate/'kv_stream.dp.cpp.o.d')]:
        assert argv.count(flag)==1;argv[argv.index(flag)+1]=str(value)
    run('compile',argv)
    archive=baseline/'libstrata_kernels.a';private=candidate/'libstrata_kernels.a';assert not private.exists()
    record['original_archive_sha256']=digest(archive);shutil.copy2(archive,private)
    members=subprocess.check_output(['/usr/bin/ar','t',str(archive)],text=True).splitlines();assert len(members)==len(set(members)) and members.count(obj.name)==1
    run('archive-replace',['/usr/bin/ar','r',str(private),str(obj)]);run('archive-index',['/usr/bin/ranlib',str(private)])
    record['archive_members']=[]
    for n in members:
        a=subprocess.check_output(['/usr/bin/ar','p',str(archive),n]);b=subprocess.check_output(['/usr/bin/ar','p',str(private),n])
        assert b==obj.read_bytes() if n==obj.name else a==b
        record['archive_members'].append({'member':n,'replaced':n==obj.name,'before_sha256':hashlib.sha256(a).hexdigest(),'after_sha256':hashlib.sha256(b).hexdigest()})
    links=[shlex.split(x) for x in commands if ' -o strata ' in x];assert len(links)==1
    tokens=links[0];assert tokens[:2]==[':','&&'] and tokens[-2:]==['&&',':'];argv=tokens[2:-2];record['original_link_argv']=argv.copy()
    inputs={str(baseline/t):digest(baseline/t) for t in argv if t.endswith(('.a','.o')) and (baseline/t).is_file()};inputs[str(nr_archive)]=digest(nr_archive);record['link_input_sha256']=inputs
    assert argv.count('libstrata_kernels.a')==1 and argv.count('libstrata_prefill.a')==1
    argv=[str(private) if t=='libstrata_kernels.a' else str(nr_archive) if t=='libstrata_prefill.a' else t for t in argv]
    binary=candidate/'strata';assert not binary.exists();argv[argv.index('-o')+1]=str(binary);run('link',argv)
    assert digest(nr_binary)==record['baseline_binary_sha256'] and digest(original)==record['original_source_sha256']
    assert all(digest(Path(p))==sha for p,sha in inputs.items())
    record.update(passed=True,baseline_inputs_unchanged=True,candidate_binary=str(binary),candidate_binary_sha256=digest(binary),candidate_object_sha256=digest(obj),candidate_archive_sha256=digest(private))
except BaseException as error:record['error']=repr(error);raise
finally:record.update(active=False,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
print(json.dumps({k:record.get(k) for k in ['passed','error','elapsed_seconds','candidate_binary_sha256','baseline_inputs_unchanged']},indent=2))
