"""Replace three pointer-object aliases with direct typed assignments."""
from pathlib import Path
import datetime,difflib,hashlib,json,os,shlex,shutil,signal,subprocess,time
base=Path(__file__).parent
root=Path("/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05")
baseline=root/'build-sycl-event-ack-registered-copy-v3-20261008/build'
original=root/'build-sycl-event-ack-registered-copy-v3-20261008/source/sycl/src/core/layer.cpp'
candidate=root/'build-sycl-kv-access-safe-v1-20261008';candidate.mkdir()
(candidate/'source').mkdir();source=candidate/'source/layer.cpp'
out=base/'kv-access-safe-v01402-build-v1';out.mkdir(mode=0o700)
def digest(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
text=original.read_text();changed=text
changes=[
 ('*(void **)&d = (uint8_t *)h','d = h'),
 ('*(void **)&m_step = (int32_t *)st.host_step','m_step = st.host_step'),
 ('*(void **)&m_pos = (int32_t *)st.host_pos','m_pos = st.host_pos')]
for a,b in changes:
 assert changed.count(a)==1,a;changed=changed.replace(a,b)
assert '*(void **)&' not in changed
source.write_text(changed)
(out/'candidate.diff').write_text(''.join(difflib.unified_diff(text.splitlines(True),changed.splitlines(True),fromfile=str(original),tofile=str(source))))
env=dict(os.environ,PATH='/usr/bin:/bin:/opt/intel/oneapi/compiler/2026.1/bin',MKLROOT='/opt/intel/oneapi/mkl/2026.1',LIBRARY_PATH='/opt/intel/oneapi/mkl/2026.1/lib:/opt/intel/oneapi/compiler/2026.1/lib:/usr/lib/x86_64-linux-gnu')
env.pop('LD_LIBRARY_PATH',None);env.pop('LD_PRELOAD',None)
record=dict(active=True,passed=False,gpu_tested=False,adopted=False,steps=[],
 started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 controller_sha256=digest(Path(__file__)),original_source_sha256=digest(original),candidate_source_sha256=digest(source),
 scope='Only three direct typed pointer assignments in layer.cpp. Avoid writing pointer objects through void** aliases while retaining the same host-USM addresses, allocation/check/error handling, queue/event/graph lifetime and data/math. Link accepted v3 KV and no-root prefill. This is not a hang attribution or GPU/full-context proof.')
started=time.monotonic()
def save():
 record['elapsed_seconds']=time.monotonic()-started;(out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
def run(label,argv,timeout=600):
 step=dict(label=label,argv=list(map(str,argv)));record['steps'].append(step);save();begin=time.monotonic()
 with (out/(label+'.stdout')).open('wb') as a,(out/(label+'.stderr')).open('wb') as b:
  p=subprocess.Popen(step['argv'],cwd=baseline,env=env,stdout=a,stderr=b,start_new_session=True);step['pid']=p.pid;save()
  try:rc=p.wait(timeout=timeout)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait();raise
 step.update(exit_code=rc,elapsed_seconds=time.monotonic()-begin);save();assert rc==0,label
save()
try:
 kvp=base/'kv-stream-safe-v01402-build-v3/record.json';kv=json.loads(kvp.read_text());assert kv['passed'] and not kv['active'] and kv['baseline_inputs_unchanged']
 nrp=base/'event-ack-no-root-prefill-v01402-build/record.json';nr=json.loads(nrp.read_text());assert nr['passed'] and not nr['active']
 prior_binary=Path(kv['candidate_binary']);assert digest(prior_binary)==kv['candidate_binary_sha256']=='1441ad556cb1d8ecddf42e525a1462cc53f9251fae4734c008046e10e8b0e599'
 kernel=prior_binary.parent/'libstrata_kernels.a';assert digest(kernel)==kv['candidate_archive_sha256']
 prefill=Path(nr['candidate_binary']).parent/'libstrata_prefill.a';assert digest(prefill)==nr['candidate_archive_sha256']
 record.update(baseline_binary_sha256=digest(prior_binary),kv_build_receipt_sha256=digest(kvp),prefill_build_receipt_sha256=digest(nrp))
 commands=subprocess.check_output(['/usr/bin/ninja','-t','commands','strata'],cwd=baseline,env=env,text=True,timeout=30).splitlines()
 compiles=[shlex.split(x) for x in commands if x.endswith(' -c '+str(original))];assert len(compiles)==1
 argv=compiles[0];record['original_compile_argv']=argv.copy();obj=candidate/'layer.cpp.o'
 for flag,value in [('-c',source),('-o',obj),('-MT',obj),('-MF',candidate/'layer.cpp.o.d')]:
  assert argv.count(flag)==1;argv[argv.index(flag)+1]=str(value)
 run('compile',argv)
 archive=baseline/'libstrata_core.a';private=candidate/'libstrata_core.a';shutil.copy2(archive,private)
 record['original_archive_sha256']=digest(archive)
 members=subprocess.check_output(['/usr/bin/ar','t',str(archive)],text=True).splitlines();assert members.count(obj.name)==1 and len(members)==len(set(members))
 run('archive-replace',['/usr/bin/ar','r',str(private),str(obj)]);run('archive-index',['/usr/bin/ranlib',str(private)])
 record['archive_members']=[]
 for n in members:
  a=subprocess.check_output(['/usr/bin/ar','p',str(archive),n]);b=subprocess.check_output(['/usr/bin/ar','p',str(private),n])
  assert b==obj.read_bytes() if n==obj.name else a==b
  record['archive_members'].append(dict(member=n,replaced=n==obj.name,before_sha256=hashlib.sha256(a).hexdigest(),after_sha256=hashlib.sha256(b).hexdigest()))
 links=[shlex.split(x) for x in commands if ' -o strata ' in x];assert len(links)==1
 tokens=links[0];assert tokens[:2]==[':','&&'] and tokens[-2:]==['&&',':'];argv=tokens[2:-2];record['original_link_argv']=argv.copy()
 inputs={str(baseline/t):digest(baseline/t) for t in argv if t.endswith(('.a','.o')) and (baseline/t).is_file()}
 inputs[str(kernel)]=digest(kernel);inputs[str(prefill)]=digest(prefill);record['link_input_sha256']=inputs
 assert argv.count('libstrata_core.a')==1 and argv.count('libstrata_kernels.a')==1 and argv.count('libstrata_prefill.a')==1
 argv=[str(private) if t=='libstrata_core.a' else str(kernel) if t=='libstrata_kernels.a' else str(prefill) if t=='libstrata_prefill.a' else t for t in argv]
 binary=candidate/'strata';argv[argv.index('-o')+1]=str(binary);run('link',argv)
 assert digest(prior_binary)==record['baseline_binary_sha256'] and digest(original)==record['original_source_sha256']
 assert all(digest(Path(p))==sha for p,sha in inputs.items())
 record.update(passed=True,baseline_inputs_unchanged=True,candidate_binary=str(binary),candidate_binary_sha256=digest(binary),candidate_object_sha256=digest(obj),candidate_archive_sha256=digest(private))
except BaseException as e:record['error']=repr(e);raise
finally:record.update(active=False,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
print(json.dumps({k:record.get(k) for k in ['passed','elapsed_seconds','error','candidate_binary_sha256','baseline_inputs_unchanged']},indent=2))
