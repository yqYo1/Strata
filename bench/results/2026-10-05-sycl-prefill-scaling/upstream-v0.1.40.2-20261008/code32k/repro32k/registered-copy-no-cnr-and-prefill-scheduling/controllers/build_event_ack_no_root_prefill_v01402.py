"""Offline scheduling-only candidate from the matched registered-copy build."""
from pathlib import Path
import datetime, difflib, hashlib, json, os, re, shlex, shutil, signal, subprocess, time

base=Path(__file__).parent
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
baseline=root/'build-sycl-event-ack-registered-copy-v3-20261008/build'
source=root/'build-sycl-event-ack-registered-copy-v3-20261008/source/sycl/src/prefill/kernels.dp.cpp'
candidate=root/'build-sycl-event-ack-no-root-prefill-20261008'
out=base/'event-ack-no-root-prefill-v01402-build'
out.mkdir(mode=0o700);(candidate/'source').mkdir(parents=True)

def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

env=dict(os.environ,PATH='/usr/bin:/bin:/opt/intel/oneapi/compiler/2026.1/bin',
         MKLROOT='/opt/intel/oneapi/mkl/2026.1',
         LIBRARY_PATH='/opt/intel/oneapi/mkl/2026.1/lib:/opt/intel/oneapi/compiler/2026.1/lib:/usr/lib/x86_64-linux-gnu')
env.pop('LD_LIBRARY_PATH',None);env.pop('LD_PRELOAD',None)
record={'active':True,'passed':False,'gpu_tested':False,'adopted':False,'steps':[],
        'scope':'Remove only14 use_root_sync properties from the prefill kernel translation unit. No root-group operations or cross-work-group atomic coordination occur in that file; inspected recurrences synchronize their own local memory and partition state by head/column. All kernel bodies, arithmetic, ranges, subgroup attributes, buffers and other queues/graphs are unchanged. This is a private scheduling experiment, not a root-cause, speed or full256K claim. Retain existing flags in the core persistent/coordination kernels.',
        'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'controller_sha256':digest(Path(__file__))}
started=time.monotonic()
def save():
    record['elapsed_seconds']=time.monotonic()-started
    (out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
def run(label,argv,timeout=300):
    step={'label':label,'argv':list(map(str,argv))};record['steps'].append(step);save();before=time.monotonic()
    with (out/(label+'.stdout')).open('wb') as stdout,(out/(label+'.stderr')).open('wb') as stderr:
        process=subprocess.Popen(step['argv'],cwd=baseline,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
        step['pid']=process.pid;save()
        try:rc=process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL);process.wait();raise
    step.update(exit_code=rc,elapsed_seconds=time.monotonic()-before);save();assert rc==0,label+' failed'

save()
try:
    receipt=base/'event-ack-registered-copy-v01402-matched-build/record.json'
    old=json.loads(receipt.read_text());assert old['passed'] and not old['active'] and not old['selected_configuration_differences']
    assert digest(baseline/'strata')==old['candidate_binary_sha256']
    record['baseline_build_receipt_sha256']=digest(receipt)
    record['baseline_binary_sha256']=old['candidate_binary_sha256']
    original=source.read_text()
    assert not any(word in original for word in ['get_root_group','root_group<','grid_group','atomic_ref','atomicAdd','atomicCAS'])
    pattern=r'^(?P<indent>[ \t]*)sycl::ext::oneapi::experimental::use_root_sync\};$'
    matches=list(re.finditer(pattern,original,re.MULTILINE));assert len(matches)==14
    sites=[]
    for match in matches:
        tail=original[match.end():match.end()+1800]
        kernel=re.search(r'class ([A-Za-z_0-9]+)',tail);assert kernel
        sites.append({'line':original[:match.start()].count('\n')+1,'kernel':kernel[1]})
    replacement,count=re.subn(pattern,lambda match:match['indent']+'};',original,flags=re.MULTILINE)
    assert count==14 and 'use_root_sync' not in replacement
    assert replacement==original.replace('sycl::ext::oneapi::experimental::use_root_sync','')
    record['removed_sites']=sites
    private_source=candidate/'source/kernels.dp.cpp'
    private_source.write_text(replacement)
    (out/'source.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),replacement.splitlines(True),fromfile='kernels.original.cpp',tofile='kernels.no-root.cpp')))
    record['original_source_sha256']=digest(source);record['candidate_source_sha256']=digest(private_source)
    commands=subprocess.check_output(['/usr/bin/ninja','-t','commands','strata'],cwd=baseline,env=env,text=True,timeout=30).splitlines()
    compiles=[shlex.split(line) for line in commands if line.endswith(' -c '+str(source))];assert len(compiles)==1
    argv=compiles[0];record['original_compile_argv']=argv.copy()
    obj=candidate/'kernels.dp.cpp.o'
    for flag,value in [('-c',private_source),('-o',obj),('-MT',obj),('-MF',candidate/'kernels.dp.cpp.o.d')]:
        assert argv.count(flag)==1;argv[argv.index(flag)+1]=str(value)
    run('compile',argv)
    archive=baseline/'libstrata_prefill.a';private_archive=candidate/'libstrata_prefill.a'
    record['original_archive_sha256']=digest(archive);shutil.copy2(archive,private_archive)
    members=subprocess.check_output(['/usr/bin/ar','t',str(archive)],text=True).splitlines()
    assert len(members)==len(set(members)) and members.count(obj.name)==1
    run('archive-replace',['/usr/bin/ar','r',str(private_archive),str(obj)])
    run('archive-index',['/usr/bin/ranlib',str(private_archive)])
    record['archive_members']=[]
    for name in members:
        before=subprocess.check_output(['/usr/bin/ar','p',str(archive),name])
        after=subprocess.check_output(['/usr/bin/ar','p',str(private_archive),name])
        assert after==obj.read_bytes() if name==obj.name else before==after
        record['archive_members'].append({'member':name,'replaced':name==obj.name,'before_sha256':hashlib.sha256(before).hexdigest(),'after_sha256':hashlib.sha256(after).hexdigest()})
    links=[shlex.split(line) for line in commands if ' -o strata ' in line];assert len(links)==1
    tokens=links[0];assert tokens[:2]==[':','&&'] and tokens[-2:]==['&&',':'];argv=tokens[2:-2]
    record['original_link_argv']=argv.copy()
    inputs={str(baseline/token):digest(baseline/token) for token in argv if token.endswith(('.o','.a')) and (baseline/token).is_file()}
    record['link_input_sha256']=inputs
    assert argv.count('libstrata_prefill.a')==1
    argv=[str(private_archive) if token=='libstrata_prefill.a' else token for token in argv]
    binary=candidate/'strata';argv[argv.index('-o')+1]=str(binary);run('link',argv)
    assert digest(baseline/'strata')==record['baseline_binary_sha256'] and digest(source)==record['original_source_sha256']
    assert all(digest(Path(path))==value for path,value in inputs.items())
    record.update(passed=True,baseline_inputs_unchanged=True,candidate_binary=str(binary),candidate_binary_sha256=digest(binary),candidate_object_sha256=digest(obj),candidate_archive_sha256=digest(private_archive))
except BaseException as error:
    record['error']=repr(error);raise
finally:
    record.update(active=False,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
print(json.dumps({key:record.get(key) for key in ['passed','error','elapsed_seconds','candidate_binary_sha256','baseline_inputs_unchanged']},indent=2))
