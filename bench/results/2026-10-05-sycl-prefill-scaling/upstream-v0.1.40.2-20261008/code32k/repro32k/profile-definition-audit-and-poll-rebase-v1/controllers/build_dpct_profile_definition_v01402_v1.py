"""Compile only the seven unprofiled DPCT users with a consistent header."""
from pathlib import Path
import datetime,hashlib,json,os,shutil,signal,subprocess,time
base=Path(__file__).parent
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
build=root/'build-sycl-event-ack-registered-copy-v3-20261008/build'
candidate=root/'build-sycl-dpct-profile-definition-objects-v1-20261008';candidate.mkdir()
out=base/'dpct-profile-definition-v01402-build-v1';out.mkdir(mode=0o700)
def digest(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
env=dict(os.environ,PATH='/usr/bin:/bin:/opt/intel/oneapi/compiler/2026.1/bin',MKLROOT='/opt/intel/oneapi/mkl/2026.1',LIBRARY_PATH='/opt/intel/oneapi/mkl/2026.1/lib:/opt/intel/oneapi/compiler/2026.1/lib:/usr/lib/x86_64-linux-gnu')
env.pop('LD_LIBRARY_PATH',None);env.pop('LD_PRELOAD',None)
record=dict(active=True,passed=False,gpu_tested=False,adopted=False,steps=[],started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),controller_sha256=digest(__file__),scope='CPU-only seven compatible DPCT object replacements on7f054 indexer candidate. The shadow header fixes class/external-inline token definitions to the translated profiling branch. Existing profiled TUs retain identical helper bodies; explicit nonprofiling copy factory is unchanged. No polling change, compiler flag/arithmetic/launch change, GPU run or adoption.')
started=time.monotonic()
def save():
    record['elapsed_seconds']=time.monotonic()-started;(out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
def run(label,argv,timeout=600):
    step=dict(label=label,argv=list(map(str,argv)));record['steps'].append(step);save();begin=time.monotonic()
    with (out/(label+'.stdout')).open('wb') as a,(out/(label+'.stderr')).open('wb') as b:
        p=subprocess.Popen(step['argv'],cwd=build,env=env,stdout=a,stderr=b,start_new_session=True);step['pid']=p.pid;save()
        try:rc=p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait();raise
    step.update(exit_code=rc,elapsed_seconds=time.monotonic()-begin);save();assert rc==0,label
save()
try:
    prior_path=base/'indexer-spare-commit-v01402-build-v1/record.json';prior=json.loads(prior_path.read_text());assert prior['passed'] and not prior['active'] and prior['baseline_inputs_unchanged']
    baseline=Path(prior['candidate_binary']);assert digest(baseline)==prior['candidate_binary_sha256']=='7f054f8338c6552a5ae6dba548d2020ff2e3bd2661f4a41bb5a2289c56d07891'
    audit_path=base/'dpct-profile-definition-linked-tu-audit-v3/record.json';audit=json.loads(audit_path.read_text());assert audit['passed'] and not audit['active'] and audit['codepin_not_in_actual_linked_dependencies']
    review_path=base/'dpct-profile-definition-v01402-source-review-v3/record.json';review=json.loads(review_path.read_text());assert review['passed'] and not review['active'] and review['candidate_identical_to_translated_original']
    shadow=root/'build-sycl-dpct-profile-definition-v3-20261008/include';header=shadow/'dpct/device.hpp';assert digest(header)==review['candidate_header_sha256']
    selected=audit['required_unprofiled_tus'];assert len(selected)==7 and all(not x['profiling_macro_defined_in_source_or_flags'] for x in selected)
    record.update(baseline_binary_sha256=digest(baseline),baseline_build_receipt_sha256=digest(prior_path),source_review_sha256=digest(review_path),linked_tu_audit_sha256=digest(audit_path),shadow_header=str(header),shadow_header_sha256=digest(header))
    argv=list(prior['steps'][-1]['argv']);assert prior['steps'][-1]['label']=='link'
    inputs={str((build/p).resolve()):digest((build/p).resolve()) for p in argv if p.endswith(('.a','.o')) and (build/p).is_file()};record['baseline_link_input_sha256']=inputs
    archives={Path(p):subprocess.check_output(['/usr/bin/ar','t',p],text=True).splitlines() for p in inputs if p.endswith('.a')}
    replacements={};record['objects']=[]
    for index,item in enumerate(selected):
        source=Path(item['source']);assert digest(source)==item['source_sha256'];name=Path(item['object']).name
        matches=[p for p,members in archives.items() if name in members];assert len(matches)==1,(name,matches)
        archive=matches[0]
        if archive not in replacements:
            private=candidate/archive.name;assert not private.exists();shutil.copy2(archive,private);replacements[archive]=private
        objdir=candidate/'objects'/str(index);objdir.mkdir(parents=True);obj=objdir/name
        command=item['argv'].copy();assert command[0].endswith('/icpx') and command.count('-c')==1
        for flag,value in [('-o',obj),('-MT',obj),('-MF',objdir/(name+'.d'))]:
            assert command.count(flag)==1;command[command.index(flag)+1]=str(value)
        command.insert(1,'-I'+str(shadow));run('compile-'+str(index),command)
        dep=objdir/(name+'.d');assert str(header) in dep.read_text(), 'shadow device.hpp not selected'
        assert str(build.parent/'source/sycl/include/dpct/device.hpp') not in dep.read_text()
        record['objects'].append(dict(source=str(source),source_sha256=digest(source),object=str(obj),object_sha256=digest(obj),dependency=str(dep),dependency_sha256=digest(dep),archive=str(archive),member=name))
        run('replace-'+str(index),['/usr/bin/ar','r',str(replacements[archive]),str(obj)])
    record['archive_members']=[]
    for old,new in replacements.items():
        run('index-'+old.stem,['/usr/bin/ranlib',str(new)])
        members=archives[old];assert len(members)==len(set(members))
        for member in members:
            before=subprocess.check_output(['/usr/bin/ar','p',str(old),member]);after=subprocess.check_output(['/usr/bin/ar','p',str(new),member]);matched=[x for x in record['objects'] if x['archive']==str(old) and x['member']==member]
            assert len(matched)<=1
            if matched:assert hashlib.sha256(after).hexdigest()==matched[0]['object_sha256']
            else:assert before==after
            record['archive_members'].append(dict(archive=str(old),member=member,replaced=bool(matched),before_sha256=hashlib.sha256(before).hexdigest(),after_sha256=hashlib.sha256(after).hexdigest()))
    def rebase(value):
        p=(build/value).resolve()
        return str(replacements[p]) if p in replacements else value
    argv=[rebase(p) for p in argv];binary=candidate/'strata';argv[argv.index('-o')+1]=str(binary);run('link',argv)
    assert digest(baseline)==record['baseline_binary_sha256'] and all(digest(p)==sha for p,sha in inputs.items())
    record.update(passed=True,baseline_inputs_unchanged=True,candidate_binary=str(binary),candidate_binary_sha256=digest(binary),replaced_objects=7,replaced_archives={str(k):str(v) for k,v in replacements.items()},pending=['Current separate indexer capacity gate','logged32K complete numerical/owned-exit gate','complete256K repeat/restore/clipped/refusal/later32K gate before adoption','separate profiler data and matched quiet>=32K speed'])
except BaseException as e:record['error']=repr(e);raise
finally:record.update(active=False,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
print(json.dumps({k:record.get(k) for k in ['passed','error','elapsed_seconds','candidate_binary_sha256','replaced_objects']},indent=2))
