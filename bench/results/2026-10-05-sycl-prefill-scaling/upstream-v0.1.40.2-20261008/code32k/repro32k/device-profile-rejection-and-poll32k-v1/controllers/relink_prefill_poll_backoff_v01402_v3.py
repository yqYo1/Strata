"""Rebase the CPU-tested polling object onto DD5, with actual linked-header users."""
from pathlib import Path
import datetime,hashlib,json,os,shlex,shutil,signal,subprocess,time

base=Path(__file__).parent
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
build=root/'build-sycl-event-ack-registered-copy-v3-20261008/build'
candidate=root/'build-sycl-prefill-poll-backoff-v3-20261008';candidate.mkdir()
out=base/'prefill-poll-backoff-v01402-relink-v3';out.mkdir(mode=0o700)
def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
env=dict(os.environ,PATH='/usr/bin:/bin:/opt/intel/oneapi/compiler/2026.1/bin',MKLROOT='/opt/intel/oneapi/mkl/2026.1',LIBRARY_PATH='/opt/intel/oneapi/mkl/2026.1/lib:/opt/intel/oneapi/compiler/2026.1/lib:/usr/lib/x86_64-linux-gnu')
env.pop('LD_LIBRARY_PATH',None);env.pop('LD_PRELOAD',None)
record={'active':True,'passed':False,'gpu_tested':False,'adopted':False,'steps':[],
        'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'controller_sha256':digest(__file__),
        'scope':'CPU-only link/source inventory: replace only DD5 prefill.cpp.o with the unchanged CPU-tested polling object. Verify every prefill archive member, all baseline link inputs and actual linked host_wait.hpp users. No GPU workload or claim that backoff fixes a pending native event.'}
started=time.monotonic()
def save():
    record['elapsed_seconds']=time.monotonic()-started
    (out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
save()
try:
    uniform_path=base/'dpct-profile-definition-v01402-build-v1/record.json'
    uniform=json.loads(uniform_path.read_text())
    assert uniform['passed'] and not uniform['active'] and uniform['baseline_inputs_unchanged']
    baseline=Path(uniform['candidate_binary'])
    assert digest(baseline)==uniform['candidate_binary_sha256']=='dd5efb9002167481d180f370b86fb6978d724f37745f7e4a923432296725ad34'
    record.update(baseline_binary_sha256=digest(baseline),baseline_build_receipt_sha256=digest(uniform_path))
    poll_path=base/'prefill-poll-backoff-v01402-build-v1/record.json'
    poll=json.loads(poll_path.read_text())
    assert poll['passed'] and not poll['active'] and poll['cpu_ownership_deadline_cancellation_checks_passed'] and poll['baseline_inputs_unchanged']
    old=Path(poll['candidate_binary']).parent
    archive=old/'libstrata_prefill.a'
    assert digest(archive)==poll['candidate_archive_sha256']
    compiled=next(x['argv'] for x in poll['steps'] if x['label']=='compile')
    dep=Path(compiled[compiled.index('-MF')+1])
    audit_path=base/'prefill-poll-backoff-v01402-source-review-v1/record.json'
    audit=json.loads(audit_path.read_text());assert audit['passed'] and not audit['active']
    assert digest(dep)==audit['dependency_file_sha256']
    shadow=old/'include/strata/host_wait.hpp'
    assert digest(shadow)==poll['candidate_header_sha256']
    assert str(shadow) in dep.read_text().replace('\\\n',' ')
    actual_source=Path(compiled[compiled.index('-c')+1])
    assert digest(actual_source)==poll['prefill_source_sha256']==audit['prefill_source_sha256']
    assert '#define DPCT_PROFILING_ENABLED' in actual_source.read_text()
    record.update(poll_build_receipt_sha256=digest(poll_path),poll_source_review_sha256=digest(audit_path),
                  actual_prefill_object=str(old/'prefill.cpp.o'),actual_prefill_object_sha256=digest(old/'prefill.cpp.o'),
                  actual_shadow_header=str(shadow),actual_shadow_header_sha256=digest(shadow),
                  actual_dependency_file=str(dep),actual_dependency_file_sha256=digest(dep),
                  prefill_source=str(actual_source),prefill_source_sha256=digest(actual_source),
                  prefill_keeps_original_profiled_dpct_definition=True)
    # The whole actual linked dependency inventory must contain this template
    # header only in prefill, rather than assuming a text search finds every TU.
    rows=[]
    for line in subprocess.check_output(['/usr/bin/ninja','-t','commands','strata'],cwd=build,text=True,timeout=30).splitlines():
        args=shlex.split(line)
        if '-c' in args and '-o' in args and Path(args[args.index('-c')+1]).is_file():
            rows.append({'object':args[args.index('-o')+1],'source':args[args.index('-c')+1]})
    deps=subprocess.check_output(['/usr/bin/ninja','-t','deps']+[r['object'] for r in rows],cwd=build,text=True,timeout=30)
    (out/'ninja-deps.txt').write_text(deps)
    users=[];current=None
    expected_header=build.parent/'source/sycl/include/strata/host_wait.hpp'
    for line in deps.splitlines():
        if line and not line.startswith(' '):
            if line.endswith(': deps not found'):
                current=line[:-len(': deps not found')]
                row=next(r for r in rows if r['object']==current)
                assert row['source'].endswith('/src/kernels/cpu/iq_avx2.cpp')
            else:
                assert ': #deps ' in line
                current=line.split(': #deps ',1)[0]
        elif line.startswith('    ') and line.strip()==str(expected_header):
            users.append(next(r for r in rows if r['object']==current))
    assert len(rows)==116 and len(users)==1 and Path(users[0]['source'])==actual_source
    record.update(actual_linked_compile_commands=len(rows),actual_linked_host_wait_header_users=users,
                  actual_header_dependency_inventory_sha256=digest(out/'ninja-deps.txt'),
                  poll_template_definition_isolated_to_prefill_tu=True)
    control=root/'build-sycl-event-ack-no-root-prefill-20261008/libstrata_prefill.a'
    assert digest(control)==poll['original_archive_sha256']
    members=subprocess.check_output(['/usr/bin/ar','t',archive],text=True).splitlines()
    assert members==[m['member'] for m in poll['archive_members']]
    assert [m['member'] for m in poll['archive_members'] if m['replaced']]==['prefill.cpp.o']
    for member in poll['archive_members']:
        for path,key in [(control,'before_sha256'),(archive,'after_sha256')]:
            data=subprocess.check_output(['/usr/bin/ar','p',str(path),member['member']])
            assert hashlib.sha256(data).hexdigest()==member[key]
    record['prefill_archive_members']=poll['archive_members']
    private=candidate/'libstrata_prefill.a';shutil.copy2(archive,private)
    assert digest(private)==digest(archive)
    argv=list(uniform['steps'][-1]['argv']);assert uniform['steps'][-1]['label']=='link'
    assert argv.count(str(control))==1
    inputs={str((build/p).resolve()):digest((build/p).resolve()) for p in argv if p.endswith(('.a','.o')) and (build/p).is_file()}
    record['baseline_link_input_sha256']=inputs
    argv=[str(private) if p==str(control) else p for p in argv]
    binary=candidate/'strata';argv[argv.index('-o')+1]=str(binary)
    step={'label':'link','argv':argv};record['steps'].append(step);save()
    with (out/'link.stdout').open('wb') as a,(out/'link.stderr').open('wb') as b:
        proc=subprocess.Popen(argv,cwd=build,env=env,stdout=a,stderr=b,start_new_session=True)
        step['pid']=proc.pid;save()
        try:rc=proc.wait(timeout=600)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL);proc.wait();raise
    step['exit_code']=rc;assert rc==0
    assert digest(baseline)==record['baseline_binary_sha256']
    assert all(digest(p)==sha for p,sha in inputs.items())
    record.update(passed=True,baseline_inputs_unchanged=True,candidate_binary=str(binary),candidate_binary_sha256=digest(binary),
                  candidate_archive_sha256=digest(private),only_prefill_link_input_replaced=True,
                  pending=['First logged32K numerical gate against DD5','Matched quiet>=32768 full-read comparison','Complete physical256K, disk restoration, clipped tail, capacity refusal and later32K before adoption'])
except BaseException as error:
    record['error']=repr(error);raise
finally:
    record.update(active=False,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
print(json.dumps({k:record.get(k) for k in ['passed','error','elapsed_seconds','candidate_binary_sha256','only_prefill_link_input_replaced','actual_linked_host_wait_header_users']},indent=2))
