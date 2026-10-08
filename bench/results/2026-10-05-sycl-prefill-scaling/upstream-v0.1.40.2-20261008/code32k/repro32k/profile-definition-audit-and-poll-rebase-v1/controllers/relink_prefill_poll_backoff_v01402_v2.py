"""CPU-only rebase of the polling object onto the indexer-spare candidate."""
from pathlib import Path
import datetime,hashlib,json,os,shutil,signal,subprocess,time
base=Path(__file__).parent
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
build=root/'build-sycl-event-ack-registered-copy-v3-20261008/build'
candidate=root/'build-sycl-prefill-poll-backoff-v2-20261008';candidate.mkdir()
out=base/'prefill-poll-backoff-v01402-relink-v2';out.mkdir(mode=0o700)
def digest(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
env=dict(os.environ,PATH='/usr/bin:/bin:/opt/intel/oneapi/compiler/2026.1/bin',MKLROOT='/opt/intel/oneapi/mkl/2026.1',LIBRARY_PATH='/opt/intel/oneapi/mkl/2026.1/lib:/opt/intel/oneapi/compiler/2026.1/lib:/usr/lib/x86_64-linux-gnu')
env.pop('LD_LIBRARY_PATH',None);env.pop('LD_PRELOAD',None)
record=dict(active=True,passed=False,gpu_tested=False,adopted=False,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),controller_sha256=digest(__file__),steps=[],scope='Only replace the accepted prefill archive in the new7f054 indexer-spare candidate link. Reuse unchanged CPU-sanitizer-tested polling object/header. Indexer correctness remains a separate active full-capacity gate; no GPU job is launched, no wait cadence or profiling definition experiment is combined beyond this isolated prefill change.')
started=time.monotonic()
def save():
    record['elapsed_seconds']=time.monotonic()-started;(out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
save()
try:
    prior_path=base/'indexer-spare-commit-v01402-build-v1/record.json';prior=json.loads(prior_path.read_text());assert prior['passed'] and not prior['active'] and prior['baseline_inputs_unchanged']
    baseline=Path(prior['candidate_binary']);assert digest(baseline)==prior['candidate_binary_sha256']=='7f054f8338c6552a5ae6dba548d2020ff2e3bd2661f4a41bb5a2289c56d07891'
    record.update(baseline_binary_sha256=digest(baseline),baseline_build_receipt_sha256=digest(prior_path))
    poll_path=base/'prefill-poll-backoff-v01402-build-v1/record.json';poll=json.loads(poll_path.read_text());assert poll['passed'] and not poll['active'] and poll['cpu_ownership_deadline_cancellation_checks_passed'] and poll['baseline_inputs_unchanged']
    old=Path(poll['candidate_binary']).parent;archive=old/'libstrata_prefill.a';assert digest(archive)==poll['candidate_archive_sha256']
    audit_path=base/'prefill-poll-backoff-v01402-source-review-v1/record.json';audit=json.loads(audit_path.read_text());assert audit['passed'] and not audit['active']
    assert digest(old/'include/strata/host_wait.hpp')==poll['candidate_header_sha256']
    record.update(poll_build_receipt_sha256=digest(poll_path),poll_source_review_sha256=digest(audit_path),actual_prefill_object=str(old/'prefill.cpp.o'),actual_prefill_object_sha256=digest(old/'prefill.cpp.o'),actual_shadow_header=str(old/'include/strata/host_wait.hpp'),actual_shadow_header_sha256=poll['candidate_header_sha256'])
    private=candidate/'libstrata_prefill.a';shutil.copy2(archive,private);assert digest(private)==digest(archive)
    argv=list(prior['steps'][-1]['argv']);assert prior['steps'][-1]['label']=='link'
    control=root/'build-sycl-event-ack-no-root-prefill-20261008/libstrata_prefill.a';assert argv.count(str(control))==1
    nr=json.loads((base/'event-ack-no-root-prefill-v01402-build/record.json').read_text());assert digest(control)==nr['candidate_archive_sha256']
    inputs={str((build/p).resolve()):digest((build/p).resolve()) for p in argv if p.endswith(('.a','.o')) and (build/p).is_file()};record['baseline_link_input_sha256']=inputs
    argv=[str(private) if p==str(control) else p for p in argv];binary=candidate/'strata';argv[argv.index('-o')+1]=str(binary)
    step=dict(label='link',argv=argv);record['steps'].append(step);save()
    with (out/'link.stdout').open('wb') as a,(out/'link.stderr').open('wb') as b:
        p=subprocess.Popen(argv,cwd=build,env=env,stdout=a,stderr=b,start_new_session=True);step['pid']=p.pid;save()
        try:rc=p.wait(timeout=600)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait();raise
    step['exit_code']=rc;assert rc==0
    assert digest(baseline)==record['baseline_binary_sha256'] and all(digest(p)==sha for p,sha in inputs.items())
    record.update(passed=True,baseline_inputs_unchanged=True,candidate_binary=str(binary),candidate_binary_sha256=digest(binary),candidate_archive_sha256=digest(private),only_prefill_link_input_replaced=True,pending=['Current indexer full-capacity sequence must complete','first diagnostic32K math gate against7f054','matched quiet>=32768 full-read comparisons','full-capacity gate before adoption'])
except BaseException as e:record['error']=repr(e);raise
finally:record.update(active=False,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
print(json.dumps({k:record.get(k) for k in ['passed','elapsed_seconds','error','candidate_binary_sha256','only_prefill_link_input_replaced']},indent=2))
