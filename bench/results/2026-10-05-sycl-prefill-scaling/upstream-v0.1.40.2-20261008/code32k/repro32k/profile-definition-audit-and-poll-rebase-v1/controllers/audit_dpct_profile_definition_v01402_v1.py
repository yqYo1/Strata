"""CPU preprocessing audit: inline DPCT definitions across actual compiled TUs."""
from pathlib import Path
import datetime,difflib,hashlib,json,os,re,shlex,signal,subprocess,time
base=Path(__file__).parent
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
compiled=root/'build-sycl-event-ack-registered-copy-v3-20261008/source'
build=root/'build-sycl-event-ack-registered-copy-v3-20261008/build'
header=compiled/'sycl/include/dpct/device.hpp'
candidate=root/'build-sycl-dpct-profile-definition-v1-20261008';(candidate/'include/dpct').mkdir(parents=True)
out=base/'dpct-profile-definition-v01402-source-review-v1';out.mkdir(mode=0o700)
def digest(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
text=header.read_text()
a='''#ifdef DPCT_PROFILING_ENABLED
            sycl::property::queue::enable_profiling(),
#endif'''
b='''#ifdef DPCT_PROFILING_ENABLED
  *event_ptr = queue->ext_oneapi_submit_barrier();
#else
  *event_ptr = queue->single_task([=]() {});
#endif'''
assert text.count(a)==text.count(b)==1
changed=text.replace(a,'            sycl::property::queue::enable_profiling(),').replace(b,'  *event_ptr = queue->ext_oneapi_submit_barrier();')
new_header=candidate/'include/dpct/device.hpp';new_header.write_text(changed)
(out/'candidate.diff').write_text(''.join(difflib.unified_diff(text.splitlines(True),changed.splitlines(True),fromfile=str(header),tofile=str(new_header))))
env=dict(os.environ,PATH='/usr/bin:/bin:/opt/intel/oneapi/compiler/2026.1/bin',MKLROOT='/opt/intel/oneapi/mkl/2026.1')
env.pop('LD_LIBRARY_PATH',None);env.pop('LD_PRELOAD',None)
record=dict(active=True,passed=False,gpu_tested=False,adopted=False,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),controller_sha256=digest(__file__),original_header_sha256=digest(header),candidate_header_sha256=digest(new_header),steps=[],scope='Preprocess actual compiler commands for translated verifier and hand-written conversation state. Compare class-inline queue factory and external-inline sync_barrier token definitions with/without source-local DPCT_PROFILING_ENABLED. This is C++ definition/lifetime correctness evidence, not a runtime fault attribution. Candidate keeps the existing profiling branch uniformly and keeps explicit copy-queue property selection unchanged; no GPU execution or binary adoption.')
started=time.monotonic()
def save():
    record['elapsed_seconds']=time.monotonic()-started;(out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
def body(text,anchor):
    i=text.index(anchor);start=text.index('{',i);depth=0
    for j in range(start,len(text)):
        if text[j]=='{':depth+=1
        elif text[j]=='}':
            depth-=1
            if depth==0:return text[i:j+1]
    raise AssertionError('unterminated body')
def tokenize(text):
    return re.findall(r'[A-Za-z_][A-Za-z_0-9]*|\d+|::|->|[^\s]',text)
try:
    commands=subprocess.check_output(['/usr/bin/ninja','-t','commands','strata'],cwd=build,env=env,text=True,timeout=30).splitlines()
    results={}
    for label,source in [('translated',compiled/'sycl/src/core/verify.cpp'),('handwritten',compiled/'sycl/src/core/conversation_state.cpp')]:
        choices=[shlex.split(x) for x in commands if ' -c ' in x and Path(shlex.split(x)[-1]).resolve()==source.resolve()];assert len(choices)==1
        original=choices[0];record.setdefault('original_commands',{})[label]=original.copy()
        record.setdefault('actual_source_sha256',{})[str(source)]=digest(source)
        argv=[];skip=False
        for v in original:
            if skip:skip=False;continue
            if v in ['-o','-MF','-MT']:skip=True;continue
            if v in ['-MD','-c']:continue
            argv.append(v)
        argv.insert(1,'-E');argv.insert(2,'-P')
        results[label]={}
        for variant in ['original','candidate']:
            args=argv.copy()
            if variant=='candidate':args.insert(1,'-I'+str(candidate/'include'))
            step=dict(label=label+'-'+variant,argv=args);record['steps'].append(step);save()
            ii=out/(step['label']+'.ii');err=out/(step['label']+'.stderr')
            with ii.open('wb') as f,err.open('wb') as e:
                p=subprocess.Popen(args,cwd=build,env=env,stdout=f,stderr=e,start_new_session=True);step['pid']=p.pid;save()
                try:rc=p.wait(timeout=180)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait();raise
            step['exit_code']=rc;assert rc==0
            data=ii.read_text();excerpts={name:body(data,anchor) for name,anchor in [('queue_factory','sycl::queue *create_queue_impl(bool enable_exception_handler'),('sync_barrier','inline void sync_barrier(event_ptr event_ptr')]}
            for name,part in excerpts.items():
                (out/(step['label']+'-'+name+'.txt')).write_text(part+'\n')
            results[label][variant]={k:tokenize(v) for k,v in excerpts.items()}
            step['preprocessed_bytes']=ii.stat().st_size;step['preprocessed_sha256']=digest(ii);save()
    for name in ['queue_factory','sync_barrier']:
        assert results['translated']['original'][name]!=results['handwritten']['original'][name]
        assert results['translated']['candidate'][name]==results['handwritten']['candidate'][name]
        assert results['translated']['candidate'][name]==results['translated']['original'][name]
    assert 'create_in_order_queue_with_profiling' in changed and changed.count('const auto properties = enable_profiling')==text.count('const auto properties = enable_profiling')==1
    assert changed[changed.index('  sycl::queue *create_in_order_queue_with_profiling'):changed.index('  sycl::queue *create_out_of_order_queue')]==text[text.index('  sycl::queue *create_in_order_queue_with_profiling'):text.index('  sycl::queue *create_out_of_order_queue')]
    record.update(passed=True,original_tokens_differ=['queue_factory','sync_barrier'],candidate_tokens_identical=['queue_factory','sync_barrier'],candidate_identical_to_translated_original=True,explicit_copy_queue_property_factory_unchanged=True,pending=['Enumerate all linked TUs that instantiate these helpers','Compile compatible private replacements before any GPU use','logged32K numerical and complete256K gates before adoption','separate profiler collection/quiet32K speed evidence'])
except BaseException as e:record['error']=repr(e);raise
finally:record.update(active=False,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
print(json.dumps({k:record.get(k) for k in ['passed','error','elapsed_seconds','original_tokens_differ','candidate_tokens_identical']},indent=2))
