"""CPU-prepare a private prefill polling candidate; do not run a GPU job."""
from pathlib import Path
import datetime,difflib,hashlib,json,os,shlex,shutil,signal,subprocess,time
base=Path(__file__).parent
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
compiled=root/'build-sycl-event-ack-registered-copy-v3-20261008/source'
build=root/'build-sycl-event-ack-registered-copy-v3-20261008/build'
original=compiled/'sycl/include/strata/host_wait.hpp'
prefill_source=compiled/'sycl/src/prefill/prefill.cpp'
candidate=root/'build-sycl-prefill-poll-backoff-v1-20261008';candidate.mkdir()
(candidate/'include/strata').mkdir(parents=True);header=candidate/'include/strata/host_wait.hpp'
out=base/'prefill-poll-backoff-v01402-build-v1';out.mkdir(mode=0o700)
def digest(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
text=original.read_text()
old='        std::this_thread::yield();'
new='''        // Keep the first readiness checks responsive, then stop driving an
        // event-status query on every scheduler yield while a DMA is pending.
        // This changes only host polling cadence; readiness still requires the
        // same operation's completion, and cancellation never frees its buffer.
        if (polls <= 32) std::this_thread::yield();
        else std::this_thread::sleep_for(std::chrono::microseconds(10));'''
assert text.count(old)==1
header.write_text(text.replace(old,new))
(out/'candidate.diff').write_text(''.join(difflib.unified_diff(text.splitlines(True),header.read_text().splitlines(True),fromfile=str(original),tofile=str(header))))
env=dict(os.environ,PATH='/usr/bin:/bin:/opt/intel/oneapi/compiler/2026.1/bin',MKLROOT='/opt/intel/oneapi/mkl/2026.1',LIBRARY_PATH='/opt/intel/oneapi/mkl/2026.1/lib:/opt/intel/oneapi/compiler/2026.1/lib:/usr/lib/x86_64-linux-gnu')
env.pop('LD_LIBRARY_PATH',None);env.pop('LD_PRELOAD',None)
record={'active':True,'passed':False,'gpu_tested':False,'adopted':False,'steps':[],
        'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'controller_sha256':digest(__file__),'original_header_sha256':digest(original),'candidate_header_sha256':digest(header),
        'prefill_source_sha256':digest(prefill_source),
        'scope':'Only prefill TU uses a shadow host_wait.hpp:32 failed polls yield, later failed polls sleep10us. Same completion queries, generation/ownership acquire/release, cancellation, five-minute budget and timeout _Exit. No GPU model/profiler is run. Not a measured CPU/GPU bottleneck or speed claim.'}
started=time.monotonic()
def save():
    record['elapsed_seconds']=time.monotonic()-started;(out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
def run(label,argv,timeout=600):
    step={'label':label,'argv':list(map(str,argv))};record['steps'].append(step);save();begin=time.monotonic()
    with (out/(label+'.stdout')).open('wb') as a,(out/(label+'.stderr')).open('wb') as b:
        p=subprocess.Popen(step['argv'],cwd=build,env=env,stdout=a,stderr=b,start_new_session=True);step['pid']=p.pid;save()
        try:rc=p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait();raise
    step.update(exit_code=rc,elapsed_seconds=time.monotonic()-begin);save();assert rc==0,label
save()
try:
    cpu=out/'poll-check.cpp'
    cpu.write_text('''#include "strata/host_wait.hpp"
#include <atomic>
#include <cassert>
#include <fstream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
int main(int argc, char** argv) {
    assert(argc==2);
    unsigned calls=0;
    assert(strata::wait_host_ready([&]{++calls;return true;},[]{return false;},"ready"));
    assert(calls==1);
    calls=0;
    assert(!strata::wait_host_ready([&]{++calls;return true;},[]{return true;},"cancel"));
    assert(calls==0); // cancellation takes priority over a fresh readiness query
    const auto start=std::chrono::steady_clock::now();
    calls=0;
    assert(strata::wait_host_ready([&]{++calls;return std::chrono::steady_clock::now()-start>=std::chrono::milliseconds(20);},
        []{return false;},"pending",std::chrono::seconds(1)));
    std::printf("pending readiness queries=%u elapsed_us=%lld\\n",calls,
        (long long)std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-start).count());
    std::atomic<bool> stop{false};
    std::thread cancel([&]{std::this_thread::sleep_for(std::chrono::milliseconds(5));stop.store(true,std::memory_order_release);});
    assert(!strata::wait_host_ready([]{return false;},[&]{return stop.load(std::memory_order_acquire);},"cancel pending",std::chrono::seconds(1)));
    cancel.join();
    const pid_t child=fork();assert(child>=0);
    if(child==0) {
        struct Guard { const char* path; ~Guard(){std::ofstream(path)<<"unsafe cleanup";} } guard{argv[1]};
        strata::wait_host_ready([]{return false;},[]{return false;},"timeout pending",std::chrono::milliseconds(20));
        return 9;
    }
    int status=0;assert(waitpid(child,&status,0)==child);
    assert(WIFEXITED(status) && WEXITSTATUS(status)==1);
    assert(access(argv[1],F_OK)!=0); // timeout must not destroy possibly-active GPU buffers
}\n''')
    for label,include in [('control',original.parents[1]),('candidate',candidate/'include')]:
        exe=out/(label+'-poll-check')
        run(label+'-cpu-compile',['/usr/bin/c++','-std=c++20','-O2','-fsanitize=address,undefined','-pthread','-I'+str(include),str(cpu),'-o',str(exe)])
        run(label+'-cpu-run',[str(exe),str(out/(label+'-unsafe-cleanup-marker'))],timeout=10)
    record['cpu_ownership_deadline_cancellation_checks_passed']=True
    prior_path=base/'visible-commit-v01402-build-v1/record.json';prior=json.loads(prior_path.read_text())
    assert prior['passed'] and not prior['active'] and prior['baseline_inputs_unchanged']
    record['baseline_binary_sha256']=digest(prior['candidate_binary']);assert record['baseline_binary_sha256']==prior['candidate_binary_sha256']=='fdca351f73bd2433d45d9db7962fde7ad1a4b83f5e254751d27782fe9f5f75c6'
    record['baseline_build_receipt_sha256']=digest(prior_path)
    old_archive=root/'build-sycl-event-ack-no-root-prefill-20261008/libstrata_prefill.a'
    nr=json.loads((base/'event-ack-no-root-prefill-v01402-build/record.json').read_text());assert digest(old_archive)==nr['candidate_archive_sha256']
    record['original_archive_sha256']=digest(old_archive)
    commands=subprocess.check_output(['/usr/bin/ninja','-t','commands','strata'],cwd=build,env=env,text=True,timeout=30).splitlines()
    compiles=[shlex.split(x) for x in commands if ' -c ' in x and Path(shlex.split(x)[-1]).resolve()==prefill_source.resolve()]
    assert len(compiles)==1
    argv=compiles[0];record['original_compile_argv']=argv.copy();obj=candidate/'prefill.cpp.o'
    for flag,value in [('-o',obj),('-MT',obj),('-MF',candidate/'prefill.cpp.o.d')]:
        assert argv.count(flag)==1;argv[argv.index(flag)+1]=str(value)
    argv.insert(1,'-I'+str(candidate/'include'));run('compile',argv)
    private=candidate/'libstrata_prefill.a';shutil.copy2(old_archive,private)
    members=subprocess.check_output(['/usr/bin/ar','t',str(old_archive)],text=True).splitlines();assert members.count(obj.name)==1 and len(members)==len(set(members))
    run('archive-replace',['/usr/bin/ar','r',str(private),str(obj)]);run('archive-index',['/usr/bin/ranlib',str(private)])
    record['archive_members']=[]
    for name in members:
        a=subprocess.check_output(['/usr/bin/ar','p',str(old_archive),name]);b=subprocess.check_output(['/usr/bin/ar','p',str(private),name])
        assert b==obj.read_bytes() if name==obj.name else a==b
        record['archive_members'].append({'member':name,'replaced':name==obj.name,'before_sha256':hashlib.sha256(a).hexdigest(),'after_sha256':hashlib.sha256(b).hexdigest()})
    argv=list(prior['steps'][-1]['argv']);assert prior['steps'][-1]['label']=='link' and argv.count(str(old_archive))==1
    argv=[str(private) if value==str(old_archive) else value for value in argv]
    inputs={str((build/p).resolve()):digest((build/p).resolve()) for p in argv if p.endswith(('.a','.o')) and (build/p).is_file()}
    record['link_input_sha256']=inputs
    binary=candidate/'strata';argv[argv.index('-o')+1]=str(binary);run('link',argv)
    assert digest(original)==record['original_header_sha256'] and digest(prefill_source)==record['prefill_source_sha256']
    assert digest(prior['candidate_binary'])==record['baseline_binary_sha256'] and all(digest(p)==sha for p,sha in inputs.items())
    record.update(passed=True,baseline_inputs_unchanged=True,candidate_binary=str(binary),candidate_binary_sha256=digest(binary),candidate_archive_sha256=digest(private))
except BaseException as e:record['error']=repr(e);raise
finally:record.update(active=False,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
print(json.dumps({k:record.get(k) for k in ['passed','elapsed_seconds','error','candidate_binary_sha256','cpu_ownership_deadline_cancellation_checks_passed']},indent=2))
