"""Verify worker masks, rejected input and actual delayed-DMA Stager lifetime."""
from pathlib import Path
import datetime,hashlib,json,os,re,shutil,subprocess,time
b=Path(__file__).parent
a=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/perf-sycl-stager-affinity-v0141-20261009')
src=a/'sycl/src/prefill/prefill.cpp'
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert digest(src)=='e14edcee574a88684de6dd4d46ccce80f975857178865b78aa05a10f966fb00f'
t=src.read_text();original=subprocess.check_output(['git','show','23268953314d12588fd3f496414a46bd426a7306:sycl/src/prefill/prefill.cpp'],cwd=a,text=True)
assert len(re.findall(r'#if defined\(__linux__\)\n.*?#endif\n',t,re.S))==2
assert re.sub(r'#if defined\(__linux__\)\n.*?#endif\n','',t,flags=re.S)==original
out=b/'stager-affinity-host-v1';out.mkdir(mode=0o700)
start=t.index('struct Stager {');end=t.index("// multi-GPU: the peer GPU's share",start)
(out/'actual-stager.hpp').write_text(t[start:end])
h=(b/'phase_stager_event_host_v1.cpp').read_text()
h=h.replace('#include <vector>','#include <vector>\n#include <future>\n#include <sstream>\n#include <pthread.h>\n#include <sched.h>\n#include <cerrno>')
old='    setenv("STRATA_STAGER_RING",argv[1],1);'
assert h.count(old)==1
h=h.replace(old,'''    setenv("STRATA_STAGER_RING",argv[1],1);
    const int main_cpu=std::stoi(std::getenv("PREF_HOST_MAIN_CPU"));
    cpu_set_t parent_mask;CPU_ZERO(&parent_mask);CPU_SET(main_cpu,&parent_mask);
    assert(pthread_setaffinity_np(pthread_self(),sizeof(parent_mask),&parent_mask)==0);
    const bool expected_failure=std::getenv("PREF_HOST_EXPECT_REJECTION")!=nullptr;
    std::vector<int> expected;
    std::string cpu_text=std::getenv("PREF_HOST_EXPECT_CPUS");std::replace(cpu_text.begin(),cpu_text.end(),',',' ');
    std::istringstream cpu_stream(cpu_text);for(int cpu;cpu_stream>>cpu;)expected.push_back(cpu);''')
old='        if(!s){s=std::make_unique<Stager>();assert(s->init(bytes,workers));}'
assert h.count(old)==1
h=h.replace(old,'''        if(!s){
            s=std::make_unique<Stager>();const bool ok=s->init(bytes,workers);
            if(expected_failure){assert(!ok);s.reset();std::puts("PASS rejected affinity safely");return 0;}
            assert(ok && !expected.empty());
            for(size_t i=0;i<s->threads.size();++i){
                cpu_set_t actual;CPU_ZERO(&actual);
                assert(pthread_getaffinity_np(s->threads[i].native_handle(),sizeof(actual),&actual)==0);
                assert(CPU_COUNT(&actual)==1 && CPU_ISSET(expected[i%expected.size()],&actual));
            }
        }''')
(out/'host-test.cpp').write_text(h)
allowed=sorted(os.sched_getaffinity(0));assert len(allowed)>=4
main=allowed[0];cpus=allowed[1:4];spread=','.join(map(str,cpus))
r={'active':True,'passed':False,'gpu_executed':False,'source_sha256':digest(src),
   'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
   'default_guard_removed_entire_original_prefill_source_byte_identical':True,
   'scope':'Actual Stager source: inherit single host CPU with opt-in off; distribute workers with opt-in on; invalid/nonexistent CPUs reject; delayed CPU DMA reuse across128 generations. CPU only; no SYCL/runtime arithmetic proof.',
   'harness_sha256':digest(out/'host-test.cpp'),'main_cpu':main,'spread_cpus':cpus,'tests':[]}
def save():(out/'record.json').write_text(json.dumps(r,indent=2)+'\n')
save()
cmd=[shutil.which('g++'),'-std=c++20','-O1','-g','-pthread','-fsanitize=address,undefined','-fno-omit-frame-pointer','-I'+str(a/'sycl/include'),'-I'+str(out),str(out/'host-test.cpp'),'-o',str(out/'host-test')]
r['compile']=cmd;save()
try:
    with (out/'build.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=60)
    for name,value,reject in [('default',None,False),('spread',spread,False),('empty','',True),('negative','-1',True),('trailing-comma','1,',True),('nonnumeric','x',True),('oversized','1024',True),('offline-cpu','1023',True)]:
        env=dict(os.environ,ASAN_OPTIONS='detect_leaks=1:abort_on_error=1',UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1',PREF_HOST_MAIN_CPU=str(main),PREF_HOST_EXPECT_CPUS=spread if value==spread else str(main))
        env.pop('STRATA_STAGER_CPU_LIST',None);env.pop('PREF_HOST_EXPECT_REJECTION',None)
        if value is not None:env['STRATA_STAGER_CPU_LIST']=value
        if reject:env['PREF_HOST_EXPECT_REJECTION']='1'
        t0=time.monotonic();v=subprocess.run([str(out/'host-test'),'16','3','128'],capture_output=True,text=True,env=env,timeout=30)
        (out/(name+'.stdout')).write_text(v.stdout);(out/(name+'.stderr')).write_text(v.stderr)
        r['tests'].append({'name':name,'setting':value,'expected_rejection':reject,'exit_code':v.returncode,'stdout':v.stdout,'elapsed_seconds':time.monotonic()-t0});save()
        assert v.returncode==0 and v.stdout.startswith('PASS '),(name,v.returncode,v.stderr)
    r['passed']=True
except BaseException as e:r['error']=repr(e);raise
finally:r.update(active=False,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
print(json.dumps({'passed':r['passed'],'tests':r['tests'],'receipt':str(out/'record.json'),'sha256':digest(out/'record.json')}))
