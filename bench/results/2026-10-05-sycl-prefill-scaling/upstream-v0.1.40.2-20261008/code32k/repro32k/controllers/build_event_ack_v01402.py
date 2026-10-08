"""Offline private DMA-event acknowledgement candidate; no GPU submissions."""
from pathlib import Path
import datetime, difflib, hashlib, json, os, re, shlex, shutil, subprocess, time
base = Path(__file__).parent
root = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
build = root/'build-sycl-e8ca-refresh-20261007'
candidate_dir = root/'build-sycl-event-ack-20261008'
out = base/'event-ack-v01402-build'; out.mkdir(mode=0o700)
source = root/'sycl/src/prefill/prefill.cpp'
candidate_source = candidate_dir/'source/prefill.cpp'
candidate_object = candidate_dir/'prefill.cpp.o'
candidate_archive = candidate_dir/'libstrata_prefill.a'
candidate_binary = candidate_dir/'strata'
original_object = build/'CMakeFiles/strata_prefill.dir/src/prefill/prefill.cpp.o'
original_archive = build/'libstrata_prefill.a'
def digest(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()
env = dict(os.environ, PATH='/usr/bin:/bin:/opt/intel/oneapi/compiler/2026.1/bin',
           MKLROOT='/opt/intel/oneapi/mkl/2026.1',
           LIBRARY_PATH='/opt/intel/oneapi/mkl/2026.1/lib:/opt/intel/oneapi/compiler/2026.1/lib:/usr/lib/x86_64-linux-gnu')
env.pop('LD_LIBRARY_PATH',None); env.pop('LD_PRELOAD',None)
record = {'active':True,'passed':False,'gpu_tested':False,'adopted':False,
          'scope':'Replace only Stager and PLE host_task DMA acknowledgements with the actual memcpy events. All GPU arithmetic, ranges, graph code and queue barriers unchanged. Private build; original sources/objects/link inputs/binary must remain unchanged. CPU ring ownership tests are not a GPU correctness or hang prevention proof.',
          'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'controller_sha256':digest(Path(__file__)),'steps':[],
          'production_source_sha256':digest(source),'production_object_sha256':digest(original_object),
          'production_archive_sha256':digest(original_archive),'production_binary_sha256':digest(build/'strata')}
started = time.monotonic()
def save(): (out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
def run(label,argv,timeout=240):
    step={'label':label,'argv':list(map(str,argv))}; record['steps'].append(step); save()
    before=time.monotonic()
    with (out/(label+'.stdout')).open('wb') as stdout, (out/(label+'.stderr')).open('wb') as stderr:
        result=subprocess.run(step['argv'],cwd=build,env=env,stdout=stdout,stderr=stderr,timeout=timeout)
    step.update(exit_code=result.returncode,elapsed_seconds=time.monotonic()-before);save()
    assert result.returncode==0, label+' failed: inspect saved output'
try:
    assert record['production_binary_sha256']=='c88f94d81bfb22227310ea00d09ecc8ab21a6e670aee9556307746530f5af714'
    assert not candidate_source.exists() and not candidate_binary.exists()
    original=source.read_text(); candidate=original
    include='#include "strata/host_completion.hpp"'
    assert candidate.count(include)==1
    query='''#include "event_completion.hpp"
namespace strata {
inline bool dma_completed(const EventCompletion<sycl::event>& state, uint64_t value) {
    return event_completed(state, value, [](const sycl::event& event) {
        return event.get_info<sycl::info::event::command_execution_status>() ==
               sycl::info::event_command_status::complete;
    });
}
} // namespace strata'''
    candidate=candidate.replace(include,query)
    assert candidate.count('strata::HostCompletion')==2
    assert candidate.count('strata::make_host_completion()')==2
    assert candidate.count('strata::host_completed(')==2
    candidate=candidate.replace('strata::HostCompletion','strata::EventCompletion<sycl::event>')
    candidate=candidate.replace('strata::make_host_completion()','strata::make_event_completion<sycl::event>()')
    candidate=candidate.replace('strata::host_completed(','strata::dma_completed(')
    old='''    // The copy queue's host task acknowledges its preceding DMA. Worker polls
    // access CPU atomics only; no GPU-written volatile marker is assumed safe.
    void issued_one(int j, dpct::queue_ptr copy) {'''
    new='''    // Retain this DMA's event and check its actual completion before reuse.
    // No CPU host task is inserted into the copy queue's dependency chain.
    void issued_one(int j, const sycl::event& dma) {'''
    assert candidate.count(old)==1; candidate=candidate.replace(old,new)
    old='strata::enqueue_host_completion(*copy, done_seq[(size_t) (j % kRing)], s);'
    new='strata::record_event_completion(done_seq[(size_t) (j % kRing)], s, dma);'
    assert candidate.count(old)==1;candidate=candidate.replace(old,new)
    pattern=r'(?P<indent> +)transfers\.copy\((?P<args>[^;]+)\);\n(?P=indent)m\.stager->issued_one\((?P<job>[^,]+), m\.copy\);'
    def pair(match):
        i=match['indent'];return i+'const auto dma = transfers.copy('+match['args']+');\n'+i+'m.stager->issued_one('+match['job']+', dma);'
    candidate,pairs=re.subn(pattern,pair,candidate); assert pairs==4, pairs
    start=candidate.index('            if (DPCT_CHECK_ERROR(m.cs->memcpy(m.ple_emb,')
    end=candidate.index(' != 0) {',candidate.index('strata::enqueue_host_completion(*m.cs,',start))+len(' != 0) {')
    replacement='''            if (DPCT_CHECK_ERROR([&] {
                    const auto dma = m.cs->memcpy(m.ple_emb, m.ple_emb_host[ple_buf], (size_t)T * N * 4);
                    m.ple_want[ple_buf] = ++m.ple_seq;
                    strata::record_event_completion(m.ple_done_seq[ple_buf], m.ple_want[ple_buf], dma);
                }()) != 0) {'''
    candidate=candidate[:start]+replacement+candidate[end:]
    assert 'enqueue_host_completion' not in candidate and 'HostCompletion' not in candidate
    (candidate_dir/'source/prefill.original.cpp').write_text(original)
    candidate_source.write_text(candidate)
    (out/'source.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),candidate.splitlines(True),fromfile='prefill.original.cpp',tofile='prefill.cpp')))
    record['changed_dma_pairs']=pairs;record['candidate_source_sha256']=digest(candidate_source)
    record['completion_header_sha256']=digest(candidate_dir/'source/event_completion.hpp')
    test_source=candidate_dir/'source/event_completion_test.cpp'
    for name,flags in [('cpu-test',['-O2']),('cpu-test-sanitized',['-O1','-g','-fsanitize=address,undefined','-fno-omit-frame-pointer'])]:
        test_binary=candidate_dir/name
        run(name+'-build',['/usr/bin/g++','-std=c++17','-pthread',*flags,str(test_source),'-o',str(test_binary)])
        run(name,[str(test_binary)],timeout=30)
    commands=subprocess.check_output(['/usr/bin/ninja','-t','commands','strata'],cwd=build,env=env,text=True,timeout=15).splitlines()
    compiles=[shlex.split(line) for line in commands if line.endswith(' -c '+str(source))]
    assert len(compiles)==1;argv=compiles[0];record['original_compile_argv']=argv.copy()
    for flag,value in [('-o',candidate_object),('-MT',candidate_object),('-MF',candidate_dir/'prefill.cpp.o.d'),('-c',candidate_source)]:
        assert argv.count(flag)==1;argv[argv.index(flag)+1]=str(value)
    run('compile',argv)
    shutil.copyfile(original_archive,candidate_archive)
    names=subprocess.check_output(['/usr/bin/ar','t',str(original_archive)],text=True).splitlines()
    assert names.count(original_object.name)==1 and len(names)==len(set(names))
    run('archive-replace',['/usr/bin/ar','r',str(candidate_archive),str(candidate_object)])
    run('archive-index',['/usr/bin/ranlib',str(candidate_archive)])
    assert subprocess.check_output(['/usr/bin/ar','t',str(candidate_archive)],text=True).splitlines()==names
    record['archive_members']=[]
    for name in names:
        before=subprocess.check_output(['/usr/bin/ar','p',str(original_archive),name])
        after=subprocess.check_output(['/usr/bin/ar','p',str(candidate_archive),name])
        assert after==candidate_object.read_bytes() if name==original_object.name else before==after
        record['archive_members'].append({'member':name,'replaced':name==original_object.name,'original_sha256':hashlib.sha256(before).hexdigest(),'candidate_sha256':hashlib.sha256(after).hexdigest()})
    links=[shlex.split(line) for line in commands if ' -o strata ' in line];assert len(links)==1
    tokens=links[0];assert tokens[:2]==[':','&&'] and tokens[-2:]==['&&',':'];argv=tokens[2:-2]
    record['original_link_argv']=argv.copy()
    inputs={str(build/token):digest(build/token) for token in argv if token.endswith(('.o','.a')) and (build/token).is_file()}
    record['link_input_sha256']=inputs
    assert argv.count('libstrata_prefill.a')==1
    argv=[str(candidate_archive) if token=='libstrata_prefill.a' else token for token in argv]
    argv[argv.index('-o')+1]=str(candidate_binary);run('link',argv)
    for name,path in [('source',source),('object',original_object),('archive',original_archive),('binary',build/'strata')]:
        assert digest(path)==record['production_'+name+'_sha256']
    assert all(digest(Path(path))==value for path,value in inputs.items())
    record.update(passed=True,production_inputs_unchanged=True,candidate_binary=str(candidate_binary),candidate_binary_sha256=digest(candidate_binary),candidate_object_sha256=digest(candidate_object),candidate_archive_sha256=digest(candidate_archive))
except BaseException as error:
    record['error']=repr(error);raise
finally:
    record.update(active=False,elapsed_seconds=time.monotonic()-started,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
print(json.dumps({key:record.get(key) for key in ['passed','error','elapsed_seconds','candidate_binary','candidate_binary_sha256','production_inputs_unchanged']},indent=2))
