"""Complete private registered-queue build using the production configure argv."""
from pathlib import Path
import datetime, hashlib, json, os, shutil, signal, subprocess, time

base = Path(__file__).parent
root = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
previous_path = base/'event-ack-registered-copy-v01402-v3-build/record.json'
previous = json.loads(previous_path.read_text())
assert previous['passed'] and not previous['active']
candidate = root/'build-sycl-event-ack-registered-copy-v3-20261008'
build = candidate/'build'
out = base/'event-ack-registered-copy-v01402-matched-build'
out.mkdir(mode=0o700)

def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

env = dict(os.environ,PATH='/usr/bin:/bin:/opt/intel/oneapi/compiler/2026.1/bin',
           MKLROOT='/opt/intel/oneapi/mkl/2026.1',
           LIBRARY_PATH='/opt/intel/oneapi/mkl/2026.1/lib:/opt/intel/oneapi/compiler/2026.1/lib:/usr/lib/x86_64-linux-gnu')
env.pop('LD_LIBRARY_PATH',None);env.pop('LD_PRELOAD',None)
record = {key:previous[key] for key in ['source_revision','production_inputs','candidate_sources','completion_header_sha256','identical_cpu_test_receipt_sha256']}
record.update(active=True,passed=False,gpu_tested=False,adopted=False,steps=[],
              scope='Same complete private registered-copy-queue sources/header. Reconfigure with the exact original production argv, including IQ2_S GCC groups, to avoid mixing CPU code generation with the queue experiment. Preserve the prior complete-build binary before the incremental rebuild. No GPU/timing/full-context claim.',
              started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),controller_sha256=digest(Path(__file__)),
              previous_build_receipt_sha256=digest(previous_path))
started = time.monotonic()
def save():
    record['elapsed_seconds']=time.monotonic()-started
    (out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
def run(label,argv,timeout):
    step={'label':label,'argv':list(map(str,argv))};record['steps'].append(step);save();before=time.monotonic()
    with (out/(label+'.stdout')).open('wb') as stdout,(out/(label+'.stderr')).open('wb') as stderr:
        process=subprocess.Popen(step['argv'],cwd=root,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
        step['pid']=process.pid;save()
        try:rc=process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL);process.wait();raise
    step.update(exit_code=rc,elapsed_seconds=time.monotonic()-before);save();assert rc==0,label+' failed'
def cache(path):
    result={}
    for line in path.read_text().splitlines():
        if not line or line.startswith(('#','//')) or '=' not in line:continue
        key,value=line.split('=',1);name,kind=key.split(':',1);result[name]=(kind,value)
    return result

save()
try:
    binary=build/'strata';assert digest(binary)==previous['candidate_binary_sha256']
    snapshot=out/'strata-before-config-match';shutil.copy2(binary,snapshot)
    record['previous_binary_snapshot']={'file':str(snapshot),'sha256':digest(snapshot),'bytes':snapshot.stat().st_size}
    assert all(digest(candidate/'source'/path)==value for path,value in record['candidate_sources'].items())
    production=json.loads((base/'upstream-e8ca-refresh-20261007/build-record.json').read_text())
    argv=next(step['argv'].copy() for step in production['steps'] if step['label']=='configure')
    argv[argv.index('-S')+1]=str(candidate/'source/sycl');argv[argv.index('-B')+1]=str(build)
    assert '-DSTRATA_IQ2S_GCC=ON' in argv and '-DCMAKE_C_COMPILER=/opt/intel/oneapi/compiler/2026.1/bin/icx' in argv
    run('configure',argv,120)
    run('build',['/usr/bin/cmake','--build',str(build),'--target','strata','--parallel','4'],600)
    original=cache(root/'build-sycl-e8ca-refresh-20261007/CMakeCache.txt');current=cache(build/'CMakeCache.txt')
    selected=[key for key,(kind,value) in original.items() if (key.startswith(('GGML_','STRATA_')) and kind in ['BOOL','STRING']) or key in ['CMAKE_C_COMPILER','CMAKE_CXX_COMPILER','CMAKE_BUILD_TYPE','CMAKE_C_FLAGS','CMAKE_CXX_FLAGS','STRATA_GGML_DIR','STRATA_IQ_FIXTURE_DIR','STRATA_PLE_FIXTURE_DIR']]
    differences={key:{'production':original[key][1],'candidate':current.get(key,(None,None))[1]} for key in selected if original[key][1]!=current.get(key,(None,None))[1]}
    record['selected_configuration_differences']=differences;record['compared_configuration_keys']=sorted(selected)
    assert not differences,'private build does not match production configuration'
    record['production_inputs_unchanged']=all(digest(Path(path))==value for path,value in record['production_inputs'].items())
    assert record['production_inputs_unchanged']
    commands=subprocess.check_output(['/usr/bin/ninja','-C',str(build),'-t','commands','strata'],env=env,text=True,timeout=20)
    (out/'build-commands.txt').write_text(commands)
    record.update(passed=True,candidate_binary=str(binary),candidate_binary_sha256=digest(binary))
except BaseException as error:
    record['error']=repr(error);raise
finally:
    record.update(active=False,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
print(json.dumps({key:record.get(key) for key in ['passed','error','elapsed_seconds','candidate_binary_sha256','selected_configuration_differences','production_inputs_unchanged']},indent=2))
