"""Offline full private rebuild: actual DMA acknowledgements and registered copy queue."""
from pathlib import Path
import datetime, difflib, hashlib, json, os, re, shutil, signal, subprocess, tarfile, time

base = Path(__file__).parent
root = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
candidate = root/'build-sycl-event-ack-registered-copy-v2-20261008'
source_root = candidate/'source'
build = candidate/'build'
out = base/'event-ack-registered-copy-v01402-v2-build'
out.mkdir(mode=0o700)
candidate.mkdir()

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

prod_paths = [root/'sycl/src/prefill/prefill.cpp', root/'sycl/include/dpct/device.hpp',
              root/'build-sycl-e8ca-refresh-20261007/strata']
env = dict(os.environ, PATH='/usr/bin:/bin:/opt/intel/oneapi/compiler/2026.1/bin',
           MKLROOT='/opt/intel/oneapi/mkl/2026.1',
           LIBRARY_PATH='/opt/intel/oneapi/mkl/2026.1/lib:/opt/intel/oneapi/compiler/2026.1/lib:/usr/lib/x86_64-linux-gnu')
env.pop('LD_LIBRARY_PATH', None)
env.pop('LD_PRELOAD', None)
record = {'active':True, 'passed':False, 'gpu_tested':False, 'adopted':False,
          'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'controller_sha256':digest(Path(__file__)), 'steps':[],
          'production_inputs':{str(path):digest(path) for path in prod_paths},
          'scope':'Full private source rebuild. Retain the actual DMA event until source reuse, and make an optional nonprofiling in-order copy queue through device_ext so its global waits still cover this queue. Default queue properties and arithmetic unchanged. Every translation unit is rebuilt with the same modified class definition; no mixed-header ODR experiment. CPU completion tests were already passed with this identical header; no GPU or timing claim.'}
started = time.monotonic()

def save():
    record['elapsed_seconds'] = time.monotonic()-started
    (out/'record.json').write_text(json.dumps(record,indent=2)+'\n')

def run(label, argv, timeout=900):
    step = {'label':label, 'argv':list(map(str,argv))}
    record['steps'].append(step)
    save()
    before = time.monotonic()
    with (out/(label+'.stdout')).open('wb') as stdout, (out/(label+'.stderr')).open('wb') as stderr:
        process = subprocess.Popen(step['argv'], cwd=root, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
        step['pid'] = process.pid
        save()
        try:
            rc = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL); process.wait()
            raise
    step.update(exit_code=rc,elapsed_seconds=time.monotonic()-before)
    save()
    assert rc==0, label+' failed; inspect saved output'

save()
try:
    assert record['production_inputs'][str(prod_paths[2])]=='c88f94d81bfb22227310ea00d09ecc8ab21a6e670aee9556307746530f5af714'
    revision = subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    record['source_revision'] = revision
    archive = candidate/'source.tar'
    with archive.open('wb') as stream:
        subprocess.run(['git','archive',revision,'sycl','src','include','third_party'],cwd=root,stdout=stream,check=True,timeout=60)
    record['archive_sha256'] = digest(archive)
    source_root.mkdir()
    with tarfile.open(archive) as tar:
        tar.extractall(source_root,filter='data')
    original_files = {str(path.relative_to(source_root)):digest(path) for path in source_root.rglob('*') if path.is_file()}
    (out/'original-source-hashes.json').write_text(json.dumps(original_files,indent=2)+'\n')
    copied_prefill = source_root/'sycl/src/prefill/prefill.cpp'
    copied_header = source_root/'sycl/include/dpct/device.hpp'
    original = copied_prefill.read_text()
    replacement = (root/'build-sycl-event-ack-v2-20261008/source/prefill.cpp').read_text()
    # Private single-object prototype normalized relative includes. Restore their
    # original spelling so this complete private tree resolves all of its sources.
    for relative in re.findall(r'#include "(\.\./[^"]+)"', original):
        absolute = str((prod_paths[0].parent/relative).resolve())
        token = '#include "'+absolute+'"'
        assert replacement.count(token)==1
        replacement = replacement.replace(token,'#include "'+relative+'"')
    assert replacement.count('#include "event_completion.hpp"')==1
    replacement = replacement.replace('#include "event_completion.hpp"','#include "strata/event_completion.hpp"')
    old = 'm.copy = dpct::get_current_device().create_in_order_queue(true)'
    new = 'm.copy = dpct::get_current_device().create_in_order_queue_with_profiling(\n                true, std::getenv("STRATA_PREFILL_TRANSFER_TIMING") != nullptr)'
    assert replacement.count(old)==1
    replacement = replacement.replace(old,new)
    copied_prefill.write_text(replacement)
    completion = root/'build-sycl-event-ack-v2-20261008/source/event_completion.hpp'
    shutil.copyfile(completion,source_root/'sycl/include/strata/event_completion.hpp')
    record['completion_header_sha256'] = digest(completion)
    prior = json.loads((base/'event-ack-v01402-v2-build/record.json').read_text())
    assert prior['passed'] and prior['completion_header_sha256']==record['completion_header_sha256']
    record['identical_cpu_test_receipt_sha256'] = digest(base/'event-ack-v01402-v2-build/record.json')
    header_original = copied_header.read_text()
    needle = '  sycl::queue *create_out_of_order_queue(bool enable_exception_handler = false) {'
    assert header_original.count(needle)==1
    factory = '''  // Register the queue so default-stream/device-wide waits include its work.
  // Existing factories retain their original profiling properties.
  sycl::queue *create_in_order_queue_with_profiling(bool enable_exception_handler,
                                                  bool enable_profiling) {
    std::lock_guard<mutex_type> lock(m_mutex);
    sycl::async_handler eh = {};
    if (enable_exception_handler) eh = exception_handler;
    const auto properties = enable_profiling
        ? sycl::property_list{sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()}
        : sycl::property_list{sycl::property::queue::in_order()};
    _queues.push_back(std::make_shared<sycl::queue>(_ctx, *this, eh, properties));
    return _queues.back().get();
  }

'''
    copied_header.write_text(header_original.replace(needle,factory+needle))
    for name, before, path in [('prefill',original,copied_prefill),('device',header_original,copied_header)]:
        (out/(name+'.diff')).write_text(''.join(difflib.unified_diff(before.splitlines(True),path.read_text().splitlines(True),fromfile=name+'.original',tofile=name+'.candidate')))
    record['candidate_sources'] = {str(path.relative_to(source_root)):digest(path) for path in [copied_prefill,copied_header,source_root/'sycl/include/strata/event_completion.hpp']}
    changed = [name for name, value in original_files.items() if digest(source_root/name)!=value]
    assert changed==sorted(['sycl/include/dpct/device.hpp','sycl/src/prefill/prefill.cpp']) or set(changed)=={'sycl/include/dpct/device.hpp','sycl/src/prefill/prefill.cpp'}
    record['changed_tracked_sources'] = changed
    run('configure',['/usr/bin/cmake','-S',str(source_root/'sycl'),'-B',str(build),'-G','Ninja',
        '-DCMAKE_CXX_COMPILER=/opt/intel/oneapi/compiler/2026.1/bin/icpx',
        '-DCMAKE_BUILD_TYPE=Release','-DSTRATA_SYCL_AOT=',
        '-DSTRATA_GGML_DIR=/home/yayoi/orca/workspaces/Strata/ggml-sycl-pinned',
        '-DSTRATA_SYCL_PARITY=ON','-DSTRATA_NATIVE_POOL_TASK_FACTOR=0'],timeout=120)
    run('build',['/usr/bin/cmake','--build',str(build),'--target','strata','--parallel','4'],timeout=900)
    commands = subprocess.check_output(['/usr/bin/ninja','-C',str(build),'-t','commands','strata'],env=env,text=True,timeout=20)
    (out/'build-commands.txt').write_text(commands)
    assert all(digest(Path(path))==value for path,value in record['production_inputs'].items())
    assert digest(copied_header)==record['candidate_sources']['sycl/include/dpct/device.hpp']
    record.update(passed=True,production_inputs_unchanged=True,candidate_binary=str(build/'strata'),candidate_binary_sha256=digest(build/'strata'))
except BaseException as error:
    record['error'] = repr(error)
    raise
finally:
    record.update(active=False,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save()
print(json.dumps({k:record.get(k) for k in ['passed','error','elapsed_seconds','candidate_binary_sha256','production_inputs_unchanged']},indent=2))
