"""Compile one host fixture, then link it with each exact kernel object.

Offline only. A successful build is not a GPU correctness or speed result.
"""
from pathlib import Path
import datetime
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import time

base = Path(__file__).parent
root = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
build = root / 'build-sycl-upstream-jit'
out = base / 'dequant-no-root-probe-build'
out.mkdir(mode=0o700)
source = base / 'dequant_actual_wrapper_probe.cpp'
original_object = build / 'CMakeFiles/strata_kernels.dir/src/kernels/cuda/iq_kernels.dp.cpp.o'
candidate_object = base / 'dequant-no-root-build/iq_kernels.dp.cpp.o'
probe_object = out / 'probe.cpp.o'
env = dict(os.environ, PATH='/usr/bin:/bin:/opt/intel/oneapi/compiler/2026.1/bin',
           LIBRARY_PATH='/opt/intel/oneapi/compiler/2026.1/lib:/usr/lib/x86_64-linux-gnu')
env.pop('LD_LIBRARY_PATH', None)
env.pop('LD_PRELOAD', None)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


record = {'scope': 'Offline construction of actual-wrapper FP16-byte/guard fixtures: shared host object, original versus property-only kernel object. No GPU execution or parity result.',
          'passed': False, 'gpu_tested': False,
          'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'controller_sha256': digest(Path(__file__)), 'source_sha256': digest(source),
          'original_kernel_object_sha256': digest(original_object),
          'candidate_kernel_object_sha256': digest(candidate_object), 'steps': []}
start = time.monotonic()


def run(label, argv):
    before = time.monotonic()
    item = {'label': label, 'argv': argv}
    record['steps'].append(item)
    with (out / (label + '.stdout')).open('wb') as stdout, (out / (label + '.stderr')).open('wb') as stderr:
        done = subprocess.run(argv, cwd=build, env=env, stdout=stdout, stderr=stderr, timeout=180)
    item.update(exit_code=done.returncode, elapsed_seconds=time.monotonic() - before)
    assert done.returncode == 0, label + ' failed; inspect saved stderr'


try:
    receipt = json.loads((base / 'dequant-no-root-build/record.json').read_text())
    assert receipt['passed'] and receipt['production_inputs_unchanged']
    assert digest(original_object) == receipt['production_object_sha256']
    assert digest(candidate_object) == receipt['candidate_object_sha256']
    shutil.copyfile(source, out / source.name)
    argv = receipt['original_compile_argv'].copy()
    for flag, value in [('-o', probe_object), ('-MT', probe_object),
                        ('-MF', out / 'probe.cpp.o.d'), ('-c', source)]:
        assert argv.count(flag) == 1
        argv[argv.index(flag) + 1] = str(value)
    run('compile', argv)
    record['probe_object_sha256'] = digest(probe_object)
    compiler = argv[0]
    links = {}
    for label, kernel in [('original', original_object), ('candidate', candidate_object)]:
        binary = out / ('dequant-' + label)
        run('link-' + label, [compiler, '-O3', '-DNDEBUG', '-fsycl',
                             '-Xsycl-target-backend=spir64', '-cl-fp32-correctly-rounded-divide-sqrt',
                             '-fsycl-device-code-split=per_kernel', str(probe_object), str(kernel), '-o', str(binary)])
        links[label] = {'path': str(binary), 'sha256': digest(binary), 'kernel_object_sha256': digest(kernel),
                        'probe_object_sha256': digest(probe_object)}
    assert digest(original_object) == record['original_kernel_object_sha256']
    assert digest(candidate_object) == record['candidate_kernel_object_sha256']
    assert digest(source) == record['source_sha256']
    record.update(binaries=links, inputs_unchanged=True, passed=True)
except BaseException as error:
    record['error'] = repr(error)
    raise
finally:
    record['elapsed_seconds'] = time.monotonic() - start
    record['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (out / 'record.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({k: v for k, v in record.items() if k != 'steps'}, indent=2))
