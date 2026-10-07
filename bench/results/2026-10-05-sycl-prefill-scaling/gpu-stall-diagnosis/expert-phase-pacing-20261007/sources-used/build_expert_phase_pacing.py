"""Compile and link a private typed-vector candidate without a GPU submission."""
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
build = root/'build-sycl-upstream-jit'
out = base/'expert-phase-pacing-build'
out.mkdir(mode=0o700)
source = root/'sycl/src/prefill/prefill.cpp'
original_object = build/'CMakeFiles/strata_prefill.dir/src/prefill/prefill.cpp.o'
candidate_source = out/'prefill.cpp'
candidate_object = out/'prefill.cpp.o'
candidate_archive = out/'libstrata_prefill.a'
candidate_binary = base/'strata-expert-phase-pacing-candidate'
assert not candidate_binary.exists()

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

env = dict(os.environ, PATH='/usr/bin:/bin:/opt/intel/oneapi/compiler/2026.1/bin',
           MKLROOT='/opt/intel/oneapi/mkl/2026.1',
           LIBRARY_PATH='/opt/intel/oneapi/mkl/2026.1/lib:/opt/intel/oneapi/compiler/2026.1/lib:/usr/lib/x86_64-linux-gnu')
env.pop('LD_LIBRARY_PATH', None)
env.pop('LD_PRELOAD', None)
record = {'scope': 'Offline private expert-region phase pacing compilation and one-object archive replacement/relink; no GPU execution, parity, capacity, throughput or hang-cause proof',
          'passed': False, 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'controller_sha256': digest(Path(__file__)), 'production_source_sha256': digest(source),
          'production_object_sha256': digest(original_object),
          'production_binary_sha256': digest(build/'strata'),
          'environment': {k: env.get(k) for k in ('PATH', 'LD_LIBRARY_PATH', 'LD_PRELOAD', 'MKLROOT', 'LIBRARY_PATH')},
          'steps': []}
start = time.monotonic()

def run(label, argv, timeout=180):
    item = {'label': label, 'argv': argv}
    record['steps'].append(item)
    before = time.monotonic()
    with (out/(label+'.stdout')).open('wb') as stdout, (out/(label+'.stderr')).open('wb') as stderr:
        completed = subprocess.run(argv, cwd=build, env=env, stdout=stdout, stderr=stderr, timeout=timeout)
    item.update(exit_code=completed.returncode, elapsed_seconds=time.monotonic()-before)
    assert completed.returncode == 0, label+' failed; inspect its saved stderr'

try:
    cpu = json.loads((base/'expert-phase-pacing-prepared/record.json').read_text())
    assert cpu['passed'] and cpu['original_source_sha256'] == record['production_source_sha256']
    assert record['production_binary_sha256'] == digest(base/'strata-workspace-reclaim-production')
    shutil.copyfile(base/'expert-phase-pacing-prepared/prefill.candidate.cpp', candidate_source)
    assert digest(candidate_source) == cpu['candidate_source_sha256']
    record['candidate_source_sha256'] = digest(candidate_source)
    commands = subprocess.run(['/usr/bin/ninja', '-t', 'commands', 'strata'], cwd=build, env=env,
                              capture_output=True, text=True, check=True, timeout=15).stdout.splitlines()
    compiles = [shlex.split(line) for line in commands if line.endswith(' -c '+str(source))]
    assert len(compiles) == 1
    argv = compiles[0]
    record['original_compile_argv'] = argv.copy()
    for flag, value in [('-o', candidate_object), ('-MT', candidate_object),
                        ('-MF', out/'prefill.cpp.o.d'), ('-c', candidate_source)]:
        assert argv.count(flag) == 1
        argv[argv.index(flag)+1] = str(value)
    run('compile', argv)
    archive = build/'libstrata_prefill.a'
    record['production_archive_sha256'] = digest(archive)
    shutil.copyfile(archive, candidate_archive)
    member = original_object.name
    ar = '/usr/bin/ar'
    old_names = subprocess.run([ar, 't', str(archive)], capture_output=True, text=True, check=True).stdout.splitlines()
    assert old_names.count(member) == 1 and len(set(old_names)) == len(old_names)
    run('archive-replace', [ar, 'r', str(candidate_archive), str(candidate_object)])
    run('archive-index', ['/usr/bin/ranlib', str(candidate_archive)])
    new_names = subprocess.run([ar, 't', str(candidate_archive)], capture_output=True, text=True, check=True).stdout.splitlines()
    assert old_names == new_names
    receipts = []
    for name in old_names:
        old = subprocess.run([ar, 'p', str(archive), name], capture_output=True, check=True).stdout
        new = subprocess.run([ar, 'p', str(candidate_archive), name], capture_output=True, check=True).stdout
        assert (new == candidate_object.read_bytes()) if name == member else (old == new)
        receipts.append({'member': name, 'original_sha256': hashlib.sha256(old).hexdigest(),
                         'candidate_sha256': hashlib.sha256(new).hexdigest(), 'replaced': name == member})
    record['archive_members'] = receipts
    links = [shlex.split(line) for line in commands if ' -o strata ' in line]
    assert len(links) == 1
    tokens = links[0]
    # Parse Ninja's known wrapper without executing it as shell text.
    assert tokens[:2] == [':', '&&'] and tokens[-2:] == ['&&', ':']
    argv = tokens[2:-2]
    record['original_link_argv'] = argv.copy()
    inputs = {str(build/token): digest(build/token) for token in argv
              if token.endswith(('.o', '.a')) and (build/token).is_file()}
    record['link_input_sha256'] = inputs
    assert argv.count('libstrata_prefill.a') == 1
    argv[argv.index('libstrata_prefill.a')] = str(candidate_archive)
    assert argv.count('-o') == 1
    argv[argv.index('-o')+1] = str(candidate_binary)
    run('link', argv)
    assert digest(source) == record['production_source_sha256']
    assert digest(original_object) == record['production_object_sha256']
    assert digest(build/'strata') == record['production_binary_sha256']
    assert all(digest(Path(path)) == value for path, value in inputs.items())
    record.update(production_inputs_unchanged=True,
                  candidate_object_sha256=digest(candidate_object),
                  candidate_archive_sha256=digest(candidate_archive),
                  candidate_binary_sha256=digest(candidate_binary), passed=True)
except BaseException as error:
    record['error'] = repr(error)
    raise
finally:
    record['elapsed_seconds'] = time.monotonic()-start
    record['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (out/'record.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps({k: v for k, v in record.items() if k not in ('archive_members', 'link_input_sha256', 'original_link_argv', 'original_compile_argv')}, indent=2))
