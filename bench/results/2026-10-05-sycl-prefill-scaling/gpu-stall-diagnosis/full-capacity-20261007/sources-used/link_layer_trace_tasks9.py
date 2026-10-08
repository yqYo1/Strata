"""Relink current trace objects with one measured CPU scheduling member."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import shlex
import shutil
import subprocess

root = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
base = Path(__file__).parent
build = root / 'build-sycl-upstream-jit'
out = base / 'layer-trace-tasks9-link'
out.mkdir(mode=0o700)
expected = 'b81a7d6fbc1c6d524cf3b64e196f5866c91ec56c76e460a66b63b28ce3aed461'

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def members(path):
    result = {}
    for name in subprocess.check_output(['/usr/bin/ar', 't', str(path)], text=True).splitlines():
        assert name not in result
        data = subprocess.check_output(['/usr/bin/ar', 'p', str(path), name])
        result[name] = {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
    return result

record = {'scope': 'Build only: same current objects and libraries, exact baseline relink, then the measured task-factor-9 CPU archive substitution; no GPU execution',
          'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'source_revision': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
          'baseline_binary_sha256': expected, 'controller_sha256': digest(Path(__file__)),
          'build_passed': False, 'gpu_verified': False}

def save():
    (out / 'record.json').write_text(json.dumps(record, indent=2) + '\n')

try:
    assert digest(build / 'strata') == expected
    assert digest(base / 'strata-prefill-layer-trace-candidate') == expected
    cpu = build / 'libstrata_kernels_cpu.a'
    old = Path('/home/yayoi/.local/state/strata-sycl/cpu-pool-scheduling-probe/production-constant9.a')
    assert digest(cpu) == '766e388d445dc0da3627f1eae91c8a2575785ac739cdeffebce16b2237f8220f'
    assert digest(old) == 'ac0a17a817eef883083214efa941ff2d6ead3e4549c0306e91b3bf78b038577c'
    pool = root / 'src/kernels/cpu/pool.cpp'
    assert digest(pool) == '41a2be640f1d1823f7ba88107beba4815cb3ad6f7ffc39e6f34575b8b1b1cd32'
    candidate_cpu = out / 'libstrata_kernels_cpu.a'
    shutil.copyfile(old, candidate_cpu)
    left, right = members(cpu), members(candidate_cpu)
    assert set(left) - set(right) == {'pool.cpp.o'}
    assert set(right) - set(left) == {'native-pool-task-factor.cpp.o'}
    assert all(left[k] == right[k] for k in set(left) & set(right))
    (out / 'archive-comparison.json').write_text(json.dumps({'original': left, 'candidate': right}, indent=2) + '\n')
    command = subprocess.check_output(['/usr/bin/ninja', '-C', str(build), '-t', 'commands', 'strata'], text=True).splitlines()[-1]
    assert command.startswith(': && ') and command.endswith(' && :')
    argv = shlex.split(command[5:-5])
    assert argv.count('libstrata_kernels_cpu.a') == 1
    inputs = {v: digest(build / v) for v in argv if v.endswith(('.a', '.o'))}
    record['link_input_sha256_before'] = inputs
    record['source_sha256'] = {str(p): digest(p) for p in [pool, root / 'sycl/src/program/generate.cpp', root / 'sycl/src/prefill/prefill.cpp']}
    record['original_cpu_archive_sha256'] = digest(cpu)
    record['candidate_cpu_archive_sha256'] = digest(candidate_cpu)
    env = dict(os.environ, PATH='/usr/bin:/bin:/opt/intel/oneapi/compiler/2026.1/bin')
    env.pop('LD_LIBRARY_PATH', None)
    env.pop('LD_PRELOAD', None)
    env.update(MKLROOT='/opt/intel/oneapi/mkl/2026.1',
               LIBRARY_PATH='/opt/intel/oneapi/mkl/2026.1/lib:/opt/intel/oneapi/compiler/2026.1/lib:/usr/lib/x86_64-linux-gnu')
    record['build_environment'] = {k: env.get(k) for k in ['PATH', 'LD_LIBRARY_PATH', 'LD_PRELOAD', 'MKLROOT', 'LIBRARY_PATH']}
    record['links'] = []
    save()
    for name, output, archive in [
            ('control', base / 'strata-layer-trace-relink-control', 'libstrata_kernels_cpu.a'),
            ('tasks9', base / 'strata-layer-trace-tasks9', str(candidate_cpu))]:
        assert not output.exists()
        args = argv.copy()
        args[args.index('-o') + 1] = str(output)
        args[args.index('libstrata_kernels_cpu.a')] = archive
        item = {'name': name, 'argv': args}
        record['links'].append(item)
        save()
        with (out / (name + '.stdout')).open('wb') as stdout, (out / (name + '.stderr')).open('wb') as stderr:
            run = subprocess.run(args, cwd=build, env=env, stdout=stdout, stderr=stderr, timeout=300)
        item['exit_code'] = run.returncode
        assert run.returncode == 0
        item['binary_sha256'] = digest(output)
        item['binary_bytes'] = output.stat().st_size
        if name == 'control':
            assert item['binary_sha256'] == expected, 'Exact link reproduction failed'
        record['link_input_sha256_after_' + name] = {v: digest(build / v) for v in inputs}
        assert record['link_input_sha256_after_' + name] == inputs
        assert digest(build / 'strata') == expected
        save()
    record['build_passed'] = True
except BaseException as error:
    record['error'] = repr(error)
finally:
    record['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save()
    print(json.dumps({k: v for k, v in record.items() if k not in ['link_input_sha256_before', 'link_input_sha256_after_control', 'link_input_sha256_after_tasks9', 'links']}, indent=2))
raise SystemExit(0 if record['build_passed'] else 1)
