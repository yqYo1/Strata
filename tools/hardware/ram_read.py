#!/usr/bin/env python3
"""Owned, bounded read-only RAM envelope. Repository LICENSE; Main-authored.

Commit source before prepare; commit prepared inputs before build; commit the
build record before measure. No GPU or inference implementation is changed.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import shlex
import statistics
import subprocess
import sys

sys.dont_write_bytecode = True
P = Path(__file__).resolve().parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
ROOT = B / 'ram-read-ceiling-v3'
RESULT = P / 'bench/results/2026-10-15-pr6-recovery-audit/cpu-roofline-current.json'
KEY = 'ram_read_ceiling_v3'
SOURCE = P / 'bench/hardware/ram_read.cpp'
PINNED = ['bench/hardware/ram_read.cpp', 'tools/hardware/ram_read.py',
          'AGENTS.md', 'docs/ARTIFACT_RETENTION.md']
CC = Path('/usr/bin/g++').resolve()
FLAGS = ['-O3', '-march=znver3', '-std=c++20', '-pthread', '-fno-lto']
ENV = {'PATH': '/usr/bin:/bin', 'LC_ALL': 'C.UTF-8', 'LANG': 'C.UTF-8',
       'HOME': '/home/yayoi', 'MALLOC_ARENA_MAX': '1'}
ORDER = [1, 3, 6, 6, 3, 1, 1, 6, 3]


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(argv, **kw):
    with subprocess.Popen(list(map(str, argv)), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=ENV, cwd=P, start_new_session=True, **kw) as child:
        try:
            out, error = child.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, 9); child.communicate(); raise
        require(child.returncode == 0, 'command failed: ' + error[-2000:])
        return out.strip()


def git(*argv):
    return run(['git', *argv])


def save(row):
    ROOT.mkdir(exist_ok=True)
    (ROOT / 'record.json').write_text(json.dumps(row, indent=2) + '\n')
    d = json.loads(RESULT.read_text()); d[KEY] = row
    RESULT.write_text(json.dumps(d, indent=2, ensure_ascii=True) + '\n')


def require_committed_row(row):
    committed = json.loads(git('show', 'HEAD:' + str(RESULT.relative_to(P))))[KEY]
    require(committed == row, 'commit the original plan/build/results before advancing')


def verify(row, binary=False):
    for rel, expected in row['source_inputs'].items():
        require(sha(P / rel) == expected, 'changed pinned source: ' + rel)
        blob = subprocess.check_output(['git', 'show', row['source_commit'] + ':' + rel], cwd=P)
        require(hashlib.sha256(blob).hexdigest() == expected, 'source Git recovery: ' + rel)
    for path, expected in row['build_inputs'].items():
        require(sha(path) == expected, 'changed compiler/header/link input: ' + path)
    if binary:
        require(sha(ROOT / 'ram-read') == row['binary_sha256'], 'actual executable mismatch')


def dependency_paths(text):
    return {str((P / p).resolve()) for p in shlex.split(text.split(':', 1)[1].replace('\\\n', ' '))}


def cpu_cgroup():
    membership = Path('/proc/self/cgroup').read_text().strip()
    require(membership.startswith('0::/'), 'this fixture records the actual cgroup2 hierarchy')
    root = Path('/sys/fs/cgroup')
    current = root / membership.split('::', 1)[1].lstrip('/')
    require(current.is_dir(), 'current cgroup must be visible')
    rows, limits = [], []
    while True:
        path = current / 'cpu.max'
        # The cgroup2 root has no cpu.max; limits belong to descendant groups.
        if path.exists():
            text = path.read_text().strip(); quota, period = text.split()
            rows.append(dict(path=str(path), value=text))
            if quota != 'max':
                limits.append(int(quota) / int(period))
        if current == root:
            break
        current = current.parent
    return dict(membership=membership, visible_limits=rows,
        effective_quota_cores=min(limits) if limits else None,
        meaning='None means no finite quota in the visible current-to-root hierarchy')


def prepare():
    require(not ROOT.exists(), 'do not replace an existing run or failed plan')
    revision = git('rev-parse', 'HEAD')
    sources = {}
    for rel in PINNED:
        require((P / rel).read_bytes() == subprocess.check_output(
            ['git', 'show', revision + ':' + rel], cwd=P), 'commit exact source first: ' + rel)
        sources[rel] = sha(P / rel)
    require('avx2' in Path('/proc/cpuinfo').read_text(), 'AVX2 hardware required')
    cpus, seen = [], set()
    for cpu in sorted(os.sched_getaffinity(0)):
        top = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        key = ((top / 'physical_package_id').read_text(), (top / 'core_id').read_text())
        if key not in seen:
            cpus.append(cpu); seen.add(key)
    require(len(cpus) >= 6, 'six available distinct physical cores required')
    deps = dependency_paths(run([CC, *FLAGS, '-M', '-MT', 'ram-read', SOURCE]))
    inputs = set(deps) | {str(CC)}
    for name in ('cc1plus', 'as', 'collect2', 'ld'):
        value = run([CC, '-print-prog-name=' + name]); path = Path(value)
        if not path.is_absolute():
            path = Path('/usr/bin') / value
        require(path.is_file(), 'compiler program missing: ' + name)
        inputs.add(str(path.resolve()))
    programs = sorted(inputs - deps)
    for program in programs:
        for token in run(['/usr/bin/ldd', program]).split():
            if token.startswith('/') and Path(token).is_file():
                inputs.add(str(Path(token).resolve()))
    for name in ('Scrt1.o', 'crti.o', 'crtbeginS.o', 'crtendS.o', 'crtn.o',
                 'libstdc++.so', 'libm.so', 'libm.so.6', 'libmvec.so.1',
                 'libgcc_s.so', 'libgcc_s.so.1', 'libgcc.a', 'libc.so', 'libc.so.6',
                 'libc_nonshared.a', 'libpthread.a', 'ld-linux-x86-64.so.2', 'liblto_plugin.so'):
        path = Path(run([CC, '-print-file-name=' + name]))
        require(path.is_file(), 'link input missing: ' + name)
        inputs.add(str(path.resolve()))
    snapshot = {p: sha(p) for p in sorted(inputs)}
    row = dict(schema='ram-read-ceiling-v1', stage='prepared', active=False, ownership_closed=True,
        source_commit=revision, source_inputs=sources, compiler=str(CC),
        compiler_version=run([CC, '--version']).splitlines()[0], flags=FLAGS,
        environment=ENV, build_inputs=snapshot, headers=sorted(deps),
        cpus=cpus[:6], order=ORDER, buffer_bytes=1 << 30,
        plan='Nine fresh processes in1,3,6,6,3,1,1,6,3physical-core order. One shared1GiB '
             'anonymous read-only buffer,disjoint slices,prefaulted deterministic integer data,guard pages. '
             'Four AVX2XOR accumulators;exact scalar slice/tail checks outside timing. Calibrate by '
             'doubling to0.75s,then3eligible batches/process with0.5s floor,no major faults and '
             'exact post-batch checksum. Store every individual wall/worker CPU/fault/byte sample. '
             'Stop first failure;no threshold changes or retries. No inference/adoption gate.',
        budgets=dict(process_wall_seconds=90, process_cpu_seconds=180, address_space_bytes=3 << 30,
                     combined_text_bytes=64 << 10, total_raw_bytes=2 << 20),
        interpretation='Observed logical continuous-read envelope,not physical DRAM/bus counters, '
             'random-expert latency,application saturation or a model-speed claim. Compare to actual '
             'pool logical bytes cautiously;both GU arithmetic and quantization remain in pool clocks.',
        consumer='Choose the next selective NT1cache/native NT2..4CPUexpert component test. '
             'Retire fixture binary/raw output after result and next decision;keep reusable source only '
             'while hardware ceilings need remeasurement.',
        runtime=dict(kernel=os.uname().release, boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            cpu_model=next(x.split(':', 1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines()
                if x.startswith('model name')), initial_cpu_cgroup=cpu_cgroup(),
            llc_size=Path(f'/sys/devices/system/cpu/cpu{cpus[0]}/cache/index3/size').read_text().strip(),
            allowed_cpus=sorted(os.sched_getaffinity(0)), initial_load_average=os.getloadavg(),
            thp=Path('/sys/kernel/mm/transparent_hugepage/enabled').read_text().strip()))
    save(row); print('prepared', len(snapshot), 'original compiler/header/link inputs; commit before build')


def build(row):
    require(row['stage'] == 'prepared', 'build only the original prepared stage')
    require_committed_row(row); verify(row)
    obj = ROOT / 'ram-read.o'
    compile_argv = [CC, *FLAGS, '-c', SOURCE, '-MD', '-MF', ROOT / 'ram-read.d', '-o', obj]
    run(compile_argv)
    require(dependency_paths((ROOT / 'ram-read.d').read_text()) == set(row['headers']), 'actual header closure differs')
    object_hash = sha(obj)
    argv = [CC, *FLAGS, obj, '-Wl,-t', '-o', ROOT / 'ram-read']
    text = run(argv)
    links = []
    for line in text.splitlines():
        value = line.strip()
        if value.startswith('/') and Path(value).is_file():
            real = str(Path(value).resolve()); links.append(real)
            require(real in row['build_inputs'] or (real == str(obj) and sha(obj) == object_hash),
                'unprepared actual linker input: ' + real)
        elif value.startswith('('):
            archive = value[1:].split(')', 1)[0]
            require(str(Path(archive).resolve()) in row['build_inputs'], 'unprepared archive: ' + archive)
        else:
            require(not value, 'unclassified linker output: ' + value)
    require(len(links) >= 10, 'missing real linker trace')
    verify(row)
    row.update(stage='built', binary_sha256=sha(ROOT / 'ram-read'), actual_link_inputs=links,
        compile_argv=list(map(str, compile_argv)), build_argv=list(map(str, argv)),
        generated_object=dict(path=str(obj), sha256=object_hash, source=str(SOURCE)), build_exit_code=0)
    save(row); print('built', row['binary_sha256'], 'links', len(links), '; commit before measure')


def limits():
    os.setsid()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CPU, (180, 180))
    resource.setrlimit(resource.RLIMIT_AS, (3 << 30, 3 << 30))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))


def measure(row):
    require(row['stage'] == 'built' and not row['active'], 'one original closed build required')
    require_committed_row(row); verify(row, True)
    require(int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines()
        if x.startswith('MemAvailable:'))) * 1024 >= 8 << 30, '8GiB available required')
    row.update(stage='measuring', active=True, ownership_closed=False, samples=[])
    row['measurement_cpu_cgroup'] = cpu_cgroup()
    row['measurement_load_average'] = os.getloadavg()
    save(row)
    try:
        for ordinal, n in enumerate(ORDER):
            verify(row, True)
            cpus = row['cpus'][:n]; argv = [str(ROOT / 'ram-read'), ','.join(map(str, cpus))]
            with (ROOT / f'process-{ordinal}.stdout').open('w') as out, (ROOT / f'process-{ordinal}.stderr').open('w') as err:
                child = subprocess.Popen(argv, stdout=out, stderr=err, env=ENV, cwd=ROOT, preexec_fn=limits)
                row['live_child'] = dict(ordinal=ordinal, pid=child.pid, argv=argv,
                    start_ticks=Path(f'/proc/{child.pid}/stat').read_text().rsplit(')', 1)[1].split()[19])
                save(row)
                try:
                    code = child.wait(timeout=90)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, 9); child.wait(); raise RuntimeError('owned child wall budget')
            text = (ROOT / f'process-{ordinal}.stdout').read_text()
            error = (ROOT / f'process-{ordinal}.stderr').read_text()
            require(len(text.encode()) + len(error.encode()) <= 64 << 10, 'text budget')
            require(code == 0 and not error, 'native process failed: ' + error[:500])
            value = json.loads(text)
            require(value['schema'] == 'ram-read-v1' and value['buffer_bytes'] == 1 << 30
                and value['threads'] == n and value['checksum_valid'] is True
                and len(value['samples']) == 3, 'native workload or checks differ')
            for s in value['samples']:
                require(s['wall_seconds'] >= .5 and s['logical_bytes'] == value['passes'] * (1 << 30)
                    and s['major_faults'] == 0 and 0 < s['worker_cpu_seconds'] <= n * s['wall_seconds'] * 1.02
                    and abs(s['logical_GBps'] - s['logical_bytes'] / s['wall_seconds'] / 1e9) < 1e-5,
                    'sample duration/CPU/fault/byte conservation')
            verify(row, True)
            row['samples'].append(dict(ordinal=ordinal, cpus=cpus, native_exit_code=code,
                ownership_closed=True, result=value))
            row.pop('live_child', None); save(row)
            print('closed', ordinal, n, 'cores median GB/s',
                statistics.median(s['logical_GBps'] for s in value['samples']), flush=True)
        summary = []
        for n in (1, 3, 6):
            processes = [s for s in row['samples'] if len(s['cpus']) == n]
            medians = [statistics.median(s['logical_GBps'] for s in p['result']['samples']) for p in processes]
            summary.append(dict(physical_cores=n, process_medians_GBps=medians,
                median_GBps=statistics.median(medians), min_GBps=min(medians), max_GBps=max(medians)))
        row.update(stage='closed', active=False, ownership_closed=True, passed=True, summary=summary)
    except BaseException as e:
        if 'child' in locals() and child.poll() is None:
            os.killpg(child.pid, 9); child.wait()
        closed = 'child' not in locals() or child.poll() is not None
        row.update(stage='failed', active=not closed, ownership_closed=closed, passed=False,
            error=type(e).__name__ + ': ' + str(e))
        save(row); raise
    save(row)


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'build', 'measure'), 'prepare|build|measure')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        action = sys.argv[1]
        if action == 'prepare':
            prepare()
        else:
            row = json.loads((ROOT / 'record.json').read_text())
            try:
                build(row) if action == 'build' else measure(row)
            except BaseException as e:
                if action == 'build':
                    row.update(stage='failed_build_admission', active=False, ownership_closed=True,
                        passed=False, error=type(e).__name__ + ': ' + str(e))
                    save(row)
                raise
