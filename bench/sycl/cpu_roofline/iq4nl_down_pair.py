"""Build and compare the Zen 3 IQ4_NL Down row loop under the measurement lock."""

import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import statistics
import sys

sys.dont_write_bytecode = True
import iq3_pair as shared
from cpu_roofline import formats, machine

P, B, need, digest = shared.P, shared.B, shared.need, shared.digest
ROOT = B / 'cpu-iq4nl-down-pair-v1'
RESULT = shared.RESULT.with_name('cpu-roofline-current.json')
KEY = 'iq4nl_down_pair_v1'
BUILD = ROOT / 'build'
VERSION_HEADER = BUILD / 'ggml/src/ggml-version.h'
EXE = BUILD / 'cpu_iq4nl_down_pair'
DEPS = ('bench/sycl/cpu_roofline/iq4nl_down_pair.cpp', 'src/kernels/cpu/native_expert.cpp',
        'ggml/src/ggml-cpu/arch/x86/quants.c')
PROJECT = (*shared.PROJECT, 'tools/sycl/cpu_roofline.py')
ORDER = (('reuse', 'AB'), ('rotate', 'AB'), ('rotate', 'BA'), ('reuse', 'BA'),
         ('reuse', 'AB'), ('rotate', 'AB'))
TENSOR_BYTES = 921600
EXPERTS = {'reuse': 1, 'rotate': 128}
CALLS = 6144
FLOOR = .1
MAX_BRACKET_SHARE = .001
LIMITS = dict(build_wall=240, build_cpu=1200, build_rss=8 << 30, check_wall=35,
              measure_wall=180, text_cap=2 << 20, file_cap=20 << 20, total_cap=384 << 20)


def plan_fixture():
    row, = [row for row in formats() if row['gu'] == 18 and row['down'] == 20]
    path = Path(row['shard']).resolve(strict=True)
    stat = shared.identity(path)
    offset = row['down_offset']
    size = EXPERTS['rotate'] * TENSOR_BYTES
    need(row['layer'] == 0 and row['down_bytes'] == TENSOR_BYTES and 0 <= offset <= stat['bytes'] - size,
         'require the actual layer-0 IQ4_NL Down tensor')
    full, first = hashlib.sha256(), None
    with path.open('rb') as stream:
        stream.seek(offset)
        for expert in range(EXPERTS['rotate']):
            packed = stream.read(TENSOR_BYTES)
            need(len(packed) == TENSOR_BYTES, 'short model Down tensor')
            if expert == 0:
                first = hashlib.sha256(packed).hexdigest()
            full.update(packed)
    need(shared.identity(path) == stat, 'model changed during Down tensor hash')
    return dict(shard=str(path), file_identity=stat, layer=row['layer'], down_type=row['down'],
                offset=offset, expert_bytes=TENSOR_BYTES, reuse_sha256=first, rotate_sha256=full.hexdigest(),
                metadata_sha256=digest(shared.METADATA))


def compact(row):
    data = {key: value for key, value in row.items() if key not in ('project', 'ggml', 'host_files')}
    for key in ('project', 'ggml', 'host_files'):
        data[key] = dict(files=len(row[key]), sha256=shared.fingerprint(row[key]))
    return data


def save(row):
    shared.write(ROOT / 'record.json', row)
    current = json.loads(RESULT.read_text())
    current[KEY] = compact(row)
    shared.write(RESULT, current)


def prepare():
    committed = json.loads(shared.git(P, 'show', 'HEAD:' + str(RESULT.relative_to(P))))
    need(not ROOT.exists() and KEY not in committed and KEY not in json.loads(RESULT.read_text()),
         'one fresh CPU component comparison')
    revision = shared.git(P, 'rev-parse', 'HEAD').decode().strip()
    need(shared.git(shared.GGML, 'rev-parse', 'HEAD').decode().strip() == shared.GGML_REV, 'pinned GGML revision')
    project = shared.pins(P, revision, PROJECT)
    ggml = shared.pins(shared.GGML, shared.GGML_REV, ('ggml',))
    need(digest(shared.SDK) == shared.SDK_SHA, 'pinned official compiler inventory')
    host = json.loads(shared.SDK.read_text())
    need(len(host) == 10945, 'official compiler inventory extent')
    env = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8',
               LD_LIBRARY_PATH='/opt/intel/oneapi/compiler/2026.2/lib', OMP_NUM_THREADS='1',
               PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', TMPDIR=str(ROOT / 'tmp'),
               XDG_CACHE_HOME=str(ROOT / 'cache'))
    links = shared.host_inventory(env)
    host.update(links['files'])
    tools = (shared.CC, shared.CXX, '/usr/bin/cmake', '/usr/bin/ninja', '/usr/bin/ldd',
             '/usr/bin/git', '/usr/bin/taskset', sys.executable)
    executables = {path: dict(resolved=str(Path(path).resolve()), sha256=digest(path)) for path in tools}
    host.update({tool['resolved']: tool['sha256'] for tool in executables.values()})
    for modules in Path('/usr/share').glob('cmake-*/Modules'):
        host.update({str(p.resolve()): digest(p) for p in modules.rglob('*') if p.is_file()})
    need(all(digest(path) == value for path, value in host.items()), 'actual host/SDK input bytes')
    cpu_state = machine()
    core = cpu_state['physical_cpus'][2]
    fixture = plan_fixture()
    template = shared.GGML / 'ggml/src/ggml-version.h.in'
    ggml_header = template.read_text().replace('@GGML_VERSION@', '0.24.0').replace(
        '@GGML_BUILD_COMMIT@', shared.git(shared.GGML, 'rev-parse', '--short', 'HEAD').decode().strip())
    need('@' not in ggml_header, 'generated GGML version header')
    run = ['/usr/bin/taskset', '--cpu-list', str(core), str(EXE)]
    plans = [dict(pattern=pattern, initial=initial, argv=run + ['time', fixture['shard'],
             str(fixture['offset']), pattern, initial]) for pattern, initial in ORDER]
    row = dict(stage='prepared', source_commit=revision, dependency_commit=shared.GGML_REV,
               project=project, ggml=ggml, host_files=host, host_packages=links['packages'],
               executables=executables, interpreter=sys.executable,
               environment=env, machine=cpu_state, cpu=core, fixture=fixture,
               generated={str(VERSION_HEADER): hashlib.sha256(ggml_header.encode()).hexdigest()},
               configure=['/usr/bin/cmake', '-S', str(P / 'bench/sycl/cpu_roofline'), '-B', str(BUILD),
                          '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
                          '-DCMAKE_C_COMPILER=' + shared.CC, '-DCMAKE_CXX_COMPILER=' + shared.CXX,
                          '-DCMAKE_EXE_LINKER_FLAGS=-Wl,-t', '-DGGML_CCACHE=OFF',
                          '-DSTRATA_GGML_DIR=' + str(shared.GGML)],
               build=['/usr/bin/cmake', '--build', str(BUILD), '--target', EXE.name, '--parallel', '4'],
               check=run + ['check'], plans=plans, limits=LIMITS,
               comparison=dict(calls_per_batch=CALLS, minimum_seconds=FLOOR, maximum_bracket_fraction=MAX_BRACKET_SHARE,
                   gate='Five order-alternated ratios per process. All three rotation process medians and the '
                        'median of the three reuse process medians must be less than one.'),
               consumer='Single CPU-only IQ4_NL Down comparison with measured generation evidence at c74d1539c.',
               review='When the component closes, retain compact samples and recover source/recipe before build retirement.',
               execution_gpu=False, adopted=False, active=False, ownership_closed=True,
               complete=False, passed=False, processes=[])
    ROOT.mkdir(mode=0o700)
    (ROOT / 'tmp').mkdir(mode=0o700)
    (ROOT / 'cache').mkdir(mode=0o700)
    save(row)
    print('Pinned CPU component plan. Commit the plan before building.', flush=True)


def verify_source(row):
    need(row['dependency_commit'] == shared.GGML_REV and
         shared.git(shared.GGML, 'rev-parse', 'HEAD').decode().strip() == shared.GGML_REV and
         shared.pins(P, row['source_commit'], PROJECT) == row['project'] and
         shared.pins(shared.GGML, row['dependency_commit'], ('ggml',)) == row['ggml'] and
         machine() == row['machine'], 'source, dependency or host changed')
    need(sys.executable == row['interpreter'] and all(
         str(Path(path).resolve()) == tool['resolved'] and digest(path) == tool['sha256']
         for path, tool in row['executables'].items()), 'planned tool identity changed')


def load():
    row = json.loads((ROOT / 'record.json').read_text())
    committed = json.loads(shared.git(P, 'show', 'HEAD:' + str(RESULT.relative_to(P))))
    need(not row['active'] and row['ownership_closed'] and committed[KEY] == compact(row) and
         json.loads(RESULT.read_text())[KEY] == committed[KEY], 'commit exact closed component stage')
    verify_source(row)
    for path, value in row.get('built_inputs', {}).items():
        need(digest(path) == value, 'changed executable/build/runtime input: ' + path)
    return row


def runtime_paths(run):
    stdout, stderr = run('ldd', ['/usr/bin/ldd', str(EXE)], 10)
    text = stdout.read_text()
    paths = {str(Path(path).resolve()) for path in re.findall(r'(/[^\s]+)', text)}
    need(paths and not stderr.stat().st_size and
         not any(word in text.lower() for word in ('not found', 'sycl', 'libze_', 'cuda', 'amdhip', 'libur_')),
         'actual CPU-only runtime libraries')
    return sorted(paths)


def verify_build(row, run):
    allowed = {**row['host_files'], **row['project'], **row['ggml'], **row['generated']}
    need(all(digest(path) == value for path, value in row['host_files'].items()), 'build compiler input changed')
    run('configure', row['configure'], 60, 120, 2 << 30)
    link_trace, _ = run('compile', row['build'], LIMITS['build_wall'], LIMITS['build_cpu'], LIMITS['build_rss'])
    dep_trace, _ = run('deps', ['/usr/bin/ninja', '-C', str(BUILD), '-t', 'deps'], 20)
    paths = {str((BUILD / line.strip()).resolve()) for line in dep_trace.read_text().splitlines()
             if line.startswith('    ') and line.strip()}
    paths.add(str(VERSION_HEADER))
    need(len(paths) > 150 and all(allowed.get(path) == digest(path) for path in paths),
         'compiler dependency list contains unpinned inputs')
    database = BUILD / 'compile_commands.json'
    units = json.loads(database.read_text())
    for name, prefix, compiler, arch in zip(DEPS, (P, P, shared.GGML), (shared.CXX, shared.CXX, shared.CC),
                                           ('-march=znver3', '-march=znver3', '-march=native'), strict=True):
        path = str((prefix / name).resolve())
        unit, = [entry for entry in units if str(Path(entry['file']).resolve()) == path]
        argv = shlex.split(unit['command'])
        need(path in paths and Path(argv[0]).resolve() == Path(compiler).resolve() and {'-O3', arch} <= set(argv)
             and not {'-ffast-math', '-Ofast', '-fsycl'} & set(argv), 'CPU comparison flags: ' + name)
    cache = BUILD / 'CMakeCache.txt'
    configuration = cache.read_text()
    need(all(re.search('^GGML_' + key + ':BOOL=OFF$', configuration, re.M) for key in
             ('CUDA', 'HIP', 'SYCL', 'OPENMP', 'BACKEND_DL')), 'CPU-only configure selection')
    linked = set()
    for line in link_trace.read_text().splitlines():
        word = line.strip()
        if not word or any(ch.isspace() for ch in word):
            continue
        path = (BUILD / word).resolve()
        if path.is_file() and (word.startswith('/') or path.suffix in ('.a', '.o')):
            linked.add(path)
    library = BUILD / 'libexpert_cpu.a'
    oracle = BUILD / 'ggml/src/libggml-cpu.a'
    external = {str(path) for path in linked if not path.is_relative_to(BUILD) and not path.is_relative_to(ROOT / 'tmp')}
    need({library, oracle} <= linked and len(external) > 8 and
         all(allowed.get(path) == digest(path) for path in external), 'actual compiler/linker inputs')
    runtime = runtime_paths(run)
    launch_tools = [str(Path(path).resolve()) for path in ('/usr/bin/taskset', '/usr/bin/ldd', sys.executable)]
    need(all(allowed.get(p) == digest(p) for p in (*runtime, *launch_tools)), 'pinned CPU runtime and launch tools')
    archives = {str(p) for p in linked if p.is_relative_to(BUILD) and p.suffix == '.a'}
    keys = {*runtime, *launch_tools, *external, *archives, *map(str, (EXE, cache, database, VERSION_HEADER))}
    row.update(stage='built', runtime_paths=runtime, built_inputs={path: digest(path) for path in sorted(keys)},
               compiler_inputs=dict(files=len(paths), sha256=shared.fingerprint({p: allowed[p] for p in paths})))


def parse_evidence(text, plan, row):
    head, *samples, tail = text.splitlines()
    kind, cpu, experts, byte_count, weights, activation, bracket = head.split()
    count = EXPERTS[plan['pattern']]
    need(kind == 'INPUT' and (int(cpu), int(experts), int(byte_count)) == (row['cpu'], count, count * TENSOR_BYTES)
         and tail == f'VALIDATED {count * 4} cases before and after' and len(samples) == 5,
         'complete real-weight component validation')
    timer = float(bracket)
    need(math.isfinite(timer), 'finite clock bracket')
    pairs = []
    for repeat, sample in enumerate(samples):
        order, calls, reference, candidate, output = sample.split()
        need(order == plan['initial'][::(-1 if repeat % 2 else 1)] and int(calls) == CALLS,
             'changed dedicated CPU samples or trial order')
        times = tuple(map(float, (reference, candidate)))
        need(all(math.isfinite(t) for t in times), 'nonfinite component timing')
        eligible = min(times) >= FLOOR and 0 < timer <= MAX_BRACKET_SHARE * min(times)
        pairs.append(dict(order=order, reference_s=times[0], candidate_s=times[1], fingerprint=int(output), eligible=eligible))
    return dict(input=dict(cpu=int(cpu), experts=count, weights=int(weights), activation=int(activation)),
                timer_s=timer, pairs=pairs)


def decide(processes):
    need(tuple((p['pattern'], p['initial']) for p in processes) == ORDER and
         len({(p['receipt']['pid'], p['receipt']['direct_identity']['start_ticks'])
              for p in processes}) == len(ORDER), 'complete independent-process matrix')
    medians = {pattern: [statistics.median(pair['candidate_s'] / pair['reference_s'] for pair in p['pairs'])
              for p in processes if p['pattern'] == pattern] for pattern in EXPERTS}
    return dict(process_ratios=medians, qualified=all(r < 1 for r in medians['rotate']) and
                statistics.median(medians['reuse']) < 1, scope='Single-core IQ4_NL Down row loop')


def execute(stage):
    row = load()
    need(stage == {'prepared': 'build', 'built': 'check', 'checked': str(len(row['processes']))}.get(row['stage']),
         'use next planned stage')
    root = ROOT / ('owned-' + stage)
    need(not root.exists(), 'inspect existing stage ownership before continuing')

    def persist():
        shared.write(ROOT / 'execution.json', dict(stage=stage, active=bool(owner.active), commands=owner.commands))

    owner = shared.owned.Owner(root, persist=persist)

    def run(label, argv, wall, cpu=180, rss=1 << 30):
        entry, stdout, stderr = owner.run(label, argv, row['environment'], ROOT, wall=wall, cpu=cpu,
             rss_cap=rss, text_cap=LIMITS['text_cap'], file_cap=LIMITS['file_cap'], total_cap=LIMITS['total_cap'])
        need(shared.owned.closed(entry) and entry['exit_code'] == 0,
             label + ' failed: ' + stderr.read_text(errors='replace')[-1200:])
        return stdout, stderr

    try:
        if stage != 'build':
            need(runtime_paths(run) == row['runtime_paths'], 'CPU runtime selection changed')
        if stage == 'build':
            verify_build(row, run)
        elif stage == 'check':
            stdout, stderr = run('fixture', row['check'], LIMITS['check_wall'], 40)
            need(not stderr.stat().st_size and stdout.read_text().strip() == 'IQ4_NL Down synthetic exact cases 60',
                 'synthetic reference and guards')
            row.update(stage='checked', correctness=dict(cases=60, passed=True,
                       receipt=shared.receipt(owner.commands[-1])))
        else:
            plan = row['plans'][int(stage)]
            need(plan_fixture() == row['fixture'], 'real Down ranges or asset metadata changed before launch')
            stdout, stderr = run('measure', plan['argv'], LIMITS['measure_wall'])
            need(not stderr.stat().st_size, 'unexpected component stderr')
            sample = dict(parse_evidence(stdout.read_text(), plan, row), pattern=plan['pattern'], initial=plan['initial'],
                          receipt=shared.receipt(owner.commands[-1]))
            prior = [p for p in row['processes'] if p['pattern'] == sample['pattern']]
            row['processes'].append(sample)
            need(all(pair['eligible'] for pair in sample['pairs']) and
                 all(p['input'] == sample['input'] and [x['fingerprint'] for x in p['pairs']] ==
                     [x['fingerprint'] for x in sample['pairs']] for p in prior) and
                 plan_fixture() == row['fixture'], 'batch eligibility, input or output evidence changed')
            if len(row['processes']) == len(ORDER):
                row.update(stage='closed', complete=True, passed=True, decision=decide(row['processes']))
        verify_source(row)
        need(all(digest(path) == value for path, value in row['built_inputs'].items()), 'post-stage build input drift')
    except BaseException as error:
        row.update(stage='failed', complete=False, passed=False, error=repr(error))
        raise
    finally:
        row.update(active=bool(owner.active), ownership_closed=owner.active is None and
                   all(shared.owned.closed(entry) for entry in owner.commands))
        row.setdefault('receipts', {})[stage] = [shared.receipt(entry) for entry in owner.commands]
        save(row)


if __name__ == '__main__':
    need(len(sys.argv) == 2, 'use prepare|build|check|0..5')
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        shared.guards.no_model_owners([])
        prepare() if sys.argv[1] == 'prepare' else execute(sys.argv[1])
