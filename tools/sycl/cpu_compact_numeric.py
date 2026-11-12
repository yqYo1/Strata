#!/usr/bin/env python3
"""Frozen CPU-only real-weight/actual-dispatch numeric gate, not speed evidence.

Main-authored; repository LICENSE. Fresh project/GGML compilation avoids
unrecorded archive reuse. Each stage is single-use and shares Main's owner lock.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
import cpu_compact_hot as hot

P, B = hot.P, hot.B
ROOT = B / 'cpu-compact-hot-numeric-v1'
PLAN = P / 'bench/sycl/cpu_compact_hot/numeric-plan.json'
RESULT = P / 'bench/results/2026-10-15-pr6-recovery-audit/cpu-roofline-current.json'
METADATA = Path('/home/yayoi/.local/share/strata-sycl/packs/qwen3.8-flash-next-iq3_s/native_experts.txt')
MODEL = Path('/home/yayoi/.local/share/strata-sycl/models/qwen3.8-flash-next-iq3_s/IQ3_S')
PRODUCTION = ('src/kernels/cpu', 'include/strata/kernels/cpu',
              'include/strata/core/progress.hpp', 'third_party/ggml')
PREFIXES = PRODUCTION + ('bench/sycl/cpu_compact_hot', 'tools/sycl/cpu_compact_numeric.py',
    'tools/sycl/cpu_compact_hot.py', 'tools/sycl/measurement_owner.py', 'tools/sycl/measurement_guards.py')
need, sha, write = hot.need, hot.sha, hot.write
CM, NJ = '/usr/bin/cmake', '/usr/bin/ninja'
CC = '/opt/intel/oneapi/compiler/2026.2/bin/icx'
OWNERSHIP_FIELDS = ('source_commit', 'component_source_commit', 'plan_sha256', 'binary',
    'passed', 'complete', 'active', 'ownership_closed', 'numeric_oracle_qualified',
    'performance_qualified', 'gates')


def sources(revision):
    names = hot.git('ls-tree', '-r', '--name-only', revision, '--', *PREFIXES).decode().splitlines()
    need(names, 'source scope empty')
    result = {}
    for name in names:
        raw = hot.git('show', revision + ':' + name)
        need((P / name).read_bytes() == raw, 'uncommitted source: ' + name)
        result[name] = hashlib.sha256(raw).hexdigest()
    return result


def identity(path):
    s = Path(path).stat()
    return dict(path=str(Path(path).resolve()), dev=s.st_dev, ino=s.st_ino,
                bytes=s.st_size, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


def cmake_snapshot():
    version = subprocess.check_output([CM, '--version'], text=True, timeout=10).splitlines()[0]
    match = re.fullmatch(r'cmake version (\d+)\.(\d+)\..+', version)
    need(match, 'CMake version format')
    root = Path('/usr/share') / ('cmake-' + match[1] + '.' + match[2])
    need((root / 'Modules').is_dir() and (root / 'Templates').is_dir(), 'installed CMake sources missing')
    pins = {str(p): sha(p) for name in ('Modules', 'Templates')
            for p in (root / name).rglob('*') if p.is_file()}
    need(len(pins) > 100, 'incomplete CMake source snapshot')
    return dict(root=str(root), version=version, files=len(pins),
                canonical_sorted_json_sha256=hashlib.sha256(json.dumps(pins, sort_keys=True).encode()).hexdigest())


def fixture_hash(row):
    before = identity(row['shard'])
    digest = hashlib.sha256()
    with Path(row['shard']).open('rb', buffering=0) as stream:
        for expert in range(32):
            for start, count in zip(row['offsets'], row['extents']):
                offset = start + expert * count
                need(0 <= offset <= before['bytes'] and count <= before['bytes'] - offset, 'slice bounds')
                stream.seek(offset)
                data = stream.read(count)
                need(len(data) == count, 'short model slice')
                digest.update(data)
    need(identity(row['shard']) == before, 'model changed during hash')
    return digest.hexdigest()


def model_rows():
    lines = METADATA.read_text().splitlines()
    first = lines[0].split('absolute offsets in ')[1].split(',')[0]
    rows = {}
    for line in lines:
        if not line or line.startswith('#'):
            continue
        fields = line.split()
        layer, gu, down, _, size, gate, up, offset = map(int, fields[:8])
        if gu != 18 or down not in (20, 42) or down in rows:
            continue
        extents = [980 * 640, 980 * 640, (360 if down == 20 else 180) * 2560]
        need(size == sum(extents), 'model metadata geometry')
        row = dict(layer=layer, gu=gu, down=down, blob_bytes=size,
            offsets=[gate, up, offset], extents=extents,
            shard=str(MODEL / (fields[8] if len(fields) == 9 else first)))
        row.update(identity=identity(row['shard']), sha256=fixture_hash(row))
        rows[down] = row
    need(set(rows) == {20, 42}, 'both actual IQ3_XXS model formats required')
    return [rows[20], rows[42]]


def prepare():
    need(not ROOT.exists() and not PLAN.exists(), 'fresh plan/root required')
    lifetime = json.loads((hot.ROOT / 'check.json').read_text())
    public = json.loads(RESULT.read_text())['compact_cache_lifetime_v1']
    need(lifetime['passed'] and lifetime['complete'] and lifetime['ownership_closed']
         and not lifetime['active'] and public['passed'] and public['binary'] == lifetime['binary'],
         'closed committed ownership gate required')
    need(hot.git('show', 'HEAD:' + str(RESULT.relative_to(P))) == RESULT.read_bytes(), 'commit ownership evidence')
    old_plan = json.loads(hot.PLAN.read_text()); hot.verify(old_plan)
    native = json.loads(Path(old_plan['native_recipe']['path']).read_text())
    dep, = [d for d in native['dependencies'] if d['revision'] == '3cf03257f219afbe7334045ff7c6a06ac68c627d']
    revision = hot.git('rev-parse', 'HEAD').decode().strip(); pins = sources(revision)
    for name, digest in pins.items():
        if any(name == prefix or name.startswith(prefix + '/') for prefix in PRODUCTION):
            need(native['project']['source_sha256'].get(name) == digest,
                 'production source differs from coherent build: ' + name)
    tools = {tool: sha(tool) for tool in (hot.CXX, CC, CM, NJ, hot.READELF, hot.PRLIMIT, hot.PYTHON)}
    need(all(tools[p] == h for p, h in native['toolchain'].items()), 'qualified toolchain differs')
    external = native['external_inputs']
    need(sha(external['path']) == external['sha256'], 'external census changed')
    flags, = [s.split('=', 1)[1] for s in native['recipe']['configure'] if s.startswith('-DCMAKE_CXX_FLAGS=')]
    env = dict(old_plan['environment']); env['TMPDIR'] = str(ROOT / 'tmp')
    plan = dict(schema='compact-cache-numeric-v1', source_commit=revision, sources=pins,
        ownership_plan_sha256=sha(hot.PLAN), tools=tools, native_recipe=old_plan['native_recipe'],
        ownership_evidence={k: public[k] for k in OWNERSHIP_FIELDS},
        compiler_inputs=old_plan['compiler_inputs'], external_inputs=external,
        cmake_inputs=cmake_snapshot(),
        GGML=dict(root=dep['root'], revision=dep['revision'], source_sha256=dep['source_sha256']),
        metadata=dict(path=str(METADATA), sha256=sha(METADATA)), formats=model_rows(),
        environment=env, native_environment=dict(env, STRATA_IQ3_PAIR='1', STRATA_IQ3_PAIR_AUDIT='1',
            STRATA_IQ2_PAIR='0', STRATA_IQ2_PAIR_AUDIT='0', STRATA_IQ_SINGLE_EXACT='0',
            STRATA_IQ_MT_MIN='2', STRATA_CPU_SERVICE='0', STRATA_CPU_WORK_CENSUS='0'),
        configure=[CM, '-S', str(P / 'bench/sycl/cpu_compact_hot'), '-B', str(ROOT / 'build'),
            '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
            '-DCMAKE_C_COMPILER=' + CC, '-DCMAKE_CXX_COMPILER=' + hot.CXX,
            '-DCMAKE_CXX_FLAGS=' + flags, '-DCMAKE_EXE_LINKER_FLAGS=-Wl,-t',
            '-DSTRATA_GGML_DIR=' + dep['root']],
        build=[CM, '--build', str(ROOT / 'build'), '--target', 'compact_cache_numeric', '--parallel', '3'],
        acceptance='Both actual Down20/42 children normal0,quiet stderr,closed owners,32real experts,'
            '4quantized activations,384serial bit comparisons,640threaded rows,phase/NT/alias/input guards,'
            'actual audit392calls/181376NT1rows/181248paired rows. Fresh project/GGML compilation with'
            ' covered compiler/link inputs;no GPU libraries. No speed/model/strictRSS qualification.',
        stopping='Exclusive attempt before admission;stop on any source/tool/model/owner/build/dispatch/numeric'
            '/budget failure. No retry,hash refresh,changed arithmetic or relaxed tolerance.',
        budgets=dict(configure_wall=90, build_wall=240, check_wall=60, rss=3 << 30,
            output=2 << 20, total=384 << 20, cpu=900),
        retention='Current ownership99KB fixture is preserved while this source-built numeric successor'
            ' is qualified. Retire successful raw output and unused predecessor at a tested boundary;'
            'keep source/recipe,compact numeric result and current auxiliary for imminent hot/cold comparison.')
    ROOT.mkdir(mode=0o700); (ROOT / 'tmp').mkdir(); write(PLAN, plan)
    print('Prepared CPU-only actual-dispatch numeric plan;commit before build/check.', flush=True)


def verify(plan, *, models=True):
    need(hot.git('show', 'HEAD:' + str(PLAN.relative_to(P))) == PLAN.read_bytes(), 'commit exact numeric plan')
    for name, expected in plan['sources'].items():
        raw = hot.git('show', plan['source_commit'] + ':' + name)
        need(hashlib.sha256(raw).hexdigest() == expected and sha(P / name) == expected,
             'source drift: ' + name)
    need(sha(hot.PLAN) == plan['ownership_plan_sha256'], 'ownership recipe drift')
    historical = json.loads(hot.git('show', plan['source_commit'] + ':' + str(RESULT.relative_to(P))))
    current = json.loads(RESULT.read_text())
    for data in (historical, current):
        need({k: data['compact_cache_lifetime_v1'][k] for k in OWNERSHIP_FIELDS}
             == plan['ownership_evidence'], 'consumed ownership qualification changed')
    lifetime = json.loads((hot.ROOT / 'check.json').read_text())
    need(lifetime['passed'] and lifetime['complete'] and lifetime['ownership_closed']
         and not lifetime['active'] and lifetime['binary'] == plan['ownership_evidence']['binary'],
         'private ownership qualification changed')
    hot.verify(json.loads(hot.PLAN.read_text()))
    need(all(sha(p) == h for p, h in plan['tools'].items()), 'tool drift')
    for name in ('native_recipe', 'compiler_inputs', 'external_inputs', 'metadata'):
        need(sha(plan[name]['path']) == plan[name]['sha256'], name + ' drift')
    ggml = plan['GGML']; root = Path(ggml['root'])
    for name, expected in ggml['source_sha256'].items():
        raw = subprocess.check_output(['git', 'show', ggml['revision'] + ':' + name], cwd=root, timeout=30)
        need(hashlib.sha256(raw).hexdigest() == expected and sha(root / name) == expected, 'GGML source drift')
    need(cmake_snapshot() == plan['cmake_inputs'], 'CMake module/template drift')
    if models:
        for row in plan['formats']:
            need(identity(row['shard']) == row['identity'] and fixture_hash(row) == row['sha256'], 'real weight drift')


def dependencies(plan, text, build_log):
    census = json.loads(Path(plan['compiler_inputs']['path']).read_text())
    known = dict(census['compiled']) | dict(census['links'])
    known.update({str(P / p): h for p, h in plan['sources'].items()})
    known.update({str(Path(plan['GGML']['root']) / p): h for p, h in plan['GGML']['source_sha256'].items()})
    known = {str(Path(p).resolve()): h for p, h in known.items()}
    result = {}
    for line in text.splitlines():
        if not line.startswith('    '):
            continue
        path = Path(line.strip())
        if not path.is_absolute(): path = ROOT / 'build' / path
        name = str(path.resolve())
        need(name in known and sha(path) == known[name], 'uncovered compiler dependency: ' + name)
        result[name] = known[name]
    need(len(result) >= 400, 'incomplete actual dependency census')
    commands = json.loads((ROOT / 'build/compile_commands.json').read_text())
    required = {str(P / 'bench/sycl/cpu_compact_hot/numeric.cpp'), str(P / 'src/kernels/cpu/native_expert.cpp'),
                str(P / 'src/kernels/cpu/iq_single_avx2.cpp')}
    need(required <= {v['file'] for v in commands}, 'actual reference/fixture compilation absent')
    for item in commands:
        path = str(Path(item['file']).resolve())
        need(path in known and sha(path) == known[path], 'uncovered compile source')
    external = {str(Path(p).resolve()): h for p, h in
                json.loads(Path(plan['external_inputs']['path']).read_text()).items()}
    links = {}
    for line in build_log.splitlines():
        path = Path(line.strip())
        if not path.is_absolute() or not path.is_file() or path.is_relative_to(ROOT): continue
        name = str(path.resolve())
        need(name in external and sha(path) == external[name], 'uncovered link input: ' + name)
        links[name] = external[name]
    need(len(links) >= 6, 'link trace incomplete')
    return dict(compiled=result, links=links, compilation_commands=len(commands),
                compile_commands_sha256=sha(ROOT / 'build/compile_commands.json'))


def execute(action):
    with (ROOT / (action + '-attempt.json')).open('x') as stream:
        json.dump(dict(action=action, no_retry=True), stream)
    state = dict(stage=action, active=True, passed=False, complete=False, GPU_executed=False,
                 numerical_qualified=False, performance_qualified=False, commands=[])
    def persist(): write(ROOT / (action + '.json'), state)
    persist()
    try:
        plan = json.loads(PLAN.read_text()); verify(plan); hot.guards.no_model_owners([])
        state.update(source_commit=plan['source_commit'], plan_sha256=sha(PLAN))
        scope = hot.owner.Owner(ROOT / 'owned', persist); state['commands'] = scope.commands
        def run(label, command, wall, *, native=False):
            entry, out, error = scope.run(action + '-' + label,
                [hot.PRLIMIT, '--as=8589934592', '--', *command],
                plan['native_environment'] if native else plan['environment'], P,
                wall=wall, text_cap=plan['budgets']['output'], rss_cap=plan['budgets']['rss'],
                total_cap=plan['budgets']['total'], file_cap=plan['budgets']['total'], cpu=plan['budgets']['cpu'])
            need(hot.owner.closed(entry) and entry['exit_code'] == 0, 'failed/unclosed child: ' + label)
            return out.read_text(), error.read_text()
        if action == 'build':
            need(not (ROOT / 'build').exists(), 'fresh compilation directory required')
            run('configure', plan['configure'], plan['budgets']['configure_wall'])
            log, error = run('compile', plan['build'], plan['budgets']['build_wall'])
            dep, err = run('dependencies', [NJ, '-C', str(ROOT / 'build'), '-t', 'deps'], 20)
            need(not err, 'dependency tool error')
            actual = dependencies(plan, dep, log + '\n' + error)
            binary = ROOT / 'build/compact_cache_numeric'
            dynamic, err = run('libraries', [hot.READELF, '--dynamic', str(binary)], 10)
            libraries = re.findall(r'Shared library: \[([^\]]+)\]', dynamic)
            need(libraries and not err and not any(bad in lib.lower() for lib in libraries
                for bad in ('sycl', 'level_zero', 'libze', 'ur_loader', 'opencl', 'cuda', 'amdhip')), 'GPU library')
            state.update(actual_inputs=actual, binary=dict(path=str(binary), sha256=sha(binary)), libraries=libraries)
        else:
            build = json.loads((ROOT / 'build.json').read_text())
            need(build['passed'] and build['complete'] and build['ownership_closed'] and not build['active'], 'closed exact build required')
            need(build['source_commit'] == plan['source_commit'] and build['plan_sha256'] == sha(PLAN)
                 and build['actual_inputs']['compile_commands_sha256'] == sha(ROOT / 'build/compile_commands.json'),
                 'compiled source/recipe identity differs')
            need(sha(build['binary']['path']) == build['binary']['sha256'], 'numeric ELF changed')
            for paths in (build['actual_inputs']['compiled'], build['actual_inputs']['links']):
                need(all(sha(p) == h for p, h in paths.items()), 'actual built dependency drift')
            results = []
            for row in plan['formats']:
                args = [build['binary']['path'], row['shard'], str(row['down']),
                        *map(str, row['offsets']), str(row['blob_bytes']), '512']
                output, error = run('down' + str(row['down']), args, plan['budgets']['check_wall'], native=True)
                need(not error, 'numeric stderr')
                result = json.loads(output)
                expected = dict(component='compact_cache_numeric', passed=True, down=row['down'], experts=32,
                    serial_cases=384, threaded_rows=640, raw_calls=392, raw_nt1_rows=181376,
                    raw_pair_rows=181248, performance_qualified=False)
                need({k: result[k] for k in expected} == expected and -128 <= result['q8_min'] < 0
                     and 0 < result['q8_max'] <= 127, 'numeric/actual route/count mismatch')
                results.append(result)
            state.update(binary=build['binary'], results=results, numerical_qualified=True)
        verify(plan)
        state.update(passed=True, complete=True, ownership_closed=True)
    except BaseException as error:
        state.update(error=repr(error), ownership_closed=all(hot.owner.closed(c) for c in state['commands']))
        raise
    finally:
        state['active'] = False; persist()


if __name__ == '__main__':
    need(sys.executable == hot.PYTHON, 'use qualified Python')
    need(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'build', 'check'), 'prepare|build|check')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); hot.guards.no_model_owners([])
        prepare() if sys.argv[1] == 'prepare' else execute(sys.argv[1])
