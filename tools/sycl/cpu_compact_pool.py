#!/usr/bin/env python3
"""Commit-pinned fresh CPU/GGML build and actual-pool numeric gate.

Main-authored; repository LICENSE. No borrowed project object, GPU execution or
speed/adoption claim. Preserve the source-qualified old CPU/model fallbacks.
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
import cpu_compact_numeric as inputs

P, B = hot.P, hot.B
ROOT = B / 'cpu-compact-production-pool-v1'
PLAN = P / 'bench/sycl/cpu_compact_hot/pool-plan-v1.json'
THIS = 'tools/sycl/cpu_compact_pool.py'
PREFIXES = ('src/kernels/cpu', 'include/strata/kernels/cpu', 'third_party/ggml',
    'include/strata/core/progress.hpp', 'bench/sycl/cpu_compact_hot/CMakeLists.txt',
    'bench/sycl/cpu_compact_hot/pool.cpp', 'bench/sycl/cpu_compact_hot/cache.hpp',
    THIS, 'tools/sycl/cpu_compact_hot.py', 'tools/sycl/cpu_compact_numeric.py',
    'tools/sycl/measurement_owner.py', 'tools/sycl/measurement_guards.py')
need, sha, write = hot.need, hot.sha, hot.write
ABSENT = {'ggml/src/CMakeFiles/ggml.dir/ggml-backend-dl.cpp.o',
          'ggml/src/CMakeFiles/ggml.dir/ggml-backend-reg.cpp.o',
          'CMakeFiles/compact_cache_numeric.dir/numeric.cpp.o'}


def std_headers():
    return {str(p.resolve()): sha(p) for root in
        ('/usr/include/c++/13', '/usr/include/x86_64-linux-gnu/c++/13')
        for p in Path(root).rglob('*') if p.is_file()}


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def topology():
    import os
    result = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        root = Path('/sys/devices/system/cpu') / ('cpu' + str(cpu)) / 'topology'
        result[str(cpu)] = {name: (root / name).read_text().strip()
                           for name in ('physical_package_id', 'core_id', 'thread_siblings_list')}
    need(len(result) == 12 and len({(v['physical_package_id'], v['core_id']) for v in result.values()}) == 6,
         'expected six-core/twelve-thread host')
    return result


def prepare():
    need(not ROOT.exists() and not PLAN.exists(), 'fresh pool plan/root required')
    revision = hot.git('rev-parse', 'HEAD').decode().strip()
    paths = hot.git('ls-tree', '-r', '--name-only', revision, '--', *PREFIXES).decode().splitlines()
    sources = {}
    for path in paths:
        raw = hot.git('show', revision + ':' + path)
        need(raw == (P / path).read_bytes(), 'commit source before prepare: ' + path)
        sources[path] = hashlib.sha256(raw).hexdigest()
    need(THIS in sources and 'src/kernels/cpu/compact_gu.cpp' in sources, 'incomplete source scope')
    native_path = B / 'coherent-prefill-native-v1/record.json'
    native = json.loads(native_path.read_text())
    need(native['ownership_closed'] and not native['active'] and
         native['native_source_commit'] == '14508acbf1b70eb994fdf37f97fb6d0f5cff7d43', 'coherent recipe required')
    ggml, = [d for d in native['dependencies'] if d['revision'] == '3cf03257f219afbe7334045ff7c6a06ac68c627d']
    recipe = dict(path=str(native_path), sha256=sha(native_path))
    flags, = [s.split('=', 1)[1] for s in native['recipe']['configure'] if s.startswith('-DCMAKE_CXX_FLAGS=')]
    flags = flags.replace('-fsycl-id-queries-range=size_t', '').strip() # CPU-only, unused in the seed recipe.
    env = dict(PATH='/opt/intel/oneapi/compiler/2026.2/bin:/usr/bin:/bin',
        LD_LIBRARY_PATH='/opt/intel/oneapi/compiler/2026.2/lib:/usr/lib/x86_64-linux-gnu',
        LANG='C.UTF-8', LC_ALL='C.UTF-8', TMPDIR=str(ROOT / 'tmp'))
    headers = std_headers()
    census = native['actual_inputs']; external = native['external_inputs']
    need(sha(census['path']) == census['sha256'] and sha(external['path']) == external['sha256'], 'seed SDK census drift')
    plan = dict(schema='compact-production-pool-v1', source_commit=revision, sources=sources,
        qualified_fallbacks=dict(engine_source=native['native_source_commit'], engine_sha256='dcb8efd4698d4c9c6de913cb5f92e2229e945b242f24bd1fca2998e4cf89e6d1',
            CPU_numeric_source='aba1ba35c242829377183eba35148d609c8284e7', CPU_numeric_ELF='557c8cd16068321e9c8e6c6d91c2a6cdf5f4661d90b641980cb2bf5a8b422633',
            timing_ELF='979d696585b163e869936761de92cc86bd6600249fd16ae983c8caf36b1a6a78'),
        native_recipe=recipe, compiler_inputs=census, external_inputs=external,
        generated_inputs=inputs.coherent_generated(recipe, census),
        GGML=ggml, cmake_inputs=inputs.cmake_snapshot(),
        standard_headers=dict(count=len(headers), sha256=signature(headers)), topology=topology(),
        tools={p: sha(p) for p in (hot.CXX, inputs.CC, inputs.CM, inputs.NJ, hot.READELF, hot.PRLIMIT, hot.PYTHON)},
        metadata=dict(path=str(inputs.METADATA), sha256=sha(inputs.METADATA)), formats=inputs.model_rows(),
        environment=env, native_environment=dict(env, STRATA_IQ3_PAIR='1', STRATA_IQ3_PAIR_AUDIT='1',
            STRATA_IQ2_PAIR='0', STRATA_IQ2_PAIR_AUDIT='0', STRATA_IQ_SINGLE_EXACT='0',
            STRATA_IQ_MT_MIN='2', STRATA_CPU_SERVICE='0', STRATA_CPU_WORK_CENSUS='0'),
        configure=[inputs.CM, '-S', str(P / 'bench/sycl/cpu_compact_hot'), '-B', str(ROOT / 'build'),
            '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
            '-DCMAKE_C_COMPILER=' + inputs.CC, '-DCMAKE_CXX_COMPILER=' + hot.CXX,
            '-DCMAKE_CXX_FLAGS=' + flags, '-DCMAKE_EXE_LINKER_FLAGS=-Wl,-t', '-DSTRATA_GGML_DIR=' + ggml['root']],
        build=[inputs.CM, '--build', str(ROOT / 'build'), '--target', 'compact_cache_pool', '--parallel', '3'],
        expected=dict(compiled_objects=36, exported_commands=39, unused_exports=sorted(ABSENT),
                      whole_pool_cases_per_format=48, fallback_nt1_rows=768000, compact_rows=376320),
        acceptance='Fresh compilation of all11project CPUobjects plus24GGMLobjects and the actual pool fixture;'
          'no seed object/archive reuse or mixed ExpertPool definitions. Exact source/tool/SDK/generated/link/ELF'
          'admission,CPU-only libraries,quiet normal0/closed owners. Both real Down20/42 formats must pass48'
          'whole-pool cases:1worker and5workers+host,Decode/Prefill,1/6/97/193jobs,NT1/NT2/mixed and'
          'one-third misses. Bit equality after GU,hidden quantization and Down;actual native paired counters'
          'prove selected compact/fallback rows. Retired cache owner retains leased GU while raw Down stays.'
          'Budget/source/duplicate/short-span refusals. No speed,driver/model,TTFT/RSS/physical262144 adoption.',
        stopping='Single-use build/check. Stop on missing/changed source,tool,model,topology,coverage,generated'
          'header,ELF,owner,budget,dispatch or numeric gate;preserve failure. No hash refresh or retry.',
        budgets=dict(configure_wall=90, build_wall=240, check_wall=120, cpu=900,
                     rss=3 << 30, output=2 << 20, total=384 << 20),
        retention='Keep the current coherent engine and source-qualified old CPUgeneration as fallbacks.'
          'This fresh CPUgeneration serves the immediate production-pool numerical/performance comparison.'
          'Current old CPUreader/legacyGPUholds remain;retire ended auxiliary products after successor'
          'qualification and source recovery. Normal output is temporary;keep compact useful case results.')
    ROOT.mkdir(mode=0o700); (ROOT / 'tmp').mkdir(); write(PLAN, plan)
    print('Prepared fresh production-pool source/recipe;commit plan before build/check.', flush=True)


def verify(plan, models=True):
    need(hot.git('show', 'HEAD:' + str(PLAN.relative_to(P))) == PLAN.read_bytes(), 'commit exact pool plan')
    for name, expected in plan['sources'].items():
        raw = hot.git('show', plan['source_commit'] + ':' + name)
        need(hashlib.sha256(raw).hexdigest() == expected == sha(P / name), 'source drift: ' + name)
    need(all(sha(p) == h for p, h in plan['tools'].items()), 'tool drift')
    for name in ('native_recipe', 'compiler_inputs', 'external_inputs', 'metadata'):
        need(sha(plan[name]['path']) == plan[name]['sha256'], name + ' drift')
    census = json.loads(Path(plan['compiler_inputs']['path']).read_text())
    sdk = {p: h for p, h in (dict(census['compiled']) | dict(census['links'])).items() if p.startswith(('/usr/', '/opt/'))}
    need(all(sha(p) == h for p, h in sdk.items()), 'SDK input drift')
    ggml = plan['GGML']; root = Path(ggml['root'])
    for name, expected in ggml['source_sha256'].items():
        raw = subprocess.check_output(['git', 'show', ggml['revision'] + ':' + name], cwd=root, timeout=30)
        need(hashlib.sha256(raw).hexdigest() == expected == sha(root / name), 'GGML source drift')
    headers = std_headers()
    need(len(headers) == plan['standard_headers']['count'] and signature(headers) == plan['standard_headers']['sha256'], 'standard header drift')
    need(inputs.cmake_snapshot() == plan['cmake_inputs'] and topology() == plan['topology'], 'CMake/topology drift')
    if models:
        for row in plan['formats']:
            need(inputs.identity(row['shard']) == row['identity'] and inputs.fixture_hash(row) == row['sha256'], 'model slice drift')


def dependencies(plan):
    build = ROOT / 'build'
    seed = json.loads(Path(plan['compiler_inputs']['path']).read_text())
    known = dict(seed['compiled']) | dict(seed['links']) | std_headers()
    known.update({str(P / p): h for p, h in plan['sources'].items()})
    known.update({str(Path(plan['GGML']['root']) / p): h for p, h in plan['GGML']['source_sha256'].items()})
    known.update({str(build / p): h for p, h in plan['generated_inputs'].items()})
    known = {str(Path(p).resolve()): h for p, h in known.items()}
    actual, objects = {}, {}; current, expected, seen = None, 0, 0
    for line in (ROOT / 'owned/build-dependencies.stdout').read_text().splitlines() + ['']:
        if line.startswith('    '):
            need(current is not None, 'orphan dependency')
            p = Path(line.strip()); p = p if p.is_absolute() else build / p
            name = str(p.resolve()); need(name in known and sha(p) == known[name], 'uncovered compiler input: ' + name)
            actual[name] = known[name]; seen += 1
        elif line:
            if current is not None: need(seen == expected, 'incomplete dependency block')
            m = re.fullmatch(r'(.+\.o): #deps (\d+), deps mtime (\d+) \(VALID\)', line)
            need(m is not None, 'stale/invalid dependency block')
            current, expected, seen = m[1], int(m[2]), 0; p = build / current
            need(current not in objects and p.is_file() and p.stat().st_mtime_ns == int(m[3]), 'object continuity')
            objects[current] = sha(p)
        else:
            if current is not None: need(seen == expected, 'truncated dependency block')
            current = None
    commands = json.loads((build / 'compile_commands.json').read_text())
    absent, matched = set(), set()
    for c in commands:
        output = c['output']
        if output in objects:
            need(output not in matched and str(Path(c['file']).resolve()) in actual, 'source/command coverage')
            matched.add(output)
        else:
            need(output in ABSENT and not (build / output).exists(), 'unexpected unbuilt target')
            absent.add(output)
    need(len(objects) == plan['expected']['compiled_objects'] and len(commands) == plan['expected']['exported_commands']
         and matched == set(objects) and absent == ABSENT, 'actual compilation coverage')
    return dict(compiled=actual, objects=objects)


def execute(action):
    with (ROOT / (action + '-attempt.json')).open('x') as f: json.dump(dict(action=action, no_retry=True), f)
    state = dict(stage=action, passed=False, active=True, complete=False, GPU_executed=False, commands=[])
    def persist(): write(ROOT / (action + '.json'), state)
    persist()
    try:
        plan = json.loads(PLAN.read_text()); verify(plan)
        scope = hot.owner.Owner(ROOT / 'owned', persist); state['commands'] = scope.commands
        state.update(source_commit=plan['source_commit'], plan_sha256=sha(PLAN))
        def run(label, argv, wall, native=False):
            entry, out, err = scope.run(action + '-' + label, [hot.PRLIMIT, '--as=8589934592', '--', *argv],
                plan['native_environment'] if native else plan['environment'], P, wall=wall,
                text_cap=plan['budgets']['output'], rss_cap=plan['budgets']['rss'],
                total_cap=plan['budgets']['total'], file_cap=plan['budgets']['total'], cpu=plan['budgets']['cpu'])
            need(hot.owner.closed(entry) and entry['exit_code'] == 0, 'failed/unclosed ' + label)
            return out.read_text(), err.read_text()
        if action == 'build':
            run('configure', plan['configure'], plan['budgets']['configure_wall'])
            for name, expected in plan['generated_inputs'].items(): need(sha(ROOT / 'build' / name) == expected, 'generated header drift')
            text, _ = run('compile', plan['build'], plan['budgets']['build_wall'])
            run('dependencies', [inputs.NJ, '-C', str(ROOT / 'build'), '-t', 'deps'], 20)
            actual = dependencies(plan)
            allowed = {str(Path(p).resolve()): h for p, h in json.loads(Path(plan['external_inputs']['path']).read_text()).items()}
            links = {}
            for line in text.splitlines():
                p = Path(line.strip())
                if not p.is_absolute() or not p.is_file() or p.is_relative_to(ROOT): continue
                name = str(p.resolve()); need(name in allowed and sha(p) == allowed[name], 'uncovered external link: ' + name)
                links[name] = allowed[name]
            need(len(links) == 22, 'external link coverage')
            text, error = run('libraries', [hot.READELF, '--dynamic', str(ROOT / 'build/compact_cache_pool')], 10)
            libraries = re.findall(r'Shared library: \[([^\]]+)\]', text)
            need(not error and set(libraries) == {'libm.so.6', 'libstdc++.so.6', 'libgcc_s.so.1', 'libc.so.6'}, 'CPU-only library admission')
            state.update(actual_inputs=actual, links=links, libraries=libraries,
                         binary=dict(path=str(ROOT / 'build/compact_cache_pool'), sha256=sha(ROOT / 'build/compact_cache_pool')))
        else:
            build = json.loads((ROOT / 'build.json').read_text())
            need(build['passed'] and build['ownership_closed'] and not build['active'] and
                build['plan_sha256'] == sha(PLAN) and dependencies(plan) == build['actual_inputs'] and
                sha(build['binary']['path']) == build['binary']['sha256'], 'exact qualified build required')
            results = []
            for row in plan['formats']:
                text, error = run('down' + str(row['down']), [build['binary']['path'], row['shard'], str(row['down']),
                    *map(str, row['offsets']), str(row['blob_bytes'])], plan['budgets']['check_wall'], True)
                result = json.loads(text)
                need(not error and result == dict(passed=True, down=row['down'], whole_pool_cases=48,
                     fallback_nt1_rows=768000, compact_rows=376320, performance_qualified=False), 'pool parity/actual dispatch differs')
                results.append(result)
            state.update(results=results, binary=build['binary'], pool_numerical_qualified=True, performance_qualified=False)
        verify(plan); state.update(passed=True, complete=True, ownership_closed=True)
    except BaseException as e:
        state.update(error=repr(e), ownership_closed=all(hot.owner.closed(c) for c in state['commands'])); raise
    finally:
        state['active'] = False; persist()


if __name__ == '__main__':
    need(sys.executable == hot.PYTHON and len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'build', 'check'), 'qualified Python;prepare|build|check')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); hot.guards.no_model_owners([])
        prepare() if sys.argv[1] == 'prepare' else execute(sys.argv[1])
