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
ROOT = B / 'cpu-compact-small-batch-pool-v1'
PLAN = P / 'bench/sycl/cpu_compact_hot/pool-small-plan-v1.json'
THIS = 'tools/sycl/cpu_compact_pool.py'
PREFIXES = ('src/kernels/cpu', 'include/strata/kernels/cpu', 'third_party/ggml',
    'include/strata/core/progress.hpp', 'bench/sycl/cpu_compact_hot/CMakeLists.txt',
    'bench/sycl/cpu_compact_hot/pool_timing.cpp', 'bench/sycl/cpu_compact_hot/numeric.cpp',
    'bench/sycl/cpu_compact_hot/cache.hpp', 'AGENTS.md', 'docs/ARTIFACT_RETENTION.md',
    THIS, 'tools/sycl/cpu_compact_hot.py', 'tools/sycl/cpu_compact_numeric.py',
    'tools/sycl/measurement_owner.py', 'tools/sycl/measurement_guards.py')
need, sha, write = hot.need, hot.sha, hot.write
ORDER = ['reference', 'compact', 'compact', 'reference', 'reference', 'compact']
EXTRA_HEADERS = ('/usr/include/x86_64-linux-gnu/sys/resource.h',
    '/usr/include/x86_64-linux-gnu/bits/resource.h',
    '/usr/include/x86_64-linux-gnu/bits/types/struct_rusage.h')
ABSENT = {'ggml/src/CMakeFiles/ggml.dir/ggml-backend-dl.cpp.o',
          'ggml/src/CMakeFiles/ggml.dir/ggml-backend-reg.cpp.o',
          }


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


def expected_pool():
    cases = rows = selected = 0
    for workers in (1, 5):
        for n in (1, 2, 3, 4, 5, 6, 7, 32, 97, 193):
            for policy in (0, 1, 2):
                for decode in (True, False):
                    for misses in (False, True):
                        round_number = cases + 11
                        for j in range(n):
                            nt = 1 if policy == 0 else 2 if policy == 1 else 1 + (j & 1)
                            if nt == 1:
                                rows += 640
                                if n <= 6 and decode and (not misses or (j + j // 17 + round_number) % 5 != 0):
                                    selected += 640
                        cases += 1
    return dict(passed=True, cases=cases, nt1_rows=rows, compact_rows=selected,
        wrong_expert_control_detected=True, forced_policy_controls=60,
        span_controls=2, retirement_controls=2, q8_range_checked=True, source_budget_controls=True)


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
    flags, = [v.split('=', 1)[1] for v in native['recipe']['configure'] if v.startswith('-DCMAKE_CXX_FLAGS=')]
    flags = flags.replace('-fsycl-id-queries-range=size_t', '').strip()
    env = dict(PATH='/opt/intel/oneapi/compiler/2026.2/bin:/usr/bin:/bin',
        LD_LIBRARY_PATH='/opt/intel/oneapi/compiler/2026.2/lib:/usr/lib/x86_64-linux-gnu',
        LANG='C.UTF-8', LC_ALL='C.UTF-8', TMPDIR=str(ROOT / 'tmp'))
    headers = std_headers(); census = native['actual_inputs']; external = native['external_inputs']
    need(sha(census['path']) == census['sha256'] and sha(external['path']) == external['sha256'], 'SDK census drift')
    old_root = B / 'cpu-compact-production-pool-v1'
    old_build = json.loads((old_root / 'build.json').read_text())
    old_check = json.loads((old_root / 'check.json').read_text())
    for v in (old_build, old_check):
        need(v['passed'] and v['complete'] and v['ownership_closed'] and not v['active']
             and all(hot.owner.closed(c) for c in v['commands']), 'qualified CPUfallback closure')
    need(old_build['binary'] == old_check['binary'] and
         sha(old_build['binary']['path']) == old_build['binary']['sha256'], 'retained CPUfallback continuity')
    plan = dict(schema='compact-small-batch-pool-v1', source_commit=revision, sources=sources,
        qualified_fallbacks=dict(engine_source=native['native_source_commit'],
            engine_sha256='dcb8efd4698d4c9c6de913cb5f92e2229e945b242f24bd1fca2998e4cf89e6d1',
            CPU_source=old_build['source_commit'], CPU_binary=old_build['binary'],
            CPU_original_recipe_commit='e2fd9e722769dc11207d133f7e5ac3ef5430f050',
            scope='Original c61CPUfallback,not current-source qualification. Rebuild all new inputs;no borrowed project objects.'),
        native_recipe=recipe, compiler_inputs=census, external_inputs=external,
        generated_inputs=inputs.coherent_generated(recipe, census), GGML=ggml, cmake_inputs=inputs.cmake_snapshot(),
        standard_headers=dict(count=len(headers), sha256=signature(headers)),
        extra_headers={p: sha(p) for p in EXTRA_HEADERS}, topology=topology(),
        tools={p: sha(p) for p in (hot.CXX, inputs.CC, inputs.CM, inputs.NJ, hot.READELF, hot.PRLIMIT, hot.PYTHON)},
        metadata=dict(path=str(inputs.METADATA), sha256=sha(inputs.METADATA)), formats=inputs.model_rows(),
        environment=env, native_environment=dict(env, STRATA_IQ3_PAIR='1', STRATA_IQ3_PAIR_AUDIT='1',
            STRATA_IQ2_PAIR='0', STRATA_IQ2_PAIR_AUDIT='0', STRATA_IQ_SINGLE_EXACT='0',
            STRATA_IQ_MT_MIN='2', STRATA_CPU_SERVICE='0', STRATA_CPU_WORK_CENSUS='0'),
        speed_environment=dict(env, STRATA_IQ3_PAIR='1', STRATA_IQ3_PAIR_AUDIT='0',
            STRATA_IQ2_PAIR='0', STRATA_IQ2_PAIR_AUDIT='0', STRATA_IQ_SINGLE_EXACT='0',
            STRATA_IQ_MT_MIN='2', STRATA_CPU_SERVICE='0', STRATA_CPU_WORK_CENSUS='0'),
        configure=[inputs.CM, '-S', str(P / 'bench/sycl/cpu_compact_hot'), '-B', str(ROOT / 'build'),
            '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
            '-DCMAKE_C_COMPILER=' + inputs.CC, '-DCMAKE_CXX_COMPILER=' + hot.CXX,
            '-DCMAKE_CXX_FLAGS=' + flags, '-DCMAKE_EXE_LINKER_FLAGS=-Wl,-t', '-DSTRATA_GGML_DIR=' + ggml['root']],
        build=[inputs.CM, '--build', str(ROOT / 'build'), '--target', 'compact_cache_pool', 'compact_cache_numeric', '--parallel', '3'],
        expected=dict(compiled_objects=37, exported_commands=39, unused_exports=sorted(ABSENT),
            pool=expected_pool(), numeric=dict(component='compact_cache_numeric', passed=True, experts=32,
                serial_cases=384, threaded_rows=640, raw_calls=392, raw_nt1_rows=181376,
                raw_pair_rows=181248, performance_qualified=False)),
        order=ORDER, pairs=[[0,1],[3,2],[4,5]],
        expected_ELF=dict(elf_class='ELF64', data="2's complement, little endian",
            machine='Advanced Micro Devices X86-64', elf_type='EXEC (Executable file)',
            interpreter='/lib64/ld-linux-x86-64.so.2'),
        scope='Fresh coherent11CPU+24GGML+2fixture objects. The actual ExpertPool entry independently limits cached GU to complete Decode batches<=6jobs;per-job NT1/format guards remain. Timed fixture includes coordinates/lookup/lease handoff,GU,hidden quantization,Down and waits;actual router/core/generate wiring is not built or qualified. Original V1/V2whole-pool and138-byte failures stay failed;134-byte codec unchanged.',
        numerical='Two actual Down20/42 layers.240regular cases/layer across1worker or5workers+host,1..7/32/97/193jobs,NT1/NT2/mixed,Decode/Prefill and misses.60forced-valid-lease controls exercise pool batch/phase/NT guards independently of dispatcher omission. Wrong-expert6job control must use cached math and differ. Short span rejected before publication then valid call works;temporary Cache really destroyed while leased Entry storage survives. Actual Q8_K range checked. Separate384case/layer GU fixture,threaded rows,source/activation guards and alias refusals.',
        timing='Separate native processes per format/arm,in enforcedR,C,C,R,R,Corder.20cells/layer:1worker or5workers+host;1/3/6/7/32jobs;reuse/stream32. Calibrate powers2calls1..65536until>=0.5s;then3batches each>=0.25s,timer overhead<0.1%. Normalize seconds/expert. Preparation/oracle/topology/correctness outside timer. All-hit lookup checks remain timed;fully frozen format/phase/NT/valid bounds make selection deterministic. These checks record bindings,not hardware counters. Numerical actual native-row counters establish the same policy;no claim of per-kernel clean-run audit instrumentation.',
        acceptance='Numerical/ELF/source/owner gates first. Judge each cached1/3/6job cell independently:all3paired gains,without minimum percentage,qualify only that CPUshape. Blocked7/32jobs must prove native numerical dispatch even with forced leases;their clean samples are nonregression diagnostics,not required positive-gain cells. No global model or70TGadmission from this campaign;commit any later shape/model plan before new evidence. Before model adoption,independent core/generate build,allocation/RAM/TTFT,prefill preservation,model/task and physical262144/RESTORE gates remain.',
        stopping='Single-use build/check/measure-N. Stop on changed/missing source,SDK,model,generated input,ELF,object/link,budget,affinity,numeric/dispatch,order or batch eligibility;preserve failure,no hash refresh/retry.',
        budgets=dict(configure_wall=90, build_wall=240, check_wall=180, measure_wall=240,
            cpu=1800, rss=3 << 30, output=2 << 20, total=384 << 20),
        retention='Keep coherent GPUfallback and original qualified c61CPUgeneration before successor qualification. One fresh CPUgeneration serves this comparison;both fixtures share coherent library. After pushed compact results select current CPUgeneration,verify original c61source/recipe and retire ended old CPUroot. Do not retain routine output. Legacy GPU/IR/deployment holds remain separate.')
    ROOT.mkdir(mode=0o700); (ROOT / 'tmp').mkdir(); write(PLAN, plan)
    print('Prepared fresh small-batch pool plan;commit before build/check/measure.', flush=True)


def verify(plan, models=True):
    need(hot.git('show', 'HEAD:' + str(PLAN.relative_to(P))) == PLAN.read_bytes(), 'commit exact pool plan')
    for name, expected in plan['sources'].items():
        raw = hot.git('show', plan['source_commit'] + ':' + name)
        need(hashlib.sha256(raw).hexdigest() == expected == sha(P / name), 'source drift: ' + name)
    need(all(sha(p) == h for p, h in plan['tools'].items()), 'tool drift')
    need(all(sha(p) == h for p, h in plan['extra_headers'].items()) and plan['order'] == ORDER, 'OS header/order drift')
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
    topo = topology()
    need(len({(topo[str(i)]['physical_package_id'], topo[str(i)]['core_id']) for i in range(6)}) == 6, 'six distinct physical cores0..5')
    need(inputs.cmake_snapshot() == plan['cmake_inputs'] and topo == plan['topology'], 'CMake/topology drift')
    if models:
        for row in plan['formats']:
            need(inputs.identity(row['shard']) == row['identity'] and inputs.fixture_hash(row) == row['sha256'], 'model slice drift')


def dependencies(plan):
    build = ROOT / 'build'
    seed = json.loads(Path(plan['compiler_inputs']['path']).read_text())
    known = dict(seed['compiled']) | dict(seed['links']) | std_headers() | plan['extra_headers']
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


def closed(record, plan):
    return record['passed'] and record['complete'] and record['ownership_closed'] and not record['active'] and \
        record['source_commit'] == plan['source_commit'] and record['plan_sha256'] == sha(PLAN) and \
        all(hot.owner.closed(c) for c in record['commands'])


def qualified_build(plan):
    build = json.loads((ROOT / 'build.json').read_text())
    need(closed(build, plan) and dependencies(plan) == build['actual_inputs'] and
         all(sha(b['path']) == b['sha256'] for b in build['binaries'].values()) and
         all(sha(p) == h for p, h in build['links'].items()), 'exact coherent build/executables required')
    return build


def execute(action):
    with (ROOT / (action + '-attempt.json')).open('x') as f: json.dump(dict(action=action, no_retry=True), f)
    state = dict(stage=action, passed=False, active=True, complete=False, GPU_executed=False, commands=[])
    def persist(): write(ROOT / (action + '.json'), state)
    persist()
    try:
        plan = json.loads(PLAN.read_text()); verify(plan)
        scope = hot.owner.Owner(ROOT / 'owned', persist); state['commands'] = scope.commands
        state.update(source_commit=plan['source_commit'], plan_sha256=sha(PLAN))
        def run(label, argv, wall, environment=None):
            entry, out, err = scope.run(action + '-' + label, [hot.PRLIMIT, '--as=8589934592', '--', *argv],
                plan['environment'] if environment is None else environment, P, wall=wall,
                text_cap=plan['budgets']['output'], rss_cap=plan['budgets']['rss'],
                total_cap=plan['budgets']['total'], file_cap=plan['budgets']['total'], cpu=plan['budgets']['cpu'])
            need(hot.owner.closed(entry) and entry['exit_code'] == 0, 'failed/unclosed ' + label)
            return out.read_text(), err.read_text()
        if action == 'build':
            need(not (ROOT / 'build').exists(), 'no unrecorded/reused build tree')
            fallback = plan['qualified_fallbacks']['CPU_binary']
            need(sha(fallback['path']) == fallback['sha256'], 'retain CPUfallback before build')
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
            binaries, headers, all_libraries = {}, {}, {}
            for key in ('pool', 'numeric'):
                path = ROOT / ('build/compact_cache_' + key)
                text, error = run('ELF-' + key, [hot.READELF, '--file-header', '--program-headers', '--dynamic', str(path)], 10)
                libraries = re.findall(r'Shared library: \[([^\]]+)\]', text)
                need(not error and set(libraries) == {'libm.so.6', 'libstdc++.so.6', 'libgcc_s.so.1', 'libc.so.6'}, 'CPU-only libraries')
                def field(name):
                    values = re.findall(r'^\s*' + re.escape(name) + r':\s*(.+)$', text, re.MULTILINE)
                    need(len(values) == 1, 'ELF field'); return values[0].strip()
                interpreter = re.findall(r'\[Requesting program interpreter: ([^\]]+)\]', text)
                need(len(interpreter) == 1, 'ELF interpreter')
                elf = dict(elf_class=field('Class'), data=field('Data'), machine=field('Machine'),
                    elf_type=field('Type'), interpreter=interpreter[0])
                need(elf == plan['expected_ELF'], 'ELF architecture/type/interpreter')
                binaries[key] = dict(path=str(path), sha256=sha(path)); headers[key] = elf; all_libraries[key] = libraries
            state.update(actual_inputs=actual, links=links, libraries=all_libraries, ELF=headers, binaries=binaries)
        else:
            build = qualified_build(plan); results = []
            if action != 'check':
                checked = json.loads((ROOT / 'check.json').read_text())
                need(closed(checked, plan) and checked['binaries'] == build['binaries'], 'numerical pass first')
                index = int(action.split('-')[1])
                for j in range(6):
                    if j < index:
                        previous = json.loads((ROOT / f'measure-{j}.json').read_text())
                        need(closed(previous, plan) and previous['binaries'] == build['binaries'], 'previous arm failed/unclosed')
                    elif j > index:
                        need(not (ROOT / f'measure-{j}-attempt.json').exists(), 'later arm already invoked')
            for row in plan['formats']:
                args = [row['shard'], str(row['down']), *map(str, row['offsets']), str(row['blob_bytes'])]
                if action == 'check':
                    text, error = run('numeric-down' + str(row['down']), [build['binaries']['numeric']['path'], *args, '512'],
                        plan['budgets']['check_wall'], plan['native_environment'])
                    numeric = json.loads(text)
                    need(not error and -127 <= numeric['q8_min'] < 0 < numeric['q8_max'] <= 127, 'actual Q8_K range')
                    need({k:v for k,v in numeric.items() if k not in ('q8_min','q8_max')} ==
                         dict(plan['expected']['numeric'], down=row['down']), 'direct GU bits/counters/source/alias guard')
                    text, error = run('pool-down' + str(row['down']), [build['binaries']['pool']['path'], *args, 'check'],
                        plan['budgets']['check_wall'], plan['native_environment'])
                    value = json.loads(text)
                    need(not error and value == plan['expected']['pool'], 'pool bits/actual dispatch/controls differ')
                    results.append(dict(down=row['down'], numeric=numeric, pool=value))
                else:
                    arm = plan['order'][int(action.split('-')[1])]
                    text, error = run('down' + str(row['down']), [build['binaries']['pool']['path'], *args, arm],
                        plan['budgets']['measure_wall'], plan['speed_environment'])
                    value = json.loads(text)
                    need(not error and value['passed'] and value['down'] == row['down'] and value['arm'] == arm and
                         value['timing_envelope'] == 'small6_cal05_floor025' and len(value['cells']) == 20, 'timed arm/envelope')
                    seen = set()
                    for cell in value['cells']:
                        key = (cell['workers'], cell['jobs_per_call'], cell['pattern']); need(key not in seen, 'duplicate cell'); seen.add(key)
                        need(cell['host_cpu'] == 0 and cell['host_works'] == (cell['workers'] == 5) and
                             cell['worker_cpus'] == list(range(1, cell['workers'] + 1)), 'actual host/worker affinity')
                        need(cell['calibration'][-1]['wall_s'] >= .5 and cell['calibration'][-1]['rounds'] == cell['rounds'], 'calibration floor/count')
                        need(len(cell['samples']) == 3 and all(s['wall_s'] >= .25 and s['calls'] == cell['rounds']
                             and s['jobs'] == cell['rounds'] * cell['jobs_per_call'] and
                             0 <= s['pool_gu_s'] + s['pool_quant_s'] + s['pool_down_s'] <= s['wall_s'] + 1e-6 and
                             0 <= value['timer_pair_s'] / s['wall_s'] < .001 for s in cell['samples']), 'timing eligibility')
                        import statistics
                        cell['median_seconds_per_job'] = statistics.median(s['wall_s'] / s['jobs'] for s in cell['samples'])
                    need(seen == {(w,n,p) for w in (1,5) for n in (1,3,6,7,32) for p in ('reuse','stream32')}, 'timed shape coverage')
                    results.append(value)
            state.update(results=results, binaries=build['binaries'], numerical_qualified=action == 'check', performance_qualified=False)
        verify(plan); state.update(passed=True, complete=True, ownership_closed=True)
    except BaseException as e:
        state.update(error=repr(e), ownership_closed=all(hot.owner.closed(c) for c in state['commands'])); raise
    finally:
        state['active'] = False; persist()


if __name__ == '__main__':
    need(sys.executable == hot.PYTHON and len(sys.argv) == 2 and
         sys.argv[1] in ('prepare', 'build', 'check', *(f'measure-{i}' for i in range(6))), 'qualified Python;prepare|build|check|measure-0..5')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); hot.guards.no_model_owners([])
        prepare() if sys.argv[1] == 'prepare' else execute(sys.argv[1])
