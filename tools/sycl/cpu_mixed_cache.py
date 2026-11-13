#!/usr/bin/env python3
"""Owned mixed-NT cache trial; repository LICENSE. No model/GPU admission.

Fresh CPU/GGML compilation and actual production pool entry in both arms.
The prior failed whole-scheduling outcomes supply no performance admission.
"""
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
import cpu_compact_pool as base

P = Path(__file__).resolve().parents[2]
B, hot, inputs = base.B, base.hot, base.inputs
base.P = hot.P = inputs.P = P
ROOT, PLAN = B / 'cpu-mixed-cache-v1', P / 'bench/sycl/cpu_mixed_cache/plan.json'
RESULT, KEY = P / 'bench/results/2026-10-15-pr6-recovery-audit/cpu-roofline-current.json', 'mixed_nt_cache_v1'
inputs.RESULT = RESULT
THIS = 'tools/sycl/cpu_mixed_cache.py'
ORDER = ['reference', 'compact', 'compact', 'reference', 'reference', 'compact']
base.ROOT, base.PLAN, base.ORDER = ROOT, PLAN, ORDER
need, sha, write = base.need, base.sha, base.write
PREFIXES = ('src/kernels/cpu', 'include/strata/kernels/cpu', 'third_party/ggml',
    'include/strata/core/progress.hpp', 'bench/sycl/cpu_native_schedule/CMakeLists.txt',
    'bench/sycl/cpu_mixed_cache/pool.cpp', THIS, 'tools/sycl/cpu_compact_pool.py',
    'tools/sycl/cpu_compact_numeric.py', 'tools/sycl/cpu_compact_hot.py',
    'tools/sycl/measurement_owner.py', 'tools/sycl/measurement_guards.py',
    'AGENTS.md', 'docs/ARTIFACT_RETENTION.md')


def fixture_hash(row):
    before = inputs.identity(row['shard']); digest = hashlib.sha256()
    with Path(row['shard']).open('rb', buffering=0) as stream:
        for expert in range(64):
            for start, count in zip(row['offsets'], row['extents']):
                pos = start + expert * count
                need(0 <= pos <= before['bytes'] and count <= before['bytes'] - pos, '64expert bounds')
                stream.seek(pos); raw = stream.read(count)
                need(len(raw) == count, '64expert short read'); digest.update(raw)
    need(inputs.identity(row['shard']) == before, 'model changed during hash')
    return digest.hexdigest()


inputs.fixture_hash = fixture_hash


def qualification():
    # Scope the reader to immutable decision fields, not the file or outcome row.
    fields = ('origin', 'cost_evidence', 'policy', 'representative_cases', 'prospective_comparison')
    def view(value):
        return {k: value[k] for k in fields}
    original = json.loads(hot.git('show', '5fe9aae5c485a0da2059b96cfb966bf8e89978ae:' +
        str(RESULT.relative_to(P))))[KEY]
    committed = json.loads(hot.git('show', 'HEAD:' + str(RESULT.relative_to(P))))[KEY]
    actual = json.loads(RESULT.read_text())[KEY]
    encode = lambda v: json.dumps(view(v), sort_keys=True, separators=(',', ':'), allow_nan=False)
    need(encode(original) == encode(committed) == encode(actual), 'changed prospective decision fields')
    return dict(source_commit='5fe9aae5c485a0da2059b96cfb966bf8e89978ae',
        consumed_fields_sha256=hashlib.sha256(encode(original).encode()).hexdigest(),
        scope='Only original decision fields are consumed;outcomes and unrelated rows may change.')


def expected_numeric(down):
    counts = [[7,0,0,0], [20,4,0,1], [33,0,0,0]] if down == 20 else [
        [9,0,0,0], [7,2,2,2], [31,2,0,0]]
    shapes = [[t + 1 for t, n in enumerate(c) for _ in range(n)] for c in counts]
    shapes += [[1] * n for n in (1, 3, 6)]
    shapes += [[1] * 6 + [nt] for nt in (5, 6, 7, 8)]
    shapes += [[1] + [4] * 12, [1] * 41, [1] * 40, [2] * 9]
    cases = rows = selected = 0
    for workers in (1, 5):
        for policy, corpus in [('mixed', shapes), ('small', [[1] * n for n in (1,3,6,7,33)])]:
            for shape in corpus:
                for decode in (True, False):
                    for misses in (False, True):
                        round_ = cases + 11
                        eligible = decode and (len(shape) <= 6 if policy == 'small' else
                            len(shape) <= 40 and sum(shape) <= 40 and max(shape) <= 4)
                        for j in range(len(shape)):
                            nt = shape[(17 * j + (round_ & 3)) % len(shape)]
                            if nt == 1:
                                rows += 640
                                if eligible and (not misses or (j + round_) % 5):
                                    selected += 640
                        cases += 1
    return dict(passed=True, cases=cases, nt1_rows=rows, compact_rows=selected,
        special_controls=14, whole_calls=0, inputs_immutable=True, guards=True,
        performance_qualified=False)


def prepare():
    need(not ROOT.exists() and not PLAN.exists(), 'fresh plan/root required')
    basis = qualification(); revision = hot.git('rev-parse', 'HEAD').decode().strip()
    paths = hot.git('ls-tree', '-r', '--name-only', revision, '--', *PREFIXES).decode().splitlines()
    sources = {}
    for name in paths:
        raw = hot.git('show', revision + ':' + name)
        need(raw == (P / name).read_bytes(), 'commit exact input: ' + name)
        sources[name] = hashlib.sha256(raw).hexdigest()
    need(THIS in sources and 'bench/sycl/cpu_mixed_cache/pool.cpp' in sources, 'complete source scope')
    native_path = B / 'coherent-prefill-native-v1/record.json'
    native = json.loads(native_path.read_text())
    need(native['ownership_closed'] and not native['active'] and
        native['native_source_commit'] == '14508acbf1b70eb994fdf37f97fb6d0f5cff7d43', 'closed coherent recipe')
    ggml, = [d for d in native['dependencies'] if d['revision'] == '3cf03257f219afbe7334045ff7c6a06ac68c627d']
    recipe = dict(path=str(native_path), sha256=sha(native_path))
    census, external = native['actual_inputs'], native['external_inputs']
    need(sha(census['path']) == census['sha256'] and sha(external['path']) == external['sha256'], 'SDK census')
    flags, = [x.split('=', 1)[1] for x in native['recipe']['configure'] if x.startswith('-DCMAKE_CXX_FLAGS=')]
    flags = flags.replace('-fsycl-id-queries-range=size_t', '').strip()
    old_root = B / 'cpu-compact-small-batch-pool-v1'
    old, checked = [json.loads((old_root / name).read_text()) for name in ('build.json', 'check.json')]
    need(all(r['passed'] and r['complete'] and r['ownership_closed'] and not r['active']
        and all(hot.owner.closed(c) for c in r['commands']) for r in (old, checked)), 'closed CPUfallback')
    fallback = old['binaries']['pool']
    need(checked['binaries'] == old['binaries'] and sha(fallback['path']) == fallback['sha256'], 'fallback continuity')
    for name in ('src/kernels/cpu/compact_gu.cpp', 'include/strata/kernels/cpu/detail/compact_iq3xxs.hpp'):
        need((P / name).read_bytes() == hot.git('show', old['source_commit'] + ':' + name), 'unchanged owned codec')
    env = dict(PATH='/opt/intel/oneapi/compiler/2026.2/bin:/usr/bin:/bin',
        LD_LIBRARY_PATH='/opt/intel/oneapi/compiler/2026.2/lib:/usr/lib/x86_64-linux-gnu',
        LANG='C.UTF-8', LC_ALL='C.UTF-8', TMPDIR=str(ROOT / 'tmp'))
    child = dict(env, STRATA_IQ3_PAIR='1', STRATA_IQ3_PAIR_AUDIT='1', STRATA_IQ2_PAIR='0',
        STRATA_IQ2_PAIR_AUDIT='0', STRATA_IQ_SINGLE_EXACT='0', STRATA_IQ_MT_MIN='2',
        STRATA_CPU_SERVICE='0', STRATA_CPU_WORK_CENSUS='0')
    headers = base.std_headers(); formats = inputs.model_rows()
    for row in formats:
        row.update(sha256=fixture_hash(row), fixture_experts=64)
    plan = dict(schema='mixed-nt-cache-v1', qualification=basis, source_commit=revision, sources=sources,
        native_recipe=recipe, compiler_inputs=census, external_inputs=external,
        generated_inputs=inputs.coherent_generated(recipe, census), GGML=ggml,
        cmake_inputs=inputs.cmake_snapshot(), standard_headers=dict(count=len(headers), sha256=base.signature(headers)),
        extra_headers={x: sha(x) for x in base.EXTRA_HEADERS}, topology=base.topology(),
        tools={x: sha(x) for x in (hot.CXX, inputs.CC, inputs.CM, inputs.NJ, hot.READELF, hot.PRLIMIT, hot.PYTHON)},
        metadata=dict(path=str(inputs.METADATA), sha256=sha(inputs.METADATA)), formats=formats,
        environment=env, native_environment=child, speed_environment=dict(child, STRATA_IQ3_PAIR_AUDIT='0'),
        qualified_fallbacks=dict(CPU_binary=fallback, source_commit=old['source_commit'],
            scope='Original closed CPUauxiliary retained;no project object/archive reuse.'),
        configure=[inputs.CM, '-S', str(P / 'bench/sycl/cpu_native_schedule'), '-B', str(ROOT / 'build'),
            '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
            '-DCMAKE_C_COMPILER=' + inputs.CC, '-DCMAKE_CXX_COMPILER=' + hot.CXX,
            '-DCMAKE_CXX_FLAGS=' + flags, '-DCMAKE_EXE_LINKER_FLAGS=-Wl,-t',
            '-DSTRATA_GGML_DIR=' + ggml['root'],
            '-DSTRATA_CPU_NATIVE_FIXTURE=' + str(P / 'bench/sycl/cpu_mixed_cache/pool.cpp')],
        build=[inputs.CM, '--build', str(ROOT / 'build'), '--target', 'cpu_native_schedule', '--parallel', '3'],
        binary_targets=dict(pool='cpu_native_schedule'), expected=dict(compiled_objects=36, exported_commands=38),
        expected_ELF=dict(elf_class='ELF64', data="2's complement, little endian", machine='Advanced Micro Devices X86-64',
            elf_type='EXEC (Executable file)', interpreter='/lib64/ld-linux-x86-64.so.2'),
        numeric={str(d): expected_numeric(d) for d in (20, 42)}, order=ORDER, pairs=[[0,1],[3,2],[4,5]],
        comparison=copy.deepcopy(json.loads(RESULT.read_text())[KEY]['prospective_comparison']),
        cases=copy.deepcopy(json.loads(RESULT.read_text())[KEY]['representative_cases']),
        scope='Fresh11CPU+24GGML+onefixture;all affected ExpertPool class consumers coherent. Actual production '
            'entry,whole_decode=false. MixedNt is explicit only in the fixture;no core/generate/model adoption.',
        numeric_scope='152regular/legacy cases per Downformat,14special controls,64actual experts,4activations. '
            'Actual NT1native audit conserves selected cache rows;NT2..4exact independent phase/NToracle. '
            'NT5..8fallback compares native outputs directly. Forced valid leases exercise phase/volume/NT/invalid '
            'policy guards;misses,span,null,retired owner,wrong-expert and immutable source/output canaries.',
        timing='Twelve fresh processes interleaved by Down20/42in each RCCRRCarm. Three observed shapes per format '
            'times reuse/stream64,5workers+host.64round complete input cycles,calibrate>=0.5s,3batches>=0.25s, '
            'measured observer estimate<0.1percent. Include bindings,policy scan,GU/quant/Down and waits;'
            'oracle/checks/preparation outside timer,setup/RSS reported. Timed producer binding counts are not '
            'kernel counters;actual leaf selection is audited numerically and immutable valid cases cannot reroute.',
        stopping='Single-use build/check/measure0..5;stop first input,source,build,ELF,numeric,route,owner,budget or '
            'batch failure. No retry/hash refresh;component passing is not model or target qualification.',
        budgets=dict(configure_wall=90, build_wall=300, check_wall=300, measure_wall=300,
            cpu=2400, rss=3 << 30, output=2 << 20, total=384 << 20),
        retention='One imminent CPUcomparison fixture;keep original CPUfallback until numerical qualification. '
            'Commit useful individual samples,then retire routine output and ended fixtures after source recovery.')
    ROOT.mkdir(mode=0o700); (ROOT / 'tmp').mkdir(); write(PLAN, plan)
    print('Prepared coherent mixed-cache source/recipe/corpus;commit before build.', flush=True)


def verify(plan):
    need(plan['order'] == ORDER and plan['qualification'] == qualification(), 'unchanged scoped qualification')
    base.verify(plan)


def execute(action):
    if action == 'build':
        original = base.verify
        base.verify = lambda plan, models=True: (need(plan['qualification'] == qualification(), 'qualification'), original(plan, models))
        base.execute('build'); return
    with (ROOT / (action + '-attempt.json')).open('x') as stream:
        json.dump(dict(action=action, no_retry=True), stream)
    state = dict(stage=action, active=True, passed=False, complete=False, commands=[], GPU_executed=False)
    def persist():
        write(ROOT / (action + '.json'), state)
    persist()
    try:
        plan = json.loads(PLAN.read_text()); verify(plan)
        build = base.qualified_build(plan); state.update(source_commit=plan['source_commit'], plan_sha256=sha(PLAN))
        if action != 'check':
            checked = json.loads((ROOT / 'check.json').read_text())
            need(base.closed(checked, plan) and checked['binaries'] == build['binaries'], 'numeric gate first')
            index = int(action.split('-')[1])
            for i in range(6):
                if i < index:
                    need(base.closed(json.loads((ROOT / f'measure-{i}.json').read_text()), plan), 'previous arm')
                elif i > index:
                    need(not (ROOT / f'measure-{i}-attempt.json').exists(), 'later arm already attempted')
        scope = hot.owner.Owner(ROOT / 'owned', persist); state['commands'] = scope.commands; results = []
        for row in plan['formats']:
            arm = 'check' if action == 'check' else ORDER[int(action.split('-')[1])]
            argv = [build['binaries']['pool']['path'], row['shard'], str(row['down']),
                *map(str, row['offsets']), str(row['blob_bytes']), arm]
            entry, out, err = scope.run(action + '-down' + str(row['down']),
                [hot.PRLIMIT, '--as=8589934592', '--', *argv],
                plan['native_environment'] if action == 'check' else plan['speed_environment'], P,
                wall=plan['budgets']['check_wall' if action == 'check' else 'measure_wall'],
                cpu=plan['budgets']['cpu'], rss_cap=plan['budgets']['rss'], text_cap=plan['budgets']['output'],
                total_cap=plan['budgets']['total'], file_cap=plan['budgets']['total'])
            need(hot.owner.closed(entry) and entry['exit_code'] == 0 and not err.read_text(), 'native child/owner failed')
            value = json.loads(out.read_text())
            if action == 'check':
                need(value == plan['numeric'][str(row['down'])], 'numeric output/counts/actual route differ')
            else:
                need(value['scope'] == 'CPU_mixed_NT_cache' and value['arm'] == arm and value['down'] == row['down']
                    and value['passed'] and value['inputs_immutable'] and not value['model_qualified'], 'timed envelope')
                expected = {(c['jobs'], tuple(c['NT_counts']), pattern) for c in plan['cases'] if c['down'] == row['down']
                    for pattern in ('reuse', 'stream64')}; seen = set()
                for cell in value['cells']:
                    key = (cell['jobs_per_call'], tuple(cell['NT_counts']), cell['pattern'])
                    need(key not in seen and cell['workers'] == 5 and cell['host_works'], 'cell/worker binding'); seen.add(key)
                    rows = sum((i + 1) * n for i, n in enumerate(cell['NT_counts']))
                    need(cell['rows_per_call'] == rows <= 40 and sum(cell['NT_counts']) == cell['jobs_per_call']
                        and cell['rounds'] >= 64 and cell['rounds'] % 64 == 0 and len(cell['samples']) == 3
                        and cell['calibration'][-1]['wall_s'] >= .5
                        and cell['calibration'][-1]['rounds'] == cell['rounds'], 'physical/calibration volume')
                    for s in cell['samples']:
                        need(s['wall_s'] >= .25 and 0 < value['observer_estimate_s'] / s['wall_s'] < .001
                            and s['calls'] == cell['rounds'] and s['jobs'] == s['calls'] * cell['jobs_per_call']
                            and s['token_expert_rows'] == s['calls'] * rows
                            and s['bound_NT1_leases'] == (s['calls'] * cell['NT_counts'][0] if arm == 'compact' else 0)
                            and 0 <= sum(s[k] for k in ('pool_gu_s','pool_quant_s','pool_down_s')) <= s['wall_s'] + 1e-6,
                            'sample duration/work/producer bindings/stage clocks')
                    cell['median_seconds_per_call'] = statistics.median(s['wall_s'] / s['calls'] for s in cell['samples'])
                need(seen == expected, 'exact per-format corpus')
            results.append(dict(down=row['down'], result=value))
        verify(plan)
        state.update(results=results, binaries=build['binaries'], passed=True, complete=True,
            ownership_closed=True, numerical_qualified=action == 'check', performance_qualified=False)
    except BaseException as error:
        state.update(error=repr(error), ownership_closed=all(hot.owner.closed(c) for c in state['commands']))
        raise
    finally:
        state['active'] = False; persist()


if __name__ == '__main__':
    need(sys.executable == hot.PYTHON and len(sys.argv) == 2 and sys.argv[1] in
        ['prepare', 'build', 'check', *[f'measure-{i}' for i in range(6)]], 'qualified Python/action')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        hot.guards.no_model_owners([])
        prepare() if sys.argv[1] == 'prepare' else execute(sys.argv[1])
