#!/usr/bin/env python3
"""Main's independent mapped-row mechanism trial; repository LICENSE.

Fresh five-TU fixture, never an engine relink. Keep the old production entry
unchanged. Commit each plan/result before the next serialized stage.
"""
import copy
from collections import Counter
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys

sys.dont_write_bytecode = True
import build_engine as source
import build_upstream_features as build
import check_cache_loan_cli as probe
import engine_once
import measurement_guards as guards
import measurement_owner as ownership
import ur_validation as validation

P, B = source.P, source.B
ROOT = B / 'mapped-rows-subgroup-v5'
RESULT = P / 'bench/results/2026-10-15-pr6-recovery-audit/fused-moe-service-current.json'
KEY, TARGET = 'mapped_rows_subgroup_v5', 'sycl_mapped_rows_bench'
THIS = 'tools/sycl/mapped_rows_experiment.py'
PREFIXES = ['CMakeLists.txt', 'cmake', 'src', 'include', 'third_party', 'tests',
            'bench/sycl', 'tools/sycl', 'AGENTS.md', 'docs/ARTIFACT_RETENTION.md']
ORDER = ['scalar', 'subgroup', 'subgroup', 'scalar', 'scalar', 'subgroup']
TUS = {'bench/sycl/mapped_rows.cpp', 'src/kernels/sycl/elementwise.cpp',
       'src/sycl/runtime.cpp', 'src/sycl/compat.cpp', 'src/sycl/command_profile.cpp'}
require, sha, signature = guards.require, source.digest, build.signature


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def compact(row):
    value = {k: v for k, v in row.items()
             if k not in ('project', 'dependencies', 'runtime_files', 'commands', 'actual_inputs')}
    value['project'] = {k: v for k, v in row['project'].items() if k != 'source_sha256'}
    value['dependencies'] = [{k: v for k, v in p.items() if k != 'source_sha256'}
                             for p in row['dependencies']]
    value['runtime_files_sha256'] = signature(row['runtime_files'])
    if 'actual_inputs' in row:
        value['actual_inputs'] = {k: dict(files=len(v), sha256=signature(v)) if isinstance(v, dict) else v
                                 for k, v in row['actual_inputs'].items()}
    return value


def save(row):
    write(ROOT / 'record.json', row)
    data = json.loads(RESULT.read_text()); data[KEY] = compact(row); write(RESULT, data)


def check(row, *, committed=True):
    if committed:
        require(validation.committed(RESULT, (KEY,)) == compact(row), 'commit exact experiment plan/result')
    build.check_inputs(row)
    runtime = row['runtime_manifest']
    require(sha(runtime['path']) == runtime['sha256'] and
            json.loads(Path(runtime['path']).read_text()) == row['runtime_files'] and
            all(sha(p) == h for p, h in row['runtime_files'].items()), 'runtime bytes changed')
    actual_probe, reuse = probe.current_probe()
    require(actual_probe == row['native_validation_probe'] and reuse == row['runtime_reuse'], 'probe changed')
    require(guards.driver_guard() == row['driver'] and
            Path('/proc/sys/kernel/random/boot_id').read_text().strip() == row['boot'], 'boot/driver changed')
    if row.get('binary'):
        require(sha(row['binary']['path']) == row['binary']['sha256'], 'fixture changed')
        for group in ('compiled', 'generated', 'links', 'objects'):
            require(all(sha(p) == h for p, h in row['actual_inputs'][group].items()), 'actual input changed: ' + group)


def environment(row, label, traced=False):
    root = ROOT / label
    env = {k: v for k, v in row['recipe']['environment'].items()
           if k in probe.ENV_KEYS}
    env.update(TMPDIR=str(root / 'tmp'), XDG_CACHE_HOME=str(root / 'cache'),
        CUDA_MODULE_LOADING='EAGER', STRATA_CPU_SERVICE='0',
        UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION',
        UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(root / 'validation.log'),
        UR_LOG_LOADER='level:warning;flush:warning;output:stderr',
        UR_LOG_LEVEL_ZERO='level:warning;flush:warning;output:stderr')
    env['ZE_ENABLE_ALT_DRIVERS'] = row['driver_library']
    if traced:
        env.update(UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
                   UR_LOG_TRACING='level:info;flush:info;output:file,' + str(root / 'ur.trace'))
    return engine_once.native_environment(row, env)


def prepare():
    require(not ROOT.exists() and KEY not in validation.committed(RESULT), 'fresh plan only')
    prior = validation.committed(RESULT, ('current_decode_device_service_v1',))
    require(prior['stage'] == 'closed' and prior['complete'] and not prior['passed'] and
            prior['ownership_closed'] and not prior['active'], 'close failed observer gate first')
    original = B / 'cpu-mixed-cache-native-v2/record.json'
    native = json.loads(original.read_text())
    require(native['passed'] and native['ownership_closed'] and native['stage'] == 'built', 'coherent recipe required')
    recipe = copy.deepcopy(native['recipe'])
    old = str(original.parent)
    recipe['configure'] = [x.replace(old, str(ROOT)) for x in recipe['configure']]
    recipe['environment'] = {k: v.replace(old, str(ROOT)) for k, v in recipe['environment'].items()}
    recipe['targets'] = recipe['build_targets'] = [TARGET]
    fallback = dict(native['binaries'])
    clean = B / 'coherent-prefill-native-v1/build/strata'
    fallback['clean_prefill'] = dict(path=str(clean), bytes=clean.stat().st_size,
        sha256='dcb8efd4698d4c9c6de913cb5f92e2229e945b242f24bd1fca2998e4cf89e6d1')
    cohort = json.loads((B / 'cpu-mixed-cache-speed-v1/record.json').read_text())
    actual_probe, reuse = probe.current_probe()
    row = dict(stage='prepared', project=source.pin(P, PREFIXES),
        dependencies=native['dependencies'], toolchain=native['toolchain'] | {str(Path(sys.executable).resolve()): sha(sys.executable)},
        external_inputs=native['external_inputs'], fallbacks=fallback, recipe=recipe,
        recipe_origin=dict(path=str(original), sha256=sha(original),
            source_commit=native['project']['revision']),
        runtime_manifest=cohort['runtime_manifest'],
        runtime_files=json.loads(Path(cohort['runtime_manifest']['path']).read_text()),
        native_validation_probe=actual_probe, runtime_reuse=reuse,
        driver_library=cohort['environment_template']['ZE_ENABLE_ALT_DRIVERS'],
        boot=cohort['boot'], driver=cohort['driver'],
        plan=dict(question='Does subgroup-shared row-mask lookup reduce the owned mapped-copy critical interval?',
            origin='Main-authored from unchanged owned scalar copy;Khronos group collective contract. No XeStrata source/design.',
            evidence='d2016f7307fecac26aaec56aec665677a0ed6891:observed scalar kernel176.6/133.2ms over322/319calls under rejected observer envelope;hypothesis only,not saturation or clean model speed.',
            build='Fresh runtime3,elementwise1,fixture1 only;preserve13diagnostic ELFs and clean engine.',
            correctness='18captures/54dynamic replays;bit-exact zeros,NaNs,infinities,subnormals;guards/source/metadata intact;empty and5invalid-span/dimension cases;checked teardown.',
            order=ORDER, repeats=3,
            timing='Captured single production kernel:graph launch plus completion wait. Host source refresh,metadata/reset/oracle/capture/validation outside timed intervals. Sum disjoint per-call intervals,not contiguous batch wall.',
            eligibility='Per-cell warmup sum>=500ms;calibration sum>=750ms;3samples per cell sum>=250ms;clock-pair estimate<=1%;max262144calls. Exact same72cells and dispatch each process;stop first failure/no retry.',
            primary='producer_update=true,quarter-hit,rows10/20/30/40,width2560. Per-shape all3paired candidate/reference time ratios<1,geometric mean<1. No arbitrary minimumgain;unqualified shapes retain scalar.',
            auxiliary='zero/allhit and static-source cells retain samples,not substitutes for primary. No physical bus bytes,coldRAM,modelTG or full-context claim.',
            gate='Traced correctness before any speed. Component gain permits a separately committed decode-only model plan,not production adoption. Prefill stays unchanged.'),
        budget=dict(wall=900, cpu=2400, rss_cap=8 << 30, text_cap=32 << 20,
                    file_cap=256 << 20, total_cap=1 << 30),
        host_command=[sys.executable, '-B', '-m', 'unittest', '-v', 'tools.sycl.test_mapped_rows_experiment'], host_tests=10,
        cases=[], commands=[], active=False, ownership_closed=True, complete=False, passed=False,
        GPU_executed=False, model_executed=False,
        retention='One standalone fixture for immediate component decision. Keep compact individual samples,source/recipe/actual inputs/failure outcomes. Retire successful API/build output after decision;no third engine generation.')
    prior_timing = validation.committed(RESULT, ('mapped_rows_subgroup_v4',))
    require(prior_timing['stage'] == 'failed' and len(prior_timing['cases']) == 2 and
            prior_timing['cases'][0]['passed'] and not prior_timing['cases'][1]['passed'] and
            prior_timing['cases'][1]['fresh_faults'] == [], 'preserve failed original timing cohort')
    previous = B / 'mapped-rows-subgroup-v4/case1'
    execution = json.loads((previous / 'execution.json').read_text())['commands']
    require(len(execution) == 1 and execution[0]['exit_code'] == 1 and execution[0]['direct_child_reaped']
            and execution[0]['session_empty'] and not execution[0]['survivors'] and
            not Path('/proc', str(execution[0]['pid'])).exists() and
            'fixture=FAIL error=ineligible duration/timer cost' in (previous / 'owned/fixture.stderr').read_text(),
            'confirm actual terminated timing failure before a new plan')
    row['previous_timing_failure'] = dict(key='mapped_rows_subgroup_v4',
        original_outcome_preserved=True, partial_samples=18, failing_batch_metrics_unrecorded=True,
        change='Keep original250ms floor and1%timer bound;add500ms per-cell warmup and750ms calibration,raise call cap262144 and native wall420s. New fixture must compile and requalify;no candidate speed was obtained in the failed cohort.')
    check(row, committed=False)
    ROOT.mkdir(mode=0o700)
    for leaf in ('tmp', 'cache'): (ROOT / leaf).mkdir()
    save(row)
    print('Prepared fresh five-TU V5 fixture and unchanged-floor repeated plan;commit before build.', flush=True)


def host_controls(row, owner):
    entry, out, err = owner.run('host', row['host_command'], row['recipe']['environment'], P,
        wall=60, cpu=60, rss_cap=1 << 30, text_cap=1 << 20, file_cap=2 << 20, total_cap=8 << 20)
    text = out.read_text() + err.read_text()
    require(ownership.closed(entry) and entry['exit_code'] == 0 and
            'Ran ' + str(row['host_tests']) + ' tests' in text and '\nOK\n' in text and
            'skipped' not in text, 'all host controls required')
    row['host_controls'] = dict(tests=row['host_tests'], skipped=0, passed=True, exit_code=0, ownership_closed=True)


def host():
    row = json.loads((ROOT / 'record.json').read_text()); check(row)
    require(row['stage'] == 'built' and not row['cases'] and 'host_controls' not in row, 'one fresh host stage')
    owner = ownership.Owner(ROOT / 'host-owned')
    try:
        host_controls(row, owner); row.update(stage='ready', passed=True)
    except BaseException as error:
        row.update(stage='failed', passed=False, error=repr(error)); raise
    finally:
        row['active'] = owner.active is not None
        row['ownership_closed'] = owner.active is None and all(ownership.closed(c) for c in owner.commands)
        save(row)
    print('All host controls pass;commit before traced correctness.', flush=True)


def census(row, compile_output):
    root = ROOT / 'build'
    commands = json.loads((root / 'compile_commands.json').read_text())
    selected = [x for x in commands if TARGET + '.dir/' in x['output'] or
                'strata_sycl_runtime.dir/' in x['output']]
    require(len(selected) == 5 and {str(Path(x['file']).relative_to(P)) for x in selected} == TUS,
            'exact five freshly compiled translation units required')
    allowed = {}
    for pin in [row['project'], *row['dependencies']]:
        allowed.update({str((Path(pin['root']) / p).resolve()): h for p, h in pin['source_sha256'].items()})
    for meta in (row['external_inputs'],):
        allowed.update({str(Path(p).resolve()): h for p, h in json.loads(Path(meta['path']).read_text()).items()})
    compiled, generated, objects = {}, {}, {}
    for item in selected:
        obj = root / item['output']; objects[str(obj)] = sha(obj)
        args = shlex.split(item['command'])
        require('-DSTRATA_ENABLE_SYCL=1' in args and '-DSTRATA_CPU_SERVICE_PROFILE=1' in args,
                'consistent runtime class conditions required')
        text = subprocess.check_output(['/usr/bin/ninja', '-t', 'deps', item['output']], cwd=root, text=True, timeout=30)
        lines = text.splitlines()
        require(len(lines) > 8 and 'VALID' in lines[0], 'missing actual compiler dependency closure')
        for line in lines[1:]:
            if not line.startswith('    '): continue
            dep = Path(line.strip()); dep = dep if dep.is_absolute() else root / dep
            name = str(dep.resolve()); digest = sha(dep)
            if name in allowed:
                require(allowed[name] == digest, 'unadmitted compiler input: ' + name); compiled[name] = digest
            else:
                require(dep.resolve().is_relative_to(root) and dep.suffix in ('.h', '.hpp'), 'unknown external compiler input: ' + name)
                generated[name] = digest
    links = {}
    produced = {str(Path(p).resolve()): h for p, h in objects.items()}
    archive = root / 'libstrata_sycl_runtime.a'; produced[str(archive.resolve())] = sha(archive)
    for line in Path(compile_output).read_text().splitlines():
        p = Path(line.strip()); p = p if p.is_absolute() else root / p
        if not p.is_file(): continue
        name = str(p.resolve())
        if name in produced: require(sha(p) == produced[name], 'link object changed')
        elif name in allowed:
            require(sha(p) == allowed[name], 'external link input changed'); links[name] = allowed[name]
        elif line.startswith('/'):
            raise RuntimeError('unknown link input: ' + name)
    require(len(compiled) > 100 and len(links) > 8, 'incomplete compiler/link evidence')
    return dict(compiled=compiled, generated=generated, links=links, objects=objects,
                translation_units=5, reused_objects=0, uncovered=0)


def compile_fixture():
    row = json.loads((ROOT / 'record.json').read_text()); check(row)
    require(row['stage'] == 'prepared' and not (ROOT / 'build-attempt.json').exists(), 'one build attempt')
    write(ROOT / 'build-attempt.json', dict(source_commit=row['project']['revision']))
    owner = ownership.Owner(ROOT / 'build-owned')
    try:
        host_controls(row, owner)
        for label, args in [('configure', row['recipe']['configure']),
                ('compile', ['/usr/bin/cmake', '--build', str(ROOT / 'build'), '--target', TARGET, '-j', '4'])]:
            check(row, committed=False)
            entry, output, _ = owner.run(label, args, row['recipe']['environment'], P, **row['budget'])
            require(ownership.closed(entry) and entry['exit_code'] == 0, label + ' failed')
        row['actual_inputs'] = census(row, output)
        path = ROOT / 'build' / TARGET
        row['binary'] = dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)
        row['native_source_commit'] = row['project']['revision']
        check(row, committed=False)
        row.update(stage='built', passed=True)
    except BaseException as error:
        row.update(stage='failed', passed=False, error=repr(error)); raise
    finally:
        row['commands'] = owner.commands; row['active'] = owner.active is not None
        row['ownership_closed'] = owner.active is None and all(ownership.closed(c) for c in owner.commands)
        save(row)
    print('Fresh five-TU fixture built;no engine rebuilt or GPU executed.', flush=True)


def samples(text, arm):
    data = [json.loads(x) for x in text.splitlines() if x.startswith('{')]
    require(len(data) == 72, 'all72individual component samples required')
    keys = set()
    for d in data:
        key = d['rows'], d['hits'], d['producer_update'], d['rep']
        require(key not in keys and d['type'] == 'sample' and d['width'] == 2560 and
                d['rows'] in (10, 20, 30, 40) and d['hits'] in (0, d['rows'] // 4, d['rows']) and
                type(d['producer_update']) is bool and d['rep'] in range(3) and
                d['subgroup_dispatch'] is (arm == 'subgroup') and
                type(d['calls']) is int and 64 <= d['calls'] <= 262144 and not d['calls'] & (d['calls'] - 1) and
                all(math.isfinite(d[k]) for k in ('elapsed_ns', 'ns_per_call', 'clock_pair_ns', 'timer_fraction', 'calibration_ns', 'warmup_ns')) and
                d['elapsed_ns'] >= 250000000 and d['calibration_ns'] >= 750000000 and d['warmup_ns'] >= 500000000 and
                0 <= d['timer_fraction'] <= .01 and d['clock_pair_ns'] >= 0 and
                abs(d['ns_per_call'] * d['calls'] - d['elapsed_ns']) <= 1 and
                abs(d['timer_fraction'] - d['clock_pair_ns'] * d['calls'] / d['elapsed_ns']) < 1e-12,
                'ineligible or changed timing/dispatch cell')
        keys.add(key)
    return data


def run_case(mode):
    row = json.loads((ROOT / 'record.json').read_text()); check(row)
    expected = ['correctness', *ORDER]
    i = len(row['cases'])
    require(row['stage'] == 'ready' and row.get('host_controls', {}).get('passed') and i < len(expected) and mode == expected[i]
            and all(x['passed'] for x in row['cases']), 'frozen order;stop on failed case')
    label = 'case' + str(i); root = ROOT / label
    root.mkdir(mode=0o700)
    for name in ('tmp', 'cache'): (root / name).mkdir()
    env = environment(row, label, mode == 'correctness')
    cursor_text = subprocess.check_output(['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15)
    cursor = next(x.split(': ', 1)[1] for x in cursor_text.splitlines() if x.startswith('-- cursor: '))
    owner = None
    result = dict(mode=mode, index=i, passed=False, complete=False)
    def persist():
        write(root / 'execution.json', dict(result, active=bool(owner and owner.active), commands=owner.commands if owner else []))
    try:
        owner = ownership.Owner(root / 'owned', persist)
        def interact(io, entry):
            lines = []
            while True:
                line = io.line(60); lines.append(line)
                require(len(lines) <= 80, 'unexpected fixture output')
                if line == 'MAPPED_ROWS_WAIT': break
            if mode == 'correctness':
                require('correctness=PASS captures=18 replays=54 refused=5 bitwise=PASS guards=PASS readonly=PASS dynamic_metadata=PASS' in lines, 'incomplete bitwise/capture/span coverage')
            else:
                result['samples'] = samples('\n'.join(lines), mode)
                require(any(x.startswith('benchmark=PASS samples=72 ') for x in lines), 'benchmark matrix incomplete')
            mapped = set()
            for line in Path('/proc', str(entry['pid']), 'maps').read_text().splitlines():
                fields = line.split(maxsplit=5)
                if len(fields) == 6 and 'x' in fields[1] and fields[5].startswith('/'):
                    require(not fields[5].endswith(' (deleted)'), 'deleted runtime mapping')
                    mapped.add(str(Path(fields[5]).resolve()))
            allowed = {str(Path(p).resolve()): h for p, h in row['runtime_files'].items()}
            for a in (row['binary'], *row['native_validation_probe'].values()): allowed[str(Path(a['path']).resolve())] = a['sha256']
            required = {str(Path(p).resolve()) for p in (env['UR_ADAPTERS_FORCE_LOAD'], env['ZE_ENABLE_ALT_DRIVERS'],
                row['native_validation_probe']['loader']['path'], row['native_validation_probe']['library']['path'])}
            require(len(mapped) >= 8 and not mapped - allowed.keys() and not required - mapped, 'unexpected/missing runtime mappings')
            require(all(sha(p) == allowed[p] for p in mapped), 'mapped runtime bytes changed')
            result['runtime_maps'] = dict(files=len(mapped), sha256=signature({p: allowed[p] for p in sorted(mapped)}), unknown=0, missing=0)
            complete_retirement(io)
        args = ['correctness'] if mode == 'correctness' else ['bench', mode]
        budget = dict(row['budget'], wall=420, cpu=840, total_cap=128 << 20, text_cap=8 << 20)
        entry, out, err = owner.run('fixture', [row['binary']['path'], *args], env, P, interaction=interact, **budget)
        require(ownership.closed(entry) and entry['exit_code'] == 0 and
                'fixture=PASS checked_teardown=PASS' in out.read_text(), 'fixture failed or did not close')
        text = err.read_text(errors='replace')
        result['validation'] = validation.preload_evidence(text, entry['pid'])
        require(Path(result['validation']['loader']).resolve() == Path(row['native_validation_probe']['loader']['path']).resolve(), 'actual preload provider differs')
        warning = root / 'validation.log'
        require(not warning.exists() or warning.stat().st_size == 0, 'validation warning/error')
        require(not re.search(r'UR_RESULT_ERROR_|fixture=FAIL|SYCL exception', text), 'unexpected runtime error')
        if mode == 'correctness':
            trace = root / 'ur.trace'
            traced = trace.read_text(errors='replace')
            result['tracing'] = dict(bytes=trace.stat().st_size, sha256=sha(trace), **graph_trace(traced))
        check(row)
        result.update(passed=True, complete=True, exit_code=0)
    except BaseException as error:
        result['error'] = repr(error)
    finally:
        result['ownership_closed'] = not owner or owner.active is None and all(ownership.closed(c) for c in owner.commands)
        try:
            text = subprocess.check_output(['journalctl', '-k', '-b', '--after-cursor', cursor, '-n', '2048', '--no-pager'], text=True, timeout=15)
            require(len(text.splitlines()) < 2048, 'incomplete fault window')
            result['fresh_faults'] = [x for x in text.splitlines() if re.search(r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))', x)]
            require(not result['fresh_faults'] and guards.driver_guard() == row['driver'] and
                    Path('/proc/sys/kernel/random/boot_id').read_text().strip() == row['boot'], 'fresh fault or boot/driver changed')
            result['fault_probe_successful'] = True
        except BaseException as error:
            result.update(passed=False, fault_error=repr(error))
        result['passed'] = bool(result['passed'] and result['ownership_closed'])
        row['cases'].append(result); row.update(active=bool(owner and owner.active),
            ownership_closed=result['ownership_closed'], stage='ready' if result['passed'] else 'failed')
        if len(row['cases']) == 7 and result['passed']:
            row.update(stage='closed', complete=True, decision=decision(row['cases'][1:]))
        row['passed'] = result['passed']; save(row); persist()
    require(result['passed'], result.get('error', result.get('fault_error', 'case failed')))
    print('Closed ' + label + ' ' + mode + ';commit before next case.', flush=True)


def decision(cases):
    require(len(cases) == 6 and [x['mode'] for x in cases] == ORDER, 'complete frozen cohort required')
    cells = []
    for rows in (10, 20, 30, 40):
        for hits in (0, rows // 4, rows):
            for update in (False, True):
                medians = [statistics.median(s['ns_per_call'] for s in c['samples']
                    if (s['rows'], s['hits'], s['producer_update']) == (rows, hits, update)) for c in cases]
                ratios = [medians[1]/medians[0], medians[2]/medians[3], medians[5]/medians[4]]
                cells.append(dict(rows=rows, hits=hits, producer_update=update,
                    scalar_ns=[medians[x] for x in (0,3,4)], subgroup_ns=[medians[x] for x in (1,2,5)],
                    paired_time_ratios=ratios, geometric_mean_ratio=math.prod(ratios)**(1/3),
                    reproducible_gain=all(x < 1 for x in ratios)))
    return dict(cells=cells, qualified_primary_rows=[x['rows'] for x in cells
        if x['producer_update'] and x['hits'] == x['rows']//4 and x['reproducible_gain']],
        production_adopted=False, model_speed_qualified=False, physical_resource_saturation_qualified=False,
        next='Only qualified primary shapes may enter a new committed decode-only model plan;keep prompt scalar and existing defaults pending model evidence.')


def complete_retirement(io):
    io.send(b'QUIT\n')
    require(io.line(30) == 'fixture=PASS checked_teardown=PASS', 'checked retirement marker missing')
    io.exit(30)


def graph_trace(text):
    names = {'urCommandBufferCreateExp': 18, 'urCommandBufferAppendKernelLaunchWithArgsExp': 18,
             'urCommandBufferFinalizeExp': 18, 'urEnqueueCommandBufferExp': 54, 'urCommandBufferReleaseExp': 18}
    begin = Counter(re.findall(r'^\s*---> (\w+)', text, re.M))
    end = Counter(re.findall(r'^\s*<--- (\w+)\(.* -> UR_RESULT_SUCCESS;', text, re.M))
    require(all(begin[k] == end[k] == n for k, n in names.items()) and
            set(re.findall(r'UR_RESULT_ERROR_[A-Z_]+', text)) <= {'UR_RESULT_ERROR_INVALID_NULL_HANDLE'},
            'incomplete or failed actual command-graph trace')
    return dict(operations=names, complete_success_pairs=True)


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'host', 'build', 'correctness', 'scalar', 'subgroup'),
            'use prepare|host|build|correctness|scalar|subgroup')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); guards.no_model_owners([])
        if sys.argv[1] == 'prepare': prepare()
        elif sys.argv[1] == 'host': host()
        elif sys.argv[1] == 'build': compile_fixture()
        else: run_case(sys.argv[1])
