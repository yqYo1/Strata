#!/usr/bin/env python3
"""Main-owned, standalone Q6_K component comparison; repository LICENSE.

No engine build or production selection. Preserve the selected nine ELFs and
all legacy provenance holds. The prepared CPU-observer engine remains gated.
"""
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
import time

sys.dont_write_bytecode = True
import build_engine as build
import build_manifest as manifest
import cache_map_model_entry as selected
import measurement_guards as guards
import measurement_owner as supervision
import ur_validation as validation
from run_q6_columns import validation_log

P, B = selected.P, selected.B
ROOT = B / 'q6-byte-load-component-v2'
RESULT = selected.model.RESULT
KEY = 'q6_byte_load_component_v2'
EXE = ROOT / 'sycl_q6_columns_bench'
CONTROLLER = Path(__file__).resolve()
CORE = 'src/kernels/sycl/native_mmvq.cpp'
HEADER = 'include/strata/kernels/native_mmvq.hpp'
FIXTURE = 'bench/sycl/q6_columns.cpp'
SOURCES = ('src/sycl/runtime.cpp', 'src/sycl/compat.cpp', 'src/sycl/command_profile.cpp',
    CORE, 'src/kernels/sycl/native_q5_head.cpp', 'src/kernels/sycl/native_q3_esimd.cpp',
    'src/kernels/sycl/native_q4_esimd.cpp', 'src/kernels/sycl/native_q2_esimd.cpp',
    'src/kernels/sycl/native_iq4xs_esimd.cpp', 'src/kernels/sycl/native_iq_esimd.cpp',
    'src/kernels/sycl/native_mmq_xmx.cpp', 'src/kernels/sycl/iq.cpp', FIXTURE)
HELPERS = ('tools/sycl/measure_q6_columns.py', 'tools/sycl/build_engine.py',
    'tools/sycl/build_manifest.py', 'tools/sycl/measurement_guards.py',
    'tools/sycl/measurement_owner.py', 'tools/sycl/ur_validation.py',
    'tools/sycl/run_q6_columns.py')
SHAPES = ((2560, 10240, 256), (6144, 2560, 256), (2560, 6144, 256),
    (2560, 640, 1024), (2560, 512, 1024), (2560, 12288, 256), (2560, 248320, 8))
require, digest = guards.require, build.digest


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def committed():
    raw = build.git(P, 'show', 'HEAD:' + str(RESULT.relative_to(P)))
    require(json.loads(raw) == json.loads(RESULT.read_text()), 'commit result plan before action')
    return json.loads(raw)


def project_sources(revision):
    # The live tree also has unrelated, unselected observer work. Recover the
    # selected composition instead of silently compiling those runtime changes.
    _, baseline = selected.model.projection(worker=True)
    overlay = (CORE, HEADER, FIXTURE, *HELPERS, 'cmake/sycl_backend.cmake')
    values = {name: raw for name, raw in baseline.items()
              if name.startswith('include/') or name in SOURCES[:-1]}
    for name in (CORE, HEADER):
        require(build.git(P, 'show', '79e240181:' + name) == baseline[name],
            'original Q6 reference differs from the selected input: ' + name)
    values.update(dict(build.blobs(P, revision, overlay)))
    return values, set(overlay)


def save(row):
    write(ROOT / 'record.json', row)
    data = json.loads(RESULT.read_text())
    keys = ('stage', 'source_commit', 'native_reference', 'compiler_inputs', 'plan',
        'build_input_pins', 'binary', 'actual_inputs', 'build_result', 'smoke', 'processes', 'decision',
        'passed', 'complete', 'active', 'ownership_closed', 'error', 'retention')
    data[KEY] = {key: row[key] for key in keys if key in row}
    write(RESULT, data)


def source_current(row):
    project = row['project_inputs']
    recovered, overlay = project_sources(project['revision'])
    require(set(recovered) == set(project['files']), 'composed source set changed')
    for name, sha in project['files'].items():
        require(hashlib.sha256(recovered[name]).hexdigest() == sha and
            (name not in overlay or digest(P / name) == sha) and
            digest(ROOT / 'source' / name) == sha,
            'frozen component source changed: ' + name)
    for name, expected in row['external_inputs']['manifests'].items():
        require(digest(expected['path']) == expected['sha256'], 'input manifest changed: ' + name)
    runtime = row['external_inputs']['manifests']['runtime_manifest']
    require(all(digest(path) == sha for path, sha in
        json.loads(Path(runtime['path']).read_text()).items()), 'runtime provider bytes changed')
    require(digest(row['compiler_inputs']['path']) == row['compiler_inputs']['sha256'],
        'compiler changed')
    require(guards.driver_guard() == row['driver'] and
        Path('/proc/sys/kernel/random/boot_id').read_text().strip() == row['boot'],
        'boot/driver changed')
    for asset in row['native_reference']['binaries'].values():
        require(digest(asset['path']) == asset['sha256'], 'selected fallback changed')
    manifest.verify(row, row['build_input_pins'])


def allowed_inputs(row):
    allowed = {}
    for value in row['external_inputs']['manifests'].values():
        allowed.update(json.loads(Path(value['path']).read_text()))
    allowed.update(row['external_inputs']['host_link_files'])
    for name, sha in row['project_inputs']['files'].items():
        allowed[str((ROOT / 'source' / name).resolve())] = sha
    return allowed


def prepare():
    require(not ROOT.exists() and KEY not in committed(), 'fresh prospective comparison only')
    failed = committed()['q6_byte_load_lowering_v3']
    require(failed['stage'] == 'failed' and failed['ownership_closed'] and
        'IR output is not supported' in failed['error'] and not failed['GPU_executed'],
        'preserve actual host/device IR refusals;no compiler-debug retry here')
    prior = committed()['q6_byte_load_component_v1']
    require(prior['stage'] == 'failed-build' and prior['ownership_closed'] and
        all(token in prior['error'] for token in
            ('no member named', 'endian_little', 'sycl::info::device')),
        'preserve the unsupported-info-query compile refusal')
    require(not (B / 'q6-byte-load-component-v1').exists(),
        'retire closed failed-v1 partial build after exact source/recipe recovery before fresh build')
    print('Verifying selected engine/source and qualified providers before component preparation.', flush=True)
    native, _, _ = selected.proof()
    require(native['source_commit'] == '479a6fdac8cbaa35a9bf3a41d2b6e23be5260e54',
        'comparison is scoped to the selected native generation')
    revision = build.git(P, 'rev-parse', 'HEAD').decode().strip()
    values, overlay = project_sources(revision)
    require(all((P / name).read_bytes() == values[name] for name in overlay),
        'commit exact component source, benchmark and recipe first')
    # Every unrelated runtime/format input is the exact selected source byte.
    for name in SOURCES[:-1]:
        if name != CORE:
            require(values[name] == (selected.model.ROOT / 'source' / name).read_bytes(),
                'unrelated runtime/format change: ' + name)
    ROOT.mkdir(mode=0o700)
    for directory in ('source', 'objects', 'tmp', 'cache'):
        (ROOT / directory).mkdir()
    for name, raw in values.items():
        path = ROOT / 'source' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    environment = dict(native['environment'])
    environment.update(TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'))
    database = json.loads(Path(native['compile_database']['path']).read_text())
    native_root = str(selected.model.ROOT / 'source')
    units = [x for x in database if x['file'] == native_root + '/' + CORE]
    require(len(units) == 1, 'one actual production Q6 compile recipe')
    original, flags, skip = shlex.split(units[0]['command']), [], False
    for arg in original[1:]:
        if skip: skip = False; continue
        if arg in ('-o', '-MF', '-MT', '-MQ'): skip = True; continue
        if arg in ('-c', '-MD', '-MMD', '-MP', units[0]['file']): continue
        flags.append(arg.replace(native_root, str(ROOT / 'source')))
    require(not skip and '-O3' in flags and '-std=c++20' in flags and
        '-ffp-contract=off' in flags and '-fsycl' in flags,
        'retain actual production optimization/language flags')
    compiler = native['compiler']
    require(original[0] == compiler['path'], 'actual production compiler differs')
    commands = [('fixture-syntax', [compiler['path'], *flags, '-fsyntax-only',
        '-MD', '-MF', str(ROOT / 'objects/fixture.syntax.d'), str(ROOT / 'source' / FIXTURE)])]
    objects = []
    for i, name in enumerate(SOURCES):
        obj = ROOT / 'objects' / (str(i) + '.o')
        objects.append(str(obj))
        commands.append(('compile-' + str(i), [compiler['path'], *flags,
            '-MD', '-MF', str(obj.with_suffix('.d')), '-c', str(ROOT / 'source' / name), '-o', str(obj)]))
    commands.append(('link', [compiler['path'], *flags, *objects, '-Wl,-t', '-o', str(EXE)]))
    row = dict(stage='prepared', source_commit=revision,
        project_inputs=dict(revision=revision, selected_native_revision=native['source_commit'],
            reference_source_base='79e240181', files={name: hashlib.sha256(raw).hexdigest()
            for name, raw in values.items()}), generated_inputs={},
        dependency_inputs=dict(selected_native=native['source_commit'], standalone=True),
        external_inputs=dict(manifests={name: native[name] for name in
            ('sdk', 'external_manifest', 'runtime_manifest', 'compiled_inputs')},
            host_link_files=native['host_link_inputs']['files']),
        compiler_inputs=compiler, environment=environment, commands=commands,
        driver=guards.driver_guard(), boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        native_reference=dict(source_commit=native['source_commit'], binaries=native['binaries'],
            validation=native['native_validation_probe']),
        plan=dict(reference='Public native_q6_k_mmvq exact layout;both arms use Q6/Q81 byte-span access and the same byte-writing Q81 quantizer',
            candidate='Explicit native_q6_k_columns_mmvq;specialize 1,2,3,4,8 only;no model caller',
            compile_recipe=dict(compiler=compiler,flags=[x.replace(str(ROOT / 'source'),'<source>')
                for x in flags],origin='Actual selected479a native_mmvq.cpp compile database;only source paths and build outputs change.'),
            model_shapes=[list(shape) for shape in SHAPES], smoke_cases=32,
            process_repetitions=3, batches_per_arm_shape=5, timing_floor_seconds=.01,
            timer='Host steady clock including public dispatch,submission and final queue completion;weights/Q8_1 already resident;no conversion,reset or correctness work inside timing',
            order='For each shape/column,batch+column+position modulo2;five rotated paired batches',
            acceptance='All reference/candidate output bits,guards,read-only inputs and finite values match. Scalar dequantized-dot checks on first/middle/last rows of every column require abs_error<=2e-5+2e-6*sum_abs_terms in double,outside timers. Check every Q81 encoded byte against a host FP32 XOR-sum/rounding reference outside timers,including all-zero and negative-constant smoke blocks. Both host references share the tested half-bit converter. Require actual little-endian device,normal exits,strict validation,no fresh fault. Any shape may advance separately only when all three independent process median candidate/reference ratios are below1;no arbitrary minimum gain.',
            stopping='Stop on first source/provider/geometry/numerical/fault/owner/budget/timing-floor failure;retain original failure. No model execution from this component alone.',
            limits='Synthetic packed weights at actual model geometries. All consumed Q6/Q81 fields and Q81 scratch writes use byte spans in both arms;this comparison does not attribute a speed change to that repair. Actual production C++20/optimization recipe replaces the earlier component C++17/-fno-fast-math recipe,so old timings are descriptive only. Host/device IR output is unavailable. Validation preload retains UR until exit. No sustained TG,prefill,PCIe traffic,register-spill proof,ordinary provider teardown or model adoption. Other formats retain their legacy typed consumers;this component does not qualify them or all grouped/prompt call paths.'),
        budgets=dict(wall=1200, cpu=2400, rss_cap=8 << 30, text_cap=4 << 20,
            file_cap=128 << 20, total_cap=768 << 20),
        passed=False, complete=False, active=False, ownership_closed=True, processes=[],
        retention=dict(consumer='Immediate Q6_K column specialization comparison only;selected nine-ELF fallback remains unchanged',
            review_point='After component decision,retire rejected candidate ELF/objects/raw logs after source recovery;no extra engine generation',
            legacy_build_inventory='Preserve selected nine-ELF engine and the already built phase generation;this is one standalone numerical/performance fixture,not another engine generation. Retire fixture/objects/logs after the scoped decision and original input recovery.'))
    row['build_input_pins'] = manifest.pins(row)
    source_current(row)
    save(row)
    print('Prepared standalone component;commit exact generated plan before build.', flush=True)


def load():
    row = json.loads((ROOT / 'record.json').read_text())
    require(committed()[KEY] == {key: row[key] for key in committed()[KEY]},
        'commit current compact result before action')
    source_current(row)
    return row


def build_component():
    row = load()
    require(row['stage'] == 'prepared' and not EXE.exists(), 'one prospective compilation')
    owner = supervision.Owner(ROOT / 'build-commands')
    allowed = allowed_inputs(row)
    deadline = time.monotonic() + 1800
    try:
        for label, command in row['commands']:
            print(label, flush=True)
            remaining = deadline - time.monotonic()
            require(remaining > 0, 'whole component build deadline')
            budgets = dict(row['budgets'], wall=min(remaining, row['budgets']['wall']))
            entry, out, err = owner.run(label, command, row['environment'], ROOT, **budgets)
            require(supervision.closed(entry) and entry['exit_code'] == 0,
                label + ' failed: ' + err.read_text()[-4000:])
        actual = {}
        for dep in sorted((ROOT / 'objects').glob('*.d')):
            for name in shlex.split(dep.read_text().replace('\\\n', ' ').split(':', 1)[1]):
                path = str(Path(name).resolve())
                require(path in allowed and digest(path) == allowed[path],
                    'uncovered/changed compiler input: ' + path)
                actual[path] = allowed[path]
        link = (ROOT / 'build-commands/link.stdout').read_text()
        linked = {}
        for line in link.splitlines():
            if not line.startswith('/'):
                continue
            path = Path(line).resolve()
            if path.is_relative_to(ROOT / 'objects') or path.is_relative_to(ROOT / 'tmp'):
                # Owned compiler products can disappear after the driver exits.
                continue
            require(path.is_file(), 'nonfile external link input: ' + line)
            name = str(path.resolve())
            require(name in allowed and digest(name) == allowed[name], 'uncovered link input: ' + name)
            linked[name] = allowed[name]
        require(len(actual) >= 300 and len(linked) >= 10, 'incomplete compiler/link input accounting')
        source_current(row)
        row.update(stage='compiled', binary=dict(path=str(EXE), sha256=digest(EXE), bytes=EXE.stat().st_size),
            build_result=dict(passed=True, compiler_inputs=len(actual), linked_inputs=len(linked),
                owner_closed=all(supervision.closed(e) for e in owner.commands)))
        write(ROOT / 'actual-inputs.json', dict(compiler=actual, linked=linked))
        row['actual_inputs'] = dict(path=str(ROOT / 'actual-inputs.json'), sha256=digest(ROOT / 'actual-inputs.json'))
    except BaseException as error:
        row.update(stage='failed-build', error=repr(error))
        raise
    finally:
        row['ownership_closed'] = owner.active is None and all(supervision.closed(e) for e in owner.commands)
        save(row)


def native_environment(row, smoke):
    env = dict(row['environment'])
    probe = row['native_reference']['validation']
    for asset in probe.values():
        require(digest(asset['path']) == asset['sha256'], 'validation provider changed')
    env.update(LD_PRELOAD=probe['library']['path'], STRATA_UR_PROBE_EXPECTED_LOADER=probe['loader']['path'])
    if not smoke:
        env['UR_ENABLE_LAYERS'] = 'UR_LAYER_PARAMETER_VALIDATION'
        env['UR_LOG_TRACING'] = 'level:error;flush:error;output:stderr'
    return env


def parse_speed(text):
    groups = {}
    for line in text.splitlines():
        if not line.startswith('Q6_SAMPLE '):
            continue
        fields = dict(word.split('=', 1) for word in line.split()[1:])
        key = tuple(int(fields[k]) for k in ('ni', 'no', 'nc'))
        arm, batch, position = (int(fields[k]) for k in ('arm', 'batch', 'position'))
        require(arm == (batch + key[2] + position) % 2 and 0 <= batch < 5 and position in (0, 1),
            'wrong declared batch order')
        bucket = groups.setdefault(key, {0: {}, 1: {}, 'calls': int(fields['calls'])})
        seconds = float(fields['seconds'])
        require(batch not in bucket[arm] and int(fields['calls']) == bucket['calls'] and
            math.isfinite(seconds) and seconds >= .01, 'duplicate/short/nonfinite timing batch')
        bucket[arm][batch] = seconds
    expected = {(ni, no, nc): calls for ni, no, calls in SHAPES for nc in (1, 2, 3, 4, 8)}
    require(set(groups) == set(expected), 'missing/extra model geometry')
    result = []
    for key, calls in expected.items():
        group = groups[key]
        require(group['calls'] == calls and all(set(group[a]) == set(range(5)) for a in (0, 1)),
            'missing timing batch/call count')
        samples = [[group[arm][batch] / calls for batch in range(5)] for arm in (0, 1)]
        result.append(dict(shape=list(key), reference_s=samples[0], candidate_s=samples[1],
            median_time_ratio=statistics.median(samples[1]) / statistics.median(samples[0])))
    return result


def journal_cursor():
    text = subprocess.check_output(['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'],
        text=True, timeout=15)
    return next(line.split(': ', 1)[1] for line in text.splitlines() if line.startswith('-- cursor: '))


def fresh_faults(cursor):
    text = subprocess.check_output(['journalctl', '-k', '-b', '--after-cursor', cursor,
        '-n', '2048', '--no-pager'], text=True, timeout=15)
    require(len(text.splitlines()) < 2048, 'incomplete bounded kernel log')
    return [line for line in text.splitlines() if re.search(
        r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))', line)]


def run(smoke):
    row = load()
    require(row['stage'] == ('compiled' if smoke else 'smoke-passed'), 'component progression gate')
    require(digest(EXE) == row['binary']['sha256'], 'actual component binary changed')
    require(digest(row['actual_inputs']['path']) == row['actual_inputs']['sha256'], 'build inputs receipt changed')
    actual = json.loads(Path(row['actual_inputs']['path']).read_text())
    allowed = allowed_inputs(row)
    for name, sha in dict(actual['compiler'], **actual['linked']).items():
        require(name in allowed and sha == allowed[name] and digest(name) == sha, 'compiled input changed')
    count = 1 if smoke else 3
    owner = supervision.Owner(ROOT / ('smoke-commands' if smoke else 'speed-commands'))
    env = native_environment(row, smoke)
    deadline = time.monotonic() + 2400
    try:
        for process in range(count):
            print('Q6 run', 'smoke' if smoke else 'speed', process, flush=True)
            cursor = journal_cursor()
            remaining = deadline - time.monotonic()
            require(remaining > 0, 'whole component execution deadline')
            budgets = dict(row['budgets'], wall=min(remaining, row['budgets']['wall']))
            entry, out, err = owner.run('process-' + str(process), [str(EXE), '--smoke' if smoke else '--speed'],
                env, ROOT, **budgets)
            require(supervision.closed(entry) and entry['exit_code'] == 0,
                'component failed: ' + (err.read_text() + out.read_text())[-4000:])
            text = out.read_text()
            require('Q6_COLUMNS_PASS mode=' + ('smoke' if smoke else 'speed') in text, 'missing normal completion')
            evidence = validation_log(err.read_text(), entry['pid'],
                row['native_reference']['validation']['loader']['path'], smoke)
            expected_cases = 32 if smoke else 35
            require(text.count('Q6_TARGET_ENDIAN_LITTLE=1') == 1 and
                text.count(' scalar_oracle=PASS') == expected_cases and
                text.count(' q8_oracle=PASS') == expected_cases,
                'missing actual endian/Q6-scalar/Q81-byte-oracle proof')
            faults = fresh_faults(cursor)
            require(faults == [], 'new GPU fault')
            source_current(row)
            result = dict(native_exit_code=0, ownership_closed=True, validation=evidence,
                passed_cases=text.count('Q6_CASE_PASS '), fresh_faults=faults)
            if smoke:
                require(result['passed_cases'] == 32, 'missing smoke case')
                row['smoke'] = result
                row['stage'] = 'smoke-passed'
            else:
                require(result['passed_cases'] == 35, 'missing performance case correctness')
                result['samples'] = parse_speed(text)
                row['processes'].append(result)
            save(row)
        if not smoke:
            decisions = []
            for i in range(35):
                samples = [p['samples'][i] for p in row['processes']]
                ratios = [s['median_time_ratio'] for s in samples]
                decisions.append(dict(shape=samples[0]['shape'], process_time_ratios=ratios,
                    all_three_faster=all(r < 1 for r in ratios)))
            row.update(stage='closed', passed=True, complete=True,
                decision=dict(shapes=decisions, inference_adopted=False,
                    next='Only individually faster shapes may enter a new explicitly decode-scoped model plan;preserve prefill/MTP and perform long-generation/quality/safety qualification.'))
    except BaseException as error:
        row.update(stage='failed-smoke' if smoke else 'failed-speed', error=repr(error))
        raise
    finally:
        row['ownership_closed'] = owner.active is None and all(supervision.closed(e) for e in owner.commands)
        save(row)


def main():
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'build', 'smoke', 'speed'),
        'expected prepare/build/smoke/speed')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners([])
        action = sys.argv[1]
        if action == 'prepare': prepare()
        elif action == 'build': build_component()
        else: run(action == 'smoke')


if __name__ == '__main__':
    main()
