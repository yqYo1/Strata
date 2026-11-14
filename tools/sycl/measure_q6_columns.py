#!/usr/bin/env python3
"""Own the complete Q6 native-half comparison; Main-authored, repository LICENSE.

Both arms include the public column entry and production checks. Closed
centering experiments remain recoverable from Git. This unselected component
does not establish phase ownership or model adoption.
"""
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import device_fp64_probe as dependencies
import mapped_row_copy_component as shared
from run_q6_columns import validation_log

P, B, RESULT = shared.P, shared.B, shared.RESULT
ROOT = B / 'q6-native-half-component-v3'
KEY, COHORT = 'q6_native_half_component_v3', 'decode_iq3_audit_speed_v2'
CPU_KEY = 'cpu_iq3s_pair_component_v1_analysis'
AUDIT_KEY = 'q6_native_half_entry_contract_20261106'
INDEX = P / 'bench/results/2026-10-15-pr6-recovery-audit/research-index.json'
EXE = ROOT / 'sycl_q6_columns_bench'
GIT = str(Path(shutil.which('git')).resolve())
CORE, FIXTURE = 'src/kernels/sycl/native_mmvq.cpp', 'bench/sycl/q6_columns.cpp'
SOURCES = ('src/sycl/runtime.cpp', 'src/sycl/compat.cpp', 'src/sycl/command_profile.cpp',
    CORE, 'src/kernels/sycl/native_q5_head.cpp', 'src/kernels/sycl/native_q3_esimd.cpp',
    'src/kernels/sycl/native_q4_esimd.cpp', 'src/kernels/sycl/native_q2_esimd.cpp',
    'src/kernels/sycl/native_iq4xs_esimd.cpp', 'src/kernels/sycl/native_iq_esimd.cpp',
    'src/kernels/sycl/native_mmq_xmx.cpp', 'src/kernels/sycl/iq.cpp', FIXTURE)
SHAPES = ((2560, 10240, 256), (6144, 2560, 256), (2560, 6144, 256),
    (2560, 640, 1024), (2560, 512, 1024), (2560, 12288, 256), (2560, 248320, 8))
require, sha, owner, guards = shared.require, shared.sha, shared.owner, shared.guards


def git(*args):
    return subprocess.check_output([GIT, *args], cwd=P, timeout=30)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def committed(key):
    original = json.loads(git('show', 'HEAD:' + str(RESULT.relative_to(P))))
    current = json.loads(RESULT.read_text())
    require(key in original and key in current and canonical(original[key]) == canonical(current[key]),
            'commit exact consumed result: ' + key)
    return current[key]


def cohort():
    row = committed(COHORT)
    require(row['stage'] == 'closed' and row['passed'] and row['complete'] and
        row['ownership_closed'] and not row['active'] and len(row['processes']) == 6 and
        [p['index'] for p in row['processes']] == list(range(6)) and
        all(p['passed'] and p['complete'] and p['ownership_closed'] and
            p['native_exit_code'] == 0 and p['fresh_faults'] == [] for p in row['processes']),
        'close all six IQ3 audit processes before Q6 preparation')
    return row


def qualifications():
    cpu = committed(CPU_KEY)
    require(cpu['stage'] == 'closed' and cpu['passed'] and cpu['complete'] and
        cpu['ownership_closed'] and not cpu['active'] and
        cpu['original_failure_preserved'] and cpu['native_executed'] is False and
        cpu['original_samples_verified'] == 240 and len(cpu['processes']) == 6 and
        [p['index'] for p in cpu['processes']] == list(range(6)) and
        all(p['exit_code'] == 0 and p['pid'] > 0 and p['start_ticks'] > 0
            for p in cpu['processes']), 'close the CPU comparison before GPU preparation')
    previous = json.loads(git('show', 'HEAD:' + str(INDEX.relative_to(P))))[AUDIT_KEY]
    audit = json.loads(INDEX.read_text())[AUDIT_KEY]
    require(canonical(previous) == canonical(audit) and
        audit['verdict'] == 'conditional_source_pass' and
        audit['ownership']['worker_completed'] and
        audit['ownership']['own_external_terminal_closed'], 'commit closed Q6 source review')
    for name, expected in audit['source_hashes'].items():
        require(hashlib.sha256(git('show', audit['source_revision'] + ':' + name)).hexdigest()
            == expected, 'original reviewed source differs: ' + name)
        if name != FIXTURE:
            require(sha(P / name) == expected, 'reviewed kernel/contract changed: ' + name)
    # Main reviewed the only fixture supplement: a host clock-pair screen
    # outside either arm. Native math and the numerical workload are unchanged.
    fixture_delta = git('diff', audit['source_revision'], 'HEAD', '--', FIXTURE)
    return dict(cpu_signature=signature(cpu), audit_signature=signature(audit),
        audit_source_revision=audit['source_revision'], source_hashes=audit['source_hashes'],
        fixture_supplement=dict(sha256=sha(P / FIXTURE),
            diff_sha256=hashlib.sha256(fixture_delta).hexdigest(),
            main_review='Host-only steady-clock pair screen outside both arms;no math/workload change.'))


def production_sources(native):
    path = B / 'decode-service-model-v1/record.json'
    record = json.loads(path.read_text())
    require(record['source_commit'] == native['source_commit'] and
        record['compile_database'] == native['compile_database'] and
        record['selected_basis'] == native['selected_basis'] and
        shared.signature(record['native_sources']) == native['native_sources'],
        'full materialized-source record differs from committed build evidence')
    return record['native_sources'], dict(path=str(path), sha256=sha(path))


def signature(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def compact(row):
    result = {k: v for k, v in row.items() if k not in ('sources', 'runtime_pins', 'commands')}
    result['source_inputs'] = shared.signature(row['sources'])
    result['runtime_inputs'] = shared.signature(row['runtime_pins'])
    result['command_inputs'] = dict(count=len(row['commands']), sha256=signature(row['commands']))
    return result


def save(row):
    write(ROOT / 'record.json', row)
    data = json.loads(RESULT.read_text()); data[KEY] = compact(row)
    RESULT.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def persist_owner(supervisor, path):
    fields = ('label', 'pid', 'direct_identity', 'owners', 'session_anchored',
        'session_empty', 'direct_child_reaped', 'exit_code', 'ownership_status',
        'errors', 'error', 'cleanup', 'survivors', 'started_utc', 'finished_utc', 'logs')
    value = dict(active_pid=supervisor.active.pid if supervisor.active is not None else None,
        commands=[dict(command_index=i, **{k: e[k] for k in fields if k in e})
                  for i, e in enumerate(supervisor.commands)])
    raw = json.dumps(value, indent=2) + '\n'
    require(len(raw.encode()) <= 4 << 20, 'bounded durable owner receipt')
    pending = path.with_suffix('.tmp')
    pending.write_text(raw)
    pending.replace(path)


def make_owner(row, phase):
    path = ROOT / (phase + '-owner.json')
    row.setdefault('owner_receipts', {})[phase] = dict(path=str(path))
    save(row)
    supervisor = owner.Owner(ROOT / (phase + '-commands'),
                             persist=lambda: persist_owner(supervisor, path))
    persist_owner(supervisor, path)
    return supervisor


def finish_owner(row, supervisor, phase):
    row['active'] = False
    row['ownership_closed'] = supervisor is None or (supervisor.active is None and
        all(owner.closed(e) for e in supervisor.commands))
    if supervisor is not None:
        path = ROOT / (phase + '-owner.json'); persist_owner(supervisor, path)
        row['owner_receipts'][phase].update(sha256=sha(path), closed=row['ownership_closed'],
                                          commands=len(supervisor.commands))
    save(row)


def native_postchecks(row, start):
    def fault_probe():
        require(start is not None, 'kernel cursor unavailable')
        return fresh_faults(start)

    result = dict(passed=True)
    for key, probe in (
        ('faults', fault_probe),
        ('inputs_boot_driver', lambda: inputs_current(row)),
        ('model_owners', lambda: guards.no_model_owners([]))):
        try:
            value = probe()
            result[key] = dict(passed=(not value if key == 'faults' else True), evidence=value)
            if key == 'faults':
                raw = '\n'.join(value).encode()
                if len(raw) > 256 << 10:
                    result[key].update(evidence=raw[:256 << 10].decode(errors='replace').splitlines(),
                        truncated=True, total_bytes=len(raw), full_sha256=hashlib.sha256(raw).hexdigest())
        except BaseException as error:
            result[key] = dict(passed=False, error=repr(error))
        result['passed'] = result['passed'] and result[key]['passed']
    return result


def final_owner_guard(row):
    try:
        row['final_owner_guard'] = dict(passed=True, evidence=guards.no_model_owners([]))
    except BaseException as error:
        row['final_owner_guard'] = dict(passed=False, error=repr(error))
        if row['stage'] in ('prepared', 'compiled', 'smoke-passed', 'closed'):
            row['stage'] = 'failed-final-owner'
        row.update(passed=False, complete=False)
    save(row)
    return row['final_owner_guard']['passed']


def prepare():
    closed = cohort()
    qualification = qualifications()
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()), 'fresh attempt only')
    revision = git('rev-parse', 'HEAD').decode().strip()
    tracked = git('ls-files', '-z', 'include', 'tools/sycl').decode().split('\0')
    names = sorted(set(SOURCES) | {n for n in tracked if n and
        (n.startswith('include/') or n.endswith('.py'))} |
        set(qualification['source_hashes']) |
        {'AGENTS.md', 'docs/ARTIFACT_RETENTION.md', 'tools/sycl/measurement_environment.json',
         'tests/core/test_q6_columns.py'})
    values = {n: git('show', revision + ':' + n) for n in names}
    require(len(names) <= 1024 and sum(map(len, values.values())) <= 32 << 20 and
        all((P / n).read_bytes() == raw for n, raw in values.items()),
        'commit exact native, fixture and transitive controller sources first')
    native = committed('decode_service_model_build_v1')
    require(native['passed'] and native['complete'] and native['ownership_closed'] and
        not native['active'], 'closed retained production build')
    # source_commit identifies the build controller/recipe. The selected model
    # image is materialized from pinned source revisions; use its actual inputs.
    materialized, production_record = production_sources(native)
    original_hashes = {name: materialized[name]
        for name in SOURCES if name != FIXTURE}
    require(all(sha(B / 'decode-service-model-v1/source' / name) == expected
        for name, expected in original_hashes.items()), 'retained production source image changed')
    changed = sorted(name for name, expected in original_hashes.items() if sha(P / name) != expected)
    require(changed == [CORE, 'src/sycl/command_profile.cpp'],
        'unexpected production GPU translation-unit delta')
    database = native['compile_database']
    require(sha(database['path']) == database['sha256'], 'actual native compile database changed')
    units = [u for u in json.loads(Path(database['path']).read_text()) if u['file'].endswith('/' + CORE)]
    require(len(units) == 1, 'one actual Q6 production recipe')
    original, flags, skip = shlex.split(units[0]['command']), [], False
    origin = units[0]['file'][:-len(CORE)]
    for token in original[1:]:
        if skip: skip = False; continue
        if token in ('-o', '-MF', '-MT', '-MQ'): skip = True; continue
        if token in ('-c', '-MD', '-MMD', '-MP', units[0]['file']): continue
        flags.append(token.replace(origin, str(ROOT / 'source') + '/'))
    require(not skip and all(f in flags for f in
        ('-O3', '-DNDEBUG', '-std=c++20', '-ffp-contract=off', '-fsycl')),
        'retain actual native optimization and numerical flags')
    flags.append('--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13')
    tools = {n: dict(path=p, realpath=str(Path(p).resolve()), sha256=sha(p)) for n, p in
        dict(compiler=original[0], git=GIT, python=sys.executable,
             assembler='/usr/bin/as', linker='/usr/bin/ld').items()}
    image = dependencies.toolchain()
    encoded = (json.dumps(image, indent=2) + '\n').encode()
    providers = closed['runtime_manifest']
    require(sha(providers['path']) == providers['sha256'], 'qualified provider manifest changed')
    runtime = json.loads(Path(providers['path']).read_text())
    require(len(runtime) == providers['files'] and all(sha(p) == h for p, h in runtime.items()),
            'qualified provider bytes changed')
    require(all(runtime.get(str(Path(a['path']).resolve())) == a['sha256']
        for a in closed['native_validation_probe'].values()), 'qualified validation provider missing')
    env = dict(closed['environment_template'])
    env.update(TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'),
        PYTHONPYCACHEPREFIX=str(ROOT / 'cache/python'), STRATA_MEASUREMENT_ROOT=str(ROOT),
        UR_LOG_VALIDATION='level:warning;flush:warning;output:stderr',
        UR_LOG_LOADER='level:warning;flush:warning;output:stderr',
        UR_LOG_LEVEL_ZERO='level:warning;flush:warning;output:stderr')
    require(all(env.get(k) == v for k, v in dict(ONEAPI_DEVICE_SELECTOR='level_zero:gpu',
        UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1', NEOReadDebugKeys='1',
        EnableDirectSubmission='0', ZES_ENABLE_SYSMAN='0', SYCL_CACHE_PERSISTENT='0').items()),
        'preserve qualified runtime safety before exec')
    commands, objects = [], []
    for i, source in enumerate(SOURCES):
        obj = ROOT / 'objects' / (str(i) + '.o'); objects.append(str(obj))
        commands.append(('compile-' + str(i), [original[0], *flags, '-MD', '-MF',
            str(obj.with_suffix('.d')), '-c', str(ROOT / 'source' / source), '-o', str(obj)]))
    commands.append(('link', [original[0], *flags, *objects, '-Wl,-t', '-o', str(EXE)]))
    row = dict(stage='preparing', source_commit=revision,
        sources={n: hashlib.sha256(raw).hexdigest() for n, raw in values.items()},
        runtime_pins=runtime, cohort_signature=signature(closed), qualifications=qualification, tools=tools,
        toolchain=dict(path=str(ROOT / 'toolchain.json'), sha256=hashlib.sha256(encoded).hexdigest(),
            files=len(image['files']), bytes=image['bytes'], packages=image['packages']),
        native_recipe_origin=dict(source_commit=native['source_commit'], compile_database=database,
            materialized_source_record=production_record, native_build_signature=signature(native),
            flags=flags, change='Private source paths plus explicit GCC13 host-header selection;no math/optimization flag change.'),
        common_source_delta=dict(original_GPU_units=original_hashes, changed=changed,
            review='native_mmvq adds the unselected audited entry. Shared command_profile adds inactive kinds/metadata;the fixture constructs no Trace,so fused_tag returns before registration. Both arms share it. Other GPU translation units match retained5c5e. Actual compiler dependencies are checked after build;this is not whole-model equivalence.'),
        compiler_version=subprocess.check_output([original[0], '--version'], env=env, text=True,
            timeout=15).splitlines()[0], environment=env,
        native_validation_probe=closed['native_validation_probe'], commands=commands,
        boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), driver=guards.driver_guard(),
        budgets=dict(wall=1200, cpu=2400, rss_cap=8 << 30, text_cap=16 << 20,
            file_cap=128 << 20, total_cap=768 << 20),
        plan=dict(reference='native_q6_k_columns_mmvq', candidate='native_q6_k_native_half_columns_mmvq',
            difference='Byte-copy half bits,bit_cast to real sycl::half,then widen Q6/Q81 scales to float;same packed data,dot,FP32 accumulation/reduction,geometry and outputs. Per-call fp16 capability check remains timed.',
            provenance='Own existing Q6 kernel and standalone SYCL conversion;no XeStrata implementation input.',
            device_identity='Exact LevelZero,vendor8086,devicee20c,PCI0000:05:00.0,fp16,in-order;descriptive name is emitted,not matched.',
            order='One traced smoke,then three fresh processes;five paired batches per shape/column,arm=(batch+column+position)%2.',
            shapes=SHAPES, columns=[1, 2, 3, 4, 8], smoke_cases=32, speed_cases=35,
            timer='Steady-clock full public dispatch and queue completion;resident weights/Q81;conversion,allocation,oracles outside timers.',
            minimum_batch_seconds=.01, timer_screen='2048 adjacent steady-clock pairs before speed cases;finite positive median <=0.1% of fastest timing batch.',
            process_repetitions=3, batches_per_arm=5,
            acceptance='All output bits,read-only inputs,guards and finite values pass;Q6 scalar and every Q81 byte oracle pass. Exact B570,in-order queue,normal0,quiet validation,closed owners,unchanged inputs/providers/boot and no fresh fault. A shape is eligible only when all three independent median candidate/reference time ratios are below1;no percentage floor.',
            stopping='Stop on first source/provider/owner/numerical/validation/fault/budget/short-timer failure;no retries or hash refresh.',
            limits='Synthetic model geometries and parameter-validation/preload envelope. Logical bytes unchanged;no physical-traffic,saturation,native-ISA,clean model-speed,ordinary runtime-teardown,prefill or262144/RESTORE qualification. No production selector is enabled.'),
        retention='One auxiliary component build for this imminent native-half comparison. Keep552b validated prefill fallback(9ELFs) and5c5e current diagnostic model(5ELFs) for their recorded consumers;no b310/CPU comparison build is needed. Retire this fixture ELF/objects/source snapshot/traces after committing the decision and verifying exact source/recipe recovery;not another model reference generation.',
        processes=[], active=False, ownership_closed=True, passed=False, complete=False,
        GPU_executed=False, model_executed=False)
    ROOT.mkdir(mode=0o700)
    try:
        save(row)
        for directory in ('source', 'objects', 'tmp', 'cache'): (ROOT / directory).mkdir()
        (ROOT / 'toolchain.json').write_bytes(encoded)
        for n, raw in values.items():
            path = ROOT / 'source' / n; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
        row['stage'] = 'prepared'; save(row)
    except BaseException as error:
        row.update(stage='failed-prepare', error=repr(error)); save(row); raise
    print('Prepared Q6 native-half;commit exact plan before build.', flush=True)


def inputs_current(row):
    require(signature(cohort()) == row['cohort_signature'], 'closed cohort evidence changed')
    require(canonical(qualifications()) == canonical(row['qualifications']),
            'closed CPU or Q6 source-review qualification changed')
    native = row['native_recipe_origin']
    require(signature(committed('decode_service_model_build_v1')) == native['native_build_signature'] and
        sha(native['materialized_source_record']['path']) == native['materialized_source_record']['sha256'] and
        sha(native['compile_database']['path']) == native['compile_database']['sha256'],
        'original production source/recipe record changed')
    require(all(sha(ROOT / 'source' / n) == h and sha(P / n) == h for n, h in row['sources'].items()),
            'frozen source/fixture/controller changed')
    require(all(sha(p) == h for p, h in row['runtime_pins'].items()), 'runtime provider changed')
    require(sha(row['toolchain']['path']) == row['toolchain']['sha256'] and
        dependencies.toolchain() == json.loads(Path(row['toolchain']['path']).read_text()),
        'compiler/package image changed')
    require(all(str(Path(t['path']).resolve()) == t['realpath'] and sha(t['path']) == t['sha256']
        for t in row['tools'].values()) and row['tools']['git']['path'] == GIT, 'tool executable changed')
    require(guards.driver_guard() == row['driver'] and
        Path('/proc/sys/kernel/random/boot_id').read_text().strip() == row['boot'], 'driver/boot changed')
    if 'binary' in row: require(sha(EXE) == row['binary']['sha256'], 'fixture executable changed')


def current():
    row = json.loads((ROOT / 'record.json').read_text())
    require(canonical(committed(KEY)) == canonical(compact(row)), 'commit exact component state')
    inputs_current(row)
    require(all(r.get('closed') and sha(r['path']) == r['sha256']
        for r in row.get('owner_receipts', {}).values()), 'closed owner receipt changed')
    return row


def actual_inputs(row):
    image = json.loads(Path(row['toolchain']['path']).read_text())
    allowed = {v['realpath']: v['sha256'] for v in image['files'].values() if v['kind'] == 'file'}
    allowed.update(row['runtime_pins'])
    allowed.update({str((ROOT / 'source' / n).resolve()): h for n, h in row['sources'].items()})
    names = []
    expected = {ROOT / 'objects' / (str(i) + '.d') for i in range(len(SOURCES))}
    require(set((ROOT / 'objects').glob('*.d')) == expected,
            'one actual dependency file per compiled translation unit')
    for dep in sorted(expected):
        target, inputs = dep.read_text().replace('\\\n', ' ').split(':', 1)
        require(shlex.split(target) == [str(dep.with_suffix('.o'))], 'dependency target differs')
        names += shlex.split(inputs)
    links = [line.strip() for line in (ROOT / 'build-commands/link.stdout').read_text().splitlines()
             if line.startswith('/')]
    require(len(names) >= 300 and len(links) >= 10, 'actual include/link accounting missing')
    actual = {}
    for name in dict.fromkeys(names + links):
        path = Path(name).resolve()
        if path.is_relative_to(ROOT / 'objects') or path.is_relative_to(ROOT / 'tmp'): continue
        require(str(path) in allowed and sha(path) == allowed[str(path)], 'unqualified build input: ' + name)
        actual[str(path)] = allowed[str(path)]
    write(ROOT / 'actual-inputs.json', actual)
    return dict(path=str(ROOT / 'actual-inputs.json'), sha256=sha(ROOT / 'actual-inputs.json'),
                signature=shared.signature(actual))


def build_component():
    row = current(); require(row['stage'] == 'prepared' and not EXE.exists(), 'one build attempt')
    row.update(stage='building', active=True, ownership_closed=False); save(row)
    supervisor = None
    try:
        supervisor = make_owner(row, 'build'); deadline = time.monotonic() + 1800
        for label, argv in row['commands']:
            require(time.monotonic() < deadline, 'whole build deadline')
            print(label, flush=True)
            limits = dict(row['budgets'], wall=min(row['budgets']['wall'], deadline-time.monotonic()))
            entry, out, err = supervisor.run(label, argv, row['environment'], ROOT, **limits)
            require(owner.closed(entry) and entry['exit_code'] == 0, label + ': ' + err.read_text()[-4000:])
        row['actual_inputs'] = actual_inputs(row)
        row['binary'] = dict(path=str(EXE), sha256=sha(EXE), bytes=EXE.stat().st_size)
        inputs_current(row)
        row['stage'] = 'compiled'
    except BaseException as error:
        row.update(stage='failed-build', error=repr(error)); raise
    finally:
        finish_owner(row, supervisor, 'build')


def timer_screen(text):
    lines = [line for line in text.splitlines() if line.startswith('Q6_TIMER ')]
    require(len(lines) == 1, 'one host clock-pair screen per speed process')
    fields = dict(word.split('=', 1) for word in lines[0].split()[1:])
    seconds = float(fields['median_seconds'])
    require(set(fields) == {'pairs', 'median_seconds'} and int(fields['pairs']) == 2048 and
        math.isfinite(seconds) and seconds > 0, 'invalid host clock-pair evidence')
    return dict(pairs=2048, median_seconds=seconds)


def parse_speed(text):
    timer = timer_screen(text)
    groups = {}
    for line in text.splitlines():
        if not line.startswith('Q6_SAMPLE '): continue
        fields = dict(word.split('=', 1) for word in line.split()[1:])
        key = tuple(int(fields[k]) for k in ('ni', 'no', 'nc'))
        arm, batch, position = (int(fields[k]) for k in ('arm', 'batch', 'position'))
        require(arm == (batch + key[2] + position) % 2 and 0 <= batch < 5 and position in (0, 1),
                'wrong paired batch order')
        bucket = groups.setdefault(key, {0: {}, 1: {}, 'calls': int(fields['calls'])})
        seconds = float(fields['seconds'])
        require(batch not in bucket[arm] and int(fields['calls']) == bucket['calls'] and
            math.isfinite(seconds) and seconds >= .01, 'duplicate/short/nonfinite timing batch')
        bucket[arm][batch] = seconds
    expected = {(ni, no, nc): calls for ni, no, calls in SHAPES for nc in (1, 2, 3, 4, 8)}
    require(set(groups) == set(expected), 'missing/extra model geometry')
    fastest = min(seconds for group in groups.values() for arm in (0, 1)
                  for seconds in group[arm].values())
    require(timer['median_seconds'] / fastest <= .001, 'clock-pair cost exceeds batch budget')
    result = []
    for key, calls in expected.items():
        g = groups[key]
        require(g['calls'] == calls and all(set(g[a]) == set(range(5)) for a in (0, 1)),
                'missing timing batch/call count')
        samples = [[g[a][b] / calls for b in range(5)] for a in (0, 1)]
        result.append(dict(shape=list(key), reference_s=samples[0], candidate_s=samples[1],
            median_time_ratio=statistics.median(samples[1])/statistics.median(samples[0])))
    return result


def cursor():
    text = subprocess.check_output(['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'],
        text=True, timeout=15)
    return next(line.split(': ', 1)[1] for line in text.splitlines() if line.startswith('-- cursor: '))


def fresh_faults(start):
    text = subprocess.check_output(['journalctl', '-k', '-b', '--after-cursor', start,
        '-n', '2048', '--no-pager'], text=True, timeout=15)
    require(len(text.splitlines()) < 2048, 'bounded kernel log incomplete')
    return [line for line in text.splitlines() if re.search(
        r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))', line)]


def run(smoke):
    row = current()
    require(row['stage'] == ('compiled' if smoke else 'smoke-passed'), 'component progression gate')
    require(sha(row['actual_inputs']['path']) == row['actual_inputs']['sha256'], 'build receipt changed')
    require(all(sha(p) == h for p, h in json.loads(Path(row['actual_inputs']['path']).read_text()).items()),
            'actual compiled/linked input changed')
    env = dict(row['environment']); probe = row['native_validation_probe']
    require(all(sha(a['path']) == a['sha256'] for a in probe.values()), 'validation provider changed')
    env.update(LD_PRELOAD=probe['library']['path'], STRATA_UR_PROBE_EXPECTED_LOADER=probe['loader']['path'],
        UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION' if smoke else 'UR_LAYER_PARAMETER_VALIDATION',
        UR_LOG_TRACING='level:info;flush:info;output:stderr' if smoke else 'level:error;flush:error;output:stderr')
    if row['GPU_executed'] is not True:
        row['GPU_executed'] = None
    row.update(stage='running-smoke' if smoke else 'running-speed', active=True,
               ownership_closed=False, GPU_execution_attempted=True); save(row)
    supervisor = None
    try:
        phase = 'smoke' if smoke else 'speed'
        supervisor = make_owner(row, phase)
        deadline = time.monotonic() + (1200 if smoke else 3600)
        for process in range(1 if smoke else 3):
            start = None
            try:
                require(time.monotonic() < deadline, 'whole component deadline')
                start = cursor(); print('Q6 native-half', phase, process, flush=True)
                entry, out, err = supervisor.run('process-' + str(process),
                    [str(EXE), '--smoke' if smoke else '--speed', '--native-half'], env, ROOT,
                    **dict(row['budgets'], wall=min(row['budgets']['wall'], deadline-time.monotonic())))
                text = out.read_text(); count = 32 if smoke else 35
                if 'Q6_TARGET_ENDIAN_LITTLE=1' in text:
                    row['GPU_executed'] = True
                require(owner.closed(entry) and entry['exit_code'] == 0,
                        'component failed: ' + (err.read_text()+text)[-4000:])
                require(text.count('Q6_COMPARISON=native-half') == 1 and text.count('Q6_TARGET_B570=1') == 1 and
                    text.count('Q6_TARGET_FP16=1') == 1 and
                    text.count('Q6_TARGET_ENDIAN_LITTLE=1') == 1 and text.count('Q6_CASE_PASS ') == count and
                    text.count(' scalar_oracle=PASS') == count and text.count(' q8_oracle=PASS') == count and
                    text.count('Q6_COLUMNS_PASS mode=' + ('smoke' if smoke else 'speed')) == 1,
                    'missing comparison/device/numerical/normal-completion evidence')
                proof = validation_log(err.read_text(), entry['pid'], probe['loader']['path'], smoke)
            finally:
                postchecks = native_postchecks(row, start)
                row.setdefault('native_postchecks', []).append(dict(phase=phase, process=process,
                                                                   **postchecks))
                save(row)
            require(postchecks['passed'], 'native post-check failed;bounded evidence retained')
            result = dict(passed_cases=count, native_exit_code=0, ownership_closed=True,
                          validation=proof, fresh_faults=[])
            if smoke: row['smoke'] = result
            else:
                result['timer_screen'] = timer_screen(text)
                result['samples'] = parse_speed(text); row['processes'].append(result)
            save(row)
        if smoke: row['stage'] = 'smoke-passed'
        else:
            decisions = []
            for i in range(35):
                ratios = [p['samples'][i]['median_time_ratio'] for p in row['processes']]
                decisions.append(dict(shape=row['processes'][0]['samples'][i]['shape'],
                    process_time_ratios=ratios, all_three_faster=all(r < 1 for r in ratios)))
            row.update(stage='closed', passed=True, complete=True,
                decision=dict(shapes=decisions, model_adopted=False,
                    next='Only individually faster shapes may enter a new decode-only model plan;preserve prefill.'))
    except BaseException as error:
        row.update(stage='failed-smoke' if smoke else 'failed-speed', error=repr(error)); raise
    finally:
        finish_owner(row, supervisor, 'smoke' if smoke else 'speed')


def main():
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'build', 'smoke', 'speed'),
            'expected prepare/build/smoke/speed')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); guards.no_model_owners([])
        error = None
        try:
            action = sys.argv[1]
            if action == 'prepare': prepare()
            elif action == 'build': build_component()
            else: run(action == 'smoke')
        except BaseException as failure:
            error = failure
            raise
        finally:
            if (ROOT / 'record.json').exists():
                row = json.loads((ROOT / 'record.json').read_text())
                passed = final_owner_guard(row)
                if error is None:
                    require(passed, 'final owner guard failed;terminal failure retained')


if __name__ == '__main__':
    main()
