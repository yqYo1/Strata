#!/usr/bin/env python3
"""Own the complete Q6 centering comparison; Main-authored, repository LICENSE.

Both arms use the existing column geometry and production checks. The old
column-specialization controller/results remain recoverable from Git. No
component action may overlap the frozen six-process IQ3 audit.
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
ROOT = B / 'q6-centering-component-v1'
KEY, COHORT = 'q6_centering_component_v1', 'decode_iq3_audit_speed_v2'
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


def signature(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def compact(row):
    result = {k: v for k, v in row.items() if k not in ('sources', 'runtime_pins', 'commands')}
    result['source_inputs'] = shared.signature(row['sources'])
    result['runtime_inputs'] = shared.signature(row['runtime_pins'])
    return result


def save(row):
    write(ROOT / 'record.json', row)
    data = json.loads(RESULT.read_text()); data[KEY] = compact(row)
    RESULT.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def prepare():
    closed = cohort()
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()), 'fresh attempt only')
    revision = git('rev-parse', 'HEAD').decode().strip()
    tracked = git('ls-files', '-z', 'include', 'tools/sycl').decode().split('\0')
    names = sorted(set(SOURCES) | {n for n in tracked if n and
        (n.startswith('include/') or n.endswith('.py'))} |
        {'AGENTS.md', 'docs/ARTIFACT_RETENTION.md', 'tools/sycl/measurement_environment.json',
         'tests/core/test_q6_columns.py'})
    values = {n: git('show', revision + ':' + n) for n in names}
    require(len(names) <= 1024 and sum(map(len, values.values())) <= 32 << 20 and
        all((P / n).read_bytes() == raw for n, raw in values.items()),
        'commit exact native, fixture and transitive controller sources first')
    native = committed('decode_iq3_pair_model_build_v1')
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
        runtime_pins=runtime, cohort_signature=signature(closed), tools=tools,
        toolchain=dict(path=str(ROOT / 'toolchain.json'), sha256=hashlib.sha256(encoded).hexdigest(),
            files=len(image['files']), bytes=image['bytes'], packages=image['packages']),
        native_recipe_origin=dict(source_commit=native['source_commit'], compile_database=database,
            flags=flags, change='Private source paths plus explicit GCC13 host-header selection;no math/optimization flag change.'),
        compiler_version=subprocess.check_output([original[0], '--version'], env=env, text=True,
            timeout=15).splitlines()[0], environment=env,
        native_validation_probe=closed['native_validation_probe'], commands=commands,
        boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), driver=guards.driver_guard(),
        budgets=dict(wall=1200, cpu=2400, rss_cap=8 << 30, text_cap=16 << 20,
            file_cap=128 << 20, total_cap=768 << 20),
        plan=dict(reference='native_q6_k_columns_mmvq', candidate='native_q6_k_centered_columns_mmvq',
            difference='Q6 centering after 0x0f/0x30 masks only;same byte loads,dot/scales,geometry and reductions.',
            order='One traced smoke,then three fresh processes;five paired batches per shape/column,arm=(batch+column+position)%2.',
            shapes=SHAPES, columns=[1, 2, 3, 4, 8], smoke_cases=32, speed_cases=35,
            timer='Steady-clock full public dispatch and queue completion;resident weights/Q81;conversion,allocation,oracles outside timers.',
            minimum_batch_seconds=.01, process_repetitions=3, batches_per_arm=5,
            acceptance='All output bits,read-only inputs,guards and finite values pass;Q6 scalar and every Q81 byte oracle pass. Exact B570,in-order queue,normal0,quiet validation,closed owners,unchanged inputs/providers/boot and no fresh fault. A shape is eligible only when all three independent median candidate/reference time ratios are below1;no percentage floor.',
            stopping='Stop on first source/provider/owner/numerical/validation/fault/budget/short-timer failure;no retries or hash refresh.',
            limits='Synthetic model geometries and parameter-validation/preload envelope. Logical bytes unchanged;no physical-traffic,saturation,native-ISA,clean model-speed,ordinary runtime-teardown,prefill or262144/RESTORE qualification. No production selector is enabled.'),
        retention='Temporary third project build generation for this imminent component comparison only. Both552b validated prefill fallback and b310 current model reference still have named consumers. Count this fixture with project builds and retire its ELF/objects/source snapshot/traces after recording the decision and verifying source recovery;do not retain it as another reference generation.',
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
    print('Prepared Q6 centering;commit exact plan before build.', flush=True)


def inputs_current(row):
    require(signature(cohort()) == row['cohort_signature'], 'closed cohort evidence changed')
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
    return row


def actual_inputs(row):
    image = json.loads(Path(row['toolchain']['path']).read_text())
    allowed = {v['realpath']: v['sha256'] for v in image['files'].values() if v['kind'] == 'file'}
    allowed.update(row['runtime_pins'])
    allowed.update({str((ROOT / 'source' / n).resolve()): h for n, h in row['sources'].items()})
    names = []
    for dep in sorted((ROOT / 'objects').glob('*.d')):
        names += shlex.split(dep.read_text().replace('\\\n', ' ').split(':', 1)[1])
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
        supervisor = owner.Owner(ROOT / 'build-commands'); deadline = time.monotonic() + 1800
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
        row['active'] = False
        row['ownership_closed'] = supervisor is None or (supervisor.active is None and
            all(owner.closed(e) for e in supervisor.commands)); save(row)


def parse_speed(text):
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
    row.update(stage='running-smoke' if smoke else 'running-speed', active=True,
               ownership_closed=False, GPU_execution_attempted=True); save(row)
    supervisor = None
    try:
        supervisor = owner.Owner(ROOT / ('smoke-commands' if smoke else 'speed-commands'))
        deadline = time.monotonic() + (1200 if smoke else 3600)
        for process in range(1 if smoke else 3):
            require(time.monotonic() < deadline, 'whole component deadline')
            start = cursor(); print('Q6 centering', 'smoke' if smoke else 'speed', process, flush=True)
            entry, out, err = supervisor.run('process-' + str(process),
                [str(EXE), '--smoke' if smoke else '--speed', '--centering'], env, ROOT,
                **dict(row['budgets'], wall=min(row['budgets']['wall'], deadline-time.monotonic())))
            text = out.read_text(); count = 32 if smoke else 35
            row['GPU_executed'] = row['GPU_executed'] or 'Q6_TARGET_ENDIAN_LITTLE=1' in text
            require(owner.closed(entry) and entry['exit_code'] == 0,
                    'component failed: ' + (err.read_text()+text)[-4000:])
            require(text.count('Q6_COMPARISON=centering') == 1 and text.count('Q6_TARGET_B570=1') == 1 and
                text.count('Q6_TARGET_ENDIAN_LITTLE=1') == 1 and text.count('Q6_CASE_PASS ') == count and
                text.count(' scalar_oracle=PASS') == count and text.count(' q8_oracle=PASS') == count and
                text.count('Q6_COLUMNS_PASS mode=' + ('smoke' if smoke else 'speed')) == 1,
                'missing comparison/device/numerical/normal-completion evidence')
            proof = validation_log(err.read_text(), entry['pid'], probe['loader']['path'], smoke)
            faults = fresh_faults(start); require(not faults, 'new GPU fault')
            inputs_current(row)
            result = dict(passed_cases=count, native_exit_code=0, ownership_closed=True,
                          validation=proof, fresh_faults=faults)
            if smoke: row['smoke'] = result
            else: result['samples'] = parse_speed(text); row['processes'].append(result)
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
        row['active'] = False
        row['ownership_closed'] = supervisor is None or (supervisor.active is None and
            all(owner.closed(e) for e in supervisor.commands)); save(row)


def main():
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'build', 'smoke', 'speed'),
            'expected prepare/build/smoke/speed')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); guards.no_model_owners([])
        action = sys.argv[1]
        if action == 'prepare': prepare()
        elif action == 'build': build_component()
        else: run(action == 'smoke')
        guards.no_model_owners([])


if __name__ == '__main__':
    main()
