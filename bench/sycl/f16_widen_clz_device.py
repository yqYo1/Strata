#!/usr/bin/env python3
"""Own the standalone half-widening screen; Main-authored, repository LICENSE.

This fixture/controller is independent of the frozen model benchmark inputs.
Every action waits for its six-process closure. No inference engine is rebuilt.
Semantic/lowering evidence here never establishes conversion or model speed.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

sys.dont_write_bytecode = True
P = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(P / 'tools/sycl'))
import build_official_model as base
import build_q6_centering_model as model
import measurement_guards as guards
import measurement_owner as owner
import run_q6_columns as gate
import ur_validation as validation

B = model.B
ROOT = B / 'q6-f16-widen-device-v1'
KEY = 'q6_f16_widen_device_v1'
COHORT = 'q6_centering_generation_speed_v1'
RESULT = model.RESULT
SELF = Path(__file__).resolve()
CPP = SELF.with_suffix('.cpp')
HEADER = P / 'include/strata/kernels/q6_f16_widen_clz.hpp'
NATIVE = model.ROOT
EXE = ROOT / 'build/sycl_f16_widen_clz_device'
CXX = '/opt/intel/oneapi/compiler/2026.2/bin/icpx'
ARMS = ('reference', 'clz', 'native')
CHECKSUM = '8c8d4b0224882325'
require, digest = guards.require, validation.digest
COMPILE_BUDGET = dict(wall=180, cpu=2160, rss_cap=8 << 30,
                      text_cap=8 << 20, file_cap=32 << 20, total_cap=64 << 20)
DEVICE_BUDGET = dict(wall=60, cpu=720, rss_cap=2 << 30,
                     text_cap=16 << 20, file_cap=32 << 20, total_cap=64 << 20)


def git(*args):
    return subprocess.check_output(['git', *args], cwd=P, timeout=30)


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def controls(revision):
    pins = {}
    for module in (base, model, guards, owner, gate, validation):
        pins.update(base.controls(revision, controller=Path(module.__file__).resolve()))
    for path in (SELF, CPP, HEADER, P / 'AGENTS.md', P / 'docs/ARTIFACT_RETENTION.md',
                 P / 'tools/sycl/measurement_environment.json'):
        name = str(path.relative_to(P))
        raw = git('show', revision + ':' + name)
        require(raw == path.read_bytes(), 'commit/freeze exact screen input: ' + name)
        pins[name] = hashlib.sha256(raw).hexdigest()
    return pins


def closed_cohort():
    row = base.committed(RESULT)[COHORT]
    require(row['stage'] == 'closed' and row['complete'] and row['passed']
            and row['ownership_closed'] and not row['active']
            and len(row['processes']) == 6
            and [p['index'] for p in row['processes']] == list(range(6))
            and all(p['complete'] and p['passed'] and p['ownership_closed']
                and p['native_exit_code'] == 0 and p['fresh_faults'] == []
                and p['centering_route']['passed'] for p in row['processes']),
            'close/adjudicate all six frozen centering processes first')
    # The new converter is orthogonal to centering. Execution closure here does
    # not require or assert that the model centering performance gate passed.
    require('centering_model_TG_gain_qualified' in row['decision'],
            'record the original paired performance decision first')
    return row


def compact(row):
    out = {k: v for k, v in row.items() if k not in
           ('sources', 'commands_executed', 'allowed_inputs', 'compiler_dependencies')}
    for key in ('sources', 'allowed_inputs', 'compiler_dependencies'):
        if key in row:
            out[key] = dict(files=len(row[key]), aggregate_sha256=signature(row[key]))
    return out


def save(row):
    base.write(ROOT / 'record.json', row)
    data = json.loads(RESULT.read_text())
    data[KEY] = compact(row)
    base.write(RESULT, data)


def environment(prior):
    names = ('HOME', 'LANG', 'LC_ALL', 'LD_LIBRARY_PATH', 'ONEAPI_DEVICE_SELECTOR',
             'UR_ADAPTERS_FORCE_LOAD', 'UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD',
             'NEOReadDebugKeys', 'EnableDirectSubmission', 'STRATA_SYCL_PREFILL_SAME_QUEUE',
             'STRATA_SYCL_VERIFY_NATIVE_CAPTURE', 'SYCL_CACHE_PERSISTENT',
             'ZES_ENABLE_SYSMAN', 'ZE_ENABLE_ALT_DRIVERS')
    env = {k: prior['environment'][k] for k in names}
    env.update(PATH='/usr/bin:/bin', TMPDIR=str(ROOT / 'tmp'),
               XDG_CACHE_HOME=str(ROOT / 'cache'))
    require(env['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD'] == env['NEOReadDebugKeys'] == '1'
            and env['EnableDirectSubmission'] == env['STRATA_SYCL_VERIFY_NATIVE_CAPTURE']
                == env['SYCL_CACHE_PERSISTENT'] == env['ZES_ENABLE_SYSMAN'] == '0'
            and env['STRATA_SYCL_PREFILL_SAME_QUEUE'] == '1'
            and '/MistVVK/' not in json.dumps(env), 'qualified official safety environment required')
    return env


def recipe():
    runtime = NATIVE / 'source/src/sycl/runtime.cpp'
    flags = ['-std=c++20', '-O3', '-fsycl', '-fsycl-id-queries-range=size_t',
             '--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13',
             '-fno-fast-math', '-ffp-contract=off', '-Wall', '-Wextra', '-Werror',
             '-save-temps=obj', '-I' + str(NATIVE / 'source/include'), '-I' + str(P / 'include')]
    commands = []
    for name, source in (('runtime', runtime), ('fixture', CPP)):
        commands.append([name, [CXX, *flags, '-MD', '-MF', str(ROOT / 'build' / (name + '.d')),
                                '-c', str(source), '-o', str(ROOT / 'build' / (name + '.o'))]])
    commands.append(['compile', [CXX, '-fsycl', '-save-temps=obj',
        str(ROOT / 'build/runtime.o'), str(ROOT / 'build/fixture.o'), '-Wl,-t', '-o', str(EXE)]])
    return commands


def prepare():
    cohort = closed_cohort()
    require(not ROOT.exists() and KEY not in base.committed(RESULT), 'fresh screen only;no retry')
    revision = git('rev-parse', 'HEAD').decode().strip()
    pins = controls(revision)
    native = json.loads((NATIVE / 'record.json').read_text())
    # Reuse the original frozen native/toolchain inputs, not a nearby checkout.
    allowed = model.inputs(native)
    prior_path = NATIVE / 'model-entry-centering0-v1/record.json'
    prior = json.loads(prior_path.read_text())
    require(prior['native_source_commit'] == native['source_commit']
            and prior['binary'] == cohort['binary'] == native['binaries']['strata']
            and prior['outcome']['native_exit_code'] == 0 and prior['passed']
            and prior['ownership_closed'] and not prior['active'], 'qualified exact native basis required')
    for path in (CPP, HEADER):
        allowed[str(path.resolve())] = pins[str(path.relative_to(P))]
    providers = prior['runtime_manifest']
    require(digest(providers['path']) == providers['sha256'], 'qualified provider inventory changed')
    provider_files = json.loads(Path(providers['path']).read_text())
    require(all(digest(p) == h for p, h in provider_files.items()), 'qualified provider bytes changed')
    probe = validation.preload_qualification()
    require(all(probe[k] == prior['native_validation_probe'][k] for k in ('loader', 'library')),
            'qualified validation probe changed')
    row = dict(stage='prepared', source_commit=revision, sources=pins,
        native_source_commit=native['source_commit'], native_record=dict(path=str(NATIVE / 'record.json'),
        sha256=digest(NATIVE / 'record.json')), toolchain_image=native['toolchain_image'],
        prior_record=dict(path=str(prior_path), sha256=digest(prior_path)),
        allowed_inputs=allowed, providers=providers, native_validation_probe=probe,
        configuration=dict(arms=list(ARMS), patterns=65536, offsets=[0, 1], timing=False),
        environment=environment(prior), commands=recipe(),
        compiler=dict(path=CXX, sha256=digest(CXX)),
        interpreter=dict(path=str(Path(sys.executable).resolve()), sha256=digest(sys.executable),
                         version=sys.version),
        boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), driver=guards.driver_guard(),
        compile_budget=COMPILE_BUDGET, device_budget=DEVICE_BUDGET,
        parser_controls=parser_controls(),
        original_cohort_decision=cohort['decision']['centering_model_TG_gain_qualified'],
        criteria='Reference/CLZ raw-bit exact over all65536patterns at both byte offsets;native finite/zero/subnormal/Inf exact and NaN class/sign. Native payload/quiet changes descriptive. Read-only input and guards unchanged. Require normal exit,pre-main validation,unchanged inputs/providers/boot and no fresh fault. Preserve any failed arm;no retry,hash refresh or timing.',
        retention='One-use standalone fixture,not an inference generation. Keep exact source/recipe and compact semantic/ownership/fault result. Retire normal output and fixture after exact lowering/code-choice consumer closes;both inference generations remain unchanged.',
        build_passed=False, passed=False, complete=False, active=False, ownership_closed=True,
        GPU_executed=False, model_executed=False)
    ROOT.mkdir(mode=0o700)
    for name in ('build', 'tmp', 'cache'): (ROOT / name).mkdir()
    save(row)


def current(row):
    require(base.committed(RESULT)[KEY] == compact(row), 'commit exact screen plan/result first')
    closed_cohort()
    require(controls(row['source_commit']) == row['sources'], 'screen sources changed')
    require(digest(row['native_record']['path']) == row['native_record']['sha256']
            and digest(row['compiler']['path']) == row['compiler']['sha256']
            and str(Path(sys.executable).resolve()) == row['interpreter']['path']
            and sys.version == row['interpreter']['version']
            and digest(sys.executable) == row['interpreter']['sha256'], 'native/compiler/interpreter changed')
    native = json.loads(Path(row['native_record']['path']).read_text())
    require(digest(row['prior_record']['path']) == row['prior_record']['sha256'],
            'qualified model-entry environment changed')
    prior = json.loads(Path(row['prior_record']['path']).read_text())
    require(row['environment'] == environment(prior)
            and row['compile_budget'] == COMPILE_BUDGET and row['device_budget'] == DEVICE_BUDGET
            and row['commands'] == recipe()
            and row['configuration'] == dict(arms=list(ARMS), patterns=65536, offsets=[0, 1], timing=False),
            'screen environment/budget/extent differs')
    model.toolchain(native)
    require(all(digest(p) == h for p, h in row['allowed_inputs'].items()), 'prebuild input changed')
    require(digest(row['providers']['path']) == row['providers']['sha256'], 'provider manifest changed')
    require(all(digest(p) == h for p, h in json.loads(Path(row['providers']['path']).read_text()).items()),
            'provider bytes changed')
    require(row['boot'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            and row['driver'] == guards.driver_guard(), 'boot/device changed')


def dependencies(row):
    actual = {}
    for name in ('runtime', 'fixture'):
        tokens = shlex.split((ROOT / 'build' / (name + '.d')).read_text().replace('\\\n', ''))
        require(tokens and tokens[0] == str(ROOT / 'build' / (name + '.o')) + ':',
                'compiler dependency target differs')
        for item in tokens[1:]:
            path = Path(item)
            require(path.is_absolute(), 'absolute compiler dependency required')
            if path.is_relative_to(ROOT / 'tmp') or path.is_relative_to(ROOT / 'build'):
                continue  # Owned integration/generated headers, not external inputs.
            real = str(path.resolve(strict=True))
            require(row['allowed_inputs'].get(real) == digest(path), 'unfrozen actual include: ' + item)
            actual[real] = row['allowed_inputs'][real]
    require(str(CPP) in actual and str(HEADER) in actual and len(actual) > 100,
            'complete actual fixture/header coverage required')
    return actual


def launch_inputs(row, action):
    commands, env = row['commands'], dict(row['environment'])
    cursor = None
    if action == 'check':
        require(digest(EXE) == row['binary']['sha256'], 'compiled screen executable changed')
        probe = validation.preload_qualification()
        require(probe == row['native_validation_probe'], 'pre-main validation basis changed')
        env.update(LD_PRELOAD=probe['library']['path'],
            UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
            UR_LOG_TRACING='level:info;flush:info;output:stderr',
            UR_LOG_VALIDATION='level:warning;flush:warning;output:stderr',
            UR_LOG_LOADER='level:warning;flush:warning;output:stderr',
            UR_LOG_LEVEL_ZERO='level:warning;flush:warning;output:stderr')
        lines = subprocess.check_output(['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'],
                                        text=True, timeout=15).splitlines()
        cursor, = [v.split(': ', 1)[1] for v in lines if v.startswith('-- cursor: ')]
        commands = [('semantic', [str(EXE)])]
    return commands, env, cursor


def execute(row, action):
    require((action == 'build' and row['stage'] == 'prepared') or
            (action == 'check' and row['stage'] == 'built' and row['build_passed']),
            'fresh ordered build/check only;no retry')
    try:
        current(row)  # All committed-input checks precede publishing an active result.
        commands, env, cursor = launch_inputs(row, action)
    except BaseException as error:
        row.update(stage='failed-admission', passed=False, complete=False, error=repr(error))
        save(row)
        raise
    supervisor = None
    receipt = ROOT / (action + '-ownership.json')
    def persist():
        fields = ('label', 'pid', 'direct_identity', 'owners', 'session_anchored',
            'session_empty', 'direct_child_reaped', 'exit_code', 'ownership_status',
            'errors', 'error', 'cleanup', 'survivors', 'started_utc', 'finished_utc', 'logs')
        value = dict(active_pid=supervisor.active.pid if supervisor and supervisor.active else None,
            commands=[{k: c[k] for k in fields if k in c}
                      for c in supervisor.commands] if supervisor else [])
        text = json.dumps(value, indent=2) + '\n'
        require(len(text.encode()) <= 4 << 20, 'bounded live ownership receipt')
        pending = receipt.with_suffix('.tmp')
        pending.write_text(text)
        pending.replace(receipt)
    row.update(stage='building' if action == 'build' else 'checking', active=True,
               ownership_closed=False)
    save(row)
    try:
        supervisor = owner.Owner(ROOT / ('owned-build' if action == 'build' else 'owned-device'), persist)
        for name, argv in commands:
            if action == 'check':
                row['device_process_attempted'] = True
                row['GPU_executed'] = None  # No absence claim before a complete device report.
            child, out, err = supervisor.run(name, argv, env, ROOT,
                **(row['compile_budget'] if action == 'build' else row['device_budget']))
            if action == 'check':
                text = err.read_text()
                row['semantic_result'] = parse(out.read_text())
                row['GPU_executed'] = True  # All six arm reports follow checked device completion.
                row['validation'] = gate.validation_log(text, child['pid'],
                    row['native_validation_probe']['loader']['path'], traced=True)
            require(owner.closed(child) and child['exit_code'] == 0, 'owned child failed: ' + name)
        require(controls(row['source_commit']) == row['sources']
                and all(digest(p) == h for p, h in row['allowed_inputs'].items()), 'post-action inputs changed')
        require(all(digest(row[k]['path']) == row[k]['sha256']
                    for k in ('native_record', 'prior_record')), 'post-action qualification records changed')
        require(digest(row['providers']['path']) == row['providers']['sha256']
                and all(digest(p) == h for p, h in
                        json.loads(Path(row['providers']['path']).read_text()).items()),
                'post-action provider inputs changed')
        model.toolchain(json.loads(Path(row['native_record']['path']).read_text()))
        if action == 'build':
            row['compiler_dependencies'] = dependencies(row)
            row['actual_link_inputs'] = base.traced_link_inputs(row['allowed_inputs'], ROOT)
            row['binary'] = dict(path=str(EXE), sha256=digest(EXE), bytes=EXE.stat().st_size)
            row['build_outcome'] = [dict(label=c['label'], exit_code=c['exit_code'],
                ownership_closed=owner.closed(c)) for c in supervisor.commands]
            row.update(build_passed=True, stage='built')
        else:
            require(row['semantic_result']['passed'], 'one or more semantic contracts failed')
            row.update(host_probe_accepted=True, complete=True, stage='closed')
    except BaseException as error:
        row.update(stage='failed', passed=False, complete=False, error=repr(error))
    finally:
        if action == 'check' and supervisor:
            validation.finish(row, supervisor, cursor, row['boot'])
            if not row['passed'] or row.get('driver_after') != row['driver']:
                row.update(stage='failed', passed=False, complete=False)
        row['ownership_closed'] = supervisor is None or (supervisor.active is None
            and all(owner.closed(c) for c in supervisor.commands))
        row['active'] = bool(supervisor and supervisor.active)
        row['commands_executed'] = supervisor.commands if supervisor else []
        persist()
        row.setdefault('ownership_receipts', {})[action] = dict(path=str(receipt),
            sha256=digest(receipt), closed=row['ownership_closed'])
        save(row)
    require(row['stage'] != 'failed' and row['ownership_closed'], row.get('error', 'screen action failed'))


def parse(text):
    lines = text.splitlines()
    require(len(lines) == 8 and lines[0] == 'F16_WIDEN_TARGET_B570=1 timing=0 offsets=0,1',
            'complete bounded semantic output required')
    rows = []
    fields = ('patterns', 'finite_mismatches', 'subnormal_mismatches', 'zero_mismatches',
              'infinity_mismatches', 'nan_class_mismatches', 'nan_sign_mismatches',
              'nan_payload_changes', 'nan_quiet_changes', 'signaling_quieted',
              'signaling_preserved', 'bit_mismatches')
    for i, line in enumerate(lines[1:7]):
        require(line.startswith('F16_WIDEN_ARM '), 'missing arm record')
        pairs = [s.split('=', 1) for s in line.split()[1:]]
        require(all(len(p) == 2 for p in pairs) and len({p[0] for p in pairs}) == len(pairs),
                'duplicate/malformed arm field')
        row = dict(pairs)
        require(set(row) == {'arm', 'offset', 'status', 'checksum', *fields}, 'unexpected arm fields')
        require(row['arm'] == ARMS[i % 3] and row['offset'] == str(i // 3)
                and row['status'] in ('PASS', 'FAIL') and re.fullmatch('[0-9a-f]{16}', row['checksum']),
                'arm identity/status/checksum differs')
        for name in fields:
            require(row[name].isdecimal(), 'unsigned semantic counter required')
            row[name] = int(row[name])
        limits = dict(finite_mismatches=63488, subnormal_mismatches=2046,
            zero_mismatches=2, infinity_mismatches=2, nan_class_mismatches=2046,
            nan_sign_mismatches=2046, nan_payload_changes=2046, nan_quiet_changes=2046,
            signaling_quieted=1022, signaling_preserved=1022, bit_mismatches=65536)
        require(row['patterns'] == 65536 and all(0 <= row[n] <= limit for n, limit in limits.items()),
                'semantic extent/counter out of range')
        require(row['zero_mismatches'] + row['subnormal_mismatches'] <= row['finite_mismatches']
                and row['finite_mismatches'] + row['infinity_mismatches'] <= row['bit_mismatches']
                and all(row[n] <= row['bit_mismatches'] for n in
                    ('nan_class_mismatches', 'nan_sign_mismatches', 'nan_payload_changes', 'nan_quiet_changes')),
                'semantic counters do not conserve mismatches')
        exact = row['bit_mismatches'] == 0 and row['checksum'] == CHECKSUM
        native = all(row[n] == 0 for n in ('finite_mismatches', 'infinity_mismatches',
            'nan_class_mismatches', 'nan_sign_mismatches'))
        passed = exact if i % 3 < 2 else native
        require((row['status'] == 'PASS') == passed, 'status and observed contract disagree')
        require(row['signaling_quieted'] + row['signaling_preserved'] <= 1022,
                'signaling population exceeds binary16 count')
        if row['nan_class_mismatches'] == 0:
            require(row['signaling_quieted'] + row['signaling_preserved'] == 1022,
                    'all signaling inputs must be accounted for')
        row['passed'] = passed
        rows.append(row)
    passed = all(r['passed'] for r in rows)
    require(lines[-1] == 'F16_WIDEN_DEVICE_STATUS=' + ('PASS' if passed else 'FAIL') +
        ' patterns_per_arm_offset=65536 arms=3 offsets=2 readonly_and_guards=PASS timing=0',
        'final aggregate/guard status differs')
    return dict(passed=passed, arms=rows, patterns_per_arm_offset=65536,
                offsets=[0, 1], guards_passed=True, input_unchanged=True, timing=False)


def parser_controls():
    # Exercise malformed/misclassified output, not a copy of the GPU converter.
    exact = ('patterns=65536 status=PASS finite_mismatches=0 subnormal_mismatches=0 '
        'zero_mismatches=0 infinity_mismatches=0 nan_class_mismatches=0 nan_sign_mismatches=0 '
        'nan_payload_changes=0 nan_quiet_changes=0 signaling_quieted=0 signaling_preserved=1022 '
        'bit_mismatches=0 checksum=' + CHECKSUM)
    lines = ['F16_WIDEN_TARGET_B570=1 timing=0 offsets=0,1']
    lines += [f'F16_WIDEN_ARM arm={arm} offset={offset} ' + exact
              for offset in (0, 1) for arm in ARMS]
    lines += ['F16_WIDEN_DEVICE_STATUS=PASS patterns_per_arm_offset=65536 '
              'arms=3 offsets=2 readonly_and_guards=PASS timing=0']
    original = '\n'.join(lines)
    require(parse(original)['passed'], 'exact native policy control')
    quiet = original.replace('arm=native offset=0 ' + exact,
        'arm=native offset=0 ' + exact.replace('nan_quiet_changes=0', 'nan_quiet_changes=1022')
            .replace('signaling_quieted=0', 'signaling_quieted=1022')
            .replace('signaling_preserved=1022', 'signaling_preserved=0')
            .replace('bit_mismatches=0', 'bit_mismatches=1022').replace(CHECKSUM, '0000000000000001'))
    require(parse(quiet)['passed'], 'native NaN quieting is allowed')
    failed = original.replace('arm=native offset=0 ' + exact,
        'arm=native offset=0 ' + exact.replace('status=PASS', 'status=FAIL')
            .replace('finite_mismatches=0', 'finite_mismatches=1')
            .replace('bit_mismatches=0', 'bit_mismatches=1'))
    failed = failed.replace('F16_WIDEN_DEVICE_STATUS=PASS', 'F16_WIDEN_DEVICE_STATUS=FAIL')
    result = parse(failed)
    require(not result['passed'] and result['arms'][0]['passed'] and result['arms'][1]['passed'],
            'keep failed native and independent reference/CLZ results')
    bad = [original.replace(lines[1] + '\n', ''),
        original.replace('arm=clz offset=0', 'arm=native offset=0'),
        original.replace('patterns=65536 status', 'patterns=65536 patterns=65536 status', 1),
        original.replace(CHECKSUM, '0000000000000001', 1),
        original.replace('zero_mismatches=0', 'zero_mismatches=1', 1),
        original.replace('signaling_preserved=1022', 'signaling_preserved=1023', 1),
        original.replace('readonly_and_guards=PASS', 'readonly_and_guards=FAIL')]
    for text in bad:
        try:
            parse(text)
        except RuntimeError:
            continue
        raise RuntimeError('invalid semantic report accepted')
    return dict(passed=True, valid=3, invalid_rejected=len(bad), GPU_executed=False)


if __name__ == '__main__':
    require(sys.argv[1:] in (['prepare'], ['build'], ['check']), 'use prepare|build|check')
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners([])
        if sys.argv[1] == 'prepare':
            prepare()
        else:
            execute(json.loads((ROOT / 'record.json').read_text()), sys.argv[1])
