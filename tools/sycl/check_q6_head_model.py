#!/usr/bin/env python3
"""Qualify the production-linked Q6 head before a full-model trial.

Main-authored; repository LICENSE. Reuse the exact closed native GPU gate
functions from484d, with fresh binary/source/runtime checks and fresh processes.
No old executable or private standalone source tree is a dependency.
"""
import ast
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import re
import select
import subprocess
import sys

sys.dont_write_bytecode = True
import benchmark_runtime as python_runtime
import build_q6_head_model as build
import measurement_guards as guards
import measurement_owner as owner
import ur_validation as validation
from q6_head_owner import parse_native, finish_gpu_probe
from run_q6_columns import validation_log
from check_verify_window_phase import SUBMISSION, PHASE

P, B, RESULT = build.P, build.B, build.RESULT
ROOT, KEY = B / 'q6-head-model-device-v1', 'q6_head_model_device_v1'
require, sha, write = guards.require, build.digest, build.write
ORIGINAL = '484d403e60c49361422f00a16dfdd774e4fe90d5'
ORIGINAL_PATH = 'tools/sycl/q6_head_owner.py'
RUNTIME = B / 'decode-iq3-audit-speed-v2/runtime-inputs.json'
RUNTIME_SHA = '43de9b527585addcc6da1f8fdc6f1e8266e2d982ba4c752d909578ed8c7bde21'


def archived_gate():
    raw = build.source.git(P, 'show', ORIGINAL + ':' + ORIGINAL_PATH)
    tree = ast.parse(raw)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and
             n.name in ('model_assets', 'run_gpu')]
    require([n.name for n in nodes] == ['model_assets', 'run_gpu'], 'exact archived GPU functions required')
    namespace = dict(globals())
    exec(compile(ast.Module(body=nodes, type_ignores=[]), ORIGINAL + ':' + ORIGINAL_PATH, 'exec'), namespace)
    return namespace['run_gpu'], hashlib.sha256(raw).hexdigest()


def compact(row):
    out = {k: v for k, v in row.items() if k not in ('controller', 'runtime_pins', 'commands')}
    out['controller'] = {k: v for k, v in row['controller'].items() if k != 'source_sha256'}
    out['commands'] = [dict(label=c['label'], exit_code=c.get('exit_code'),
        ownership_closed=owner.closed(c), error=c.get('error'), errors=c.get('errors', []))
        for c in row['commands']]
    return out


def save(row):
    write(ROOT / 'record.json', row)
    d = json.loads(RESULT.read_text()); d[KEY] = compact(row); write(RESULT, d)


def native():
    row = json.loads((build.ROOT / 'record.json').read_text()); build.current(row)
    require(row['stage'] == 'built' and row['passed'] and row['complete'] and
        row['ownership_closed'] and row['actual_inputs']['uncovered'] == 0, 'closed full-engine build required')
    for binary in row['binaries'].values():
        require(sha(binary['path']) == binary['sha256'] and
            Path(binary['path']).stat().st_size == binary['bytes'], 'fresh same-build executable changed')
    actual = row['actual_inputs']
    require(sha(actual['path']) == actual['sha256'], 'actual-input inventory changed')
    return row


def verify(row):
    require(validation.committed(RESULT, (KEY,)) == compact(row), 'commit exact device plan/result first')
    pin = row['controller']
    require(build.signature(pin['source_sha256']) == pin['aggregate_sha256'] and
        {p: hashlib.sha256(raw).hexdigest() for p, raw in build.source.blobs(P,
            pin['revision'], pin['source_sha256'])} == pin['source_sha256'] and
        all(sha(P / p) == h for p, h in pin['source_sha256'].items()), 'device controller changed')
    n = native()
    require(sha(build.ROOT / 'record.json') == row['native_record']['sha256'] and
        n['binaries']['sycl_q6_head_owner_test'] == row['binary'], 'native build changed')
    require(sha(RUNTIME) == RUNTIME_SHA and json.loads(RUNTIME.read_text()) == row['runtime_pins'] and
        all(sha(p) == h for p, h in row['runtime_pins'].items()), 'qualified runtime changed')
    require(archived_gate()[1] == row['archived_gate']['sha256'], 'archived gate changed')
    for asset in row['native_validation_probe'].values():
        if isinstance(asset, dict) and 'path' in asset:
            require(sha(asset['path']) == asset['sha256'], 'native preload/provider changed')
    require(guards.driver_guard() == row['driver'] and
        Path('/proc/sys/kernel/random/boot_id').read_text().strip() == row['boot'], 'boot/driver changed')
    guards.no_model_owners([a['path'] for a in row['assets']])


def prepare():
    require(not ROOT.exists() and KEY not in build.public(), 'fresh device plan only')
    n = native(); old = build.public()['q6_head_owner_v2']
    require(old['passed'] and old['complete'] and old['ownership_closed'], 'closed owner prerequisite')
    require(sha(RUNTIME) == RUNTIME_SHA, 'fixed84ELF runtime manifest required')
    env = copy.deepcopy(old['environment'])
    env.update(TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'))
    require(env['STRATA_PREFILL_RELEASE_HEADS'] == '1' and env['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD'] == '1'
        and env['EnableDirectSubmission'] == '0' and env['STRATA_SYCL_VERIFY_NATIVE_CAPTURE'] == '0',
        'keep qualified immutable safety/head environment')
    row = dict(stage='prepared', source_commit=build.source.git(P, 'rev-parse', 'HEAD').decode().strip(),
        controller=build.source.pin(P, ['tools/sycl']), native_source_commit=n['project']['revision'],
        native_record=dict(path=str(build.ROOT / 'record.json'), sha256=sha(build.ROOT / 'record.json')),
        binary=n['binaries']['sycl_q6_head_owner_test'], binaries=n['binaries'],
        archived_gate=dict(commit=ORIGINAL, path=ORIGINAL_PATH, sha256=archived_gate()[1],
            functions=['model_assets', 'run_gpu']),
        environment=env, assets=old['assets'], runtime_manifest=dict(path=str(RUNTIME), sha256=RUNTIME_SHA),
        runtime_pins=json.loads(RUNTIME.read_text()), native_validation_probe=old['native_validation_probe'],
        boot=old['boot'], driver=old['driver'], native_budget=old['native_budget'],
        host_budget=old['host_budget'],
        plan='Five fresh CPU-only empty-owner/invalid-switch processes,then submission10 and phase-cache16'
             ' traced checks,then fresh reference0/packed1 real-GGUF owner processes with the exact484d'
             ' two-gate/native-body/finite-budget controller. Full shard hashes checked before and after'
             ' each head process. Four numerical cases,two retire/overwrite/restore cycles,12graph replays'
             ' per arm. Commit host closure before device. Stop first failure;no retries.',
        limits='Fresh production-linked fixture only. Not full model/production Verifier ownership,clean'
               ' PP/TTFT/TG,task quality,physical262144/RESTORE,ordinary teardown or default adoption.',
        retention='Current model-entry prerequisite. Retire normal traces/caches and obsolete standalone'
                  ' source/build roots after committed useful results and exact input recovery;preserve'
                  ' selected90f and552b/e1b phase guards. No new engine copy.',
        results=[], commands=[], passed=False, complete=False, active=False, ownership_closed=True,
        GPU_executed=False, model_executed=False)
    ROOT.mkdir(mode=0o700)
    for name in ('tmp', 'cache'): (ROOT / name).mkdir(mode=0o700)
    save(row)
    print('Prepared production-linked head device gate;commit before host.', flush=True)


def execute(mode):
    row = json.loads((ROOT / 'record.json').read_text()); verify(row)
    require(row['stage'] == ('prepared' if mode == 'host' else 'host_passed'), 'no retry/duplicate device attempt')
    scope = None
    def persist():
        write(ROOT / 'execution.json', dict(active=bool(scope and scope.active),
            commands=scope.commands if scope else []))
    try:
        scope = owner.Owner(ROOT / ('owned-' + mode), persist)
        if mode == 'host':
            cases = [('empty', ['--host'], {}, 'HOST owner_negative_cases=3 GPU=0'),
                ('flag2', ['--reject-switch', 'must be 0 or 1'], {'STRATA_SYCL_DECODE_Q6_DPAS_HEAD': '2'}, 'REJECT_SWITCH passed=1 GPU=0'),
                ('empty-flag', ['--reject-switch', 'must be 0 or 1'], {'STRATA_SYCL_DECODE_Q6_DPAS_HEAD': ''}, 'REJECT_SWITCH passed=1 GPU=0'),
                ('release0', ['--reject-switch', 'requires STRATA_PREFILL_RELEASE_HEADS=1'],
                    {'STRATA_SYCL_DECODE_Q6_DPAS_HEAD': '1', 'STRATA_PREFILL_RELEASE_HEADS': '0'}, 'REJECT_SWITCH passed=1 GPU=0'),
                ('release-absent', ['--reject-switch', 'requires STRATA_PREFILL_RELEASE_HEADS=1'],
                    {'STRATA_SYCL_DECODE_Q6_DPAS_HEAD': '1', 'STRATA_PREFILL_RELEASE_HEADS': None}, 'REJECT_SWITCH passed=1 GPU=0')]
            for label, args, changes, expected in cases:
                env = dict(row['environment']); env.pop('LD_PRELOAD', None)
                for k, v in changes.items():
                    if v is None: env.pop(k, None)
                    else: env[k] = v
                e, out, err = scope.run(label, [row['binary']['path'], *args], env, ROOT, **row['host_budget'])
                require(owner.closed(e) and e['exit_code'] == 0 and out.read_text().strip() == expected and
                    not re.search(r'--->\s*ur[A-Z]|<---\s*ur[A-Z]', err.read_text()), 'CPU owner/switch gate: ' + label)
                row['results'].append(dict(mode=label, passed=True, ownership_closed=True, GPU_executed=False))
            row['stage'] = 'host_passed'
        else:
            env = dict(row['environment'], LD_PRELOAD=row['native_validation_probe']['library']['path'],
                STRATA_UR_PROBE_EXPECTED_LOADER=row['native_validation_probe']['loader']['path'])
            cursor = next(v.split(': ', 1)[1] for v in subprocess.check_output(
                ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15).splitlines()
                if v.startswith('-- cursor: '))
            for label, target, args, expected in [
                ('submission', 'sycl_submission_test', ['--submission'], SUBMISSION),
                ('phase', 'sycl_verifier_retirement_test', ['--phase-cache'], [PHASE, 'phase_fixture_status=PASS cases=1'])]:
                outcome = dict(mode=label, passed=False)
                try:
                    e, out, err = scope.run(label, [row['binaries'][target]['path'], *args], env, ROOT, **row['native_budget'])
                    text, diagnostic = out.read_text(), err.read_text()
                    require(owner.closed(e) and e['exit_code'] == 0, 'small native fixture failed: ' + label)
                    require(([v for v in text.splitlines() if v.startswith('case ')] == expected and
                        text.count('fixture_status=PASS mode=submission') == 1) if label == 'submission'
                        else text.splitlines() == expected, 'small fixture extent differs: ' + label)
                    outcome.update(validation=validation_log(diagnostic, e['pid'],
                        row['native_validation_probe']['loader']['path'], True), host_probe_accepted=True, native_exit_code=0)
                finally:
                    validation.finish(outcome, scope, cursor, row['boot'])
                    row['results'].append(outcome)
                require(outcome['passed'], 'small fixture/fault/closure gate failed: ' + label)
                print('Production-linked device gate passed: ' + label, flush=True)
            run_gpu, _ = archived_gate()
            for arm in ('reference', 'packed'):
                outcome = dict(mode=arm, passed=False)
                row['results'].append(outcome)
                run_gpu(row, scope, arm, outcome)
                print('Production-linked device gate passed: ' + arm, flush=True)
            row.update(stage='closed', passed=True, complete=True)
    except BaseException as error:
        row.update(stage='failed', passed=False, complete=False, error=repr(error))
    finally:
        row['commands'] += scope.commands if scope else []
        row['active'] = bool(scope and scope.active)
        row['ownership_closed'] = not row['active'] and all(owner.closed(c) for c in row['commands'])
        row['passed'] = bool(row['passed'] and row['ownership_closed'])
        save(row)
    require(row['stage'] != 'failed' and row['ownership_closed'], row.get('error', 'device gate did not close'))
    print('Closed production-linked head ' + mode + ';commit before advancing.', flush=True)


if __name__ == '__main__':
    require(sys.argv[1:] in (['prepare'], ['host'], ['device']), 'check_q6_head_model.py prepare|host|device')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare() if sys.argv[1] == 'prepare' else execute(sys.argv[1])
