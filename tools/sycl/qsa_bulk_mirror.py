#!/usr/bin/env python3
"""Build and screen the opt-in streamed QSA Host-mirror transfer candidate.

Main-authored from the maintained owned build/screen controls; repository LICENSE.
This component is a prerequisite for a model comparison, not a speed claim.
"""
import fcntl
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
import check_qsa16_stream as original

P, B, RESULT = original.P, original.B, original.RESULT
ROOT, KEY = B / 'qsa16-bulk-mirror-components-v1', 'qsa16_bulk_mirror_components_v1'
FLAG = 'STRATA_SYCL_PREFILL_QSA_BULK_HOST_MIRROR'
CASES = [('small-off', 'small', '0'), ('small-on', 'small', '1'),
         ('full-off', 'full', '0'), ('full-on', 'full', '1')]
NATIVE = B / 'qsa16-stream-model-native-v2/record.json'
CHANGED = {'src/prefill/sycl/qsa_native16.cpp', 'include/strata/prefill/qsa_native16.hpp',
           'tests/sycl/qsa_native16.cpp'}
require, sha, write = original.require, original.sha, original.write


def context():
    spec = importlib.util.spec_from_file_location('qsa_bulk_private_component', original.__file__)
    ctx = importlib.util.module_from_spec(spec); spec.loader.exec_module(ctx)
    ctx.ROOT, ctx.KEY = ROOT, KEY
    return ctx


def prepare():
    ctx = context()
    native = json.loads(NATIVE.read_text())
    public = json.loads(ctx.common.source.git(P, 'show', 'HEAD:' + str(RESULT.relative_to(P))))
    require(public['qsa16_stream_model_native_v2'] == ctx.compact(native) and
        native['stage'] == 'built' and native['passed'] and native['complete'] and
        native['ownership_closed'] and not native['active'], 'preserve exact current b9aa generation')
    before = native['project']['source_sha256']
    after = ctx.common.source.pin(P, ctx.common.PREFIXES)['source_sha256']
    prefixes = ('src/', 'include/', 'tests/', 'cmake/', 'third_party/')
    changed = {p for p in set(before) | set(after) if
        (p.startswith(prefixes) or p == 'CMakeLists.txt') and before.get(p) != after.get(p)}
    require(changed == CHANGED, 'only mirror production/header and partial-page oracle may change')
    ctx.prepare()
    row = json.loads((ROOT / 'record.json').read_text())
    row['preserved_binaries'] += list(native['binaries'].values())
    row.update(native_reference=dict(path=str(NATIVE), sha256=sha(NATIVE),
        native_source_commit=native['native_source_commit'], binaries=native['binaries']),
        native_changes={p: after[p] for p in sorted(changed)}, cases=CASES,
        plan='Commit every build/result stage. Reuse exact qualified compiler/SDK/safety recipe;build only the existing five-TU native16 QSA fixture. Fresh immutable processes in order small-off,small-on,full-off,full-on. Small uses tracing+parameter validation;full uses parameter validation+stage receipts. Independent existing dyadic code/scale oracle,nonfinite handling,owner/alias/domain refusals,all five partial-page offsets,physical262144 full fill and last-cell overwrite must pass with normal0,closed owners,quiet validation,unchanged boot/driver and no fresh fault. Candidate receipts must match accepted copy counts/bytes and full262144+1written cells. No environment mutation,retry,pin refresh,model/performance/default adoption.',
        retention='Three existing named engines552b/a347/b9aa stay preserved;this is one temporary component fixture,not a fourth engine. Its sole consumer is the imminent same-source model-build qualification and off/on comparison. Commit useful results and verify original inputs before retiring its build/ELF at that transition;do not retain raw successful traces/JIT/host/build logs after reduction. Legacy deployment/provenance holds stay separate.',
        boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        driver=ctx.guards.driver_guard())
    ctx.current(row, committed=False); ctx.save(row)
    print('Prepared bulk-mirror component/off-on plan;commit before configure.', flush=True)


def screen(label):
    ctx = context(); row = json.loads((ROOT / 'record.json').read_text()); ctx.current(row)
    require(row['cases'] == [list(v) for v in CASES], 'committed off/on case order differs')
    index = len(row['outcomes'])
    require(index < len(CASES) and label == CASES[index][0] and
        row['stage'] == ('built' if index == 0 else 'checked-' + str(index)),
        'require next fresh case;never retry/relabel an original failure')
    _, mode, flag = CASES[index]
    root = ROOT / label; root.mkdir(mode=0o700)
    for name in ('tmp', 'cache'): (root / name).mkdir(mode=0o700)
    env = dict(row['recipe']['environment'], **{FLAG: flag}, STRATA_TRACE='1',
        TMPDIR=str(root / 'tmp'), XDG_CACHE_HOME=str(root / 'cache'),
        LD_PRELOAD=row['native_validation_probe']['library']['path'],
        STRATA_UR_PROBE_EXPECTED_LOADER=row['native_validation_probe']['loader']['path'],
        UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION' if mode == 'small' else 'UR_LAYER_PARAMETER_VALIDATION',
        UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(root / 'validation.log'))
    if mode == 'full': env.pop('UR_LOG_TRACING', None)
    require(all(env.get(k) == v for k, v in dict(UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1',
        NEOReadDebugKeys='1', EnableDirectSubmission='0').items()), 'qualified runtime safety changed')
    require(Path('/proc/sys/kernel/random/boot_id').read_text().strip() == row['boot'] and
        ctx.guards.driver_guard() == row['driver'], 'frozen platform changed')
    cursor = subprocess.check_output(['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'],
        text=True, timeout=15).strip().split('-- cursor: ')[-1]
    require(cursor and '\n' not in cursor, 'kernel journal cursor required')
    scope = ctx.owned(root); outcome = dict(passed=False, complete=False, flag=flag, mode=mode)
    try:
        command = [row['binary']['path']] + (['--streamed-full-extent'] if mode == 'full' else [])
        entry, out, err = scope.run(label, command, env, P, **row['budgets'][mode])
        text, errors = out.read_text(), err.read_text()
        marker = ('QSA16 streamed KV producer physical262144 component PASS; attention/model/RESTORE/performance unqualified'
            if mode == 'full' else 'QSA16 local component screen PASS; attention/model/physical262144/RESTORE/performance unqualified')
        require(ctx.ownership.closed(entry) and entry['exit_code'] == 0 and text.splitlines() == [marker],
            'component oracle did not complete normally')
        proof = ctx.validation.preload_evidence(errors, entry['pid'])
        require(not (root / 'validation.log').exists() or (root / 'validation.log').stat().st_size == 0,
            'runtime parameter validation warning')
        receipts = re.findall(r'^strata QSA mirror bulk: tokens=(\d+) pos0=(\d+) segments=(\d+) copies=(\d+) requested_D2H_bytes=(\d+) queue=same accepted=1 host_quantization=0$', errors, re.M)
        require(bool(receipts) == (flag == '1'), 'actual candidate dispatch differs from immutable flag')
        cells, copies, payload = 0, 0, 0
        for receipt in receipts:
            tokens, pos, segments, count, size = map(int, receipt)
            require(0 < tokens <= 8192 and pos + tokens <= 262144 and 0 < segments <= 5
                and count == 4 * segments and size == tokens * 1056, 'accepted mirror receipt differs')
            cells += tokens; copies += count; payload += size
        if mode == 'full' and flag == '1':
            require(cells == 262145 and len(receipts) == 33 and copies == 136 and payload == 262145 * 1056,
                'full fill plus final-cell overwrite not observed')
        ctx.current(row)
        outcome.update(passed=True, complete=True, host_probe_accepted=True, native_exit_code=0,
            validation=proof, physical_cells=262144 if mode == 'full' else 24,
            actual_bulk_calls=len(receipts), accepted_copy_commands=copies, requested_D2H_bytes=payload,
            scope='Independent component numerical/owner/tail/full-fill oracles only;no attention/model/RESTORE/speed qualification')
    except BaseException as error:
        outcome.update(error=repr(error), host_probe_accepted=False)
    finally:
        ctx.validation.finish(outcome, scope, cursor, row['boot'])
        if ctx.guards.driver_guard() != row['driver']: outcome['passed'] = False
        outcome['passed'] = bool(outcome.get('passed') and outcome['complete'])
        row['outcomes'][label] = outcome; row['commands'] += scope.commands
        row.update(stage=('closed' if index == len(CASES) - 1 else 'checked-' + str(index + 1))
            if outcome['passed'] else 'failed', passed=outcome['passed'],
            complete=index == len(CASES) - 1 and outcome['passed'], active=bool(scope.active),
            ownership_closed=outcome['ownership_closed'])
        ctx.save(row)
    print(json.dumps({k: outcome.get(k) for k in ('mode', 'flag', 'passed', 'complete', 'fresh_faults', 'error')}), flush=True)
    require(outcome['passed'], outcome.get('error', 'component case failed'))


if __name__ == '__main__':
    require(len(sys.argv) == 2, 'use prepare|configure|compile|small-off|small-on|full-off|full-on')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); original.guards.no_model_owners([])
        action = sys.argv[1]
        if action == 'prepare': prepare()
        elif action in ('configure', 'compile'): context().build(action)
        else: screen(action)
