#!/usr/bin/env python3
"""Qualify restored same-build owners before model or phase comparisons.

Repository LICENSE. Independently authored using this fork's owned runtime
qualification APIs. No model, performance, physical-context or adoption gate.
"""
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import re
import select
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import prefill_restore as restored

runtime = restored.runtime
validation = runtime.validation
guards, supervision = restored.guards, restored.supervision
P, B = restored.P, restored.B
ROOT = B / 'selected-prefill-restoration-device-v3'
RESULT = restored.RESULT.with_name('prefill-restore-device-current.json')
write, digest = restored.write, restored.build.digest


def reused_checks(previous):
    guards.require(previous['stage'] == 'failed' and not previous['passed'] and not previous['active']
        and previous['ownership_closed'] and previous['boot_unchanged'] and previous['fresh_faults'] == []
        and len(previous['outcomes']) == 12
        and previous['failure_diagnosis']['stderr'] ==
            'prefill scratch fixture: one byte short was admitted', 'closed layout-configuration refusal required')
    pins = restored.gen.benchmark_pins(previous['controller_source_commit'], match_worktree=False)
    guards.require(len(pins) == previous['controller_files'] and
        restored.native.fingerprint(pins) == previous['controller_fingerprint'], 'original controller cannot recover')
    checks = [c for c in previous['outcomes'] if c['label'] != 'scratch-smoke']
    guards.require(len(checks) == 11 and sum(c['kind'] == 'host' for c in checks) == 8
        and [c['label'] for c in checks if c['kind'] == 'device'] == ['submission', 'graph0', 'graph1']
        and all(c['passed'] and c['ownership_closed'] and c['exit_code'] == 0 for c in checks)
        and all(c['parameter_validation'] and c['fresh_faults'] == [] for c in checks if c['kind'] == 'device'),
        'individual unchanged host/submission/graph qualifications incomplete')
    return checks


def current(row):
    guards.require(restored.gen.committed(RESULT) == runtime.compact(row), 'commit exact device plan')
    build = json.loads((restored.ROOT / 'record.json').read_text())
    restored.current(build, controller_revision=row['controller_source_commit'])
    linked(build)
    guards.require(build['source_commit'] == row['build_source_commit']
        and build['assets'] == row['assets']
        and digest(restored.ROOT / 'record.json') == row['build_record_sha256'],
        'same-build native/source identity differs')
    pins = restored.gen.benchmark_pins(row['controller_source_commit'])
    guards.require(len(pins) == row['controller_files'] and
        restored.native.fingerprint(pins) == row['controller_fingerprint'], 'new controller snapshot changed')
    for asset in row['assets'].values():
        guards.require(Path(asset['path']).stat().st_size == asset['bytes']
            and digest(asset['path']) == asset['sha256'], 'same-build artifact changed')
    qualified = restored.gen.committed(runtime.RESULT)
    guards.require(qualified['passed'] and qualified['complete'] and qualified['ownership_closed']
        and not qualified['active'] and qualified['fresh_faults'] == []
        and qualified['preload'] == row['preload'] and qualified['runtime_files'] == row['runtime_files']
        and qualified['boot'] == row['boot'] and qualified['driver'] == row['driver'],
        'qualified safety environment/runtime differs')
    ref = row['prior_failed_gate']
    raw = restored.gen.git('show', ref['commit'] + ':' + ref['path'])
    guards.require(hashlib.sha256(raw).hexdigest() == ref['sha256'], 'original failed gate changed')
    previous = json.loads(raw)
    same_environment = lambda env: {k: v for k, v in env.items() if k not in ('TMPDIR', 'XDG_CACHE_HOME')}
    guards.require(row['outcomes'][:11] == reused_checks(previous)
        and previous['assets'] == row['assets'] and previous['build_source_commit'] == row['build_source_commit']
        and all(previous[k] == row[k] for k in ('runtime_files', 'preload', 'boot', 'driver'))
        and same_environment(previous['environment']) == same_environment(row['environment'])
        and same_environment(previous['host_environment']) == same_environment(row['host_environment']),
        'reused fixture bytes,original source or effective settings differ')


def linked(build):
    """Reuse successful compilation, without promoting failed host observation."""
    commands = build['commands']
    guards.require(build['stage'] == 'failed' and not build['passed'] and not build['active']
        and len(commands) == 4 and [c['label'] for c in commands[:3]] ==
            ['configure', 'link', 'compiler-census']
        and all(supervision.closed(c) and c['exit_code'] == 0 for c in commands[:3])
        and build['compiler_closure']['files'] == 1416 and build['compiler_closure']['uncovered'] == 0
        and not build['host'] and len(build['assets']) == 17, 'closed compilation/census required')
    failed = commands[-1]
    guards.require(failed['label'] == 'host-prefill_expert_bank_plan_test'
        and failed['error'] == "RuntimeError('artifact cap')" and failed['direct_child_reaped']
        and failed['session_empty'] and not failed['survivors']
        and build['build_diagnosis']['actual_ownership_closed'], 'reconcile failed host supervision first')


def prepare():
    guards.require(not ROOT.exists(), 'fresh single-use device gate')
    previous = restored.gen.committed(RESULT)
    reused = reused_checks(previous)
    guards.require(previous['artifact_retirement']['complete'], 'retire closed layout-refusal output first')
    prior_failed = dict(commit=restored.gen.git('log', '-1', '--format=%H', '--',
        str(RESULT.relative_to(P))).decode().strip(), path=str(RESULT.relative_to(P)), sha256=digest(RESULT))
    build = json.loads((restored.ROOT / 'record.json').read_text())
    revision = restored.gen.git('rev-parse', 'HEAD').decode().strip()
    restored.current(build, controller_revision=revision)
    linked(build)
    qualified = restored.gen.committed(runtime.RESULT)
    guards.require(qualified['passed'] and qualified['complete'], 'qualified runtime required')
    pins = restored.gen.benchmark_pins(revision)
    guards.require('tools/sycl/prefill_restore_device.py' in pins, 'commit exact host/device controller')
    cases = [dict(label=label, target=target, args=args, marker=marker) for label, target, args, marker in (
        ('scratch-smoke', 'sycl_prefill_scratch_test', ['--case', 'smoke'],
            'prefill_scratch_device_fixture mode=smoke cases=1 status=PASS'),
        ('scratch-full', 'sycl_prefill_scratch_test', ['--case', 'full'],
            'prefill_scratch_device_fixture mode=full cases=15 status=PASS'),
        ('scratch-loan', 'sycl_prefill_scratch_test', ['--case', 'loan'],
            'prefill_cache_loan_fixture cases=20 cycles=3 model_PP=UNRUN status=PASS'),
        ('expert-private', 'sycl_expert_bank_test', ['--packed-smoke'],
            'expert_bank_fixture PASS model=UNRUN physical262144_RESTORE=UNQUALIFIED'),
        ('restore-smoke', 'sycl_conversation_snapshot_test', ['--restore-smoke'], '(restore smoke)'))]
    layout = dict(STRATA_SYCL_PREFILL_SCRATCH_ALIAS='1', STRATA_PREFILL_MMQ='0',
        STRATA_PREFILL_MOE_COMPACT_FP16='1', STRATA_PREFILL_RING='8',
        STRATA_GR_UNFUSED='0', STRATA_PREFILL_BF16X2='0', STRATA_PREFILL_LAYER_MAJOR='1')
    for case in cases:
        case['kind'] = 'device'
        case['environment_delta'] = layout if case['target'] == 'sycl_prefill_scratch_test' else {}
    ROOT.mkdir(mode=0o700)
    for name in ('tmp', 'cache'):
        (ROOT / name).mkdir()
    row = dict(stage='prepared', controller_source_commit=revision, build_source_commit=build['source_commit'],
        prior_failed_gate=prior_failed,
        controller_files=len(pins), controller_fingerprint=restored.native.fingerprint(pins),
        build_record_sha256=digest(restored.ROOT / 'record.json'), assets=build['assets'], cases=cases,
        preload=qualified['preload'], runtime_files=qualified['runtime_files'],
        boot=qualified['boot'], driver=qualified['driver'],
        environment=dict(qualified['environment'], TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache')),
        budgets=dict(wall=90, cpu=120, rss_cap=4 << 30, text_cap=16 << 20,
            file_cap=32 << 20, total_cap=64 << 20),
        host_environment=dict(build['environment'], TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache')),
        criteria='Reuse only8 host and3 submission/graph passes from the closed failed gate after exact source,'
            '17-asset,runtime and effective-environment proof. Scratch smoke with its changed layout is NOT reused. '
            'Five fresh device cases:Alias1/MMQ0 scratch smoke/full/loan,private bank,RESTORE. '
            'Stop first failure,no retry;original failed gates remain failed. All8 host gates precede GPU work. '
            'Require exit0,numerical/bounds/loan/publication/RESTORE markers,pre-main validation,silent parameter '
            'validation and an observed pinned updated NEO. Trace enqueue work where expected;smoke refusal may '
            'perform allocation only. Check actual process closure,boot and fresh driver faults after every case.',
        retention='Keep compact individual fixture outcomes and source/artifact/runtime identities. Delete successful '
            'API traces,maps and repeated logs after committed replay. Keep this generation only for imminent '
            'model/state and independent prefill/decode comparisons;do not select it from fixture success.',
        pending='Model admission/state handoff,matched repeated long prefill and independent decode,task/agent/RESTORE '
            'and physical262144 qualification. Retained UR reference means ordinary runtime teardown is unqualified.',
        outcomes=reused, reused_outcomes=11, active=False, ownership_closed=True, complete=False, passed=False,
        model_qualified=False, clean_speed_qualified=False, production_selected=False)
    write(ROOT / 'record.json', row); write(RESULT, runtime.compact(row))
    print('Prepared restored same-build device gate;commit before run.', flush=True)


def run(row):
    current(row)
    guards.require(row['stage'] == 'prepared' and not (ROOT / 'owned').exists(), 'single-use device gate')
    initial, owner = copy.deepcopy(row), None
    cursor_text = subprocess.check_output(['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'],
        text=True, timeout=15)
    cursor = next(s.split(': ', 1)[1] for s in cursor_text.splitlines() if s.startswith('-- cursor: '))

    def persist():
        write(ROOT / 'record.json', dict(row, commands=owner.commands if owner else [],
            active=bool(owner and owner.active is not None)))

    try:
        owner = supervision.Owner(ROOT / 'owned', persist)
        for case in row['cases']:
            current(initial)
            asset = row['assets'][case['target']]
            guards.require(sum(c['kind'] == 'host' and c['passed'] for c in row['outcomes']) == 8,
                'all host gates before the first device')
            trace, warnings = ROOT / (case['label'] + '.trace'), ROOT / (case['label'] + '.validation')
            env = dict(row['environment'], **case['environment_delta'])
            env.update(UR_LOG_TRACING='level:info;flush:info;output:file,' + str(trace),
                UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(warnings))
            seen = set()

            def observe(io, entry):
                deadline = time.monotonic() + 85
                while not io.exit_watch.poll(0) or not io.eof:
                    io.poll()
                    guards.require(time.monotonic() < deadline, 'device observation deadline')
                    try:
                        maps = Path('/proc', str(io.proc.pid), 'maps').read_text()
                    except (FileNotFoundError, ProcessLookupError):
                        maps = ''
                    for path in runtime.executable_mappings(maps):
                        guards.require(path in row['runtime_files'] or path == str(Path(asset['path']).resolve()),
                            'unplanned executable mapping: ' + path)
                        if path in row['runtime_files']:
                            seen.add(path)
                    if select.select([io.proc.stdout], [], [], .01)[0]:
                        io.read(); io.pending.clear()
                io.exit(1)

            entry, out, err = owner.run(case['label'], [asset['path'], *case['args']], env, P,
                interaction=observe, **row['budgets'])
            guards.require(supervision.closed(entry) and entry['exit_code'] == 0, 'device fixture failed: ' + case['label'])
            evidence = validation.preload_evidence(err.read_text(), entry['pid'])
            guards.require(evidence['loader'] == row['preload']['loader'], 'actual UR provider differs')
            guards.require(not warnings.exists() or warnings.stat().st_size == 0, 'parameter validation warning')
            guards.require(any('libze_intel_gpu.so.1.17.39758' in p for p in seen), 'updated NEO not observed')
            guards.require(trace.exists() and trace.stat().st_size > 0, 'device trace missing')
            if case['label'] != 'scratch-smoke':
                guards.require('urEnqueue' in trace.read_text(), 'device enqueue trace missing')
            text = out.read_text()
            guards.require(case['marker'] in text, 'fixture success marker missing')
            checks = None
            if case['label'] == 'restore-smoke':
                match = re.search(r'conversation_snapshot_test: ([1-9][0-9]*) checks passed \(restore smoke\)', text)
                guards.require(match is not None, 'RESTORE verification count missing')
                checks = int(match[1])
            probe = dict(host_probe_accepted=True)
            validation.finish(probe, owner, cursor, row['boot'])
            guards.require(probe['passed'] and probe['driver_after'] == row['driver'], 'fresh fault/closure check failed')
            row['outcomes'].append(dict(label=case['label'], kind='device', exit_code=0, passed=True, ownership_closed=True,
                parameter_validation=True, fresh_faults=[], restore_checks=checks,
                mapped_runtime_fingerprint=restored.native.fingerprint(sorted(seen)),
                stdout_sha256=digest(out), stderr_sha256=digest(err), trace_bytes=trace.stat().st_size))
            persist()
        current(initial)
        row.update(stage='device-checked', host_probe_accepted=True, complete=True)
    except BaseException as error:
        row.update(stage='failed', error=repr(error))
    finally:
        if owner is not None:
            validation.finish(row, owner, cursor, row['boot'])
        row['active'] = bool(owner and owner.active is not None)
        row['passed'] = bool(row.get('passed') and row['complete'] and len(row['outcomes']) == 16)
        persist(); write(RESULT, runtime.compact(row))
    print(json.dumps({k: row.get(k) for k in ('stage', 'passed', 'complete', 'ownership_closed', 'error')}), flush=True)
    return int(not row['passed'])


if __name__ == '__main__':
    guards.require(sys.argv[1:] in (['prepare'], ['run']), 'prepare|run')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners([])
        if sys.argv[1:] == ['prepare']:
            prepare()
        else:
            sys.exit(run(json.loads((ROOT / 'record.json').read_text())))
