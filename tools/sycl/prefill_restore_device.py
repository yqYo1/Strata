#!/usr/bin/env python3
"""Qualify restored same-build owners before model or phase comparisons.

Repository LICENSE. Independently authored using this fork's owned runtime
qualification APIs. No model, performance, physical-context or adoption gate.
"""
import copy
import fcntl
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
ROOT = B / 'selected-prefill-restoration-device-v1'
RESULT = restored.RESULT.with_name('prefill-restore-device-current.json')
write, digest = restored.write, restored.build.digest


def current(row):
    guards.require(restored.gen.committed(RESULT) == runtime.compact(row), 'commit exact device plan')
    build = json.loads((restored.ROOT / 'record.json').read_text())
    restored.current(build)
    guards.require(build['stage'] == 'build-host-closed' and build['passed'] and build['complete']
        and build['ownership_closed'] and not build['active'] and len(build['host']) == 8
        and build['source_commit'] == row['build_source_commit']
        and build['assets'] == row['assets']
        and digest(restored.ROOT / 'record.json') == row['build_record_sha256'],
        'closed same-build source/host qualification differs')
    for asset in row['assets'].values():
        guards.require(Path(asset['path']).stat().st_size == asset['bytes']
            and digest(asset['path']) == asset['sha256'], 'same-build artifact changed')
    qualified = restored.gen.committed(runtime.RESULT)
    guards.require(qualified['passed'] and qualified['complete'] and qualified['ownership_closed']
        and not qualified['active'] and qualified['fresh_faults'] == []
        and qualified['preload'] == row['preload'] and qualified['runtime_files'] == row['runtime_files']
        and qualified['boot'] == row['boot'] and qualified['driver'] == row['driver'],
        'qualified safety environment/runtime differs')


def prepare():
    guards.require(not ROOT.exists() and not RESULT.exists(), 'fresh single-use device gate')
    build = json.loads((restored.ROOT / 'record.json').read_text())
    restored.current(build)
    guards.require(build['passed'] and build['stage'] == 'build-host-closed', 'build/host gates first')
    qualified = restored.gen.committed(runtime.RESULT)
    guards.require(qualified['passed'] and qualified['complete'], 'qualified runtime required')
    revision = restored.gen.git('rev-parse', 'HEAD').decode().strip()
    pins = restored.gen.benchmark_pins(build['source_commit'])
    guards.require('tools/sycl/prefill_restore_device.py' in pins, 'commit controller before native build')
    cases = [dict(label=label, target=target, args=args, marker=marker) for label, target, args, marker in (
        ('submission', 'sycl_submission_test', ['--submission'], 'fixture_status=PASS mode=submission'),
        ('graph0', 'sycl_graph_envelope_test', ['--profile=0'],
            'numerical=PASS guards=PASS aliases=PASS checked_teardown=PASS support=NOT_ENABLED'),
        ('graph1', 'sycl_graph_envelope_test', ['--profile=1'],
            'numerical=PASS guards=PASS aliases=PASS checked_teardown=PASS support=SUPPORTED'),
        ('scratch-smoke', 'sycl_prefill_scratch_test', ['--case', 'smoke'],
            'prefill_scratch_device_fixture mode=smoke cases=1 status=PASS'),
        ('scratch-full', 'sycl_prefill_scratch_test', ['--case', 'full'],
            'prefill_scratch_device_fixture mode=full cases=15 status=PASS'),
        ('scratch-loan', 'sycl_prefill_scratch_test', ['--case', 'loan'],
            'prefill_cache_loan_fixture cases=20 cycles=3 model_PP=UNRUN status=PASS'),
        ('expert-private', 'sycl_expert_bank_test', ['--packed-smoke'],
            'expert_bank_fixture PASS model=UNRUN physical262144_RESTORE=UNQUALIFIED'),
        ('restore-smoke', 'sycl_conversation_snapshot_test', ['--restore-smoke'], '(restore smoke)'))]
    ROOT.mkdir(mode=0o700)
    for name in ('tmp', 'cache'):
        (ROOT / name).mkdir()
    row = dict(stage='prepared', controller_source_commit=revision, build_source_commit=build['source_commit'],
        build_record_sha256=digest(restored.ROOT / 'record.json'), assets=build['assets'], cases=cases,
        preload=qualified['preload'], runtime_files=qualified['runtime_files'],
        boot=qualified['boot'], driver=qualified['driver'],
        environment=dict(qualified['environment'], TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache')),
        budgets=dict(wall=90, cpu=120, rss_cap=4 << 30, text_cap=16 << 20,
            file_cap=32 << 20, total_cap=64 << 20),
        criteria='Eight ordered fresh owned processes on the exact restored same-build fixtures;stop first failure,no retry. '
            'Require exit0,numerical/bounds/loan/publication/RESTORE markers,pre-main validation,silent parameter '
            'validation and an observed pinned updated NEO. Trace enqueue work where expected;smoke refusal may '
            'perform allocation only. Check actual process closure,boot and fresh driver faults after every case.',
        retention='Keep compact individual fixture outcomes and source/artifact/runtime identities. Delete successful '
            'API traces,maps and repeated logs after committed replay. Keep this generation only for imminent '
            'model/state and independent prefill/decode comparisons;do not select it from fixture success.',
        pending='Model admission/state handoff,matched repeated long prefill and independent decode,task/agent/RESTORE '
            'and physical262144 qualification. Retained UR reference means ordinary runtime teardown is unqualified.',
        outcomes=[], active=False, ownership_closed=True, complete=False, passed=False,
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
            trace, warnings = ROOT / (case['label'] + '.trace'), ROOT / (case['label'] + '.validation')
            env = dict(row['environment'], UR_LOG_TRACING='level:info;flush:info;output:file,' + str(trace),
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
            row['outcomes'].append(dict(label=case['label'], exit_code=0, passed=True, ownership_closed=True,
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
        row['passed'] = bool(row.get('passed') and row['complete'] and len(row['outcomes']) == 8)
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
