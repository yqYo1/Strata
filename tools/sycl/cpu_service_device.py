#!/usr/bin/env python3
"""Qualify the original-plus-observer build's small device fixtures.

Repository LICENSE. Uses the maintained runtime-probe/owner contracts; does
not invoke the model, select inference changes or qualify observer overhead.
"""
import copy
import fcntl
import json
from pathlib import Path
import select
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import cpu_service_native as engine

gen, native, runtime = engine.gen, engine.native, engine.runtime
validation = runtime.validation
P, B, RESULT = engine.P, engine.B, engine.RESULT
ROOT = engine.ROOT / 'device-probe-v1'
RECORD = ROOT / 'record.json'
require, digest, write = engine.require, engine.digest, engine.write


def compact(row):
    return {k: v for k, v in row.items() if k != 'commands'}


def engine_gate():
    row = json.loads(engine.RECORD.read_text())
    engine.current(row)
    require(row['stage'] == 'linked' and row['passed'] and row['complete'] and
        row['ownership_closed'] and not row['active'] and
        row['dependency_closure']['uncovered_files'] == 0 and
        set(row['assets']) == set(engine.TARGETS), 'closed exact three-target build required')
    for asset in row['assets'].values():
        require(digest(asset['path']) == asset['sha256'] and
            Path(asset['path']).stat().st_size == asset['bytes'], 'linked asset changed')
    require(native.fingerprint(row['actual_compiled_inputs']) == row['dependency_closure']['sha256'],
        'linked compiler closure changed')
    for path, sha in row['actual_compiled_inputs'].items():
        require(digest(path) == sha, 'linked compiler input changed: ' + path)
    return row


def prepare():
    container = gen.committed(RESULT)
    require('native_device' not in container and not ROOT.exists(), 'fresh device gate only; no retry')
    built = engine_gate()
    qualified = gen.committed(runtime.RESULT)  # engine.current verified the complete runtime gate.
    revision = gen.git('rev-parse', 'HEAD').decode().strip()
    pins = gen.benchmark_pins(revision)
    require('tools/sycl/cpu_service_device.py' in pins, 'commit exact device controller first')
    ROOT.mkdir(mode=0o700)
    for name in ('tmp', 'cache'):
        (ROOT / name).mkdir()
    environment = dict(qualified['environment'], TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'))
    cases = [dict(label=label, binary=built['assets'][target], args=args)
        for label, target, args in (
            ('submission', 'sycl_submission_test', ['--submission']),
            ('graph0', 'sycl_graph_envelope_test', ['--profile=0']),
            ('graph1', 'sycl_graph_envelope_test', ['--profile=1']))]
    row = dict(stage='prepared', source_commit=revision,
        benchmark_files=len(pins), benchmark_fingerprint=native.fingerprint(pins),
        native_build_reference=dict(path=str(engine.RECORD), sha256=digest(engine.RECORD)),
        runtime_reference=built['runtime_reference'], environment=environment, cases=cases,
        boot=qualified['boot'], driver=qualified['driver'],
        budgets=dict(wall=90, cpu=120, rss_cap=4 << 30, text_cap=16 << 20,
            file_cap=32 << 20, total_cap=64 << 20),
        criteria='Three ordered fresh owned processes from the new admitted build: submission/USM/events,graph profile0,graph profile1. Require exit0,numerical/guard/teardown markers,pre-main parameter validation,silent validation log,enqueue trace,only pinned executable mappings and actual updated NEO in each device child. Verify all source/build/runtime assets and boot/driver/no-fresh-fault after each child. Stop first failure;no retry.',
        limits='Small same-build fixture qualification only. No model,observer overhead,clean speed,physical context,numerical task quality,ordinary runtime destruction or inference adoption. Retained UR reference until exit remains required.',
        retention='Keep source/recipe,three compact outcomes and limits for imminent original-plus-observer model gate. Retire successful trace/map/log output after evidence is committed and verified;keep original fallback and this single named comparison generation.',
        outcomes=[], active=False, ownership_closed=True, passed=False, complete=False,
        model_qualified=False, production_selected=False, clean_speed_qualified=False)
    write(RECORD, row)
    container['native_device'] = compact(row)
    write(RESULT, container)
    print('Prepared three small same-build device checks; commit plan before execution.', flush=True)


def current(row):
    require(gen.committed(RESULT)['native_device'] == compact(row), 'commit exact device gate plan')
    built = engine_gate()
    require(digest(engine.RECORD) == row['native_build_reference']['sha256'] and
        built['runtime_reference'] == row['runtime_reference'], 'native/runtime qualification changed')
    pins = gen.benchmark_pins(row['source_commit'])
    require(len(pins) == row['benchmark_files'] and native.fingerprint(pins) == row['benchmark_fingerprint'],
        'device controller/helper changed')
    qualified = gen.committed(runtime.RESULT)
    require(qualified['boot'] == row['boot'] and qualified['driver'] == row['driver'],
        'qualified boot/driver changed')
    return qualified


def run():
    container = gen.committed(RESULT)
    row = json.loads(RECORD.read_text())
    qualified = current(row)
    require(row['stage'] == 'prepared' and not (ROOT / 'owned').exists(), 'single-use device gate')
    initial = copy.deepcopy(row)
    owner = None
    cursor_text = subprocess.check_output(['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'],
        text=True, timeout=15)
    cursor = next(line.split(': ', 1)[1] for line in cursor_text.splitlines() if line.startswith('-- cursor: '))

    def persist():
        write(RECORD, dict(row, commands=owner.commands if owner else [],
            active=bool(owner and owner.active is not None)))

    try:
        owner = native.supervision.Owner(ROOT / 'owned', persist)
        for case in row['cases']:
            current(initial)
            label = case['label']
            trace, warnings = ROOT / (label + '.trace'), ROOT / (label + '.validation')
            env = dict(row['environment'], UR_LOG_TRACING='level:info;flush:info;output:file,' + str(trace),
                UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(warnings))
            seen = set()

            def observe(io, entry):
                deadline = time.monotonic() + 85
                while not io.exit_watch.poll(0) or not io.eof:
                    io.poll()
                    require(time.monotonic() < deadline, 'device fixture observation deadline')
                    try:
                        maps = Path('/proc', str(io.proc.pid), 'maps').read_text()
                    except (FileNotFoundError, ProcessLookupError):
                        maps = ''
                    for path in runtime.executable_mappings(maps):
                        require(path in qualified['runtime_files'] or
                            path == str(Path(case['binary']['path']).resolve()), 'unplanned executable mapping: ' + path)
                        if path in qualified['runtime_files']:
                            seen.add(path)
                    if select.select([io.proc.stdout], [], [], .01)[0]:
                        io.read()
                        io.pending.clear()
                io.exit(1)

            entry, out, err = owner.run(label, [case['binary']['path'], *case['args']], env, P,
                interaction=observe, **row['budgets'])
            require(native.supervision.closed(entry) and entry['exit_code'] == 0, 'device fixture failed: ' + label)
            evidence = validation.preload_evidence(err.read_text(), entry['pid'])
            require(evidence['loader'] == qualified['preload']['loader'], 'actual UR provider differs')
            require(not warnings.exists() or not warnings.stat().st_size, 'parameter validation warning')
            neo = next(path for path in qualified['runtime_files'] if 'libze_intel_gpu.so.' in path)
            require(neo in seen and trace.exists() and 'urEnqueue' in trace.read_text(), 'new NEO/enqueue trace missing')
            marker = 'fixture_status=PASS mode=submission' if label == 'submission' else (
                'numerical=PASS guards=PASS aliases=PASS checked_teardown=PASS support=' +
                ('NOT_ENABLED' if label == 'graph0' else 'SUPPORTED'))
            require(marker in out.read_text(), 'fixture numerical/guard/teardown markers missing')
            probe = dict(host_probe_accepted=True)
            validation.finish(probe, owner, cursor, row['boot'])
            require(probe['passed'] and probe['driver_after'] == row['driver'], 'fresh fault/closure check failed')
            row['outcomes'].append(dict(label=label, exit_code=0, passed=True, ownership_closed=True,
                parameter_validation=True, fresh_faults=[], mapped_runtime_file_count=len(seen),
                updated_NEO_sha256=qualified['runtime_files'][neo], stdout_sha256=digest(out),
                stderr_sha256=digest(err), trace_bytes=trace.stat().st_size))
            persist()
        current(initial)
        row.update(stage='device-checked', host_probe_accepted=True, complete=True)
    except BaseException as error:
        row.update(stage='failed', error=repr(error))
    finally:
        if owner is not None:
            validation.finish(row, owner, cursor, row['boot'])
        row['active'] = False
        row['passed'] = bool(row.get('passed') and row['complete'] and len(row['outcomes']) == 3)
        persist()
        container['native_device'] = compact(row)
        write(RESULT, container)
    print(json.dumps({key: row.get(key) for key in ('stage', 'passed', 'ownership_closed', 'error')}), flush=True)
    return int(not row['passed'])


if __name__ == '__main__':
    require(sys.argv[1:] in (['prepare'], ['run']), 'prepare|run')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        native.guards.no_model_owners([])
        if sys.argv[1] == 'prepare':
            prepare()
        else:
            sys.exit(run())
