#!/usr/bin/env python3
"""Bounded requalification after the20261024 NEO update; no model or rebuild.

Repository LICENSE; independently authored from maintained owner/fixture APIs.
Original runtime qualifications and the interrupted comparison remain unchanged.
"""
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import select
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import build_engine as build
import check_full_context_source as selected
import measurement_guards as guards
import measurement_owner as supervision
import ur_validation as validation

P, B = build.P, build.B
ROOT = B / 'neo-39758-runtime-probe-v2'
RESULT = selected.RESULT.with_name('runtime-requalification-current.json')
SETTINGS = P / 'tools/sycl/measurement_environment.json'
INPUTS = ('tools/sycl/runtime_probe.py', 'tools/sycl/check_full_context_source.py',
    'tools/sycl/build_engine.py', 'tools/sycl/measurement_guards.py',
    'tools/sycl/measurement_owner.py', 'tools/sycl/ur_validation.py',
    'tools/sycl/measurement_environment.json')
REPLACEMENTS = {
    '/usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1.17.39395':
        '/usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1.17.39758',
    '/usr/lib/x86_64-linux-gnu/libigdml.so.1.0.231':
        '/usr/lib/x86_64-linux-gnu/libigdml.so.1.0.240'}
EXTRA = '/usr/lib/x86_64-linux-gnu/libmetee.so.6.2.5.0'
DECLARED_CHANGES = {
    '/usr/lib/x86_64-linux-gnu/libz.so.1.3':
        'f61677c7bdb7c2aa2a243ea781bfd16afcb361cede7b4ef76127a9d8eb9a5242',
    '/usr/lib/x86_64-linux-gnu/libigdml.so.1.0.231':
        '1806ee390b7693dd61dd5385a25744dd94479295d3a7d15730e578e91b18e782',
    '/usr/lib/x86_64-linux-gnu/libigdgmm.so.12.10.0':
        '427847d14c2c36889efa8b0be6a8af6afac32cc3c744dcfc8e010d4ca210c957',
    '/usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1.17.39395':
        '52e29a602f92d3542eac91e20717743e63ca7d4b257d0c79a3c4d6c64c94ea49'}
write, require, digest = selected.write, guards.require, build.digest


def pins(revision):
    result = {}
    for name, raw in build.blobs(P, revision, INPUTS):
        require(raw == (P / name).read_bytes(), 'commit exact runtime probe input: ' + name)
        result[name] = hashlib.sha256(raw).hexdigest()
    return result


def components():
    # Reuse unchanged host-only native artifacts, not the old GPU-runtime gate.
    result = validation.committed(validation.INDEX,
        ('UR_silent_validation_native_child_scope', 'preload_component'))
    require(result['passed'] and result['ownership_closed'] and not result['fresh_faults'] and
        result['provider_mismatch_stopped_before_body'], 'closed original preload component required')
    ref = result['original_plan']
    raw = validation.git('show', ref['commit'] + ':' + ref['path'])
    require(hashlib.sha256(raw).hexdigest() == ref['sha256'], 'original preload recipe differs')
    original = json.loads(raw)['UR_silent_validation_native_child_scope']['canary_plan']
    require(original['source_commit'] == result['source_commit'], 'preload source identity differs')
    for name, sha in original['inputs'].items():
        path = Path(name)
        if path.is_relative_to(P):
            require(hashlib.sha256(validation.git('show', result['source_commit'] + ':' +
                str(path.relative_to(P)))).hexdigest() == sha, 'original preload source cannot recover')
            if path.suffix != '.c':
                continue
        require(digest(path) == sha, 'preload compiled input changed: ' + name)
    for asset in (result['library'], result['canary']):
        require(digest(asset['path']) == asset['sha256'], 'preload/canary artifact changed')
    loader = result['preload_evidence']['loader']
    require(digest(loader) == original['runtime_pins'][loader], 'UR provider changed')
    return dict(library=result['library'], canary=result['canary'], loader=loader,
        original_plan=ref, source_commit=result['source_commit'],
        retained_reference_until_exit=True, ordinary_runtime_teardown_qualified=False)


def runtime_files(preload):
    settings = validation.committed(SETTINGS)
    client = validation.committed(RESULT.with_name('client-resources-current.json'))
    out = {}
    for name, original in settings['runtime_elfs'].items():
        target = REPLACEMENTS.get(name, name)
        sha = digest(target)
        expected = client.get('runtime_asset_delta', {}).get(name, {}).get('candidate_sha256', original['sha256'])
        require(sha == DECLARED_CHANGES.get(name, expected), 'undeclared additional runtime change: ' + name)
        out[target] = sha
    for name in (EXTRA, preload['library']['path']):
        out[name] = digest(name)
    return out


def compact(row):
    return {k: v for k, v in row.items() if k not in ('environment', 'commands')}


def current(row):
    require(RESULT.read_bytes() == validation.git('show', 'HEAD:' + str(RESULT.relative_to(P))) and
        json.loads(RESULT.read_text()) == compact(row), 'commit exact runtime plan')
    require(pins(row['source_commit']) == row['input_sha256'], 'runtime probe source changed')
    full, proof = selected.baseline()
    require(proof == row['original_build'] and components() == row['preload'] and
        runtime_files(row['preload']) == row['runtime_files'], 'native/runtime inputs changed')
    require(guards.driver_guard() == row['driver'] and
        Path('/proc/sys/kernel/random/boot_id').read_text().strip() == row['boot'], 'boot/driver changed')
    return full


def prepare():
    require(not ROOT.exists(), 'fresh bounded runtime gate only; no retry')
    failure = validation.committed(RESULT)
    require(failure['stage'] == 'preparation-failed' and not failure['root_created'] and
        not failure['native_launched'] and failure['ownership_closed'] and not failure['active'],
        'preserve the closed metadata-only preparation refusal')
    failure_ref = dict(commit=validation.git('log', '-1', '--format=%H', '--',
        str(RESULT.relative_to(P))).decode().strip(), path=str(RESULT.relative_to(P)), sha256=digest(RESULT))
    prior = validation.committed(RESULT.with_name('mtp-width-clean-current.json'))
    require(prior['stage'] == 'admission-failed' and len(prior['processes']) == 4 and
        prior['postfailure_full_model_hashes'] == 15 and not prior['active'] and prior['ownership_closed'],
        'close interrupted comparison before a new runtime gate')
    revision = validation.git('rev-parse', 'HEAD').decode().strip()
    inputs = pins(revision)
    full, proof = selected.baseline()
    preload = components()
    files = runtime_files(preload)
    ROOT.mkdir(mode=0o700)
    for name in ('tmp', 'cache'):
        (ROOT / name).mkdir()
    environment = dict(validation.committed(SETTINGS)['environment'],
        LD_PRELOAD=preload['library']['path'], STRATA_UR_PROBE_EXPECTED_LOADER=preload['loader'],
        TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'))
    cases = [dict(label='canary', binary=preload['canary'], args=[])]
    for label, fixture, args in (
            ('submission', 'sycl_submission_test', ['--submission']),
            ('graph0', 'sycl_graph_envelope_test', ['--profile=0']),
            ('graph1', 'sycl_graph_envelope_test', ['--profile=1'])):
        cases.append(dict(label=label, binary=full['fixtures'][fixture], args=args))
    row = dict(stage='prepared', source_commit=revision, input_sha256=inputs,
        prior_preparation_failure=failure_ref, declared_runtime_changes=DECLARED_CHANGES,
        original_build=proof, preload=preload, runtime_files=files, environment=environment, cases=cases,
        boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), driver=guards.driver_guard(),
        criteria='Four ordered fresh owned processes: host-only preload canary,submission/USM/events,ordinary graph profile0,profile1. All exit0,expected numerical/guard/profiling markers,pre-main parameter validation,silent validation log,observed ELF subset pinned and actual new NEO observed in device children. Trace must record enqueue work;fresh fault/boot/source/runtime/owner checks after every child. Stop first failure,no retry. Not model,clean speed,physical context or ordinary runtime destruction qualification.',
        budgets=dict(wall=90, cpu=120, rss_cap=4 << 30, text_cap=16 << 20, file_cap=32 << 20, total_cap=64 << 20),
        retention='Keep exact source/recipe,runtime hashes,individual fixture outcomes and limits. Retire successful traces/maps/logs after committed replay;retain provider manifest for imminent model requalification/fresh comparison. No new native engine generation.',
        outcomes=[], active=False, ownership_closed=True, passed=False, complete=False,
        model_qualified=False, production_selected=False, clean_speed_qualified=False)
    write(ROOT / 'record.json', row)
    write(RESULT, compact(row))
    print('Prepared updated-runtime host/device gate; commit before execution.', flush=True)


def run(row):
    current(row)
    require(row['stage'] == 'prepared' and not (ROOT / 'owned').exists(), 'single-use runtime gate')
    initial = copy.deepcopy(row)
    owner = None
    cursor_text = subprocess.check_output(['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15)
    cursor = next(line.split(': ', 1)[1] for line in cursor_text.splitlines() if line.startswith('-- cursor: '))

    def persist():
        write(ROOT / 'record.json', dict(row, commands=owner.commands if owner else [],
            active=bool(owner and owner.active is not None)))

    try:
        owner = supervision.Owner(ROOT / 'owned', persist)
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
                    require(time.monotonic() < deadline, 'fixture observation deadline')
                    try:
                        maps = Path('/proc', str(io.proc.pid), 'maps').read_text()
                    except (FileNotFoundError, ProcessLookupError):
                        maps = ''
                    for line in maps.splitlines():
                        fields = line.split(maxsplit=5)
                        if len(fields) == 6 and fields[5].startswith('/') and '.so' in Path(fields[5]).name:
                            require(not fields[5].endswith(' (deleted)'), 'runtime asset removed while mapped')
                            path = str(Path(fields[5]).resolve())
                            require(path in row['runtime_files'], 'unplanned mapped runtime: ' + path)
                            seen.add(path)
                    if select.select([io.proc.stdout], [], [], .01)[0]:
                        io.read()
                        io.pending.clear()
                io.exit(1)

            entry, out, err = owner.run(label, [case['binary']['path'], *case['args']], env, P,
                interaction=observe, **row['budgets'])
            require(supervision.closed(entry) and entry['exit_code'] == 0, 'runtime fixture failed: ' + label)
            evidence = validation.preload_evidence(err.read_text(), entry['pid'])
            require(evidence['loader'] == row['preload']['loader'], 'actual UR provider differs')
            require(not warnings.exists() or not warnings.stat().st_size, 'parameter validation warning')
            text = out.read_text()
            if label == 'canary':
                require(json.loads(text) == dict(pid=entry['pid'], initialized=0, checked=46,
                    expected=46, version=(1 << 32) - 1, closed=0), 'host canary result differs')
            else:
                require(REPLACEMENTS[next(k for k in REPLACEMENTS if 'libze_intel_gpu' in k)] in seen,
                    'actual updated NEO provider not observed')
                require(trace.exists() and 'urEnqueue' in trace.read_text(), 'device enqueue trace missing')
                marker = 'fixture_status=PASS mode=submission' if label == 'submission' else (
                    'numerical=PASS guards=PASS aliases=PASS checked_teardown=PASS support=' +
                    ('NOT_ENABLED' if label == 'graph0' else 'SUPPORTED'))
                require(marker in text, 'fixture numerical/profiling/teardown markers missing')
            probe = dict(host_probe_accepted=True)
            validation.finish(probe, owner, cursor, row['boot'])
            require(probe['passed'] and probe['driver_after'] == row['driver'], 'fresh fault/closure check failed')
            row['outcomes'].append(dict(label=label, exit_code=0, passed=True, ownership_closed=True,
                parameter_validation=True, fresh_faults=[], mapped_runtime_files=sorted(seen),
                stdout_sha256=digest(out), stderr_sha256=digest(err),
                trace_bytes=trace.stat().st_size if trace.exists() else 0))
            persist()
        current(initial)
        row.update(stage='device-checked', host_probe_accepted=True, complete=True)
    except BaseException as error:
        row.update(stage='failed', error=repr(error))
    finally:
        if owner is not None:
            validation.finish(row, owner, cursor, row['boot'])
        row['active'] = False
        row['passed'] = bool(row.get('passed') and row['complete'] and len(row['outcomes']) == 4)
        persist()
        write(RESULT, compact(row))
    print(json.dumps({k: row.get(k) for k in ('stage', 'passed', 'complete', 'ownership_closed', 'error')}), flush=True)
    return int(not row['passed'])


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'run'), 'prepare|run')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners([])
        if sys.argv[1] == 'prepare':
            prepare()
        else:
            sys.exit(run(json.loads((ROOT / 'record.json').read_text())))
