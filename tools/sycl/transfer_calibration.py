#!/usr/bin/env python3
"""Main-owned, finite transfer/API calibration; repository LICENSE applies.

Independently authored. No inference changes, third-party application source,
model execution, safety-setting changes or cross-domain timestamp subtraction.
"""
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import re
import select
import shlex
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import build_fused_profile as n

r, P, B = n.r, n.P, n.B
ROOT = B / 'transfer-calibration-v1'
RESULT = n.RESULT
KEY = 'transfer_calibration'
require, digest, write = n.require, n.digest, n.write
PREFIXES = ['tools/sycl', 'bench/sycl/transfer_api.cpp', 'include/strata/sycl', 'src/sycl/runtime.cpp']


def scope(container):
    return container['event_reconciliation']['fused_command_profile']


def retained():
    """Immutable proof, not a relabelled old broad-snapshot qualification."""
    evidence = []
    for name, count in [('record.json', 1385), ('engine-build.json', 1403)]:
        row = json.loads((n.ROOT / name).read_text())
        require(row['passed'] and row['ownership_closed'] and not row['active'], 'retained build not closed')
        for group in [row['project'], *row['dependencies']]:
            recovered = {k: hashlib.sha256(v).hexdigest() for k, v in
                         n.build.blobs(Path(group['root']), group['revision'], group['source_sha256'])}
            require(recovered == group['source_sha256'], 'retained source unrecoverable')
        manifest = row['compiled_inputs']['manifest']
        require(digest(manifest['path']) == manifest['sha256'], 'retained manifest differs')
        values = json.loads(Path(manifest['path']).read_text())
        require(len(values) == count and all(digest(p) == h for p, h in values.items()),
                'retained actual compiled input differs')
        assets = row['fixtures'].values() if name == 'record.json' else [row['binary']]
        for item in assets:
            require(digest(item['path']) == item['sha256'], 'retained native artifact differs')
        evidence.append(dict(record=name, sha256=digest(n.ROOT / name), source_commit=row['source_commit'],
            compiled_inputs=count, manifest=manifest, assets=3 if name == 'record.json' else 1))
    return evidence


def prepare():
    container = r.validation.committed(RESULT)
    cost = scope(container)['observer_cost']
    require(cost['stage'] == 'closed' and cost['passed'] and cost['ownership_closed'] and
            cost['summary']['observer_cost_qualified'], 'closed cost-qualified attribution required')
    require(not ROOT.exists() and KEY not in scope(container), 'fresh calibration only; no retry')
    rt = r.validation.committed(r.RESULT)
    r.current(rt)
    old = json.loads((n.ROOT / 'record.json').read_text())
    source = n.build.pin(P, PREFIXES)
    compiler = old['tools']['sycl']
    require(digest(compiler['path']) == compiler['sha256'], 'compiler changed')
    ROOT.mkdir(mode=0o700)
    for name in ('tmp', 'cache'):
        (ROOT / name).mkdir()
    env = dict(old['environment'], TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'))
    commands = []
    for name, filename in [('fixture', 'bench/sycl/transfer_api.cpp'), ('runtime', 'src/sycl/runtime.cpp')]:
        commands.append([compiler['path'], '-std=c++20', '-O3', '-fsycl', '-fsycl-device-code-split=per_kernel',
            '-I' + str(P / 'include'), '-MD', '-MF', str(ROOT / (name + '.d')),
            '-c', str(P / filename), '-o', str(ROOT / (name + '.o'))])
    commands.append([compiler['path'], '-fsycl', str(ROOT / 'fixture.o'), str(ROOT / 'runtime.o'),
                     '-o', str(ROOT / 'sycl_transfer_api')])
    row = dict(stage='prepared-build', passed=False, complete=False, active=False, ownership_closed=True,
        native_engine_changed=False, GPU_executed=False, production_selected=False,
        source_commit=source['revision'], source=source, sdk=n.sdk_inputs(), compiler=compiler,
        runtime_sha256=digest(r.RESULT), runtime_files=rt['runtime_files'], preload=rt['preload'],
        retained_proof=retained(), environment=env, build_commands=commands,
        boot=rt['boot'], driver=rt['driver'],
        build_limits=dict(wall=600, cpu=1800, rss_cap=8 << 30, text_cap=8 << 20,
                          file_cap=128 << 20, total_cap=512 << 20),
        device_limits=dict(wall=180, cpu=300, rss_cap=3 << 30, text_cap=8 << 20,
                           file_cap=16 << 20, total_cap=128 << 20),
        plan='One full-traced smoke,then three separate parameter-validated processes. '
            'Each two1.9712/2.6624MB payload sizes,four rotated memory arms,one first-use and five subsequent batches. '
            'Fixed128 H2D or1024 D2D copies/batch,64source segments. '
            'Status ordinary/hostUSM with/without independent finite GPU work,rotated four arms,same repetitions. '
            'No adaptive retry or omitted slow sample.',
        criteria='Every process exits normally,all bytewise inputs/outputs/guards pass;preload/provider/runtime '
            'and fault/boot gates pass. Copy wall and device-span floors10ms;queued status work floor1ms. '
            'Below floor is ineligible,not silently repeated. No target speed or adoption gate. '
            'Diagnose enqueue/explicit-wait redistribution using total wall/CPU and GPU service separately. '
            'Requested payload rates only,not physical bus/GDDR traffic or historical cause.',
        safety_audit='Fixed positive extents,maximum allocation170.4MB below device admission;max~700MB live;'
            'single profiled in-order production Runtime queue;host mappings read-only;USM/runtime owners retained;'
            'four staging slots reused only after their own events complete;all pending work drains before '
            'free/unmap including exceptions;no barrier/collective/graph/CPU inference/environment mutation. '
            'Status producer bounded262144items*256 unsigned iterations;each unique output in bounds. '
            'Bytewise validation outside timing;first-use separate;only device-local timestamps subtracted.',
        retention='Component fixture only for immediate transfer/status decision. Original6e40+12fixtures and '
            'diagnostic engine+3fixtures remain unchanged. Retire objects,deps,routine logs/JIT after extraction;'
            'retire this fixture on calibration/consumer closure,not a third model-engine generation.')
    write(ROOT / 'record.json', row); scope(container)[KEY] = compact(row); write(RESULT, container)
    print('Prepared build plan;commit exact generated inputs before compile.', flush=True)


def compact(row):
    out = {k: v for k, v in row.items() if k not in ('source', 'device_source', 'runtime_files', 'commands')}
    out['source'] = {k: row['source'][k] for k in ('revision', 'count', 'aggregate_sha256')}
    if 'device_source' in row:
        out['device_source'] = {k: row['device_source'][k] for k in ('revision', 'count', 'aggregate_sha256')}
    return out


def admit(row):
    require(n.build.pin(P, PREFIXES)['source_sha256'] == row.get('device_source', row['source'])['source_sha256'], 'frozen source differs')
    require(n.sdk_inputs() == row['sdk'] and digest(row['compiler']['path']) == row['compiler']['sha256'], 'SDK differs')
    require(digest(r.RESULT) == row['runtime_sha256'], 'runtime qualification differs')
    rt = r.validation.committed(r.RESULT); r.current(rt)
    require(retained() == row['retained_proof'], 'retained native proof differs')
    require(r.guards.driver_guard() == row['driver'] and
        Path('/proc/sys/kernel/random/boot_id').read_text().strip() == row['boot'], 'boot/driver differs')


def prepare_device():
    container = r.validation.committed(RESULT)
    row = json.loads((ROOT / 'record.json').read_text())
    require(scope(container)[KEY] == compact(row) and row['stage'] == 'built-not-executed' and
        row['passed'] and row['ownership_closed'] and not row['active'], 'closed committed build required')
    original = row['source']
    recovered = {k: hashlib.sha256(v).hexdigest() for k, v in
        n.build.blobs(P, original['revision'], original['source_sha256'])}
    require(recovered == original['source_sha256'], 'original fixture source unrecoverable')
    fresh = n.build.pin(P, PREFIXES)
    changed = {k for k in set(fresh['source_sha256']) | set(recovered)
               if fresh['source_sha256'].get(k) != recovered.get(k)}
    require(changed == {'tools/sycl/transfer_calibration.py'}, 'unexpected device-admission source change')
    manifest = row['compiled_inputs']
    require(digest(manifest['path']) == manifest['sha256'], 'original fixture manifest differs')
    compiled = json.loads(Path(manifest['path']).read_text())
    require(not {str(P / k) for k in changed} & set(compiled) and
        all(digest(p) == h for p, h in compiled.items()) and
        digest(row['binary']['path']) == row['binary']['sha256'], 'actual native fixture changed')
    row['device_source'] = fresh
    row['controller_only_admission'] = dict(original_native_source_commit=row['source_commit'],
        original_build_environment_unchanged=True, new_controller_commit=fresh['revision'],
        changed={k: dict(before=recovered[k], after=fresh['source_sha256'][k]) for k in changed},
        compiled_inputs_verified=len(compiled), binary_sha256=row['binary']['sha256'],
        limits='No recompilation or GPU qualification inherited for new native code. Original source,build '
            'recipe/environment and successful build stay intact. Only uncompiled owned-controller changes;'
            'new pre-exec child environment explicitly sets qualified ZES_ENABLE_SYSMAN=0 and warning logs.')
    settings = r.validation.committed(P / 'tools/sycl/measurement_environment.json')['environment']
    row['device_environment'] = dict(row['environment'], **{k: v for k, v in settings.items()
        if k.startswith('UR_LOG_') or k == 'ZES_ENABLE_SYSMAN'})
    row.update(stage='prepared-device', complete=False, passed=False)
    admit(row)
    write(ROOT / 'record.json', row); scope(container)[KEY] = compact(row); write(RESULT, container)
    print('Prepared device plan;commit before smoke/calibration.', flush=True)


def dependencies(row):
    expected = {str((P / k).resolve()): v for k, v in row['source']['source_sha256'].items()}
    expected.update({str(Path(v['path']).resolve()): n.SYSTEM_HEADER_UPDATES.get(v['path'], (None, v['sha256']))[1]
                     for v in json.loads(n.lineage.SDK.read_text())})
    found = set()
    for name in ('fixture.d', 'runtime.d'):
        text = (ROOT / name).read_text().replace('\\\n', ' ')
        found.update(str(Path(v).resolve()) for v in shlex.split(text.split(':', 1)[1]))
    require({str(P / 'bench/sycl/transfer_api.cpp'), str(P / 'src/sycl/runtime.cpp')} <= found,
            'native closure incomplete')
    for p in found:
        require(p in expected and digest(p) == expected[p], 'unplanned/changed actual compiler input: ' + p)
    values = {p: digest(p) for p in sorted(found)}
    write(ROOT / 'compiler-inputs.json', values)
    return dict(files=len(values), path=str(ROOT / 'compiler-inputs.json'), sha256=digest(ROOT / 'compiler-inputs.json'))


def execute(mode):
    container = r.validation.committed(RESULT); row = json.loads((ROOT / 'record.json').read_text())
    require(scope(container)[KEY] == compact(row), 'commit exact calibration plan/result')
    require(row['stage'] == ('prepared-build' if mode == 'build' else 'prepared-device'), 'fresh expected stage only')
    initial = copy.deepcopy(row); admit(initial)
    owner = None
    cursor = None
    def persist():
        row['active'] = bool(owner and owner.active is not None)
        write(ROOT / 'record.json', dict(row, commands=owner.commands if owner else []))
    def monitor(io, entry):
        seen = set()
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll()
            if mode != 'build':
                try:
                    maps = Path('/proc', str(io.proc.pid), 'maps').read_text()
                except (FileNotFoundError, ProcessLookupError):
                    maps = ''
                for p in r.executable_mappings(maps):
                    require(p in row['runtime_files'] or p == row['binary']['path'], 'unplanned executable map: ' + p)
                    if p in row['runtime_files']: seen.add(p)
            if select.select([io.proc.stdout], [], [], .01)[0]:
                io.read(); io.pending.clear()
        io.exit(1); entry['runtime_seen'] = sorted(seen)
    try:
        owner = r.supervision.Owner(ROOT / ('owned-' + mode), persist)
        if mode == 'build':
            for i, command in enumerate(row['build_commands']):
                entry, _, err = owner.run('compile' + str(i), command, row['environment'], P,
                    interaction=monitor, **row['build_limits'])
                require(r.supervision.closed(entry) and entry['exit_code'] == 0,
                        'build failed: ' + err.read_text(errors='replace')[-3000:])
            row['compiled_inputs'] = dependencies(row)
            p = ROOT / 'sycl_transfer_api'
            row['binary'] = dict(path=str(p), sha256=digest(p), bytes=p.stat().st_size)
            row.update(stage='built-not-executed', passed=True, complete=True)
        else:
            text = subprocess.check_output(['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15)
            cursor = next(v.split(': ', 1)[1] for v in text.splitlines() if v.startswith('-- cursor: '))
            require(digest(row['binary']['path']) == row['binary']['sha256'], 'fixture bytes changed')
            values = json.loads(Path(row['compiled_inputs']['path']).read_text())
            require(digest(row['compiled_inputs']['path']) == row['compiled_inputs']['sha256'] and
                    all(digest(p) == h for p, h in values.items()), 'fixture closure differs')
            row['samples'] = []
            for index in range(-1, 3):
                admit(initial)
                label = 'smoke' if index < 0 else 'process' + str(index)
                trace, warnings = ROOT / (label + '.trace'), ROOT / (label + '.validation')
                env = dict(row['device_environment'], LD_PRELOAD=row['preload']['library']['path'],
                    STRATA_UR_PROBE_EXPECTED_LOADER=row['preload']['loader'],
                    UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION',
                    UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(warnings))
                if index < 0:
                    env.update(UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
                        UR_LOG_TRACING='level:info;flush:info;output:file,' + str(trace))
                command = [row['binary']['path'], '--smoke'] if index < 0 else [row['binary']['path'], '--bench', str(index)]
                row['GPU_executed'] = True; persist()
                entry, out, err = owner.run(label, command, env, P, interaction=monitor, **row['device_limits'])
                require(r.supervision.closed(entry) and entry['exit_code'] == 0,
                        'fixture failed: ' + err.read_text(errors='replace')[-2000:])
                proof = r.validation.preload_evidence(err.read_text(), entry['pid'])
                require(proof['loader'] == row['preload']['loader'], 'actual provider mismatch')
                neo = next(p for p in row['runtime_files'] if 'libze_intel_gpu' in p)
                require(neo in entry['runtime_seen'], 'actual NEO not observed')
                require(not warnings.exists() or warnings.stat().st_size == 0, 'parameter validation warning')
                text = out.read_text()
                require('fixture_status=PASS no_model_or_physical_traffic_qualification=1' in text, 'missing success marker')
                if index < 0: require(trace.exists() and 'urEnqueue' in trace.read_text(), 'smoke device trace absent')
                samples = [json.loads(v) for v in text.splitlines() if v.startswith('{')]
                require(len(samples) == (12 if index < 0 else 72), 'sample count differs')
                probe = dict(host_probe_accepted=True)
                r.validation.finish(probe, owner, cursor, row['boot'])
                require(probe['passed'] and probe['driver_after'] == row['driver'], 'fault/closure gate failed')
                row['samples'].extend(samples if index >= 0 else [])
                print(label + ': normal closure,bytewise guards/provider/fault gates PASS,samples=' + str(len(samples)), flush=True)
                persist()
            row.update(stage='closed', host_probe_accepted=True, complete=True)
        admit(initial)
    except BaseException as error:
        row.update(stage='failed', passed=False, error=repr(error))
    finally:
        if mode != 'build' and owner is not None and cursor:
            r.validation.finish(row, owner, cursor, row['boot'])
            row['passed'] = bool(row['passed'] and row.get('host_probe_accepted') and row['stage'] == 'closed')
        row['active'] = bool(owner and owner.active is not None)
        row['ownership_closed'] = bool(owner and not row['active'] and all(r.supervision.closed(v) for v in owner.commands))
        row['passed'] = bool(row['passed'] and row['ownership_closed'])
        persist(); scope(container)[KEY] = compact(row); write(RESULT, container)
    print(json.dumps({k: row.get(k) for k in ('stage', 'passed', 'ownership_closed', 'error')}), flush=True)
    return int(not row['passed'])


if __name__ == '__main__':
    require(sys.argv[1:] in (['prepare'], ['build'], ['prepare-device'], ['device']), 'prepare|build|prepare-device|device')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        r.guards.no_model_owners([])
        prepare() if sys.argv[1] == 'prepare' else prepare_device() if sys.argv[1] == 'prepare-device' else sys.exit(execute(sys.argv[1]))
