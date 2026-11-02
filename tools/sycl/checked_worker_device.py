#!/usr/bin/env python3
"""Bounded real-copy worker retirement check; Main-authored, repository LICENSE.

Build a component from the new engine's exact runtime/cache sources. No model,
speed, mapping arithmetic, ordinary runtime teardown or physical-context claim.
"""
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import select
import shlex
import sys
import time

sys.dont_write_bytecode = True
import build_cache_map_model as model
import engine_once
import ur_validation

base, component = model.base, model.component
P, B, RESULT = model.P, model.B, model.RESULT
ROOT, KEY = B / 'checked-adaptive-worker-device-v2', 'checked_adaptive_worker_device_v2'
CPP = 'tests/sycl/checked_worker_device.cpp'
ENV_KEY = 'adaptive_cache_model_entry_adaptive-positive-throughput_model_v1'
require, digest, write = base.require, base.digest, base.write
BUILD = dict(wall=600, cpu=1200, rss_cap=8 << 30, text_cap=4 << 20,
             file_cap=256 << 20, total_cap=512 << 20)
DEVICE = dict(wall=180, cpu=180, rss_cap=4 << 30, text_cap=4 << 20,
              file_cap=32 << 20, total_cap=512 << 20)


def compact(row):
    out = {k: v for k, v in row.items() if k not in
           ('sources', 'commands_executed', 'build_environment', 'environment_template')}
    out['sources'] = dict(files=len(row['sources']), aggregate_sha256=hashlib.sha256(
        json.dumps(row['sources'], sort_keys=True).encode()).hexdigest())
    for name in ('build_environment', 'environment_template'):
        out[name + '_sha256'] = hashlib.sha256(json.dumps(row[name], sort_keys=True).encode()).hexdigest()
    return out


def save(row):
    write(ROOT / 'record.json', row)
    data = json.loads(RESULT.read_text()); data[KEY] = compact(row); write(RESULT, data)


def sources(revision):
    pins = base.controls(revision, controller=Path(__file__).resolve())
    raw = base.basis.build.git(P, 'show', revision + ':' + CPP)
    require((P / CPP).read_bytes() == raw, 'commit exact device fixture')
    pins[CPP] = hashlib.sha256(raw).hexdigest()
    return pins


def dependencies(path):
    raw = path.read_text().replace('\\\n', ' ')
    return {str(Path(value).resolve()) for value in shlex.split(raw.split(':', 1)[1])}


def allowed_inputs(native):
    # model.verify() has just checked these exact inventories, source, actual
    # dependencies and linker bytes. Do not repeat all package-manager queries.
    allowed = {}
    for name in ('sdk', 'external_manifest', 'runtime_manifest', 'compiled_inputs'):
        inv = native[name]
        require(digest(inv['path']) == inv['sha256'], 'verified inventory changed: ' + name)
        values = json.loads(Path(inv['path']).read_text())
        require(len(values) == inv['files'], 'verified inventory shape changed: ' + name)
        allowed.update(values)
    allowed.update(native['actual_link_inputs']['frozen_inputs'])
    return allowed


def current(row):
    require(base.committed(RESULT)[KEY] == compact(row) and
            sources(row['source_commit']) == row['sources'], 'commit/freeze exact device fixture inputs')
    native = model.verify()
    require(native['source_commit'] == row['engine_source_commit'] and
            native['binaries']['strata'] == row['engine'] and native['boot'] == row['boot'] and
            native['driver'] == row['driver'], 'new engine/source/environment changed')
    original = json.loads(base.basis.build.git(P, 'show', row['source_commit'] + ':' +
        str(RESULT.relative_to(P))))[ENV_KEY]['environment']
    require(row['environment_template'] == original and row['build_environment'] ==
            dict(native['environment'], TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache')),
            'original build/device environment is not recoverable')
    inv = row['dependencies']; require(digest(inv['path']) == inv['sha256'], 'compiler map changed')
    used = json.loads(Path(inv['path']).read_text())
    require(len(used) == inv['files'] and all(digest(path) == sha for path, sha in used.items()),
            'actual component compiler input changed')
    if 'binary' in row:
        require(digest(row['binary']['path']) == row['binary']['sha256'] and
                row['actual_link_inputs']['uncovered_external_inputs'] == 0 and all(
                digest(path) == sha for path, sha in row['actual_link_inputs']['frozen_inputs'].items()),
                'device fixture ELF/link input changed')
    return native


def prepare():
    data = base.committed(RESULT)
    require(KEY not in data and not ROOT.exists(), 'one fresh device fixture')
    failed = data['checked_adaptive_worker_device_v1']
    original = json.loads((B / 'checked-adaptive-worker-device-v1/record.json').read_text())
    require(compact(original) == failed and failed['stage'] == 'failed' and
            not failed['active'] and not original['commands_executed'] and
            'TimeoutExpired' in failed['error'] and 'dpkg-query' in failed['error'] and
            'dependencies' not in failed and not (B / 'checked-adaptive-worker-device-v1/fixture').exists(),
            'preserve the exact pre-owner package-query timeout;no device or compile attempt occurred')
    native = model.verify()
    old = json.loads((component.ROOT / 'record.json').read_text())
    require(data[component.KEY] == component.shared.compact(old) and old['passed'] and
            old['complete'] and old['ownership_closed'] and not old['active'],
            'copy only the closed qualified device environment;no old ELF dependency')
    require(old['environment_template'] == data[ENV_KEY]['environment'],
            'device environment must be recoverable from the committed canonical result')
    revision = base.basis.build.git(P, 'rev-parse', 'HEAD').decode().strip()
    pins = sources(revision)
    ROOT.mkdir(mode=0o700)
    for name in ('objects', 'tmp', 'cache'): (ROOT / name).mkdir()
    env = dict(native['environment'], TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'))
    database = json.loads(Path(native['compile_database']['path']).read_text())
    units = [unit for unit in database if 'strata_sycl_runtime.dir' in unit['command']]
    require(len(units) == 3, 'exact production runtime units')
    core = next(unit for unit in database if unit['file'] == str(model.ROOT / 'source' / component.CORE))
    specs = [(Path(unit['file']).stem, unit, unit['file']) for unit in units]
    specs += [('cache', core, core['file']), ('fixture', core, str(P / CPP))]
    commands, preprocess, depfiles, objects = [], [], [], []
    for label, unit, source in specs:
        flags = shlex.split(unit['command']); index = flags.index('-o')
        del flags[index:index + 2]; flags.remove('-c'); flags.remove(unit['file'])
        header, actual = ROOT / (label + '.headers.d'), ROOT / (label + '.build.d')
        obj = str(ROOT / 'objects' / (label + '.o')); objects.append(obj)
        preprocess.append([label, [*flags, '-M', '-MF', str(header), source]])
        commands.append([label, [*flags, '-MD', '-MF', str(actual), '-c', source, '-o', obj]])
        depfiles.append([str(header), str(actual)])
    commands.append(['compile', [base.COMPILER, '-fsycl', '-pthread', *objects,
                                '-ldl', '-lm', '-Wl,-t', '-o', str(ROOT / 'fixture')]])
    row = dict(stage='preparing', source_commit=revision, sources=pins,
        engine_source_commit=native['source_commit'], engine=native['binaries']['strata'],
        boot=native['boot'], driver=native['driver'], native_validation_probe=native['native_validation_probe'],
        runtime_manifest=native['runtime_manifest'], build_environment=env,
        environment_template=old['environment_template'], commands=commands, depfiles=depfiles,
        environment_recovery=dict(record_commit=revision, result_key=ENV_KEY,
            field='environment', build_environment_key=model.KEY,
            component_overrides=['TMPDIR', 'XDG_CACHE_HOME'],
            note='Exact templates are recovered from committed results;per-case tracing paths are set by the pinned controller before exec.'),
        previous_failure=dict(result_key='checked_adaptive_worker_device_v1',
            original_result_commit=revision,
            original_status_preserved=True, before_owned_commands=True, GPU_executed=False,
            closure='Controller exited1;check_output uses subprocess.run,which kills and waits for its timed-out child. No Owner was constructed,no compile command ran,and no component ELF exists. Original ownership flag remains unchanged.',
            change='Reuse the just-verified source/dependency/link inventories instead of repeating model.inputs package queries.'),
        plan=dict(order=['exception', 'tracked', 'failed-drain'], stop_on_first_failure=True,
            expected_exits=[0, 0, 1], build_budget=BUILD, device_budget=DEVICE,
            contract='Pending4096-byte real H2D,worker bad_alloc,production all-queue drain,payload D2H equality,main-unwind join,and production cache close with tracking off/on. Software failed-drain status is injected only after known completion;expect fail-stop before owner release. No deliberately invalid GPU work or environment mutation.',
            consumer='Qualify changed worker/cache error lifetime before positive-swap model work;retire raw traces,objects and component ELF after that qualification.'),
        passed=False, complete=False, active=False, ownership_closed=True, cases=[])
    owner = None
    try:
        allowed = allowed_inputs(native); allowed[str((P / CPP).resolve())] = pins[CPP]
        owner = base.supervision.Owner(ROOT / 'prepare-owned')
        for label, argv in preprocess:
            entry, _, _ = owner.run(label, argv, env, P, **BUILD)
            require(base.supervision.closed(entry) and entry['exit_code'] == 0, 'preprocessing failed: ' + label)
        found = set().union(*(dependencies(Path(header)) for header, _ in depfiles))
        require(all(allowed.get(path) == digest(path) for path in found), 'unplanned component input')
        write(ROOT / 'compiler-inputs.json', {path: allowed[path] for path in sorted(found)})
        row.update(stage='prepared', dependencies=dict(path=str(ROOT / 'compiler-inputs.json'),
            sha256=digest(ROOT / 'compiler-inputs.json'), files=len(found)))
    except BaseException as error: row.update(stage='failed', error=repr(error))
    finally:
        row.update(active=bool(owner and owner.active), commands_executed=owner.commands if owner else [],
            ownership_closed=bool(owner and owner.active is None and all(
                base.supervision.closed(value) for value in owner.commands)))
        save(row)
    require(row['stage'] == 'prepared' and row['ownership_closed'], row.get('error', 'preparation failed'))
    print('Prepared real-copy fixture;commit before compilation.', flush=True)


def build():
    row = json.loads((ROOT / 'record.json').read_text()); native = current(row)
    require(row['stage'] == 'prepared' and not (ROOT / 'owned-build').exists(), 'one component build')
    allowed = allowed_inputs(native); allowed[str((P / CPP).resolve())] = row['sources'][CPP]
    owner = None
    try:
        owner = base.supervision.Owner(ROOT / 'owned-build')
        for label, argv in row['commands']:
            entry, out, err = owner.run(label, argv, row['build_environment'], P, **BUILD)
            require(base.supervision.closed(entry) and entry['exit_code'] == 0,
                    label + ' failed: ' + err.read_text(errors='replace')[-3000:])
        require(set().union(*(dependencies(Path(actual)) for _, actual in row['depfiles'])) ==
                set(json.loads(Path(row['dependencies']['path']).read_text())), 'actual dependency set differs')
        current(row)
        # All generated object paths are below this component root, not external inputs.
        allowed.update({str(path.resolve()): digest(path) for path in (ROOT / 'objects').glob('*.o')})
        row.update(stage='built', binary=dict(path=str(ROOT / 'fixture'),
            sha256=digest(ROOT / 'fixture'), bytes=(ROOT / 'fixture').stat().st_size),
            actual_link_inputs=base.traced_link_inputs(allowed, ROOT))
    except BaseException as error: row.update(stage='failed', error=repr(error))
    finally:
        row.update(active=bool(owner and owner.active), commands_executed=owner.commands if owner else [],
            ownership_closed=bool(owner and owner.active is None and all(
                base.supervision.closed(value) for value in owner.commands)))
        save(row)
    require(row['stage'] == 'built' and row['ownership_closed'], row.get('error', 'build failed'))
    print('Built real-copy fixture;commit before GPU execution.', flush=True)


def check():
    row = json.loads((ROOT / 'record.json').read_text()); current(row)
    require(row['stage'] == 'built' and not row['cases'], 'one ordered device check')
    frozen = copy.deepcopy(row)
    providers = model.prior.fallback.retained.runtime.inventory(row['runtime_manifest'])
    for mode, expected in zip(row['plan']['order'], row['plan']['expected_exits']):
        root = ROOT / mode; root.mkdir()
        env = component.shared.device_environment(row, root)
        target = engine_once.native_environment(row, env)
        phases, mapped, owner = [], set(), None
        outcome = dict(mode=mode, expected_exit_code=expected, passed=False, host_probe_accepted=False)
        cursor = component.profile.cursor()
        def interact(io, entry):
            for phase in ('before-runtime', 'runtime-ready',
                          'before-failstop' if expected else 'before-exit'):
                while True:
                    line = io.line(45)
                    if line.startswith('COMPONENT_READY '): break
                require(line == 'COMPONENT_READY ' + phase, 'device checkpoint differs')
                require(Path('/proc', str(entry['pid']), 'exe').resolve() == Path(row['binary']['path']),
                        'actual fixture executable differs')
                actual = dict(value.split('=', 1) for value in Path('/proc', str(entry['pid']),
                    'environ').read_text().split('\0') if '=' in value)
                component.profile.same_environment(actual, target)
                seen = component.cache.shared.runtime.executable_mappings(
                    Path('/proc', str(entry['pid']), 'maps').read_text())
                require(seen <= set(providers) | {row['binary']['path']}, 'unplanned loaded provider')
                phases.append(phase); mapped.update(seen); io.send(b'X')
            io.proc.stdin.close()
            while not io.exit_watch.poll(0) or not io.eof:
                io.poll()
                if select.select([io.proc.stdout], [], [], .02)[0]: io.read(); io.pending.clear()
            io.exit(1)
        try:
            owner = base.supervision.Owner(root / 'owned')
            entry, out, err = owner.run(mode, [row['binary']['path'], mode], target, P,
                interaction=interact, **DEVICE)
            require(base.supervision.closed(entry) and entry['exit_code'] == expected, 'unexpected native exit')
            stdout, stderr = out.read_text(), err.read_text()
            require(stdout.count('DEVICE_COPY_ACCEPTED_PENDING') == 1 and
                    stdout.count('DEVICE_DRAIN_PAYLOAD_PASS') == 1 and
                    stdout.count('DEVICE_OWNERS_RELEASED') == (0 if expected else 1) and
                    stdout.count('checked_worker_device=PASS') == (0 if expected else 1) and
                    stderr.count('Strata SYCL worker:') == 1 and
                    stderr.count('Strata SYCL cleanup:') == (1 if expected else 0), 'retirement coverage differs')
            activation = ur_validation.preload_evidence(stderr, entry['pid'])
            required = {row['native_validation_probe']['library']['path'],
                row['native_validation_probe']['loader']['path'], env['ZE_ENABLE_ALT_DRIVERS']}
            require(required <= mapped and activation['loader'] == row['native_validation_probe']['loader']['path']
                and not any(value in stderr for value in ('UR_RESULT_ERROR_', 'ZE_RESULT_ERROR_', '[WARNING]', '[ERROR]'))
                and (not (root / 'validation.log').exists() or not (root / 'validation.log').stat().st_size),
                'runtime validation failed')
            current(frozen)
            outcome.update(native_exit_code=entry['exit_code'], complete=True, host_probe_accepted=True,
                           validation=activation, phases=phases, pending_copy=True, payload_equal=True,
                           owner_release=not bool(expected))
        except BaseException as error: outcome['error'] = repr(error)
        finally:
            if owner: ur_validation.finish(outcome, owner, cursor, row['boot'])
            outcome['passed'] = bool(outcome.get('passed') and outcome.get('complete') and
                outcome.get('driver_after') == row['driver'] and outcome.get('ownership_closed'))
            row['cases'].append(outcome)
            row.update(stage='checking' if outcome['passed'] else 'failed',
                active=bool(owner and owner.active), ownership_closed=bool(outcome.get('ownership_closed')))
            write(ROOT / 'record.json', row)
            if not outcome['passed']: save(row)
        require(outcome['passed'], outcome.get('error', 'device fixture failed'))
    row.update(stage='closed', passed=True, complete=True,
        decision='Worker/error/cache-close component passed;fresh model zero/positive-swap,task,independent prefill,repeated clean speed and262144/RESTORE remain unqualified.')
    save(row); print('Real-copy retirement passed in three fresh validated processes.', flush=True)


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'build', 'check'), 'use prepare|build|check')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base.guards.no_model_owners([]); base.guards.driver_guard()
        prepare() if sys.argv[1] == 'prepare' else build() if sys.argv[1] == 'build' else check()
