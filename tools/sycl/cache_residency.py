#!/usr/bin/env python3
"""Bounded production-cache identity oracle; Main-authored, repository LICENSE.

Rebuild three unchanged runtime units and the changed cache for one component
fixture. Neither retained inference generation is rebuilt or overwritten.
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
import profile_window as shared

cache, profile = shared.cache, shared.profile
P, B, RESULT = shared.P, shared.B, shared.RESULT
ROOT = cache.model.adaptive.ROOT / 'residency-map-v2'
KEY = 'adaptive_cache_residency_map_v2'
BEFORE = '3e22e21b0b00eea9c51f1ca62407b368d340a4ed'
CPP = 'tests/sycl/cache_residency.cpp'
CORE = 'src/core/expert_cache.cpp'
HEADER = 'include/strata/core/expert_cache.hpp'
PROGRAM = 'src/program/generate.cpp'
PROJECT = (CPP, CORE, HEADER, PROGRAM, 'cmake/sycl_backend.cmake')
require, digest, write = shared.require, shared.digest, shared.write
BUILD = dict(wall=900, cpu=1800, rss_cap=8 << 30, text_cap=8 << 20,
             file_cap=256 << 20, total_cap=512 << 20)
DEVICE = shared.DEVICE_BUDGET


def save(row):
    write(ROOT / 'record.json', row)
    data = json.loads(RESULT.read_text()); data[KEY] = shared.compact(row); write(RESULT, data)


def sources(revision):
    pins = cache.build.controls(revision, controller=Path(__file__).resolve())
    for name in PROJECT:
        raw = cache.shared.gen.git('show', revision + ':' + name)
        require((P / name).read_bytes() == raw, 'uncommitted oracle input: ' + name)
        pins[name] = hashlib.sha256(raw).hexdigest()
    return pins


def projection(revision):
    _, old = cache.model.adaptive.projection()
    for name in (CORE, HEADER):
        require(cache.shared.gen.git('show', BEFORE + ':' + name) == old[name],
                'cache baseline differs from measured composition')
    result = {name: cache.shared.gen.git('show', revision + ':' + name) for name in (CORE, HEADER)}
    result[PROGRAM] = cache.build.basis.merge_delta(old[PROGRAM], PROGRAM, BEFORE, revision)
    require(cache.build.basis.merge_delta(result[PROGRAM], PROGRAM, revision, BEFORE) == old[PROGRAM],
            'program change is not the reversible adaptive-only delta')
    return result


def dependencies(path):
    raw = path.read_text().replace('\\\n', ' ')
    require(':' in raw, 'compiler dependency target missing')
    return {str(Path(p).resolve()) for p in shlex.split(raw.split(':', 1)[1])}


def current(row):
    require(cache.build.committed(RESULT)[KEY] == shared.compact(row) and
            sources(row['source_commit']) == row['sources'], 'commit/freeze oracle inputs and results')
    native, prior, providers, _ = shared.basis()
    require(native['compiler'] == row['compiler'] and native['boot'] == row['boot']
            and native['driver'] == row['driver'] and native['binaries']['strata'] == row['engine']
            and prior['environment'] == row['environment_template']
            and prior['native_validation_probe'] == row['native_validation_probe'], 'qualified basis changed')
    inv = row['runtime_manifest']
    require(digest(inv['path']) == inv['sha256'] and json.loads(Path(inv['path']).read_text()) == providers,
            'oracle runtime inventory changed')
    for name, raw in projection(row['source_commit']).items():
        p = ROOT / 'projection' / name
        require(p.read_bytes() == raw, 'projected oracle input changed')
    inv = row['dependencies']
    require(digest(inv['path']) == inv['sha256'], 'oracle dependency manifest changed')
    deps = json.loads(Path(inv['path']).read_text())
    require(len(deps) == inv['files'] and all(digest(p) == sha for p, sha in deps.items()), 'compiler input changed')
    if 'binary' in row:
        require(digest(row['binary']['path']) == row['binary']['sha256'], 'oracle ELF changed')
    return native, providers


def prepare():
    require(not ROOT.exists() and KEY not in cache.build.committed(RESULT), 'fresh oracle only')
    previous = cache.build.committed(RESULT)['adaptive_cache_residency_map_v1']
    require(previous['stage'] == 'failed-admission' and not previous['GPU_executed']
            and previous['ownership_closed'], 'preserve the original failed CMake admission')
    revision = cache.shared.gen.git('rev-parse', 'HEAD').decode().strip()
    pins = sources(revision)
    native, prior, providers, _ = shared.basis()
    generated = projection(revision)
    ROOT.mkdir(mode=0o700)
    for name, raw in generated.items():
        p = ROOT / 'projection' / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(raw)
    for name in ('tmp', 'cache', 'objects'): (ROOT / name).mkdir()
    write(ROOT / 'runtime-inputs.json', providers)
    database = json.loads(Path(native['compile_database']['path']).read_text())
    units = [x for x in database if 'strata_sycl_runtime.dir' in x['command']]
    require(len(units) == 3, 'exact three qualified runtime translation units required')
    core = next(x for x in database if x['file'].endswith('/' + CORE))
    program = next(x for x in database if x['file'].endswith('/' + PROGRAM))

    def prefix(unit):
        args = shlex.split(unit['command']); index = args.index('-o')
        del args[index:index + 2]; args.remove('-c'); args.remove(unit['file'])
        return [args[0], '-I' + str(ROOT / 'projection/include'), *args[1:]]

    specs = [(Path(x['file']).stem, prefix(x), x['file'], False) for x in units]
    specs += [('cache', prefix(core), str(ROOT / 'projection' / CORE), False),
              ('fixture', prefix(core), str(P / CPP), False),
              ('program-syntax', prefix(program), str(ROOT / 'projection' / PROGRAM), True)]
    commands, preprocess, depfiles, objects = [], [], [], []
    for label, flags, source, syntax in specs:
        pre = ROOT / (label + '.headers.d'); dep = ROOT / (label + '.build.d')
        preprocess.append([label, [*flags, '-M', '-MT', 'fixture', '-MF', str(pre), source]])
        command = [*flags, '-MD', '-MT', 'fixture', '-MF', str(dep)]
        if syntax: command += ['-fsyntax-only', source]
        else:
            obj = str(ROOT / 'objects' / (label + '.o')); objects.append(obj)
            command += ['-c', source, '-o', obj]
        commands.append([label, command]); depfiles.append([str(pre), str(dep)])
    commands.append(['link', [native['compiler']['path'], '-fsycl', '-pthread', *objects,
                             '-ldl', '-lm', '-o', str(ROOT / 'fixture')]])
    env = dict(native['environment'], TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'))
    row = dict(stage='preparing', source_commit=revision, sources=pins,
        compiler=native['compiler'], sdk=native['sdk'], boot=native['boot'], driver=native['driver'],
        engine=native['binaries']['strata'], native_validation_probe=prior['native_validation_probe'],
        environment_template=copy.deepcopy(prior['environment']), build_environment=env,
        runtime_manifest=dict(path=str(ROOT / 'runtime-inputs.json'), sha256=digest(ROOT / 'runtime-inputs.json')),
        commands=commands, depfiles=depfiles, cases=[], passed=False, complete=False, active=False,
        ownership_closed=True, plan=dict(order='Build only component and syntax-check the projected generator;then fresh legacy and fixed traced/validated processes. Stop on first failure;no retries.',
            contract='130claimed slots,two layers,one forced tail swap. Check host/device/callback identity,pending exclusion,incoming payload,six invalid maps,same-slot128-tail scratch loan/refill and unlent prefix. Legacy omits publication calls in the same fixture;not an old-ELF speed comparison.',
            limits='No model arithmetic,actual failed-task causality,speed/adoption,multi-GPU,ordinary runtime teardown or262144qualification. Two inference generations remain unchanged.',
            build_budget=BUILD,device_budget=DEVICE,whole_bytes=512 << 20,
            consumer='Qualify the adaptive mapping correction before a new native generation and task/speed plan. Retire routine raw/build/JIT output after useful evidence;keep component ELF only until changed-engine qualification.'))
    owner = None
    def persist():
        write(ROOT / 'prepare-execution.json', dict(active=bool(owner and owner.active),
            commands=owner.commands if owner else []))
    try:
        owner = cache.build.supervision.Owner(ROOT / 'prepare-owned', persist)
        for label, argv in preprocess:
            entry, _, _ = owner.run(label, argv, env, P, interaction=monitor, **BUILD)
            require(cache.build.supervision.closed(entry) and entry['exit_code'] == 0, 'oracle preprocessing failed: ' + label)
        allowed = cache.model.adaptive.inputs(native)
        allowed[str((P / CPP).resolve())] = pins[CPP]
        allowed.update({str((ROOT / 'projection' / name).resolve()): hashlib.sha256(raw).hexdigest()
                        for name, raw in generated.items()})
        found = set().union(*(dependencies(Path(pre)) for pre, _ in depfiles))
        require(all(p in allowed and digest(p) == allowed[p] for p in found), 'unqualified compiler dependency')
        write(ROOT / 'compiler-inputs.json', {p: allowed[p] for p in sorted(found)})
        row.update(stage='prepared', dependencies=dict(path=str(ROOT / 'compiler-inputs.json'),
            sha256=digest(ROOT / 'compiler-inputs.json'), files=len(found)))
    except BaseException as error: row.update(stage='failed', error=repr(error))
    finally:
        row.update(active=bool(owner and owner.active), ownership_closed=bool(owner and owner.active is None
            and all(cache.build.supervision.closed(x) for x in owner.commands)))
        save(row)
    require(row['stage'] == 'prepared' and row['ownership_closed'], row.get('error', 'preparation not closed'))
    print('Prepared residency oracle;commit before component build.', flush=True)


def build():
    row = json.loads((ROOT / 'record.json').read_text()); current(row)
    require(row['stage'] == 'prepared' and not (ROOT / 'build-owned').exists(), 'fresh component build only')
    owner = None
    def persist():
        write(ROOT / 'build-execution.json', dict(active=bool(owner and owner.active),
            commands=owner.commands if owner else []))
    try:
        owner = cache.build.supervision.Owner(ROOT / 'build-owned', persist)
        for label, argv in row['commands']:
            print('CPU component build:', label, flush=True)
            entry, _, _ = owner.run(label, argv, row['build_environment'], P, interaction=monitor, **BUILD)
            require(cache.build.supervision.closed(entry) and entry['exit_code'] == 0, 'component build failed: ' + label)
        expected = json.loads(Path(row['dependencies']['path']).read_text())
        found = set().union(*(dependencies(Path(dep)) for _, dep in row['depfiles']))
        require(found == set(expected), 'actual compiler dependency set changed')
        current(row)
        row.update(stage='built', binary=dict(path=str(ROOT / 'fixture'), sha256=digest(ROOT / 'fixture'),
                                             bytes=(ROOT / 'fixture').stat().st_size))
    except BaseException as error: row.update(stage='failed', error=repr(error))
    finally:
        row.update(active=bool(owner and owner.active), ownership_closed=bool(owner and owner.active is None
            and all(cache.build.supervision.closed(x) for x in owner.commands)))
        save(row)
    require(row['stage'] == 'built' and row['ownership_closed'], row.get('error', 'component build not closed'))
    print('Built component oracle;commit before GPU checks.', flush=True)


def monitor(io, entry):
    io.proc.stdin.close()
    while not io.exit_watch.poll(0) or not io.eof:
        io.poll()
        require(cache.build.supervision.artifact_bytes(ROOT) <= 512 << 20, 'whole oracle output budget')
        if select.select([io.proc.stdout], [], [], .02)[0]: io.read(); io.pending.clear()
    io.exit(1)


def check():
    row = json.loads((ROOT / 'record.json').read_text()); _, providers = current(row)
    require(row['stage'] == 'built' and not row['cases'], 'fresh ordered oracle checks only')
    frozen = copy.deepcopy(row)
    for mode in ('legacy', 'fixed'):
        root = ROOT / mode; root.mkdir()
        env = shared.device_environment(row, root)
        target = cache.model.engine_once.native_environment(row, env)
        owner, phases, mapped = None, [], set()
        outcome = dict(mode=mode, passed=False, complete=False, host_probe_accepted=False)
        cursor = profile.cursor()
        def persist():
            write(root / 'execution.json', dict(outcome, active=bool(owner and owner.active),
                                               commands=owner.commands if owner else []))
        def interact(io, entry):
            for phase in ('before-runtime', 'runtime-ready', 'before-exit'):
                while True:
                    line = io.line(45)
                    if line.startswith('COMPONENT_READY '): break
                require(line == 'COMPONENT_READY ' + phase, 'oracle phase differs')
                phases.append(phase)
                require(Path('/proc', str(entry['pid']), 'exe').resolve() == Path(row['binary']['path']).resolve(), 'oracle ELF identity changed')
                actual = dict(v.split('=', 1) for v in Path('/proc', str(entry['pid']), 'environ').read_text().split('\0') if '=' in v)
                profile.same_environment(actual, target)
                seen = cache.shared.runtime.executable_mappings(Path('/proc', str(entry['pid']), 'maps').read_text())
                require(seen <= set(providers) | {row['binary']['path']}, 'unplanned oracle provider')
                mapped.update(seen); persist(); io.send(b'X')
            io.proc.stdin.close(); deadline = time.monotonic() + 45
            while not io.exit_watch.poll(0) or not io.eof:
                io.poll(); require(time.monotonic() < deadline, 'oracle teardown deadline')
                require(cache.build.supervision.artifact_bytes(ROOT) <= 512 << 20, 'whole oracle output budget')
                if select.select([io.proc.stdout], [], [], .02)[0]: io.read(); io.pending.clear()
            io.exit(1)
        try:
            owner = cache.build.supervision.Owner(root / 'owned', persist)
            entry, out, err = owner.run(mode, [row['binary']['path'], mode], target, P, interaction=interact, **DEVICE)
            require(cache.build.supervision.closed(entry) and entry['exit_code'] == 0, 'oracle native failure')
            outcome['native_exit_code'] = 0; persist()
            stdout, stderr = out.read_text(), err.read_text()
            marker = ('legacy_stale_map=REPRODUCED overwritten_slot=129 incoming_payload=PASS' if mode == 'legacy' else
                'residency_map=PASS pending_exclusion=PASS invalid_maps=6 loan_refill=PASS payload=PASS prefix=PASS')
            require(stdout.count(marker) == 1 and stdout.count('fixture_status=PASS mode=cache-residency') == 1,
                    'oracle result markers missing')
            activation = cache.shared.runtime.validation.preload_evidence(stderr, entry['pid'])
            required = {row['native_validation_probe']['loader']['path'], row['native_validation_probe']['library']['path'], env['ZE_ENABLE_ALT_DRIVERS']}
            require(required <= mapped and activation['loader'] == row['native_validation_probe']['loader']['path']
                and not any(x in stderr for x in ('UR_RESULT_ERROR_', 'ZE_RESULT_ERROR_', '[WARNING]', '[ERROR]'))
                and (not (root / 'validation.log').exists() or not (root / 'validation.log').stat().st_size), 'oracle validation failed')
            current(frozen)
            outcome.update(complete=True, host_probe_accepted=True, validation=activation, checks=marker)
        except BaseException as error: outcome['error'] = repr(error)
        finally:
            if owner: cache.shared.runtime.validation.finish(outcome, owner, cursor, row['boot'])
            outcome['passed'] = bool(outcome.get('passed') and outcome.get('complete')
                and outcome.get('driver_after') == row['driver'] and outcome.get('ownership_closed'))
            row['cases'].append(outcome)
            row.update(stage='checking' if outcome.get('passed') else 'failed', active=bool(owner and owner.active),
                       ownership_closed=bool(outcome.get('ownership_closed')))
            write(ROOT / 'record.json', row)
            if not outcome.get('passed'): save(row)
        require(outcome.get('passed'), outcome.get('error', 'oracle check failed'))
    row.update(stage='checked', passed=True, complete=True)
    row['decision'] = dict(mapping_component_qualified=True, inference_adopted=False, speed_qualified=False,
        next='Retire the now-superseded older engine after source/consumer reconciliation;compose and qualify the changed engine,then fixed-token/task and repeated clean speed checks.')
    save(row); print('Mapping component checks passed;no model or speed qualification.', flush=True)


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'build', 'check'), 'cache_residency.py prepare|build|check')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); cache.build.guards.no_model_owners([])
        {'prepare': prepare, 'build': build, 'check': check}[sys.argv[1]]()
