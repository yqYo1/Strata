#!/usr/bin/env python3
"""Build the independently authored cache-map correction; repository LICENSE.

Compose exactly three changed native files on the measured adaptive engine.
Keep that engine as the development fallback, and retire its older comparison
generation before creating this one. Compilation supplies no model, numerical,
performance, ordinary teardown or physical-context qualification.
"""
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import select
import shlex
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import adaptive_cache_model as prior
import cache_residency as component

base, P, B, RESULT = prior.base, prior.P, prior.B, prior.RESULT
require, digest, write = base.require, base.digest, base.write
ROOT = B / 'cache-map-model-v1'
KEY = 'cache_map_model_build_v1'
RETIREMENT_KEY = 'gdn16_model_fallback_retirement_v1'
CONTROLLER = Path(__file__).resolve()
CORRECTION = '9974789b0ec57c4146beedb35dfc811595cc15c8'
CHANGED = (component.CORE, component.HEADER, component.PROGRAM)


def save(row):
    write(ROOT / 'record.json', row)
    data = json.loads(RESULT.read_text())
    data[KEY] = base.compact(row)
    write(RESULT, data)


def projection():
    full, native = prior.projection()
    before = dict(native)
    patches = component.projection(CORRECTION)
    require(set(patches) == set(CHANGED), 'exact cache correction files required')
    for name, raw in patches.items():
        native[name] = raw
    require(len(native) == 557 and
            {name for name in native if native[name] != before[name]} == set(CHANGED),
            'correction changed unrelated measured native inputs')
    return full, native


def qualified_component(*, actual=False):
    # The closed qualification is evidence, not an ELF input to this engine.
    # Reopen its actual artifacts at preparation, without making future engine
    # admission depend indefinitely on the older fallback/fixture generation.
    row = base.committed(RESULT)[component.KEY]
    require(row['source_commit'] == CORRECTION and row['stage'] == 'checked' and
            row['passed'] and row['complete'] and row['ownership_closed'] and
            not row['active'] and row['decision']['mapping_component_qualified'],
            'closed mapping/payload/loan component qualification required')
    if actual:
        record = json.loads((component.ROOT / 'record.json').read_text())
        require(component.shared.compact(record) == row, 'original component receipt changed')
        component.current(record)
        for name, raw in component.projection(CORRECTION).items():
            require((component.ROOT / 'projection' / name).read_bytes() == raw,
                    'actual component-qualified native projection changed: ' + name)
    return {key: row[key] for key in ('source_commit', 'binary', 'dependencies')}


def retired_comparison(data, fallback):
    row = data.get(RETIREMENT_KEY, {})
    require(row.get('original_inputs_recovered') is True and
            row.get('deleted_verified') is True and
            row.get('current_admission_after_deletion') is True and
            row.get('selected_fallback_root') == str(prior.ROOT) and
            row.get('selected_fallback_engine_sha256') == fallback['binaries']['strata']['sha256'],
            'select and verify the adaptive fallback;recover and retire the old comparison first')
    require(all(not Path(asset['path']).exists() for asset in
                data[prior.fallback.KEY]['binaries'].values()),
            'old comparison still retains engine/fixture executables')
    require(not (prior.fallback.ROOT / 'build').exists(),
            'old comparison still retains its disposable build tree')
    return row


def inputs(row):
    require(base.controls(row['source_commit'], controller=CONTROLLER) == row['controller_inputs'],
            'cache-map build controller changed')
    require(qualified_component() == row['component_qualification'],
            'component qualification changed')
    full, native = projection()
    require({name: hashlib.sha256(raw).hexdigest() for name, raw in native.items()} ==
            row['native_sources'], 'projected cache-map native source changed')
    allowed = base.sdk()
    allowed.update(prior.fallback.retained.runtime.inventory(row['external_manifest']))
    providers = prior.fallback.retained.runtime.inventory(row['runtime_manifest'])
    allowed.update(providers)
    require(prior.fallback.retained.runtime.qualified_probe(base.committed(RESULT), providers) ==
            row['native_validation_probe'], 'qualified native validation probe changed')
    for name, raw in native.items():
        path = ROOT / 'source' / name
        require(path.is_file() and not path.is_symlink() and path.read_bytes() == raw,
                'frozen cache-map native input changed: ' + name)
        allowed[str(path.resolve())] = row['native_sources'][name]
    for group in full['dependencies']:
        recovered = {name: hashlib.sha256(raw).hexdigest() for name, raw in base.basis.build.blobs(
            Path(group['root']), group['revision'], group['source_sha256'])}
        require(recovered == group['source_sha256'], 'dependency Git recovery changed')
        for name, sha in recovered.items():
            path = (Path(group['root']) / name).resolve()
            require(digest(path) == sha, 'actual dependency changed: ' + str(path))
            allowed[str(path)] = sha
    allowed.update(row['generated_expected'])
    require(base.host_link_inventory.inventory(base.COMPILER, row['environment']) ==
            row['host_link_inputs'], 'compiler/linker inputs changed')
    packages = subprocess.check_output(['dpkg-query', '-W', '-f=${Package} ${Version}\n',
        'intel-oneapi-dpcpp-cpp-2026.2', 'intel-oneapi-compiler-dpcpp-cpp-runtime-2026.2',
        'intel-oneapi-umf-1.2'], text=True, timeout=15).splitlines()
    require(packages == row['packages'] and base.guards.driver_guard() == row['driver'] and
            Path('/proc/sys/kernel/random/boot_id').read_text().strip() == row['boot'],
            'qualified packages/boot/driver changed')
    return allowed


def prepare():
    data = base.committed(RESULT)
    require(KEY not in data and not ROOT.exists(), 'one fresh cache-map engine generation')
    fallback = prior.verify()
    require(len(fallback['binaries']) == 9, 'preserve the nine-ELF adaptive fallback')
    retired = retired_comparison(data, fallback)
    qualification = qualified_component(actual=True)
    full, native = projection()
    revision = base.basis.build.git(P, 'rev-parse', 'HEAD').decode().strip()
    controls = base.controls(revision, controller=CONTROLLER)
    providers = prior.fallback.retained.runtime.inventory(fallback['runtime_manifest'])
    ROOT.mkdir(mode=0o700)
    for name in ('source', 'tmp', 'cache'):
        (ROOT / name).mkdir()
    for name, raw in native.items():
        path = ROOT / 'source' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    write(ROOT / 'runtime-inputs.json', providers)
    env = dict(fallback['environment'], TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'))
    cfg = [value.replace(str(prior.ROOT), str(ROOT)) for value in fallback['configure_argv']]
    row = dict(stage='prepared', source_commit=revision, controller_inputs=controls,
        native_sources={name: hashlib.sha256(raw).hexdigest() for name, raw in native.items()},
        composition=dict(selected_native_commit=prior.SELECTED,
            adaptive_fix_before=prior.FIX_BEFORE, adaptive_fix_commit=prior.FIX,
            correction_before=component.BEFORE, correction_commit=CORRECTION,
            changed_from_adaptive=list(CHANGED), independent_from_XeStrata=True),
        component_qualification=qualification,
        fallback=dict(root=str(prior.ROOT), source_commit=fallback['source_commit'],
            binaries=fallback['binaries'],
            consumer='Development fallback until changed-engine device/model checks and comparison closure;not full inference adoption.'),
        retired_comparison=dict(result_key=RETIREMENT_KEY,
            manifest=retired['manifest']),
        **{key: fallback[key] for key in ('compiler', 'sdk', 'external_manifest',
            'host_link_inputs', 'packages', 'dependencies', 'native_validation_probe', 'boot', 'driver')},
        runtime_manifest=dict(path=str(ROOT / 'runtime-inputs.json'),
            sha256=digest(ROOT / 'runtime-inputs.json'), files=len(providers)),
        environment=env, configure_argv=cfg, targets=fallback['targets'],
        generated_expected={path.replace(str(prior.ROOT), str(ROOT)): sha
            for path, sha in fallback['generated_expected'].items()},
        expected_compiler_inputs=fallback['compiled_inputs']['files'],
        commands=[['configure', cfg], ['compile', ['/usr/bin/cmake', '--build', str(ROOT / 'build'),
            '--target', *fallback['targets'], '-j', '4']],
            ['dependencies', ['/usr/bin/ninja', '-C', str(ROOT / 'build'), '-t', 'deps']]],
        budgets=dict(wall=1800, cpu=7200, rss_cap=12 << 30, text_cap=16 << 20,
            file_cap=512 << 20, total_cap=4 << 30), whole_wall_seconds=3600,
        commands_executed=[], active=False, ownership_closed=True,
        passed=False, GPU_executed=False, model_executed=False, inference_adopted=False,
        next='Fresh bounded submission/wrapper and actual-model zero/positive-swap checks,including next-GEN payload/loan identity. Then fixed-token/task,independent prefill,and repeated clean strong/weak-MTP timing. No prior fixture qualification is inherited by these changed native inputs.')
    try:
        inputs(row)
    except BaseException as error:
        row.update(stage='failed-admission', error=repr(error))
        save(row)
        raise
    save(row)
    print('Prepared cache-map engine recipe;commit before CPU compilation.', flush=True)


def build():
    row = json.loads((ROOT / 'record.json').read_text())
    require(base.committed(RESULT)[KEY] == base.compact(row) and row['stage'] == 'prepared' and
            not (ROOT / 'owned-build').exists(), 'one frozen fresh cache-map build')
    retired_comparison(base.committed(RESULT), prior.verify())
    allowed = inputs(row)
    frozen, owner = copy.deepcopy(row), None
    deadline = time.monotonic() + row['whole_wall_seconds']

    def monitor(io, entry):
        next_size = 0.
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll()
            require(time.monotonic() < deadline, 'whole cache-map build deadline')
            if time.monotonic() >= next_size:
                require(base.supervision.artifact_bytes(ROOT) <= row['budgets']['total_cap'],
                        'whole cache-map generation byte budget')
                next_size = time.monotonic() + 1
            if select.select([io.proc.stdout], [], [], .02)[0]:
                io.read()
                io.pending.clear()
        io.exit(1)

    try:
        owner = base.supervision.Owner(ROOT / 'owned-build')
        for label, argv in row['commands']:
            print('CPU build:', label, flush=True)
            entry, out, err = owner.run(label, argv, row['environment'], ROOT, interaction=monitor,
                **dict(row['budgets'], wall=min(row['budgets']['wall'], deadline-time.monotonic())))
            require(base.supervision.closed(entry) and entry['exit_code'] == 0,
                    label + ' failed: ' + (out.read_text(errors='replace') + err.read_text(errors='replace'))[-5000:])
        inputs(frozen)
        found = {str((Path(value.strip()) if Path(value.strip()).is_absolute() else
                      ROOT / 'build' / value.strip()).resolve())
                 for value in out.read_text().splitlines() if value.startswith('    ') and value.strip()}
        require(len(found) == row['expected_compiler_inputs'] and
                str(ROOT / 'source' / component.PROGRAM) in found and
                all(allowed.get(path) == digest(path) and '/MistVVK/' not in path for path in found),
                'unplanned or incomplete actual compiler inputs')
        write(ROOT / 'compiler-inputs.json', {path: allowed[path] for path in sorted(found)})
        db = ROOT / 'build/compile_commands.json'
        commands, units = json.loads(db.read_text()), []
        for name in (component.PROGRAM, component.CORE,
                     prior.fallback.source.KERNEL, prior.fallback.source.FIXTURE):
            matches = [value for value in commands if value['file'] == str(ROOT / 'source' / name)]
            if name == prior.fallback.source.KERNEL:
                matches = [value for value in matches if '-o CMakeFiles/strata_kernels.dir/' in value['command']]
            require(len(matches) == 1, 'one actual production compile target required: ' + name)
            argv = shlex.split(matches[0]['command'])
            require(argv[0] == base.COMPILER and '-O3' in argv and '-fsycl' in argv and
                    not any(value in argv for value in ('-ffast-math', '-Ofast', '-ffp-contract=fast')),
                    'production compile options changed: ' + name)
            units.append(dict(source=name, argv_sha256=hashlib.sha256(json.dumps(argv).encode()).hexdigest()))
        row.update(stage='built-not-executed', passed=True,
            compiled_inputs=dict(path=str(ROOT / 'compiler-inputs.json'),
                sha256=digest(ROOT / 'compiler-inputs.json'), files=len(found), uncovered=0),
            actual_link_inputs=base.traced_link_inputs(allowed, ROOT),
            compile_database=dict(path=str(db), sha256=digest(db), units=units),
            binaries={name: dict(path=str(ROOT / 'build' / name),
                sha256=digest(ROOT / 'build' / name), bytes=(ROOT / 'build' / name).stat().st_size)
                for name in row['targets']})
    except BaseException as error:
        row.update(stage='failed', passed=False, error=repr(error))
    finally:
        row.update(active=bool(owner and owner.active), commands_executed=owner.commands if owner else [],
            ownership_closed=bool(owner and all(base.supervision.closed(value) for value in owner.commands)))
        require(owner is None or owner.active is None, 'build ownership unresolved')
        save(row)
    print(json.dumps({key: row.get(key) for key in ('stage', 'passed', 'error')}), flush=True)
    require(row['passed'], 'cache-map model build failed')


def verify():
    row = json.loads((ROOT / 'record.json').read_text())
    require(base.committed(RESULT)[KEY] == base.compact(row) and row['passed'] and
            row['stage'] == 'built-not-executed' and row['ownership_closed'] and not row['active'] and
            all(base.supervision.closed(value) for value in row['commands_executed']),
            'closed complete cache-map build required')
    allowed = inputs(row)
    asset = row['compiled_inputs']
    require(digest(asset['path']) == asset['sha256'], 'actual compiler map changed')
    found = json.loads(Path(asset['path']).read_text())
    require(len(found) == asset['files'] and
            all(allowed.get(path) == sha == digest(path) for path, sha in found.items()),
            'actual compiler inputs changed')
    require(digest(row['compile_database']['path']) == row['compile_database']['sha256'] and
            row['actual_link_inputs']['uncovered_external_inputs'] == 0 and
            all(allowed.get(path) == sha == digest(path)
                for path, sha in row['actual_link_inputs']['frozen_inputs'].items()),
            'actual compile/link recipe changed')
    external = prior.fallback.retained.runtime.inventory(row['external_manifest'])
    require(not any(Path(path).is_relative_to(old) for old in (prior.ROOT, prior.fallback.ROOT)
                    for path in (*found, *row['actual_link_inputs']['frozen_inputs'], *external)),
            'new generation still consumes old build products')
    require(set(row['binaries']) == set(row['targets']) and all(
        digest(asset['path']) == asset['sha256'] and Path(asset['path']).stat().st_size == asset['bytes']
        for asset in row['binaries'].values()), 'same-build binaries changed')
    return row


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ('projection', 'prepare', 'build', 'verify'),
            'use projection|prepare|build|verify')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base.guards.no_model_owners([])
        if sys.argv[1] == 'projection':
            qualified_component(actual=True)
            _, native = projection()
            print(json.dumps(dict(native_files=len(native), changed=list(CHANGED),
                inference_build_created=False, GPU_executed=False)), flush=True)
        elif sys.argv[1] == 'prepare':
            prepare()
        elif sys.argv[1] == 'build':
            build()
        else:
            print(json.dumps(dict(binaries=len(verify()['binaries']), GPU_executed=False)), flush=True)
