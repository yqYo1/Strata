#!/usr/bin/env python3
"""Build the owned adaptive-refill fix on the measured GDN16 model; MIT.

Only the reversible generate.cpp fix enters this composed native source.
Compilation is CPU-only; new binaries require fresh bounded GPU/model checks.
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
import gdn_model_build as fallback

base, P, B, RESULT = fallback.base, fallback.P, fallback.B, fallback.RESULT
require, digest, write = base.require, base.digest, base.write
ROOT, KEY = B / 'adaptive-cache-model-v1', 'adaptive_cache_model_build_v1'
CONTROLLER = Path(__file__).resolve()
SELECTED = '521c1757b289ab9129378aace21a5f0ffcfb2af5'
FIX_BEFORE = '5465f65d0a2763374400255283102596821dcd0c'
FIX = 'f8f632341bb4c965c0fb67dee998ab67ad6fd941'
PROGRAM = 'src/program/generate.cpp'


def save(row):
    write(ROOT / 'record.json', row)
    data = json.loads(RESULT.read_text()); data[KEY] = base.compact(row); write(RESULT, data)


def projection():
    full, native = fallback.source.sources(SELECTED)
    before = native[PROGRAM]
    native[PROGRAM] = base.basis.merge_delta(before, PROGRAM, FIX_BEFORE, FIX)
    require(base.basis.merge_delta(native[PROGRAM], PROGRAM, FIX, FIX_BEFORE) == before,
            'adaptive fix changed unrelated native source')
    require(len(native) == 557 and native[PROGRAM] != before, 'one owned program delta required')
    return full, native


def inputs(row):
    require(base.controls(row['source_commit'], controller=CONTROLLER) == row['controller_inputs'],
            'adaptive build controller changed')
    full, native = projection()
    require({n: hashlib.sha256(raw).hexdigest() for n, raw in native.items()} == row['native_sources'],
            'adaptive projected source changed')
    allowed = base.sdk()
    allowed.update(fallback.retained.runtime.inventory(row['external_manifest']))
    providers = fallback.retained.runtime.inventory(row['runtime_manifest'])
    allowed.update(providers)
    require(fallback.retained.runtime.qualified_probe(base.committed(RESULT), providers) ==
            row['native_validation_probe'], 'qualified native validation probe changed')
    for name, raw in native.items():
        path = ROOT / 'source' / name
        require(path.is_file() and not path.is_symlink() and path.read_bytes() == raw,
                'frozen adaptive native input changed: ' + name)
        allowed[str(path.resolve())] = row['native_sources'][name]
    for group in full['dependencies']:
        recovered = {n: hashlib.sha256(raw).hexdigest() for n, raw in base.basis.build.blobs(
            Path(group['root']), group['revision'], group['source_sha256'])}
        require(recovered == group['source_sha256'], 'dependency Git recovery changed')
        for name, sha in recovered.items():
            path = (Path(group['root']) / name).resolve()
            require(digest(path) == sha, 'actual dependency changed: ' + str(path))
            allowed[str(path)] = sha
    allowed.update(row['generated_expected'])
    require(base.host_link_inventory.inventory(base.COMPILER, row['environment']) == row['host_link_inputs'],
            'compiler/linker inputs changed')
    packages = subprocess.check_output(['dpkg-query', '-W', '-f=${Package} ${Version}\n',
        'intel-oneapi-dpcpp-cpp-2026.2', 'intel-oneapi-compiler-dpcpp-cpp-runtime-2026.2',
        'intel-oneapi-umf-1.2'], text=True, timeout=15).splitlines()
    require(packages == row['packages'] and base.guards.driver_guard() == row['driver'] and
            Path('/proc/sys/kernel/random/boot_id').read_text().strip() == row['boot'],
            'qualified packages/boot/driver changed')
    return allowed


def prepare():
    data = base.committed(RESULT)
    require(KEY not in data and not ROOT.exists(), 'one fresh adaptive model generation')
    retired = data['gdn16_parent_build_retirement']
    require(retired['deleted_verified'] and retired['original_inputs_recovered'] and
            retired['current_admission_after_deletion'], 'retire the unused eight-ELF parent first')
    before = fallback.verify()
    require(before['source_commit'] == SELECTED and len(before['binaries']) == 9,
            'preserve the validated nine-ELF fallback')
    full, native = projection()
    revision = base.basis.build.git(P, 'rev-parse', 'HEAD').decode().strip()
    gate = data['gdn16_model_component_device_v1']
    providers = fallback.retained.runtime.inventory(gate['runtime_manifest'])
    ROOT.mkdir(mode=0o700)
    for part in ('source', 'tmp', 'cache'): (ROOT / part).mkdir()
    for name, raw in native.items():
        path = ROOT / 'source' / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
    write(ROOT / 'runtime-inputs.json', providers)
    env = dict(before['environment'], TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'))
    cfg = [v.replace(str(fallback.ROOT), str(ROOT)) for v in before['configure_argv']]
    row = dict(stage='prepared', source_commit=revision,
        controller_inputs=base.controls(revision, controller=CONTROLLER),
        native_sources={n: hashlib.sha256(raw).hexdigest() for n, raw in native.items()},
        composition=dict(selected_native_commit=SELECTED, fix_before=FIX_BEFORE, fix_commit=FIX,
            changed_native_files=[PROGRAM], independent_from_XeStrata=True),
        fallback=dict(native_source_commit=SELECTED, binaries=before['binaries'],
            consumer='Validated same-engine TTFT/throughput and zero-swap fallback until fixed-source qualification.'),
        **{k: before[k] for k in ('compiler', 'sdk', 'external_manifest', 'host_link_inputs',
            'packages', 'dependencies', 'native_validation_probe', 'boot', 'driver')},
        runtime_manifest=dict(path=str(ROOT / 'runtime-inputs.json'),
            sha256=digest(ROOT / 'runtime-inputs.json'), files=len(providers)),
        environment=env, configure_argv=cfg, targets=before['targets'],
        generated_expected={p.replace(str(fallback.ROOT), str(ROOT)): sha
            for p, sha in before['generated_expected'].items()},
        expected_compiler_inputs=before['compiled_inputs']['files'],
        commands=[['configure', cfg], ['compile', ['/usr/bin/cmake', '--build', str(ROOT / 'build'),
            '--target', *before['targets'], '-j', '4']],
            ['dependencies', ['/usr/bin/ninja', '-C', str(ROOT / 'build'), '-t', 'deps']]],
        budgets=dict(wall=1800, cpu=7200, rss_cap=12 << 30, text_cap=16 << 20,
            file_cap=512 << 20, total_cap=4 << 30),
        whole_wall_seconds=3600, commands_executed=[], active=False, ownership_closed=True,
        passed=False, GPU_executed=False, model_executed=False, inference_adopted=False,
        next='Small traced/validated submission and production wrapper checks;then bounded zero/positive-swap model lifecycle checks before long-generation timing. Preserve both prefill strategies and independent decode;no qualification inherited from compilation.')
    inputs(row); save(row)
    print('Prepared fixed-source nine-ELF generation;commit recipe before CPU build.', flush=True)


def build():
    row = json.loads((ROOT / 'record.json').read_text())
    require(base.committed(RESULT)[KEY] == base.compact(row) and row['stage'] == 'prepared' and
            not (ROOT / 'owned-build').exists(), 'one frozen fresh adaptive build')
    allowed = inputs(row); frozen = copy.deepcopy(row); owner = None
    deadline = time.monotonic() + row['whole_wall_seconds']

    def monitor(io, entry):
        next_size = 0.
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll(); require(time.monotonic() < deadline, 'whole adaptive build deadline')
            if time.monotonic() >= next_size:
                require(base.supervision.artifact_bytes(ROOT) <= row['budgets']['total_cap'],
                        'whole adaptive generation byte budget'); next_size = time.monotonic() + 1
            if select.select([io.proc.stdout], [], [], .02)[0]: io.read(); io.pending.clear()
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
        found = {str((Path(v.strip()) if Path(v.strip()).is_absolute() else ROOT / 'build' / v.strip()).resolve())
            for v in out.read_text().splitlines() if v.startswith('    ') and v.strip()}
        require(len(found) == row['expected_compiler_inputs'] and str(ROOT / 'source' / PROGRAM) in found and
                all(allowed.get(p) == digest(p) and '/MistVVK/' not in p for p in found),
                'unplanned or incomplete actual compiler inputs')
        write(ROOT / 'compiler-inputs.json', {p: allowed[p] for p in sorted(found)})
        db = ROOT / 'build/compile_commands.json'; commands = json.loads(db.read_text()); units = []
        for name in (PROGRAM, fallback.source.KERNEL, fallback.source.FIXTURE):
            matches = [v for v in commands if v['file'] == str(ROOT / 'source' / name)]
            if name == fallback.source.KERNEL:
                matches = [v for v in matches if '-o CMakeFiles/strata_kernels.dir/' in v['command']]
            require(len(matches) == 1, 'one actual production compile target required: ' + name)
            argv = shlex.split(matches[0]['command'])
            require(argv[0] == base.COMPILER and '-O3' in argv and '-fsycl' in argv and
                    not any(v in argv for v in ('-ffast-math', '-Ofast', '-ffp-contract=fast')),
                    'production compile options changed: ' + name)
            units.append(dict(source=name, argv_sha256=hashlib.sha256(json.dumps(argv).encode()).hexdigest()))
        row.update(stage='built-not-executed', passed=True,
            compiled_inputs=dict(path=str(ROOT / 'compiler-inputs.json'),
                sha256=digest(ROOT / 'compiler-inputs.json'), files=len(found), uncovered=0),
            actual_link_inputs=base.traced_link_inputs(allowed, ROOT),
            compile_database=dict(path=str(db), sha256=digest(db), units=units),
            binaries={n: dict(path=str(ROOT / 'build' / n), sha256=digest(ROOT / 'build' / n),
                bytes=(ROOT / 'build' / n).stat().st_size) for n in row['targets']})
    except BaseException as error:
        row.update(stage='failed', passed=False, error=repr(error))
    finally:
        row.update(active=bool(owner and owner.active), commands_executed=owner.commands if owner else [],
            ownership_closed=bool(owner and all(base.supervision.closed(v) for v in owner.commands)))
        require(owner is None or owner.active is None, 'build ownership unresolved'); save(row)
    print(json.dumps({k: row.get(k) for k in ('stage', 'passed', 'error')}), flush=True)
    require(row['passed'], 'adaptive model build failed')


def verify():
    row = json.loads((ROOT / 'record.json').read_text())
    require(base.committed(RESULT)[KEY] == base.compact(row) and row['passed'] and
            row['stage'] == 'built-not-executed' and row['ownership_closed'] and not row['active'] and
            all(base.supervision.closed(v) for v in row['commands_executed']), 'closed complete build required')
    allowed = inputs(row); asset = row['compiled_inputs']
    require(digest(asset['path']) == asset['sha256'], 'actual compiler map changed')
    found = json.loads(Path(asset['path']).read_text())
    require(len(found) == asset['files'] and all(allowed.get(p) == sha == digest(p) for p, sha in found.items()),
            'actual compiler inputs changed')
    require(digest(row['compile_database']['path']) == row['compile_database']['sha256'] and
            row['actual_link_inputs']['uncovered_external_inputs'] == 0 and all(allowed.get(p) == sha == digest(p)
            for p, sha in row['actual_link_inputs']['frozen_inputs'].items()), 'actual compile/link recipe changed')
    require(not any(Path(p).is_relative_to(fallback.ROOT) for p in
            (*found, *row['actual_link_inputs']['frozen_inputs'],
             *fallback.retained.runtime.inventory(row['external_manifest']))),
            'new generation still consumes fallback build products')
    require(set(row['binaries']) == set(row['targets']) and all(digest(v['path']) == v['sha256'] and
            Path(v['path']).stat().st_size == v['bytes'] for v in row['binaries'].values()), 'same-build binaries changed')
    return row


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'build', 'verify'), 'use prepare|build|verify')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); base.guards.no_model_owners([])
        if sys.argv[1] == 'prepare': prepare()
        elif sys.argv[1] == 'build': build()
        else: print(json.dumps(dict(binaries=len(verify()['binaries']), GPU_executed=False)), flush=True)
