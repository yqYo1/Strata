#!/usr/bin/env python3
"""Build and check one observer-only comparison; Main owns serialization.

Repository LICENSE; independently authored. Preserve selected552b and its eight
fixtures. Compilation/host parity never imply model attribution or a speed gain.
"""
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import re
import select
import shlex
import sys
import time

sys.dont_write_bytecode = True
import cpu_service_sources as source

selected, P, B = source.selected, source.P, source.B
base = selected.base
require, digest, write = base.require, base.digest, base.write
ROOT = B / 'cpu-service-cache-map-v1'
RESULT = selected.RESULT
KEY = 'cache_map_cpu_service_build_v1'
CONTROLLER = Path(__file__).resolve()


def compact(row):
    out = base.compact(row)
    out.pop('commands', None)
    recipe = out.pop('recipe')
    out['recipe'] = dict(helper='tools/sycl/cpu_service_sources.py:recipe',
        sha256=hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest(),
        selected_native=source.NATIVE, observer_build_option='STRATA_CPU_SERVICE_PROFILE=ON',
        targets=recipe['targets'], host_modes=recipe['host_modes'])
    if 'host_result' in out:
        out['host_result'] = {key: value for key, value in out['host_result'].items() if key != 'commands'}
    return out


def save(row):
    write(ROOT / 'record.json', row)
    data = json.loads(RESULT.read_text())
    data[KEY] = compact(row)
    write(RESULT, data)


def selected_basis():
    native, gate, providers = selected.proof()
    require(native['source_commit'] == source.NATIVE and
            native['binaries']['strata']['sha256'] == source.ENGINE,
            'preserve the named validated engine generation')
    return native, dict(root=str(selected.model.ROOT),
        native_source_commit=native['source_commit'],
        record_sha256=digest(selected.model.ROOT / 'record.json'),
        binaries=native['binaries'], runtime_manifest=native['runtime_manifest'],
        validation_provider=native['native_validation_probe'],
        unchanged_fixture_basis=gate['mathematical_fixture_basis'])


def retention_gate():
    """Do not create another generation from an unreconciled binary inventory."""
    review = base.committed(RESULT).get('retained_builds_review_20261103', {})
    require(review.get('stage') == 'reconciled' and review.get('passed') is True and
            review.get('selected_root') == str(selected.model.ROOT) and
            review.get('next_comparison_root') == str(ROOT) and
            review.get('original_inputs_recovered_or_failure_recorded') is True,
            'finish the bounded existing-build recovery/retirement review before compiling')
    manifest = review['retirement_manifest']
    require(digest(manifest['path']) == manifest['sha256'], 'retirement evidence changed')
    retired = json.loads(Path(manifest['path']).read_text())['files']
    require(retired and all(not Path(item['path']).exists() for item in retired),
            'recorded build retirement is not physically complete')
    retained = review['retained_project_binaries']
    require(retained and all(type(item.get('consumer')) is str and item['consumer'] and
            Path(item['path']).is_file() and digest(item['path']) == item['sha256']
            for item in retained), 'named retained binary consumers changed')
    require({asset['path'] for asset in selected_basis()[0]['binaries'].values()} <=
            {item['path'] for item in retained}, 'selected same-build fixtures are not protected')
    return review


def inputs(row):
    revision = base.basis.build.git(P, 'rev-parse', 'HEAD').decode().strip()
    require(base.controls(row['source_commit'], controller=CONTROLLER) == row['controller_inputs'],
            'commit and freeze the exact observer build controller')
    native, proof = selected_basis()
    require(proof == row['selected_basis'], 'selected qualification inputs changed')
    _, values = source.projection()
    require(source.aggregate(values)[0] == row['native_sources'] and
            source.recipe(native, ROOT) == row['recipe'], 'observer source/recipe changed')
    allowed = selected.model.inputs(native, reuse_revision=revision,
                                    controller_changes=selected.NATIVE_CONTROLS)
    # The extra build must not silently compile or link from selected build outputs.
    allowed = {path: sha for path, sha in allowed.items()
               if not Path(path).is_relative_to(selected.model.ROOT)}
    for name, raw in values.items():
        path = ROOT / 'source' / name
        require(path.is_file() and not path.is_symlink() and path.read_bytes() == raw,
                'frozen observer input changed: ' + name)
        allowed[str(path.resolve())] = row['native_sources'][name]
    allowed.update(row['generated_expected'])
    return allowed


def prepare():
    require(not ROOT.exists() and KEY not in base.committed(RESULT),
            'one fresh observer generation;inspect an existing owner before any retry')
    native, proof = selected_basis()
    _, values = source.projection()
    revision = base.basis.build.git(P, 'rev-parse', 'HEAD').decode().strip()
    recipe = source.recipe(native, ROOT)
    old = str(selected.model.ROOT)
    row = dict(stage='prepared', source_commit=revision,
        controller_inputs=base.controls(revision, controller=CONTROLLER),
        native_sources=source.aggregate(values)[0], selected_basis=proof, recipe=recipe,
        generated_expected={path.replace(old, str(ROOT)): sha
                            for path, sha in native['generated_expected'].items()},
        commands=[['configure', recipe['configure']],
            ['compile', ['/usr/bin/cmake', '--build', str(ROOT / 'build'),
                         '--target', *recipe['targets'], '-j', '4']],
            ['dependencies', ['/usr/bin/ninja', '-C', str(ROOT / 'build'), '-t', 'deps']]],
        budgets=dict(wall=1800, cpu=7200, rss_cap=12 << 30, text_cap=8 << 20,
                     file_cap=512 << 20, total_cap=2 << 30), whole_wall_seconds=3600,
        host_plan=dict(modes=[0, 1], wall_each=300, text_cap_each=1 << 20,
            expected_cases=28, require_identical_checksum=True,
            scope='Five real pool workers and a dedicated dispatch host;seven formats,NT1..4,96-expert boundary,canaries,phase isolation and counter/owner conservation. Same-leaf parity,not an independent mathematical oracle.'),
        retention='Retire this extra generation after its named observer comparison closes, '
            'recovering exact original inputs first. Selected552b/eight fixtures remain the fallback. '
            'Prepare may precede retention reconciliation;compile may not.',
        commands_executed=[], active=False, ownership_closed=True, passed=False,
        complete=False, GPU_executed=False, model_executed=False,
        observer_overhead_qualified=False, inference_adopted=False)
    ROOT.mkdir(mode=0o700)
    for folder in ('tmp', 'cache', 'source'):
        (ROOT / folder).mkdir(mode=0o700)
    for name, raw in values.items():
        path = ROOT / 'source' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    write(ROOT / 'record.json', row)
    try:
        inputs(row)
    except BaseException as error:
        row.update(stage='failed-admission', error=repr(error))
    save(row)
    require(row['stage'] == 'prepared', 'observer source admission failed;inspect the preserved result')
    print('Prepared observer-only source/recipe;commit before compilation. No build/GPU execution.', flush=True)


def monitor(deadline, io, entry):
    next_size = 0.
    while not io.exit_watch.poll(0) or not io.eof:
        io.poll()
        require(time.monotonic() < deadline, 'observer build deadline')
        if time.monotonic() >= next_size:
            require(base.supervision.artifact_bytes(ROOT) <= 2 << 30, 'observer generation byte budget')
            next_size = time.monotonic() + 1
        if select.select([io.proc.stdout], [], [], .02)[0]:
            io.read()
            io.pending.clear()
    io.exit(1)


def build():
    row = json.loads((ROOT / 'record.json').read_text())
    require(base.committed(RESULT)[KEY] == compact(row) and row['stage'] == 'prepared' and
            not (ROOT / 'owned-build').exists(), 'one frozen fresh observer build')
    retention_gate()
    allowed = inputs(row)
    frozen, owner = copy.deepcopy(row), None
    deadline = time.monotonic() + row['whole_wall_seconds']
    def persist():
        # Live ownership belongs in the external job receipt. Keep the committed
        # qualification file frozen until final input verification has finished.
        row.update(active=bool(owner and owner.active), commands_executed=owner.commands if owner else [],
            ownership_closed=bool(owner and all(base.supervision.closed(item) for item in owner.commands)))
        write(ROOT / 'record.json', row)
    try:
        owner = base.supervision.Owner(ROOT / 'owned-build', persist=persist)
        for label, argv in row['commands']:
            print('CPU build:', label, flush=True)
            entry, out, err = owner.run(label, argv, row['recipe']['environment'], ROOT,
                interaction=lambda io, entry: monitor(deadline, io, entry),
                **dict(row['budgets'], wall=min(row['budgets']['wall'], deadline-time.monotonic())))
            require(base.supervision.closed(entry) and entry['exit_code'] == 0,
                    label + ' failed: ' + (out.read_text(errors='replace') +
                                           err.read_text(errors='replace'))[-4000:])
        inputs(frozen)
        paths = {str((Path(line.strip()) if Path(line.strip()).is_absolute() else
                       ROOT / 'build' / line.strip()).resolve())
                 for line in out.read_text().splitlines() if line.startswith('    ') and line.strip()}
        required = {ROOT / 'source' / name for name in
                    (source.PROGRAM, source.POOL[1], source.HEADER, source.FIXTURES[1])}
        require(required <= {Path(path) for path in paths} and len(paths) > 600 and
                all(allowed.get(path) == digest(path) for path in paths),
                'incomplete or unplanned actual observer compiler inputs')
        write(ROOT / 'compiler-inputs.json', {path: digest(path) for path in sorted(paths)})
        database = ROOT / 'build/compile_commands.json'
        units = json.loads(database.read_text())
        for name in (source.PROGRAM, source.POOL[1], source.FIXTURES[1]):
            matches = [unit for unit in units if unit['file'] == str(ROOT / 'source' / name)]
            require(len(matches) == 1, 'one actual production compile unit required: ' + name)
            argv = shlex.split(matches[0]['command'])
            require(argv[0] == base.COMPILER and '-O3' in argv and
                    '-DSTRATA_CPU_SERVICE_PROFILE=1' in argv and
                    not any(flag in argv for flag in ('-ffast-math', '-Ofast', '-ffp-contract=fast')),
                    'unplanned observer production compile flags: ' + name)
        row.update(stage='built-not-executed', passed=True, complete=True,
            compiled_inputs=dict(path=str(ROOT / 'compiler-inputs.json'),
                sha256=digest(ROOT / 'compiler-inputs.json'), files=len(paths), uncovered=0),
            compile_database=dict(path=str(database), sha256=digest(database)),
            actual_link_inputs=base.traced_link_inputs(allowed, ROOT),
            binaries={name: dict(path=str(ROOT / 'build' / name), sha256=digest(ROOT / 'build' / name),
                bytes=(ROOT / 'build' / name).stat().st_size) for name in row['recipe']['targets']})
    except BaseException as error:
        row.update(stage='failed', passed=False, complete=False, error=repr(error))
    finally:
        row.update(active=bool(owner and owner.active), commands_executed=owner.commands if owner else [],
            ownership_closed=bool(owner and all(base.supervision.closed(item) for item in owner.commands)))
        save(row)
    require(row['ownership_closed'] and row['passed'], 'observer compilation failed;inspect existing owner/results')
    print('Observer build closed;no model/device/overhead qualification.', flush=True)


def verify():
    row = json.loads((ROOT / 'record.json').read_text())
    require(base.committed(RESULT)[KEY] == compact(row) and row['passed'] and row['complete'] and
            row['stage'] == 'built-not-executed' and row['ownership_closed'] and not row['active'] and
            all(base.supervision.closed(item) for item in row['commands_executed']),
            'commit the closed observer build before execution')
    allowed = inputs(row)
    asset = row['compiled_inputs']
    require(digest(asset['path']) == asset['sha256'], 'observer compiler inventory changed')
    found = json.loads(Path(asset['path']).read_text())
    require(len(found) == asset['files'] and all(allowed.get(path) == sha == digest(path)
            for path, sha in found.items()), 'actual observer compiler inputs changed')
    require(digest(row['compile_database']['path']) == row['compile_database']['sha256'] and
            all(allowed.get(path) == sha == digest(path)
                for path, sha in row['actual_link_inputs']['frozen_inputs'].items()) and
            all(digest(asset['path']) == asset['sha256'] for asset in row['binaries'].values()),
            'observer executable/link/recipe changed')
    return row


def host():
    row = verify()
    require('host_result' not in row and not (ROOT / 'owned-host').exists(),
            'one same-build host parity check;never retry a live/failed owner')
    owner, result = None, dict(passed=False, complete=False, cases=[], GPU_executed=False)
    def persist():
        result.update(commands=owner.commands if owner else [], active=bool(owner and owner.active),
            ownership_closed=bool(owner and all(base.supervision.closed(item) for item in owner.commands)))
        write(ROOT / 'record.json', dict(row, host_result=result))
    try:
        owner = base.supervision.Owner(ROOT / 'owned-host', persist=persist)
        for mode in row['host_plan']['modes']:
            env = dict(row['recipe']['environment'], STRATA_CPU_SERVICE=str(mode))
            argv = [row['binaries']['cpu_service_pool_test']['path'], str(mode)]
            entry, out, err = owner.run('pool-' + str(mode), argv, env, ROOT,
                wall=300, cpu=1200, rss_cap=2 << 30, text_cap=1 << 20,
                file_cap=1 << 20, total_cap=4 << 20)
            require(base.supervision.closed(entry) and entry['exit_code'] == 0,
                    'same-build pool check failed: ' + err.read_text(errors='replace')[-3000:])
            match = re.fullmatch(r'cpu_service_pool PASS observer=' + str(mode) +
                r' formats=7 nt=1\.\.4 cases=28 checksum_fnv1a64_le=([0-9a-f]{16})\n', out.read_text())
            require(match is not None, 'incomplete or unexpected real-pool parity footer')
            result['cases'].append(dict(mode=mode, exit_code=0, cases=28, checksum=match.group(1)))
        require(result['cases'][0]['checksum'] == result['cases'][1]['checksum'],
                'observer changed real-pool output bits')
        result.update(passed=True, complete=True)
    except BaseException as error:
        result['error'] = repr(error)
    finally:
        result.update(commands=owner.commands if owner else [], active=bool(owner and owner.active),
            ownership_closed=bool(owner and all(base.supervision.closed(item) for item in owner.commands)))
        row['host_result'] = result
        save(row)
    require(result['passed'] and result['ownership_closed'], 'observer host parity failed')
    print('Both observer modes pass28same-build pool cases with identical output;no GPU/model execution.', flush=True)


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'build', 'host', 'verify'),
            'usage: build_cpu_service_model.py prepare|build|host|verify')
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        {'prepare': prepare, 'build': build, 'host': host, 'verify': verify}[sys.argv[1]]()
