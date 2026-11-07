#!/usr/bin/env python3
"""Main's source-pinned upstream-feature compilation; never launches the engine.

Reuse the qualified compiler/dependency recipe, without replacing either fast
fallback. One temporary candidate serves the current DFlash2/v3.1 qualification.
Prepared plans and closed outcomes live in the maintained upstream status.
"""
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import build_engine as source
import measurement_guards as guards
import measurement_owner as ownership

P, B = source.P, source.B
ROOT = B / 'upstream-features-146df3c4'
REFERENCE = B / 'adaptive-payload-audit-model-v2'
STATUS = P / 'bench/results/2026-10-15-pr6-recovery-audit/upstream-status.json'
TARGETS = ['strata', 'sycl_dflash2_test', 'dflash2_pack_test', 'sycl_submission_test']
PREFIXES = ['CMakeLists.txt', 'cmake', 'src', 'include', 'third_party', 'tests', 'bench/sycl',
            'tools/sycl/build_upstream_features.py', 'tools/sycl/build_engine.py',
            'tools/sycl/measurement_guards.py', 'tools/sycl/measurement_owner.py']
require = guards.require


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def committed_plan():
    return json.loads(source.git(P, 'show', 'HEAD:' + str(STATUS.relative_to(P))))['current_sync']['native_feature_build']


def check_inputs(row):
    for pin in [row['project'], *row['dependencies']]:
        root = Path(pin['root'])
        recovered = {n: hashlib.sha256(raw).hexdigest() for n, raw in
                     source.blobs(root, pin['revision'], pin['source_sha256'])}
        require(recovered == pin['source_sha256'] and
                all(source.digest(root / n) == h for n, h in recovered.items()), 'source input changed')
    for path, value in row['toolchain'].items():
        require(source.digest(path) == value, 'compiler/tool input changed: ' + path)
    external = row['external_inputs']
    require(source.digest(external['path']) == external['sha256'], 'external input map changed')
    for path, value in json.loads(Path(external['path']).read_text()).items():
        require(source.digest(path) == value, 'SDK/dependency input changed: ' + path)
    for value in row['fallbacks'].values():
        require(Path(value['path']).stat().st_size == value['bytes'] and
                source.digest(value['path']) == value['sha256'], 'preserve both fast fallbacks')


def compact(row):
    return dict(stage=row['stage'], source_commit=row['project']['revision'],
        project_source_sha256=row['project']['aggregate_sha256'],
        dependency_revisions={p['root']: p['revision'] for p in row['dependencies']},
        recipe=row['recipe'], toolchain=row['toolchain'], fallbacks=row['fallbacks'],
        external_inputs=row['external_inputs'],
        record=dict(path=str(ROOT / 'record.json'), sha256=source.digest(ROOT / 'record.json')),
        binaries=row.get('binaries', {}), passed=row.get('passed', False),
        ownership_closed=row.get('ownership_closed', True), error=row.get('error'),
        commands=[dict(label=c['label'], exit_code=c.get('exit_code'),
                       ownership_closed=ownership.closed(c), errors=c.get('errors', []), cleanup=c.get('cleanup', []))
                  for c in row.get('commands', [])],
        retention='Temporary third generation: preserve552b prefill and e1b011 decode. '
                  'This candidate is only for immediate native feature compilation and bounded DFlash2 qualification. '
                  'Retire it if qualification fails; at the feature comparison boundary select validated outputs '
                  'and retire superseded generations after original-source/recipe recovery. No adoption from build success.')


def publish(row):
    write(ROOT / 'record.json', row)
    status = json.loads(STATUS.read_text())
    status['current_sync']['native_feature_build'] = compact(row)
    write(STATUS, status)


def prepare():
    require(not ROOT.exists(), 'one fresh feature build; close/reconcile an existing attempt before replacement')
    status = json.loads(STATUS.read_text())
    require('native_feature_build' not in status['current_sync'], 'existing feature build must close first')
    baseline = json.loads((REFERENCE / 'record.json').read_text())
    recipe = copy.deepcopy(baseline['recipe'])
    recipe['configure'] = [v.replace(str(REFERENCE / 'source'), str(P)).replace(str(REFERENCE), str(ROOT))
                           for v in recipe['configure']]
    recipe['environment'] = {k: v.replace(str(REFERENCE), str(ROOT)) for k, v in recipe['environment'].items()}
    recipe['targets'] = TARGETS
    recipe.pop('retention', None)
    project = source.pin(P, PREFIXES)
    dependencies = []
    for prefix in ('-DSTRATA_GGML_DIR:PATH=', '-DSYCL_TLA_SOURCE_DIR='):
        paths = [v[len(prefix):] for v in recipe['configure'] if v.startswith(prefix)]
        require(len(paths) == 1, 'one exact dependency')
        dependencies.append(source.pin(Path(paths[0]), ['CMakeLists.txt', 'cmake', 'src', 'include']))
    toolchain = {p: source.digest(p) for p in ['/usr/bin/cmake', '/usr/bin/ninja',
        '/opt/intel/oneapi/compiler/2026.2/bin/icpx', '/opt/intel/oneapi/compiler/2026.2/bin/icx']}
    external = baseline['external_inputs']
    require(source.digest(external['path']) == external['sha256'], 'qualified external input map changed')
    fallbacks = {
        'prefill': dict(path=str(B / 'cache-map-model-v1/build/strata'), bytes=0,
            sha256='552b1616d21ad1b9aa6a317cb940d375c003246b896a255a935e09292804c96a'),
        'decode': dict(path=str(REFERENCE / 'build/strata'), bytes=24518560,
            sha256='e1b01116ccdc4adc3d59cbccb78c365ebbcc937dfa343463656263c1044e1eb2')}
    for value in fallbacks.values():
        value['bytes'] = Path(value['path']).stat().st_size
        require(source.digest(value['path']) == value['sha256'], 'fallback identity')
    row = dict(stage='prepared', project=project, dependencies=dependencies, recipe=recipe,
        toolchain=toolchain, external_inputs=external, fallbacks=fallbacks,
        reference_recipe_record_sha256=source.digest(REFERENCE / 'record.json'),
        budgets=dict(wall=2400, cpu=7200, rss_cap=12 << 30, total_cap=4 << 30,
                     text_cap=48 << 20, file_cap=512 << 20), commands=[], passed=False, ownership_closed=True,
        qualification='Compilation only, no GPU execution, no speed, no model-route or full-context qualification.')
    check_inputs(row)
    ROOT.mkdir(mode=0o700)
    for name in ['tmp', 'cache']: (ROOT / name).mkdir(mode=0o700)
    publish(row)
    print('Prepared feature compilation; commit upstream-status before build.', flush=True)


def build():
    row = json.loads((ROOT / 'record.json').read_text())
    frozen = compact(row)
    require(row['stage'] == 'prepared' and committed_plan() == frozen, 'commit the prepared build first')
    require(json.loads(STATUS.read_text())['current_sync']['native_feature_build'] == frozen, 'pending plan edits')
    check_inputs(row); guards.no_model_owners([])
    monitor = None
    def persist():
        snapshot = dict(row, active=bool(monitor and monitor.active),
                        commands=monitor.commands if monitor else row['commands'])
        write(ROOT / 'active.json', snapshot)
    try:
        monitor = ownership.Owner(ROOT / 'owned', persist)
        row['stage'] = 'building'
        for label, argv in [
            ('configure', row['recipe']['configure']),
            ('compile', ['/usr/bin/cmake', '--build', str(ROOT / 'build'), '--target', *TARGETS, '-j', '4'])]:
            require(committed_plan() == frozen, 'frozen build plan changed')
            check_inputs(row); guards.no_model_owners([])
            entry, stdout, stderr = monitor.run(label, argv, row['recipe']['environment'], P, **row['budgets'])
            row['commands'] = monitor.commands
            print(json.dumps(dict(label=label, exit_code=entry.get('exit_code'),
                                  closed=ownership.closed(entry))), flush=True)
            require(ownership.closed(entry) and entry['exit_code'] == 0,
                    label + ' failed; read bounded diagnostics in the owned output')
        check_inputs(row)
        row['binaries'] = {n: dict(path=str(ROOT / 'build' / n), sha256=source.digest(ROOT / 'build' / n),
                                  bytes=(ROOT / 'build' / n).stat().st_size) for n in TARGETS}
        row.update(stage='built', passed=True)
    except BaseException as error:
        row.update(stage='build_failed', error=repr(error))
    finally:
        row['commands'] = monitor.commands if monitor else []
        row['ownership_closed'] = not monitor or monitor.active is None and all(ownership.closed(c) for c in monitor.commands)
        publish(row)
    require(row['passed'] and row['ownership_closed'], row.get('error', 'build did not close'))
    print('Feature build complete; commit its exact binary identities before GPU checks.', flush=True)


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ['prepare', 'build'], 'use prepare or build')
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (prepare if sys.argv[1] == 'prepare' else build)()
