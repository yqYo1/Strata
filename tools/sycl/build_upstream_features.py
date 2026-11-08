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
PLAN_KEY = 'native_feature_build'
REFERENCE = B / 'adaptive-payload-audit-model-v2'
STATUS = P / 'bench/results/2026-10-15-pr6-recovery-audit/upstream-status.json'
TARGETS = ['strata', 'sycl_dflash2_test', 'sycl_dflash2_ring_test', 'sycl_dflash2_mask_test', 'dflash2_pack_test', 'sycl_submission_test']
PREFIXES = ['CMakeLists.txt', 'cmake', 'src', 'include', 'third_party', 'tests', 'bench/sycl',
            'tools/sycl/build_upstream_features.py', 'tools/sycl/build_engine.py',
            'tools/sycl/measurement_guards.py', 'tools/sycl/measurement_owner.py']
require = guards.require


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def committed_plan():
    return json.loads(source.git(P, 'show', 'HEAD:' + str(STATUS.relative_to(P))))['current_sync'][PLAN_KEY]


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
    baseline = row.get('baseline_feature_build')
    if baseline:
        require(source.digest(baseline['path']) == baseline['sha256'], 'original feature build receipt changed')


def compact(row):
    result = dict(stage=row['stage'], source_commit=row['project']['revision'],
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
        retention=row.get('retention', 'Temporary third generation: preserve552b prefill and e1b011 decode. '
                  'This candidate is only for immediate native feature compilation and bounded DFlash2 qualification. '
                  'Retire it if qualification fails; at the feature comparison boundary select validated outputs '
                  'and retire superseded generations after original-source/recipe recovery. No adoption from build success.'))
    for key in ['attempt', 'previous_attempt', 'unchanged_outputs', 'host_policy_gate', 'baseline_feature_build']:
        if key in row: result[key] = row[key]
    return result


def publish(row):
    write(ROOT / 'record.json', row)
    status = json.loads(STATUS.read_text())
    status['current_sync'][PLAN_KEY] = compact(row)
    write(STATUS, status)


def prepare():
    previous = None
    status = json.loads(STATUS.read_text())
    loan = PLAN_KEY == 'dflash2_cache_loan_native_build'
    require(not loan or not ROOT.exists(), 'one fresh isolated cache-loan build only')
    if ROOT.exists():
        previous = json.loads((ROOT / 'record.json').read_text())
        failed_compile = previous['stage'] == 'build_failed' and not previous['passed']
        gate = status['current_sync'].get('dflash2_device_check', {})
        ring = status['current_sync'].get('dflash2_ring_check', {})
        model = status['current_sync'].get('dflash2_model_check', {})
        public = json.loads(source.git(P, 'show', 'HEAD:' + str(STATUS.relative_to(P))))['current_sync']
        for key in ['dflash2_device_check', 'dflash2_ring_check', 'dflash2_mask_check',
                    'dflash2_model_check', 'dflash2_native_port']:
            require(status['current_sync'].get(key) == public.get(key), 'commit component evidence before build reuse')
        superseded_component = (previous['stage'] == 'built' and previous['passed'] and (
            (ring.get('stage') == 'full_passed' and ring.get('passed') and ring.get('ownership_closed') and
            ring.get('native_build', {}).get('source_commit') == previous['project']['revision'] and
            ring.get('native_build', {}).get('record') == compact(previous)['record']) or
            (gate.get('stage') == 'passed' and gate.get('passed') and gate.get('ownership_closed') and
            gate.get('numerical_cases_passed') == 16 and
            gate.get('native_build', {}).get('source_commit') == previous['project']['revision'] and
            gate.get('fixture') == previous['binaries']['sycl_dflash2_test'])) and
            status['current_sync']['dflash2_native_port']['state'] ==
                'compiled_small_q4_fp32_bf16_prompt_gate_passed_model_route_unqualified')
        failed_model = (previous['stage'] == 'built' and previous['passed'] and
            model.get('stage') == 'failed' and not model.get('passed') and model.get('ownership_closed') and
            len(model.get('commands', [])) == 1 and model['commands'][0]['exit_code'] == -6 and
            model['commands'][0]['ownership_closed'] and not model.get('fresh_faults') and
            model.get('native_build', {}).get('record') == compact(previous)['record'] and
            status['current_sync']['dflash2_native_port']['state'] == 'model_runtime_failed_host_draft_result_extent')
        bounded_model = (previous['stage'] == 'built' and previous['passed'] and
            model.get('stage') == 'passed' and model.get('passed') and model.get('ownership_closed') and
            len(model.get('commands', [])) == 1 and model['commands'][0]['exit_code'] == 0 and
            model['commands'][0]['ownership_closed'] and not model.get('fresh_faults') and
            model.get('native_build', {}).get('record') == compact(previous)['record'] and
            not model.get('speed_eligible', True) and not model.get('model_adoption_qualified', True) and
            status['current_sync']['dflash2_native_port']['state'] ==
                'bounded_target_model_passed_long_phase_and_speed_unqualified')
        head_plan = status['current_sync'].get('dflash2_prefill_heads', {})
        reviewed_unexecuted_heads = (previous['stage'] == 'built' and previous['passed'] and
            head_plan.get('stage') == 'compiled_review_corrections_pending' and
            head_plan.get('compiled_candidate_record') == compact(previous)['record'] and
            head_plan.get('native_model_launched') is False and 'dflash2_prefill_heads_check' not in public and
            head_plan.get('source_review', {}).get('sha256') ==
                '89971fad6d52c32f5ff15c2fc9cdce59a0548a4eef1fb61ac7e29037f726789b')
        require(head_plan == public.get('dflash2_prefill_heads', {}), 'commit head candidate disposition before reuse')
        replace_candidate = superseded_component or failed_model or bounded_model or reviewed_unexecuted_heads
        require((failed_compile or replace_candidate) and previous['ownership_closed'] and
                all(ownership.closed(c) for c in previous['commands']) and
                committed_plan() == compact(previous), 'commit and close failed compilation before a new source attempt')
        if replace_candidate:
            # This candidate has no qualified model route or adoption. Recover
            # its original inputs before replacing its disposable output path;
            # both validated phase engines remain hash-checked fallbacks.
            for pin in [previous['project'], *previous['dependencies']]:
                recovered = {n: hashlib.sha256(raw).hexdigest() for n, raw in
                    source.blobs(Path(pin['root']), pin['revision'], pin['source_sha256'])}
                require(recovered == pin['source_sha256'] and signature(recovered) == pin['aggregate_sha256'],
                        'original candidate inputs not recoverable')
            source.git(P, 'merge-base', '--is-ancestor', previous['project']['revision'], 'origin/feature/sycl')
            for value in previous['binaries'].values():
                require(source.digest(value['path']) == value['sha256'], 'original candidate executable changed')
        require(source.git(P, 'rev-parse', 'HEAD').decode().strip() != previous['project']['revision'], 'new committed source required')
    else:
        require(PLAN_KEY not in status['current_sync'], 'existing feature build must close first')
    baseline = json.loads((REFERENCE / 'record.json').read_text())
    recipe = copy.deepcopy(baseline['recipe'])
    recipe['configure'] = [v.replace(str(REFERENCE / 'source'), str(P)).replace(str(REFERENCE), str(ROOT))
                           for v in recipe['configure']]
    recipe['environment'] = {k: v.replace(str(REFERENCE), str(ROOT)) for k, v in recipe['environment'].items()}
    recipe['targets'] = TARGETS
    recipe['build_targets'] = TARGETS if loan else ['strata']
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
    if loan:
        result_path = P / 'bench/results/2026-10-15-pr6-recovery-audit/fused-moe-service-current.json'
        raw = source.git(P, 'show', 'HEAD:' + str(result_path.relative_to(P)))
        require(raw == result_path.read_bytes(), 'commit host result before native build')
        host = json.loads(raw)['dflash2_cache_loan_heads_host_v1']
        require(host['passed'] and host['complete'] and host['ownership_closed'] and
                not host['active'] and not host['GPU_executed'] and
                host['result'] == dict(kind='prefill_head_scope', cases=38, checks=854, status='PASS'),
                'exact closed cache-loan host policy required')
        row['host_policy_gate'] = dict(evidence_commit=source.git(P, 'rev-parse', 'HEAD').decode().strip(),
            key='dflash2_cache_loan_heads_host_v1', source_commit=host['source_commit'], result=host['result'])
        host_names = ['tests/sycl/prefill_head_scope.cpp', 'include/strata/core/prefill_head_scope.hpp',
                      'include/strata/core/cache_scratch_lease.hpp']
        require(all(source.git(P, 'show', host['source_commit'] + ':' + name) == (P / name).read_bytes()
                    for name in host_names), 'native host policy changed after its passing check')
        row['host_policy_gate']['native_inputs'] = {name: source.digest(P / name) for name in host_names}
        reference_path = B / 'upstream-features-146df3c4/record.json'
        reference = json.loads(reference_path.read_text())
        require(status['current_sync']['native_feature_build']['record'] ==
                dict(path=str(reference_path), sha256=source.digest(reference_path)),
                'original feature build must match its committed qualification record')
        require(reference['passed'] and reference['ownership_closed'] and reference['stage'] == 'built',
                'preserve closed f633 feature engine and its same-build fixtures')
        for pin in [reference['project'], *reference['dependencies']]:
            recovered = {n: hashlib.sha256(raw).hexdigest() for n, raw in
                source.blobs(Path(pin['root']), pin['revision'], pin['source_sha256'])}
            require(recovered == pin['source_sha256'], 'original f633 source must remain recoverable')
        row['baseline_feature_build'] = dict(path=str(reference_path), sha256=source.digest(reference_path),
            source_commit=reference['project']['revision'], binaries=reference['binaries'])
        row['fallbacks'].update({'feature_' + name: value for name, value in reference['binaries'].items()})
        row['retention'] = ('One isolated candidate for the imminent borrowed-vs-owned DFlash2 gate. '
            'Preserve f633 and all5fixtures until same-build small/model admission selects a replacement;'
            'then retire the superseded feature engine,or retire this candidate on failure,after exact source/recipe recovery. '
            'Keep552b/e1b phase references;this finite transition temporarily has four engine generations,not a new quota. '
            'Legacy56ELF provenance holds remain separate and unresolved;review their current consumers at closure. '
            'No speed,device/model qualification or adoption follows from compilation.')
    row['attempt'] = previous.get('attempt', 1) + 1 if previous else 1
    if previous:
        if previous.get('unchanged_outputs'):
            row['unchanged_outputs'] = {n: v for n, v in previous['unchanged_outputs'].items()
                                        if n not in recipe['build_targets']}
        row['previous_attempt'] = dict(source_commit=previous['project']['revision'], stage=previous['stage'],
            record_sha256=source.digest(ROOT / 'record.json'), error=previous.get('error'),
            ownership_closed=True, evidence_commit=source.git(P, 'rev-parse', 'HEAD').decode().strip(),
            reason='Reuse the closed unqualified candidate cache; exact source/recipe/outcome remain in Git. Preserve both validated phase references.')
        if replace_candidate:
            row['unchanged_outputs'] = {name: dict(value, origin_source_commit=
                value.get('origin_source_commit', previous['project']['revision']))
                for name, value in previous['binaries'].items() if name not in recipe['build_targets']}
            row['previous_attempt']['retirement'] = dict(
                qualification='Closed bounded candidate only;prior model outcome remains in its original committed record.No adopted long/model-speed route.',
                original_source_recovered=True, source_sha256=previous['project']['aggregate_sha256'],
                source_files=previous['project']['count'], original_binaries_verified=previous['binaries'],
                consumer='Replace the unadopted feature candidate with optional-MTP prefill-head ownership;rebuild engine only,preserve all5original component outputs.',
                point='Same new binary supplies release0/1 arms after host/admission gates;keep552b/e1b011.No fourth generation or f50baseline copy.')
    check_inputs(row)
    ROOT.mkdir(mode=0o700, exist_ok=bool(previous))
    for name in ['tmp', 'cache']: (ROOT / name).mkdir(mode=0o700, exist_ok=bool(previous))
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
        output = 'owned' if row.get('attempt', 1) == 1 else 'owned-' + str(row['attempt'])
        monitor = ownership.Owner(ROOT / output, persist)
        row['stage'] = 'building'
        for label, argv in [
            ('configure', row['recipe']['configure']),
            ('compile', ['/usr/bin/cmake', '--build', str(ROOT / 'build'), '--target', *row['recipe']['build_targets'], '-j', '4'])]:
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
        for name, binary in row['binaries'].items():
            prior = row.get('unchanged_outputs', {}).get(name)
            if prior:
                require(binary['sha256'] == prior['sha256'] and binary['bytes'] == prior['bytes'],
                        'preserved native output changed: ' + name)
            binary['origin_source_commit'] = (prior.get('origin_source_commit',
                row['previous_attempt']['source_commit']) if prior else row['project']['revision'])
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
    require(len(sys.argv) == 2 and sys.argv[1] in ['prepare', 'build', 'prepare-loan', 'build-loan'],
            'use prepare|build or prepare-loan|build-loan')
    if sys.argv[1].endswith('-loan'):
        ROOT, PLAN_KEY = B / 'dflash2-cache-loan-native-v1', 'dflash2_cache_loan_native_build'
        sys.argv[1] = sys.argv[1].removesuffix('-loan')
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (prepare if sys.argv[1] == 'prepare' else build)()
