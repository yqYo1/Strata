#!/usr/bin/env python3
"""Fresh native qualification of the coherent full build; Main serializes work.

Main-authored using this fork's owned gates and scalar GU oracle. Repository
LICENSE applies. Historical partial-relink results grant no qualification here.
"""
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
import build_coherent_prefill as build
import check_moe_census_model as previous
import check_gu_compact as gu

P, B, RESULT = build.P, build.B, build.RESULT
ROOT, KEY = B/'coherent-prefill-device-v1', 'coherent_prefill_device_v1'
THIS = 'tools/sycl/check_coherent_prefill.py'
STAGES = (*previous.STAGES, 'gu-smoke', 'gu-numerics')
original = previous.original
require, sha = build.require, build.sha


def native(expected=None, *, observers=None):
    observers = {THIS} if observers is None else set(observers)
    require(THIS in observers, 'declare this qualification observer')
    row = json.loads((build.ROOT/'record.json').read_text())
    require(original.validation.committed(RESULT, (build.KEY,)) == build.context().compact(row)
        and row['stage'] == 'built' and row['passed'] and row['complete']
        and row['ownership_closed'] and not row['active'] and not row['GPU_executed'],
        'committed closed coherent build required')
    before = row['project']['source_sha256']
    after = original.common.source.pin(P, original.common.PREFIXES)['source_sha256']
    recovered = {p: hashlib.sha256(raw).hexdigest() for p, raw in
        original.common.source.blobs(P, row['project']['revision'], before)}
    require(recovered == before and set(after)-set(before) == observers
        and not set(before)-set(after) and all(after[p] == h for p, h in before.items()),
        'only explicitly named committed observers may extend the native source snapshot')
    actual = row['actual_inputs']
    require(row['native_source_commit'] == row['project']['revision'] == actual['source_commit']
        and actual['translation_units'] == 134 and actual['uncovered'] == 0
        and len(row['planned_compilation']) == 137
        and row['duplicate_source_admission'] == build.DUPLICATES
        and set(row['binaries']) == set(build.TARGETS), 'exact twelve-target full-build census required')
    coherence = actual['class_definition_coherence']
    require(coherence['header_sha256'] == sha(build.HEADER)
        and coherence['includer_count'] >= 3 and coherence['same_SYCL_class_condition']
        and coherence['all_objects_from_fresh_build'] and not coherence['partial_archive_reuse']
        and build.prerequisite() == row['build_prerequisite'], 'fresh consistent Gemm definitions required')
    for dep in row['dependencies']:
        recovered = {p: hashlib.sha256(raw).hexdigest() for p, raw in
            original.common.source.blobs(Path(dep['root']), dep['revision'], dep['source_sha256'])}
        require(recovered == dep['source_sha256'] and
            all(sha(Path(dep['root'])/p) == h for p, h in recovered.items()), 'dependency source changed')
    require(sha(build.SEED) == row['recipe_seed']['sha256']
        and sha(row['external_inputs']['path']) == row['external_inputs']['sha256']
        and all(sha(p) == h for p, h in row['toolchain'].items())
        and all(sha(p) == h for p, h in json.loads(Path(row['external_inputs']['path']).read_text()).items()),
        'compiler,SDK or original recipe changed')
    seed = json.loads(build.SEED.read_text())
    recipe = copy.deepcopy(seed['recipe']); old = str(build.SEED.parent)
    recipe['configure'] = [str(build.ROOT)+v[len(old):] if v.startswith(old) else v
        for v in recipe['configure']]
    recipe['environment'] = {k: v.replace(old, str(build.ROOT)) for k, v in recipe['environment'].items()}
    recipe['targets'] = recipe['build_targets'] = build.TARGETS
    require(recipe == row['recipe'], 'coherent build must retain its exact original recipe')
    python_seed = json.loads(original.candidate.SEED.read_text())
    require(original.common.HOST_PYTHON == python_seed['host_python']
        and sha(original.common.HOST_PYTHON) == python_seed['host_python_sha256'], 'host interpreter changed')
    for item in (*row['preserved_binaries'], *row['binaries'].values()):
        require(Path(item['path']).stat().st_size == item['bytes'] and
            sha(item['path']) == item['sha256'], 'same-build fixture or phase reference changed')
    require(original.candidate.probe_qualification() ==
        (row['native_validation_probe'], row['validation_qualification']), 'runtime probe changed')
    require(all(sha(p) == h for p, h in (row['configured_files'] | row['generated_inputs']).items())
        and sha(actual['path']) == actual['sha256'], 'generated or compiler census input changed')
    census = json.loads(Path(actual['path']).read_text())
    require(all(sha(p) == h for family in ('compiled', 'generated', 'links')
        for p, h in census[family].items()), 'actual compiler/header/link input changed')
    proof = dict(native_source_commit=row['native_source_commit'], original_source_files=len(before),
        observer_source_files=len(after), observer_source_sha256=original.common.signature(after),
        added_observers={p: after[p] for p in sorted(observers)}, exact_twelve_ELFs_census_recipe=True,
        class_coherence=coherence, exact_seed_recipe_sha256=hashlib.sha256(
            json.dumps(recipe, sort_keys=True).encode()).hexdigest(),
        native_recompiled=False, GPU_result_inherited=False)
    require(expected is None or proof == expected, 'native observer admission changed')
    return dict(row, observer_admission=proof)


def context():
    driver = previous.context()
    driver.ROOT, driver.KEY, driver.THIS = ROOT, KEY, THIS
    driver.build, driver.native, driver.STAGES = build, native, STAGES
    original_host = driver.host
    def host(device, ctx, row, scope, outcome):
        env = dict(row['environment']); env.pop('LD_PRELOAD', None)
        binary = row['binaries']['full_context_diagnostic_host_test']
        event, out, err = scope.run('physical-context-host', [binary['path']], env, P, **row['host_budget'])
        text = out.read_text()
        match = re.fullmatch(r'cases=([1-9]\d*) full_context_request_and_terminal_bounds=PASS\n'
            r'bounds_cases=\1 kv_hash_cases=([1-9]\d*) byte_order_and_bounds=PASS full_extent_spans=1 GPU=UNRUN\n', text)
        require(original.owner.closed(event) and event['exit_code'] == 0 and match
            and not err.read_text(), 'full-context admission/hash-order host fixture failed')
        original_host(device, ctx, row, scope, outcome)
        outcome['physical_context_host'] = dict(passed=True, GPU_executed=False,
            bounds_cases=int(match[1]), hash_cases=int(match[2]),
            actual_model_physical_fill_qualified=False)
    driver.host = host
    return driver


def prepare():
    driver = context(); ctx = driver.context(); ctx.prepare()
    row = json.loads((ROOT/'record.json').read_text())
    row['environment'].update(STRATA_PREFILL_LAYER_MAJOR_PROFILE='0', STRATA_SYCL_PREFILL_MOE_CENSUS='0',
        STRATA_SYCL_PREFILL_NATIVE16='bf16', STRATA_SYCL_PREFILL_NATIVE16_BACKEND='vllm',
        STRATA_SYCL_PREFILL_NATIVE16_VLLM_TILE='auto', STRATA_SYCL_PREFILL_GU_SWIGLU='large')
    row['plan'].update(order=list(STAGES),
        host='Fresh physical-context request/hash-order host fixture,literal MoE census,54reader controls,'
             '11CPU-only owner/switch refusals,phase policy and selected decode-module equality with a347.',
        small='Fresh full trace and parameter validation for submission and phase fixtures before head work.',
        attention='Fresh small/full BF16 numerical/guard checks and off/on producer including262144cells.',
        gu='Fresh same-build actual GU smoke then44numerical cases,oneFP16ULP/exactBF16capture,'
           'guard/immutableinput/integer-dot and dispatch-boundary checks. Only these separate native '
           'children use immutable GU=auto to exercise both policy arms;the next model policy is large. '
           'No compact timing gate or adoption is reopened.',
        next='Only after all11stages pass,commit a new actual-model handoff and repeated32K/64K plan.'
             'Preserve independent decode,agent/task,physical262144/tails/RESTORE and clean-speed gates.')
    row['limits'] = ('All12ELFs are from the coherent full build,not old fixture/archive reuse. '
        'A host bounds test or262144-cell component does not qualify physical model fill. '
        'No partial-relink numerical/speed result,ordinary teardown,task or model adoption inherited.')
    row['retention'] = ('Keep compact gate outcomes for the imminent coherent-model comparison. '
        'Retire successful traces/cache and migrate old partial-build proof readers at closure. '
        'Existing97fda/56cad9/a347 named references and separate provenance holds remain unchanged.')
    ctx.context(row['native_observer_admission']).save(row)
    print('Prepared11fresh coherent native stages;commit before CPU/GPU execution.', flush=True)


def gu_stage(stage):
    driver = context(); ctx = driver.context()
    row = json.loads((ROOT/'record.json').read_text()); head = ctx.context(row['native_observer_admission'])
    head.verify(row)
    require(row['stage'] == 'in_progress' and len(row['results']) == STAGES.index(stage)
        and all(v['passed'] and v['ownership_closed'] for v in row['results']), 'fresh next GU stage required')
    root = ROOT/stage; root.mkdir(mode=0o700)
    for name in ('tmp', 'cache'):
        (root/name).mkdir(mode=0o700)
    scope = None
    def persist():
        original.common.write(root/'execution.json', dict(active=bool(scope and scope.active),
            commands=scope.commands if scope else []))
    cursor = subprocess.check_output(['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'],
        text=True, timeout=15).strip().split('-- cursor: ')[-1]
    require(cursor and '\n' not in cursor, 'kernel cursor required')
    mode = stage.removeprefix('gu-')
    result = dict(mode=stage, passed=False, complete=False, GPU_executed=False)
    try:
        scope = original.owner.Owner(root/'owned', persist)
        trace = mode == 'smoke'
        env = dict(row['environment'], STRATA_TEST_MAIN_GATES='0', STRATA_TRACE='1',
            STRATA_SYCL_PREFILL_GU_SWIGLU='auto', TMPDIR=str(root/'tmp'), XDG_CACHE_HOME=str(root/'cache'),
            LD_PRELOAD=row['native_validation_probe']['library']['path'],
            STRATA_UR_PROBE_EXPECTED_LOADER=row['native_validation_probe']['loader']['path'],
            UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION' if trace else 'UR_LAYER_PARAMETER_VALIDATION',
            UR_LOG_VALIDATION='level:warning;flush:warning;output:file,'+str(root/'validation.log'))
        if not trace:
            env.pop('UR_LOG_TRACING', None)
        result['native_execution_attempted'] = True
        event, out, err = scope.run(mode, [row['binaries']['sycl_gu_policy_test']['path'], '--'+mode],
            env, P, wall=240, cpu=720, rss_cap=4<<30, text_cap=64<<20, file_cap=64<<20, total_cap=80<<20)
        errors = err.read_text()
        proof = original.validation.preload_evidence(errors, event['pid'])
        previous.component.diagnostics(errors, event['pid'])
        require(original.owner.closed(event) and event['exit_code'] == 0
            and proof['loader'] == env['STRATA_UR_PROBE_EXPECTED_LOADER'], 'normal validated GU closure required')
        log = root/'validation.log'
        require(not log.exists() or log.stat().st_size == 0, 'runtime parameter validation warning')
        require(not trace or 'urEnqueue' in errors, 'actual traced GU submission missing')
        result['numerics'] = gu.parsed(out.read_text(), mode)
        head.verify(row)
        result.update(passed=True, complete=True, native_exit_code=0, host_probe_accepted=True,
            GPU_executed=True, validation=proof)
    except BaseException as error:
        result.update(passed=False, error=repr(error))
    finally:
        original.validation.finish(result, scope, cursor, row['boot'])
        result['passed'] = bool(result['passed'] and result['complete']
            and original.guards.driver_guard() == row['driver'])
        row['results'].append(result)
        row['commands'] += scope.commands if scope else []
        row['active'] = bool(scope and scope.active)
        row['ownership_closed'] = all(original.owner.closed(c) for c in row['commands']) and not row['active']
        row['complete'] = len(row['results']) == len(STAGES) and all(v['passed'] for v in row['results'])
        row['passed'] = row['complete']
        row['stage'] = 'failed' if not result['passed'] else 'closed' if row['complete'] else 'in_progress'
        head.save(row)
    require(result['passed'] and row['ownership_closed'], result.get('error', 'GU gate failed'))
    print(stage+' closed0', flush=True)


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', *STAGES), 'prepare|'+'|'.join(STAGES))
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); original.guards.no_model_owners([])
        if sys.argv[1] == 'prepare':
            prepare()
        elif sys.argv[1].startswith('gu-'):
            gu_stage(sys.argv[1])
        elif sys.argv[1] in original.STAGES:
            context().context().execute(sys.argv[1])
        else:
            context().extra(sys.argv[1])
