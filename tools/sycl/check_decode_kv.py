#!/usr/bin/env python3
"""Fresh same-build qualification of the decode KV successor; Main owns jobs.

Uses the owned coherent numerical/trace gates with the current reference ELF.
No old GPU results or model qualification are inherited. Repository LICENSE.
"""
import copy
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
import build_decode_kv as build
import check_coherent_prefill as previous
import fixture_build as common

P, B, RESULT = build.P, build.B, build.RESULT
ROOT, KEY = B / 'decode-kv-device-v1', 'decode_kv_device_v1'
THIS = 'tools/sycl/check_decode_kv.py'
STAGES = previous.STAGES
require, sha, signature = build.require, build.sha, build.signature


def native(expected=None, *, observers=None):
    observers = {THIS} if observers is None else set(observers)
    require(THIS in observers and all(p.startswith('tools/sycl/') and p.endswith('.py')
        for p in observers), 'declare exact uncompiled Python observers')
    row = json.loads((build.ROOT / 'record.json').read_text())
    require(common.validation.committed(RESULT, (build.KEY,)) == build.context().compact(row)
        and row['stage'] == 'built' and not row['active'] and not row['GPU_executed']
        and all(row[k] for k in ('passed', 'complete', 'ownership_closed')),
        'committed closed full-engine build required')
    before = row['project']['source_sha256']
    recovered = {p: hashlib.sha256(raw).hexdigest() for p, raw in
        common.source.blobs(P, row['project']['revision'], before)}
    after = common.source.pin(P, common.PREFIXES)['source_sha256']
    require(recovered == before and set(after) - set(before) == observers
        and not set(before) - set(after) and all(after[p] == h for p, h in before.items()),
        'only this committed observer may extend the exact native snapshot')
    for pin in row['dependencies']:
        found = {p: hashlib.sha256(raw).hexdigest() for p, raw in
            common.source.blobs(Path(pin['root']), pin['revision'], pin['source_sha256'])}
        require(found == pin['source_sha256'] and all(sha(Path(pin['root']) / p) == h
            for p, h in found.items()), 'compiled dependency source changed')
    require(sha(build.SEED) == row['recipe_seed']['sha256']
        and sha(row['external_inputs']['path']) == row['external_inputs']['sha256']
        and all(sha(p) == h for p, h in row['toolchain'].items()), 'original recipe/tool inputs changed')
    sdk = json.loads(Path(row['external_inputs']['path']).read_text())
    require(all(sha(p) == h for p, h in sdk.items()), 'original SDK inputs changed')
    seed = json.loads(build.SEED.read_text())
    recipe = copy.deepcopy(seed['recipe']); old = str(build.SEED.parent)
    recipe['configure'] = [str(build.ROOT) + v[len(old):] if v.startswith(old) else v
        for v in recipe['configure']]
    recipe['configure'][recipe['configure'].index('-S') + 1] = str(P)
    recipe['environment'] = {k: v.replace(old, str(build.ROOT)) for k, v in recipe['environment'].items()}
    recipe['environment']['STRATA_SYCL_DECODE_KV_COALESCED'] = '0'
    recipe['targets'] = recipe['build_targets'] = build.TARGETS
    require(recipe == row['recipe'] and build.prerequisite() == row['build_prerequisite'],
        'exact native recipe or prospective component decision changed')
    actual = row['actual_inputs']
    require(actual['source_commit'] == row['native_source_commit'] == row['project']['revision']
        and actual['translation_units'] == 135 and actual['uncovered'] == 0
        and len(row['planned_compilation']) == 138
        and set(row['binaries']) == set(build.TARGETS)
        and sha(actual['path']) == actual['sha256'], 'full native compiler census changed')
    coherence = actual['class_definition_coherence']
    require(set(coherence['headers']) == set(build.HEADERS)
        and coherence['fresh_complete_build'] and not coherence['partial_archive_reuse']
        and all(v['header_sha256'] == sha(P / p) and v['all_objects_from_fresh_build']
            for p, v in coherence['headers'].items()), 'fresh class-definition coherence required')
    for p, h in (row['configured_files'] | row['generated_inputs']).items():
        require(sha(p) == h, 'configured/generated input changed')
    census = json.loads(Path(actual['path']).read_text())
    require(all(sha(p) == h for family in ('compiled', 'generated', 'links')
        for p, h in census[family].items()), 'actual compiler/link input changed')
    for image in (*row['preserved_binaries'], *row['binaries'].values()):
        require(Path(image['path']).stat().st_size == image['bytes']
            and sha(image['path']) == image['sha256'], 'candidate/reference executable changed')
    require(build.base.probe_qualification() ==
        (row['native_validation_probe'], row['validation_qualification']), 'qualified runtime probe changed')
    proof = dict(native_source_commit=row['native_source_commit'], original_source_files=len(before),
        observer_source_files=len(after), observer_source_sha256=signature(after),
        added_observers={p: after[p] for p in sorted(observers)}, actual_census_sha256=actual['sha256'],
        class_coherence=coherence, exact_thirteen_ELFs_census_recipe=True,
        exact_seed_recipe_sha256=signature(recipe), native_recompiled=False, GPU_result_inherited=False)
    require(expected is None or proof == expected, 'frozen native observer admission changed')
    return dict(row, observer_admission=proof)


def controller():
    spec = importlib.util.spec_from_file_location('decode_kv_coherent_checks', previous.__file__)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.ROOT, module.KEY, module.THIS, module.build, module.native = ROOT, KEY, THIS, build, native
    original_context = module.context
    def context():
        driver = original_context()
        # The old host reader compared with the now-retired dense predecessor.
        # Keep its exact selected-module comparison, using the retained clean ELF.
        driver.original = SimpleNamespace(**vars(driver.original))
        driver.original.candidate = SimpleNamespace(**vars(driver.original.candidate))
        driver.original.candidate.SEED = build.SEED
        original_host = driver.host
        def host(device, ctx, row, scope, outcome):
            env = dict(row['environment']); env.pop('LD_PRELOAD', None)
            image = row['binaries']['sycl_expert_dispatch_bounds_test']
            entry, out, err = scope.run('native-dispatch-bounds', [image['path']], env, P,
                **row['host_budget'])
            require(common.ownership.closed(entry) and entry['exit_code'] == 0
                and out.read_text() == 'expert-dispatch-bounds: passed cases=6 phase=2 GPU_executed=0\n'
                and not err.read_text(), 'same-build native MAXT guard failed')
            original_host(device, ctx, row, scope, outcome)
            outcome['native_dispatch_bounds'] = dict(cases=6, phases=2, passed=True, GPU_executed=False)
        driver.host = host
        return driver
    module.context = context
    return module


def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()), 'fresh native qualification only')
    n = native(); module = controller(); driver = module.context()
    head = driver.context().context(n['observer_admission'])
    prior = common.validation.committed(RESULT, ('coherent_prefill_device_v1',))
    require(prior['stage'] == 'closed' and not prior['active']
        and all(prior[k] for k in ('passed', 'complete', 'ownership_closed')),
        'qualified baseline environment required')
    row = copy.deepcopy(json.loads((B / 'coherent-prefill-device-v1/record.json').read_text()))
    require(head.compact(row) == prior, 'original qualified environment receipt changed')
    env = {k: v.replace(str(B / 'coherent-prefill-device-v1'), str(ROOT))
        for k, v in row['environment'].items()}
    env.update(STRATA_SYCL_DECODE_KV_COALESCED='1', STRATA_SYCL_KV_COPY_GROUPS='0')
    require(env['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD'] == '1'
        and env['EnableDirectSubmission'] == '0' and env['STRATA_SYCL_VERIFY_NATIVE_CAPTURE'] == '0'
        and env['STRATA_SYCL_PREFILL_SAME_QUEUE'] == '1', 'preserve qualified safety settings')
    require(common.guards.driver_guard() == prior['driver']
        and Path('/proc/sys/kernel/random/boot_id').read_text().strip() == prior['boot']
        and all(sha(p) == h for p, h in row['runtime_pins'].items()), 'qualified boot/driver/runtime changed')
    row.update(stage='prepared', source_commit=common.source.git(P, 'rev-parse', 'HEAD').decode().strip(),
        controller=common.source.pin(P, ['tools/sycl']), native_source_commit=n['native_source_commit'],
        native_record=dict(path=str(build.ROOT / 'record.json'), sha256=sha(build.ROOT / 'record.json')),
        native_observer_admission=n['observer_admission'], binary=n['binaries']['sycl_q6_head_owner_test'],
        binaries=n['binaries'], prior_environment_source='coherent_prefill_device_v1', environment=env,
        results=[], commands=[], passed=False, complete=False, active=False, ownership_closed=True,
        GPU_executed=False, model_executed=False)
    row['plan'].update(order=list(STAGES),
        host='Fresh same-build six-case native MAXT guard,physical-context admission/hash ordering,literal MoE census,54reader controls,11CPU-only owner/switch refusals,phase/shape/cache policy and exact selected head module against retained14508. No old ELF invocation.',
        next='After all11fresh stages pass,commit a separate32K/short/32Kproduction handoff proving actual decode-only cooperative captures. Repair API/native-segment journal binding before sustained thinking speed comparison. Preserve dedicated-prefill,agent/task and actual262144/RESTORE obligations.')
    row['limits'] = ('Fresh same-build CPU/GPU prerequisite only. Component795oracle/six-cell speed evidence '
        'does not qualify this model route. These small native fixtures may select the scalar geometry fallback;'
        'actual target-shape cooperative capture requires model evidence. No model speed/task/agent,physical'
        '262144/RESTORE,ordinary provider teardown or default adoption.')
    row['retention'] = ('One validated14508engine and one immediate330405candidate plus the1e776component;'
        'no new binary generation. Preserve compact numeric/closure/fault and actual route evidence;retire '
        'successful raw traces/cache at closure after original-input recovery. Three separate legacy source '
        'recovery holds remain recorded in the KVproduction-preparation result.')
    ROOT.mkdir(mode=0o700)
    for name in ('tmp', 'cache'):
        (ROOT / name).mkdir(mode=0o700)
    head.save(row)
    print('Prepared11fresh same-build native stages;commit before CPU/GPU execution.', flush=True)


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', *STAGES), 'prepare|' + '|'.join(STAGES))
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); common.guards.no_model_owners([])
        if sys.argv[1] == 'prepare':
            prepare()
        elif sys.argv[1].startswith('gu-'):
            controller().gu_stage(sys.argv[1])
        elif sys.argv[1] in previous.original.STAGES:
            controller().context().context().execute(sys.argv[1])
        else:
            controller().context().extra(sys.argv[1])
