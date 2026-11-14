#!/usr/bin/env python3
"""Matched phase measurements of the six-file naming unit, Main only.

Owned Strata build recovery and bench.sh readers; repository LICENSE.
This compares one unit, without TU extraction or scalar restoration.
"""
import copy
import fcntl
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import socket
import statistics
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
import check_named_kv_model as model
import coherent_native_reuse as recovery
import decode_kv_prefill as prefill

kv = prefill.kv
P, B, RESULT = model.P, model.B, model.RESULT
THIS, TEST = 'tools/sycl/phase_naming_bench.py', 'tools/test_phase_naming_bench.py'
PYTHON, RUNS, ARMS, WORKLOAD_ID = kv.PYTHON, prefill.RUNS, kv.ARMS, kv.WORKLOAD_ID
MODE, ROOT, KEY = None, None, None
require, sha, signature = model.require, model.sha, model.signature
BUILDS = {
    'reference': ('decode-kv-native-v1', 'decode_kv_native_v1',
        '6c05d9a4dd6d813627672d9c9badd61daa173b90',
        '1eec28c77f393d34814dc748502113514973ea8e',
        '1ab18db544e7af8cf00e355550dc6e25aed721ab09b011bbdb5e045947ae7923',
        '330405370dab1d52995540e79c9dad5a3decf46d4e2007ab096dcc57e602d21e',
        13, 138, 135, 1471),
    'candidate': ('phase-kv-naming-native-v1', 'phase_kv_naming_native_v1',
        '9ff38147e5d9bbbaec96dc8dcae86ffd969f457c',
        '9849e35803f011aaafb2753c1e0f31411e40308e',
        '2dc7d4f3ead6068be12afa6ef36c589324f75d31ddd95f6fa96421552da57c9c',
        '6d2b4f7d6a79424c3da5ac6e35d7e1336016604e2f7bdb68b84031d2879903de',
        14, 140, 136, 1473),
}


def configure(mode):
    global MODE, ROOT, KEY
    require(mode in ('prefill', 'decode'), 'explicit phase required')
    MODE = mode
    ROOT = B / ('phase-kv-naming-' + mode + '-v1')
    KEY = 'phase_kv_naming_' + mode + '_v1'


def cases():
    return prefill.CASES if MODE == 'prefill' else [(None, arm) for arm in ARMS]


def historical(arm):
    root, key, evidence, source, record_sha, image_sha, images, objects, tus, inputs = BUILDS[arm]
    record = B / root / 'record.json'
    require(sha(record) == record_sha, 'original build record changed')
    row = json.loads(record.read_text())
    builder = model.native_device.build if arm == 'candidate' else model.native_device.build.previous
    committed = model.common.validation.committed(RESULT, (key,))
    original = json.loads(model.common.source.git(P, 'show', evidence + ':' + str(RESULT.relative_to(P))))[key]
    require(committed == original == builder.context().compact(row)
            and row['stage'] == 'built' and not row['active']
            and all(row[k] for k in ('passed', 'complete', 'ownership_closed'))
            and row['native_source_commit'] == row['project']['revision'] == source,
            'exact original coherent build evidence required')
    recovered = recovery.recover(row['project'])
    for pin in row['dependencies']: recovered.update(recovery.recover(pin))
    require(sha(row['recipe_seed']['path']) == row['recipe_seed']['sha256']
            and sha(row['external_inputs']['path']) == row['external_inputs']['sha256']
            and all(sha(p) == h for p, h in row['toolchain'].items()), 'original recipe/tool inputs changed')
    external = json.loads(Path(row['external_inputs']['path']).read_text())
    require(all(sha(p) == h for p, h in external.items()), 'original SDK/link input changed')
    for p, h in (row['configured_files'] | row['generated_inputs']).items():
        require(recovered.get(p) == h if p in recovered else sha(p) == h,
                'configured/generated input differs from the actual build')
    census = row['actual_inputs']
    require(census['source_commit'] == source and census['uncovered'] == 0
            and census['translation_units'] == tus and len(row['planned_compilation']) == objects
            and sha(census['path']) == census['sha256'], 'original compiler census changed')
    actual = json.loads(Path(census['path']).read_text())
    allowed = recovered | external | row['generated_inputs']
    require(len(actual['compiled']) == inputs and len(actual['generated']) == 1
            and len(actual['links']) == 29 and all(allowed.get(p) == h
                and (p in recovered or sha(p) == h) for family in ('compiled', 'generated', 'links')
                for p, h in actual[family].items()), 'recover exact historical compiler/link inputs')
    coherence = census['class_definition_coherence']
    require(coherence['fresh_complete_build'] and not coherence['partial_archive_reuse']
            and len(coherence['headers']) == 5 and all(v['all_objects_from_fresh_build']
                and v['header_sha256'] == recovered[str(Path(row['project']['root']) / n)]
                for n, v in coherence['headers'].items()), 'original class-definition closure changed')
    require(len(row['binaries']) == images and row['binaries']['strata']['sha256'] == image_sha,
            'wrong retained native generation')
    for v in [*row['binaries'].values(), *row['preserved_binaries']]:
        require(Path(v['path']).stat().st_size == v['bytes'] and sha(v['path']) == v['sha256'],
                'retained engine,fixture or faster phase reference changed')
    for v in row['native_validation_probe'].values():
        require(sha(v['path']) == v['sha256'], 'qualified validation provider changed')
    proof = dict(source_commit=source, build_evidence=evidence, record_sha256=record_sha,
        source_dependency_files=len(recovered), actual_census_sha256=census['sha256'],
        original_recipe_sha256=signature(row['recipe']), binaries=row['binaries'],
        historical_source_recovered=True, current_source_not_substituted=True)
    return row, proof


def source_pin(revision):
    values = kv.core.pins(revision)
    for name in (THIS, TEST, kv.THIS):
        raw = model.common.source.git(P, 'show', revision + ':' + name)
        require(raw == (P / name).read_bytes(), 'commit exact benchmark adapter and controls')
        values[name] = sha(P / name)
    return values


def basis():
    native, proofs, rows = {}, {}, {}
    for arm in BUILDS:
        row, proofs[arm] = historical(arm); rows[arm] = row
        native[arm] = dict(binary=row['binaries']['strata'], native_source_commit=row['native_source_commit'],
            native_validation_probe=row['native_validation_probe'])
    # Current native files must still select naming alone. Only the explicit
    # uncompiled measurement adapter and observer additions may differ.
    model.native_device.build.unit_sources()
    original = rows['candidate']['project']['source_sha256']
    current = model.common.source.pin(P, model.common.PREFIXES)
    additions = {model.native_device.THIS, model.oracle.THIS, model.THIS, THIS}
    changed = {n for n in original if n in current['source_sha256']
               and original[n] != current['source_sha256'][n]}
    require(set(current['source_sha256']) - set(original) == additions
            and not set(original) - set(current['source_sha256']) and changed == {kv.THIS},
            'only declared uncompiled benchmark additions/adapter may extend naming')
    actual = json.loads(Path(rows['candidate']['actual_inputs']['path']).read_text())
    require(not any(str(P / n) in actual[f] for n in additions | changed
                    for f in ('compiled', 'generated', 'links')), 'a benchmark delta is compiled native input')
    proofs['uncompiled_deltas'] = {n: dict(before=original.get(n), after=current['source_sha256'][n])
                                  for n in sorted(additions | changed)}
    gates = {}
    for arm, keys in [('reference', (kv.device.KEY, kv.model.KEY)),
                     ('candidate', (model.native_device.KEY, model.oracle.KEY, model.KEY))]:
        for key in keys:
            g = model.common.validation.committed(RESULT, (key,))
            require(g['stage'] == 'closed' and not g['active']
                    and all(g[k] for k in ('passed', 'complete', 'ownership_closed'))
                    and g['native_source_commit'] == native[arm]['native_source_commit'], 'closed same-build gates required')
            outcomes = g.get('results', g.get('cases', []))
            require(outcomes and all(v['passed'] and v['ownership_closed']
                    and v['native_exit_code'] == 0 and (not v.get('GPU_executed', True)
                        or v['fresh_faults'] == []) for v in outcomes), 'original qualification outcome changed')
            gates[key] = signature(g)
    candidate = model.common.validation.committed(RESULT, (model.KEY,))
    require(candidate['results'][0]['routes']['decode_KV']['actual_decode4_handoffs'] == 2
            and candidate['results'][0]['routes']['decode_KV']['prefill_cooperative_calls'] == 0
            and candidate['public_retention']['private_raw_retirement_key'] == 'phase_kv_naming_raw_retirement_v1',
            'curated closed production handoff required;do not resurrect its raw parser')
    static = model.common.validation.committed(RESULT, ('phase_kv_naming_elf_evidence',))
    image = static['runs']['v2']
    require(image['passed'] and image['complete'] and image['controls']['passed']
            and image['comparison']['equivalent'] and image['comparison']['raw_loads_equal']
            and image['comparison']['raw_load_segments'] == 4
            and image['comparison']['equal_section_count'] == 36,
            'compatibility-only naming control requires exact host/device loaded-image proof')
    static_source = model.common.source.git(P, 'show', image['controller_commit']
        + ':tools/sycl/check_naming_elf_equivalence.py')
    require(hashlib.sha256(static_source).hexdigest() == image['controller_sha256']
            and sha(B / 'naming-elf-equivalence-v2/record.json') == image['record_sha256'],
            'original static identity proof/source changed')
    gates['phase_kv_naming_elf_evidence'] = signature(static)
    prior = json.loads((B / 'coherent-generations-prefill-v1/record.json').read_text())
    require(model.common.validation.committed(RESULT, ('coherent_generations_prefill_v1',))
            == kv.compact(prior), 'recover the exact retained configuration/model/environment manifest')
    require(prior['stage'] == 'closed' and prior['complete'] and prior['passed']
            and prior['ownership_closed'] and not prior['active'], 'closed common frontend/configuration required')
    require(native['reference']['native_validation_probe'] == native['candidate']['native_validation_probe']
            == prior['native_validation_probe'], 'both arms need the same qualified validation provider')
    files = json.loads(Path(prior['runtime_manifest']['path']).read_text())
    require(sha(prior['runtime_manifest']['path']) == prior['runtime_manifest']['sha256']
            and all(sha(p) == h for p, h in files.items())
            and kv.build.guards.driver_guard() == prior['driver']
            and Path('/proc/sys/kernel/random/boot_id').read_text().strip() == prior['boot'],
            'qualified runtime,boot or driver changed')
    for value in native.values(): value['boot'] = prior['boot']
    return native, dict(proofs=proofs, gates=gates), prior, files, current


def compact(row):
    out = kv.compact(row)
    out['live_snapshot'] = {k: v for k, v in row['live_snapshot'].items() if k != 'source_sha256'}
    return out


def save(row):
    kv.core.write(ROOT / 'record.json', row)
    data = json.loads(RESULT.read_text()); data[KEY] = compact(row); kv.core.write(RESULT, data)


def native_context(row, plan):
    index = plan['index']
    require(type(index) is int and 0 <= index < len(cases())
            and (plan.get('read'), plan['kernel']) == cases()[index]
            and plan['heads'] == 1 and plan['adapt_swaps'] == 0, 'unplanned native arm or phase')
    native = row['native_builds'][plan['kernel']]
    expected = BUILDS[plan['kernel']]
    require(native['native_source_commit'] == expected[3]
            and native['binary']['sha256'] == expected[5], 'wrong source/image selected for arm')
    return native


configuration = kv.configuration


def environment(row, plan):
    native_context(row, plan)
    return dict(kv.base.environment(row, plan), STRATA_SYCL_DECODE_KV_COALESCED='1',
                STRATA_SYCL_KV_COPY_GROUPS='0')


def invariant(row, plan):
    return signature(json.dumps(dict(config=configuration(row, plan), environment=environment(row, plan)),
        sort_keys=True).replace(plan['root'], '<process>'))


def command(plan):
    if MODE == 'prefill': return prefill.command(plan)
    return ['/bin/sh', str(P / 'bench.sh'), '--config', str(Path(plan['root']) / 'config.json'),
        '--port', str(plan['port']), '--long-generation-only', '--think', '--runs', '1',
        '--profile-request-gates', '--workload-id', WORKLOAD_ID, '--fail-fast-engine-error',
        '--ready-timeout', '600', '--out', str(Path(plan['root']) / 'bench')]


def current(row):
    require(model.common.validation.committed(RESULT, (KEY,)) == compact(row)
            and source_pin(row['source_commit']) == row['sources'], 'freeze exact source and current plan')
    native, proofs, prior, files, live = basis()
    require(native == row['native_builds'] and proofs == row['native_proofs']
            and live['source_sha256'] == row['live_snapshot']['source_sha256']
            and signature(prior) == row['configuration_evidence_sha256'], 'frozen build/source/qualification changed')
    contract, manifest = kv.runner.model.python_manifest()
    require(contract == row['python_contract'] and manifest['selected'] == row['python_selected']
            and sha(row['template']['path']) == row['template']['sha256']
            and kv.build.supervision.artifact_bytes(ROOT) <= row['whole_bytes']
            and len(row['plans']) == len(cases()), 'frontend/template/capture budget changed')
    for i, plan in enumerate(row['plans']):
        root = Path(plan['root']); selected = native_context(row, plan)
        require(plan['index'] == i and plan['command'] == command(plan)
                and json.loads((root / 'config.json').read_text()) == configuration(row, plan)
                and json.loads((root / 'plan.json').read_text()) ==
                    {k: selected[k] for k in ('binary', 'native_validation_probe')}
                and all(sha(root / p) == h for p, h in plan['generated'].items())
                and invariant(row, plan) == row['invariant_signature']
                and kv.runner.comparison.fingerprint(environment(row, plan)) == plan['environment_sha256'],
                'exact generated launcher/configuration/environment changed')
    kv.shared.models.model_pins(row['configuration'], row['model_assets'], full=False)
    return files, manifest


def request_route(text, plan):
    out = kv.base.prefill.context().request_route(text, plan)
    out['KV_captures'] = kv.kv_captures(text, True)
    require(any(c['phase'] == 'decode' and c['width'] == 4 for c in out['KV_captures']), 'actual cooperative decode4 required')
    footers = kv.base.copy_records(text, arm='reference')
    require(len(footers) == 1 and footers[0]['read_tokens'] == plan['read']
            and footers[0]['generated_tokens'] == 24, 'exact fresh24-output prefill extent required')
    kv.base.clean_observers(text)
    return out


def decision(processes):
    require(len(processes) == len(cases()) and all(p['index'] == i
            and p['kernel'] == cases()[i][1]
            and p['native_source_commit'] == BUILDS[p['kernel']][3]
            and p['binary']['sha256'] == BUILDS[p['kernel']][5]
            for i, p in enumerate(processes)), 'complete phase/source-bound cohort required')
    if MODE == 'prefill':
        result = prefill.decision(processes)
        ratios = [metric['paired_ratios'] for point in result['points']
                  for scope in point['scopes'].values() for metric in scope.values()]
        result['naming_control_checks'] = [paired_control(values) for values in ratios]
        result['naming_control_eligible'] = all(v['no_clear_slowdown'] for v in result['naming_control_checks'])
        return result
    require(all(p['passed'] and p['complete'] and p['ownership_closed'] and p['native_exit_code'] == 0
            and p['fresh_faults'] == [] and len(p['samples']) == 2 for p in processes), 'complete closed decode cohort required')
    result = kv.decision(processes)
    for value in result['workloads'].values():
        value['TG_median_preserved'] = statistics.median(value['paired_ratios'][i]['TG'] for i in range(3)) >= 1
        value['naming_control'] = paired_control([1 / p['TG'] for p in value['paired_ratios']])
    result['median_preservation_screen_passed'] = all(v['TG_median_preserved']
        and v['reference_MTP_class_confirmed'] for v in result['workloads'].values())
    result['question'] = 'Naming-only image versus unchanged1eec at identical KV1,not an expected gain.'
    result['naming_control_eligible'] = all(v['naming_control']['no_clear_slowdown']
                                          and v['reference_MTP_class_confirmed']
                                          for v in result['workloads'].values())
    return result


def paired_control(ratios):
    """Prospective naming-control screen,never an optimization/adoption rule.

    For three independent paired log time ratios,the two-sided95% Student
    interval uses df2: F(t)=1/2+t/(2*sqrt(t*t+2));solve F(t)=0.975.
    The independent full loaded-image proof is mandatory in basis().
    """
    require(len(ratios) == 3 and all(type(v) in (int, float) and math.isfinite(v)
            and v > 0 for v in ratios), 'three finite positive paired time ratios required')
    values = [math.log(v) for v in ratios]
    mean = statistics.mean(values)
    critical = math.sqrt(2 * .95 ** 2 / (1 - .95 ** 2))
    half = critical * statistics.stdev(values) / math.sqrt(3)
    return dict(paired_time_ratios=ratios, log_mean=mean, log_95_lower=mean-half,
        log_95_upper=mean+half, no_clear_slowdown=mean-half <= 0,
        meaning='Paired naming-only compatibility screen backed by exact loaded-image identity;'
                'not proof of speedup,statistical equivalence alone or general inference adoption.')


def context():
    if MODE == 'prefill':
        return SimpleNamespace(ROOT=ROOT, KEY=KEY, ORDER=prefill.ORDER, RUNS=RUNS, current=current,
            environment=environment, native_context=native_context, request_route=request_route,
            check_sample=prefill.check_sample, decision=decision, compact=compact, save=save,
            validate_model_text=kv.runner.generation.gdn.validate_model_text)
    spec = importlib.util.spec_from_file_location('naming_decode_owned', kv.__file__)
    ctx = importlib.util.module_from_spec(spec); spec.loader.exec_module(ctx)
    for key, value in dict(ROOT=ROOT, KEY=KEY, current=current, native_context=native_context,
            configuration=configuration, environment=environment, invariant=invariant,
            decision=decision, compact=compact, save=save).items(): setattr(ctx, key, value)
    ctx.api_window = lambda text, sample, row, arm: kv.api_window(text, sample, row, 'candidate')
    return ctx


def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()), 'one fresh phase plan')
    native, proofs, prior, files, live = basis(); contract, manifest = kv.runner.model.python_manifest()
    row = {k: copy.deepcopy(prior[k]) for k in ('configuration', 'environment_template', 'template',
        'boot', 'driver', 'runtime_manifest', 'model_assets', 'native_validation_probe')}
    revision = model.common.source.git(P, 'rev-parse', 'HEAD').decode().strip()
    row.update(source_commit=revision, sources=source_pin(revision), live_snapshot=live,
        native_builds=native, native_proofs=proofs, configuration_evidence_sha256=signature(prior),
        python_contract=contract, python_selected=manifest['selected'],
        # Compatibility for host/frontend readers only;native_context selects each child.
        binary=native['reference']['binary'], native_source_commit=native['reference']['native_source_commit'],
        insertions=kv.insertion_recipe(row['configuration']), stage='prepared', plans=[], processes=[],
        request_hashes={}, host_passed=False, passed=False, complete=False, active=False, ownership_closed=True,
        whole_bytes=(832 if MODE == 'prefill' else 448) << 20,
        budgets=dict(wall=1800 if MODE == 'prefill' else 5400, cpu=36000, rss_cap=104 << 30,
            text_cap=16 << 20, file_cap=32 << 20, total_cap=64 << 20),
        plan=dict(unit='A:only six-file mechanical decode names;no TU split or scalar restoration.',
            order='Twelve32K/64Kprocesses in the unchanged alternating plan,three long requests each,first/warmed separate.'
                  if MODE == 'prefill' else 'Six fresh R,C,C,R,R,Cprocesses;four official warmups,then regular-code and varied-prose once,thinking on,cap16384.',
            constants='Same current bench.sh/server/engine_once/placement,262144logical/20480residentINT8KV,'
                      '8192throughputBF16/largeGU/bulk1/staging0,1256static slots,5workers,pcie0/adapt0,spec4,'
                      'head2/dense1,KVcoalesced1/copygroups0 in both immutable child environments. Only native ELF/source differs.',
            acceptance='Recover original sources/recipes/compiler/class/SDK/census and all retained images before launch.'
                       'Freeze current live source,controller/tests/generated launcher plans/providers/Python/models.'
                       'Exact fresh read/zero reuse/valid clientTTFT/journal/work and actual phase routes required.'
                       'Decode needs>=1Knative outputs with thinking and an answer plus observed strong/weak MTPclasses.'
                       'Normal0,reaped empty owner,quiet validation,no fresh fault,unchangedboot/driver.'
                       'Hash all15model assets before this cohort and after its final process.',
            decision='Keep every sample and all three paired ratios. No expected speed gain from renaming.'
                     'Retain the eight-series median prefill/TTFT screen and median TGscreen as separate diagnostics,'
                     'including every failed series. This is a naming-only compatibility control,not engine adoption:'
                     'require the separately committed exact host/device loaded-image identity,numerical/handoff'
                     'gates and three paired log-time ratios per series. A two-sided95%Student interval(df2)'
                     'with lower bound>0 is a clear slowdown and blocks control eligibility;otherwise report'
                     'the interval without claiming speedup or equivalence from timing alone. TGuses log(R/C),'
                     'and observed strong/weak reference MTPclasses remain mandatory.'
                     'Any failed screen remains unresolved;no posthoc exclusion,threshold change or added samples.'
                     'Both naming-control phases must pass before replacing the unchanged comparison generation'
                     'or advancing B1. This prospective control-only criterion changes no earlier failed KVflag'
                     'or cross-build decision and does not apply to optimized/different loaded images.',
            stopping='Commit source/plan;host controls;commit;each fresh process in order with closure commits.'
                     'Stop first source/owner/count/clock/route/provider/validation/fault/budget/eligibility failure;no retry or pin refresh.',
            limits='No new native build or default inference adoption;no physical262144fill/RESTORE/agent/task/ordinaryteardown or hardware-saturation claim.'
                   'Previous failed KVflag and cross-build screens remain unchanged.',
            retention='Keep compact individual samples/source/configuration/decisions;retire successful raw/cache on closure.'
                      'Only the already declared three native generations;select/retire the superseded comparison image before a B1build.'))
    kv.shared.models.model_pins(row['configuration'], row['model_assets'], full=True)
    ROOT.mkdir(mode=0o700)
    for i, (read, arm) in enumerate(cases()):
        root = ROOT / str(i); root.mkdir(mode=0o700)
        for name in ('tmp', 'cache'): (root / name).mkdir()
        with socket.socket() as sock: sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
        plan = dict(index=i, kernel=arm, heads=1, adapt_swaps=0, root=str(root), port=port)
        if read is not None: plan['read'] = read
        selected = native_context(row, plan)
        kv.core.write(root / 'config.json', configuration(row, plan))
        kv.core.write(root / 'plan.json', {k: selected[k] for k in ('binary', 'native_validation_probe')})
        plan.update(command=command(plan), generated={n: sha(root / n) for n in ('config.json', 'plan.json')})
        plan['environment_sha256'] = kv.runner.comparison.fingerprint(environment(row, plan)); row['plans'].append(plan)
    row['invariant_signature'] = invariant(row, row['plans'][0])
    require(all(invariant(row, p) == row['invariant_signature'] for p in row['plans']), 'undeclared comparison arm difference')
    save(row); print('Prepared naming-only '+MODE+' comparison;commit before host/native.', flush=True)


if __name__ == '__main__':
    require(sys.executable == PYTHON and len(sys.argv) == 3, 'qualified Python;prefill|decode prepare|host|index')
    configure(sys.argv[1]); action = sys.argv[2]
    require(action in ('prepare', 'host', *map(str, range(len(cases())))), 'one declared stage')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); kv.build.guards.no_model_owners([])
        if action == 'prepare': prepare()
        else:
            with (ROOT / ('host-attempt.json' if action == 'host' else action + '/attempt.json')).open('x') as f:
                json.dump(dict(action=action, result_key=KEY), f)
            try:
                if action == 'host': kv.runner.comparison.host(context())
                elif MODE == 'prefill': kv.runner.comparison.run(int(action), context())
                else: context().run(int(action))
            except BaseException as error:
                row = json.loads((ROOT / 'record.json').read_text())
                if row['stage'] != 'failed':
                    row.update(stage='failed', passed=False, complete=False, error=repr(error), failed_action=action); save(row)
                raise
