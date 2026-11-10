#!/usr/bin/env python3
"""Prospective control-only admission for the qualified bulk-mirror native build.

Main-authored from this fork's exact source/recipe/census and closed model
admission. Repository LICENSE. Historical comparator ELFs remain provenance in
their immutable receipts; the current engine and all four same-build fixtures
remain required bytes. This does not change any original validation outcome.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import qsa_bulk_bench as previous_bench
import check_qsa16_stream_model as legacy
import compare_official_prefill as comparison

model, original = previous_bench.model, previous_bench.original
common, candidate, validation = legacy.common, legacy.candidate, legacy.validation
require, sha = legacy.require, legacy.sha
P, B, RESULT = legacy.P, legacy.B, legacy.RESULT
NATIVE, NATIVE_KEY = model.device.build.ROOT, model.device.build.KEY
THIS = 'tools/sycl/qsa_bulk_controls.py'
FLAG = previous_bench.FLAG
UPDATED_CONTROLLERS = {'tools/sycl/compare_official_prefill.py',
    'tools/sycl/qsa16_stream_bench.py'}


def control_delta(before, after, observers):
    """Permit the declared Python observers; every other original input is exact."""
    changed = {p for p, h in before.items() if p in after and after[p] != h}
    require(set(after) - set(before) == set(observers) and not set(before) - set(after)
        and changed <= UPDATED_CONTROLLERS
        and all(p.startswith('tools/sycl/') and p.endswith('.py')
            for p in changed | set(observers)),
        'undeclared native/control source change')
    return {p: dict(original=before[p], prospective=after[p]) for p in sorted(changed)}


def verify_current_binaries(binaries, inspect=None):
    require(set(binaries) == set(legacy.TARGETS), 'all five current native ELFs required')
    def actual(item):
        path = Path(item['path'])
        return path.stat().st_size, sha(path)
    inspect = actual if inspect is None else inspect
    for name, item in binaries.items():
        require(Path(item['path']) == NATIVE / 'build' / name
            and inspect(item) == (item['bytes'], item['sha256']),
            'current engine/fixture changed or missing')


def check_recipe(actual, seed):
    # Match the actual relocated bulk build, not the legacy b9aa default root.
    return model.device.context().check_recipe(actual, seed)


def check_sample_exact(sample, count, read, minimum_ms=10000):
    comparison.check_sample(sample, count, read, minimum_ms)
    require(type(sample['generated_tokens']) is int and sample['generated_tokens'] == 24,
        'prospective prefill workload requires exactly24 generated tokens')


def host_controls():
    good = dict(ok=True, eligible=True, stream_complete=True, thinking_requested=True,
        reasoning_observed=True, actual_read_tokens=32769, actual_prompt_tokens=32769,
        actual_reused_tokens=0, request_sha256='body', generated_tokens=24,
        prompt_tps_source='engine', prompt_ms=40000.0, ttft_s=40.1, total_s=41.0)
    count = dict(counted_input_tokens=32769, body_sha256='body')
    check_sample_exact(good, count, 32769)
    rejected = []
    for n in (0, 1, 23, 25, True):
        altered = dict(good, generated_tokens=n)
        try: check_sample_exact(altered, count, 32769)
        except RuntimeError: rejected.append(str(n))
        else: raise RuntimeError('exact output gate accepted ' + str(n))
    # Legacy users keep their declared range; a new policy explicitly selects exact24.
    comparison.check_sample(dict(good, generated_tokens=23), count, 32769)
    before = {'src/example.cpp': 'native', 'tools/sycl/compare_official_prefill.py': 'old'}
    after = dict(before, **{THIS: 'observer'})
    after['tools/sycl/compare_official_prefill.py'] = 'new'
    control_delta(before, after, {THIS})
    for altered in (dict(after, **{'src/example.cpp': 'changed'}),
            {p:h for p,h in after.items() if p != 'src/example.cpp'},
            dict(after, **{'src/unplanned.cpp': 'extra'})):
        try: control_delta(before, altered, {THIS})
        except RuntimeError: rejected.append('source')
        else: raise RuntimeError('native source mutation accepted')
    entries = {n:dict(path=str(NATIVE/'build'/n), bytes=i+1, sha256=str(i))
               for i,n in enumerate(legacy.TARGETS)}
    verify_current_binaries(entries, lambda v:(v['bytes'],v['sha256']))
    for name in legacy.TARGETS:
        altered = {k:v for k,v in entries.items() if k != name}
        try: verify_current_binaries(altered, lambda v:(v['bytes'],v['sha256']))
        except RuntimeError: rejected.append('missing-'+name)
        else: raise RuntimeError('missing current fixture accepted')
        try: verify_current_binaries(entries,
            lambda v:(v['bytes'], 'wrong') if v is entries[name] else (v['bytes'],v['sha256']))
        except RuntimeError: rejected.append('changed-'+name)
        else: raise RuntimeError('mutated current fixture accepted')
    return dict(passed=True, exact24_accepted=True, exact_output_rejections=5,
        native_source_rejections=3, missing_current_ELF_rejections=5,
        changed_current_ELF_rejections=5, legacy_range_preserved=True, GPU_executed=False)


def native(expected=None,*,observers):
    observers=set(observers)
    require(THIS in observers,'prospective admission observer must remain pinned')
    row=json.loads((NATIVE/'record.json').read_text())
    require(validation.committed(RESULT,(NATIVE_KEY,))==candidate.compact(row) and
        row['stage']=='built' and row['passed'] and row['complete'] and
        row['ownership_closed'] and not row['active'] and not row['GPU_executed'],
        'closed exact full-engine build required')
    original=row['project'];current=common.source.pin(P,common.PREFIXES)
    a,b=original['source_sha256'],current['source_sha256']
    require(a=={p:hashlib.sha256(raw).hexdigest() for p,raw in
        common.source.blobs(P,original['revision'],a)},
        'original complete native project inputs must remain recoverable')
    changes=control_delta(a,b,observers)
    require(row['native_source_commit']==original['revision']==row['actual_inputs']['source_commit']
        and row['actual_inputs']['translation_units']==128 and row['actual_inputs']['uncovered']==0,
        'original native source and complete compiler census required')
    for pin in row['dependencies']:
        require(pin['source_sha256']=={p:hashlib.sha256(raw).hexdigest() for p,raw in
            common.source.blobs(Path(pin['root']),pin['revision'],pin['source_sha256'])} and
            all(sha(Path(pin['root'])/p)==h for p,h in pin['source_sha256'].items()),
            'original dependency source changed')
    require(sha(candidate.SEED)==row['recipe_seed']['sha256'] and
        sha(row['external_inputs']['path'])==row['external_inputs']['sha256'] and
        all(sha(p)==h for p,h in row['toolchain'].items()) and
        all(sha(p)==h for p,h in json.loads(Path(row['external_inputs']['path']).read_text()).items()),
        'compiler/SDK/recipe seed changed')
    seed=json.loads(candidate.SEED.read_text())
    recipe_sha256=check_recipe(row['recipe'],seed)
    require(common.HOST_PYTHON==seed['host_python'] and
        sha(common.HOST_PYTHON)==seed['host_python_sha256'],
        'qualified host-control interpreter changed')
    verify_current_binaries(row['binaries'])
    require(candidate.probe_qualification()==(row['native_validation_probe'],row['validation_qualification']),
        'qualified pre-main loader/runtime proof changed')
    require(all(sha(p)==h for p,h in (row['configured_files']|row['generated_inputs']).items()) and
        sha(row['actual_inputs']['path'])==row['actual_inputs']['sha256'],
        'original configured/generated/compiler census changed')
    census=json.loads(Path(row['actual_inputs']['path']).read_text())
    require(all(sha(p)==h for field in ('compiled','generated','links')
        for p,h in census[field].items()),'actual compiled/header/link inputs changed')
    proof=dict(native_source_commit=row['native_source_commit'],original_source_files=len(a),
        observer_source_files=len(b),observer_source_sha256=common.signature(b),
        added_observers={p:b[p] for p in sorted(observers)},exact_five_ELFs_census_recipe=True,
        exact_seed_recipe_sha256=recipe_sha256,
        controller_only_changes=changes,
        current_required_ELFs=sorted(row['binaries']),
        historical_reference_ELFs_are_provenance_only=len(row['preserved_binaries']),
        native_recompiled=False,GPU_result_inherited=False)
    require(expected is None or proof==expected,'native observer admission changed')
    return dict(row,observer_admission=proof)


def basis(extra_observers=()):
    dev = model.device.context()
    observers = {dev.THIS, model.THIS, previous_bench.THIS, THIS, *extra_observers}
    current_native = native(observers=observers)
    prior = json.loads((model.ROOT / 'record.json').read_text())
    require(dev.validation.committed(RESULT, (model.KEY,)) == model.original.previous.compact(prior)
        and prior['stage'] == 'closed' and all(prior[k] for k in
            ('passed', 'complete', 'ownership_closed', 'model_executed'))
        and not prior['active'] and prior['native_source_commit'] == current_native['native_source_commit']
        and prior['native_build']['binaries'] == current_native['binaries']
        and sha(prior['native_record']['path']) == prior['native_record']['sha256'],
        'closed exact candidate-model/source/five-ELF qualification required')
    result, = prior['results']
    routes = result['routes']
    require(result['passed'] and result['native_exit_code'] == 0
        and result['ownership_closed'] and result['fresh_faults'] == []
        and result['boot_unchanged'] and result['driver_after'] == prior['driver']
        and [(v['prompt_tokens'], v['generated'], v['complete']) for v in result['requests']]
            == [(32769, 32, True), (82, 32, True), (32769, 32, True)]
        and routes['actual_mode'] == 1 and routes['typed_QSA_entry']
        and routes['packed_layer_entries'] == 96 and routes['staged_bank_headers'] == 2
        and routes['actual_batched_ring_callbacks'] == 8
        and routes['cache_binding_and_final_drains']
        and routes['MTP_expert_restore_bytes'] == [707788800] * 2
        and routes['MTP_round_recaptures'] >= 2
        and routes['decode_routes']['head']['actual_decode_C4_replays'] > 0
        and routes['actual_bulk_calls'] == 96 and routes['accepted_copy_commands'] == 384
        and routes['requested_D2H_bytes'] == 830472192,
        'actual typed attention/bulk publication/fast placement/decode handoff required')
    admitted = prior['native_observer_admission']
    observed = current_native['observer_admission']
    require(admitted['native_source_commit'] == observed['native_source_commit']
        and admitted['exact_seed_recipe_sha256'] == observed['exact_seed_recipe_sha256']
        and admitted['exact_five_ELFs_census_recipe'] and admitted['fresh_five_stage_controls']
        and set(admitted['added_observers']) == {dev.THIS, model.THIS}
        and all(observed['added_observers'][p] == h for p, h in admitted['added_observers'].items())
        and admitted['device_key'] == model.device.KEY
        and admitted['device_evidence_sha256'] == prior['device_evidence_sha256']
        == original.source.signature(dev.validation.committed(RESULT, (model.device.KEY,))),
        'original exact fresh device qualification and both qualified observers changed')
    pin = prior['controller']; before = pin['source_sha256']
    recovered = {p: hashlib.sha256(raw).hexdigest() for p, raw in
        dev.common.source.blobs(P, pin['revision'], before)}
    after = dev.common.source.pin(P,
        ['tools/sycl', 'tools/strata_tokenizer.py', 'data/expert-profile.bin'])['source_sha256']
    require(recovered == before and original.source.signature(before) == pin['aggregate_sha256'],
        'original qualified model controls must remain recoverable')
    control_delta(before,after,{previous_bench.THIS,THIS,*extra_observers})
    require(all(prior['environment'].get(k) == v for k, v in original.FAST.items())
        and prior['environment'][FLAG] == '1'
        and sha(prior['runtime_manifest']['path']) == prior['runtime_manifest']['sha256']
        and all(sha(p) == h for p, h in prior['runtime_files'].items())
        and model.original.guards.driver_guard() == prior['driver']
        and Path('/proc/sys/kernel/random/boot_id').read_text().strip() == prior['boot'],
        'exact actual-model fast configuration/runtime/boot/driver required')
    model.original.previous.assets(prior)
    diagnostic = dict(key=model.KEY,
        evidence_commit='f79a5cd2145efb91ff85445710bbb31f62f00e61',
        original_controller_source=prior['source_commit'],
        original_native_source=prior['native_source_commit'],
        receipt_sha256=sha(model.ROOT / 'record.json'),
        recovered_controller_inputs=len(recovered), changed_controller=[previous_bench.THIS,THIS,*sorted(extra_observers)],
        model_assets_sha256=original.source.signature(prior['models']),
        meaning='Exact closed bulk-model final15full hashes reused with unchanged identities;'
            'rehash all15after this cohort. No inherited clean speed or adoption.')
    proof = dict(native=observed, original_model_source=prior['source_commit'],
        original_model_evidence_sha256=original.source.signature(model.original.previous.compact(prior)),
        original_model_receipt_sha256=sha(model.ROOT / 'record.json'),
        original_model_controls_recovered=len(before), added_observers={p:after[p] for p in (previous_bench.THIS,THIS,*sorted(extra_observers))},
        fast_configuration_diagnostic=diagnostic,
        collector_qualification_inherited=False, native_recompiled=False,
        earlier_clean_result_inherited=False, prospective_host_controls=host_controls())
    return current_native, dict(prior, observer_admission=proof), prior['runtime_files'], None


def admit():
    """Record the prospective reader without relabelling the closed speed cohort."""
    n, prior, _, _ = basis({'tools/sycl/qsa_bulk_profile.py'})
    row = json.loads((previous_bench.ROOT / 'record.json').read_text())
    data = json.loads(RESULT.read_text())
    require(validation.committed(RESULT, (previous_bench.KEY,)) == original.compact(row)
        and row['complete'] and row['passed'] and row['ownership_closed']
        and not row['active'] and 'prospective_control_admission' not in row,
        'one closed original cohort without a previous reconciliation required')
    proof = dict(controller_source_commit=common.source.git(P, 'rev-parse', 'HEAD').decode().strip(),
        native_source_commit=n['native_source_commit'], native_recompiled=False,
        native_observer_admission=n['observer_admission'],
        model_qualification=prior['observer_admission'],
        original_measurement_source_unchanged=row['source_commit'],
        original_samples_and_outcomes_unchanged=True,
        current_required_ELFs=sorted(n['binaries']),
        historical_reference_ELFs_required=False,
        scope='Prospective Python-control reconciliation only. Same exact native/recipe/five fixtures,'
            'closed device/model evidence,assets/providers/boot. No new GPU result or speed/adoption.')
    row['prospective_control_admission'] = proof
    original.write(previous_bench.ROOT / 'record.json', row)
    data[previous_bench.KEY] = original.compact(row)
    original.write(RESULT, data)
    print('Prospective native/control admission passed;original measured source and results preserved.', flush=True)


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] == 'admit', 'qsa_bulk_controls.py admit')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        model.original.guards.no_model_owners([])
        admit()
