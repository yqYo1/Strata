#!/usr/bin/env python3
"""Matched clean bulk-mirror comparison using the maintained bench.sh runner.

Main-authored from this fork's owned measurement controls; repository LICENSE.
This reuses exact native/model qualification, not earlier performance results.
"""
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
import qsa_bulk_model_entry as model
import qsa16_stream_bench as original

P, B, RESULT = model.P, model.B, original.RESULT
THIS = 'tools/sycl/qsa_bulk_bench.py'
ROOT = B / 'qsa16-bulk-mirror-prefill-compare-v1'
KEY = 'qsa16_bulk_mirror_prefill_compare_v1'
FLAG = 'STRATA_SYCL_PREFILL_QSA_BULK_HOST_MIRROR'
CASES = tuple((read, arm) for read in (32769, 65537) for arm in (0, 1, 1, 0))
RUNS = 3
require, sha = model.require, model.sha


def basis():
    dev = model.device.context()
    native = dev.native(observers={dev.THIS, model.THIS, THIS})
    prior = json.loads((model.ROOT / 'record.json').read_text())
    require(dev.validation.committed(RESULT, (model.KEY,)) == model.original.previous.compact(prior)
        and prior['stage'] == 'closed' and all(prior[k] for k in
            ('passed', 'complete', 'ownership_closed', 'model_executed'))
        and not prior['active'] and prior['native_source_commit'] == native['native_source_commit']
        and prior['native_build']['binaries'] == native['binaries']
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
    observed = native['observer_admission']
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
    require(recovered == before and original.source.signature(before) == pin['aggregate_sha256']
        and set(after) - set(before) == {THIS} and not set(before) - set(after)
        and all(after[p] == h for p, h in before.items()),
        'only this new committed benchmark observer may extend the qualified model')
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
        recovered_controller_inputs=len(recovered), changed_controller=[THIS],
        model_assets_sha256=original.source.signature(prior['models']),
        meaning='Exact closed bulk-model final15full hashes reused with unchanged identities;'
            'rehash all15after this cohort. No inherited clean speed or adoption.')
    proof = dict(native=observed, original_model_source=prior['source_commit'],
        original_model_evidence_sha256=original.source.signature(model.original.previous.compact(prior)),
        original_model_receipt_sha256=sha(model.ROOT / 'record.json'),
        original_model_controls_recovered=len(before), added_observers={THIS: after[THIS]},
        fast_configuration_diagnostic=diagnostic,
        collector_qualification_inherited=False, native_recompiled=False,
        earlier_clean_result_inherited=False)
    return native, dict(prior, observer_admission=proof), prior['runtime_files'], None


def decision(processes):
    require(len(processes) == len(CASES) and all(p['passed'] and p['complete']
        and p['ownership_closed'] and p['native_exit_code'] == 0 and not p['fresh_faults']
        and p['read'] == CASES[i][0] and len(p['samples']) == RUNS
        for i, p in enumerate(processes)), 'all eight ordered clean processes required')
    points = []
    for offset, read in ((0, 32769), (4, 65537)):
        local = processes[offset:offset + 4]
        def arm_summary(arm):
            selected = [p for i, p in enumerate(local) if CASES[offset + i][1] == arm]
            warm_ms = [p['subsequent_median_ms'] for p in selected]
            first_ms = [p['samples'][0]['prompt_ms'] for p in selected]
            warm_ttft = [statistics.median(s['ttft_s'] for s in p['samples'][1:]) for p in selected]
            first_ttft = [p['samples'][0]['ttft_s'] for p in selected]
            pp = [read * 1000 / ms for ms in warm_ms]
            return dict(processes=len(selected), subsequent_samples_per_process=RUNS - 1,
                subsequent_median_prompt_ms=statistics.median(warm_ms),
                subsequent_PP=read * 1000 / statistics.median(warm_ms),
                subsequent_process_PP_min=min(pp), subsequent_process_PP_max=max(pp),
                subsequent_median_client_TTFT_s=statistics.median(warm_ttft),
                first_median_prompt_ms=statistics.median(first_ms),
                first_PP=read * 1000 / statistics.median(first_ms),
                first_median_client_TTFT_s=statistics.median(first_ttft))
        off, on = arm_summary(0), arm_summary(1)
        paired = [local[a]['subsequent_median_ms'] / local[b]['subsequent_median_ms']
                  for a, b in ((0, 1), (3, 2))]
        points.append(dict(read=read, off=off, on=on,
            subsequent_prompt_speedup=off['subsequent_median_prompt_ms'] / on['subsequent_median_prompt_ms'],
            subsequent_TTFT_speedup=off['subsequent_median_client_TTFT_s'] / on['subsequent_median_client_TTFT_s'],
            paired_subsequent_prompt_ratios=paired,
            both_independent_pairs_favor_bulk=all(v > 1 for v in paired)))
    gain = all(p['both_independent_pairs_favor_bulk'] for p in points)
    slower = all(all(r <= 1 for r in p['paired_subsequent_prompt_ratios']) for p in points)
    slopes = {arm: (points[1][arm]['subsequent_median_prompt_ms']
        - points[0][arm]['subsequent_median_prompt_ms']) / (65537 - 32769) for arm in ('off', 'on')}
    return dict(points=points, clean_speed=True, matched_same_binary=True,
        independent_processes_per_arm_per_length=2,
        mechanism_evidence=('repeated_improvement_at_both_lengths' if gain else
            'no_repeated_improvement' if slower else 'mixed_requires_fresh_repeats'),
        advance_to_further_model_gates=gain,
        local_adjacent_ms_per_token=slopes,
        curve_scope='Two lengths give local slopes only;no fitted law,physical fixed-cost inference or extrapolation. Extend the existing multi-length characterization after a useful change.',
        prefill1000_on_64K_subsequent_median_observed=points[1]['on']['subsequent_PP'] >= 1000,
        first_use_is_separate=True, sustained_decode_qualified=False,
        agent_task_quality_qualified=False, physical262144_qualified=False,
        resource_saturation_qualified=False, inference_adopted=False,
        next='If paired evidence is mixed,commit a fresh repetition plan;do not relabel noise or drop samples. Preserve both selectors. Before adoption compare the retained clean reference and independent sustained decode/agent/task/full262144/RESTORE gates. No prefill gain excuses decode loss.')


def context():
    spec = importlib.util.spec_from_file_location('qsa_bulk_clean_benchmark', original.__file__)
    ctx = importlib.util.module_from_spec(spec); spec.loader.exec_module(ctx)
    ctx.ROOT, ctx.KEY, ctx.THIS = ROOT, KEY, THIS
    ctx.READS = tuple(read for read, _ in CASES)
    ctx.ORDER = [(read, 1) for read, _ in CASES]
    ctx.RUNS, ctx.CLEAN_FAST, ctx.basis, ctx.decision = RUNS, True, basis, decision
    base_environment, base_current, base_route = ctx.environment, ctx.current, ctx.request_route
    def environment(row, plan):
        index = plan['index']
        require(type(index) is int and 0 <= index < len(CASES)
            and plan['read'] == CASES[index][0], 'unplanned bulk comparison arm')
        env = base_environment(row, plan)
        env[FLAG] = str(CASES[index][1])
        return env
    def current(row):
        require(row['arm_order'] == [list(v) for v in CASES]
            and len(row['plans']) == len(CASES)
            and all((p['read'], p['bulk_mirror']) == CASES[i]
                and p['index'] == i and p['heads'] == 1 for i, p in enumerate(row['plans'])),
            'frozen ABBA arms/lengths changed')
        return base_current(row)
    def request_route(text, plan):
        route = base_route(text, plan)
        route.update(bulk_mirror=CASES[plan['index']][1],
            selector_evidence='Observed typed-QSA request entry plus immutable environment checked at both live request gates and exact recovered production append dispatch. No per-launch bulk telemetry in clean mode.',
            actual_bulk_command_count_measured=False)
        return route
    ctx.environment, ctx.current, ctx.request_route = environment, current, request_route
    return ctx


def prepare():
    ctx = context(); ctx.prepare()
    row = json.loads((ROOT / 'record.json').read_text())
    row['arm_order'] = [list(v) for v in CASES]
    for index, plan in enumerate(row['plans']):
        plan['bulk_mirror'] = CASES[index][1]
        require(plan['environment_sha256'] == ctx.comparison.fingerprint(ctx.environment(row, plan)),
            'generated pre-exec arm environment changed')
    row['whole_bytes'] = 1024 << 20
    row['plan'] = dict(question='Does bounded same-queue bulk Host mirror publication improve actual long-prefill PP/clientTTFT over direct Host-USM stores?',
        order='Eight fresh processes:32769off/on/on/off,then65537off/on/on/off. Three exact-count thinking-on bench.sh requests each,24outputs. First measured long request separate from subsequent2. Two independent processes per arm/length. Identical per-length nonce/body sequence across arms;zero reuse. Commit every closed process before advancing.',
        settings='Same56cad9/611a0fd6native and five exact fixtures,logical262144/resident20480INT8,owned264MiBstage,8192chunks,1256decode slots,5workers,spec4,throughput,head2/dense1/backing0 and all nine qualified FAST settings. Only immutable bulk flag0/1 and private process paths differ. Preserve qualified runtime safety;remove trace/timing/audit observers. No native rebuild or decode selection change.',
        acceptance='Existing fresh host/source/provider/ownership/request-journal/count/clock/clientTTFT/quiet-validation/normal0/unchanged-boot/no-fault gates. Actual per-request8192BF16QSA,all residual rows onGPU,staged384bank,ordinary accumulation,checked drains,head/MTP restore. Native environment is observed at both gates. Clean-mode bulk selection additionally follows exact recovered deterministic dispatch;no per-launch command or physical traffic claim.',
        comparison='Use per-process subsequent2 median times,and AB/BA paired off/on ratios at each length. Repeated improvement requires both independent pairs favor bulk at both lengths;no arbitrary minimum gain. Mixed results need a fresh committed repetition plan. First PP/clientTTFT and spread remain separate,not a throughput veto or pooled warm sample.',
        stopping='Stop first source/owner/provider/count/clock/route/validation/fault/finite-budget failure. No retries,pin refresh,changed criteria,speed-based early stop or dropped samples. Reuse original closed-model final15hashes with exact identities;hash all15again after this cohort.',
        limits='Local same-binary prefill mechanism comparison only;24outputs do not qualify sustainedTG. No actual task/agent quality,full262144model/RESTORE,ordinary-provider teardown,paired comparison against b9aa,hardware saturation or overall adoption. Two lengths permit local slopes,not a new fitted law.',
        retention='Keep24compact individual samples,actual routes,source/native/environment/workload and decision. Retire successful raw stage/journal/cache history after cohort closure. Keep the declared four temporary engine generations only through matched phase selection;recover original inputs and retire the superseded generation. No new binary generation.')
    ctx.save(row)
    print('Prepared exact eight-process bulk off/on comparison;commit before host/native.', flush=True)


if __name__ == '__main__':
    require(sys.executable == model.PYTHON, 'Use pinned prompt-generator interpreter: ' + model.PYTHON)
    require(len(sys.argv) == 2 and (sys.argv[1] in ('prepare', 'host')
        or sys.argv[1].isdigit() and int(sys.argv[1]) < len(CASES)), 'prepare|host|INDEX0..7')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        model.original.guards.no_model_owners([])
        if sys.argv[1] == 'prepare': prepare()
        elif sys.argv[1] == 'host': original.comparison.host(context())
        else: original.comparison.run(int(sys.argv[1]), context())
