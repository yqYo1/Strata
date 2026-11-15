#!/usr/bin/env python3
"""Repeat bench.sh long-prefill timing on the coherent, model-checked engine.

Main-authored from the maintained owned benchmark controls; repository LICENSE.
This is same-build reproduction,not an engine speedup or model adoption test.
"""
import copy
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
import coherent_prefill_model as model
import gu_small_bench as dispatch_reader
import qsa16_stream_bench as core

P, B, RESULT = model.P, model.B, model.RESULT
ROOT, KEY = B/'coherent-prefill-speed-v1', 'coherent_prefill_speed_v1'
THIS = 'tools/sycl/coherent_prefill_bench.py'
READS, RUNS = (32769, 65537, 65537, 32769, 32769, 65537), 3
require, sha = model.require, model.sha


def basis():
    n = model.native(observers={THIS})
    prior = json.loads((model.ROOT/'record.json').read_text())
    committed = model.original.validation.committed(RESULT, (model.KEY,))
    require(committed == model.original.previous.compact(prior)
        and prior['stage'] == 'closed' and all(prior[k] for k in
            ('passed', 'complete', 'ownership_closed', 'model_executed'))
        and not prior['active'] and prior['native_build']['binaries'] == n['binaries']
        and prior['native_source_commit'] == n['native_source_commit']
        and sha(prior['native_record']['path']) == prior['native_record']['sha256'],
        'closed exact coherent model admission required')
    source = model.original.device.common.source
    proof = copy.deepcopy(n['model_observer_admission'])
    current = source.pin(P, model.original.device.common.PREFIXES)['source_sha256']
    require(THIS in proof['added_observers'] and THIS in current, 'declare new benchmark observer')
    del proof['added_observers'][THIS]; del current[THIS]
    proof['observer_source_files'] -= 1
    proof['observer_source_sha256'] = model.original.device.common.signature(current)
    proof['additional_observers'] = []
    require(proof == prior['native_observer_admission'],
        'only this observer may extend original coherent model/native controls')
    pin = prior['controller']; before = pin['source_sha256']
    recovered = {p: hashlib.sha256(raw).hexdigest() for p, raw in
        source.blobs(P, pin['revision'], before)}
    after = source.pin(P, ['tools/sycl', 'tools/strata_tokenizer.py',
        'data/expert-profile.bin'])['source_sha256']
    require(recovered == before and set(after)-set(before) == {THIS}
        and not set(before)-set(after) and all(after[p] == h for p, h in before.items()),
        'recover original controls;do not refresh changed source pins')
    outcome, = prior['results']
    require(outcome['native_exit_code'] == 0 and outcome['ownership_closed']
        and not outcome['fresh_faults'] and outcome['boot_unchanged']
        and outcome['driver_after'] == prior['driver']
        and [(x['prompt_tokens'], x['generated'], x['complete']) for x in outcome['requests']]
            == [(32769, 32, True), (82, 32, True), (32769, 32, True)]
        and outcome['routes']['actual_attention_math'] == 'bf16-prefix-ordered-exp'
        and outcome['routes']['actual_bulk_calls'] == 96
        and outcome['routes']['cache_binding_and_final_drains']
        and all(v['compact_jobs'] == 0 for v in outcome['routes']['actual_GU_dispatch']),
        'actual large-GU/typed-attention/state handoff required')
    providers = prior['runtime_files']
    require(sha(prior['runtime_manifest']['path']) == prior['runtime_manifest']['sha256']
        and all(sha(p) == h for p, h in providers.items())
        and core.source.model.guards.driver_guard() == prior['driver']
        and Path('/proc/sys/kernel/random/boot_id').read_text().strip() == prior['boot'],
        'runtime,driver or boot changed')
    model.original.previous.assets(prior)
    diagnostic = dict(key=model.KEY, evidence_sha256=core.source.signature(committed),
        original_controller_source=prior['source_commit'], original_native_source=n['native_source_commit'],
        model_handoff_passed=True, model_speed_task_full_context_inherited=False,
        final15full_hashes_reused_only_after_current_identity_check=True)
    prior = copy.deepcopy(prior)
    prior['observer_admission'] = dict(native=n['observer_admission'],
        original_model_evidence=diagnostic, fast_configuration_diagnostic=diagnostic,
        original_model_controls_recovered=len(before), new_benchmark_observer=THIS,
        native_recompiled=False, earlier_speed_inherited=False)
    return n, prior, providers, None


def decision(processes):
    require(len(processes) == len(READS) and all(p['passed'] and p['complete']
        and p['ownership_closed'] and p['native_exit_code'] == 0 and not p['fresh_faults']
        and p['read'] == READS[i] and len(p['samples']) == RUNS
        for i, p in enumerate(processes)), 'all six ordered processes/eighteen samples required')
    points = []
    for read in (32769, 65537):
        group = [p for p in processes if p['read'] == read]
        warm = [p['subsequent_median_ms'] for p in group]
        first = [p['samples'][0]['prompt_ms'] for p in group]
        rates = [read*1000/ms for ms in warm]
        points.append(dict(read=read, subsequent_prompt_ms=statistics.median(warm),
            subsequent_PP=read*1000/statistics.median(warm), subsequent_process_PP=rates,
            subsequent_client_TTFT_s=statistics.median(statistics.median(
                s['ttft_s'] for s in p['samples'][1:]) for p in group),
            first_PP=read*1000/statistics.median(first),
            first_client_TTFT_s=statistics.median(p['samples'][0]['ttft_s'] for p in group),
            all_warmed_processes_above600=all(x>600 for x in rates),
            all_warmed_processes_at_least1000=all(x>=1000 for x in rates)))
    return dict(points=points,
        prefill600_reproduced=all(x['all_warmed_processes_above600'] for x in points),
        prefill1000_at64K_reproduced=points[1]['all_warmed_processes_at_least1000'],
        local_adjacent_ms_per_token=(points[1]['subsequent_prompt_ms']-
            points[0]['subsequent_prompt_ms'])/(65537-32769),
        fitted_physical_law=False, new_engine_speedup_claim=False, inference_adopted=False,
        model_quality_decode_full_context_or_whole_goal_qualified=False)


def context():
    spec = importlib.util.spec_from_file_location('coherent_prefill_speed_context', core.__file__)
    ctx = importlib.util.module_from_spec(spec); spec.loader.exec_module(ctx)
    ctx.ROOT, ctx.KEY, ctx.THIS = ROOT, KEY, THIS
    ctx.READS, ctx.ORDER, ctx.RUNS = READS, [(n, 1) for n in READS], RUNS
    ctx.CLEAN_FAST, ctx.basis, ctx.decision = True, basis, decision
    old_env, old_route, old_current = ctx.environment, ctx.request_route, ctx.current
    def environment(row, plan):
        env = old_env(row, plan)
        env.update(STRATA_SYCL_PREFILL_QSA_MATH='bf16-prefix-ordered-exp',
            STRATA_SYCL_PREFILL_QSA_BULK_HOST_MIRROR='1', STRATA_SYCL_PREFILL_EXPERT_UPLOAD_STAGING='0',
            STRATA_SYCL_PREFILL_MOE_CENSUS='0', STRATA_PREFILL_LAYER_MAJOR_PROFILE='0',
            STRATA_SYCL_PREFILL_GU_SWIGLU='large')
        return env
    def current(row):
        require(row['read_order'] == list(READS) and len(row['plans']) == len(READS),
            'frozen length order changed')
        return old_current(row)
    def route(text, plan):
        result = old_route(text, plan)
        result['actual_GU_dispatch'] = dispatch_reader.dispatch(text, 'large')
        require('softmax=parallel-exp-ascending-sum' in text and 'softmax=subgroup' not in text,
            'actual ordered-exp attention required')
        return result
    def sample(sample, count, read, minimum_ms=10000):
        core.comparison.check_sample(sample, count, read, minimum_ms)
        require(sample['generated_tokens'] == 24, 'exact24outputs required')
    ctx.environment, ctx.current, ctx.request_route, ctx.check_sample = environment, current, route, sample
    return ctx


def prepare():
    ctx = context(); ctx.prepare()
    row = json.loads((ROOT/'record.json').read_text()); row['read_order'] = list(READS)
    row['whole_bytes'] = 1024 << 20
    row['plan'] = dict(question='Reproduce current coherent large-GU PP/time/clientTTFT at32K/64K.',
        order='Six fresh processes at32769,65537,65537,32769,32769,65537actual inputs. Three thinking-on exact24-output bench.sh requests per process. Keep first separate from subsequent2;all18samples and process spreads. Commit each closed process before advancing.',
        settings='Coherent14508full-build dcb8 ELF,logical262144/resident20480K8V8,8192chunks,throughput,FAST9,ordered-exp/bulk1/staging0/source-view1/private-map1,384slots/ordinary,head2/dense1/backing0,MTPFP16 and GUlarge. No engine/math/phase-selector change;immutable pre-exec safety and quiet retained-reference validation remain charged.',
        acceptance='Current-source frontend CPU controls,exact source/build/12ELFs/provider/model/workload/owner identities. Normal SSE,exact fresh counts,thinking,24outputs,valid clocks/clientTTFT,actual all-GPU384-slot route and GU/Down/job conservation,quiet validation,normal0/reaped/empty ownership,no new fault. No API/profiler/stage trace.',
        target='Report600and1000reproduction independently. 1000at64Krequires all3warmed process medians>=1000;600requires all3warmed medians>600atbothlengths. No engine speedup/adoption claim,automatic exclusion or performance-based stop. First-use medians and TTFT remain separate.',
        stopping='Stop first input/provider/owner/token/clock/route/validation/fault/budget failure. No retry,pin refresh,criteria change or speed-based exclusion. Reuse exact closed-model15full hashes only after identity/source checks;rehash all15after the sixth process.',
        limits='Same-build repeated baseline,not sustainedTG,agent/task quality,physical262144/tails/RESTORE,ordinary teardown,hardware saturation or whole-goal completion. Two lengths give an adjacent slope,not a fitted physical law. The historical mixed-definition samples stay held.',
        retention='Keep compact individual18samples/route counts and source/runtime/configuration identities. Retire successful raw output after verification. Existing old-reference reader dependencies remain until the next scoped reader migration,before another engine build.')
    ctx.save(row)
    print('Prepared six coherent long-prefill processes;commit before host/native.', flush=True)


if __name__ == '__main__':
    require(sys.executable == model.PYTHON, 'Use pinned benchmark interpreter: '+model.PYTHON)
    require(len(sys.argv) == 2 and (sys.argv[1] in ('prepare', 'host')
        or sys.argv[1].isdigit() and int(sys.argv[1]) < len(READS)), 'prepare|host|0..5')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); model.original.guards.no_model_owners([])
        if sys.argv[1] == 'prepare':
            prepare()
        elif sys.argv[1] == 'host':
            core.comparison.host(context())
        else:
            core.comparison.run(int(sys.argv[1]), context())
