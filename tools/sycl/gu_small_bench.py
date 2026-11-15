#!/usr/bin/env python3
"""Matched bench.sh comparison of prefill GU policies in one owned engine.

Main-authored from this fork's maintained benchmark controls; repo LICENSE.
No kernel, provider, decode placement or native executable changes.
"""
import copy
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import statistics
import sys
sys.dont_write_bytecode = True
import build_gu_small_model as build
import gu_small_model_entry as model
import qsa16_stream_bench as core

P, B, RESULT = build.P, build.B, build.RESULT
ROOT, KEY = B/'gu-small-prefill-compare-v1', 'gu_small_prefill_compare_v1'
THIS = 'tools/sycl/gu_small_bench.py'
FIXTURE = 'tests/data/gpu_roles_vectors.json'
ARMS = (0, 1, 1, 0, 0, 1)
CASES = tuple((read, arm) for read in (32769, 65537) for arm in ARMS)
POLICIES = ('large', 'auto')
RUNS = 3
require, sha = build.require, build.sha


def basis(extra_observers=()):
    extras = set(extra_observers)
    require(all(p.startswith('tools/sycl/') and p.endswith('.py') for p in extras),
        'additional observers must be named Python measurement controls')
    n = build.native(observers={model.THIS, THIS, FIXTURE, *extras})
    prior = json.loads((model.ROOT/'record.json').read_text())
    committed = model.original.validation.committed(RESULT, (model.KEY,))
    require(committed == model.original.previous.compact(prior)
        and prior['stage'] == 'closed' and prior['passed'] and prior['complete']
        and prior['ownership_closed'] and not prior['active'] and prior['model_executed']
        and prior['native_build']['binaries'] == n['binaries']
        and sha(prior['native_record']['path']) == prior['native_record']['sha256'],
        'closed exact candidate/model handoff required')
    native_proof = copy.deepcopy(n['observer_admission'])
    for name in (THIS, FIXTURE, *sorted(extras)):
        require(name in native_proof['declared_inputs']['added_observers'], 'new observer must be explicit')
        del native_proof['declared_inputs']['added_observers'][name]
        native_proof['declared_inputs']['observer_source_files'] -= 1
    require(all(prior['native_observer_admission'][k] == v for k, v in native_proof.items()),
        'only named new benchmark and uncompiled host fixture may extend native qualification')
    fixture = build.original.common.source.git(P, 'show', 'HEAD:'+FIXTURE)
    require(hashlib.sha256(fixture).hexdigest() == sha(P/FIXTURE)
        == 'b5b8032be5c3ecf39177b9748d0c0e3b1be854c31d7a4ed0ff805775c13b0a50',
        'exact independent Niko host-only fixture required')
    outcome, = prior['results']
    require(outcome['passed'] and outcome['native_exit_code'] == 0 and outcome['ownership_closed']
        and outcome['fresh_faults'] == [] and outcome['boot_unchanged']
        and outcome['driver_after'] == prior['driver']
        and [(v['prompt_tokens'], v['generated'], v['complete']) for v in outcome['requests']]
            == [(32769, 32, True), (82, 32, True), (32769, 32, True)]
        and all(v['accepted'] > 0 and v['offered'] > 0 for v in outcome['requests'])
        and outcome['routes']['actual_attention_math'] == 'bf16-prefix-ordered-exp'
        and outcome['routes']['actual_bulk_calls'] == 96
        and outcome['routes']['accepted_copy_commands'] == 384
        and outcome['routes']['MTP_expert_restore_bytes'] == [707788800]*2
        and outcome['routes']['cache_binding_and_final_drains'], 'actual typed route and independent decode handoff required')
    require(prior['environment']['STRATA_SYCL_PREFILL_DOWN_SCATTER'] == 'large'
        and prior['environment']['STRATA_SYCL_PREFILL_NATIVE16_BACKEND'] == 'vllm'
        and prior['environment']['STRATA_SYCL_PREFILL_NATIVE16_VLLM_TILE'] == 'auto',
        'equal GU/Down wide eligibility requires the exact qualified backend policy')
    pin = prior['controller']
    after = build.original.common.source.pin(P,
        ['tools/sycl', 'tools/strata_tokenizer.py', 'data/expert-profile.bin'])['source_sha256']
    before = pin['source_sha256']
    require(before == {p: hashlib.sha256(raw).hexdigest() for p, raw in
        build.original.common.source.blobs(P, pin['revision'], before)}
        and set(after)-set(before) == {THIS, *extras} and not set(before)-set(after)
        and all(after[p] == h for p, h in before.items()),
        'recover original model controls;only this new benchmark may extend them')
    require(sha(prior['runtime_manifest']['path']) == prior['runtime_manifest']['sha256'],
        'original provider manifest changed')
    providers = json.loads(Path(prior['runtime_manifest']['path']).read_text())
    require(all(sha(p) == h for p, h in providers.items())
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
        model_control_sources_recovered=len(before), new_benchmark_observer=THIS,
        additional_uncompiled_fixture=FIXTURE, native_recompiled=False, earlier_speed_inherited=False)
    return n, prior, providers, None


def dispatch(text, policy):
    def one(label):
        found = re.findall(r'^strata MoE '+label+r' fusion: fused_jobs=(\d+) fused_rows=(\d+) '
            r'separate_jobs=(\d+) min_rows=2334$', text, re.M)
        require(len(found) == 1, 'one actual completed '+label+' aggregate required')
        return tuple(map(int, found[0]))
    gu, down = one('GU'), one('Down')
    values = re.findall(r'^strata MoE accumulator window: windows=(\d+) dequants=(\d+) jobs=(\d+) '
        r'source_H2D_bytes=(\d+) GPU_resident_windows_only=1 operator_max=8192 maximum_rows=(\d+)$', text, re.M)
    require(len(values) == 1, 'one completed job aggregate required')
    windows, dequants, jobs, requested_maps, maximum_rows = map(int, values[0])
    require(jobs > 0 and windows > 0 and dequants > 0 and 0 < maximum_rows <= 262144
        and gu[0]+gu[2] == down[0]+down[2] == jobs
        and 0 <= down[0] <= jobs and 2334*down[0] <= down[1] <= 8192*down[0],
        'completed GU/Down/job conservation failed')
    compact_jobs, compact_rows = gu[0]-down[0], gu[1]-down[1]
    if policy == 'large':
        require(gu == down, 'same wide eligibility must select equal GU/Down outcomes')
    else:
        require(policy == 'auto' and compact_jobs > 0
            and compact_jobs <= compact_rows <= 96*compact_jobs,
            'actual compact decisions must augment the identical wide subset')
    return dict(gu=dict(zip(('fused_jobs', 'fused_rows', 'separate_jobs'), gu)),
        down=dict(zip(('fused_jobs', 'fused_rows', 'separate_jobs'), down)),
        compact_jobs=compact_jobs, compact_rows=compact_rows, jobs=jobs,
        windows=windows, dequants=dequants, maximum_physical_window_rows=maximum_rows,
        requested_source_map_bytes=requested_maps,
        scope='Existing actual returned-boolean aggregates after checked drains. Same>=2334 eligibility for GU/Down is proved in the pinned source;auto difference is compact1..96. Legacy GU min_rows label names wide only. No per-job trace,physical bytes or saturation claim.')


def decision(processes):
    require(len(processes) == len(CASES) and all(p['passed'] and p['complete']
        and p['ownership_closed'] and p['native_exit_code'] == 0 and not p['fresh_faults']
        and p['read'] == CASES[i][0] and len(p['samples']) == RUNS for i,p in enumerate(processes)),
        'all12 ordered processes and36 eligible samples required')
    points = []
    for offset, read in ((0,32769), (6,65537)):
        local = processes[offset:offset+6]
        arms = {}
        for arm, name in enumerate(POLICIES):
            selected = [p for i,p in enumerate(local) if ARMS[i] == arm]
            warm = [p['subsequent_median_ms'] for p in selected]
            first = [p['samples'][0]['prompt_ms'] for p in selected]
            arms[name] = dict(subsequent_prompt_ms=statistics.median(warm),
                subsequent_PP=read*1000/statistics.median(warm),
                subsequent_process_PP=[read*1000/v for v in warm],
                subsequent_client_TTFT_s=statistics.median(statistics.median(s['ttft_s'] for s in p['samples'][1:]) for p in selected),
                first_PP=read*1000/statistics.median(first),
                first_client_TTFT_s=statistics.median(p['samples'][0]['ttft_s'] for p in selected))
        ratios = [local[b]['subsequent_median_ms']/local[a]['subsequent_median_ms']
            for a,b in ((0,1),(3,2),(4,5))]
        points.append(dict(read=read, **arms, paired_subsequent_time_ratios=ratios,
            geometric_mean_time_ratio=statistics.geometric_mean(ratios), all_pairs_favor_auto=all(v<1 for v in ratios)))
    gain = all(p['all_pairs_favor_auto'] for p in points)
    return dict(points=points, advance_to_further_model_gates=gain,
        mechanism_evidence='repeated_improvement' if gain else 'not_consistently_faster',
        matched_same_binary=True, parameter_validation_charged=True, observer_free=False,
        prefill1000_at64K_subsequent_median=points[1]['auto']['subsequent_PP'] >= 1000,
        local_adjacent_ms_per_token={a:(points[1][a]['subsequent_prompt_ms']-points[0][a]['subsequent_prompt_ms'])/(65537-32769) for a in POLICIES},
        sustained_decode_qualified=False, task_agent_qualified=False, physical262144_RESTORE_qualified=False,
        hardware_saturation_qualified=False, inference_adopted=False,
        next='Compare the selected policy with retained56cad9 under matched32K/64Ksettings,and independently preserve sustained decode against a347. Inconclusive small gains remain eligible for a newly committed repetition plan;keep every original sample.')


def context():
    spec = importlib.util.spec_from_file_location('compact_gu_prefill_comparison', core.__file__)
    ctx = importlib.util.module_from_spec(spec); spec.loader.exec_module(ctx)
    ctx.ROOT, ctx.KEY, ctx.THIS = ROOT, KEY, THIS
    ctx.READS, ctx.ORDER, ctx.RUNS = tuple(r for r,_ in CASES), [(r,1) for r,_ in CASES], RUNS
    ctx.CLEAN_FAST, ctx.basis, ctx.decision = True, basis, decision
    old_env, old_current, old_route = ctx.environment, ctx.current, ctx.request_route
    def environment(row, plan):
        require(0 <= plan['index'] < len(CASES) and plan['read'] == CASES[plan['index']][0], 'planned arm only')
        env = old_env(row, plan)
        env.update(STRATA_SYCL_PREFILL_QSA_MATH='bf16-prefix-ordered-exp',
            STRATA_SYCL_PREFILL_QSA_BULK_HOST_MIRROR='1', STRATA_SYCL_PREFILL_EXPERT_UPLOAD_STAGING='0',
            STRATA_SYCL_PREFILL_MOE_CENSUS='0', STRATA_PREFILL_LAYER_MAJOR_PROFILE='0',
            STRATA_SYCL_PREFILL_GU_SWIGLU=POLICIES[CASES[plan['index']][1]])
        return env
    def current(row):
        require(row['arm_order'] == [list(v) for v in CASES] and len(row['plans']) == len(CASES)
            and all(p['gu_policy'] == POLICIES[CASES[i][1]] for i,p in enumerate(row['plans'])),
            'frozen policy/order changed')
        return old_current(row)
    def request_route(text, plan):
        route = old_route(text, plan)
        policy = POLICIES[CASES[plan['index']][1]]
        route['actual_GU_dispatch'] = dispatch(text, policy)
        require('softmax=parallel-exp-ascending-sum' in text and 'softmax=subgroup' not in text,
            'unchanged ordered-exp attention required')
        return route
    ctx.environment, ctx.current, ctx.request_route = environment, current, request_route
    def check_sample(sample, count, read, minimum_ms=10000):
        core.comparison.check_sample(sample, count, read, minimum_ms)
        require(sample['generated_tokens'] == 24, 'exactly24outputs required;short completion is ineligible')
    ctx.check_sample = check_sample
    return ctx


def prepare():
    ctx = context(); ctx.prepare()
    row = json.loads((ROOT/'record.json').read_text()); row['arm_order'] = [list(v) for v in CASES]
    for i,p in enumerate(row['plans']): p['gu_policy'] = POLICIES[CASES[i][1]]
    row['whole_bytes'] = 1536 << 20
    row['plan'] = dict(question='Does compact1..96 GU fusion improve whole long-prefill over existing large-only fusion in the same dcb8 engine?',
        order='12fresh processes:32K ABBAAB then64K ABBAAB. Three independent processes per arm/length,three exact-input thinking-on bench.sh requests each,24outputs. First use separate from subsequent2;retain all36samples and spreads. Commit each closed process before advancing.',
        settings='Same dcb8 ELF,model/provider/safety,logical262144/resident20480INT8,8192chunk,FAST9,ordered-exp/bulk1/staging0/head2/dense1/backing0/MTPFP16. Only pre-exec GUlarge/auto and private paths differ;no runtime environment mutation.',
        acceptance='Current frontend CPU controls then exact source/ELF/provider/assets,normal SSE,read/reuse/output/thinking/clocks/TTFT,actual placement/head/MTP handoff and existing GU/Down branch aggregates. Same wide eligibility,positive observed compact delta only inauto,job conservation,normal0/reaped/empty/quiet validation/no new fault. Require exactly24outputs in every eligible sample.',
        comparison='All3paired subsequent-request prompt-time ratios favorauto at BOTH lengths for repeated-gain advancement,without a minimum effect threshold. PP,clientTTFT,first use and strategy tradeoffs remain separately reported. Preserve mixed/no-gain outcomes;additional repetitions require a new plan.',
        stopping='Stop first input/provider/owner/token/clock/route/validation/fault/budget failure. No retry,pin refresh,speed exclusion or criteria change. Reuse exact closed-model final15full asset hashes after identity checks;hash all15again after the complete cohort.',
        limits='Parameter validation remains charged;not observer-free. Prefill-only comparison,not sustainedTG,task/agent,physical262144/RESTORE,ordinary teardown,resource saturation,whole-goal or default adoption. Two lengths give adjacent slopes,not a fitted physical law.',
        retention='Keep compact individual samples/paired decisions and dispatch counts;retire successful logs/journals/caches after verification. No new engine. Next compare with fastest56cad9 and independently preserve a347decode;current97fda remains a concrete build-proof reader dependency until prospectively migrated.')
    ctx.save(row); print('Prepared12process compact-GU comparison;commit before host/native.', flush=True)


if __name__ == '__main__':
    require(sys.executable == model.PYTHON, 'Use pinned benchmark interpreter: '+model.PYTHON)
    require(len(sys.argv) == 2 and (sys.argv[1] in ('prepare','host')
        or sys.argv[1].isdigit() and int(sys.argv[1]) < len(CASES)), 'prepare|host|0..11')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); model.original.guards.no_model_owners([])
        if sys.argv[1] == 'prepare': prepare()
        elif sys.argv[1] == 'host': core.comparison.host(context())
        else: core.comparison.run(int(sys.argv[1]), context())
