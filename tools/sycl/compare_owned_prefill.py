#!/usr/bin/env python3
"""Compare two retained owned prefill engines; repository LICENSE.

The older executable is qualified from its original committed inputs and closed
model evidence. Its obsolete preserved executables are not runtime dependencies.
No native code, numerical contract, provider or decode setting is changed here.
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
import build_engine as source
import build_qsa_bulk_model as older
import gu_small_bench as gu

P, B, RESULT = gu.P, gu.B, gu.RESULT
ROOT, KEY = B/'owned-prefill-pair-v2', 'owned_prefill_pair_v2'
THIS = 'tools/sycl/compare_owned_prefill.py'
OLD_SHA = '56cad9eb45f8f5fc9158eef968504424ac83a47e75cef7ad1bfa1ea8c436a93d'
OLD_SOURCE = '611a0fd6fecc4593e17160c294103cd4464a292b'
OLD_SPEED_KEY = 'qsa16_bulk_mirror_prefill_compare_v1'
RUNS = 3
ARMS = ('current', 'retained', 'retained', 'current')
CASES = tuple((read, arm) for read in (32769, 65537) for arm in ARMS)
require, sha = gu.require, gu.sha


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def recover_project(pin):
    root = Path(pin['root'])
    actual = {name: hashlib.sha256(raw).hexdigest() for name, raw in
              source.blobs(root, pin['revision'], pin['source_sha256'])}
    require(actual == pin['source_sha256'] and len(actual) == pin['count']
        and signature(actual) == pin['aggregate_sha256'], 'original Git source/aggregate differs')
    return {str(root/name): value for name, value in actual.items()}


def retained_reference(providers, prior):
    private = json.loads((older.ROOT/'record.json').read_text())
    committed = gu.model.original.validation.committed(RESULT, (older.KEY,))
    require(older.context().compact(private) == committed and private['stage'] == 'built'
        and private['passed'] and private['complete'] and private['ownership_closed']
        and not private['active'] and private['native_source_commit'] == OLD_SOURCE
        and private['project']['revision'] == OLD_SOURCE
        and private['binary']['sha256'] == OLD_SHA, 'retained original build identity differs')
    native = private['binary']
    require(sha(native['path']) == native['sha256']
        and Path(native['path']).stat().st_size == native['bytes'], 'retained executable changed')
    recovered = recover_project(private['project'])
    roots = [private['project']['root']]
    for dependency in private['dependencies']:
        recovered.update(recover_project(dependency)); roots.append(dependency['root'])
    descriptor = private['actual_inputs']
    require(descriptor['source_commit'] == OLD_SOURCE and descriptor['uncovered'] == 0
        and descriptor['translation_units'] == 128
        and sha(descriptor['path']) == descriptor['sha256'], 'original compiler-input coverage differs')
    actual = json.loads(Path(descriptor['path']).read_text())
    require(len(actual['compile_commands']) == 128
        and len(actual['compiled']) == descriptor['compiler_files']
        and len(actual['generated']) == descriptor['generated_files']
        and len(actual['links']) == descriptor['external_link_files'], 'compiler inventory changed')
    # The original broad dependency prefix did not include ggml/'s nested
    # sources. Recover every such compiler input from that same immutable Git
    # revision; a current file hash alone is insufficient source provenance.
    additional = 0
    for pin in (private['project'], *private['dependencies']):
        root = Path(pin['root'])
        missing = {str(Path(p).relative_to(root)): value
            for p,value in actual['compiled'].items()
            if p not in recovered and Path(p).is_relative_to(root)}
        original = {p: hashlib.sha256(raw).hexdigest() for p,raw in
            source.blobs(root,pin['revision'],missing)}
        require(original == missing, 'additional compiler source does not match original Git revision')
        recovered.update({str(root/p): value for p,value in original.items()})
        additional += len(original)
    for name, value in actual['compiled'].items():
        if name in recovered:
            require(recovered[name] == value, 'compiled source does not match recovered Git input')
        else:
            require(not any(Path(name).is_relative_to(root) for root in roots),
                'compiled project/dependency input absent from original source inventory')
            require(sha(name) == value, 'external compiler input changed')
    for name, value in (actual['generated'] | actual['links'] | private['toolchain']).items():
        require(sha(name) == value, 'original generated/SDK/link/tool input changed')
    for name, value in (private['generated_inputs'] | private['configured_files']).items():
        require(sha(name) == value, 'original build configuration changed')
    external = private['external_inputs']
    require(sha(external['path']) == external['sha256'], 'original SDK manifest changed')
    require(all(sha(p) == h for p,h in json.loads(Path(external['path']).read_text()).items()),
        'original SDK/runtime input differs')
    speed = gu.model.original.validation.committed(RESULT, (OLD_SPEED_KEY,))
    require(speed['stage'] == 'closed' and speed['passed'] and speed['complete']
        and speed['ownership_closed'] and not speed['active']
        and speed['binary'] == native and speed['native_source_commit'] == OLD_SOURCE
        and speed['boot'] == prior['boot'] and speed['driver'] == prior['driver']
        and speed['native_validation_probe'] == prior['probe']
        and speed['model_assets'] == prior['models'] and len(speed['processes']) == 8
        and all(p['passed'] and p['complete'] and p['ownership_closed']
            and p['native_exit_code'] == 0 and not p['fresh_faults'] for p in speed['processes']),
        'closed retained model/runtime/workload qualification differs')
    runtime = speed['runtime_manifest']
    require(sha(runtime['path']) == runtime['sha256']
        and json.loads(Path(runtime['path']).read_text()) == providers,
        'both engines must use the same qualified providers')
    return dict(binary=native, native_source_commit=OLD_SOURCE,
        original_build_evidence_sha256=signature(committed),
        original_model_evidence_sha256=signature(speed),
        original_project_files=private['project']['count'],
        recovered_source_files=len(recovered), additional_compiler_sources_recovered=additional,
        compiler_files=len(actual['compiled']),
        original_recipe_sha256=signature(private['recipe']),
        inherited_fixture_or_task_qualification=False,
        source_recovery='Original Git project/dependency blobs,compiler-input coverage,recipe and live SDK/generated inputs;not current-worktree source equivalence or executable hash alone.')


def basis():
    native, prior, providers, _ = gu.basis(extra_observers={THIS})
    retained = retained_reference(providers, prior)
    cohort = gu.model.original.validation.committed(RESULT, (gu.KEY,))
    require(cohort['stage'] == 'closed' and cohort['passed'] and cohort['complete']
        and cohort['ownership_closed'] and not cohort['active'] and len(cohort['processes']) == 12
        and not cohort['decision']['advance_to_further_model_gates']
        and cohort['decision']['mechanism_evidence'] == 'not_consistently_faster'
        and cohort['binary'] == native['binaries']['strata'],
        'preserve failed compact-gain decision;only existing GUlarge is eligible here')
    prior = copy.deepcopy(prior)
    prior['observer_admission'].update(retained_reference=retained,
        closed_current_cohort_sha256=signature(cohort),
        comparison_scope='Retained best-owned routes with GUlarge;not another compact-GU trial or adoption.')
    return native, prior, providers, None


def decision(processes):
    require(len(processes) == len(CASES) and all(p['passed'] and p['complete']
        and p['ownership_closed'] and p['native_exit_code'] == 0 and not p['fresh_faults']
        and len(p['samples']) == RUNS for p in processes), 'all8processes/24requests required')
    points = []
    for offset, read in ((0,32769), (4,65537)):
        local = processes[offset:offset+4]; arms = {}
        for arm in ('current','retained'):
            selected = [p for i,p in enumerate(local) if ARMS[i] == arm]
            times = [p['subsequent_median_ms'] for p in selected]
            first = [p['samples'][0]['prompt_ms'] for p in selected]
            arms[arm] = dict(subsequent_prompt_ms=statistics.median(times),
                subsequent_PP=read*1000/statistics.median(times),
                subsequent_process_PP=[read*1000/v for v in times],
                subsequent_client_TTFT_s=statistics.median(statistics.median(s['ttft_s']
                    for s in p['samples'][1:]) for p in selected),
                first_PP=read*1000/statistics.median(first),
                first_client_TTFT_s=statistics.median(p['samples'][0]['ttft_s'] for p in selected))
        ratios = [local[a]['subsequent_median_ms']/local[b]['subsequent_median_ms']
            for a,b in ((0,1),(3,2))]
        points.append(dict(read=read, **arms, current_over_retained_time_ratios=ratios,
            all_pairs_favor_current=all(v<1 for v in ratios), all_pairs_favor_retained=all(v>1 for v in ratios)))
    current = all(p['all_pairs_favor_current'] for p in points)
    retained = all(p['all_pairs_favor_retained'] for p in points)
    return dict(points=points, development_reference='current' if current else 'retained' if retained else 'inconclusive_keep_both_until_next_bounded_decision',
        matched_workloads=True, same_binary=False, GUcompact=False,
        parameter_validation_charged=True, observer_free=False,
        inference_adopted=False, sustained_decode_qualified=False,
        physical262144_RESTORE_qualified=False, hardware_saturation_qualified=False,
        next='Preserve independent a347decode;qualify physical full context/task/agent. Implement selected-expert run coalescing separately only after source-contract and exact-byte/slot/lifetime tests.')


def context():
    spec = importlib.util.spec_from_file_location('owned_prefill_pair', gu.core.__file__)
    ctx = importlib.util.module_from_spec(spec); spec.loader.exec_module(ctx)
    # Reuse the production request/owner/clock/provider checks, with a new plan.
    ctx.ROOT, ctx.KEY, ctx.THIS = ROOT, KEY, THIS
    ctx.READS, ctx.ORDER, ctx.RUNS = tuple(r for r,_ in CASES), [(r,1) for r,_ in CASES], RUNS
    ctx.CLEAN_FAST, ctx.basis, ctx.decision = True, basis, decision
    base_current = ctx.current
    base_environment = ctx.environment
    base_route = ctx.request_route
    ctx.check_sample = gu.context().check_sample
    def current(row):
        require(row['arm_order'] == [list(v) for v in CASES]
            and len(row['plans']) == len(CASES), 'frozen retained/current order changed')
        for plan in row['plans']:
            expected = native_context(row, plan)
            actual = json.loads((Path(plan['root'])/'plan.json').read_text())
            require(actual == dict(binary=expected['binary'],
                native_validation_probe=expected['native_validation_probe']),
                'actual engine_once launch plan must select the declared executable')
        return base_current(row)
    def native_context(row, plan):
        read, arm = CASES[plan['index']]
        require(plan['read'] == read and plan.get('arm',arm) == arm, 'unplanned engine arm')
        if arm == 'retained':
            item = row['observer_admission']['retained_reference']
            binary, revision = item['binary'], item['native_source_commit']
        else:
            binary, revision = row['binary'], row['native_source_commit']
        return dict(binary=binary,native_source_commit=revision,
            native_validation_probe=row['native_validation_probe'],boot=row['boot'])
    def environment(row, plan):
        env = base_environment(row, plan)
        env.update(STRATA_SYCL_PREFILL_QSA_MATH='bf16-prefix-ordered-exp',
            STRATA_SYCL_PREFILL_QSA_BULK_HOST_MIRROR='1',STRATA_SYCL_PREFILL_EXPERT_UPLOAD_STAGING='0',
            STRATA_SYCL_PREFILL_MOE_CENSUS='0',STRATA_PREFILL_LAYER_MAJOR_PROFILE='0',
            STRATA_SYCL_PREFILL_GU_SWIGLU='large')
        return env
    def request_route(text, plan):
        route = base_route(text, plan)
        route['actual_GU_dispatch'] = gu.dispatch(text, 'large')
        require('softmax=parallel-exp-ascending-sum' in text and 'softmax=subgroup' not in text,
            'same ordered-exp attention required')
        return route
    ctx.current, ctx.native_context, ctx.environment, ctx.request_route = current,native_context,environment,request_route
    return ctx


def prepare():
    ctx = context(); ctx.prepare()
    row = json.loads((ROOT/'record.json').read_text()); row['arm_order'] = [list(v) for v in CASES]
    for i,p in enumerate(row['plans']):
        p['arm'] = CASES[i][1]
        selected = ctx.native_context(row,p)
        ctx.write(Path(p['root'])/'plan.json',dict(binary=selected['binary'],
            native_validation_probe=selected['native_validation_probe']))
        p['generated']['plan.json'] = sha(Path(p['root'])/'plan.json')
    row['whole_bytes'] = 1536 << 20
    row['previous_preparation'] = dict(key='owned_prefill_pair_v1',
        source_commit='df5bef02752b919ce3c0b787e47a0e7c6c7375ec',
        status='refused_before_campaign_creation_or_native_execution',
        reason='52compiled ggml inputs were outside the original declared dependency prefix;V2recovers them from the same pinned Git revision and requires their original compiler hashes. No recorded hash or comparison criterion is changed.')
    row['plan'] = dict(question='Which retained owned GUlarge engine is faster under matched32K/64K settings?',
        order='8fresh processes:current/retained/retained/current at32K then64K;3thinking-on exact24-output bench.sh requests/process. Separate first measured long request from subsequent2. Commit each closed process before advancing.',
        settings='Same model,providers,safety,logical262144/resident20480K8V8,8192chunk,FAST9,ordered-exp/bulk1,384slots/source_view1/private-map,upload_staging0,head2/dense1/backing0/MTPFP16. Only native executable/source identity and private paths differ. GUlarge in both arms;no environment mutation.',
        acceptance='Exact committed control/generated inputs,original source/recipe/ELF/provider/model proof,normal SSE/read/reuse/thinking/output24/clocks/TTFT,actual typed placement and completed GUwide-only counts,normal0/reaped/empty/quiet validation/no fresh fault. All24requests and final15asset hashes required.',
        comparison='Choose development reference only if both paired warmed time ratios favor the same engine at BOTH lengths;no minimum percentage threshold. Keep first-use/PP/TTFT tradeoffs distinct. Mixed evidence stays inconclusive with all original samples;further repetitions need a new committed plan.',
        stopping='Stop first source/provider/token/clock/route/owner/validation/fault/budget failure;no retry,pin refresh,excluded sample or changed gate.',
        limits='Reference selection only;no compact-GU adoption or new precision. Parameter validation/preload charged,not observer-free. No sustainedTG,task/agent,physical262144/RESTORE,ordinary teardown or saturation qualification.',
        retention='Use existing two executables;no new native build. Retire successful raw output after extraction;select the performance reference then retire superseded build only after exact source/recipe recovery and reader-consumer migration.')
    ctx.save(row); print('Prepared8process retained/current comparison;commit before host/native.',flush=True)


if __name__ == '__main__':
    require(sys.executable == gu.model.PYTHON, 'Use pinned benchmark interpreter: '+gu.model.PYTHON)
    require(len(sys.argv)==2 and (sys.argv[1] in ('prepare','host')
        or sys.argv[1].isdigit() and int(sys.argv[1])<len(CASES)), 'prepare|host|0..7')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);gu.model.original.guards.no_model_owners([])
        if sys.argv[1]=='prepare':prepare()
        elif sys.argv[1]=='host':gu.core.comparison.host(context())
        else:gu.core.comparison.run(int(sys.argv[1]),context())
