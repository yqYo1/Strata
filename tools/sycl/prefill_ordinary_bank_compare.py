"""Compare the retained 64K reference and the two attention stores at 384 expert slots."""

import fcntl
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
import coherent_prefill_bench as original
import prefill_attention_bank_capacity as capacity
import prefill_attention_bank_measure as store
import prefill_attention_bank_restore as restore

build, basis, require = capacity.build, capacity.basis, capacity.require
READ = 65537
ARMS = ('reference', 'copy', 'direct', 'copy', 'direct', 'reference', 'direct', 'reference', 'copy')
THIS = 'tools/sycl/prefill_ordinary_bank_compare.py'
SMOKE = 'prefill_ordinary_bank_device_after_stall_v1'
FAILED_SUITE = 'prefill_attention_bank_generation_v2'
RAGGED = 'prefill_ordinary_bank_ragged_v1'


def device_smoke():
    failure = basis.validation.committed(basis.RESULT, (FAILED_SUITE,))
    require(failure['stage'] == 'failed' and failure['ownership_closed'] and not failure['active']
            and len(failure['processes']) == 1 and failure['processes'][0]['fresh_faults'],
            'closed faulted process required')
    root = basis.B / 'ordinary-bank-device-after-stall-v1'
    root.mkdir(mode=0o700)
    store.stage.schedule.bank_smoke(controller=THIS, key=SMOKE, arg='--fullfit-eager-smoke',
        marker='prefill_fullfit_eager_fixture PASS model=UNRUN physical262144_RESTORE=UNQUALIFIED',
        expected=[(4, 256, 2, 0, 4), (4, 512, 2, 2, 4)],
        build_root=root, native_reader=capacity.retained_build)


def cohort_basis():
    candidate, speed = capacity.retained_build()
    smoke = basis.validation.committed(basis.RESULT, (SMOKE,))
    require(smoke['passed'] and smoke['ownership_closed'] and not smoke['fresh_faults']
            and smoke['native_exit_code'] == 0 and smoke['binary'] == candidate['binaries']['sycl_expert_bank_test'],
            'same-build isolated traced GPU check required')
    prior = build.read(original.model.ROOT / 'record.json')
    require(prior['models'] == speed['model_assets'] and prior['probe'] == speed['native_validation_probe'],
            'historical configuration or model assets differ')
    historical = original.context()
    plans = [plan for plan in speed['plans'] if plan['read'] == READ]
    require(len(plans) == 3 and all(basis.comparison.fingerprint(historical.environment(speed, p))
            == p['environment_sha256'] for p in plans), 'original 64K environment differs')
    point, = (p for p in speed['decision']['points'] if p['read'] == READ)
    require(len(point['subsequent_process_PP']) == 3 and min(point['subsequent_process_PP']) >= 1000,
            'retained repeated 64K reference required')
    prior['observer_admission'] = dict(
        fast_configuration_diagnostic=speed['observer_admission']['fast_configuration_diagnostic'],
        historical_binary=speed['binary'], historical_source=speed['native_source_commit'],
        historical_requests=speed['request_hashes'][str(READ)], candidate_build=capacity.BUILD_EVIDENCE)
    native = dict(binaries=candidate['binaries'], native_source_commit=candidate['source_commit'])
    return native, prior, build.read(Path(speed['runtime_manifest']['path'])), None


def native_context(row, plan, read, arms):
    index = plan['index']
    require(0 <= index < len(arms) and (plan['read'], plan['heads']) == (read, 1),
            'unexpected process or request size')
    identity = {key: row[key] for key in ('binary', 'native_source_commit', 'boot', 'native_validation_probe')}
    if arms[index] == 'reference':
        admission = row['observer_admission']
        identity.update(binary=admission['historical_binary'], native_source_commit=admission['historical_source'])
    return identity


def check_work(outcome, row, plan):
    if plan['read'] == READ:
        require([s['request_sha256'] for s in outcome['samples']] == row['observer_admission']['historical_requests'],
                'actual requests differ from historical 64K inputs')
    def counts(process):
        return [(route['actual_GU_dispatch'], route['fast_window'], sample['drafts_offered'], sample['drafts_accepted'])
                for route, sample in zip(process['routes'], process['samples'], strict=True)]
    if row['processes']:
        require(counts(outcome) == counts(row['processes'][0]), 'MoE work, transfer requests or MTP work differ')


def decision(processes):
    require(len(processes) == len(ARMS) and all(p['passed'] for p in processes), 'complete three triplets first')
    triplets = []
    speeds = {arm: [] for arm in ARMS[:3]}
    for offset in range(0, len(ARMS), 3):
        times = {ARMS[i]: store.stage.schedule.timing(processes[i]) for i in range(offset, offset + 3)}
        for arm, timing in times.items():
            speeds[arm].append(READ * 1000 / timing['warm_prompt_ms'])
        triplets.append(dict(times=times,
            direct_over_copy=times['direct']['warm_prompt_ms'] / times['copy']['warm_prompt_ms'],
            direct_over_reference=times['direct']['warm_prompt_ms'] / times['reference']['warm_prompt_ms']))
    return dict(triplets=triplets, warmed_process_PP=speeds,
                warmed_median_PP={arm: statistics.median(values) for arm, values in speeds.items()},
                store_improves_every_triplet=all(t['direct_over_copy'] < 1 for t in triplets),
                direct_beats_reference_every_triplet=all(t['direct_over_reference'] < 1 for t in triplets),
                repeated_1000={arm: min(values) >= 1000 for arm, values in speeds.items()})


def context(ragged):
    historical = original.context()
    ctx = original.core.source.device.load_module('ordinary_bank_comparison', original.core.__file__)
    ctx.KEY = RAGGED if ragged else 'prefill_ordinary_bank_comparison_v1'
    ctx.ROOT, ctx.THIS = basis.B / ctx.KEY.replace('_', '-'), THIS
    read, arms = (16098, ARMS[:3]) if ragged else (READ, ARMS)
    ctx.READS, ctx.ORDER = (read,) * len(arms), [(read, 1)] * len(arms)
    ctx.CLEAN_FAST, ctx.RUNS, ctx.check_sample = True, (2 if ragged else historical.RUNS), historical.check_sample
    ctx.MIN_PROMPT_MS = 0 if ragged else 10000
    ctx.basis, ctx.after_execution = cohort_basis, check_work
    ctx.native_context = lambda row, p: native_context(row, p, read, arms)
    ctx.decision = (lambda processes: dict(completed_arms=list(arms), input_tokens=read,
                    all_passed=all(p['passed'] for p in processes), speed_qualified=False)) if ragged else decision
    base_current = ctx.current

    def environment(row, plan):
        env = historical.environment(row, plan)
        if ragged:
            env['STRATA_TRACE'] = '1'
        if arms[plan['index']] != 'reference':
            env[store.SELECTOR] = str(int(arms[plan['index']] == 'direct'))
        return env

    def route(text, plan):
        result = historical.request_route(text, plan)
        arm = arms[plan['index']]
        require('strata MoE prefix expert upload layer:' not in text and
                'strata MoE prefix router: requested=early' not in text, 'unexpected early expert upload')
        if arm != 'reference':
            result['attention_store'] = restore.require_store(text, dict(direct=arm == 'direct'), plan)
        return result

    def current(row):
        require(row['read_order'] == list(ctx.READS) and [p['variant'] for p in row['plans']] == list(arms),
                'frozen triplet order differs')
        for p in row['plans']:
            expected = ctx.native_context(row, p)
            require(build.read(Path(p['root']) / 'plan.json') ==
                    {k: expected[k] for k in ('binary', 'native_validation_probe')}, 'native plan differs')
        return base_current(row)

    ctx.environment, ctx.request_route, ctx.current = environment, route, current
    return ctx


def prepare_plans(ctx, row, ragged):
    speed = build.read(original.ROOT / 'record.json')
    original_plan = next(p for p in speed['plans'] if p['read'] == READ)
    original_env = original.context().environment(speed, original_plan)
    arms = ARMS[:3] if ragged else ARMS
    for p, arm in zip(row['plans'], arms, strict=True):
        p['variant'] = arm
        env = ctx.environment(row, p)
        env.pop(store.SELECTOR, None)
        if ragged:
            require(env.pop('STRATA_TRACE') == '1', 'ragged diagnostic needs layer progress')
            p['trace_progress'] = True
            p['command'].append('--prefill-short-diagnostic')
        require({k: v.replace(p['root'], original_plan['root']) for k, v in env.items()} == original_env
                and row['configuration']['args'] == speed['configuration']['args'],
                'frozen comparison must reproduce the original 64K environment and arguments')
        path = Path(p['root']) / 'plan.json'
        identity = ctx.native_context(row, p)
        ctx.write(path, {k: identity[k] for k in ('binary', 'native_validation_probe')})
        p['generated']['plan.json'] = basis.sha(path)
    row.update(read_order=list(ctx.READS), whole_bytes=len(arms) * row['budgets']['total_cap'] + (8 << 20), plan=dict(
        question='Measure the direct BF16 attention store on the ordinary 384-slot 64K prefill path.',
        order=list(arms), runs_per_process=ctx.RUNS, historical_source=speed['native_source_commit'],
        acceptance='Compare first and subsequent requests separately. Require matching input bodies, actual '
                   '384-slot route, jobs, H2D requests and MTP counts. A repeatable store gain requires lower '
                   'warmed prompt time in all three direct/copy comparisons. Compare the retained historical '
                   'ELF separately as a whole-engine speed reference. Record TTFT and the 1000 target separately.',
        stopping='Stop on the first ownership, workload, route, validation, fault or budget failure.'))
    if ragged:
        row['plan'].update(question='Check ordinary same-queue prefill at the previous 16098-token stall shape.',
            acceptance='Two fresh exact requests in each arm with trace and parameter validation. Check the '
                       '8192+7905 batched rows, final token, 24 generated tokens, native closure and fresh faults.')


if __name__ == '__main__':
    require(sys.executable == original.model.PYTHON, 'use pinned benchmark interpreter')
    modes = sys.argv[1:]
    ragged = modes[:1] == ['ragged']
    action, = modes[1:] if ragged else modes
    with (basis.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        build.builder.guards.no_model_owners([])
        if action == 'device-smoke':
            device_smoke()
            sys.exit(0)
        if not ragged:
            gate = basis.validation.committed(basis.RESULT, (RAGGED,))
            require(gate['stage'] == 'closed' and gate['passed'] and gate['complete']
                    and gate['ownership_closed'] and not gate['active'], 'closed 384-slot ragged gate required')
        experiment = context(ragged)
        if action == 'prepare':
            publish = experiment.save
            def publish_prepared(row):
                prepare_plans(experiment, row, ragged)
                publish(row)
            experiment.save = publish_prepared
            experiment.prepare()
        elif action == 'host':
            basis.comparison.host(experiment)
        else:
            basis.comparison.run(int(action), experiment)
