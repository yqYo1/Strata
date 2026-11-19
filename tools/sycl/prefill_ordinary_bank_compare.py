"""Compare the retained 64K reference and the two attention stores at 384 expert slots."""

import fcntl
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
import coherent_prefill_bench as original
import prefill_attention_bank_capacity as capacity
import prefill_attention_bank_measure as store

build, basis, require = capacity.build, capacity.basis, capacity.require
READ = 65537
ARMS = ('reference', 'copy', 'direct', 'copy', 'direct', 'reference', 'direct', 'reference', 'copy')
THIS = 'tools/sycl/prefill_ordinary_bank_compare.py'


def cohort_basis():
    candidate, speed = capacity.retained_build()
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


def native_context(row, plan):
    index = plan['index']
    require(0 <= index < len(ARMS) and (plan['read'], plan['heads']) == (READ, 1),
            'unexpected process or request size')
    identity = {key: row[key] for key in ('binary', 'native_source_commit', 'boot', 'native_validation_probe')}
    if ARMS[index] == 'reference':
        admission = row['observer_admission']
        identity.update(binary=admission['historical_binary'], native_source_commit=admission['historical_source'])
    return identity


def check_work(outcome, row, _plan):
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


def context():
    historical = original.context()
    ctx = original.core.source.device.load_module('ordinary_bank_comparison', original.core.__file__)
    ctx.ROOT = basis.B / 'prefill-ordinary-bank-comparison-v1'
    ctx.KEY, ctx.THIS = 'prefill_ordinary_bank_comparison_v1', THIS
    ctx.READS = (READ,) * len(ARMS)
    ctx.ORDER = [(READ, 1)] * len(ARMS)
    ctx.CLEAN_FAST, ctx.RUNS, ctx.check_sample = True, historical.RUNS, historical.check_sample
    ctx.basis, ctx.native_context, ctx.after_execution, ctx.decision = cohort_basis, native_context, check_work, decision
    base_current = ctx.current

    def environment(row, plan):
        env = historical.environment(row, plan)
        if ARMS[plan['index']] != 'reference':
            env[store.SELECTOR] = str(int(ARMS[plan['index']] == 'direct'))
        return env

    def route(text, plan):
        result = historical.request_route(text, plan)
        arm = ARMS[plan['index']]
        if arm != 'reference':
            result['attention_store'] = store.check_store(text, dict(plan, variant=arm))
        return result

    def current(row):
        require(row['read_order'] == list(ctx.READS) and [p['variant'] for p in row['plans']] == list(ARMS),
                'frozen triplet order differs')
        for p in row['plans']:
            expected = native_context(row, p)
            require(build.read(Path(p['root']) / 'plan.json') ==
                    {k: expected[k] for k in ('binary', 'native_validation_probe')}, 'native plan differs')
        return base_current(row)

    ctx.environment, ctx.request_route, ctx.current = environment, route, current
    return ctx


def prepare_plans(ctx, row):
    speed = build.read(original.ROOT / 'record.json')
    original_plan = next(p for p in speed['plans'] if p['read'] == READ)
    original_env = original.context().environment(speed, original_plan)
    for p, arm in zip(row['plans'], ARMS, strict=True):
        p['variant'] = arm
        env = ctx.environment(row, p)
        env.pop(store.SELECTOR, None)
        require({k: v.replace(p['root'], original_plan['root']) for k, v in env.items()} == original_env
                and row['configuration']['args'] == speed['configuration']['args'],
                'frozen comparison must reproduce the original 64K environment and arguments')
        path = Path(p['root']) / 'plan.json'
        identity = native_context(row, p)
        ctx.write(path, {k: identity[k] for k in ('binary', 'native_validation_probe')})
        p['generated']['plan.json'] = basis.sha(path)
    row.update(read_order=list(ctx.READS), whole_bytes=len(ARMS) * row['budgets']['total_cap'] + (8 << 20), plan=dict(
        question='Measure the direct BF16 attention store on the ordinary 384-slot 64K prefill path.',
        order=list(ARMS), runs_per_process=ctx.RUNS, historical_source=speed['native_source_commit'],
        acceptance='Compare first and subsequent requests separately. Require matching input bodies, actual '
                   '384-slot route, jobs, H2D requests and MTP counts. A repeatable store gain requires lower '
                   'warmed prompt time in all three direct/copy comparisons. Compare the retained historical '
                   'ELF separately as a whole-engine speed reference. Record TTFT and the 1000 target separately.',
        stopping='Stop on the first ownership, workload, route, validation, fault or budget failure.'))


if __name__ == '__main__':
    require(sys.executable == original.model.PYTHON, 'use pinned benchmark interpreter')
    action, = sys.argv[1:]
    with (basis.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        build.builder.guards.no_model_owners([])
        experiment = context()
        if action == 'prepare':
            publish = experiment.save
            def publish_prepared(row):
                prepare_plans(experiment, row)
                publish(row)
            experiment.save = publish_prepared
            experiment.prepare()
        elif action == 'host':
            basis.comparison.host(experiment)
        else:
            basis.comparison.run(int(action), experiment)
