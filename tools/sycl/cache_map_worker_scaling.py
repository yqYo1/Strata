#!/usr/bin/env python3
"""Same-engine long-decode CPU worker scaling; repository LICENSE.

Main-authored policy over the maintained official benchmark supervisor. Keep
the selected native generation, math, cache membership and GPU safety fixed.
The three diagnostic processes establish configuration sensitivity, not clean
speed, physical-memory bandwidth or an adopted inference configuration.
"""
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import cache_map_decode_census as census

short, entry, base = census.short, census.entry, census.base
maintained, gen = census.maintained, census.gen
P, B, RESULT = census.P, census.B, census.RESULT
require, digest = census.require, census.digest
ROOT = entry.model.ROOT / 'worker-scaling-v1'
KEY = 'cache_map_worker_scaling_v1'
SOURCE = Path(__file__).resolve()
WORKERS = (5, 3, 9)


def cpu_resources():
    allowed = sorted(os.sched_getaffinity(0))
    topology = {}
    for cpu in allowed:
        root = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        topology[str(cpu)] = [(root / n).read_text().strip()
                             for n in ('physical_package_id', 'core_id')]
    line, = Path('/proc/self/cgroup').read_text().splitlines()
    require(line.startswith('0::/'), 'this plan requires the observed cgroup-v2 hierarchy')
    root = Path('/sys/fs/cgroup')
    current = root / line[3:].lstrip('/')
    quotas = {}
    while current.is_relative_to(root):
        quotas[str(current)] = {n: (current / n).read_text().strip()
                               for n in ('cpu.max', 'cpuset.cpus.effective')
                               if (current / n).exists()}
        if current == root:
            break
        current = current.parent
    return dict(allowed_cpus=allowed, package_core_ids=topology, cgroup_ancestors=quotas)


def pinned_sources(revision):
    pins = census.pinned_sources(revision)
    pins.update(base.controls(revision, controller=SOURCE))
    return pins


def basis(revision):
    closed = json.loads((census.ROOT / 'record.json').read_text())
    require(base.committed(RESULT)[census.KEY] == maintained.compact(closed)
            and closed['stage'] == 'closed' and closed['passed']
            and closed['complete'] and closed['ownership_closed']
            and not closed['active'] and closed['census']['passed']
            and closed['census']['completed_requests_verified'] == 6
            and closed['generation_route_check_passed'],
            'closed same-engine representative long CPU census required')
    original = {n: hashlib.sha256(raw).hexdigest() for n, raw in
                gen.blobs(P, closed['source_commit'], closed['sources'])}
    current = pinned_sources(revision)
    require(original == closed['sources']
            and all(current.get(n) == sha for n, sha in original.items())
            and set(current) - set(original) == {str(SOURCE.relative_to(P))},
            'only this new worker policy may extend qualified inputs')
    prior = json.loads((short.ROOT / 'record.json').read_text())
    providers, manifest = short.context().current(prior)
    require(prior['stage'] == 'closed' and prior['passed']
            and prior['complete'] and prior['ownership_closed']
            and not prior['active'] and prior['binary'] == closed['binary']
            and prior['native_source_commit'] == closed['native_source_commit']
            and prior['census']['passed'] and prior['model_route_check_passed'],
            'unchanged selected-engine short qualification required')
    pool = closed['compiled_pool_sources']
    require(all(digest(entry.model.ROOT / 'source' / n) == sha
                for n, sha in pool.items()), 'compiled pool source differs')
    return prior, providers, manifest, pool


def environment(row, plan):
    require(plan['kernel'] == 'reference' and plan['adapt_swaps'] == 0
            and plan['heads'] == 1 and 0 <= plan['index'] < len(WORKERS),
            'only the static-cache selected native path is allowed')
    # Reuse the unchanged baseline startup environment. Its cache-policy
    # selector is the single reference arm; worker count belongs in CLI config.
    env = maintained.environment(row, dict(plan, index=0))
    env.update(STRATA_TRACE='1', STRATA_CPU_WORK_CENSUS='1',
               STRATA_DECODE_TIMING='1')
    require(env['STRATA_SYCL_ADAPT_SYNC'] == '1'
            and env['EnableDirectSubmission'] == '0'
            and env['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD'] == '1',
            'cache synchronization and GPU safety settings must stay fixed')
    return env


def invariant(row, plan):
    value = dict(config=json.loads((Path(plan['root']) / 'config.json').read_text()),
                 environment=environment(row, plan))
    args = value['config']['args']
    workers = int(args[args.index('--pool-workers') + 1])
    if 'pool_workers' in plan:
        require(workers == plan['pool_workers'] == WORKERS[plan['index']],
                'unplanned CPU worker count')
    else:
        # The maintained preparer first writes identical baseline configs;
        # this policy freezes its worker fields before any host/model launch.
        require(workers == 5, 'unexpected preparation baseline')
    require(args[args.index('--adapt-swaps') + 1] == '0'
            and '--no-host-worker' not in args and '--no-split-rows' not in args,
            'host participation, static cache and native row path stay fixed')
    args[args.index('--pool-workers') + 1] = '<workers>'
    return hashlib.sha256(json.dumps(value, sort_keys=True).replace(
        plan['root'], '<process>').encode()).hexdigest()


def decision(processes):
    require(len(processes) == len(WORKERS)
            and all(p['passed'] and p['complete'] and p['ownership_closed']
                    for p in processes), 'complete diagnostic sweep required')
    workloads = {}
    for i, name in enumerate(census.WORKLOADS):
        samples = []
        for workers, process in zip(WORKERS, processes):
            sample = process['samples'][i]
            require(sample['workload'] == name, 'workload ordering differs')
            samples.append(dict(pool_workers=workers,
                exact_TG=sample['generated_tokens'] * 1000 / sample['decode_ms'],
                read_tokens=sample['read_tokens'], generated_tokens=sample['generated_tokens'],
                acceptance=sample['acceptance_rate'],
                observed_MTP_class=('strong' if sample['acceptance_rate'] >= .60
                                    else 'weak' if sample['acceptance_rate'] <= .40
                                    else 'intermediate')))
        for sample in samples:
            sample['TG_over_5workers'] = sample['exact_TG'] / samples[0]['exact_TG']
        workloads[name] = samples
    return dict(diagnostic_scaling_complete=True, workloads=workloads,
        clean_speed_qualified=False, repeated_gain_qualified=False,
        physical_resource_saturation_qualified=False, inference_adopted=False,
        next='Choose a worker configuration only for a new prospectively committed clean repeated comparison. One process per count and nested pool wall do not establish physical RAM bandwidth or a useful-CPU-work ceiling. Preserve dedicated prefill and independent decode qualification.')


def context():
    spec = importlib.util.spec_from_file_location('strata_worker_scaling', maintained.__file__)
    ctx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ctx)
    ctx.select_mode('cohort')
    ctx.ROOT, ctx.KEY = ROOT, KEY
    ctx.ARMS = tuple(('reference', 0) for _ in WORKERS)
    ctx.model = entry.context('zero')
    ctx.model.pinned_sources = pinned_sources
    ctx.basis, ctx.environment, ctx.invariant = basis, environment, invariant
    ctx.clean_observers, ctx.decision = census.diagnostic_output, decision
    ctx.selfcheck = maintained.selfcheck
    ctx.census_boundary_prefixes = []

    def boundary(root, expected):
        raw, history = maintained.cache_boundary(root, expected)
        census.diagnostic_output(raw.decode())
        path, = (root / 'bench').glob('*/requests.jsonl')
        events = [json.loads(v) for v in path.read_text().splitlines()]
        samples = [e['sample'] for e in events if e['kind'] == 'result']
        require(len(samples) in (expected - 1, expected), 'native/API extent differs')
        if len(samples) == expected:
            census.work_evidence(root, raw.decode(), samples)
            ctx.census_boundary_prefixes.append(expected)
        return raw, history

    ctx.cache_boundary = boundary
    native_current = ctx.current

    def current(row):
        require(cpu_resources() == row['cpu_resources'],
                'CPU affinity, topology or effective ancestor quota changed')
        return native_current(row)

    ctx.current = current
    return ctx


def prepare(ctx):
    ctx.prepare()
    row = json.loads((ROOT / 'record.json').read_text())
    require(len(row['plans']) == len(WORKERS), 'three worker configurations required')
    for plan, workers in zip(row['plans'], WORKERS):
        root = Path(plan['root'])
        config = json.loads((root / 'config.json').read_text())
        config['args'][config['args'].index('--pool-workers') + 1] = str(workers)
        base.write(root / 'config.json', config)
        plan['pool_workers'] = workers
        plan['generated']['config.json'] = digest(root / 'config.json')
    row['invariant_signature'] = invariant(row, row['plans'][0])
    require(all(invariant(row, p) == row['invariant_signature'] for p in row['plans']),
            'worker count must be the only arm difference')
    row.update(whole_bytes=256 << 20, budgets=dict(wall=3600, cpu=43200,
        rss_cap=104 << 30, text_cap=32 << 20, file_cap=32 << 20, total_cap=64 << 20))
    row['cpu_resources'] = cpu_resources()
    require(row['cpu_resources']['allowed_cpus'] == list(range(12))
            and len(set(map(tuple, row['cpu_resources']['package_core_ids'].values()))) == 6
            and all(v.get('cpu.max', 'max').split()[0] == 'max'
                    for v in row['cpu_resources']['cgroup_ancestors'].values()),
            'prospective six-physical-core unrestricted CPU plan differs')
    row['qualification_reuse'] = dict(model_result_key=census.KEY,
        native_source_commit=row['native_source_commit'], engine_sha256=row['binary']['sha256'],
        scope='Same selected engine,eight mathematical fixtures,checked worker and conserved short/long static-cache CPU census;only this Python worker policy is new. No new native generation.')
    row['plan'] = dict(mode='cohort', workload_version=5,
        question='Does actual long decode improve with fewer physical workers or additional unpinned SMT-capable workers,with native math/cache/GPU settings unchanged?',
        order='Three fresh processes,5then3then9workers;each four official warmups,regular-code then varied-prose once. Before each long request reconcile every completed prior native/API/census footer. One process per count;diagnostic screen,not repeated clean gain.',
        constants='Selected479a/552b engine,oneAPI2026.2 providers and safety/preload;throughput BF16 residual/GDN16/MTP16,heads1,131072INT8 context,1256static slots,spec4,chunk8192,pcie0,caches0,adapt0,20ms spin,host worker enabled. Fixed v5 bodies,thinking on,uncapped reasoning,temperature0,16384output budget. Only --pool-workers changes.',
        topology='Ryzen5600X,6physical/12logical,allowed CPUs0..11,no ancestor cgroup CPU quota at preparation. The compiled nonhybrid topology pins the first five workers to physical CPUs1..5;workers beyond that are unpinned and may contend with the host/driver. Host participates in the drain,so5workers means6compute participants. Do not call9workers ten physical cores or presume actual SMT placement.',
        source_audit='The selected constructor allocates scratch/state/thread storage by n_workers;each worker owns its indexed scratch and checked completion protocol. Explicit positive counts override auto selection. Extra workers get core=-1 rather than an out-of-bounds topology lookup. No GPU allocation,math,queue/event,provider or runtime-environment change. Four reconciled official warmups precede long generation.',
        observers='Existing STRATA_CPU_WORK_CENSUS1,STRATA_DECODE_TIMING1,STRATA_TRACE1,parameter validation/private retained activation preload. Immutable before exec;no API trace or hardware profiler. Timing includes counters and spinning.',
        acceptance='Every process and both workloads must satisfy unchanged official source/ELF/fixture/provider/body/route/journal/clock/ownership gates,>=32768fresh read,>=1024generated with nonempty thought and>=512answer chars,>=256drafts. Independently conserve all six CPU census/NT/format/cache/pool-clock bindings and actual startup worker count;normal exits,quiet validation,no fresh fault. Full15model hashes before and after sweep. Classify actual MTP,do not require identical output.',
        decision='Keep every configuration and actual TG ratio versus5workers,NT mix and nested pool wall. No speed-based exclusion or early stop. A promising count can proceed only to a separately committed clean matched repeated comparison;dedicated prefill nonregression remains separate.',
        stopping='First source/owner/provider/budget/route/sample/census/clock/validation/fault failure stops the sweep. No retry or refreshed hashes. Commit each closed process before the next.',
        interpretation='Worker sensitivity is configuration evidence,not disjoint useful leaf CPU time,physical RAM/PCIe traffic,GPU service or hardware saturation. No clean speed,task/agent,full262144/RESTORE or adoption claim.',
        retention='Keep three processes/two-workload individual samples,worker counts,compact format/NT/pool clocks and comparison decision. Retire successful full logs/journals/JIT after verified binding. Review after clean candidate selection. No new native generation.')
    ctx.save(row)
    print('Prepared diagnostic5/3/9-worker sweep;commit before frontend/model execution.', flush=True)


def run(ctx, index):
    try:
        ctx.run(index)
    except BaseException as error:
        row = json.loads((ROOT / 'record.json').read_text())
        if len(row['processes']) > index:
            row['processes'][index]['census_boundary_prefixes'] = ctx.census_boundary_prefixes
        else:
            row.update(stage='failed', passed=False, complete=False,
                       admission_error=repr(error))
        ctx.save(row)
        raise
    row = json.loads((ROOT / 'record.json').read_text())
    process, plan = row['processes'][index], row['plans'][index]
    process['pool_workers'] = plan['pool_workers']
    process['census_boundary_prefixes'] = ctx.census_boundary_prefixes
    try:
        require(ctx.census_boundary_prefixes == [4, 5], 'both prior-census gates required')
        root = Path(plan['root'])
        path, = (root / 'bench').glob('*/requests.jsonl')
        samples = [e['sample'] for e in map(json.loads, path.read_text().splitlines())
                   if e['kind'] == 'result']
        text = (root / 'engine.stderr').read_text()
        work, timing = census.work_evidence(root, text, samples)
        require(len(work) == len(timing) == len(samples) == 6, 'all six request bindings required')
        marker = f'strata generate: {plan["pool_workers"]} expert-pool workers + the host thread\n'
        require(text.count(marker) == 1, 'actual startup worker count differs')
        process['cpu_work'] = [dict(workload=name,
            work={k:w[k] for k in ('windows','rows','cpu_jobs','cpu_entries','vram_entries',
                  'jobs_NT','NT1_fraction','pool_ms','logical_job_blob_bytes')},
            format_bins=[{k:b[k] for k in ('gu','down','calls','pool_ns','jobs_NT','CPU','VRAM')}
                         for b in w['format_bins']],
            timing={k:t[k] for k in ('windows','ms_per_window','clock_binding')})
            for name,w,t in zip(census.WORKLOADS,work[4:],timing[4:])]
        process['census_passed'] = True
    except BaseException as error:
        process.update(passed=False, census_passed=False, census_error=repr(error))
        row.update(stage='failed', passed=False, complete=False)
    ctx.save(row)
    require(process['passed'], process.get('census_error','worker scaling failed'))
    print(f'Closed workers={plan["pool_workers"]} with both long census bindings.', flush=True)


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare','host','0','1','2'),
            'use prepare|host|0|1|2')
    ctx = context()
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base.guards.no_model_owners([])
        if sys.argv[1] == 'prepare': prepare(ctx)
        elif sys.argv[1] == 'host': ctx.comparison.host(ctx)
        else: run(ctx, int(sys.argv[1]))
