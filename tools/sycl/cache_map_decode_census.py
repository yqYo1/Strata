#!/usr/bin/env python3
"""Measure actual long-decode CPU format/NT work with the selected engine.

Repository LICENSE. Main owns serialized execution. Reuse the maintained
official-generation supervision in a private policy context, without modifying
its clean comparison or the native generation. Counter collection is diagnostic;
pool wall and logical blob visits do not establish CPU/DRAM/PCIe saturation.
"""
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import cache_map_cpu_census as short
import cpu_work_census as census
import pool_spin as maintained

entry, base, gen = short.entry, short.base, short.gen
P, B, RESULT = short.P, short.B, short.RESULT
require, digest = short.require, short.digest
ROOT = entry.model.ROOT / 'decode-census-v1'
KEY = 'cache_map_decode_census_v1'
SOURCE = Path(__file__).resolve()
WORKLOADS = ('regular-code', 'varied-prose')


def pinned_sources(revision):
    pins = short.pinned_sources(revision)
    pins.update(base.controls(revision, controller=SOURCE))
    return pins


def basis(revision):
    ctx = short.context()
    prior = json.loads((short.ROOT / 'record.json').read_text())
    providers, manifest = ctx.current(prior)
    require(prior['stage'] == 'closed' and prior['passed'] and prior['complete']
            and prior['ownership_closed'] and not prior['active']
            and prior['census_reader_tests']['passed'] and prior['census']['passed']
            and len(prior['census']['requests']) == 3
            and prior['model_route_check_passed']
            and prior['outcome']['native_exit_code'] == 0
            and prior['outcome']['fresh_faults'] == [],
            'closed current-engine short census/route/counter qualification required')
    current = pinned_sources(revision)
    require(all(current.get(n) == sha for n, sha in prior['sources'].items())
            and set(current) - set(prior['sources']) == {str(SOURCE.relative_to(P))},
            'only this new long-census controller may extend the qualified short inputs')
    native = json.loads((entry.model.ROOT / 'record.json').read_text())
    names = ('src/kernels/cpu/pool.cpp', 'include/strata/kernels/cpu/pool.hpp')
    pool = {n: native['native_sources'][n] for n in names}
    require(native['binaries']['strata'] == prior['binary'] and
            all(digest(entry.model.ROOT / 'source' / n) == sha for n, sha in pool.items()),
            'the unchanged qualified native pool is required')
    return prior, providers, manifest, pool


def environment(row, plan):
    require((plan['index'], plan['kernel'], plan['adapt_swaps'], plan['heads']) ==
            (0, 'reference', 0, 1), 'one static-cache diagnostic process only')
    env = maintained.environment(row, plan)
    env.update(STRATA_CPU_WORK_CENSUS='1', STRATA_DECODE_TIMING='1', STRATA_TRACE='1')
    require(env['STRATA_SYCL_ADAPT_SYNC'] == '1' and
            env['EnableDirectSubmission'] == '0' and
            env['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD'] == '1',
            'immutable checked cache and GPU safety settings required')
    return env


def invariant(row, plan):
    value = dict(config=json.loads((Path(plan['root']) / 'config.json').read_text()),
                 environment=environment(row, plan))
    args = value['config']['args']
    require(args[args.index('--adapt-swaps') + 1] == '0' and
            '--no-split-rows' not in args, 'static native-row census route required')
    return hashlib.sha256(json.dumps(value, sort_keys=True).replace(
        plan['root'], '<process>').encode()).hexdigest()


def diagnostic_output(text):
    require(not any(v in text for v in ('UR_RESULT_ERROR_', 'ZE_RESULT_ERROR_',
        'terminate called', 'teardown uncertain', 'strata serve: ADAPT_')),
        'runtime failure or unexpected adaptive route')
    require(all(v['valid'] == '1' and v['error'] == 'none'
                for kind, v in census.rows(text) if kind == 'end'),
            'native census invalidated before the next request')


def work_evidence(root, text, samples):
    config = json.loads((root / 'config.json').read_text())
    args = config['args']
    formats = census.manifest_formats(Path(args[args.index('--pack') + 1]) / 'native_experts.txt')
    membership = census.expected_membership(args[args.index('--expert-profile') + 1])
    timing = gen.decode_timing_evidence(text, samples)
    bindings = [dict(prompt_tokens=s['prompt_tokens'], windows=t['windows'],
        rows=t['work']['target_rows'], cpu_jobs=t['work']['cpu_unique_jobs'],
        cpu_entries=t['work']['cpu_entries'],
        pool_ms=t['ms_per_window']['CPU_pool_run'] * t['windows'])
        for s, t in zip(samples, timing)]
    return census.extract(text, bindings, formats, membership), timing


def decision(processes):
    require(len(processes) == 1 and processes[0]['passed'] and
            processes[0]['complete'] and processes[0]['ownership_closed'],
            'one closed diagnostic process required')
    samples = processes[0]['samples']
    require([s['workload'] for s in samples] == list(WORKLOADS), 'both long workloads required')
    return dict(diagnostic_requests_qualified=True,
        observed_code_strong_MTP=samples[0]['acceptance_rate'] >= .60,
        observed_prose_weak_MTP=samples[1]['acceptance_rate'] <= .40,
        clean_speed_qualified=False, repeated_gain_qualified=False,
        physical_resource_saturation_qualified=False, inference_adopted=False,
        next='Bind the complete same-request census before selecting a CPU leaf experiment. Actual NT/format/pool work is not useful-arithmetic CPU time or physical RAM traffic.')


def context():
    spec = importlib.util.spec_from_file_location('strata_cache_map_decode_census', maintained.__file__)
    ctx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ctx)
    ctx.select_mode('cohort')
    ctx.ROOT, ctx.KEY, ctx.ARMS = ROOT, KEY, (('reference', 0),)
    ctx.model = entry.context('zero')
    ctx.model.pinned_sources = pinned_sources
    ctx.basis, ctx.environment, ctx.invariant = basis, environment, invariant
    ctx.clean_observers, ctx.decision = diagnostic_output, decision
    # The original clean controller's selfcheck remains unchanged. This is its
    # baseline reader/contract control, not a clean-envelope test for this policy.
    ctx.selfcheck = maintained.selfcheck
    ctx.census_boundary_prefixes = []
    def boundary(root, expected):
        raw, history = maintained.cache_boundary(root, expected)
        diagnostic_output(raw.decode())
        path, = (root / 'bench').glob('*/requests.jsonl')
        events = [json.loads(v) for v in path.read_text().splitlines()]
        samples = [e['sample'] for e in events if e['kind'] == 'result']
        require(len(samples) in (expected - 1, expected), 'native/API boundary extent differs')
        # At begin the prior API results are durable. At end the current API
        # result is appended only after the owner acknowledges this gate.
        if len(samples) == expected:
            work_evidence(root, raw.decode(), samples)
            ctx.census_boundary_prefixes.append(expected)
        return raw, history
    ctx.cache_boundary = boundary
    return ctx


def prepare(ctx):
    ctx.prepare()
    row = json.loads((ROOT / 'record.json').read_text())
    require(len(row['plans']) == 1 and row['plans'][0]['adapt_swaps'] == 0,
            'no cache-policy comparison in this diagnostic')
    row.update(whole_bytes=96 << 20, budgets=dict(wall=3600, cpu=21600,
        rss_cap=104 << 30, text_cap=32 << 20, file_cap=32 << 20, total_cap=64 << 20))
    row['qualification_reuse'] = dict(model_result_key=short.KEY,
        native_source_commit=row['native_source_commit'], engine_sha256=row['binary']['sha256'],
        scope='Same selected engine,eight mathematical fixtures,checked worker and static-cache short census. Only this new controller extends the unchanged qualified inputs;no new native generation.')
    row['plan'] = dict(mode='cohort', workload_version=5,
        question='What CPU native format/job-NT mix and nested pool wall occur in actual thinking-on long decode?',
        order='One fresh static-cache process;four unchanged official warmups,regular-code then varied-prose,once each. Before each long request join every prior completed native/API request and its CPU census. No retries.',
        constants='Selected479a/552b engine,oneAPI2026.2 providers and qualified safety/preload;throughput BF16 residual/GDN16/MTP16,heads1,131072INT8 context,1256static expert slots,5workers,spec4,prefill8192,pcie0,caches0,adapt0,20ms spin. Fixed original v5 workload ID/bodies;thinking on,uncapped reasoning,temperature0,total output room16384.',
        observers='STRATA_TRACE1,STRATA_CPU_WORK_CENSUS1,STRATA_DECODE_TIMING1,parameter validation/private activation preload. Immutable before native exec. No full API trace,perf hardware counters or new CPU leaf observer. Diagnostic latency includes counters;it is not clean performance.',
        acceptance='Existing official source/body/token/journal/owner/provider/route/clock/fault gates. Both fresh inputs>=32768,generated>=1024including nonempty thinking,>=512answer characters and>=256drafts. All six requests must have valid static membership,format/NT/job/entry/window conservation and rounded pool clock joins. Normal exits,quiet validation,no freshfault;all15model hashes before/after. Record actual strong/weak classification;names do not establish it.',
        interpretation='Logical expert-blob visits and dot-equivalent work only;exclude separate draft. Pool wall includes worker scheduling/waits,not useful-arithmetic CPU time. No physical RAM/PCIe traffic,bandwidth ceiling,hardware saturation,clean speed,matched gain,task/agent,full262144/RESTORE or adoption qualification.',
        retention='Keep the two long samples,compact per-format NT/job/pool counts,request joins and source/fault/owner outcomes for CPU-method selection. Retire successful full journals/logs/JIT after binding;review the short census and this diagnostic when its leaf-method consumer closes. No new native generation.')
    ctx.save(row)
    print('Prepared one diagnostic long-generation CPU census;commit before host/model execution.', flush=True)


def run(ctx):
    try:
        ctx.run(0)
    except BaseException:
        row = json.loads((ROOT / 'record.json').read_text())
        row['census_boundary_prefixes'] = ctx.census_boundary_prefixes
        ctx.save(row)
        raise
    row = json.loads((ROOT / 'record.json').read_text())
    row['generation_route_check_passed'] = row['passed']
    row['census_boundary_prefixes'] = ctx.census_boundary_prefixes
    try:
        require(ctx.census_boundary_prefixes == [4, 5], 'both long-request prior-census gates required')
        root = Path(row['plans'][0]['root'])
        path, = (root / 'bench').glob('*/requests.jsonl')
        events = [json.loads(v) for v in path.read_text().splitlines()]
        samples = [e['sample'] for e in events if e['kind'] == 'result']
        work, timing = work_evidence(root, (root / 'engine.stderr').read_text(), samples)
        require(len(work) == len(timing) == len(samples) == 6, 'all warmup/long census requests required')
        row['census'] = dict(passed=True, completed_requests_verified=6,
            warmup_requests_verified=4, long_requests=[dict(workload=name, work=w, timing=t)
                for name, w, t in zip(WORKLOADS, work[4:], timing[4:])],
            physical_traffic_qualified=False, hardware_saturation_qualified=False,
            clean_speed_qualified=False, inference_adopted=False)
    except BaseException as error:
        row.update(stage='failed', passed=False, complete=False, census_error=repr(error))
    ctx.save(row)
    require(row['passed'], row.get('census_error', 'long census failed'))
    print('Both long workloads conserve actual CPU format/NT jobs and pool clocks;physical traffic and leaf CPU attribution remain separate.', flush=True)


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'host', 'run'), 'use prepare|host|run')
    ctx = context()
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base.guards.no_model_owners([])
        {'prepare': lambda: prepare(ctx), 'host': lambda: ctx.comparison.host(ctx),
         'run': lambda: run(ctx)}[sys.argv[1]]()
