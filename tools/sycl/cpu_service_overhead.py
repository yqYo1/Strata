#!/usr/bin/env python3
"""Repeated incremental CPU-service observer cost through bench.sh.

Main-authored; repository LICENSE. Reuse the owned long-generation supervisor
in a private context. Both arms retain the independent census, decode clocks
and identical diagnostics. Only the immutable leaf observer0/1 differs. This
does not measure the common observers' cost or constitute clean model speed.
"""
import copy
import fcntl
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
import cpu_service_model_entry as short
import pool_spin as protocol

P, B, RESULT, base = short.P, short.B, short.RESULT, short.base
ROOT = short.build.ROOT / 'cpu-service-overhead-v1'
KEY = 'verify_window_cpu_service_overhead_v1'
SELF = Path(__file__).resolve()
ARMS = (('reference', 0), ('candidate', 0), ('candidate', 0),
        ('reference', 0), ('reference', 0), ('candidate', 0))
WORKLOADS = ('regular-code', 'varied-prose')
MAX_LATENCY_RATIO = 1.05
require, digest = short.require, short.digest


def pinned_sources(revision):
    pins = short.pinned_sources(revision)
    pins.update(base.controls(revision, controller=SELF))
    return pins


def basis(revision):
    ctx = short.context()
    prior = json.loads((short.ROOT / 'record.json').read_text())
    providers, manifest = ctx.current(prior)
    require(prior['stage'] == 'closed' and prior['passed'] and prior['complete']
        and prior['ownership_closed'] and not prior['active'] and prior['host_passed']
        and prior['model_route_check_passed'] and prior['cpu_service']['passed']
        and prior['cpu_service']['completed_requests'] == 3
        and prior['model_process']['exit_code'] == prior['outcome']['native_exit_code'] == 0
        and prior['outcome']['fresh_faults'] == [],
        'closed same-build three-request CPU service/census entry required')
    current = pinned_sources(revision)
    require(all(current.get(n) == sha for n, sha in prior['sources'].items())
        and set(current) - set(prior['sources']) == {str(SELF.relative_to(P))},
        'only the new overhead controller may extend qualified model inputs')
    native = json.loads((short.build.ROOT / 'record.json').read_text())
    names = ('src/kernels/cpu/pool.cpp', 'include/strata/kernels/cpu/pool.hpp')
    pool = {n: native['native_sources'][n] for n in names}
    require(native['binaries']['strata'] == prior['binary'] and all(
        digest(short.build.ROOT / 'source' / n) == sha for n, sha in pool.items()),
        'same compiled observer pool and engine required')
    return prior, providers, manifest, pool


def diagnostic_output(text):
    require(not any(s in text for s in ('UR_RESULT_ERROR_', 'ZE_RESULT_ERROR_',
        'terminate called', 'teardown uncertain', 'strata serve: ADAPT_')),
        'runtime failure or unexpected adaptive route')
    require(all(v['valid'] == '1' and v['error'] == 'none'
        for kind, v in short.census.rows(text) if kind == 'end'),
        'native census invalidated before next request')


def work_evidence(root, text, samples, enabled):
    config = json.loads((root / 'config.json').read_text())
    args = config['args']
    formats = short.census.manifest_formats(Path(args[args.index('--pack') + 1]) / 'native_experts.txt')
    membership = short.census.expected_membership(args[args.index('--expert-profile') + 1])
    clocks = short.gen.decode_timing_evidence(text, samples)
    bindings = [dict(prompt_tokens=s['prompt_tokens'], windows=t['windows'],
        rows=t['work']['target_rows'], cpu_jobs=t['work']['cpu_unique_jobs'],
        cpu_entries=t['work']['cpu_entries'],
        pool_ms=t['ms_per_window']['CPU_pool_run'] * t['windows'])
        for s, t in zip(samples, clocks)]
    work = short.census.extract(text, bindings, formats, membership)
    if enabled:
        leaves = short.service.extract(text, bindings, formats, membership)
        require(len(leaves) == len(samples), 'every enabled observer record must bind')
    else:
        require(not short.service.records(text), 'disabled CPU-service observer emitted records')
        leaves = [None] * len(samples)
    return work, clocks, leaves


def retained_request(work, clock, leaf, sample, name):
    # Format/NT aggregation is enough to compare logical work and select a leaf;
    # do not retain the repeated 48-layer census or complete per-thread output.
    formats = {}
    for item in work['format_bins']:
        key = item['gu'], item['down']
        counts = formats.setdefault(key, [0] * 8)
        for nt, count in enumerate(item['jobs_NT']):
            counts[nt] += count
    format_jobs = [dict(gu=k[0], down=k[1], jobs_NT=v) for k, v in sorted(formats.items())]
    fields = ('request', 'windows', 'rows', 'cpu_jobs', 'cpu_entries', 'vram_entries',
        'jobs_NT', 'NT1_fraction', 'pool_ms', 'membership_sha256', 'logical_job_blob_bytes')
    signature = dict(generated_tokens=sample['generated_tokens'],
        **{k: work[k] for k in ('windows', 'rows', 'cpu_jobs', 'cpu_entries', 'vram_entries')},
        format_jobs_NT=format_jobs)
    result = dict(workload=name, work={k: work[k] for k in fields},
        format_jobs_NT=format_jobs, clock_binding=clock['clock_binding'],
        work_signature_sha256=hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest(),
        service_enabled=leaf is not None, same_request_census=True)
    if leaf is not None:
        result['service'] = {k: leaf[k] for k in ('record_version', 'completion_checked',
            'thread_count', 'calls', 'wall_ns', 'cpu_ns')}
        result['operations'] = {op: {k: sum(v[k] for v in leaf['format_bins'] if v['op'] == op)
            for k in ('calls', 'rows', 'token_rows', 'wall_ns', 'cpu_ns')} for op in short.service.OPS}
        # These actual aggregate bins have an immediate consumer: choose the
        # dominant native format/NT/operation for the next CPU leaf experiment.
        result['leaf_bins'] = leaf['format_bins']
    return result


def decision(processes):
    require(len(processes) == len(ARMS) and all(p['passed'] and p['complete']
        and p['ownership_closed'] and (p['kernel'], p['adapt_swaps']) == ARMS[i]
        and p['cpu_census']['passed'] for i, p in enumerate(processes)),
        'six closed generation/census processes required')
    require(len({(p['native_identity']['pid'], p['native_identity']['start_ticks'])
        for p in processes}) == len(ARMS), 'independent native processes required')
    results = {}
    for i, name in enumerate(WORKLOADS):
        ratios, matched, arms = [], [], {k: [] for k in ('reference', 'candidate')}
        for start in range(0, len(ARMS), 2):
            pair = {p['kernel']: p for p in processes[start:start + 2]}
            off, on = pair['reference']['samples'][i], pair['candidate']['samples'][i]
            ratio = (on['decode_ms'] / on['generated_tokens']) / (off['decode_ms'] / off['generated_tokens'])
            require(math.isfinite(ratio) and ratio > 0, 'finite positive per-output latency required')
            ratios.append(ratio)
            matched.append(pair['reference']['cpu_census']['long_requests'][i]['work_signature_sha256'] ==
                pair['candidate']['cpu_census']['long_requests'][i]['work_signature_sha256'])
            for arm, proc in pair.items():
                sample = proc['samples'][i]
                arms[arm].append(dict(exact_tg=sample['generated_tokens'] * 1000 / sample['decode_ms'],
                    acceptance=sample['acceptance_rate'], generated_tokens=sample['generated_tokens']))
        expected_class = all(v['acceptance'] >= .60 if name == 'regular-code' else
            v['acceptance'] <= .40 for values in arms.values() for v in values)
        results[name] = dict(paired_on_over_off_per_output_latency=ratios,
            paired_logical_work_matches=matched,
            geometric_mean_latency_ratio=math.exp(statistics.mean(math.log(v) for v in ratios)),
            expected_MTP_class_observed=expected_class,
            incremental_observer_bound_qualified=all(matched) and expected_class
                and all(v <= MAX_LATENCY_RATIO for v in ratios),
            arms={k: dict(samples=v, median_tg=statistics.median(s['exact_tg'] for s in v))
                for k, v in arms.items()})
    return dict(repeated_comparison_complete=True, max_per_pair_latency_ratio=MAX_LATENCY_RATIO,
        workloads=results, incremental_observer_overhead_qualified=all(
            v['incremental_observer_bound_qualified'] for v in results.values()),
        interpretation='Observed incremental CPU_SERVICE cost only,with all common diagnostics retained. Exact logical-work matching gates the overhead interpretation,not text/numerical correctness. A mismatch is inconclusive and preserves every sample. No common-observer overhead,clean speed,physical resource saturation,70TG,task/agent,full262144/RESTORE or adoption qualification.',
        next='Use the admitted long-request format/NT/operation service to select a CPU leaf method. Keep physical RAM traffic and arithmetic saturation unknown until independently measured.')


def decision_controls():
    rows = []
    for i, (arm, swaps) in enumerate(ARMS):
        rows.append(dict(kernel=arm, adapt_swaps=swaps, passed=True, complete=True,
            ownership_closed=True, native_identity=dict(pid=i + 1, start_ticks=i + 10),
            samples=[dict(generated_tokens=1024, decode_ms=1020 if arm == 'candidate' else 1000,
                acceptance_rate=.8 if name == 'regular-code' else .3) for name in WORKLOADS],
            cpu_census=dict(passed=True, long_requests=[dict(work_signature_sha256='fixed') for _ in WORKLOADS])))
    require(decision(rows)['incremental_observer_overhead_qualified'], 'two-percent positive control')
    bad = copy.deepcopy(rows)
    bad[1]['samples'][0]['decode_ms'] = 1080
    require(not decision(bad)['incremental_observer_overhead_qualified'],
        'one over-budget pair must survive a favorable median')
    mixed = copy.deepcopy(rows)
    mixed[1]['cpu_census']['long_requests'][0]['work_signature_sha256'] = 'different'
    require(not decision(mixed)['incremental_observer_overhead_qualified'] and
        decision(mixed)['repeated_comparison_complete'], 'changed work is inconclusive,not discarded')
    duplicate = copy.deepcopy(rows)
    duplicate[1]['native_identity'] = duplicate[0]['native_identity']
    for invalid in (rows[:-1], duplicate):
        try:
            decision(invalid)
        except RuntimeError:
            continue
        raise RuntimeError('incomplete/reused native cohort accepted')
    return dict(passed=True, positive=1, over_budget_pair_retained=True,
        changed_work_inconclusive=True, invalid_cohorts_rejected=2, GPU_executed=False)


def context(index=None):
    spec = importlib.util.spec_from_file_location('strata_cpu_service_overhead_context', protocol.__file__)
    ctx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ctx)
    ctx.select_mode('cohort')
    ctx.ROOT, ctx.KEY, ctx.ARMS = ROOT, KEY, ARMS
    ctx.model = short.context()
    ctx.model.pinned_sources = pinned_sources
    ctx.basis, ctx.clean_observers = basis, diagnostic_output
    original_environment, original_route = ctx.environment, ctx.route
    ctx.census_boundary_prefixes = []

    def environment(row, plan):
        env = original_environment(row, plan)
        env.update(STRATA_CPU_SERVICE='1' if plan['kernel'] == 'candidate' else '0',
            STRATA_CPU_WORK_CENSUS='1', STRATA_DECODE_TIMING='1', STRATA_TRACE='1',
            STRATA_SYCL_DECODE_Q6_COLUMNS='1')
        require(env['STRATA_SYCL_ADAPT_SYNC'] == '1' and env['EnableDirectSubmission'] == '0'
            and env['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD'] == '1', 'preserve immutable safety')
        return env

    def invariant(row, plan):
        config = json.loads((Path(plan['root']) / 'config.json').read_text())
        require(config['args'] == row['configuration']['args'] and plan['adapt_swaps'] == 0,
            'only the leaf observer may vary;native arguments fixed')
        env = environment(row, plan)
        env.pop('STRATA_CPU_SERVICE')
        return hashlib.sha256(json.dumps(dict(config=config, environment=env), sort_keys=True)
            .replace(plan['root'], '<process>').encode()).hexdigest()

    def route(text, read):
        result = original_route(text, read)
        result['Q6_captures'] = short.entry.captures(text, 'columns')
        return result

    def boundary(root, expected):
        raw, history = protocol.cache_boundary(root, expected)
        diagnostic_output(raw.decode())
        path, = (root / 'bench').glob('*/requests.jsonl')
        events = [json.loads(v) for v in path.read_text().splitlines()]
        samples = [e['sample'] for e in events if e['kind'] == 'result']
        require(len(samples) in (expected - 1, expected), 'native/API boundary extent differs')
        if len(samples) == expected:
            require(index is not None, 'actual observer arm required')
            work_evidence(root, raw.decode(), samples, ARMS[index][0] == 'candidate')
            ctx.census_boundary_prefixes.append(expected)
        return raw, history

    def checks():
        return dict(common_baseline_controls=protocol.selfcheck(),
            common_control_scope='Unchanged footer/workload controls only;not clean-envelope admission.',
            overhead_decision_controls=decision_controls())

    ctx.environment, ctx.invariant, ctx.route = environment, invariant, route
    ctx.cache_boundary, ctx.selfcheck = boundary, checks
    ctx.decision = lambda _: dict(generation_complete=True, CPU_service_binding_pending=True,
        incremental_observer_overhead_qualified=False, inference_adopted=False)
    return ctx


def prepare(ctx):
    ctx.prepare()
    row = json.loads((ROOT / 'record.json').read_text())
    row['qualification_reuse'] = dict(build_result_key=short.build.KEY,
        model_result_key=short.KEY, native_source_commit=row['native_source_commit'],
        engine_sha256=row['binary']['sha256'],
        scope='Same563native inputs/four ELFs,28-case off/on pool parity,small device and three actual v2 model/census joins. Only this new long-generation controller extends the unchanged299source inputs.')
    row['plan'] = dict(mode='cohort', workload_version=5,
        question='What incremental CPU_SERVICE cost and actual CPU leaf format/NT/operation service occur in long thinking-on decode?',
        order='Six independent processes off/on,on/off,off/on. Each four unchanged official warmups,then fixed v5 regular-code and varied-prose once each. No retries. Commit each closed process before the next.',
        constants='Same6e0730observer engine/native8e249,providers/safety/retained preload,throughput BF16 HC/GDN16/MTP16,heads1,131072INT8resident,1256static cache,5workers,spec4,chunk8192,short-read0,caches0,adapt0,20ms spin,Q6columns1. Only pre-exec STRATA_CPU_SERVICE0/1 changes.',
        observers='CPU_WORK_CENSUS1,DECODE_TIMING1,STRATA_TRACE1 and parameter validation in both arms;private retained validation preload. No full API trace,profiler,hardware counters or environment mutation. This isolates incremental leaf-observer cost,not all diagnostics.',
        acceptance='All6processes/12long requests must satisfy maintained source/ELF/model/provider/body/route/token/journal/clock/fault/owner gates,>=32768fresh input,>=1024generated including thinking,>=512answer characters,>=256drafts,normal native/benchmark0,quiet validation. Each process binds all6native/API censuses;on requires complete v2 service for all6,off requires no service rows. Validate completed prefixes4/5 before long requests. All15model hashes before/after cohort.',
        decision='Retain every paired per-output decode-latency ratio. Incremental overhead bound requires each of3ratios<=1.05 in both workloads,matching generated length/windows/target rows/CPU+VRAM entries and aggregate format/NT jobs,and observed code strong>=.60/prose weak<=.40 in every sample. No exact-text correctness requirement. Changed work or class makes overhead interpretation inconclusive without discarding samples or stopping collection. This is not a speedup adoption condition.',
        stopping='Stop first source/owner/provider/budget/route/sample/clock/census/service/validation/fault failure. No replacement,exclusion,speed stop,hash refresh or changed criteria.',
        retention='Keep individual samples,compact logical work and on-arm format/NT/operation CPU/wall bins for the immediate CPU-leaf-method decision;these bins answer which leaf dominates. Retire successful full logs/journals/JIT at each closed process. Retain global record/runtime and generated plans only while admission consumes them. No new binary generation.')
    row['budgets'] = dict(wall=3600, cpu=43200, rss_cap=104 << 30,
        text_cap=16 << 20, file_cap=32 << 20, total_cap=64 << 20)
    ctx.save(row)
    print('Prepared six same-binary observer processes/12long requests;commit before host probes.', flush=True)


def run(ctx, index):
    try:
        ctx.run(index)
    except BaseException:
        row = json.loads((ROOT / 'record.json').read_text())
        row.setdefault('census_boundary_prefixes', {})[str(index)] = ctx.census_boundary_prefixes
        ctx.save(row)
        raise
    row = json.loads((ROOT / 'record.json').read_text())
    proc = row['processes'][index]
    proc['model_route_check_passed'] = proc['passed']
    try:
        require(ctx.census_boundary_prefixes == [4, 5], 'both long-request prior-census gates required')
        root = Path(row['plans'][index]['root'])
        path, = (root / 'bench').glob('*/requests.jsonl')
        events = [json.loads(v) for v in path.read_text().splitlines()]
        samples = [e['sample'] for e in events if e['kind'] == 'result']
        work, clocks, leaves = work_evidence(root, (root / 'engine.stderr').read_text(),
            samples, ARMS[index][0] == 'candidate')
        require(len(work) == len(clocks) == len(samples) == 6, 'all six actual requests must bind')
        proc['cpu_census'] = dict(passed=True, completed_requests=6,
            warmup_requests=4, boundary_prefixes=ctx.census_boundary_prefixes,
            long_requests=[retained_request(w, t, leaf, s, name)
                for w, t, leaf, s, name in zip(work[4:], clocks[4:], leaves[4:], samples[4:], WORKLOADS)],
            physical_traffic_qualified=False, hardware_saturation_qualified=False,
            clean_speed_qualified=False)
        if row['passed']:
            row['decision'] = decision(row['processes'])
    except BaseException as error:
        row.update(stage='failed', passed=False, complete=False, cpu_service_error=repr(error))
    ctx.save(row)
    require(row['stage'] != 'failed', row.get('cpu_service_error', 'long CPU service failed'))
    print('Long CPU census/service bound for process ' + str(index) + ';incremental overhead decision needs all three pairs.', flush=True)


if __name__ == '__main__':
    args = sys.argv[1:]
    require(len(args) == 1 and (args[0] in ('prepare', 'host') or
        args[0].isdigit() and int(args[0]) < len(ARMS)), 'use prepare|host|INDEX0..5')
    index = int(args[0]) if args[0].isdigit() else None
    ctx = context(index)
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base.guards.no_model_owners([])
        if args[0] == 'prepare':
            prepare(ctx)
        elif args[0] == 'host':
            ctx.comparison.host(ctx)
        else:
            run(ctx, index)
