#!/usr/bin/env python3
"""Bound incremental schema2 caller observation in long thinking-on generation.

Main-authored, repository LICENSE. Reuse the owned bench.sh cohort privately;
CPU leaf service/census stay enabled in both arms. This is not clean speed.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
P = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(P / 'tools/sycl'))
import decode_service_model_entry as short
import cpu_service_overhead as leaf

B, RESULT, base = short.B, short.RESULT, short.base
ROOT, KEY = B / 'decode-service-overhead-v1', 'decode_service_overhead_v1'
SELF = Path(__file__).resolve()
ARMS, WORKLOADS = leaf.ARMS, leaf.WORKLOADS
require, digest = short.require, short.digest
CLOCK_SOURCES = ('include/strata/sycl/decode_service_profile.hpp',
    'bench/sycl/decode_clock_cost.cpp', 'bench/sycl/qualify_decode_clock_cost.py',
    'tools/sycl/decode_service_contract.py', 'tools/sycl/measurement_guards.py',
    'tools/sycl/measurement_owner.py', 'AGENTS.md', 'docs/ARTIFACT_RETENTION.md')


def pinned_sources(revision):
    pins = short.pinned_sources(revision)
    for path in (SELF, *(P / n for n in CLOCK_SOURCES)):
        pins.update(base.controls(revision, controller=path))
    return pins


def clock_proof(native):
    row = base.committed(RESULT)['decode_clock_cost_v1']
    cost = row['clock_cost']
    require(row['stage'] == 'closed' and row['passed'] is True
        and row['complete'] is True and row['ownership_closed'] is True
        and row['active'] is False and len(row['results']) == 2
        and all(r['passed'] is True and r['exit_code'] == 0 for r in row['results'])
        and set(row['sources']) == set(CLOCK_SOURCES)
        and all(digest(P / n) == sha for n, sha in row['sources'].items())
        and cost['passed'] is True and cost['host_cpu'] == 0
        and max(cost['monotonic_resolution_ns'], cost['thread_cpu_resolution_ns']) <= 1000
        and len(cost['samples']) == 5 and all(s['reads'] == 50000
            and 2000000 <= s['wall_ns'] <= 500000000
            and 2000000 <= s['cpu_ns'] <= 500000000 for s in cost['samples'])
        and row['native_header_binding'] == dict(source_commit=native['source_commit'],
            engine_sha256=native['binaries']['strata']['sha256'],
            header_sha256=native['native_sources'][CLOCK_SOURCES[0]]),
        'closed unchanged production-header clock calibration required')
    return row


def basis(revision):
    prior = json.loads((short.ROOT / 'record.json').read_text())
    providers, manifest = short.context().current(prior)
    service = prior['decode_service']
    require(prior['stage'] == 'closed' and prior['passed'] and prior['complete']
        and prior['ownership_closed'] and not prior['active'] and prior['host_passed']
        and prior['model_route_check_passed'] and service['passed']
        and service['schema2_request_joins'] and service['completed_requests'] == 3
        and prior['outcome']['native_exit_code'] == 0
        and prior['outcome']['fresh_faults'] == [],
        'closed same-build caller/leaf/census model entry required')
    current = pinned_sources(revision)
    extra = {str(SELF.relative_to(P)), *CLOCK_SOURCES} - set(prior['sources'])
    require(all(current.get(n) == sha for n, sha in prior['sources'].items())
        and set(current) - set(prior['sources']) == extra,
        'only this new controller and qualified clock inputs may extend model admission')
    native = json.loads((short.checks.model.ROOT / 'record.json').read_text())
    clock_proof(native)
    names = ('src/kernels/cpu/pool.cpp', 'include/strata/kernels/cpu/pool.hpp')
    pool = {n: native['native_sources'][n] for n in names}
    require(native['binaries']['strata'] == prior['binary'] and all(
        digest(short.checks.model.ROOT / 'source' / n) == sha for n, sha in pool.items()),
        'same compiled diagnostic engine and pool required')
    return prior, providers, manifest, pool


def evidence(root, text, samples, enabled):
    work, clocks, leaves = leaf.work_evidence(root, text, samples, True)
    ids = [w['request'] for w in work]
    require(ids == list(range(1, len(samples) + 1)), 'exact completed request order required')
    callers = short.reader.extract(text, ids if enabled else [])
    if enabled:
        require(all(r['wall_ns'] + 100000 >= s['decode_ms'] * 1000000
            for r, s in zip(callers, samples)),
            'caller interval must enclose reported decode and final commit')
    dispatches = short.dispatches(text, 1, positive=False)
    require([r['request'] for r in dispatches] == ids,
        'every caller/leaf/census request must join actual IQ3 dispatch')
    return work, clocks, leaves, callers


def decision(processes):
    require(all(p['decode_service']['passed'] is True
        and p['decode_service']['profile_enabled'] == (p['kernel'] == 'candidate')
        and p['decode_service']['completed_requests'] == 6 for p in processes),
        'complete caller admission or disabled absence required for every process')
    result = leaf.decision(processes)
    result.update(interpretation='Incremental DECODE_SERVICE_PROFILE cost only;CPU_SERVICE,CENSUS,DECODE_TIMING,TRACE and validation remain common. Exact logical work gates overhead interpretation,not text correctness. Retain mismatches as inconclusive. No clean speed,common-observer cost,physical traffic or saturation,70TG,agent/task,physical262144/RESTORE or adoption qualification.',
        next='If overhead qualifies,use long-request caller regions/nested waits and worker operation totals to select the next bounded critical-path optimization. Keep overlapping clocks and physical resource ceilings distinct.')
    return result


def context(index=None):
    ctx = short.checks.load('strata_decode_service_overhead_context', leaf.protocol.__file__)
    ctx.select_mode('cohort')
    ctx.ROOT, ctx.KEY, ctx.ARMS = ROOT, KEY, ARMS
    ctx.model = short.context()
    ctx.model.pinned_sources = pinned_sources
    ctx.basis, ctx.clean_observers = basis, leaf.diagnostic_output
    original_environment, original_route = ctx.environment, ctx.route
    ctx.census_boundary_prefixes, ctx.boundary_calls = [], 0

    def environment(row, plan):
        env = original_environment(row, plan)
        env.update(STRATA_SYCL_DECODE_SERVICE_PROFILE=str(int(plan['kernel'] == 'candidate')),
            STRATA_CPU_SERVICE='1', STRATA_CPU_WORK_CENSUS='1',
            STRATA_DECODE_TIMING='1', STRATA_TRACE='1',
            STRATA_SYCL_DECODE_Q6_COLUMNS='1', STRATA_SYCL_DECODE_Q6_CENTERING='0',
            STRATA_IQ3_PAIR='1', STRATA_IQ3_PAIR_AUDIT='1', STRATA_IQ_SINGLE_EXACT='0')
        require(env['STRATA_SYCL_ADAPT_SYNC'] == '1' and env['EnableDirectSubmission'] == '0'
            and env['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD'] == '1',
            'preserve immutable qualified runtime safety')
        return env

    def invariant(row, plan):
        config = json.loads((Path(plan['root']) / 'config.json').read_text())
        require(config['args'] == row['configuration']['args'] and plan['adapt_swaps'] == 0,
            'only caller observation may vary;native arguments fixed')
        env = environment(row, plan)
        env.pop('STRATA_SYCL_DECODE_SERVICE_PROFILE')
        return hashlib.sha256(json.dumps(dict(config=config, environment=env), sort_keys=True)
            .replace(plan['root'], '<process>').encode()).hexdigest()

    def route(text, read):
        result = original_route(text, read)
        result['centering_captures'] = short.centering_captures(text, 0)
        selected = short.dispatches(text, 1, positive=True)
        require(len(selected) == 1, 'exactly one actual long-request IQ3 footer required')
        result['IQ3_dispatch'] = selected[0]
        result['dispatch_proof'] = ('Observed native16/GDN3 and MTP mode2 callbacks,'
            'plus actual Q6 columns/centering0 and IQ3 pair1 dispatch receipts.'
            'The exact same-build short model supplies prior typed-storage qualification.')
        return result

    def boundary(root, expected):
        ordinal = ctx.boundary_calls
        require(ordinal < 4 and expected == (4, 5, 5, 6)[ordinal],
            'fixed begin/end gate order required')
        raw, history = leaf.protocol.cache_boundary(root, expected)
        text = raw.decode()
        leaf.diagnostic_output(text)
        path, = (root / 'bench').glob('*/requests.jsonl')
        events = [json.loads(v) for v in path.read_text().splitlines()]
        samples = [e['sample'] for e in events if e['kind'] == 'result']
        require(len(samples) in (expected - 1, expected), 'native/API boundary extent differs')
        if ordinal % 2 == 0:
            require(index is not None and len(samples) == expected,
                'all completed warmup/prior records required before a long request')
            evidence(root, text, samples, ARMS[index][0] == 'candidate')
            ctx.census_boundary_prefixes.append(expected)
        ctx.boundary_calls += 1
        return raw, history

    ctx.environment, ctx.invariant, ctx.route = environment, invariant, route
    ctx.cache_boundary = boundary
    ctx.selfcheck = lambda: dict(common_baseline_controls=leaf.protocol.selfcheck(),
        overhead_decision_controls=leaf.decision_controls(),
        scope='Unchanged workload/cache and paired-overhead arithmetic controls;not clean timing.')
    ctx.decision = lambda _: dict(generation_complete=True, caller_binding_pending=True,
        incremental_observer_overhead_qualified=False, inference_adopted=False)
    return ctx


def prepare(ctx):
    ctx.prepare()
    row = json.loads((ROOT / 'record.json').read_text())
    row['qualification_reuse'] = dict(build_key=short.checks.model.KEY,
        model_key=short.KEY, clock_key='decode_clock_cost_v1',
        native_source_commit=row['native_source_commit'], engine_sha256=row['binary']['sha256'],
        scope='Exact566native inputs/five diagnostic outputs,six CPU checks,submission10/phase16,actual C++ reader/28 corrupt controls,three real-model caller/leaf/census joins and five production-clock bands. No new native generation.')
    row['plan'] = dict(mode='cohort', workload_version=5,
        question='Does schema2 caller observation stay within the declared incremental latency bound,and which caller regions and nested waits dominate representative long generation?',
        order='Six independent processes off/on,on/off,off/on;each four official warmups,then fixed v5 regular-code and varied-prose once. No retry;commit each closed process before the next.',
        constants='Same5c5e engine/native70169,providers/safety/preload,throughput BF16 HC/GDN16/MTP16,heads1,131072INT8resident,1256static cache,5workers,spec4,chunk8192,short-read0,caches0,adapt0,20ms spin,Q6columns1/centering0,IQ3PAIR1/AUDIT1. Only immutable pre-exec STRATA_SYCL_DECODE_SERVICE_PROFILE0/1 varies.',
        observers='CPU_SERVICE1,CPU_WORK_CENSUS1,DECODE_TIMING1,TRACE1 and parameter validation common to both arms. No full API trace,hardware profiler or environment mutation. This measures only incremental caller-ledger cost.',
        acceptance='All6processes/12long requests satisfy source/build/ELF,15assets,providers,bodies,route,token,journal,clock,validation,fault and ownership gates. Each long request>=32768fresh input,>=1024generated including thinking,>=512answer chars,>=256drafts. Bind all6caller/leaf/census/native/API requests per process;profile0 must emit no caller records,profile1 requires complete conserved schema2 with contained nested waits and caller enclosing rounded decode. Strict completed prefixes4/5 before each long request. Normal native/benchmark0,quiet validation,no fresh fault. All15assets hashed before/after cohort.',
        decision='Each of3paired on/off per-output latency ratios<=1.05 for both workloads,exact aggregate logical work signatures,and observed code acceptance>=.60/prose<=.40 throughout. Changed work/class makes overhead inconclusive;retain all samples and finish collection. Logical-work matching is not exact-text correctness. No minimum speedup threshold.',
        stopping='Stop first source/provider/owner/budget/route/sample/clock/census/caller/validation/fault failure. No replacement,exclusion,speed stop,hash refresh or changed criteria.',
        retention='Keep individual long samples,compact work/signature,worker operation totals,nine caller-region/nested-wait totals and admission/absence counts. Omit full per-thread/per-layer tensors and leaf histograms. Retire successful logs/journals/JIT after each committed closure. Keep global record/runtime/generated plans only through admission. No new binary generation.')
    row['budgets'] = dict(wall=3600, cpu=43200, rss_cap=104 << 30,
        text_cap=16 << 20, file_cap=32 << 20, total_cap=64 << 20)
    ctx.save(row)
    print('Prepared six same-binary caller-profile processes;commit before frontend probes.', flush=True)


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
    proc['decode_service'] = dict(passed=False)
    try:
        require(ctx.boundary_calls == 4 and ctx.census_boundary_prefixes == [4, 5],
            'both strict prior-request gates required')
        root = Path(row['plans'][index]['root'])
        path, = (root / 'bench').glob('*/requests.jsonl')
        samples = [e['sample'] for e in map(json.loads, path.read_text().splitlines())
            if e['kind'] == 'result']
        enabled = ARMS[index][0] == 'candidate'
        work, clocks, leaves, callers = evidence(root,
            (root / 'engine.stderr').read_text(), samples, enabled)
        require(len(work) == len(clocks) == len(leaves) == len(samples) == 6,
            'all six actual model requests must bind')
        requests = []
        for w, t, l, s, name in zip(work[4:], clocks[4:], leaves[4:], samples[4:], WORKLOADS):
            item = leaf.retained_request(w, t, l, s, name)
            item.pop('leaf_bins', None)
            requests.append(item)
        proc['cpu_census'] = dict(passed=True, completed_requests=6, warmup_requests=4,
            boundary_prefixes=ctx.census_boundary_prefixes, long_requests=requests,
            physical_traffic_qualified=False, hardware_saturation_qualified=False,
            clean_speed_qualified=False)
        proc['decode_service'] = dict(passed=True, profile_enabled=enabled,
            completed_requests=6, caller_records=len(callers), clock_metrology_passed=True,
            long_requests=[dict(workload=name, **short.reader.compact(r))
                for name, r in zip(WORKLOADS, callers[4:])],
            caller_and_worker_cpu_additive=False, clean_speed_qualified=False)
        if row['passed']:
            row['decision'] = decision(row['processes'])
    except BaseException as error:
        row.update(stage='failed', passed=False, complete=False, caller_service_error=repr(error))
    ctx.save(row)
    require(row['stage'] != 'failed', row.get('caller_service_error', 'long caller admission failed'))
    print('Caller/leaf/census bound for process' + str(index)
        + ';overhead decision requires all three pairs.', flush=True)


if __name__ == '__main__':
    args = sys.argv[1:]
    require(len(args) == 1 and (args[0] in ('prepare', 'host') or
        args[0].isdigit() and int(args[0]) < len(ARMS)), 'use prepare|host|INDEX0..5')
    index = int(args[0]) if args[0].isdigit() else None
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base.guards.no_model_owners([])
        ctx = context(index)
        if args[0] == 'prepare':
            prepare(ctx)
        elif args[0] == 'host':
            ctx.comparison.host(ctx)
        else:
            run(ctx, index)
