#!/usr/bin/env python3
"""Conserve current-engine CPU leaves, caller regions and nested waits.

Main-authored; repository LICENSE. Static cache only. Compare caller ledger
off/on while CPU leaf/census/decode clocks remain common. Not clean speed.
"""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics
import sys

sys.dont_write_bytecode = True
import compare_adaptive_generation as clean
import cpu_service_overhead as leaf
import decode_footer as footer
import decode_service_evidence as caller

SELF = Path(__file__).resolve()
ARMS = tuple((arm, 0) for arm, _ in clean.ARMS)
MAX_RATIO = 1.05
HEADERS = ('include/strata/core/cpu_work_census.hpp',
    'include/strata/kernels/cpu/service_profile.hpp',
    'include/strata/sycl/decode_service_profile.hpp')
require, digest, signature = clean.require, clean.digest, clean.signature


def prerequisites(full=False):
    normal, prior, contract, manifest, providers, refs = clean.basis(full=full)
    data = clean.base.committed(clean.RESULT)
    for name in clean.CAPS:
        row = data['adaptive_generation_' + name.replace('-', '_') + '_v2']
        require(row['stage'] == 'closed' and row['passed'] and row['complete'] and
            row['ownership_closed'] and not row['active'] and len(row['processes']) == 6 and
            row['binary'] == normal['binary'] and all(p['passed'] and p['complete'] and
                p['ownership_closed'] and p['native_exit_code'] == 0 and not p['fresh_faults']
                for p in row['processes']), 'close both frozen speed cohorts first')
    build = json.loads((clean.positive.admission.build.ROOT / 'record.json').read_text())
    require(build['binaries']['strata'] == normal['binary'] and
        '-DSTRATA_CPU_SERVICE_PROFILE:BOOL=ON' in build['recipe']['configure'],
        'same observer-enabled engine required')
    for name in HEADERS:
        require(digest(clean.P / name) == build['native_sources'][name] ==
            digest(clean.positive.admission.build.ROOT / 'source' / name),
            'current compiled observer header differs: ' + name)
    readers = data['decode_service_reader_contract_v1']
    clocks = data['decode_clock_cost_v1']
    for row in (readers, clocks):
        require(row['stage'] == 'closed' and row['passed'] and row['complete'] and
            row['ownership_closed'] and not row['active'] and
            row['native_header_binding']['header_sha256'] == build['native_sources'][HEADERS[-1]],
            'unchanged historical producer/reader and clock implementation required')
        # Policy snapshots are historical metadata. Reuse only unchanged actual
        # implementation/fixture bytes; this fresh plan pins current policy.
        for name, sha in row['sources'].items():
            if name not in ('AGENTS.md', 'docs/ARTIFACT_RETENTION.md'):
                require(digest(clean.P / name) == sha, 'observer implementation drift: ' + name)
    require(readers['reader_controls']['corrupt_cases_rejected'] == 28 and
        readers['reader_controls']['exact_integer_clock_conservation'] and
        clocks['clock_cost']['passed'], 'reader/metrology controls must pass')
    proof = data['cache_map_cpu_service_sources_v1']['reader_v2_host']
    require(proof['passed'] and proof['complete'] and proof['ownership_closed'] and
        not proof['active'] and all(digest(clean.P / n) == sha
            for n, sha in proof['source_inputs'].items()), 'unchanged leaf/census reader required')
    return normal, prior, contract, manifest, providers, refs


def timings(text, samples, width):
    """Bind real T1/Tcap traces to rounded clocks without rewriting evidence."""
    require(width in (2, 4) and 0 < len(text) <= 64 << 20, 'bounded fixed-width timing input')
    headers = list(re.finditer(footer.HEADER, text, re.M))
    require(0 < len(headers) == len(samples) <= 64 and
        text.count('strata serve: prompt ') == len(headers) and
        text.count('strata decode timing:') == len(headers) == len(re.findall(footer.TIMING, text, re.M)) and
        text.count('strata serve: SYCL_EXPERT_COPY') == len(headers) == len(re.findall(footer.COPY, text, re.M)),
        'complete clock/copy/header framing')
    clocks, previous = [], 0
    for i, (header, sample) in enumerate(zip(headers, samples)):
        require(tuple(map(int, header.groups())) == tuple(sample[k] for k in ('prompt_tokens',
            'reused_tokens', 'read_tokens', 'generated_tokens', 'drafts_accepted', 'drafts_offered')) and
            sample['reused_tokens'] == 0, 'native/client request binding')
        segment = text[previous:header.start()]
        tail = text[header.end():headers[i+1].start() if i+1 < len(headers) else len(text)]
        native = re.findall(footer.TIMING, segment, re.M)
        copies = re.findall(footer.COPY, tail, re.M)
        require(len(native) == len(copies) == 1, 'timing/copy escaped its request')
        windows = int(native[0][0])
        values = dict(zip(footer.LABELS, map(float, native[0][1:])))
        counters = dict(zip(footer.COPY_KEYS, map(int, copies[0])))
        traced = [tuple(map(int, v)) for v in re.findall(r'^strata trace: window (\d+) (\d+)$', segment, re.M)]
        require(windows > 0 and len(traced) == windows and
            segment.count('strata trace: window ') == windows and
            traced[0] == (sample['read_tokens']-1, 1) and all(t in (1, width) for _, t in traced) and
            all(1 <= b[0]-a[0] <= a[1] for a, b in zip(traced, traced[1:])) and
            sum(t-1 for _, t in traced) == sample['drafts_offered'], 'actual target trace differs')
        require(all(math.isfinite(v) and v >= 0 for v in values.values()) and
            all(0 <= v < 2**64 for v in counters.values()) and
            all(counters[k] == 0 for k in footer.COPY_KEYS[:6]) and values['PCIe'] == 0,
            'clock bounds or expert-copy route')
        rows = sum(t for _, t in traced)
        jobs, entries = counters['cpu_unique_misses'], counters['cpu_entries']
        require(0 < jobs <= entries <= max(t for _, t in traced)*jobs and entries <= rows*480,
                'actual CPU job/entry conservation')
        for key, expected in (('average_T', rows/windows),
            ('tokens_per_window', sample['generated_tokens']/windows),
            ('CPU_experts', jobs/(windows*48)), ('CPU_entries', entries/(windows*48)),
            ('VRAM_hits', (rows*480-entries)/(windows*48))):
            require(abs(values[key]-expected) <= .00501, 'rounded work binding: ' + key)
        error = abs(values['total']*windows-sample['decode_ms'])
        rounding = .005*windows+.05
        require(error <= rounding+1e-8, 'rounded decode clock does not bind')
        residual = values['total']-sum(values[k] for k in ('verify', 'commit_emit', 'draft'))
        nested = values['per_layer_host']-sum(values[k] for k in ('plan', 'activation_quant', 'jobs', 'CPU_pool_run'))
        verify = values['verify']-sum(values[k] for k in ('GPU_reach_wait', 'per_layer_host', 'stage'))
        require(min(residual, nested, verify) >= -.025, 'nested clocks exceed parent')
        clocks.append(dict(windows=windows, ms_per_window=values,
            outer_residual_ms_per_window=residual, host_nested_residual_ms_per_window=nested,
            verify_residual_ms_per_window=verify,
            work=dict(target_rows=rows, cpu_unique_jobs=jobs, cpu_entries=entries),
            clock_binding=dict(absolute_total_difference_ms=error, rounding_bound_ms=rounding)))
        previous = header.end()
    require('strata trace: window ' not in text[previous:], 'unbound trailing target window')
    return clocks


def evidence(root, text, samples, enabled):
    args = json.loads((root / 'config.json').read_text())['args']
    clocks = timings(text, samples, int(args[args.index('--mtp-max-t')+1]))
    bindings = [dict(prompt_tokens=s['prompt_tokens'], windows=t['windows'],
        rows=t['work']['target_rows'], cpu_jobs=t['work']['cpu_unique_jobs'],
        cpu_entries=t['work']['cpu_entries'], pool_ms=t['ms_per_window']['CPU_pool_run']*t['windows'])
        for s, t in zip(samples, clocks)]
    formats = leaf.short.census.manifest_formats(Path(args[args.index('--pack')+1]) / 'native_experts.txt')
    membership = leaf.short.census.expected_membership(args[args.index('--expert-profile')+1])
    work = leaf.short.census.extract(text, bindings, formats, membership)
    leaves = leaf.short.service.extract(text, bindings, formats, membership)
    ids = [w['request'] for w in work]
    require(ids == list(range(1, len(samples) + 1)), 'complete ordered request prefix')
    callers = caller.extract(text, ids if enabled else [])
    if enabled:
        require(all(r['wall_ns'] + 100000 >= s['decode_ms'] * 1000000
            for r, s in zip(callers, samples)), 'caller must enclose rounded decode clock')
    return work, clocks, leaves, callers


def decision(processes):
    require(len(processes) == 6 and all(p['passed'] and p['complete'] and p['ownership_closed'] and
        (p['kernel'], p['adapt_swaps']) == ARMS[i] and p['native_exit_code'] == 0 and
        not p['fresh_faults'] and p['resources']['passed'] and
        p['resources']['completed_requests'] == 5 and p['resources']['warmup_prefix'] == 4 and
        p['resources']['caller_enabled'] == (p['kernel'] == 'candidate') and
        p['resources']['caller_records'] == (5 if p['kernel'] == 'candidate' else 0)
        for i, p in enumerate(processes)), 'six closed observer processes required')
    require(len({(p['native_identity']['pid'], p['native_identity']['start_ticks'])
        for p in processes}) == 6, 'fresh native processes required')
    ratios, matches = [], []
    for start in (0, 2, 4):
        pair = {p['kernel']: p for p in processes[start:start+2]}
        a, b = pair['candidate']['samples'][0], pair['reference']['samples'][0]
        ratio = (a['decode_ms'] / a['generated_tokens']) / (b['decode_ms'] / b['generated_tokens'])
        require(math.isfinite(ratio) and ratio > 0, 'finite positive observer latency')
        ratios.append(ratio)
        matches.append(pair['candidate']['resources']['long_request']['work_signature_sha256'] ==
            pair['reference']['resources']['long_request']['work_signature_sha256'])
    return dict(paired_caller_on_over_off_latency=ratios, paired_logical_work_matches=matches,
        geometric_mean_latency_ratio=math.exp(statistics.mean(map(math.log, ratios))),
        max_per_pair_latency_ratio=MAX_RATIO,
        incremental_caller_overhead_qualified=all(matches) and all(v <= MAX_RATIO for v in ratios),
        limits='CPU_SERVICE,CENSUS and DECODE_TIMING remain common and charged. Only incremental caller observation is bounded;not total diagnostic overhead or clean speed. Logical aggregate equality is not identical physical accesses. Cap2 acceptance has its own denominator. No physical traffic,saturation,quality,adoption,physical262144/RESTORE or70TG qualification.')


def controls():
    for width in (2, 4):
        sample = dict(prompt_tokens=10, reused_tokens=0, read_tokens=10,
            generated_tokens=2, drafts_accepted=0, drafts_offered=width-1, decode_ms=60.)
        text = ('strata trace: window 9 1\n' + f'strata trace: window 10 {width}\n'
            f'strata decode timing: 2 windows, avg T {(1+width)/2:.2f}, 1.00 tokens/window, '
            '30.00 ms/window = verify 20.00 (GPU-reach wait 5.00 + per-layer host 12.00 '
            '[plan 1.00 actq 1.00 jobs 1.00 CPU 8.00] + stage 3.00) + commit/emit 5.00 + draft 5.00; '
            f'per layer-window: CPU experts 0.02 (0.03 entries), VRAM hits {((1+width)*480-3)/96:.2f}, PCIe 0.00\n'
            f'strata serve: prompt 10 tokens = 0 reused + 10 read, 2 generated, drafts accepted 0 of {width-1}, done\n'
            'strata serve: SYCL_EXPERT_COPY ' + ' '.join(k+'='+str(2 if k=='cpu_unique_misses' else
                3 if k=='cpu_entries' else 0) for k in footer.COPY_KEYS) + '\n')
        require(timings(text, [sample], width)[0]['work']['target_rows'] == 1+width,
                'valid fixed-width timing fixture')
        corrupt = [text.replace('window 9 1', 'window 8 1'),
            text.replace(f'window 10 {width}', 'window 10 3'),
            text.replace('1.00 tokens/window', '2.00 tokens/window'),
            text.replace('CPU experts 0.02', 'CPU experts 0.05'),
            text.replace('cpu_unique_misses=2', 'cpu_unique_misses=0'),
            text.replace('submitted_bytes=0', 'submitted_bytes=1'),
            text.replace('CPU 8.00', 'CPU 18.00'), text.replace('30.00 ms/window', '31.00 ms/window'),
            text.replace('strata decode timing:', 'missing clock:'), text+text,
            text+'strata trace: window 12 1\n',
            text.replace('strata trace: window 9 1', 'strata trace: window bad 1\nstrata trace: window 9 1')]
        for bad in corrupt:
            try:
                timings(bad, [sample], width)
            except RuntimeError:
                continue
            raise RuntimeError('corrupt fixed-width timing accepted')
    rows = []
    for i, (arm, swaps) in enumerate(ARMS):
        rows.append(dict(kernel=arm, adapt_swaps=swaps, passed=True, complete=True,
            ownership_closed=True, native_identity=dict(pid=i+1, start_ticks=i+10),
            native_exit_code=0, fresh_faults=[],
            samples=[dict(generated_tokens=1024, decode_ms=1020 if arm == 'candidate' else 1000)],
            resources=dict(passed=True, completed_requests=5, warmup_prefix=4,
                caller_enabled=arm == 'candidate', caller_records=5 if arm == 'candidate' else 0,
                long_request=dict(work_signature_sha256='same'))))
    require(decision(rows)['incremental_caller_overhead_qualified'], 'positive overhead control')
    changed = json.loads(json.dumps(rows))
    changed[1]['samples'][0]['decode_ms'] = 1080
    require(not decision(changed)['incremental_caller_overhead_qualified'], 'keep over-budget pair')
    changed = json.loads(json.dumps(rows))
    changed[1]['resources']['long_request']['work_signature_sha256'] = 'changed'
    require(not decision(changed)['incremental_caller_overhead_qualified'], 'changed work is inconclusive')
    duplicate = json.loads(json.dumps(rows))
    duplicate[1]['native_identity'] = duplicate[0]['native_identity']
    for bad in (rows[:-1], duplicate):
        try:
            decision(bad)
        except RuntimeError:
            continue
        raise RuntimeError('incomplete/reused observer cohort accepted')
    for workload in clean.CAPS:
        ctx = context(workload)
        row = json.loads((clean.B / ('adaptive-generation-' + workload + '-v2') / 'record.json').read_text())
        plans = [dict(index=i, kernel=ARMS[i][0], adapt_swaps=0, heads=1,
            root=str(ctx.ROOT / ('control-' + str(i)))) for i in (0, 1)]
        require(ctx.invariant(row, plans[0]) == ctx.invariant(row, plans[1]),
                'only caller observer and private paths may differ')
        for plan in plans:
            env = ctx.environment(row, plan)
            require(all(env[k] == '1' for k in ('STRATA_TRACE', 'STRATA_CPU_SERVICE',
                'STRATA_CPU_WORK_CENSUS', 'STRATA_DECODE_TIMING')) and
                env['STRATA_SYCL_DECODE_SERVICE_PROFILE'] == str(plan['index']) and
                env['EnableDirectSubmission'] == '0' and env['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD'] == '1',
                'immutable common observers/runtime safety')
        try:
            ctx.environment(row, dict(plans[1], adapt_swaps=8))
        except RuntimeError:
            pass
        else:
            raise RuntimeError('adaptive cache admitted to static census')
    return dict(passed=True, positive=1, over_budget_retained=True,
        changed_work_inconclusive=True, invalid_cohorts_rejected=2,
        timing_widths=[2, 4], corrupt_timing_cases=24,
        immutable_environment_controls=4, adaptive_census_refused=True, GPU_executed=False)


def context(workload, index=None):
    require(workload in clean.CAPS, 'unknown focused resource workload')
    spec = importlib.util.spec_from_file_location('strata_current_decode_resources', clean.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ARMS, module.SELF, module.basis = ARMS, SELF, prerequisites
    base_environment = module.environment

    def environment(row, plan):
        require(plan['adapt_swaps'] == 0 and (plan['kernel'], 0) == ARMS[plan['index']],
                'census requires a static cache')
        env = base_environment(row, plan)
        env.update(STRATA_CPU_SERVICE='1', STRATA_CPU_WORK_CENSUS='1', STRATA_DECODE_TIMING='1', STRATA_TRACE='1',
            STRATA_SYCL_DECODE_SERVICE_PROFILE=str(int(plan['kernel'] == 'candidate')))
        return env

    def invariant(row, plan):
        env = environment(row, plan)
        env.pop('STRATA_SYCL_DECODE_SERVICE_PROFILE')
        value = dict(configuration=module.configuration(row, plan), environment=env)
        return signature(json.loads(json.dumps(value).replace(plan['root'], '<process>')))

    def source_pins(revision):
        pins = clean.source_pins(revision)
        pins.update(clean.base.controls(revision, controller=SELF))
        for name in HEADERS:
            raw = clean.base.basis.build.git(clean.P, 'show', revision + ':' + name)
            require((clean.P / name).read_bytes() == raw, 'commit observer header: ' + name)
            pins[name] = hashlib.sha256(raw).hexdigest()
        return pins

    module.environment, module.invariant, module.source_pins = environment, invariant, source_pins
    module.controls = lambda: dict(baseline=clean.controls(), resource_decision=controls())
    ctx = module.context(workload)
    ctx.ROOT = clean.B / ('decode-resource-' + workload + '-v1')
    ctx.KEY = 'decode_resource_' + workload.replace('-', '_') + '_v1'
    ctx.WORKLOAD_ID = 'decode_resource_20261107_' + workload
    ctx.resource_prefixes, ctx.boundary_calls = [], 0
    original_boundary = ctx.cache_boundary

    def boundary(root, expected):
        ordinal = ctx.boundary_calls
        require(ordinal < 2 and expected == (4, 5)[ordinal], 'focused request gate order')
        raw, history = original_boundary(root, expected)
        diagnostic(raw.decode())
        path, = (root / 'bench').glob('*/requests.jsonl')
        samples = [e['sample'] for e in map(json.loads, path.read_text().splitlines())
                   if e['kind'] == 'result']
        require(len(samples) in (expected-1, expected), 'native/API prefix extent')
        if ordinal == 0:
            require(index is not None and len(samples) == 4, 'all four warmups before long request')
            evidence(root, raw.decode(), samples, ARMS[index][0] == 'candidate')
            ctx.resource_prefixes.append(4)
        ctx.boundary_calls += 1
        return raw, history

    ctx.cache_boundary, ctx.clean_observers, ctx.invariant = boundary, diagnostic, invariant
    ctx.decision = lambda _: dict(model_execution_closed=True, resource_binding_pending=True,
        incremental_caller_overhead_qualified=False, inference_adopted=False)
    ctx.prepare_resources = lambda: prepare(module, ctx)
    return ctx


def diagnostic(text):
    require(not any(value in text for value in ('UR_RESULT_ERROR_', 'ZE_RESULT_ERROR_',
        'terminate called', 'teardown uncertain', 'strata serve: ADAPT_', 'ADAPT_RECORD_DIAGNOSTIC')),
        'runtime failure or adaptive mutation in static observer')
    require(all(v['valid'] == '1' and v['error'] == 'none'
        for kind, v in leaf.short.census.rows(text) if kind == 'end'), 'invalid CPU census')


def prepare(module, ctx):
    module.prepare(ctx)
    row = json.loads((ctx.ROOT / 'record.json').read_text())
    row['plan'] = ('Six fresh processes caller0/1,1/0,0/1 on the same429b engine. '
        'Four official warmups then one focused v5 thinking-on32K+/1K+ request with an answer. '
        'Code cap4,prose cap2 in independent cohorts. Static cache0/every8,half1 and all other '
        'native placement/runtime safety stay fixed. CPU_SERVICE1,CENSUS1,DECODE_TIMING1,TRACE1 are '
        'common;only immutable pre-exec DECODE_SERVICE_PROFILE0/1 changes. Require all four '
        'warmup records to conserve before the long request,then all five records and normal '
        'native/benchmark0,quiet validation,original route/body/clock/provider/owner and no-fault '
        'gates. Three paired per-output latency ratios<=1.05 with matching aggregate logical '
        'work bound incremental caller cost only. Work mismatch or an over-budget pair stays '
        'inconclusive/failed-overhead;retain every sample and finish the frozen cohort. Do not '
        'require identical text as quality or relabel cap2 MTP acceptance. Historical reader '
        'and metrology reuse covers unchanged implementation bytes only;new current policy and '
        'source are pinned prospectively. Stop first safety/record/ownership/budget failure;'
        'no retries,hash refresh or criteria change. Dedicated prefill and physical262144 '
        'qualification remain separate.')
    row['retention'] = ('Keep per-process samples,logical format/NT work signatures,CPU operation '
        'totals,top3 GU and top3 DOWN CPU bins,and nine caller regions with nested waits. '
        'No full per-thread/per-layer or repeated histograms. Retire successful raw logs/JIT '
        'after pushed results/source recovery. No new engine generation.')
    ctx.save(row)
    print('Prepared focused same-engine static resource cohort;commit before frontend probes.', flush=True)


def run(ctx, index):
    ctx.run(index)
    row = json.loads((ctx.ROOT / 'record.json').read_text())
    proc = row['processes'][index]
    proc['native_route_check_passed'] = proc['passed']
    try:
        require(ctx.boundary_calls == 2 and ctx.resource_prefixes == [4], 'strict warmup prefix required')
        root = Path(row['plans'][index]['root'])
        path, = (root / 'bench').glob('*/requests.jsonl')
        samples = [e['sample'] for e in map(json.loads, path.read_text().splitlines())
                   if e['kind'] == 'result']
        enabled = ARMS[index][0] == 'candidate'
        work, clocks, leaves, callers = evidence(root, (root / 'engine.stderr').read_text(), samples, enabled)
        require(len(work) == len(clocks) == len(leaves) == len(samples) == 5, 'all five model records required')
        request = leaf.retained_request(work[-1], clocks[-1], leaves[-1], samples[-1], row['workload'])
        bins = request.pop('leaf_bins')
        request['dominant_CPU_bins'] = {op: [{k: v[k] for k in ('gu', 'down', 'nt', 'jobs',
            'calls', 'rows', 'token_rows', 'wall_ns', 'cpu_ns')} for v in sorted(
                (v for v in bins if v['op'] == op), key=lambda v: v['cpu_ns'], reverse=True)[:3]]
            for op in ('GU', 'DOWN')}
        if enabled:
            request['caller'] = caller.compact(callers[-1])
        request['decode_timing'] = {k: v for k, v in clocks[-1].items() if k not in ('work', 'clock_binding')}
        proc['resources'] = dict(passed=True, completed_requests=5, warmup_prefix=4,
            caller_enabled=enabled, caller_records=len(callers), long_request=request,
            physical_traffic_qualified=False, hardware_saturation_qualified=False,
            total_observer_overhead_qualified=False, clean_speed_qualified=False)
        if len(row['processes']) == 6:
            row['decision'] = decision(row['processes'])
    except BaseException as error:
        proc.update(passed=False, resource_error=repr(error))
        row.update(stage='failed', passed=False, complete=False, resource_error=repr(error))
        raise
    finally:
        ctx.save(row)
    print('Current-engine static caller/leaf/census joined for process' + str(index), flush=True)


if __name__ == '__main__':
    args = sys.argv[1:]
    require(len(args) == 2 and args[0] in clean.CAPS and (args[1] in ('prepare', 'host') or
        args[1].isdigit() and int(args[1]) < 6), 'use regular-code|varied-prose prepare|host|INDEX0..5')
    index = int(args[1]) if args[1].isdigit() else None
    with (clean.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        clean.positive.fcntl.flock(lock, clean.positive.fcntl.LOCK_EX | clean.positive.fcntl.LOCK_NB)
        clean.base.guards.no_model_owners([])
        ctx = context(args[0], index)
        if args[1] == 'prepare':
            ctx.prepare_resources()
        elif args[1] == 'host':
            ctx.comparison.host(ctx)
        else:
            run(ctx, index)
