#!/usr/bin/env python3
"""Source-prepared adaptive CPU leaf cost/attribution; repository LICENSE.

Main-authored. Reuse owned benchmark execution and the independently tested
leaf/footer reader. No new native engine, static census or GPU instrumentation.
"""
import copy
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics
import sys

sys.dont_write_bytecode = True
import adaptive_cpu_leaf_evidence as leaves
import compare_adaptive_payload96_generation as speed

clean, require = speed.clean, speed.require
SELF = Path(__file__).resolve()
ORDER = ('reference', 'candidate', 'candidate', 'reference', 'reference', 'candidate')
MAX_RATIO = 1.05
PRODUCERS = ('include/strata/kernels/cpu/service_profile.hpp', 'src/kernels/cpu/pool.cpp')
POLICY = ('AGENTS.md', 'docs/ARTIFACT_RETENTION.md')
FIXTURE = 'tools/test_adaptive_cpu_leaf_evidence.py'


def choice(workload):
    data = clean.base.committed(clean.RESULT)
    key = 'adaptive_payload96_generation_' + workload.replace('-', '_') + '_v1'
    row = data[key]
    require(row['stage'] == 'closed' and row['passed'] and row['complete'] and
            row['ownership_closed'] and not row['active'] and len(row['processes']) == 6 and
            all(p['passed'] and p['complete'] and p['ownership_closed'] and
                p['native_exit_code'] == 0 and not p['fresh_faults'] for p in row['processes']),
            'close this workload32/96cohort before a resource plan')
    budget = 96 if row['decision']['repeated_TG_gain'] else 32
    return budget, dict(key=key, decision_sha256=clean.signature(row['decision']),
                       budget=budget, scope='Focused decode candidate only;not inference adoption.')


def prerequisites(workload, budget, full=False):
    chosen, reference = choice(workload)
    require(chosen == budget, 'frozen decode budget choice changed')
    normal, prior, contract, manifest, providers, refs = speed.prerequisites(workload, full=full)
    build = json.loads((clean.B / 'adaptive-payload-audit-model-v2/record.json').read_text())
    require(build['binaries']['strata'] == normal['binary'] and
            '-DSTRATA_CPU_SERVICE_PROFILE:BOOL=ON' in build['recipe']['configure'],
            'same profile-enabled native engine required')
    for name in PRODUCERS:
        require(clean.digest(clean.P / name) == build['native_sources'][name] ==
                clean.digest(clean.B / 'adaptive-payload-audit-model-v2/source' / name),
                'compiled CPU leaf producer changed: ' + name)
    proof = clean.base.committed(clean.RESULT)['adaptive_cpu_leaf_reader_host_v1']
    require(proof['stage'] == 'closed' and proof['passed'] and proof['complete'] and
            proof['ownership_closed'] and not proof['active'] and not proof['GPU_executed'] and
            len(proof['tests']) == 1 and proof['tests'][0]['tests'] == 5 and
            proof['tests'][0]['exit_code'] == 0 and proof['tests'][0]['skipped'] == 0,
            'qualified CPU leaf/footer reader required')
    for name, sha in proof['sources'].items():
        if name not in POLICY:
            require(clean.digest(clean.P / name) == sha, 'leaf reader implementation changed: ' + name)
    refs.update(decode_budget=reference, leaf_reader=clean.signature(proof))
    return normal, prior, contract, manifest, providers, refs


def evidence(text, samples, width, enabled):
    headers = list(re.finditer(leaves.footer.HEADER, text, re.M))
    copies = list(re.finditer(leaves.footer.COPY, text, re.M))
    require(len(headers) == len(copies) == len(samples) and len(samples) > 0 and
            text.count('strata serve: prompt ') == len(headers) and
            text.count('strata serve: SYCL_EXPERT_COPY') == len(copies), 'complete native CPU footers')
    native = []
    for i, (header, copy_, sample) in enumerate(zip(headers, copies, samples)):
        require(tuple(map(int, header.groups())) == tuple(sample[k] for k in leaves.FIELDS) and
                header.end() < copy_.start() and
                (i + 1 == len(headers) or copy_.end() < headers[i + 1].start()),
                'native/client CPU request binding')
        values = dict(zip(leaves.footer.COPY_KEYS, (leaves.census.integer(v) for v in copy_.groups())))
        require(not any(values[k] for k in leaves.footer.COPY_KEYS[:6]) and
                0 <= values['cpu_unique_misses'] <= values['cpu_entries'] <= width * values['cpu_unique_misses'],
                'unexpected CPU/copy work')
        native.append({k: values[k] for k in ('cpu_unique_misses', 'cpu_entries')})
    if enabled:
        parsed = leaves.extract(text, samples, width)
        require(len(parsed) == len(samples), 'complete leaf request prefix required')
    else:
        require(not leaves.service.records(text), 'leaf records in observer-off arm')
        parsed = []
    return native, parsed


def summarize(row):
    operations = {op: dict(calls=0, wall_ns=0, cpu_ns=0) for op in leaves.service.OPS}
    jobs = []
    top = {op: [] for op in ('GU', 'DOWN')}
    for group in row['format_nt']:
        jobs.append({k: group[k] for k in ('gu', 'down', 'nt', 'jobs')})
        for op in operations:
            leaves.add(operations[op], group['operations'][op])
        for op in top:
            top[op].append(dict(gu=group['gu'], down=group['down'], nt=group['nt'],
                                jobs=group['jobs'], **group['operations'][op]))
    return dict(cpu_jobs=row['cpu_jobs'], cpu_entries=row['cpu_entries'],
                concurrent_cpu_ns=row['cpu_ns'], concurrent_wall_ns=row['wall_ns'],
                thread_count=row['thread_count'], operations=operations, format_nt_jobs=jobs,
                top_cpu_bins={op: sorted(values, key=lambda v: v['cpu_ns'], reverse=True)[:3]
                              for op, values in top.items()})


def decision(processes):
    require(len(processes) == 6 and
            len({p['adapt_swaps'] for p in processes}) == 1 and processes[0]['adapt_swaps'] in (32, 96) and
            len({(p['native_identity']['pid'], p['native_identity']['start_ticks'])
                                       for p in processes}) == 6 and
            all(p['passed'] and p['complete'] and p['ownership_closed'] and
                p['native_exit_code'] == 0 and not p['fresh_faults'] and p['resources']['passed'] and
                p['resources']['completed_requests'] == 5 and p['resources']['warmup_prefix'] == 4 and
                p['kernel'] == ORDER[i] and
                p['resources']['leaf_enabled'] == (ORDER[i] == 'candidate') for i, p in enumerate(processes)),
            'six closed leaf observer processes required')
    ratios, matches = [], []
    for start in (0, 2, 4):
        pair = {p['kernel']: p for p in processes[start:start + 2]}
        a, b = pair['candidate']['samples'][0], pair['reference']['samples'][0]
        require(min(a['decode_ms'], b['decode_ms'], a['generated_tokens'], b['generated_tokens']) > 0,
                'positive observer timing extent')
        ratio = (a['decode_ms'] / a['generated_tokens']) / (b['decode_ms'] / b['generated_tokens'])
        require(math.isfinite(ratio) and ratio > 0, 'finite observer latency')
        ratios.append(ratio)
        matches.append(pair['candidate']['resources']['work_signature_sha256'] ==
                       pair['reference']['resources']['work_signature_sha256'])
    return dict(paired_leaf_on_over_off_latency=ratios, paired_work_matches=matches,
                geometric_mean_latency_ratio=math.exp(statistics.mean(map(math.log, ratios))),
                max_per_pair_latency_ratio=MAX_RATIO,
                leaf_observer_cost_qualified=all(matches) and all(v <= MAX_RATIO for v in ratios),
                limits='Logical aggregate and long-output hashes constrain this cost comparison,not task quality. '
                       'Concurrent CPU leaf sums include memory stalls;not additive critical path,physical RAM/PCIe '
                       'traffic,arithmetic saturation,clean speed,dynamic-map conservation or inference adoption.')


def context(workload, budget, index=None):
    require(workload in clean.CAPS and budget in (32, 96), 'known workload/qualified adaptive budget required')
    spec = importlib.util.spec_from_file_location('strata_adaptive_cpu_leaf_profile', clean.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ARMS = tuple((arm, budget) for arm in ORDER)
    module.SELF, module.basis, module.controls = SELF, lambda full=False: prerequisites(workload, budget, full), controls
    original_environment = module.environment

    def environment(row, plan):
        require((plan['kernel'], plan['adapt_swaps']) == module.ARMS[plan['index']], 'fixed cache policy required')
        env = original_environment(row, plan)
        env.update(STRATA_CPU_SERVICE=str(int(plan['kernel'] == 'candidate')),
                   STRATA_CPU_WORK_CENSUS='0', STRATA_SYCL_DECODE_SERVICE_PROFILE='0')
        return env

    def invariant(row, plan):
        env = environment(row, plan)
        env.pop('STRATA_CPU_SERVICE')
        value = dict(configuration=module.configuration(row, plan), environment=env)
        return clean.signature(json.loads(json.dumps(value).replace(plan['root'], '<process>')))

    def source_pins(revision):
        pins = clean.source_pins(revision)
        pins.update(clean.base.controls(revision, controller=SELF))
        for name in (*PRODUCERS, *POLICY, FIXTURE):
            raw = clean.base.basis.build.git(clean.P, 'show', revision + ':' + name)
            require(raw == (clean.P / name).read_bytes(), 'commit leaf profile input: ' + name)
            pins[name] = clean.digest(clean.P / name)
        return pins

    module.environment, module.invariant, module.source_pins = environment, invariant, source_pins
    ctx = module.context(workload)
    ctx.ROOT = clean.B / ('adaptive-cpu-leaves-' + workload + '-v1')
    ctx.KEY = 'adaptive_cpu_leaves_' + workload.replace('-', '_') + '_v1'
    ctx.WORKLOAD_ID = 'adaptive_cpu_leaves_20261107_' + workload
    ctx.configuration, ctx.invariant = module.configuration, invariant
    original_current = ctx.current

    def current(row):
        require(row['selected_decode_budget'] == budget, 'selected leaf-profile cache budget changed')
        return original_current(row)

    ctx.current = current
    ctx.boundary_calls, ctx.resource_prefixes = 0, []
    original_boundary, original_clean = ctx.cache_boundary, ctx.clean_observers
    enabled = index is not None and ORDER[index] == 'candidate'

    def diagnostic(text):
        records = leaves.service.records(text)
        require(enabled or not records, 'observer-off arm emitted leaf records')
        for _, kind, values in records:
            if kind == 'CPU_SERVICE':
                require({k: v for k, v in values.items() if k != 'request'} == leaves.service.HEADER,
                        'invalid native leaf profile')
        original_clean('\n'.join(line for line in text.splitlines()
                                 if not line.startswith('strata CPU_SERVICE')))

    def boundary(root, expected):
        ordinal = ctx.boundary_calls
        require(index is not None and ordinal < 2 and expected == (4, 5)[ordinal], 'leaf request gate order')
        raw, history = original_boundary(root, expected)
        diagnostic(raw.decode())
        if ordinal == 0:
            path, = (root / 'bench').glob('*/requests.jsonl')
            samples = [e['sample'] for e in map(json.loads, path.read_text().splitlines()) if e['kind'] == 'result']
            require(len(samples) == 4, 'all four warmups before leaf-profile long request')
            evidence(raw.decode(), samples, clean.CAPS[workload], enabled)
            ctx.resource_prefixes.append(4)
        ctx.boundary_calls += 1
        return raw, history

    ctx.cache_boundary, ctx.clean_observers = boundary, diagnostic
    ctx.decision = lambda _: dict(model_execution_closed=True, leaf_binding_pending=True,
                                  leaf_observer_cost_qualified=False, inference_adopted=False)
    ctx.prepare_leaves = lambda: prepare(module, ctx, budget)
    return ctx


def prepare(module, ctx, budget):
    module.prepare(ctx)
    row = json.loads((ctx.ROOT / 'record.json').read_text())
    row['selected_decode_budget'] = budget
    row['plan'] = ('Six fresh owned processes leaf0/1,1/0,0/1 on unchanged e1b011,fixed selected adaptive budget,'
        'every8rounds/default decay0.7. Close this workload32/96cohort;choose96only after its repeated TGgain,'
        'else32. Four official warmups then first v5thinking-on32K+/1K+generation with an answer;'
        'cap4code/cap2prose,16Koutput allowance. Only pre-exec CPU_SERVICE differs. Static census,caller,'
        'decode timing,full trace and other heavy observers stay off;all runtime safety,placement and precision '
        'remain fixed. Qualify the CPU reader first. Bind four warmup leaf/native records before the long '
        'request and all five at normal exit. Keep source/body/actual route/head/clock/provider/quiet-validation/'
        'owner/no-fault gates and15asset hashes. Compare exact per-token latency;all3pairs<=1.05 with '
        'matching aggregate CPU/token/draft work and long-output hashes qualify observer cost only. '
        'Hash/work mismatch is inconclusive for cost,not task-quality failure. Keep all samples;no retries,'
        'exclusions,pin refresh or criteria change. Commit each closed process before the next.')
    row['retention'] = ('Keep samples,limited work signatures,three operation totals,format/NT job counts and '
        'top3GU/Down CPU bins for the named current working-set calibration. Do not retain full warmup or '
        'per-thread/per-layer histograms. Retire successful raw output after verification/source recovery. '
        'No new native build or inference adoption;physical traffic,critical path and saturation stay unknown.')
    ctx.save(row)


def run(ctx, index):
    ctx.run(index)
    row = json.loads((ctx.ROOT / 'record.json').read_text())
    process = row['processes'][index]
    process['native_route_check_passed'] = process['passed']
    try:
        require(ctx.boundary_calls == 2 and ctx.resource_prefixes == [4], 'qualified leaf warmup prefix required')
        root = Path(row['plans'][index]['root'])
        path, = (root / 'bench').glob('*/requests.jsonl')
        samples = [e['sample'] for e in map(json.loads, path.read_text().splitlines()) if e['kind'] == 'result']
        enabled = ORDER[index] == 'candidate'
        native, parsed = evidence((root / 'engine.stderr').read_text(), samples, row['actual_width_cap'], enabled)
        require(len(samples) == 5, 'exact five completed leaf requests')
        long = process['samples'][0]
        signature = dict(native_cpu_work=native,
                         tokens=[{k: s[k] for k in leaves.FIELDS} for s in samples],
                         long_output={k: long[k] for k in ('request_sha256', 'output_sha256', 'reasoning_sha256')})
        process['resources'] = dict(passed=True, leaf_enabled=enabled, completed_requests=5, warmup_prefix=4,
                                    work_signature_sha256=clean.signature(signature),
                                    long_request=summarize(parsed[-1]) if enabled else None)
        if index == 5:
            row['decision'] = decision(row['processes'])
    except BaseException as error:
        process.update(passed=False, complete=False, resource_error=repr(error))
        row.update(stage='failed', passed=False, complete=False)
    finally:
        ctx.save(row)
    require(process['passed'], process.get('resource_error', 'leaf resource binding failed'))


def controls():
    spec = importlib.util.spec_from_file_location('strata_leaf_control_fixture', clean.P / FIXTURE)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    text, sample = fixture.fixture()
    native, parsed = evidence(text, [sample], 4, True)
    quiet = '\n'.join(line for line in text.splitlines() if not line.startswith('strata CPU_SERVICE'))
    off_native, off_parsed = evidence(quiet, [sample], 4, False)
    require(native == off_native and not off_parsed, 'shared native work must survive observer-off parsing')
    compact = summarize(parsed[0])
    require(compact['cpu_jobs'] == 5 and compact['cpu_entries'] == 14 and
            sum(v['cpu_ns'] for v in compact['operations'].values()) == compact['concurrent_cpu_ns'] == 155,
            'compact leaf summary must conserve operation CPU time')
    for changed, enabled in ((text, False), (quiet, True)):
        try: evidence(changed, [sample], 4, enabled)
        except RuntimeError: continue
        raise RuntimeError('wrong leaf observer presence accepted')
    rows = [dict(kernel=arm, adapt_swaps=32, passed=True, complete=True, ownership_closed=True, native_exit_code=0,
                 fresh_faults=[], native_identity=dict(pid=i + 1, start_ticks=i + 1),
                 samples=[dict(generated_tokens=2048, decode_ms=1040 if arm == 'candidate' else 1000)],
                 resources=dict(passed=True, completed_requests=5, warmup_prefix=4,
                                leaf_enabled=arm == 'candidate', work_signature_sha256='same'))
            for i, arm in enumerate(ORDER)]
    require(decision(rows)['leaf_observer_cost_qualified'], 'bounded leaf cost control')
    for change in ('latency', 'work', 'owner', 'budget'):
        bad = copy.deepcopy(rows)
        if change == 'latency': bad[1]['samples'][0]['decode_ms'] = 1051
        elif change == 'work': bad[1]['resources']['work_signature_sha256'] = 'changed'
        elif change == 'owner': bad[1]['native_identity'] = bad[0]['native_identity']
        else: bad[1]['adapt_swaps'] = 96
        if change in ('owner', 'budget'):
            try: decision(bad)
            except RuntimeError: continue
            raise RuntimeError('reused observer process or changed cache budget accepted')
        require(not decision(bad)['leaf_observer_cost_qualified'], 'retain failed/inconclusive observer cost')
    prior = json.loads((clean.positive.width.SPEED_ROOT / 'record.json').read_text())
    env = clean.positive.width.generation_context(0).environment(prior, prior['plans'][0])
    original = clean.signature(dict(configuration=prior['configuration'], environment=env))
    for workload in clean.CAPS:
        for budget in (32, 96):
            ctx = context(workload, budget)
            row = dict(configuration=copy.deepcopy(prior['configuration']), environment_template=env, workload=workload)
            plans = [dict(index=i, kernel=ORDER[i], adapt_swaps=budget, heads=1,
                          root=str(ctx.ROOT / ('control-' + str(i)))) for i in (0, 1)]
            require(ctx.invariant(row, plans[0]) == ctx.invariant(row, plans[1]), 'only leaf flag/private paths may differ')
            for i, plan in enumerate(plans):
                actual = ctx.environment(row, plan)
                args = ctx.configuration(row, plan)['args']
                require(actual['STRATA_CPU_SERVICE'] == str(i) and actual['STRATA_CPU_WORK_CENSUS'] == '0' and
                        actual['STRATA_SYCL_DECODE_SERVICE_PROFILE'] == '0' and
                        not any(k in actual for k in clean.OBSERVERS) and
                        actual['EnableDirectSubmission'] == '0' and actual['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD'] == '1',
                        'immutable leaf-only observers and runtime safety required')
                require(args[args.index('--adapt-swaps') + 1] == str(budget) and
                        args[args.index('--adapt-every') + 1] == '8' and
                        args[args.index('--mtp-max-t') + 1] == str(clean.CAPS[workload]),
                        'actual cache cadence,budget and width selectors required')
    require(clean.signature(dict(configuration=prior['configuration'], environment=env)) == original,
            'CPU controls mutated original configuration/environment')
    return dict(passed=True, GPU_executed=False, immutable_environment_controls=8,
                over_budget_retained=True, work_mismatch_inconclusive=True,
                reused_owner_refused=True, changed_cache_budget_refused=True,
                leaf_fixture_on_off=True, observer_presence_mismatch_refused=True,
                summary_conservation=True)


if __name__ == '__main__':
    args = sys.argv[1:]
    require(args == ['controls'] or len(args) == 2 and args[0] in clean.CAPS and
            (args[1] in ('prepare', 'host') or args[1].isdigit() and int(args[1]) < 6),
            'use controls | regular-code|varied-prose prepare|host|INDEX0..5')
    with (clean.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        clean.positive.fcntl.flock(lock, clean.positive.fcntl.LOCK_EX | clean.positive.fcntl.LOCK_NB)
        clean.base.guards.no_model_owners([])
        if args == ['controls']:
            revision = clean.base.basis.build.git(clean.P, 'rev-parse', 'HEAD').decode().strip()
            pins = clean.base.controls(revision, controller=SELF)
            raw = clean.base.basis.build.git(clean.P, 'show', revision + ':' + FIXTURE)
            require(raw == (clean.P / FIXTURE).read_bytes(), 'commit CPU leaf control fixture')
            pins[FIXTURE] = clean.digest(clean.P / FIXTURE)
            print(json.dumps(dict(controls(), source_commit=revision, source_count=len(pins),
                                  source_signature_sha256=clean.signature(pins)), sort_keys=True))
        else:
            budget, _ = choice(args[0])
            ctx = context(args[0], budget, int(args[1]) if args[1].isdigit() else None)
            if args[1] == 'prepare': ctx.prepare_leaves()
            elif args[1] == 'host': ctx.comparison.host(ctx)
            else: run(ctx, int(args[1]))
