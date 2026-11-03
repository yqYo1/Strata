#!/usr/bin/env python3
"""Current native generation's request-length/TTFT curve; repository LICENSE.

Use the maintained bench.sh exact-count mode and serialized request gates.
Progress-observed curves and a clean117K anchor are descriptive,not adoption
or a causal speed comparison. No native rebuild or shared context mutation.
"""
import ast
import copy
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import socket
import statistics
import sys
from functools import lru_cache

sys.dont_write_bytecode = True
import cache_map_prefill_speed as speed
import prefill_scaling_bench as fit

entry, base, comparison = speed.entry, speed.base, speed.comparison
P, B, RESULT = speed.P, speed.B, speed.RESULT
ROOT = entry.model.ROOT / 'prefill-curve-v5'
KEY = 'cache_map_prefill_curve_v5'
PARSER_KEY = 'cache_map_prefill_curve_short_route_v4'
FAILED_KEY = 'cache_map_prefill_curve_v3'
SOURCE = 'tools/sycl/cache_map_prefill_curve.py'
SNAPSHOT = '2880a857f40c94d8bc11824fa513466368079858'
EXISTING_HELPER = 'tools/sycl/cache_map_prefill_resources.py'
HELPER_COMMIT = 'b13d99d13bbd975aa8e723907b2f430c64e9e3a0'
CHANGED = ('tools/sycl/build_cache_map_model.py', 'tools/sycl/compare_official_prefill.py',
           'tools/sycl/official_model_entry.py')
BENCHMARK_CHANGED = {'tools/strata_bench.py', 'tools/test_strata_bench.py'}
CASES = ((1286, True), (117489, False), (117489, True), (39254, True),
         (4956, True), (78445, True), (19665, True), (65537, True),
         (24577, True), (8193, True))
ORDER, RUNS, MIN_PROMPT_MS = [(n, 1) for n, _ in CASES], 4, 0
require, digest, write, compact = speed.require, speed.digest, speed.write, speed.compact


def save(row):
    write(ROOT / 'record.json', row)
    data = json.loads(RESULT.read_text()); data[KEY] = compact(row); write(RESULT, data)


def pinned_sources(revision):
    pins = speed.pinned_sources(revision)
    pins.update(base.controls(revision, controller=Path(__file__).resolve()))
    return pins


@lru_cache(maxsize=1)
def original_evidence():
    return json.loads(speed.shared.gen.git('show', SNAPSHOT + ':' + str(RESULT.relative_to(P))))


def qualification(revision):
    # This is an explicit controller-only reuse,not a relabeling of native code.
    native = entry.model.verify(reuse_revision=revision, controller_changes=CHANGED)
    prior = json.loads((speed.ROOT / 'record.json').read_text())
    expected = original_evidence()[speed.KEY]
    def qualification_fields(value):
        return {k: v for k, v in value.items() if k not in
                ('closed_output_retirement', 'closed_qualification_snapshot')}
    require(qualification_fields(compact(prior)) == qualification_fields(expected)
            == qualification_fields(base.committed(RESULT)[speed.KEY]),
            'original closed clean qualification changed beyond ended-consumer metadata')
    require(prior['stage'] == 'closed' and prior['passed'] and prior['complete'] and
            prior['ownership_closed'] and not prior['active'] and prior['host_passed'] and
            prior['decision'] == speed.decision(prior['processes']) and
            prior['binary'] == native['binaries']['strata'] and
            prior['native_source_commit'] == native['source_commit'], 'same qualified native generation required')
    recovered = {n: hashlib.sha256(raw).hexdigest() for n, raw in
                 speed.shared.gen.blobs(P, prior['source_commit'], prior['sources'])}
    current = pinned_sources(revision)
    helper_sha = hashlib.sha256(speed.shared.gen.git('show', HELPER_COMMIT + ':' + EXISTING_HELPER)).hexdigest()
    require(recovered == prior['sources'] and set(current) - set(recovered) == {SOURCE, EXISTING_HELPER}
            and not set(recovered) - set(current) and
            {n for n in recovered if recovered[n] != current[n]} == set(CHANGED) | BENCHMARK_CHANGED
            and current[EXISTING_HELPER] == helper_sha,
            'only declared policy changes,new curve and unchanged committed resource helper may extend clean inputs')
    inv = prior['runtime_manifest']; providers = json.loads(Path(inv['path']).read_text())
    require(digest(inv['path']) == inv['sha256'] and len(providers) == inv['files'],
            'qualified native provider inventory changed')
    contract, manifest = entry.template.python_manifest()
    require(prior['python_contract'] == contract and prior['python_selected'] == manifest['selected'],
            'qualified Python provider changed')
    return native, providers, manifest, prior


def native_context(row, plan):
    require(type(plan['index']) is int and 0 <= plan['index'] < len(CASES) and
            (plan['read'], plan['trace_progress']) == CASES[plan['index']] and plan['heads'] == 1,
            'unplanned curve process')
    return {k: row[k] for k in ('binary', 'native_source_commit', 'native_validation_probe', 'boot')}


def environment(row, plan):
    native_context(row, plan)
    env = comparison.environment(row, plan)
    require(all(env.get(k) == v for k, v in dict(STRATA_SYCL_ADAPT_SYNC='1',
        STRATA_MTP_BATCH_F16='1', STRATA_SYCL_PREFILL_GDN_FP16='1',
        STRATA_SYCL_PREFILL_GDN_BARRIERS3='1', STRATA_SYCL_PREFILL_GDN_COLUMNS='4',
        STRATA_SYCL_PREFILL_GDN_CHUNK64='0', STRATA_SYCL_PREFILL_DOWN_SPLIT_KERNEL='original').items()),
        'preserve qualified typed kernels and runtime settings')
    if plan['trace_progress']:
        env['STRATA_TRACE'] = '1'
    return env


def records(text, prefix):
    found = []
    for line in text.splitlines():
        if line.startswith(prefix):
            pairs = re.findall(r'([A-Za-z_][A-Za-z_0-9-]*)=([^\s;]+)', line)
            require(len(pairs) == len(dict(pairs)), 'duplicate progress field')
            found.append(dict(pairs))
    return found


def request_route(text, plan):
    n = plan['read'] - 1
    if not plan['trace_progress']:
        require(plan['read'] == 117489, 'only the matched117K anchor may be clean')
        return speed.request_route(text, dict(plan, kernel='candidate'))
    long = n > 8192
    policies = records(text, 'strata prefill policy: ')
    require(len(policies) == 1 and set(policies[0]) == {'requested', 'route', 'tokens', 'chunk'},
            'missing/duplicate actual request policy')
    policy = policies[0]; chunk = int(policy['chunk'])
    require(policy['requested'] == 'throughput' and policy['tokens'] == str(n)
            and policy['route'] == ('layer-major' if long else 'chunked')
            and (chunk == 8192 if long else n <= chunk <= 8192),
        'actual request strategy/route/extent differs')
    states = records(text, 'strata prefill dense: phase=')
    require([s['phase'] for s in states] == ['retired', 'bank' if long else 'cache-loan', 'restored'],
            'dense/cache-loan phase ordering differs')
    require(all(s['residency'] == 'Full' and s['activations'] == '0' for s in states)
            if not long else states[0]['residency'] == states[-1]['residency'] == 'Full'
            and states[1]['residency'] == 'Bank' and states[-1]['activations'] == '48',
            'actual phase owners or activation count differ')
    mtp = entry.template.mtp_route(text, n)
    if not long:
        require(not any(s in text for s in ('strata prefill: layer-major candidate, ',
            'strata prefill heads: target detached ', 'strata prefill heads: draft detached ',
            'strata mtp: prompt expert phase=', 'strata GDN dispatch: ', 'strata GDN storage: ')),
            'short input entered a dedicated long-prefill route')
        return dict(rows=n, actual_route='chunked', actual_operator_chunk=chunk, dense_activations=0,
            dense_head_MTP_weights_retained=True, cache_loan_retired_and_restored=True, MTP_prefill=mtp)
    route = comparison.route(text, plan['read'], 1, expected_rows=n)
    require(route['residual_gpu_rows'] == n and route['residual_host_bytes'] == 0,
            'curve requires the declared full-GPU long route')
    speed.routes.gdn_selection(text, 'candidate'); speed.routes.native16_route(text, plan['read'])
    progress = entry.template.original.progress(text, rows=n, split_active=route['split'])
    require(progress['jobs'] == route['jobs'], 'progress and work census disagree')
    chunks = math.ceil(n / 8192)
    storage = [int(v) for v in re.findall(r'^strata GDN storage: fp16;state=fp32;tokens=(\d+)$', text, re.M)]
    calls = records(text, 'strata GDN dispatch: ')
    require(len(storage) == len(calls) == 36 * chunks and sum(storage) == 36 * n
            and all(0 < t <= 8192 for t in storage) and
            sum(int(v['tokens']) for v in calls) == 36 * n and
            all(v['schedule'] == 'barriers3' and v['partitions'] == '4' and
                0 < int(v['tokens']) <= 8192 for v in calls), 'typed GDN extent differs')
    route.update(actual_route='layer-major', actual_operator_chunk=chunk, checkpoint_pairs=progress['checkpoint_pairs'],
                 GDN_FP16_storage_calls=len(storage), GDN3_calls=len(calls), MTP_prefill=mtp)
    return route


def validate_model_text(text, plan):
    if plan['read'] > 8193:
        speed.validate_model_text(text, plan)


def decision(processes):
    require(len(processes) == len(CASES) and all(p['index'] == i and p['read'] == CASES[i][0]
            and p['clean_speed'] == (not CASES[i][1]) and p['complete'] and p['passed']
            and p['ownership_closed'] and not p['fresh_faults'] and len(p['samples']) == RUNS
            for i, p in enumerate(processes)), 'complete ordered length cohort required')
    require(len({(p['native_identity']['pid'], p['native_identity']['start_ticks']) for p in processes})
            == len(CASES), 'native process identity reused')
    points = []
    for p in processes:
        if not CASES[p['index']][1]:
            continue
        routes = p['routes']
        require(len(routes) == RUNS and len({(r['actual_route'], r['actual_operator_chunk']) for r in routes}) == 1,
                'request route changed within a length')
        points.append(dict(actual_read_tokens=p['read'], actual_route=routes[0]['actual_route'],
            actual_operator_chunk=routes[0]['actual_operator_chunk'],
            staged=routes[0].get('staged', False), split=routes[0].get('split', False),
            prompt_ms=p['subsequent_median_ms'],
            ttft_ms=1000 * statistics.median(s['ttft_s'] for s in p['samples'][1:])))
    points.sort(key=lambda x: x['actual_read_tokens'])
    groups = {name: [p for p in points if predicate(p)] for name, predicate in
              [('chunked', lambda p: p['actual_route'] == 'chunked'),
               ('layer-major-bank384', lambda p: p['actual_route'] == 'layer-major' and not p['split'])]}
    models = {}
    for name, group in groups.items():
        models[name] = dict(prompt_affine=fit.linear_fit(group), prompt_models=fit.approximation_models(group),
            TTFT_models=fit.approximation_models([dict(p, prompt_ms=p['ttft_ms']) for p in group]))
    clean, observed = processes[1], processes[2]
    ratio = observed['subsequent_median_ms'] / clean['subsequent_median_ms']
    return dict(points=points, route_specific_models=models,
        observed_over_clean117K=ratio, observer_cost_screen_passed=ratio <= 1.15,
        adjacent_ms_per_token=[dict(from_tokens=a['actual_read_tokens'], to_tokens=b['actual_read_tokens'],
            prompt=(b['prompt_ms']-a['prompt_ms'])/(b['actual_read_tokens']-a['actual_read_tokens']),
            TTFT=(b['ttft_ms']-a['ttft_ms'])/(b['actual_read_tokens']-a['actual_read_tokens']),
            same_route_and_split=(a['actual_route'], a['split']) == (b['actual_route'], b['split']))
            for a, b in zip(points, points[1:])],
        inference_adopted=False, paired_gain_qualified=False, physical_saturation_qualified=False,
        limits='Progress-observed descriptive curve;one independent process per length,first use separate from3subsequent requests. The clean/observed cost screen covers117K only,not short inputs. Never infer physical fixed cost from mixed routes or extrapolate past measured counts. Split117K remains distinct. No sustained TG,task/agent,physical262144/RESTORE or inference adoption.')


def selfcheck():
    inherited = dict(geometry=comparison.selfcheck(), progress=entry.template.original.selfcheck())
    plan = dict(read=1286, trace_progress=True)
    text = ('strata prefill policy: requested=throughput route=chunked tokens=1285 chunk=1536\n'
        'strata prefill dense: phase=retired residency=Full activations=0\n'
        'strata prefill dense: phase=cache-loan residency=Full activations=0\n'
        'strata serve: DRAFT_PREFILL path=batched mode=2 cells=1285\n'
        'strata prefill dense: phase=restored residency=Full activations=0\n')
    require(request_route(text, plan)['rows'] == 1285, 'short-route positive control')
    bad = [text + text, text.replace('tokens=1285', 'tokens=1284'),
        text.replace('route=chunked', 'route=layer-major'), text.replace('residency=Full', 'residency=Bank'),
        text.replace('activations=0', 'activations=1'), text.replace('cells=1285', 'cells=1284'),
        text.replace('phase=restored', 'phase=missing'), text + 'strata GDN dispatch: tokens=1285\n',
        text.replace('chunk=1536', 'chunk=1024'), text.replace('chunk=1536', 'chunk=0'),
        text.replace('chunk=1536', 'chunk=8193')]
    require(request_route(text.replace('chunk=1536', 'chunk=8192'), plan)['actual_operator_chunk'] == 8192,
            'valid larger short scratch lease rejected')
    for value in bad:
        try: request_route(value, plan)
        except (RuntimeError, ValueError, KeyError): continue
        raise RuntimeError('invalid short-route evidence accepted')
    sample = dict(ok=True, eligible=True, stream_complete=True, thinking_requested=True,
        reasoning_observed=True, actual_read_tokens=1286, actual_prompt_tokens=1286,
        actual_reused_tokens=0, request_sha256='a'*64, generated_tokens=24,
        prompt_tps_source='engine', prompt_ms=1000, ttft_s=1.1, total_s=2)
    count = dict(counted_input_tokens=1286, body_sha256='a'*64)
    comparison.check_sample(sample, count, 1286, MIN_PROMPT_MS)
    invalid = [dict(sample, **p) for p in (dict(prompt_ms=0), dict(prompt_ms=float('nan')),
        dict(ttft_s=3), dict(reasoning_observed=False), dict(actual_reused_tokens=1),
        dict(request_sha256='b'*64), dict(actual_read_tokens=1285))]
    for value, floor in [(sample, 10000)] + [(v, MIN_PROMPT_MS) for v in invalid]:
        try: comparison.check_sample(value, count, 1286, floor)
        except RuntimeError: continue
        raise RuntimeError('invalid clock/workload or changed legacy floor accepted')
    synthetic = [dict(actual_read_tokens=n, prompt_ms=10000 + n * .75)
                 for n in (19665, 24577, 39254, 65537, 78445)]
    require(abs(fit.linear_fit(synthetic)['ms_per_token'] - .75) < 1e-9
            and fit.approximation_models(synthetic)['candidates']['affine']['max_held_out_relative_error'] < 1e-9,
            'length model/held-out controls failed')
    for read, _ in CASES:
        argv = ['--prefill-exact-tokens', str(read), '--think', '--runs', str(RUNS),
                '--profile-request-gates', '--workload-id', 'native_gdn_20261018', '--fail-fast-engine-error']
        if read < 32768: argv.append('--prefill-short-diagnostic')
        args = speed.shared.client.parse_arguments(argv)
        require(args.prefill_only and args.think and args.prefill_exact_tokens == read
                and args.prefill_short_diagnostic == (read < 32768), 'actual frozen CLI selection differs')
    return dict(passed=True, inherited=inherited, short_route_negative=len(bad),
                workload_clock_negative=len(invalid)+1, actual_CLI_vectors=len(CASES),
                legacy_clock_floor_preserved=True, GPU_executed=False)


def scope_projection(source):
    """Only scope labels and their proof reader may differ from the replayed code."""
    nodes = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.Import) and len(node.names) == 1 and \
                node.names[0].name == 'ast' and node.names[0].asname is None:
            continue
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and \
                isinstance(node.targets[0], ast.Name) and node.targets[0].id in ('ROOT', 'KEY'):
            name = node.targets[0].id
            declarations = [f"ROOT = entry.model.ROOT / 'prefill-curve-v{v}'" if name == 'ROOT'
                            else f"KEY = 'cache_map_prefill_curve_v{v}'" for v in (4, 5)]
            require(ast.dump(node, include_attributes=False) in
                    [ast.dump(ast.parse(s).body[0], include_attributes=False) for s in declarations],
                    'scope declaration contains more than an approved label')
            nodes.append(name + ' versioned scope label')
            continue
        if isinstance(node, ast.FunctionDef) and node.name in ('scope_projection', 'parser_qualification'):
            continue
        nodes.append(ast.dump(node, include_attributes=False))
    return nodes


def parser_qualification():
    parser = base.committed(RESULT)[PARSER_KEY]
    fields = ('source_commit', 'passed', 'complete', 'ownership_closed',
        'GPU_executed', 'controller_sha256', 'fixture', 'failed_scope', 'actual_route')
    proof_commit = '7532cbe1b4f8c8e448df58673689faa30ffc1cf8'
    original = json.loads(speed.shared.gen.git('show', proof_commit + ':' + str(RESULT.relative_to(P))))[PARSER_KEY]
    frozen = {k: parser[k] for k in fields}
    require(comparison.fingerprint(frozen) == comparison.fingerprint({k: original[k] for k in fields}),
            'original immutable parser replay proof changed')
    legacy = speed.shared.gen.git('show', original['source_commit'] + ':' + SOURCE)
    require(parser['passed'] and parser['complete'] and parser['ownership_closed']
            and not parser['GPU_executed'] and parser['controller_sha256'] == hashlib.sha256(legacy).hexdigest()
            and parser['failed_scope'] == FAILED_KEY and parser['actual_route']['rows'] == 1285
            and parser['actual_route']['actual_operator_chunk'] == 1536,
            'recover the unchanged original captured short-route replay')
    source = Path(__file__).read_text()
    require(scope_projection(legacy.decode()) == scope_projection(source),
            'only scope labels and the explicit proof reader may extend the replayed controller')
    frozen['source_equal_reuse'] = dict(current_controller_sha256=digest(__file__),
        original_proof_commit=proof_commit, original_controller_sha256=parser['controller_sha256'],
        unchanged_projection_sha256=comparison.fingerprint(scope_projection(source)),
        allowed_changes=['ast import', 'ROOT/KEY scope labels', 'scope_projection/parser_qualification proof readers'],
        raw_fixture_replayed_again=False, model_or_native_qualification_added=False)
    return frozen


def replay():
    data = base.committed(RESULT); failed = data[FAILED_KEY]
    require(PARSER_KEY not in data and failed['stage'] == 'failed' and not failed['passed']
            and failed['ownership_closed'] and not failed['active'] and
            failed['posthoc_owner_closure']['original_outcome_retained'], 'preserve the closed failed model scope')
    fixture = failed['parser_replay_consumer']; path = Path(fixture['path']); raw = path.read_bytes()
    require(len(raw) == fixture['bytes'] <= 16384 and digest(path) == fixture['sha256']
            and 0 <= fixture['begin'] < fixture['end'] == len(raw), 'original failed capture changed')
    route = request_route(raw[fixture['begin']:fixture['end']].decode(), dict(read=1286, trace_progress=True))
    evidence = dict(source_commit=speed.shared.gen.git('rev-parse', 'HEAD').decode().strip(),
        passed=True, complete=True, ownership_closed=True, GPU_executed=False,
        controller_sha256=digest(__file__), fixture={k: fixture[k] for k in ('path', 'sha256', 'bytes', 'begin', 'end')},
        failed_scope=FAILED_KEY, actual_route=route,
        limits='Parser replay of the original unchanged bytes only;V3remains failed,incomplete and externally terminated. No model,numerical,normal-teardown or speed qualification.')
    data[PARSER_KEY] = evidence; write(RESULT, data)
    print(json.dumps(evidence), flush=True)


def prepare():
    revision = speed.shared.gen.git('rev-parse', 'HEAD').decode().strip()
    controls = selfcheck(); parser = parser_qualification()
    native, providers, manifest, prior = qualification(revision)
    require(not ROOT.exists() and KEY not in base.committed(RESULT), 'one fresh curve only')
    assets = speed.shared.models.model_pins(prior['configuration'], prior['model_assets'], full=True)
    ROOT.mkdir(mode=0o700); write(ROOT/'runtime-inputs.json', providers)
    row = dict(stage='prepared', source_commit=revision, sources=pinned_sources(revision), model_assets=assets,
        binary=native['binaries']['strata'], native_source_commit=native['source_commit'],
        native_validation_probe=prior['native_validation_probe'], configuration=copy.deepcopy(prior['configuration']),
        environment_template=copy.deepcopy(prior['environment_template']), python_contract=prior['python_contract'],
        python_selected=manifest['selected'], template=prior['template'], boot=prior['boot'], driver=prior['driver'],
        runtime_manifest=dict(path=str(ROOT/'runtime-inputs.json'), sha256=digest(ROOT/'runtime-inputs.json'), files=len(providers)),
        native_qualification_reuse=dict(original_closed_snapshot=SNAPSHOT, result_key=speed.KEY,
            native_recompiled=False, unchanged_native_files=len(native['native_sources']), same_build_ELFs=9,
            declared_changed_python_policies=list(CHANGED), new_controller=SOURCE,
            declared_benchmark_changes=sorted(BENCHMARK_CHANGED),
            unchanged_existing_helper=dict(path=EXISTING_HELPER, commit=HELPER_COMMIT)),
        controls=controls, parser_qualification=parser, plans=[], processes=[], request_hashes={}, host_passed=False, passed=False,
        complete=False, active=False, ownership_closed=True, inference_adopted=False, whole_bytes=512 << 20,
        budgets=dict(wall=1200, cpu=3600, rss_cap=104 << 30, text_cap=16 << 20, file_cap=16 << 20, total_cap=64 << 20),
        plan=dict(question='Characterize same-engine prompt time and client TTFT across three short and six long lengths,including the actual chunked/layer-major and bank/split boundaries.',
            order=[dict(read=n, trace_progress=t) for n, t in CASES], repetitions='One fresh process per case,four exact-count bench.sh requests,thinking on,zero reuse,up to24outputs. First use separate;median of subsequent3 is the length unit. No independent-process variance estimate per length.',
            configuration='Keep qualified552b native,throughput,GDN16/MTP16,BF16 HC,INT8resident131072,8192chunk,1256decode slots,5workers,spec4,adapt0 and immutable ADAPT_SYNC1/runtime safety. No placement or kernel change.',
            observers='Request progress STRATA_TRACE=1 for every curve length,with quiet parameter validation and unchanged native loader probe. One adjacent clean117K anchor. No full API trace or resource profiler. Short-input observer overhead remains unqualified.',
            acceptance='Require exact actual counts,zero reuse,body and journal,finite positive prompt time/client TTFT,actual phase/route/typed callback coverage,all source/native/provider/owner/fault/final-validation gates. Each length is diagnostic;no1000tok/s gate and no speed-based exclusion. The117K observed/clean time ratio<=1.15 is a local cost screen only.',
            analysis='Fit affine/quadratic/nlogn on five homogeneous non-split layer-major lengths with entire lengths held out;retain residuals,first-use samples and adjacent slopes. Three short lengths get an affine description only. Keep117K split separate;no physical intercept or extrapolation claim.',
            stopping='Stop on any budget,source/provider,owner,clock,body,route,validation or fresh-fault failure;no retry/replacement/hash refresh. Commit each process result before the next. All15model hashes before/after cohort;quick identity checks throughout.',
            retention='Keep compact individual samples and route/coverage conservation,fit coefficients/residuals and source/native/config identities. Retire successful raw text,request bodies and JIT after closure and evidence/source recovery. Existing retained root receipts support this admission;no extra native build generation.'))
    for index, (read, traced) in enumerate(CASES):
        root = ROOT/str(index); root.mkdir(mode=0o700)
        for name in ('cache', 'tmp'): (root/name).mkdir()
        config = copy.deepcopy(row['configuration']); config['log'] = str(root/'engine.stderr')
        write(root/'config.json', config)
        write(root/'plan.json', dict(binary=row['binary'], native_validation_probe=row['native_validation_probe']))
        with socket.socket() as sock: sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
        plan = dict(index=index, read=read, heads=1, trace_progress=traced, root=str(root),
            generated={name: digest(root/name) for name in ('config.json', 'plan.json')},
            command=['/bin/sh', str(P/'bench.sh'), '--config', str(root/'config.json'), '--port', str(port),
                '--prefill-exact-tokens', str(read), '--think', '--runs', str(RUNS), '--profile-request-gates',
                '--workload-id', 'native_gdn_20261018', '--fail-fast-engine-error', '--ready-timeout', '600', '--out', str(root/'bench')])
        if read < 32768: plan['command'].append('--prefill-short-diagnostic')
        args = speed.shared.client.parse_arguments(plan['command'][2:])
        require(args.prefill_exact_tokens == read and args.prefill_short_diagnostic == (read < 32768),
                'generated command differs from its validated CLI plan')
        plan['environment_sha256'] = comparison.fingerprint(environment(row, plan)); row['plans'].append(plan)
    save(row); print('Prepared nine-length observed curve and one clean anchor;commit before host probes.', flush=True)


def current(row):
    require(base.committed(RESULT)[KEY] == compact(row), 'commit exact current curve state')
    require(parser_qualification() == row['parser_qualification'], 'consumed parser qualification changed')
    native, providers, manifest, _ = qualification(row['source_commit'])
    require(pinned_sources(row['source_commit']) == row['sources'] and row['binary'] == native['binaries']['strata'],
            'frozen curve inputs changed')
    inv = row['runtime_manifest']
    require(digest(inv['path']) == inv['sha256'] and json.loads(Path(inv['path']).read_text()) == providers
            and len(providers) == inv['files'] and digest(row['template']['path']) == row['template']['sha256']
            and manifest['selected'] == row['python_selected'], 'curve provider/template changed')
    require(base.supervision.artifact_bytes(ROOT) <= row['whole_bytes'], 'whole curve output budget')
    speed.shared.models.model_pins(row['configuration'], row['model_assets'], full=False)
    return providers, manifest


if __name__ == '__main__':
    args = sys.argv[1:]
    require(args in (['selfcheck'], ['replay'], ['prepare'], ['host']) or len(args) == 1 and
            args[0].isdigit() and int(args[0]) < len(CASES), 'use selfcheck|replay|prepare|host|INDEX0..9')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); base.guards.no_model_owners([])
        if args == ['selfcheck']: print(json.dumps(selfcheck()), flush=True)
        elif args == ['replay']: replay()
        elif args == ['prepare']: prepare()
        elif args == ['host']: comparison.host(sys.modules[__name__])
        else: comparison.run(int(args[0]), sys.modules[__name__])
