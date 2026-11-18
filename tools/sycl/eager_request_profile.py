#!/usr/bin/env python3
"""Matched device-only PTI profile of the established 64K eager prefill route.

Main-authored under the repository LICENSE. The launcher and clock hooks come
from the owned a1f90cdff controller. The 64K workload and settings come from
the owned a9a90b203 length curve. No native build or other engine input.
"""
import copy
import fcntl
import json
from pathlib import Path
import socket
import statistics
import sys

sys.dont_write_bytecode = True
import eager_pti_probe as probe
import prefill_workspace_admission as workspace
import request_profile_reader as reader

capacity, core = workspace.capacity, workspace.core
comparison = core.comparison
P, B, RESULT = capacity.P, capacity.B, capacity.RESULT
ROOT, KEY = B / 'eager-request-profile-v2', 'eager_request_profile_v2'
THIS = 'tools/sycl/eager_request_profile.py'
READER = 'tools/sycl/request_resource_summary.py'
CURVE_COMMIT = '90a1bc130f47fe87938b3d4db2b769f5c9b64169'
CURVE_KEY = 'eager_prefill_length_curve_v2'
EARLIER_COMMIT = '9b17274579273235f6c142eea8d48d0ae21273ea'
EARLIER_KEY = 'eager_request_profile_v1'
PYTHON = capacity.reference.model.PYTHON
ARMS = ('trace', 'clean')
READ, RUNS = 65537, 3
ORDER = [(READ, 1)] * len(ARMS)
NATIVE_BUDGET = dict(wall=1500, cpu=9000, rss_cap=104 << 30,
    text_cap=16 << 20, file_cap=reader.MAX_BYTES, total_cap=1280 << 20)
ANALYSIS_BUDGET = dict(wall=240, cpu=210, rss_cap=12 << 30,
    text_cap=128 << 10, file_cap=2 << 20, total_cap=4 << 20)
require, sha, signature = capacity.require, capacity.sha, capacity.signature


def save(row):
    core.write(ROOT / 'record.json', row)
    data = json.loads(RESULT.read_text())
    data[KEY] = core.compact(row)
    core.write(RESULT, data)


def normalize_route(route):
    result = copy.deepcopy(route)
    del result['layout']['free_gpu_bytes']
    return result


def earlier_pair():
    path = str(RESULT.relative_to(P))
    prior = json.loads(probe.pti.blob(P, EARLIER_COMMIT, path))[EARLIER_KEY]
    require(prior['stage'] == 'failed' and not prior['passed'] and not prior['complete']
        and prior['ownership_closed'] and not prior['active']
        and len(prior['processes']) == 2 and prior['post_run_analysis_v2']['stage'] == 'closed',
        'original failed cohort and separately recovered trace required')
    clean, traced = prior['processes']
    require(clean['passed'] and clean['complete'] and clean['clean_speed']
        and not traced['passed'] and not traced['clean_speed']
        and all(p['native_exit_code'] == 0 and p['ownership_closed'] and p['boot_unchanged']
            and p['fresh_faults'] == [] and len(p['samples']) == RUNS for p in (clean, traced)),
        'original exact native timing samples required')
    return dict(commit=EARLIER_COMMIT, key=EARLIER_KEY, timings=timing_ratios(clean, traced),
        original_cohort_passed=False,
        use='Descriptive cross-check from a separately recovered trace. Original cohort remains failed.')


def basis():
    _, original_proof, prior = capacity.basis()
    fixture = json.loads((probe.ROOT / 'record.json').read_text())
    probe.current(fixture)
    require(fixture['stage'] == 'closed' and fixture['passed'] and fixture['complete']
        and fixture['ownership_closed'] and not fixture['active']
        and fixture['raw_capture_retirement']['complete']
        and [r['label'] for r in fixture['results']] == list(probe.LABELS)
        and all(r['passed'] and r['complete'] and r['ownership_closed'] and r['native_exit_code'] == 0
            and r['boot_unchanged'] and r['fresh_faults'] == [] for r in fixture['results'])
        and fixture['results'][1]['raw_clock_mapping_checked'], 'same-build closed collector probe required')
    native = json.loads((workspace.device.build.ROOT / 'record.json').read_text())
    curve = workspace.device.build.ur_validation.committed(RESULT, (CURVE_KEY,))
    source = workspace.device.build.source
    original = json.loads(source.git(P, 'show', CURVE_COMMIT + ':' + str(RESULT.relative_to(P))))[CURVE_KEY]
    require(curve['stage'] == 'closed' and curve['passed'] and curve['complete']
        and curve['ownership_closed'] and not curve['active'] and curve['raw_capture_retirement']['complete']
        and curve['original_evidence_archive']['commit'] == CURVE_COMMIT
        and curve['original_evidence_archive']['original_row_sha256'] == signature(original)
        and native['binaries']['strata'] == curve['binary'] == original['binary']
        and native['source_commit'] == fixture['native_source_commit'] == curve['native_source_commit']
        and fixture['component_gate'] == curve['component_gate']
        and fixture['native_validation_probe'] == curve['native_validation_probe']
        and fixture['boot'] == curve['boot'] and fixture['driver'] == curve['driver'],
        'same native generation and closed exact 64K evidence required')
    require(all(curve[k] == prior[k] for k in ('configuration','template','boot','driver',
        'runtime_manifest','python_contract','python_selected','native_validation_probe'))
        and all(curve[k] == core.compact(prior)[k] for k in ('environment_template_sha256','model_assets_sha256')),
        'model configuration, environment, providers or assets changed')

    previous = original['processes'][2]
    require(previous['index'] == 2 and previous['read'] == READ and previous['clean_speed']
        and previous['passed'] and previous['complete'] and previous['ownership_closed']
        and previous['native_exit_code'] == 0 and previous['fresh_faults'] == [],
        'original clean 64K curve process required')
    reference = dict(closed_curve_sha256=signature(curve), source_commit=curve['source_commit'],
        original_archive=CURVE_COMMIT, original_reference_sha256=signature(original_proof),
        probe_receipt_sha256=signature(probe.compact(fixture)),
        request_hashes=[s['request_sha256'] for s in previous['samples']],
        route_hashes=[signature(normalize_route(r)) for r in previous['routes']])
    require(reference['request_hashes'] == original['request_hashes'][str(READ)],
        'original 64K request bodies do not match the closed design')
    previous_plan = original['plans'][2]
    require(comparison.fingerprint(environment(prior, previous_plan)) == previous_plan['environment_sha256'],
        'effective environment differs from the original clean request')
    return prior, curve, fixture, reference


def source_pins(revision):
    pins = probe.source_pins(revision)
    require({THIS, READER, 'tools/test_request_resource_summary.py',
        'tools/test_request_profile_reader.py', 'tools/test_eager_request_profile.py'} <= pins.keys(),
        'commit the actual campaign and tests first')
    for name in capacity.POLICIES:
        raw = probe.pti.blob(P, revision, name)
        require(raw == (P / name).read_bytes(), 'policy differs from the committed campaign')
        pins[name] = probe.pti.pti.sha(raw)
    return pins


def environment(row,plan):
    env=comparison.environment(row,plan)
    env.update(STRATA_SYCL_PREFILL_EXPERT_BANK_SLOTS='384',STRATA_SYCL_PREFILL_MOE_WINDOW='10',
        STRATA_SYCL_PREFILL_MOE_RELOCATE_JOB_BUFFERS='0',STRATA_SYCL_PREFILL_FULLFIT_UPLOAD='eager',
        STRATA_SYCL_PREFILL_MOE_CENSUS='0',STRATA_PREFILL_LAYER_MAJOR_PROFILE='0')
    require(env['STRATA_SYCL_PREFILL_SAME_QUEUE']=='1'
        and env['STRATA_SYCL_PREFILL_EXPERT_UPLOAD_STAGING']=='0'
        and env['STRATA_SYCL_PREFILL_EXPERT_BANK_PACKED']=='1'
        and env['STRATA_SYCL_CACHE_SCRATCH_LEASE']==env['STRATA_SYCL_PREFILL_MIXED_SCRATCH']=='1'
        and env['STRATA_SYCL_PREFILL_DOWN_SPLIT_KERNEL']=='original'
        and env['STRATA_SYCL_PREFILL_MOE_SPLIT_ACCUMULATOR']=='1','fixed qualified queue/layout required')
    return env


def configuration(row,plan):
    value=copy.deepcopy(row['configuration']);value['log']=str(Path(plan['root'])/'engine.stderr')
    require(value['exe']==str(P/'tools/sycl/engine_once.py'),'preserve production wrapper')
    return value


def profile(row,plan):
    tools=row['admission']['tools']['artifacts'];root=Path(plan['root'])
    return dict(mode='whole_request',launcher=tools['unitrace'],collector=tools['libunitrace_tool.so'],session='',
        environment=dict(UNITRACE_SystemTime='1',UNITRACE_FollowChildProcess='1',
            UNITRACE_TraceOutputDir=str(root/'trace')))


def native_context(row,plan):
    require(type(plan['index']) is int and 0<=plan['index']<len(ARMS) and plan['arm']==ARMS[plan['index']]
            and plan['read']==READ and plan['heads']==1,'declared profile arm required')
    out={k:row[k] for k in ('binary','native_source_commit','native_validation_probe','boot')}
    if plan['arm']=='trace':out['native_profile']=profile(row,plan)
    return out


def launch_plan(row,plan):
    native=native_context(row,plan)
    return {k:v for k,v in native.items() if k in ('binary','native_validation_probe','native_profile')}


def check_native_environment(observed,expected,plan):
    actual={k.decode():v.decode() for k,v in observed.items()}
    wanted=dict(expected)
    if plan['arm']=='trace':
        prof=json.loads((Path(plan['root'])/'plan.json').read_text())['native_profile']
        shim=wanted['LD_PRELOAD']
        wanted.update(prof['environment'],STRATA_UNITRACE_TARGET_PRELOAD=shim,
            LD_PRELOAD=prof['collector']['path']+':'+shim,
            UNITRACE_DeviceTiming='1',UNITRACE_ChromeDeviceLogging='1',
            UNITRACE_ChromeEventBufferSize='128',UNITRACE_LD_PRELOAD_OLD=shim,
            ZE_ENABLE_TRACING_LAYER='1',INTEL_LIBITTNOTIFY64='libunitrace_tool.so',
            UNITRACE_VERSION='2.4.1 ('+reader.PTI_REVISION+')')
    changed=sorted(k for k,v in wanted.items() if actual.get(k)!=v)
    extras=sorted(set(actual)-set(wanted)-{'PWD','OLDPWD','SHLVL','_'})
    require(not changed and not extras,'actual native environment differs;changed='+str(changed)+';extra='+str(extras))


def native_extra_maps(plan):
    if plan['arm']=='clean':return set()
    return {json.loads((Path(plan['root'])/'plan.json').read_text())['native_profile']['collector']['path']}


def observe_gate(outcome,request,stage,stamp):
    require(type(stamp) is int and stamp>0,'positive RAW gate clock required')
    windows=outcome.setdefault('external_request_raw_windows',[])
    if stage=='begin':
        require(len(windows)==request and (not windows or 'end' in windows[-1]),'ordered request begin required')
        windows.append(dict(begin=stamp))
    else:
        require(stage=='end' and len(windows)==request+1 and 'end' not in windows[-1]
                and stamp>windows[-1]['begin'],'ordered request end required')
        windows[-1]['end']=stamp


def observe_lifetime(outcome,stage,stamp):
    bounds=outcome.setdefault('enclosing_owned_run_raw_ns',{})
    require(stage in ('begin','end') and stage not in bounds
            and (stage=='begin' or stamp>bounds['begin']),'ordered enclosing RAW bounds required')
    bounds[stage]=stamp


def command(plan):
    root=Path(plan['root'])
    return ['/bin/sh',str(P/'bench.sh'),'--config',str(root/'config.json'),'--port',str(plan['port']),
        '--prefill-exact-tokens',str(plan['read']),'--think','--runs',str(RUNS),'--profile-request-gates',
        '--workload-id','native_gdn_20261018','--fail-fast-engine-error','--ready-timeout','600',
        '--out',str(root/'bench')]


def request_route(text, plan):
    require(plan['read'] == READ, 'fixed 64K model request required')
    return workspace.request_route(text, plan)


def check_sample(sample, count, read, minimum_ms=10000):
    comparison.check_sample(sample, count, read, minimum_ms)
    require(read == READ and sample['long_input_extent'] is True and sample['generated_tokens'] == 24,
        'exact long request and 24-output extent required')


def check_closed_process(outcome, row, index):
    require(outcome['index'] == index and outcome['heads'] == 1 and outcome['read'] == READ
        and outcome['kernel'] == 'eager' and outcome['binary'] == row['binary']
        and outcome['native_source_commit'] == row['native_source_commit']
        and outcome['clean_speed'] == (ARMS[index] == 'clean')
        and outcome['complete'] and outcome['passed'] and outcome['ownership_closed']
        and outcome['native_exit_code'] == 0 and outcome['boot_unchanged'] and not outcome['fresh_faults'],
        'qualified closed 64K request process required')
    samples, routes = outcome['samples'], outcome['routes']
    require(len(samples) == len(routes) == RUNS
        and all(s['actual_read_tokens'] == READ and s['actual_reused_tokens'] == 0
                and s['generated_tokens'] == 24 for s in samples)
        and [s['request_sha256'] for s in samples] == row['reference']['request_hashes']
        and [signature(normalize_route(r)) for r in routes] == row['reference']['route_hashes'],
        'request bodies or actual eager routing/placement/work differ from the 64K curve')
    bounds = outcome['enclosing_owned_run_raw_ns']
    windows = outcome['external_request_raw_windows']
    previous = bounds['begin']
    require(len(windows) == RUNS, 'three observed request intervals required')
    for window in windows:
        require(previous <= window['begin'] < window['end'] <= bounds['end'],
            'request gate clocks differ')
        previous = window['end']


def after_execution(outcome, row, plan):
    check_closed_process(outcome, row, plan['index'])
    if plan['arm'] == 'clean':
        return
    root = Path(plan['root'])
    traces = list((root / 'trace').glob('*.json'))
    require(len(traces) == 1 and traces[0].name.endswith(f".{outcome['native_identity']['pid']}.json")
        and 0 < traces[0].stat().st_size <= reader.MAX_BYTES, 'one bounded actual native trace required')
    trace = traces[0]
    bounds = outcome['enclosing_owned_run_raw_ns']
    windows = outcome['external_request_raw_windows']
    receipt = dict(schema=reader.SCHEMA, clock_domain=reader.CLOCK, pti_revision=reader.PTI_REVISION,
        unitrace_system_time='1', level_zero_only=True, native_exit_code=outcome['native_exit_code'],
        ownership_closed=outcome['ownership_closed'], fresh_faults=outcome['fresh_faults'],
        boot_unchanged=outcome['boot_unchanged'], trace_sha256=sha(trace),
        source=dict(native_source_commit=row['native_source_commit'], binary_sha256=row['binary']['sha256']),
        native_lifetime=dict(begin_raw_ns=bounds['begin'], end_raw_ns=bounds['end']),
        requests=[dict(request_id=str(i), request_sha256=s['request_sha256'],
            actual_read_tokens=s['actual_read_tokens'], expected_read_tokens=READ,
            actual_reused_tokens=s['actual_reused_tokens'], generated_tokens=s['generated_tokens'],
            thinking=True, sample_scope='first_long' if i == 0 else 'warmed',
            begin_raw_ns=w['begin'], end_raw_ns=w['end'])
            for i, (s, w) in enumerate(zip(outcome['samples'], windows))])
    analysis_root = root / 'analysis'
    analysis_root.mkdir(mode=0o700)
    require(not trace.is_relative_to(analysis_root), 'input trace must be outside the analysis output cap')
    contract_path = analysis_root / 'input.json'
    result_path = analysis_root / 'resources.json'
    core.write(contract_path, receipt)
    command = [PYTHON, '-B', str(P / READER), str(trace),
        str(contract_path), str(result_path)]
    guards = comparison.build.guards
    guards.no_model_owners([])
    owner = comparison.build.supervision.Owner(analysis_root / 'owned',
        lambda: core.write(analysis_root / 'execution.json', owner.commands))
    entry, _, err = owner.run('summarize', command, dict(PATH='/usr/bin:/bin', LANG='C'), P,
        **ANALYSIS_BUDGET)
    outcome['analysis'] = dict(command=command, exit_code=entry['exit_code'],
        ownership_closed=comparison.build.supervision.closed(entry), input_sha256=sha(contract_path),
        budget_error=entry.get('error'), direct_child_reaped=entry['direct_child_reaped'],
        session_empty=entry['session_empty'])
    require(outcome['analysis']['ownership_closed'] and entry['exit_code'] == 0,
        'closed trace analysis failed: ' + str(entry.get('error')) + ' ' + err.read_text()[-1800:])
    result = json.loads(result_path.read_text())
    require(result['totals_match_request_reader'] and result['trace_sha256'] == receipt['trace_sha256']
        and len(result['requests']) == RUNS, 'trace conservation or identity differs')
    outcome['resources'] = result


def timing_ratios(clean, traced):
    points = {}
    for scope in ('first_long', 'warmed'):
        fields = {}
        for field in ('prompt_ms', 'ttft_s', 'total_s'):
            def value(process):
                samples = process['samples']
                return samples[0][field] if scope == 'first_long' else statistics.median(
                    sample[field] for sample in samples[1:])
            ratio = value(traced) / value(clean)
            fields[field] = dict(trace_over_clean=ratio, within_10_percent=.9 <= ratio <= 1.1)
        points[scope] = fields
    return points


def decision(processes):
    require(len(processes) == len(ARMS)
        and len({(p['native_identity']['pid'], p['native_identity']['start_ticks']) for p in processes}) == len(ARMS)
        and all(p['index'] == i and p['passed'] and p['complete'] and p['ownership_closed']
                and p['clean_speed'] == (ARMS[i] == 'clean')
                and ('resources' in p) == (ARMS[i] == 'trace') for i, p in enumerate(processes)),
        'two closed distinct planned model processes required')
    traced, clean = processes
    points = timing_ratios(clean, traced)
    return dict(observer_cost=points,
        fresh_pair_within_observer_budget=all(v['within_10_percent'] for fields in points.values() for v in fields.values()),
        full_original_observer_cohort_qualified=False,
        physical_saturation_qualified=False, phase_attribution_qualified=False,
        clean_speed_gain_qualified=False, inference_adopted=False,
        limits='Whole request includes 24 decode tokens. PCIe/VRAM bytes and arithmetic saturation remain unmeasured.')


def current(row):
    require(workspace.device.build.ur_validation.committed(RESULT, (KEY,)) == core.compact(row)
        and source_pins(row['source_commit']) == row['sources'], 'commit exact frozen source and prior stage')
    prior, curve, fixture, reference = basis()
    require(reference == row['reference'] and earlier_pair() == row['earlier_pair']
        and curve['binary'] == row['binary']
        and curve['native_source_commit'] == row['native_source_commit']
        and fixture['tools'] == row['admission']['tools']
        and all(row[k] == prior[k] for k in ('configuration','environment_template','model_assets',
            'template','boot','driver','native_validation_probe','runtime_manifest')),
        'frozen 64K route, collector or model inputs changed')
    require(row['budgets'] == NATIVE_BUDGET and row['analysis_budgets'] == ANALYSIS_BUDGET
        and comparison.build.supervision.artifact_bytes(ROOT) <= row['whole_bytes'],
        'frozen resource envelope differs')
    contract, python = core.frontend.python_manifest()
    require(contract == row['python_contract'] and python['selected'] == row['python_selected'],
        'frontend interpreter changed')
    for i, plan in enumerate(row['plans']):
        root = Path(plan['root'])
        require(plan['index'] == i and plan['arm'] == ARMS[i] and plan['read'] == READ
            and plan['bank'] == 384 and plan['window'] == 10 and plan['relocate'] == 0
            and plan['upload'] == plan['kernel'] == 'eager'
            and plan['command'] == command(plan)
            and json.loads((root / 'config.json').read_text()) == configuration(row, plan)
            and json.loads((root / 'plan.json').read_text()) == launch_plan(row, plan)
            and all(sha(root / name) == pin for name, pin in plan['generated'].items())
            and comparison.fingerprint(environment(row, plan)) == plan['environment_sha256'],
            'frozen generated plan differs')
    for i, outcome in enumerate(row['processes']):
        check_closed_process(outcome, row, i)
        execution = json.loads((ROOT / str(i) / 'execution.json').read_text())
        require(not execution['active'] and all(execution.get(k) == v for k, v in outcome.items())
            and all(comparison.build.supervision.closed(c) for c in execution['commands']),
            'original closed process receipt differs')
        if ARMS[i] == 'trace':
            directory = ROOT / str(i) / 'analysis'
            analyses = json.loads((directory / 'execution.json').read_text())
            require(len(analyses) == 1 and comparison.build.supervision.closed(analyses[0])
                and analyses[0]['exit_code'] == 0
                and outcome['analysis']['input_sha256'] == sha(directory / 'input.json')
                and outcome['resources'] == json.loads((directory / 'resources.json').read_text()),
                'closed trace analysis receipt differs')
    comparison.shared.models.model_pins(row['configuration'], row['model_assets'], full=False)
    return json.loads(Path(row['runtime_manifest']['path']).read_text()), python


def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()), 'fresh profiling plan required')
    prior, curve, fixture, reference = basis()
    revision = workspace.device.build.source.git(P, 'rev-parse', 'HEAD').decode().strip()
    sources = source_pins(revision)
    comparison.shared.models.model_pins(prior['configuration'], prior['model_assets'], full=True)
    row = {k: copy.deepcopy(prior[k]) for k in ('configuration','environment_template','model_assets',
        'template','boot','driver','native_validation_probe','runtime_manifest','python_contract','python_selected')}
    row.update(stage='prepared', source_commit=revision, sources=sources, reference=reference,
        earlier_pair=earlier_pair(),
        binary=curve['binary'], native_source_commit=curve['native_source_commit'],
        native_composition_sha256=curve['native_composition_sha256'],
        admission=dict(tools=fixture['tools']), plans=[], processes=[],
        request_hashes={str(READ): reference['request_hashes']}, host_passed=False,
        passed=False, complete=False, active=False, ownership_closed=True,
        budgets=NATIVE_BUDGET, analysis_budgets=ANALYSIS_BUDGET, whole_bytes=2560 << 20,
        plan=dict(question='Time copy and kernel activity on the same eager 64K route and measure the collector cost.',
            order=list(ARMS), samples='Three exact 65537-read, thinking-on, 24-output requests per fresh process.',
            isolation='Only device collector injection differs. Compare first and warmed process medians separately.',
            acceptance='The fresh reversed pair must be in 0.9..1.1 for prompt, TTFT and total duration. '
                'The earlier native timing pair is a descriptive cross-check. Its stopped cohort remains failed.',
            scope='Use enclosing RAW request windows. Retain clip/overlap, crossing counts and conserved service. '
                'Measure whole requests, including 24 decode tokens. No hardware saturation claim.',
            stopping='Stop on source, provider, ownership, fault, route, clock or finite budget failure. '
                'Preserve any failed observer screen and report its traced samples as perturbed.',
            retention='Reuse existing native generation and probe receipt. Keep actual samples and compact resource '
                'summaries. Retire full traces and transient receipts after verified closure.'))
    ROOT.mkdir(mode=0o700)
    for i, arm in enumerate(ARMS):
        root = ROOT / str(i)
        for name in ('tmp', 'cache', 'trace'):
            (root / name).mkdir(parents=True, mode=0o700)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        plan = dict(index=i, arm=arm, read=READ, heads=1, root=str(root), port=port,
            kernel='eager', upload='eager', bank=384, window=10, relocate=0, trace_progress=False)
        core.write(root / 'config.json', configuration(row, plan))
        core.write(root / 'plan.json', launch_plan(row, plan))
        plan.update(command=command(plan),
            generated={name:sha(root / name) for name in ('config.json', 'plan.json')},
            environment_sha256=comparison.fingerprint(environment(row, plan)))
        row['plans'].append(plan)
    save(row)
    print('Prepared matched 64K device-only profile. Commit generated inputs before execution.', flush=True)


if __name__ == '__main__':
    require(sys.executable == PYTHON and len(sys.argv) == 2
        and sys.argv[1] in ('prepare', 'host', '0', '1'), 'pinned Python: prepare | host | 0..1')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        comparison.build.guards.no_model_owners([])
        action = sys.argv[1]
        if action == 'prepare':
            prepare()
        elif action == 'host':
            comparison.host(sys.modules[__name__])
        else:
            comparison.run(int(action), sys.modules[__name__])
