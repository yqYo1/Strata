#!/usr/bin/env python3
"""Bounded resource attribution on the closed official joint-head generation.

Main-authored; repository LICENSE. No new native build. A clean/observed pair
checks observer cost separately from speed, quality and hardware saturation.
"""
import copy
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import select
import socket
import statistics
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import compare_official_prefill as comparison
import bench_engine as reader
import command_service

entry, shared, build = comparison.entry, comparison.shared, comparison.build
P, B, RESULT = comparison.P, comparison.B, comparison.RESULT
ROOT, KEY = B / 'official-joint-resources-v1', 'official_joint_resources_v1'
READ, ROWS, RUNS = 117489, 117488, 4
GROUPS = ((1, 15), (16, 31), (32, 63), (64, 127), (128, 255),
          (256, 511), (512, 1023), (1024, 2333), (2334, 8192))
CHANGED = {'tools/sycl/prefill_resources.py', 'tools/sycl/bench_engine.py'}
OBSERVERS = dict(STRATA_TRACE='1', STRATA_PREFILL_CPU_WAIT_PROFILE='1',
    STRATA_PREFILL_LAYER_MAJOR_PROFILE='1', STRATA_PREFILL_LAYER_MAJOR_QSA_PROFILE='1',
    STRATA_PREFILL_LAYER_MAJOR_GDN_PROFILE='1', STRATA_PREFILL_TRANSITION_PROFILE='1',
    STRATA_SYCL_PREFILL_COMMAND_PROFILE='1')
require, digest, write = comparison.require, comparison.digest, comparison.write


def compact(row):
    return comparison.compact(row)


def save(row):
    write(ROOT / 'record.json', row)
    data = json.loads(RESULT.read_text())
    data[KEY] = compact(row)
    write(RESULT, data)


def qualification():
    built, _, providers = entry.proof()
    cohort = build.committed(RESULT)[comparison.KEY]
    external = json.loads((comparison.ROOT / 'record.json').read_text())
    require(compact(external) == cohort and cohort['passed'] and cohort['complete']
            and cohort['ownership_closed'] and not cohort['active']
            and len(cohort['processes']) == 12
            and comparison.decision(cohort['processes']) == cohort['decision'],
            'close and commit the full frozen comparison first')
    gate = cohort['decision']['lengths'][str(READ)]
    require(gate['mechanism_screen_passed'] and gate['candidate_capacity_screen_passed'],
            '117K mechanism and capacity screens must permit further attribution')
    require(cohort['binary'] == built['binaries']['strata']
            and cohort['native_source_commit'] == built['source_commit'],
            'same seven-executable native generation required')
    inventory = cohort['runtime_manifest']
    require(digest(inventory['path']) == inventory['sha256'], 'closed provider manifest changed')
    observed_providers = json.loads(Path(inventory['path']).read_text())
    require(observed_providers == providers and len(observed_providers) == inventory['files']
            and all(digest(path) == sha for path, sha in observed_providers.items()),
            'closed comparison providers changed')
    return external, providers


def environment(row, plan):
    env = comparison.environment(row, plan)
    if plan['observed']:
        env.update(OBSERVERS)
    else:
        env['STRATA_PREFILL_CPU_WAIT_PROFILE'] = '0'
        env['STRATA_SYCL_PREFILL_COMMAND_PROFILE'] = '0'
        env['STRATA_PREFILL_LAYER_MAJOR_QSA_PROFILE'] = '0'
        env['STRATA_PREFILL_LAYER_MAJOR_GDN_PROFILE'] = '0'
    require(not env.get('LD_PRELOAD') and env['STRATA_PREFILL_RELEASE_HEADS'] == '1'
            and env['STRATA_SYCL_PREFILL_TOPK_EARLY'] == '0'
            and env['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD'] == '1'
            and env['EnableDirectSubmission'] == '0'
            and env['STRATA_SYCL_PREFILL_SAME_QUEUE'] == '1',
            'observer must preserve qualified native safety and placement')
    return env


def prepare():
    controls = selfcheck()
    cohort, providers = qualification()
    require(not ROOT.exists() and KEY not in build.committed(RESULT), 'fresh diagnostic only')
    revision = shared.gen.git('rev-parse', 'HEAD').decode().strip()
    sources = shared.pins(revision)
    require(sources.keys() == cohort['sources'].keys()
            and {n for n in sources if sources[n] != cohort['sources'][n]} == CHANGED,
            'only the resource controller and bounded diagnostic reader may change')
    contract, manifest = entry.python_manifest()
    require(contract == cohort['python_contract'] and manifest['selected'] == cohort['python_selected'],
            'frontend interpreter differs from the closed comparison')
    assets = shared.models.model_pins(cohort['configuration'], cohort['model_assets'], full=True)
    ROOT.mkdir(mode=0o700)
    write(ROOT / 'runtime-inputs.json', providers)
    row = dict(stage='prepared', source_commit=revision, sources=sources,
        native_source_commit=cohort['native_source_commit'], binary=cohort['binary'],
        native_validation_probe=cohort['native_validation_probe'], model_assets=assets,
        configuration=cohort['configuration'], environment_template=cohort['environment_template'],
        python_contract=contract, python_selected=manifest['selected'], template=cohort['template'],
        boot=cohort['boot'], driver=cohort['driver'],
        runtime_manifest=dict(path=str(ROOT / 'runtime-inputs.json'),
            sha256=digest(ROOT / 'runtime-inputs.json'), files=len(providers)),
        comparison_fingerprint=comparison.fingerprint(compact(cohort)),
        request_hashes=cohort['request_hashes'][str(READ)], plans=[], processes=[],
        host_passed=False, complete=False, passed=False, active=False, ownership_closed=True,
        budgets=cohort['budgets'], controls=controls, inference_adopted=False,
        plan=dict(question='Which host regions, GPU queue intervals and sampled kernel shapes consume the admitted117K path?',
            order='Two fresh processes:clean then observed;each4exact-count thinking-on requests with24output tokens. '
                'First use separate,subsequent-three medians;body hashes match the closed comparison.',
            constants='Same ELF,seven fixtures,official providers,131072 INT8 context,1256decode slots,5workers,'
                'spec4,8192chunks,BF16/vLLM,split/original Down,Top-k0,head retirement1.',
            acceptance='Both processes close normally;all samples eligible,all117488rows GPU-resident,128staged slots,'
                'checked split/lease restoration,quiet parameter validation,no fresh fault. Observed requests each '
                'pass all48window/work,CPU partition,GDN/QSA and4sampled-layer command gates. '
                'Observed/clean warmed-time ratio<=1.15 is an observer-cost screen,not a statistical cost claim.',
            stopping='Stop first input,owner,clock,budget,warning,fault,route,observer or sample failure. '
                'No retry,exclusion,hash refresh or speed-based replacement. Commit each closed process before next.',
            limitations='Queue intervals include host gaps;sampled kernel service is not device utilization or an '
                'additive full critical path. Logical/requested bytes are not physical bus/VRAM traffic. '
                'Hardware ceilings and saturation remain unmeasured. No clean speed gain,quality,decode,'
                'physical262144/RESTORE or adoption qualification.',
            retention='Keep individual latency samples,compact aggregate host/phase/shape evidence and limits. '
                'Discard successful raw journals,traces,JIT and full layer/histogram arrays after verification.'))
    for index, observed in enumerate((False, True)):
        root = ROOT / str(index)
        root.mkdir(mode=0o700)
        for name in ('tmp', 'cache'):
            (root / name).mkdir()
        config = copy.deepcopy(row['configuration'])
        config['log'] = str(root / 'engine.stderr')
        write(root / 'config.json', config)
        write(root / 'plan.json', dict(binary=row['binary'],
            native_validation_probe=row['native_validation_probe']))
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        plan = dict(index=index, read=READ, heads=1, observed=observed, root=str(root),
            generated={n: digest(root / n) for n in ('config.json', 'plan.json')},
            command=['/bin/sh', str(P / 'bench.sh'), '--config', str(root / 'config.json'),
                '--port', str(port), '--prefill-exact-tokens', str(READ), '--think',
                '--runs', str(RUNS), '--profile-request-gates', '--workload-id', 'native_gdn_20261018',
                '--fail-fast-engine-error', '--ready-timeout', '600', '--out', str(root / 'bench')])
        plan['environment_sha256'] = comparison.fingerprint(environment(row, plan))
        row['plans'].append(plan)
    save(row)
    print('Prepared current-source resource pair;commit before host probes.', flush=True)


def current(row):
    require(build.committed(RESULT)[KEY] == compact(row), 'commit exact diagnostic state')
    cohort, providers = qualification()
    require(comparison.fingerprint(compact(cohort)) == row['comparison_fingerprint'],
            'closed comparison evidence changed')
    require(shared.pins(row['source_commit']) == row['sources'], 'resource inputs changed')
    inventory = row['runtime_manifest']
    require(digest(inventory['path']) == inventory['sha256']
            and json.loads(Path(inventory['path']).read_text()) == providers
            and len(providers) == inventory['files'], 'resource provider inventory differs')
    require(row['binary'] == cohort['binary'] and row['driver'] == build.guards.driver_guard()
            and digest(row['template']['path']) == row['template']['sha256'], 'native/platform/template differs')
    shared.models.model_pins(row['configuration'], row['model_assets'], full=False)
    contract, manifest = entry.python_manifest()
    require(contract == row['python_contract'] and manifest['selected'] == row['python_selected'],
            'resource Python contract changed')
    return providers, manifest


def host():
    row = json.loads((ROOT / 'record.json').read_text())
    _, manifest = current(row)
    require(row['stage'] == 'prepared' and not (ROOT / 'host').exists(), 'fresh host probes only')
    owner = build.supervision.Owner(ROOT / 'host')
    reports = []
    try:
        for plan in row['plans']:
            for role, target in (('benchmark', 'tools/strata_bench.py'), ('server', 'serve/server.py')):
                command = [manifest['selected']['path'], str(shared.python_runtime.SOURCE),
                    'probe', str(P / target), row['template']['path']]
                item, out, err = owner.run(f'{plan["index"]}-{role}', command, environment(row, plan), P,
                    wall=45, cpu=30, rss_cap=512 << 20, text_cap=512 << 10, file_cap=2 << 20, total_cap=8 << 20)
                require(build.supervision.closed(item) and item['exit_code'] == 0
                        and err.stat().st_size == 0, 'resource frontend probe failed')
                result = json.loads(out.read_text())
                allowed = {v['realpath'] for v in manifest['inventory']['files'].values()}
                require(str(Path(result['executable']).resolve()) == manifest['selected']['file']['realpath'],
                        'host interpreter differs')
                for name in [*result['modules'], *result['code_maps']]:
                    path = Path(name).resolve()
                    require(str(path.relative_to(P)) in row['sources'] if path.is_relative_to(P)
                            else str(path) in allowed, 'unfrozen frontend module/provider')
                reports.append(dict(arm=plan['index'], role=role, passed=True))
        row.update(host_passed=True, stage='ready', host_results=reports)
    except BaseException as error:
        row.update(stage='failed', error=repr(error))
    finally:
        row.update(active=bool(owner.active), ownership_closed=owner.active is None
            and all(build.supervision.closed(v) for v in owner.commands))
        save(row)
    require(row['host_passed'] and row['ownership_closed'], row.get('error', 'host probes incomplete'))


def sampled_service(profile):
    groups = {}
    for sample in profile['samples']:
        key = (sample['kind'], int(sample['bin']))
        item = groups.setdefault(key, dict(kind=key[0], bin=key[1], service_ns=[],
            eligible=0, selected=0, logical_gemm_flops=0))
        item['service_ns'].append(sample['service_ns'])
        if key[0] in {'gu', 'gu_swiglu', 'down', 'down_scatter'}:
            m, n, k = (int(sample[name]) for name in ('m', 'n', 'k'))
            require(1 <= m <= 8192 and (n, k) == (
                (1280, 2560) if key[0] in {'gu', 'gu_swiglu'} else (2560, 640)),
                'sampled MoE GEMM geometry differs')
            item['logical_gemm_flops'] += 2 * m * n * k
    for group in profile['groups']:
        item = groups[(group['kind'], int(group['bin']))]
        item['eligible'] += int(group['eligible'])
        item['selected'] += int(group['selected'])
    out = []
    for key in sorted(groups):
        item = groups[key]
        values = item.pop('service_ns')
        require(item['selected'] == len(values) and item['eligible'] >= len(values), 'sample reduction lost work')
        out.append(dict(item, sample_service_ns=sum(values), median_service_ns=statistics.median(values),
            min_service_ns=min(values), max_service_ns=max(values)))
    return dict(sampled_layers=[0, 15, 31, 47], groups=out, complete=True,
        scope='Selected stratum samples only;logical GEMM FLOPs exclude non-GEMM epilogues and are not DPAS '
            'instruction counts. No full-model service,occupancy,saturation or ceiling claim.')


def transition(text):
    frames = list(re.finditer(r'^strata native dense host profile: (.+)$', text, re.M))
    refills = list(re.finditer(r'^strata trace: refilled (\d+) slots on (\d+) stage\(s\) in ([\d.eE+\-]+) ms$', text, re.M))
    require(len(frames) == len(refills) == 1 and frames[0].end() < refills[0].start(),
            'missing/duplicate/reordered transition and final refill')
    tokens = frames[0][1].split()
    require(all(t.count('=') == 1 for t in tokens), 'malformed transition fields')
    fields = dict(t.split('=', 1) for t in tokens)
    require(len(fields) == len(tokens) and all(fields.get(k) == v for k, v in dict(
        v='2', complete='1', outcome='success', entry_done='1', run_returned='1', restored='1',
        intervals='NONEXCLUSIVE', outer_startup_snapshot_unmeasured='1').items()), 'incomplete transition')
    times = {k: float(fields[k]) for k in ('entry_ms', 'run_ms', 'restore_ms')}
    refill = float(refills[0][3])
    require(all(math.isfinite(v) and v >= 0 for v in [*times.values(), refill])
            and int(refills[0][1]) > 0 and int(refills[0][2]) == 1, 'invalid transition clock/extent')
    return dict(native_dense=times, cache_refill_wall_ms=refill, refilled_slots=int(refills[0][1]),
        intervals_nonexclusive=True, refill_outside_native_dense=True,
        limits='Wrapper clocks include lookup,submission and waits;do not add nested queue intervals or infer PCIe service.')


def observed(text, route):
    record = reader.extract_resource_profile(text, True, True,
        'Nonexclusive queue spans;not additive wall time or hardware saturation.',
        cpu_version=4, expected_window_pairs=[(0, ROWS)], maximum_window_rows=ROWS)
    result = reader.compact_host_resource_profile(record, GROUPS)
    require(result['window_census']['jobs'] == route['jobs']
            and result['window_census']['routed_rows'] == ROWS * 10 * 48
            and result['window_census']['requested_router_D2H_bytes'] == ROWS * 10 * 4 * 48
            and result['window_census']['requested_map_H2D_bytes'] == ROWS * 10 * 8 * 48,
            'whole-window route/CPU census differs')
    result['sampled_gpu_service'] = sampled_service(record['command_service_profile'])
    result['transition'] = transition(text)
    return result


def run(index):
    row = json.loads((ROOT / 'record.json').read_text())
    providers, manifest = current(row)
    require(row['host_passed'] and row['stage'] in ('ready', 'running')
            and len(row['processes']) == index < 2
            and all(p['passed'] for p in row['processes']), 'fresh ordered resource process required')
    plan = row['plans'][index]
    root, env = Path(plan['root']), environment(row, plan)
    require(plan['index'] == index and plan['observed'] is bool(index)
            and not (root / 'execution.json').exists()
            and comparison.fingerprint(env) == plan['environment_sha256']
            and all(digest(root / n) == sha for n, sha in plan['generated'].items()), 'resource plan differs')
    cursor = next(v.split(': ', 1)[1] for v in subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'],
        text=True, timeout=15).splitlines() if v.startswith('-- cursor: '))
    outcome, owner = dict(index=index, observed=plan['observed'], complete=False, model_executed=False), None
    initial = (time.clock_gettime_ns(time.CLOCK_BOOTTIME), time.monotonic_ns())
    log_windows = []

    def persist():
        write(root / 'execution.json', dict(outcome, active=bool(owner and owner.active),
            commands=owner.commands if owner else []))

    def interact(io, command):
        identity = None
        for request in range(RUNS):
            offsets = {}
            for stage in ('begin', 'end'):
                while True:
                    line = io.line(900)
                    if line.startswith('STRATA_PROFILE_GATE '):
                        break
                require(json.loads(line.removeprefix('STRATA_PROFILE_GATE ')) ==
                        dict(stage=stage, phase='prefill-only', run=request), 'resource request gate differs')
                delta = [a - b for a, b in zip(
                    (time.clock_gettime_ns(time.CLOCK_BOOTTIME), time.monotonic_ns()), initial)]
                require(min(delta) >= 0 and abs(delta[0] - delta[1]) <= 50000000,
                        'resource clock/suspend discontinuity')
                claim = json.loads((root / 'engine-start.json').read_text())['native']
                actual = build.supervision.identity(claim['pid'])
                require(actual and actual['start_ticks'] == claim['start_ticks']
                        and actual['sid'] == command['pid']
                        and str(Path('/proc', str(claim['pid']), 'exe').resolve()) == row['binary']['path']
                        and (identity is None or identity == claim), 'resource native owner differs')
                identity = claim
                expected = comparison.engine_once.native_environment(row, env)
                actual_env = dict(v.split(b'=', 1) for v in
                    Path('/proc', str(claim['pid']), 'environ').read_bytes().split(b'\0') if b'=' in v)
                require(all(actual_env.get(k.encode()) == v.encode() for k, v in expected.items())
                        and set(actual_env) - {k.encode() for k in expected} <= {b'PWD', b'OLDPWD', b'SHLVL', b'_'},
                        'resource native environment differs')
                mapped = shared.runtime.executable_mappings(
                    Path('/proc', str(claim['pid']), 'maps').read_text())
                require(mapped <= set(providers) | {row['binary']['path']}
                        and {row['native_validation_probe']['loader']['path'],
                             row['native_validation_probe']['library']['path'],
                             env['ZE_ENABLE_ALT_DRIVERS']} <= mapped, 'resource loaded providers differ')
                shared.python_runtime.check_running(command['pid'], P / 'tools/strata_bench.py',
                    manifest['selected'], env,
                    {v['realpath'] for v in manifest['inventory']['files'].values()}, plan['command'][2:])
                require(not (root / 'validation.log').exists()
                        or (root / 'validation.log').stat().st_size == 0, 'resource validation warning/error')
                outcome.update(model_executed=True, native_identity=claim)
                offsets[stage] = (root / 'engine.stderr').stat().st_size
                print(f'Resource process{index} observed{int(plan["observed"])} request{request} {stage}', flush=True)
                io.send(b'continue\n')
            require(offsets['end'] > offsets['begin'], 'missing resource log window')
            log_windows.append(offsets)
        io.proc.stdin.close()
        deadline = time.monotonic() + 180
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll()
            require(time.monotonic() < deadline, 'resource teardown deadline')
            if select.select([io.proc.stdout], [], [], .02)[0]:
                io.read()
                io.pending.clear()
        io.exit(1)

    try:
        owner = build.supervision.Owner(root / 'owned', persist)
        command, _, _ = owner.run('bench', plan['command'], env, P,
            interaction=interact, **row['budgets'])
        require(build.supervision.closed(command) and command['exit_code'] == 0, 'resource process failed')
        outputs = list((root / 'bench').glob('*/results.json'))
        require(len(outputs) == 1, 'missing/duplicate resource benchmark output')
        result = json.loads(outputs[0].read_text())
        cohort = result['results']['prefill_only']
        samples, counts = cohort['requests'], cohort['count_preflight']
        require(result['complete'] and not result['results']['failures']
                and len(samples) == len(counts) == len(log_windows) == RUNS
                and [s['request_sha256'] for s in samples] == row['request_hashes'],
                'resource request extent/bodies differ')
        for sample, count in zip(samples, counts):
            require(all(sample[k] is True for k in
                    ('ok', 'eligible', 'stream_complete', 'thinking_requested', 'reasoning_observed'))
                    and sample['actual_read_tokens'] == sample['actual_prompt_tokens'] ==
                        count['counted_input_tokens'] == READ
                    and sample['actual_reused_tokens'] == 0
                    and sample['request_sha256'] == count['body_sha256']
                    and 1 <= sample['generated_tokens'] <= 24
                    and sample['prompt_tps_source'] == 'engine'
                    and math.isfinite(sample['prompt_ms']) and sample['prompt_ms'] > 10000,
                    'resource sample/clock differs')
        require(shared.client.verify_request_journal(outputs[0].parent)['valid'], 'resource request journal invalid')
        current(row)
        config = json.loads((root / 'config.json').read_text())
        end = shared.gen.native_end(root, row, dict(config=config), row['boot'], command['pid'])
        raw = (root / 'engine.stderr').read_bytes()
        text = raw.decode()
        validation = shared.runtime.validation.preload_evidence(text, end['native']['pid'])
        require(validation['loader'] == row['native_validation_probe']['loader']['path']
                and not any(s in text for s in ('UR_RESULT_ERROR_', 'ZE_RESULT_ERROR_',
                    'terminate called', 'teardown uncertain')), 'resource native failure')
        routes, diagnostics = [], []
        for window in log_windows:
            part = raw[window['begin']:window['end']].decode()
            route = comparison.route(part, READ, 1)
            one_map = route.pop('requested_dequant_H2D_bytes')
            require(one_map == ROWS * 10 * 4 * 48
                    and route['residual_gpu_rows'] == ROWS and route['residual_host_bytes'] == 0
                    and route['split_guard_checked'] and route['prefill_slots'] == 128,
                    'resource all-GPU/split/map-byte route differs')
            route['requested_one_routing_map_H2D_bytes'] = one_map
            routes.append(route)
            if plan['observed']:
                diagnostics.append(observed(part, route))
            else:
                require(not any(s in part for s in
                    ('strata lm profile:', 'strata command profile:', 'strata prefill cpu wait:',
                     'strata trace:', 'strata MoE progress:')), 'clean resource arm has observers')
        fields = ('actual_read_tokens', 'actual_reused_tokens', 'prompt_ms', 'prompt_tps', 'ttft_s', 'total_s',
            'generated_tokens', 'request_sha256', 'drafts_offered', 'drafts_accepted', 'timing_frontend_state')
        outcome.update(complete=True, native_exit_code=0, validation=validation, routes=routes,
            samples=[{k: s[k] for k in fields} for s in samples],
            subsequent_median_ms=statistics.median(s['prompt_ms'] for s in samples[1:]))
        if plan['observed']:
            require(len(diagnostics) == RUNS, 'incomplete observed attribution')
            outcome['attribution'] = diagnostics
            shared.models.model_pins(config, row['model_assets'], full=True)
    except BaseException as error:
        outcome.update(error=repr(error), complete=False)
    finally:
        if owner:
            shared.runtime.validation.finish(outcome, owner, cursor, row['boot'])
        outcome['passed'] = bool(outcome.get('passed') and outcome['complete'])
        persist()
        row['processes'].append(outcome)
        row.update(stage='running' if outcome['passed'] else 'failed',
            active=bool(owner and owner.active), ownership_closed=outcome.get('ownership_closed', False))
        if outcome['passed'] and len(row['processes']) == 2:
            clean, measured = row['processes']
            ratio = measured['subsequent_median_ms'] / clean['subsequent_median_ms']
            if clean['native_identity'] == measured['native_identity'] or not math.isfinite(ratio) or ratio <= 0:
                outcome.update(passed=False, error='Invalid independent process or observer ratio')
                row.update(stage='failed', complete=False, passed=False)
            else:
                row.update(stage='closed', complete=True, passed=True, decision=dict(
                    observed_over_clean_time=ratio, observer_cost_screen_passed=ratio <= 1.15,
                    attribution_eligible=ratio <= 1.15, hardware_saturation_qualified=False,
                    throughput_gain_claimed=False, inference_adopted=False))
        save(row)
    print(json.dumps({k: outcome.get(k) for k in
        ('index', 'observed', 'passed', 'complete', 'fresh_faults', 'error')}), flush=True)
    require(outcome['passed'], 'resource process did not qualify')


def selfcheck():
    # A new bounded117K reader must not change the historical default81920 gate.
    def census(rows):
        quotient, tail = divmod(rows * 10, 8192)
        job_count = quotient + bool(tail)
        histogram = ([[tail, 1]] if tail else []) + [[8192, quotient]]
        expert_histogram = [[0, 512 - job_count], [1, job_count]]
        windows = [dict(version=1, layer=layer, first_row=0, rows=rows, jobs=job_count,
            job_rows_histogram=histogram, jobs_per_expert_histogram=expert_histogram)
            for layer in range(48)]
        text = ''.join('strata MoE planned job census: ' + json.dumps(w) + '\n' for w in windows)
        return text + ('strata MoE accumulator window: windows=48 dequants=48 jobs=' +
            str(job_count * 48) + ' source_H2D_bytes=' + str(rows * 10 * 4 * 48) +
            ' GPU_resident_windows_only=1 operator_max=8192 maximum_rows=' + str(rows) + '\n')

    require(len(reader.extract_planned_moe_work(census(65536))['planned_moe_windows']) == 48,
            'default ordinary-window reader changed')
    require(len(reader.extract_planned_moe_work(census(ROWS), [(0, ROWS)], ROWS)['planned_moe_windows']) == 48,
            'prospective split-window reader refused')
    rejected = []
    for name, operation in (
        ('legacy117K', lambda: reader.extract_planned_moe_work(census(ROWS))),
        ('over117K', lambda: reader.extract_planned_moe_work(census(ROWS + 1), maximum_window_rows=ROWS)),
        ('unplanned_bound', lambda: reader.extract_planned_moe_work(census(65536), maximum_window_rows=131072)),
        ('bool_bound', lambda: reader.extract_planned_moe_work(census(65536), maximum_window_rows=True)),
        ('map_byte_mismatch', lambda: reader.extract_planned_moe_work(
            census(ROWS).replace(str(ROWS * 10 * 4 * 48), '1'), maximum_window_rows=ROWS)),
        ('missing_layer', lambda: reader.extract_planned_moe_work(
            '\n'.join(census(ROWS).splitlines()[1:]), maximum_window_rows=ROWS)),
    ):
        try:
            operation()
        except (RuntimeError, KeyError, ValueError, TypeError):
            rejected.append(name)
        else:
            raise RuntimeError('invalid resource control accepted: ' + name)
    # Group boundaries preserve the128automatic tile switch and2334fusion switch.
    windows = [dict(layer=0, rows=1, jobs=4, gu_nk=[1280, 2560], down_nk=[2560, 640],
        gu_down_storage='bf16', hidden_storage='fp16',
        job_rows_histogram=[[127, 1], [128, 1], [2333, 1], [2334, 1]],
        requested_router_D2H_bytes=40, requested_map_H2D_bytes=80)]
    windows[0]['rows'] = (127 + 128 + 2333 + 2334) // 10
    # Reduction independently conserves jobs and rows; raw histogram remains input only.
    mini = dict(planned_moe_windows=windows, resource_diagnostics=[],
        host_counter_conservation={})
    # Use an exactly divisible routed-row sum for this reduction fixture.
    windows[0]['job_rows_histogram'].append([78, 1])
    windows[0]['rows'] = 500
    windows[0]['jobs'] = 5
    folded = reader.compact_host_resource_profile(mini, GROUPS)['window_census']
    require(sum(v['jobs'] for v in folded['job_size_groups']) == 5
            and sum(v['routed_rows'] for v in folded['job_size_groups']) == 5000
            and folded['job_size_groups'][3]['jobs'] == 2
            and folded['job_size_groups'][4]['jobs'] == 1
            and folded['job_size_groups'][7]['jobs'] == 1
            and folded['job_size_groups'][8]['jobs'] == 1, 'job boundary reduction lost work')
    for name, bad in (('overlapping_groups', ((1, 127), (127, 8192))),
                      ('gapped_groups', ((1, 127), (129, 8192))),
                      ('truncated_groups', ((1, 8191),))):
        try:
            reader.compact_host_resource_profile(mini, bad)
        except (RuntimeError, KeyError, ValueError, TypeError):
            rejected.append(name)
        else:
            raise RuntimeError('invalid resource control accepted: ' + name)
    samples = [dict(kind='gu', bin=3, m=127, n=1280, k=2560, service_ns=t)
               for t in (1000, 2000)]
    groups = [dict(kind='gu', bin=3, eligible=n, selected=1) for n in (10, 20)]
    service = sampled_service(dict(samples=samples, groups=groups))['groups'][0]
    require(service['selected'] == 2 and service['eligible'] == 30
            and service['sample_service_ns'] == 3000 and service['median_service_ns'] == 1500
            and service['logical_gemm_flops'] == 4 * 127 * 1280 * 2560,
            'sampled-service reduction lost extent or time')
    return dict(ordinary_and_split_controls=2, rejected=rejected, shape_boundaries=[127, 128, 2333, 2334],
        defaults_preserved=True, GPU_executed=False)


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ('selfcheck', 'prepare', 'host', '0', '1'),
            'use selfcheck|prepare|host|0|1')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        build.guards.no_model_owners([])
        if sys.argv[1] == 'selfcheck':
            print(json.dumps(selfcheck()))
        else:
            {'prepare': prepare, 'host': host}.get(sys.argv[1], lambda: run(int(sys.argv[1])))()
