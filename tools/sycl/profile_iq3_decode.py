#!/usr/bin/env python3
"""Current-IQ3 active collector control and short device diagnostic.

Main-authored from the fork's owned collector/model supervision; repository
LICENSE. Reuse the current native generation, not the fallback engine's
qualification. This instrumented diagnostic does not measure clean speed,
physical traffic, saturation or sustained generation.
"""
import copy
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import select
import sys
import time

sys.dont_write_bytecode = True
import iq3_model_entry as iq3
import generation_engine as gen
import profile_cache_map_decode as previous
import profile_selected_compat as compat
import decode_window_trace as window

entry, base, profile = iq3.entry, iq3.base, compat.profile
P, B, RESULT = iq3.P, iq3.B, iq3.RESULT
ROOT = B / 'decode-iq3-device-service-v1'
KEY = 'decode_iq3_device_service_v1'
CONTROL_ROOT = B / 'decode-iq3-collector-active-v1'
CONTROL_KEY = 'decode_iq3_collector_active_v1'
SOURCE = Path(__file__).resolve()
ARM = 1
require, digest, write = base.require, base.digest, base.write
client_decode_window = previous.client_decode_window


def selected_basis():
    data = base.committed(RESULT)
    cohort = data['decode_iq3_generation_speed_v1']
    require(cohort['stage'] == 'closed' and cohort['complete'] and cohort['passed']
        and cohort['ownership_closed'] and not cohort['active']
        and len(cohort['processes']) == 6
        and all(r['passed'] and r['complete'] and r['native_exit_code'] == 0
                and r['ownership_closed'] and r['fresh_faults'] == []
                for r in cohort['processes']), 'close the frozen IQ3 cohort first')
    native, _, providers = iq3.proof()
    admitted = iq3.context(ARM)
    prior = json.loads((admitted.ROOT / 'record.json').read_text())
    iq3.contract(prior, ARM)
    require(data[admitted.KEY] == admitted.compact(prior)
        and prior['stage'] == 'closed' and prior['passed'] and prior['complete']
        and prior['host_passed'] and prior['ownership_closed'] and not prior['active']
        and prior['model_executed'] and prior['outcome']['native_exit_code'] == 0
        and prior['outcome']['fresh_faults'] == [] and prior['outcome']['boot_unchanged']
        and prior['binary'] == native['binaries']['strata']
        and prior['native_source_commit'] == native['source_commit']
        and iq3.pinned_sources(prior['source_commit']) == prior['sources'],
        'closed current-generation selector/body qualification required')
    inv = prior['runtime_manifest']
    require(digest(inv['path']) == inv['sha256']
        and json.loads(Path(inv['path']).read_text()) == providers,
        'matching admitted model provider inventory required')
    fixture = native['binaries']['sycl_submission_test']
    require(digest(fixture['path']) == fixture['sha256']
        and Path(fixture['path']).stat().st_size == fixture['bytes'],
        'same-build active collector fixture changed')
    assets = profile.collector_assets(data)
    return native, prior, providers, assets


def collector_context():
    # Private module globals isolate this current-generation control from the
    # historical fallback and paused-control records. No imported module is
    # rebound, and no process environment is modified.
    spec = importlib.util.spec_from_file_location('strata_iq3_collector', compat.__file__)
    ctx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ctx)
    ctx.ROOT, ctx.KEY, ctx.SOURCE = CONTROL_ROOT, CONTROL_KEY, SOURCE
    ctx.ACTIVE_FROM_EXEC = True
    ctx.entry = entry
    ctx.basis = selected_basis

    def current(row):
        require(base.committed(RESULT)[CONTROL_KEY] == ctx.compact(row)
            and base.controls(row['source_commit'], controller=SOURCE) == row['sources'],
            'commit exact current-generation collector inputs')
        native, prior, providers, assets = selected_basis()
        require(row['native_source_commit'] == native['source_commit']
            and row['engine'] == native['binaries']['strata']
            and row['binary'] == native['binaries']['sycl_submission_test']
            and row['native_validation_probe'] == prior['native_validation_probe']
            and row['environment_template'] == prior['environment']
            and row['boot'] == prior['boot'] and row['driver'] == prior['driver']
            and row['assets'] == assets and row['budgets'] == ctx.BUDGET
            and all(row['profile'][k] == v for k, v in assets.items())
            and row['whole_bytes'] == 128 << 20 and row['profile']['mode'] == 'aggregate',
            'current collector/native/provider/control envelope differs')
        return providers

    ctx.current = current
    return ctx


def prepare_control():
    ctx = collector_context()
    # Use the already maintained fixture launch, gates, validation and flush
    # reducer, but give it a fresh current-native basis and an explicit plan.
    ctx.prepare()
    row = json.loads((CONTROL_ROOT / 'record.json').read_text())
    row['plan'].update(
        native='Exact current9922/b310 engine and same-build submission fixture;no rebuild and no fallback-control inheritance.',
        order='One fresh active-from-exec compatibility-graph fixture process;aggregate collector mode,ChromeKernelLogging1,event buffer128. Both before-runtime/runtime-ready gates must observe the active session;keep capture active through normal exit.',
        next='After normal fixture/control closure and all62 joined named kernels,prepare the separately bounded current-IQ3 one-request device diagnostic. No extra native build.',
        limits='Current-generation collector-mode qualification only;not model coverage,observer cost,clean speed,physical traffic,saturation or adoption.')
    ctx.save(row)
    print('Prepared fresh current-IQ3 active collector control;commit before check.', flush=True)


def collector_basis(native):
    ctx = collector_context()
    row = json.loads((CONTROL_ROOT / 'record.json').read_text())
    ctx.current(row)
    require(row['stage'] == 'closed' and row['passed'] and row['complete']
        and row['ownership_closed'] and not row['active']
        and row['native_source_commit'] == native['source_commit']
        and row['engine'] == native['binaries']['strata']
        and row['binary'] == native['binaries']['sycl_submission_test']
        and len(row['cases']) == 1 and row['cases'][0]['passed']
        and row['cases'][0]['trace']['collection_active_from_exec']
        and row['cases'][0]['trace']['observed_kernels'] == 62,
        'fresh closed same-build active collector control required')
    return row['assets']


def proof():
    native, gate, providers = iq3.proof()
    collector_basis(native)
    return native, gate, providers


def pinned_sources(revision):
    pins = iq3.pinned_sources(revision)
    pins.update(base.controls(revision, controller=SOURCE))
    name = 'tests/core/test_decode_window_trace.py'
    raw = base.basis.build.git(P, 'show', revision + ':' + name)
    require((P / name).read_bytes() == raw, 'commit exact guarded-window tests')
    pins[name] = hashlib.sha256(raw).hexdigest()
    return pins


def dispatch(text, full):
    chosen = iq3.dispatches(text, ARM, positive=True)
    all_requests = iq3.dispatches(full, ARM, positive=False)
    require(len(chosen) == 1 and chosen[0]['request'] == 2
        and [r['request'] for r in all_requests] == [1, 2]
        and chosen[0] == all_requests[1],
        'bind one warmup and one measured IQ3 request,not the two-run entry shape')
    return chosen[0]


def dispatch_controls():
    warm = 'strata IQ3_GU_DELTA request=1 arm=1 calls=0 nt1_rows=0 pair_rows=0\n'
    long = 'strata IQ3_GU_DELTA request=2 arm=1 calls=4 nt1_rows=128 pair_rows=128\n'
    dispatch(long, warm + long)
    bad = [(long, long), (long, warm + long + long.replace('request=2', 'request=3')),
        (long + long, warm + long), (long.replace('request=2', 'request=3'), warm + long),
        (long.replace('calls=4', 'calls=5'), warm + long),
        (long.replace('arm=1', 'arm=0'), warm + long),
        (long.replace('nt1_rows=128 pair_rows=128', 'nt1_rows=0 pair_rows=0'), warm + long)]
    for selected, whole in bad:
        try:
            dispatch(selected, whole)
        except RuntimeError:
            continue
        raise RuntimeError('invalid one-request IQ3 binding accepted')
    return dict(passed=True, positive=1, negative=len(bad), GPU_executed=False)


def environment(prior):
    source_root = str(Path(prior['runtime_manifest']['path']).parent)
    env = {k: v.replace(source_root, str(ROOT)) for k, v in prior['environment'].items()}
    env.update(UNITRACE_ChromeKernelLogging='1', UNITRACE_ChromeEventBufferSize='128')
    return env


def contract(row, prior, assets):
    expected_config = copy.deepcopy(prior['configuration'])
    expected_config['log'] = str(ROOT / 'engine.stderr')
    require(row['runs'] == 1 and row['read_tokens'] == 32769 and row['rows'] == 32768
        and row['configuration'] == expected_config and row['environment'] == environment(prior)
        and row['environment']['STRATA_IQ3_PAIR'] == str(ARM)
        and row['environment']['STRATA_CPU_WORK_CENSUS'] == '0'
        and row['environment']['STRATA_CPU_SERVICE'] == '0'
        and row['environment']['STRATA_DECODE_TIMING'] == '0'
        and row['native_validation_probe'] == prior['native_validation_probe']
        and row['native_profile']['mode'] == 'aggregate'
        and all(row['native_profile'][k] == v for k, v in assets.items())
        and row['budgets'] == dict(wall=1200, cpu=7200, rss_cap=104 << 30,
            text_cap=64 << 20, file_cap=512 << 20, total_cap=768 << 20)
        and row['whole_bytes'] == 1024 << 20
        and row['maximum_observer_ratio'] == 1.5,
        'freeze current IQ3 body,selector,observers,configuration and one-request extent')
    sample = prior['outcome']['samples'][0]
    require(row['request_sha256'] == sample['request_sha256']
        and row['observer_reference_prompt_ms'] == sample['prompt_ms'],
        'matching arm/body observer reference differs')


def context():
    ctx = entry.context('columns')
    ctx.ROOT, ctx.KEY, ctx.RUNS = ROOT, KEY, 1
    ctx.proof, ctx.pinned_sources = proof, pinned_sources
    original_current, original_route = ctx.current, ctx.profile_route

    def current(row):
        _, prior, providers, assets = selected_basis()
        contract(row, prior, assets)
        files, manifest = original_current(row)
        require(files == providers, 'model provider inventory differs from selected native')
        return files, manifest

    def route(text):
        result = original_route(text)
        result['IQ3_dispatch'] = dispatch(text, (ROOT / 'engine.stderr').read_text())
        return result

    ctx.current, ctx.profile_route = current, route
    return ctx


def prepare(ctx):
    _, prior, _, assets = selected_basis()
    controls = dispatch_controls()
    ctx.prepare()
    row = json.loads((ROOT / 'record.json').read_text())
    row['environment'] = environment(prior)
    (ROOT / 'trace').mkdir()
    row['native_profile'] = dict(profile.profiling(ROOT, assets), mode='aggregate')
    write(ROOT / 'plan.json', dict(binary=row['binary'],
        native_validation_probe=row['native_validation_probe'], native_profile=row['native_profile']))
    row['generated']['plan.json'] = digest(ROOT / 'plan.json')
    row['request_sha256'] = prior['outcome']['samples'][0]['request_sha256']
    row['observer_reference_prompt_ms'] = prior['outcome']['samples'][0]['prompt_ms']
    row['maximum_observer_ratio'] = 1.5
    row['IQ3_dispatch_controls'] = controls
    row['budgets'] = dict(wall=1200, cpu=7200, rss_cap=104 << 30,
        text_cap=64 << 20, file_cap=512 << 20, total_cap=768 << 20)
    row['whole_bytes'] = 1024 << 20
    row['profile_plan'] = dict(
        question='Which observed current-b310 GPU kernels,copies and service gaps occur inside guarded client decode?',
        order='After fresh current same-build active fixture control,one new model process,one ordinary short warmup and one exact32769-input/thinking-on request with1..24outputs. Active collection from native exec through normal exit;no pause or environment mutation.',
        constants='Current9922/b310,IQ3arm1,Q6columns1,static1256cache,5workers,spec4,chunk8192,short-read0,throughput BF16/GDN16/MTP16,heads1,131072INT8resident. Same closed arm1 model-entry body and immutable environment,plus active Chrome kernel collection. CPU_SERVICE0,CPU_WORK_CENSUS0,DECODE_TIMING0;the latter enables native timing by variable presence. No new disjoint-clock native code is compiled into this ELF.',
        acceptance='Committed source/build/provider/owner/body/count/journal/clock/validation/fault gates. Both frontend probes,guarded-window reader and one-request footer controls must pass. Require one warmup ID1 and long IQ3 ID2,positive long calls/NT1 rows and pair_rows=NT1_rows. Require actual native16/GDN/MTP/Q6 routes,one flushed trace,named kernels,unique native host-flow joins and correct PCI lane. Prompt time / matching closed arm1 same-body first-use prompt time<=1.5;this is a local observer-cost screen,not clean speed.',
        clocks='Four ordered RAW/MONOTONIC pairs with<=1ms brackets and<=50ms observed offset envelope. Same native/client/controller time namespace;BOOTTIME/MONOTONIC elapsed difference<=50ms. Interior starts after first nonempty client output+maximum observed offset+100ms and ends at client end+minimum offset-100ms;require>=100ms. Sampled drift is not a bound on unsampled error.',
        raw_budget='Bounded kernel-activity trace:512MiB/file,768MiB/model job,1GiBwhole;separate fixture control keeps its original180s/4GiB/64MiB job and128MiBwhole caps. Model stop-control accounting includes the completed trace,not only stop stdout. Retire complete raw traces/logs/JIT after required joins and compact reduction.',
        stopping='Stop first source/provider/owner/body/route/token/clock/validation/fault/budget/overhead failure;no retry,hash refresh or criterion change.',
        limits='Instrumented1..24-output diagnostic with the matching short-entry default pool-spin policy,not the long cohort20ms policy. Not representative long generation,observed strong/weak MTP class,clean PP/TG,full device coverage,physical traffic,saturation,critical path,task/agent,physical262144/RESTORE,ordinary-provider teardown or adoption.')
    row['criteria'] = row['profile_plan']['acceptance']
    row['retention'] = 'Keep compact device families,service unions,requested bytes,clock/request/selector evidence and original outcomes. Retire raw timeline and successful launch logs after required joins. No new inference build.'
    contract(row, prior, assets)
    ctx.save(row)
    print('Prepared current-IQ3 short device diagnostic;commit before host checks.', flush=True)


def host(ctx):
    ctx.host()
    row = json.loads((ROOT / 'record.json').read_text())
    owner = base.supervision.Owner(ROOT / 'window-reader-tests')
    command = [row['python_selected']['path'], '-B', str(P / 'tests/core/test_decode_window_trace.py')]
    case, out, err = owner.run('reader', command, row['environment'], P,
        wall=30, cpu=20, rss_cap=512 << 20, text_cap=64 << 10,
        file_cap=2 << 20, total_cap=8 << 20)
    good = base.supervision.closed(case) and case['exit_code'] == 0 and not out.stat().st_size
    row['window_reader_tests'] = dict(passed=good, native_exit_code=case.get('exit_code'),
        ownership_closed=base.supervision.closed(case), summary=err.read_text()[-512:])
    if not good:
        row.update(stage='failed', passed=False, host_passed=False)
    ctx.save(row)
    require(good, 'guarded-window reader tests failed')
    print('Current-source frontends and guarded-window controls passed;commit before capture.', flush=True)


def expected_environment(row):
    expected = profile.expected_target_environment(row, row['environment'], row['native_profile'])
    for name in ('UNITRACE_StartPaused', 'UNITRACE_HostTiming', 'UNITRACE_ChromeCallLogging'):
        expected.pop(name)
    expected['UNITRACE_Verbose'] = '1'
    return expected


def run(ctx):
    row = json.loads((ROOT / 'record.json').read_text())
    providers, manifest = ctx.current(row)
    require(row['stage'] == 'frontend-qualified' and row['host_passed']
            and row['window_reader_tests']['passed'] and not (ROOT / 'execution.json').exists(),
            'fresh frontend-qualified model only')
    env, prof = row['environment'], row['native_profile']
    require(not profile.session_path(prof['session']).exists(), 'preexisting collector session')
    require(env['EnableDirectSubmission'] == '0' and env['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD'] == '1'
            and env['ZES_ENABLE_SYSMAN'] == '0' and env['STRATA_SYCL_VERIFY_NATIVE_CAPTURE'] == '0',
            'qualified safety settings differ')
    outcome = dict(complete=False, passed=False, host_probe_accepted=False, model_executed=False)
    owner = controls = None
    offsets, clocks, identity = {}, [], None
    initial = time.clock_gettime_ns(time.CLOCK_BOOTTIME), time.monotonic_ns()
    cursor = profile.cursor()

    def persist():
        write(ROOT / 'execution.json', dict(outcome,
            active=bool((owner and owner.active) or (controls and controls.active)),
            commands=owner.commands if owner else [], controls=controls.commands if controls else []))

    def interact(io, command):
        nonlocal identity
        for stage in ('begin', 'end'):
            while True:
                line = io.line(900)
                if line.startswith('STRATA_PROFILE_GATE '): break
            clocks.append(profile.clock_pair())
            require(json.loads(line.removeprefix('STRATA_PROFILE_GATE ')) ==
                    dict(stage=stage, phase='prefill-only', run=0), 'request gate differs')
            claim = json.loads((ROOT / 'engine-start.json').read_text())['native']
            actual = base.supervision.identity(claim['pid'])
            require(actual and actual['start_ticks'] == claim['start_ticks']
                    and actual['sid'] == command['pid']
                    and str(Path('/proc', str(actual['pid']), 'exe').resolve()) == row['binary']['path']
                    and (identity is None or claim == identity), 'native ownership/executable differs')
            identity = claim
            namespace = os.readlink('/proc/self/ns/time')
            require(all(os.readlink(f'/proc/{pid}/ns/time') == namespace
                        for pid in (actual['pid'], command['pid'])), 'time namespace differs')
            actual_env = dict(v.split('=', 1) for v in Path('/proc', str(actual['pid']), 'environ').read_text().split('\0') if '=' in v)
            profile.same_environment(actual_env, expected_environment(row))
            mapped = entry.template.shared.runtime.executable_mappings(Path('/proc', str(actual['pid']), 'maps').read_text())
            require(mapped <= set(providers) | {row['binary']['path'], prof['collector']['path']}
                    and {row['native_validation_probe']['loader']['path'],
                         row['native_validation_probe']['library']['path'], prof['collector']['path'],
                         env['ZE_ENABLE_ALT_DRIVERS']} <= mapped, 'profile provider differs')
            entry.template.shared.python_runtime.check_running(command['pid'], P / 'tools/strata_bench.py',
                manifest['selected'], env, {v['realpath'] for v in manifest['inventory']['files'].values()}, row['command'][2:])
            profile.session_state(prof['session'], 0)
            entry.template.comparison.require_quiet_validation(ROOT)
            require(base.supervision.artifact_bytes(ROOT) <= row['whole_bytes'], 'whole output budget')
            delta = [a-b for a,b in zip((time.clock_gettime_ns(time.CLOCK_BOOTTIME), time.monotonic_ns()), initial)]
            require(min(delta) >= 0 and abs(delta[0]-delta[1]) <= 50000000, 'suspend/clock discontinuity')
            offsets[stage] = (ROOT / 'engine.stderr').stat().st_size
            outcome.update(model_executed=True, native_identity=claim)
            persist()
            print('Selected decode GPU request0 ' + stage, flush=True)
            io.send(b'continue\n')
        io.proc.stdin.close()
        deadline = time.monotonic() + 180
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll()
            require(time.monotonic() < deadline, 'model teardown deadline')
            if select.select([io.proc.stdout], [], [], .02)[0]:
                io.read(); io.pending.clear()
        io.exit(1)

    try:
        owner = base.supervision.Owner(ROOT / 'owned', persist)
        controls = base.supervision.Owner(ROOT / 'controls', persist)
        clocks.append(profile.clock_pair())
        command, _, _ = owner.run('bench', row['command'], env, P, interaction=interact, **row['budgets'])
        clocks.append(profile.clock_pair())
        require(base.supervision.closed(command) and command['exit_code'] == 0, 'owned model failed')
        outputs = list((ROOT / 'bench').glob('*/results.json'))
        require(len(outputs) == 1, 'one official model result required')
        result = json.loads(outputs[0].read_text())
        cohort = result['results']['prefill_only']
        require(result['complete'] and not result['results']['failures']
                and len(cohort['requests']) == len(cohort['count_preflight']) == 1, 'incomplete official result')
        sample, counted = cohort['requests'][0], cohort['count_preflight'][0]
        require(all(sample.get(k) is True for k in ('ok','eligible','stream_complete','thinking_requested','reasoning_observed'))
                and sample['actual_read_tokens'] == sample['actual_prompt_tokens'] == counted['counted_input_tokens'] == 32769
                and sample['actual_reused_tokens'] == 0 and 1 <= sample['generated_tokens'] <= 24
                and sample['request_sha256'] == counted['body_sha256'] == row['request_sha256'],
                'exact request/body/count differs')
        entry.template.comparison.require_valid_client_ttft(sample)
        journal = entry.template.shared.client.verify_request_journal(outputs[0].parent)
        require(journal['valid'] and journal['complete'] and not journal['pending'], 'request journal invalid')
        events = [json.loads(v) for v in (outputs[0].parent / 'requests.jsonl').read_text().splitlines()]
        samples = [e['sample'] for e in events if e['kind'] == 'result']
        require(len(samples) == 2 and [e['phase'] for e in events if e['kind'] == 'result'] ==
                ['warm-up/chat','prefill-only'], 'warmup/request history differs')
        end = gen.native_end(ROOT, row, dict(config=row['configuration']), row['boot'], command['pid'])
        require(identity == end['native'], 'live/final native identity differs')
        raw = (ROOT / 'engine.stderr').read_bytes(); text = raw.decode()
        require(not any(v in text for v in ('UR_RESULT_ERROR_','ZE_RESULT_ERROR_','terminate called',
            'teardown uncertain','[WARNING]','[ERROR]')) and
            not re.search(r'(?i)(?:dropped|lost)\s*[:=]\s*[1-9]', text), 'runtime/collector error')
        activation = entry.template.shared.runtime.validation.preload_evidence(text, identity['pid'])
        entry.template.comparison.require_quiet_validation(ROOT)
        route = ctx.profile_route(raw[offsets['begin']:offsets['end']].decode())
        timing = gen.decode_timing_evidence(text, samples)
        traces = list((ROOT / 'trace').rglob('*.json'))
        require(len(traces) == 1 and base.supervision.artifact_bytes(ROOT) <= row['whole_bytes'], 'trace/output extent')
        trace = json.loads(traces[0].read_text())['traceEvents']
        capture = profile.trace_summary(trace, identity['pid'], clocks[0]['raw_before_ns'], clocks[-1]['raw_after_ns'])
        selected = client_decode_window(sample['client_interval'], clocks)
        reduced = window.summarize(trace, native_pid=identity['pid'],
            begin_ns=selected['begin_ns'], end_ns=selected['end_ns'],
            capture_begin_ns=clocks[0]['raw_before_ns'], capture_end_ns=clocks[-1]['raw_after_ns'])
        ratio = sample['prompt_ms'] / row['observer_reference_prompt_ms']
        outcome.update(sample={k:sample[k] for k in ('actual_read_tokens','generated_tokens','prompt_ms','ttft_s','total_s','client_interval','request_sha256')},
            route=route, validation=activation, native_exit_code=0, clocks=clocks, selected_window=selected,
            observed_decode=reduced, full_capture_counts={k:capture[k] for k in ('gpu_events','host_events','operation_counts')},
            trace=dict(path=str(traces[0]),sha256=digest(traces[0]),bytes=traces[0].stat().st_size),
            observer_time_ratio=ratio, observer_cost_qualified=ratio <= row['maximum_observer_ratio'],
            IQ3_requests_verified=2, decode_timing=timing[-1],
            native_decode_timing_enabled_by_presence=True, clean_speed_qualified=False,
            complete_device_coverage_qualified=False, inference_adopted=False)
        require(outcome['observer_cost_qualified'], 'profile overhead exceeds frozen bound')
        ctx.current(row)
        entry.template.shared.models.model_pins(row['configuration'], row['model_assets'], full=True)
        outcome.update(complete=True, host_probe_accepted=True)
    except BaseException as error:
        outcome['error'] = repr(error)
    finally:
        if owner and owner.active is None and profile.session_path(prof['session']).exists():
            try:
                command, _, _ = controls.run('stop', [prof['launcher']['path'],'--stop',prof['session']],
                    env, P, wall=15,cpu=15,rss_cap=2<<30,text_cap=1<<20,file_cap=2<<20,
                    # Owner accounts output.parent,including the completed
                    # model trace. Keep the already declared whole-job cap.
                    total_cap=row['budgets']['total_cap'])
                require(base.supervision.closed(command) and command['exit_code'] == 0, 'collector close failed')
            except BaseException as error: outcome['control_close_error'] = repr(error)
        if owner: entry.template.shared.runtime.validation.finish(outcome, owner, cursor, row['boot'])
        controls_closed = bool(controls and controls.active is None and
            all(base.supervision.closed(v) for v in controls.commands))
        outcome['ownership_closed'] = bool(outcome.get('ownership_closed') and controls_closed)
        outcome['passed'] = bool(outcome.get('passed') and outcome['complete'] and outcome['ownership_closed']
            and outcome.get('driver_after') == row['driver'] and not profile.session_path(prof['session']).exists()
            and 'control_close_error' not in outcome)
        row.update(outcome=outcome, stage='closed' if outcome['passed'] else 'failed', passed=outcome['passed'],
            complete=outcome['passed'], active=bool((owner and owner.active) or (controls and controls.active)),
            ownership_closed=outcome['ownership_closed'], model_executed=outcome['model_executed'])
        persist(); ctx.save(row)
    print(json.dumps({k:outcome.get(k) for k in ('passed','complete','ownership_closed','observer_time_ratio','error')}),flush=True)
    require(outcome['passed'],outcome.get('error','decode GPU diagnostic failed'))


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in
        ('control-prepare', 'control-check', 'prepare', 'host', 'run'),
        'use control-prepare|control-check|prepare|host|run')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base.guards.no_model_owners([])
        mode = sys.argv[1]
        if mode == 'control-prepare':
            prepare_control()
        elif mode == 'control-check':
            collector_context().check()
        else:
            ctx = context()
            {'prepare': lambda: prepare(ctx), 'host': lambda: host(ctx),
             'run': lambda: run(ctx)}[mode]()
