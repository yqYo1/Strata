#!/usr/bin/env python3
"""Bounded selected-engine decode GPU attribution; repository LICENSE.

Main-authored composition of the maintained official benchmark, selected native
qualification and owned collector. No native rebuild or XeStrata input. Keep
active collection through graph construction; reduce an interior client window.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import select
import sys
import time

sys.dont_write_bytecode = True
import cache_map_cpu_census as short
import decode_window_trace as window
import profile_selected_compat as collector

entry, base, profile = short.entry, short.base, collector.profile
P, B, RESULT = short.P, short.B, short.RESULT
ROOT = entry.model.ROOT / 'decode-gpu-service-v1'
KEY = 'cache_map_decode_gpu_service_v1'
SOURCE = Path(__file__).resolve()
require, digest, write = base.require, base.digest, base.write
CONTROL_KEY = 'cache_map_compat_collector_active_v1'
CONTROL_ROOT = entry.model.ROOT / 'compat-collector-active-v1'


def client_decode_window(interval, clocks, *, guard_ns=100000000):
    """Select a guarded RAW interval inside the client's observed decode.

    Use the whole observed offset envelope rather than treating MONOTONIC as
    RAW or equating the first client output with an exact device boundary.
    Same time namespaces and suspend continuity remain caller prerequisites.
    """
    names = ('request_begin_monotonic_ns', 'first_output_monotonic_ns',
             'request_end_monotonic_ns')
    window.require(type(interval) is dict and set(interval) == set(names), 'client clock fields')
    begin, first, end = (interval[n] for n in names)
    window.require(all(type(v) is int and v > 0 for v in (begin, first, end))
            and begin < first < end, 'client clock order/units')
    window.require(type(clocks) is list and len(clocks) == 4
            and type(guard_ns) is int and guard_ns >= 100000000, 'clock/guard plan')
    lower, upper, previous = [], [], None
    for clock in clocks:
        window.require(type(clock) is dict and set(clock) ==
                {'raw_before_ns', 'monotonic_ns', 'raw_after_ns'}, 'clock pair fields')
        a, m, b = (clock[n] for n in ('raw_before_ns', 'monotonic_ns', 'raw_after_ns'))
        window.require(all(type(v) is int and v > 0 for v in (a, m, b))
                and 0 <= b - a <= 1000000, 'clock pair bracket')
        window.require(previous is None or previous[0] < a and previous[1] < m,
                'clock pair order')
        lower.append(a - m)
        upper.append(b - m)
        previous = b, m
    window.require(clocks[1]['monotonic_ns'] <= begin < end <= clocks[2]['monotonic_ns'],
            'client interval outside request gates')
    offset_min, offset_max = min(lower), max(upper)
    window.require(offset_max - offset_min <= 50000000, 'observed RAW/MONOTONIC offset drift')
    start, finish = first + offset_max + guard_ns, end + offset_min - guard_ns
    window.require(finish - start >= 100000000, 'guarded decode window too short')
    return dict(begin_ns=start, end_ns=finish, guard_ns=guard_ns,
        observed_offset_min_ns=offset_min, observed_offset_max_ns=offset_max,
        clock_pair_count=len(clocks),
        limits='Interior client decode only;observed clock-offset envelope is not a '
               'proof of unsampled clock error or an exact prefill/device boundary.')


def collector_basis(native):
    data = base.committed(RESULT)
    control = json.loads((CONTROL_ROOT / 'record.json').read_text())
    require(data[CONTROL_KEY] == collector.compact(control)
            and control['passed'] and control['complete'] and control['ownership_closed']
            and not control['active'] and control['native_source_commit'] == native['source_commit']
            and control['engine'] == native['binaries']['strata']
            and control['binary'] == native['binaries']['sycl_submission_test']
            and len(control['cases']) == 1
            and control['cases'][0]['trace']['collection_active_from_exec']
            and control['cases'][0]['trace']['observed_kernels'] == 62,
            'closed selected active-from-exec collector control required')
    require(base.controls(control['source_commit'], controller=collector.SOURCE) == control['sources'],
            'original collector control source changed')
    assets = profile.collector_assets(data)
    require(control['assets'] == assets, 'collector assets differ')
    return assets


def proof():
    native, gate, providers = entry.proof()
    collector_basis(native)
    return native, gate, providers


def pinned_sources(revision):
    pins = entry.pinned_sources(revision)
    pins.update(base.controls(revision, controller=SOURCE))
    name = 'tests/core/test_decode_window_trace.py'
    raw = base.basis.build.git(P, 'show', revision + ':' + name)
    require((P / name).read_bytes() == raw, 'commit exact window reader tests')
    pins[name] = hashlib.sha256(raw).hexdigest()
    return pins


def context():
    ctx = entry.context('zero')
    ctx.ROOT, ctx.KEY, ctx.RUNS = ROOT, KEY, 1
    ctx.proof, ctx.pinned_sources = proof, pinned_sources
    return ctx


def prepare(ctx):
    entry.prepare(ctx, 'zero')
    row = json.loads((ROOT / 'record.json').read_text())
    prior = json.loads((short.ROOT / 'record.json').read_text())
    require(base.committed(RESULT)[short.KEY] == short.context().compact(prior)
            and prior['passed'] and prior['ownership_closed'] and not prior['active']
            and prior['binary'] == row['binary'], 'closed matched short census basis required')
    row['environment'].update(UNITRACE_ChromeKernelLogging='1',
        UNITRACE_ChromeEventBufferSize='128', STRATA_CPU_WORK_CENSUS='1', STRATA_DECODE_TIMING='1')
    assets = profile.collector_assets(base.committed(RESULT))
    (ROOT / 'trace').mkdir()
    row['native_profile'] = profile.profiling(ROOT, assets)
    row['native_profile']['mode'] = 'aggregate'
    write(ROOT / 'plan.json', dict(binary=row['binary'],
        native_validation_probe=row['native_validation_probe'], native_profile=row['native_profile']))
    row['generated']['plan.json'] = digest(ROOT / 'plan.json')
    row['request_sha256'] = prior['outcome']['samples'][0]['request_sha256']
    row['observer_reference_prompt_ms'] = prior['outcome']['samples'][0]['prompt_ms']
    row['maximum_observer_ratio'] = 1.5
    row['budgets'] = dict(wall=1200, cpu=7200, rss_cap=104 << 30,
        text_cap=64 << 20, file_cap=512 << 20, total_cap=768 << 20)
    row['whole_bytes'] = 1024 << 20
    row['profile_plan'] = dict(
        question='Which observed device kernels,copies and service gaps occur after first output in the unchanged selected engine?',
        order='One new process;unchanged short warmup then one exact32769-input/thinking-on request with1..24outputs. Collect from native exec through normal exit;no collection pause or environment mutation.',
        constants='Selected479a/552b and9ELFs,original128KINT8resident capacity,1256static slots,5workers+host,spec4,chunk8192,throughput BF16/GDN16/MTP16,heads1,adapt0,qualified safety/validation. Same exact request0body as the short census.',
        acceptance='Commit this source/plan and pass both frontend probes plus window reader tests first. Require original native/fixture/provider/model/body/count/route/journal/clock/validation/fault/closure checks. Collect from exec with the fixture-qualified aggregate mode plus immutable ChromeKernelLogging1,128event buffer. Require unique native graph flow IDs and correct PCI lane,nonempty named kernels/host activities,and no reported lost-event errors. Require profile prompt time / same-body first-use short-census time<=1.5;this single local overhead screen is not clean speed.',
        clocks='Same native/client/controller time namespace;four RAW/MONOTONIC bracketing samples,each<=1ms,ordered,observed offset envelope<=50ms. BOOTTIME/MONOTONIC elapsed drift<=50ms. Select first client output+maximum observed offset+100ms to client end+minimum observed offset-100ms;require>=100ms remaining. No claim that sampled drift bounds every unsampled offset.',
        raw_budget='The prior32Kfull-API trace was266967750bytes;kernel-activity-only capture is bounded to512MiB/file,768MiB/job and1GiBwhole. Required temporarily to bind graph capture IDs to replay and select decode after flush. Retire full timeline/logs/JIT after compact attribution is verified.',
        limits='Diagnostic24-output sample,not sustained TG,MTP strong/weak,clean speed,full graph critical path,complete device coverage,physical bytes,hardware saturation,task/agent,262144/RESTORE or adoption.',
        stopping='Stop first source/provider/owner/fault/validation/body/clock/count/budget/overhead failure;no retry,hash refresh or changed criteria.')
    row['criteria'] = row['profile_plan']['acceptance']
    row['retention'] = 'Keep compact device families/request identity,observed service unions,requested copy bytes,clock guards,observer ratio and original outcomes. Retire successful raw output after attribution and required joins;no new native build.'
    ctx.save(row)
    print('Prepared one active-from-exec32K/24-output decode diagnostic;commit before host checks.', flush=True)


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
    if not good: row.update(stage='failed', passed=False, host_passed=False)
    ctx.save(row)
    require(good, 'decode window reader tests failed')
    print('Frontend and guarded-window controls passed;commit before model execution.', flush=True)


def expected_environment(row):
    expected = profile.expected_target_environment(row, row['environment'], row['native_profile'])
    for key in ('UNITRACE_StartPaused', 'UNITRACE_HostTiming', 'UNITRACE_ChromeCallLogging'):
        expected.pop(key)
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
        end = short.gen.native_end(ROOT, row, dict(config=row['configuration']), row['boot'], command['pid'])
        require(identity == end['native'], 'live/final native identity differs')
        raw = (ROOT / 'engine.stderr').read_bytes(); text = raw.decode()
        require(not any(v in text for v in ('UR_RESULT_ERROR_','ZE_RESULT_ERROR_','terminate called',
            'teardown uncertain','[WARNING]','[ERROR]')) and
            not re.search(r'(?i)(?:dropped|lost)\s*[:=]\s*[1-9]', text), 'runtime/collector error')
        activation = entry.template.shared.runtime.validation.preload_evidence(text, identity['pid'])
        entry.template.comparison.require_quiet_validation(ROOT)
        route = ctx.profile_route(raw[offsets['begin']:offsets['end']].decode())
        timing = short.gen.decode_timing_evidence(text, samples)
        args = row['configuration']['args']
        formats = short.census.manifest_formats(Path(args[args.index('--pack')+1]) / 'native_experts.txt')
        membership = short.census.expected_membership(args[args.index('--expert-profile')+1])
        bindings = [dict(prompt_tokens=s['prompt_tokens'], windows=t['windows'], rows=t['work']['target_rows'],
            cpu_jobs=t['work']['cpu_unique_jobs'], cpu_entries=t['work']['cpu_entries'],
            pool_ms=t['ms_per_window']['CPU_pool_run'] * t['windows']) for s,t in zip(samples,timing)]
        work = short.census.extract(text, bindings, formats, membership)
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
            census_requests_verified=len(work), decode_timing=timing[-1], clean_speed_qualified=False,
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
                    env, P, wall=15,cpu=15,rss_cap=2<<30,text_cap=1<<20,file_cap=2<<20,total_cap=8<<20)
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
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare','host','run'), 'use prepare|host|run')
    ctx = context()
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base.guards.no_model_owners([])
        {'prepare':lambda:prepare(ctx),'host':lambda:host(ctx),'run':lambda:run(ctx)}[sys.argv[1]]()
