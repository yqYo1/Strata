#!/usr/bin/env python3
"""Observe one existing prefill route between two clean controls; repository LICENSE."""

import collections
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import current_pti_admission as pti
import prefill_prefix_expert_host_stage as stage
import profile_split as profile

from prefill_prefix_expert_host_stage import B, RESULT, basis, build, builder, require, signature

THIS = 'tools/sycl/prefill_prefix_expert_profile.py'
ROOT = B / 'prefill-prefix-expert-stage-profile-v1'
KEY = 'prefill_prefix_expert_stage_profile_v1'
TIMING = 'prefill_prefix_expert_host_stage_timing_v1'
ROLES = ('clean-before', 'profile', 'clean-after')
READ, RUNS = 32769, 2
OBSERVERS = dict(STRATA_TRACE='1', STRATA_PREFILL_CPU_WAIT_PROFILE='1',
    STRATA_SYCL_PREFILL_MOE_CENSUS='1', UNITRACE_ChromeKernelLogging='1',
    UNITRACE_ChromeCallLogging='1', UNITRACE_ChromeEventBufferSize='128')


def assets():
    tools = pti.verify_tools(pti.committed_json(pti.RESULT, pti.CHECKOUT))
    return dict(launcher=tools['artifacts']['unitrace'],
        collector=tools['artifacts']['libunitrace_tool.so']), signature(tools)


def read(path):
    return json.loads(path.read_bytes())


def campaign():
    stage_timing = stage.model_context('timing')
    stage_numerical = stage.model_context('numerical')
    spec = importlib.util.spec_from_file_location('prefix_expert_stage_profile', basis.core.__file__)
    ctx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ctx)
    ctx.ROOT, ctx.KEY, ctx.THIS = ROOT, KEY, THIS
    ctx.READS, ctx.ORDER, ctx.RUNS, ctx.CLEAN_FAST = (READ,) * 3, [(READ, 1)] * 3, RUNS, True
    base_current, base_native = ctx.current, ctx.native_context

    def environment(row, plan):
        env = stage_timing.environment(row, plan)
        env[stage.SELECTOR] = 'all'
        if plan['index'] == 1:
            env.update(OBSERVERS)
        return env

    def current(row):
        providers = base_current(row)
        earlier = basis.validation.committed(RESULT, (TIMING,))
        require(signature(earlier) == row['speed_reference'] and earlier['passed']
                and earlier['complete'] and earlier['ownership_closed']
                and earlier['binary'] == row['binary'], 'same ELF clean 64K basis required')
        tools, proof = assets()
        require(proof == row['collector_build_proof'] and row['clock_controls'] == ctx.profile_window_controls()
                and [p['role'] for p in row['plans']] == list(ROLES), 'frozen observer proof differs')
        prof = row['plans'][1]['native_profile']
        require(prof['mode'] == 'aggregate' and all(prof[k] == v for k, v in tools.items())
                and all((p['trace_progress'], 'native_profile' in p) == (i == 1, i == 1)
                        and p['read'] == READ and p['variant'] == 'all'
                        for i, p in enumerate(row['plans'])), 'frozen observer comparison differs')
        return providers

    def native_context(row, plan):
        native = base_native(row, plan)
        if 'native_profile' in plan:
            native['native_profile'] = plan['native_profile']
        return native

    def request_route(log, plan):
        route = stage_numerical.request_route(log, plan)
        if plan['trace_progress']:
            cpu = profile.d.gen.common.extract_cpu_observers(log, 4)
            host, wait = cpu['host'], cpu['wait']
            status = profile.status_summary(log)
            require(status['scopes'] == 48, 'expected one CPU status scope per layer')
            route['cpu_partition'] = dict(
                thread={k: float(wait[k]) for k in ('wall_s', 'main_cpu_s', 'wait_wall_s',
                    'wait_cpu_s', 'outside_wait_cpu_s')},
                unclassified={k: float(host['unclassified_' + k + '_s']) for k in ('wall', 'cpu')},
                host_outside_known_waits={name: {k: float(host[name + '_outside_wait_' + k + '_s'])
                    for k in ('wall', 'cpu')} for name in profile.d.gen.common.CPU_HOST_BASE
                    + profile.d.gen.common.CPU_HOST_MOE},
                nested_status=status)
        return route

    ctx.basis, ctx.current, ctx.environment = stage_timing.basis, current, environment
    ctx.native_context, ctx.request_route, ctx.check_sample = native_context, request_route, stage_timing.check_sample
    ctx.decision = ctx.profile_decision
    return ctx


def prepare(ctx):
    earlier = basis.validation.committed(RESULT, (TIMING,))
    require(earlier['stage'] == 'closed' and earlier['complete'] and earlier['ownership_closed'],
            'closed clean same-ELF timing reference required')
    tools, proof = assets()
    ctx.prepare()
    row = read(ROOT / 'record.json')
    fixed_environments = []
    for i, plan in enumerate(row['plans']):
        plan.update(role=ROLES[i], variant='all', bank=512, window=10, relocate=0,
                    trace_progress=i == 1, upload='jit')
        if i == 1:
            root = Path(plan['root'])
            prof = profile.profiling(root, tools)
            prof['mode'] = 'aggregate'
            plan['native_profile'] = prof
            config = root / 'plan.json'
            ctx.write(config, dict(read(config), native_profile=prof))
            plan['generated']['plan.json'] = ctx.digest(config)
        env = ctx.environment(row, plan)
        require(basis.comparison.fingerprint(env) == plan['environment_sha256'],
                'prepared environment changed')
        fixed_environments.append({k: v.replace(plan['root'], '<case>') for k, v in env.items()
                                   if k not in OBSERVERS})
    require(all(env == fixed_environments[0] for env in fixed_environments),
            'non-observer environment differs across arms')
    row.update(speed_reference=signature(earlier), collector_build_proof=proof,
        clock_controls=ctx.profile_window_controls(), budgets=dict(row['budgets'], file_cap=1 << 30,
        total_cap=1536 << 20), whole_bytes=2 << 30,
        plan=dict(question='How do requested copies, kernel intervals and host waits overlap on the all-staging route?',
            fixed='Same qualified ELF, native1/host USM/early issue/deferred reader/all staging, 512 slots, '
                  'window10, BF16, 32769 input tokens, 24 generated tokens including thinking, immutable decode.',
            comparison='Fresh clean, active-from-exec PTI plus in-process CPU scopes, fresh clean. '
                       'Two requests per process. Compare first use and second request separately. '
                       'Same copies/selected IDs/dispatch and draft counts. CPU scopes and API durations overlap.',
            gate='Verify owned closure, exact runtime and GPU identity, no new faults, unique trace IDs, '
                 'request brackets and complete CPU accounting. Require every observed prompt and client TTFT '
                 'within 1.20x of both clean controls at the corresponding request position.',
            stopping='Stop at the first invalid run, fault or ownership/budget failure. No retry or pin refresh.',
            limits='Observed API payload and service/overlap only. Physical PCIe traffic, arithmetic utilization '
                   'and same-layer dependencies are unmeasured. No performance adoption or speedup decision.',
            retention='Keep compact samples and CPU/GPU request summaries, retaining an unresolved diagnostic '
                      'only with a named consumer. Retire closed raw traces and benchmark output.'))
    ctx.save(row)


def window_summary(events, window, selected):
    begin, end = (window[k] / 1000 for k in ('begin_raw_ns', 'end_raw_ns'))
    clipped, h2d, kernels, copies = [], [], [], collections.defaultdict(list)
    for event in events:
        if event.get('ph') != 'X' or event.get('cat') not in ('gpu_op', 'cpu_op'):
            continue
        ts, finish = event['ts'], event['ts'] + event['dur']
        lo, hi = max(ts, begin), min(finish, end)
        if lo >= hi:
            continue
        op = dict(event, ts=lo, dur=hi - lo)
        clipped.append(op)
        if event['cat'] == 'gpu_op':
            h2d_match = re.fullmatch(r'zeCommandListAppendMemoryCopy\(H2D\)\[(\d+)\]', event['name'])
            if h2d_match:
                h2d.append(op)
                copies[(event['pid'], event['tid'])].append((int(h2d_match[1]), lo == ts and hi == finish))
            elif not event['name'].startswith('zeCommandList'):
                kernels.append(op)
    require(kernels and h2d, 'request kernels or H2D intervals missing')
    lanes = [dict(pid=pid, tid=tid, count=len(items), bytes=sum(n for n, _ in items),
                  boundary_crossing=sum(not whole for _, whole in items)) for (pid, tid), items in sorted(copies.items())]
    matching = [lane for lane in lanes if (lane['count'], lane['bytes'], lane['boundary_crossing']) ==
                (selected['experts'], selected['bytes'], 0)]
    wait = [op for op in clipped if op['cat'] == 'cpu_op' and op['name'] == 'zeCommandListHostSynchronize']
    ku, hu, either = map(profile.union_us, (kernels, h2d, kernels + h2d))
    return dict(clock=window, services=profile.family_summary(clipped, window['begin_raw_ns'], window['end_raw_ns']),
        interval_s=dict(kernel_only=(either - hu) / 1e6, h2d_only=(either - ku) / 1e6,
            kernel_and_h2d=max(0., ku + hu - either) / 1e6, neither=max(0., end - begin - either) / 1e6),
        command_list_host_synchronize=dict(calls=len(wait), window_union_s=profile.union_us(wait) / 1e6),
        h2d_lanes=lanes, prefix_count_byte_matching_lanes=matching,
        limits='A same-count/bytes lane is a candidate correlation. API wall time may overlap CPU scopes.')


def compare_work(outcome, row, plan):
    def work(process):
        keys = ('uploads', 'requested_bank_H2D_bytes', 'requested_source_map_H2D_bytes',
                'actual_GU_dispatch', 'copy_transport', 'stage_order')
        return [({k: route[k] for k in keys}, route['prefix_upload'], sample['drafts_offered'],
                 sample['drafts_accepted']) for route, sample in zip(process['routes'], process['samples'], strict=True)]

    require(all(work(outcome) == work(p) for p in row['processes']),
            'matched selected experts, copies, dispatch or MTP acceptance differs')
    outcome['role'] = plan['role']


def run(ctx, index):
    row = read(ROOT / 'record.json')
    plan = row['plans'][index]
    prof, root = plan.get('native_profile'), Path(plan['root'])
    ctx.after_execution = compare_work
    if not prof:
        def quiet(log, _plan):
            require(not any(marker in log for marker in ('strata prefill cpu wait:',
                'strata prefill status:', 'UNITRACE Profiling Result')), 'clean control has observer output')

        ctx.validate_model_text = quiet
        basis.comparison.run(index, ctx)
        return
    session = profile.session_path(prof['session'])
    require(not session.exists(), 'fresh owned collector session required')
    clocks, capture, routes = [profile.clock_pair()], {}, []

    def check_environment(observed, _expected, selected):
        claim = read(root / 'engine-start.json')['native']
        require(os.readlink(f'/proc/{claim["pid"]}/ns/time') == os.readlink('/proc/self/ns/time'),
                'native/controller time namespace differs')
        profile.session_state(prof['session'], 0)
        expected = profile.expected_target_environment(ctx.native_context(row, selected),
                                                       ctx.environment(row, selected), prof)
        for key in ('UNITRACE_StartPaused', 'UNITRACE_HostTiming'):
            expected.pop(key)
        expected['UNITRACE_Verbose'] = '1'
        profile.same_environment({k.decode(): v.decode() for k, v in observed.items()}, expected)
        clocks.append(profile.clock_pair())

    original_route = ctx.request_route

    def check_route(log, selected):
        route = original_route(log, selected)
        routes.append(route)
        return route

    def reduce(log, _plan):
        require(not any(s in log for s in ('[WARNING]', '[ERROR]')), 'collector warning or error')
        clocks.append(profile.clock_pair())
        path, = (root / 'trace').rglob('*.json')
        require(path.stat().st_size <= row['budgets']['file_cap'], 'collector trace exceeds budget')
        native = read(root / 'engine-exit.json')['native']['pid']
        events = read(path)['traceEvents']
        summary = profile.trace_summary(events, native, clocks[0]['raw_before_ns'], clocks[-1]['raw_after_ns'])
        output, = (root / 'bench').glob('*/results.json')
        samples = read(output)['results']['prefill_only']['requests']
        require(len(samples) == len(routes) == RUNS and len(clocks) == 2 + 2 * RUNS, 'trace clocks/requests differ')
        windows = [window_summary(events, ctx.profile_window(s['client_interval'], clocks), route['prefix_upload'])
                   for s, route in zip(samples, routes, strict=True)]
        capture.update(trace=dict(path=str(path), sha256=ctx.digest(path), bytes=path.stat().st_size),
                       whole_process=summary, windows=windows, clocks=clocks)

    original_save = ctx.save

    def finish(value):
        outcome = value['processes'][-1]
        cleanup = None
        try:
            require(outcome['ownership_closed'], 'model ownership must close before collector cleanup')
            cleanup = ctx.close_profile_session(value, plan)
            outcome['collector_cleanup'] = cleanup
            require(cleanup['control_closed'] and cleanup['session_removed'] and not cleanup.get('error'),
                    'collector session cleanup failed')
        except Exception as error:
            outcome.update(passed=False, complete=False, ownership_closed=False)
            outcome.setdefault('error', repr(error))
            value.update(stage='failed', passed=False, complete=False, ownership_closed=False)
        finally:
            value['active'] |= bool(cleanup and cleanup['active'])
        outcome.update(resource_observation=capture, collector_session_removed=not session.exists())
        original_save(value)

    ctx.check_native_environment = check_environment
    ctx.native_extra_maps = lambda _: {prof['collector']['path']}
    ctx.request_route, ctx.validate_model_text, ctx.save = check_route, reduce, finish
    basis.comparison.run(index, ctx)


if __name__ == '__main__':
    require(sys.executable == basis.reference.model.PYTHON, 'use pinned Python')
    action, = sys.argv[1:]
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        builder.guards.no_model_owners([])
        build.ROOT, build.KEY = build.STAGE_ROOT, build.STAGE_BUILD
        ctx = campaign()
        if action == 'prepare':
            prepare(ctx)
        elif action == 'host':
            basis.comparison.host(ctx)
        else:
            run(ctx, int(action))
