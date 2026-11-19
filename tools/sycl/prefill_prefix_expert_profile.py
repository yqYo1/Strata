#!/usr/bin/env python3
"""Measure CPU wait scopes in the fixed prefill route; repository LICENSE."""

import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import profile_split as profile
from prefill_prefix_expert_host_stage import B, RESULT, basis, build, builder, require, signature
import prefill_prefix_expert_host_stage as stage

THIS = 'tools/sycl/prefill_prefix_expert_profile.py'
ROOT = B / 'prefill-prefix-expert-cpu-wait-v2'
KEY = 'prefill_prefix_expert_cpu_wait_v2'
PREPARE_REFUSAL = 'prefill_prefix_expert_cpu_wait_v1'
PREVIOUS = 'prefill_prefix_expert_stage_profile_v2'
TIMING = 'prefill_prefix_expert_host_stage_timing_v1'
ROLES = ('clean-before', 'cpu-wait-scopes', 'clean-after')
READ, RUNS = 32769, 2
SELECTOR = 'STRATA_PREFILL_CPU_WAIT_PROFILE'
SCOPES = ('strata prefill cpu wait:', 'strata prefill cpu host:', 'strata prefill status:')
DECISIONS = 'docs/SYCL_NATIVE16_PREFILL.md'


def read(path):
    return json.loads(path.read_bytes())


def cpu_partition(log):
    common = profile.d.gen.common
    cpu = common.extract_cpu_observers(log, 4)
    host, wait = cpu['host'], cpu['wait']
    status = profile.status_summary(log)
    require(status['scopes'] == 48, 'expected one CPU status scope per layer')
    waits = {name: dict(calls=int(wait[name + '_count']), cpu_s=float(wait[name + '_cpu_s']),
                        wall_s=float(wait[name + '_wall_s']))
             for name in ('compute', 'transfer', 'event', 'compat', 'direct')}
    require(sum(v['calls'] for v in waits.values()) == int(wait['wait_count'])
            and all(abs(sum(v[k] for v in waits.values()) - float(wait['wait_' + k])) < 1e-7
                    for k in ('cpu_s', 'wall_s')), 'wait kinds do not conserve')
    return dict(thread={k: float(wait[k]) for k in ('wall_s', 'main_cpu_s', 'wait_wall_s',
                    'wait_cpu_s', 'outside_wait_cpu_s')}, waits=waits,
        unclassified={k: float(host['unclassified_' + k + '_s']) for k in ('wall', 'cpu')},
        host_outside_known_waits={name: {k: float(host[name + '_outside_wait_' + k + '_s'])
            for k in ('wall', 'cpu')} for name in common.CPU_HOST_BASE + common.CPU_HOST_MOE},
        nested_status=status)


def reference(row):
    timing = basis.validation.committed(RESULT, (TIMING,))
    previous = basis.validation.committed(RESULT, (PREVIOUS,))
    require(timing['stage'] == previous['stage'] == 'closed'
            and all(p['passed'] and p['complete'] and not p['active'] and p['ownership_closed']
                    and p['binary'] == row['binary'] for p in (timing, previous))
            and not previous['decision']['normal_route_attribution_qualified'],
            'exact closed speed and observer references required')
    return {TIMING: signature(timing), PREVIOUS: signature(previous)}


def admitted_stage_basis():
    row = read(build.STAGE_ROOT / 'record.json')
    require(basis.validation.committed(RESULT, (build.STAGE_BUILD,)) == builder.compact(row)
            and row['stage'] == 'built' and all(row[k] for k in ('passed', 'complete', 'ownership_closed'))
            and not row['active'] and row['host_control']['passed'], 'the exact closed stage build is required')
    native, foundation, speed = basis.basis(expected=row['foundation'])
    archive = build.archive()
    require(row['archive'] == {key: signature(value) for key, value in archive.items()}
            and row['foundation'] == foundation and row['recipe'] == build.recipe(archive)
            and row['driver'] == builder.guards.driver_guard()
            and row['retained_decode_reference'] == builder.decode_reference(), 'native basis or recipe changed')

    controls = row['controller_sources']
    recovered = dict(build.source.blobs(build.P, row['source_commit'], controls))
    require(build.hashed(recovered) == controls, 'original committed controller inputs differ')
    current = build.source.pin(build.P, list(controls))['source_sha256']
    refusal = basis.validation.committed(RESULT, (PREPARE_REFUSAL,))
    changes = {k: dict(original=h, current=current.get(k)) for k, h in controls.items() if h != current.get(k)}
    inputs = row['actual_inputs']
    require(refusal['stage'] == 'blocked-at-prepare' and not refusal['GPU_executed']
            and changes == refusal['changed_pinned_inputs'] and set(changes) == {DECISIONS}
            and current.keys() == controls.keys() and DECISIONS not in row['source_files']
            and all(Path(path).name != Path(DECISIONS).name
                    for field in ('compiled', 'generated', 'links', 'objects', 'produced') for path in inputs[field]),
            'only the previously recorded committed documentation delta is permitted')
    source = build.recover(native, archive, row['source_commit'])
    require(build.hashed(source) == row['source_files']
            and all(build.sha(build.STAGE_ROOT / 'source' / path) == digest
                    for path, digest in row['source_files'].items())
            and all(build.sha(Path(dep['root']) / path) == digest for dep in native['dependencies']
                    for path, digest in dep['source_sha256'].items())
            and all(build.sha(path) == digest for path, digest in row['configured_files'].items()),
            'native source, dependency or configured-file mismatch')
    require(all(build.sha(path) == digest for field in ('compiled', 'generated', 'links', 'objects', 'produced')
                for path, digest in inputs[field].items())
            and all(build.sha(binary['path']) == binary['sha256'] for binary in row['binaries'].values()),
            'actual compiled inputs or executable differ')

    smoke = basis.validation.committed(RESULT, (stage.SMOKE,))
    numerical = basis.validation.committed(RESULT, (stage.NUMERICAL,))
    native_digest = signature(builder.compact(row))
    require(smoke['passed'] and smoke['ownership_closed'] and smoke['native_exit_code'] == 0
            and smoke['native_build'] == native_digest and numerical['stage'] == 'closed'
            and numerical['passed'] and numerical['complete'] and numerical['ownership_closed']
            and not numerical['active'] and numerical['decision']['matched_work']
            and numerical['binary'] == row['binaries']['strata'], 'same-build smoke and numerical gates required')
    model = read(basis.reference.model.ROOT / 'record.json')
    require(model['models'] == speed['model_assets'] and model['probe'] == speed['native_validation_probe'],
            'exact model assets and validation probe required')
    model['observer_admission'] = dict(gates={stage.SMOKE: signature(smoke),
            'stage_order_numerical': signature(numerical)}, native_build=native_digest,
        fast_configuration_diagnostic=speed['observer_admission']['fast_configuration_diagnostic'],
        original_controller_count=len(controls), original_controller_digest=signature(controls),
        historical_documentation=changes, original_preparation_refusal=signature(refusal))
    providers = read(Path(speed['runtime_manifest']['path']))
    return dict(binaries=row['binaries'], native_source_commit=row['source_commit']), model, providers, None


def campaign():
    stage_timing = stage.model_context('timing')
    spec = importlib.util.spec_from_file_location('prefix_expert_cpu_wait', basis.core.__file__)
    ctx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ctx)
    ctx.ROOT, ctx.KEY, ctx.THIS = ROOT, KEY, THIS
    ctx.READS, ctx.ORDER, ctx.RUNS, ctx.CLEAN_FAST = (READ,) * 3, [(READ, 1)] * 3, RUNS, True
    base_current = ctx.current

    def environment(row, plan):
        env = stage_timing.environment(row, plan)
        env.update({stage.SELECTOR: 'all', SELECTOR: str(int(plan['index'] == 1))})
        require(not any(key.startswith('UNITRACE_') for key in env)
                and not env.get('STRATA_TRACE') and env['STRATA_SYCL_PREFILL_MOE_CENSUS'] == '0',
                'unexpected instrumentation in the CPU wait comparison')
        return env

    def current(row):
        providers = base_current(row)
        require(reference(row) == row['admission'] and [p['role'] for p in row['plans']] == list(ROLES)
                and all(not p['trace_progress'] and 'native_profile' not in p and p['read'] == READ
                        and p['variant'] == 'all' and p['cpu_wait_scopes'] == (i == 1)
                        for i, p in enumerate(row['plans'])), 'frozen CPU wait comparison differs')
        return providers

    def request_route(log, plan):
        route = stage_timing.request_route(log, plan)
        if plan['cpu_wait_scopes']:
            route['cpu_partition'] = cpu_partition(log)
        else:
            require(all(scope not in log for scope in SCOPES), 'CPU scopes appeared in a clean control')
        return route

    def validate_model_text(log, plan):
        count = RUNS if plan['cpu_wait_scopes'] else 0
        require(all(sum(s.startswith(scope) for s in log.splitlines()) == count for scope in SCOPES)
                and 'UNITRACE Profiling Result' not in log, 'unexpected CPU scope count/collector output')

    def decision(processes):
        result = ctx.profile_decision(processes)
        result.update(limits='Compare first use and warmed requests against the corresponding positions in '
                             'both clean controls. The CPU scopes account for calling-thread intervals only. '
                             'Native waits without an instrumented wrapper can be outside known waits. '
                             'Physical resource saturation and critical-path latency are unmeasured.')
        return result

    ctx.basis, ctx.current, ctx.environment = admitted_stage_basis, current, environment
    ctx.request_route, ctx.check_sample = request_route, stage_timing.check_sample
    ctx.validate_model_text, ctx.after_execution, ctx.decision = validate_model_text, compare_work, decision
    return ctx


def prepare(ctx):
    ctx.prepare()
    row = read(ROOT / 'record.json')
    fixed = []
    for i, plan in enumerate(row['plans']):
        plan.update(role=ROLES[i], variant='all', bank=512, window=10, relocate=0,
                    trace_progress=False, cpu_wait_scopes=i == 1, upload='jit')
        env = ctx.environment(row, plan)
        require(basis.comparison.fingerprint(env) == plan['environment_sha256'],
                'prepared environment changed')
        fixed.append({k: v.replace(plan['root'], '<case>') for k, v in env.items() if k != SELECTOR})
    require(all(env == fixed[0] for env in fixed), 'another variable changed across arms')
    row.update(admission=reference(row), plan=dict(
        question='How much of the prefill calling thread is spent in known waits, grouping and submission?',
        fixed='Same validated ELF, native1 host-USM prefix issue, deferred first reader, all-at-once staging, '
              '512 slots, window10, BF16, 32769 actual input tokens, 24 outputs with thinking, immutable decode.',
        comparison='Three independent processes with two exact requests each. Only CPU wait scopes switch on '
                   'in the middle process. Match actual routes, selected experts, payload and MTP work.',
        gate='Require complete accounting and normal owned exits with no faults. Require prompt and client '
             'TTFT ratios at most 1.20 against both clean controls at both request positions.',
        stopping='Stop first source, owner, route, accounting, driver or budget failure. No retry or pin refresh.',
        limits='CPU scope diagnostics only. Nested status time is included in the enclosing CPU partition.',
        retention='Retain compact samples and disjoint calling-thread results. Retire raw captures at closure.'))
    ctx.save(row)


def compare_work(outcome, row, plan):
    def work(process):
        keys = ('uploads', 'requested_bank_H2D_bytes', 'requested_source_map_H2D_bytes',
                'actual_GU_dispatch', 'copy_transport', 'stage_order')
        return [({k: route[k] for k in keys}, route['prefix_upload'], sample['drafts_offered'],
                 sample['drafts_accepted']) for route, sample in zip(process['routes'], process['samples'], strict=True)]

    require(all(work(outcome) == work(p) for p in row['processes']),
            'matched selected experts, copies, dispatch or MTP acceptance differs')
    outcome.update(role=plan['role'], clean_speed=not plan['cpu_wait_scopes'])



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
            basis.comparison.run(int(action), ctx)
