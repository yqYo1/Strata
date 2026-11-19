#!/usr/bin/env python3
"""Compare the order of host staging and native copies; repository LICENSE."""

import fcntl
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import prefill_prefix_expert_net as net

schedule = net.schedule
build, basis, builder = schedule.build, schedule.basis, schedule.builder
B, RESULT = build.B, build.RESULT
require, signature = build.require, build.signature
THIS = 'tools/sycl/prefill_prefix_expert_host_stage.py'
SMOKE = 'prefill_prefix_expert_host_stage_smoke_v1'
NUMERICAL = 'prefill_prefix_expert_host_stage_numerical_v1'
SELECTOR = 'STRATA_SYCL_PREFILL_PREFIX_EXPERT_STAGE_ORDER'
ARMS = {
    'numerical': ('all', 'per-span'),
    'timing': ('all', 'per-span', 'per-span', 'all', 'all', 'per-span'),
}


def bank_smoke():
    schedule.bank_smoke(controller=THIS, key=SMOKE, arg='--fullfit-host-stage-order-smoke',
        marker='prefill_fullfit_host_stage_order_fixture PASS waits_retained=1',
        expected=[(512, 256, 2, 0, 4), (4, 512, 2, 0, 4), (4, 512, 2, 2, 4)] * 2)


def model_context(cohort):
    order = ARMS[cohort]
    traced = cohort == 'numerical'
    read, runs = (32769, 2) if traced else (65537, 3)
    reference_settings = schedule.model_context(cohort)
    spec = importlib.util.spec_from_file_location('prefix_expert_host_stage', basis.core.__file__)
    ctx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ctx)
    ctx.ROOT = B / f'prefill-prefix-expert-host-stage-{cohort}-v1'
    ctx.KEY, ctx.THIS = f'prefill_prefix_expert_host_stage_{cohort}_v1', THIS
    ctx.READS, ctx.ORDER, ctx.RUNS, ctx.CLEAN_FAST = (read,) * len(order), [(read, 1)] * len(order), runs, True
    base_current = ctx.current

    def reference():
        value, _ = schedule.native()
        smoke = basis.validation.committed(RESULT, (SMOKE,))
        require(smoke['passed'] and smoke['ownership_closed'] and smoke['native_exit_code'] == 0
                and smoke['native_build'] == signature(builder.compact(value)), 'same-build stage-order fixture required')
        gates = {SMOKE: signature(smoke)}
        if not traced:
            gate = basis.validation.committed(RESULT, (NUMERICAL,))
            require(gate['stage'] == 'closed' and gate['passed'] and gate['complete'] and gate['ownership_closed']
                    and not gate['active'] and gate['decision']['matched_work']
                    and gate['binary'] == value['binaries']['strata'], 'matched numerical stage-order cohort required')
            gates['stage_order_numerical'] = signature(gate)
        _, _, speed = basis.basis()
        old = json.loads((basis.reference.model.ROOT / 'record.json').read_bytes())
        require(old['models'] == speed['model_assets'] and old['probe'] == speed['native_validation_probe'],
                'exact coherent model assets and probe required')
        old['observer_admission'] = dict(gates=gates, native_build=signature(builder.compact(value)),
            fast_configuration_diagnostic=speed['observer_admission']['fast_configuration_diagnostic'])
        return dict(binaries=value['binaries'], native_source_commit=value['source_commit']), old, \
            json.loads(Path(speed['runtime_manifest']['path']).read_bytes()), None

    def environment(row, plan):
        env = reference_settings.environment(row, plan)
        env.update({schedule.ISSUE: 'prefix', schedule.DEFER: '1', SELECTOR: order[plan['index']]})
        return env

    def current(row):
        require([p['variant'] for p in row['plans']] == list(order)
                and all((p['bank'], p['window'], p['relocate'], p['trace_progress'], p['upload'], p['read'])
                    == (512, 10, 0, traced, 'jit', read) for p in row['plans']), 'frozen stage-order comparison differs')
        return base_current(row)

    def route(log, plan):
        route = schedule.request_route(log, dict(plan, variant='prefix-deferred'))
        require(net.workspace.one(log, 'strata MoE prefix upload stage order: ') ==
                dict(effective=plan['variant']), 'actual host staging order differs')
        route['stage_order'] = plan['variant']
        return route

    ctx.basis, ctx.environment, ctx.current = reference, environment, current
    ctx.request_route, ctx.check_sample = route, reference_settings.check_sample
    ctx.after_execution = net.compare_work
    ctx.decision = lambda processes: net.decision(processes, order, traced)
    return ctx


def prepare(ctx, cohort):
    ctx.prepare()
    row = json.loads((ctx.ROOT / 'record.json').read_bytes())
    shared = []
    for plan, mode in zip(row['plans'], ARMS[cohort], strict=True):
        plan.update(variant=mode, bank=512, window=10, relocate=0, trace_progress=cohort == 'numerical', upload='jit')
        env = ctx.environment(row, plan)
        require(plan['environment_sha256'] == basis.comparison.fingerprint(env), 'prepared environment differs')
        shared.append({k: v.replace(plan['root'], '<case>') for k, v in env.items() if k != SELECTOR})
    require(all(env == shared[0] for env in shared), 'another environment variable changed between arms')
    row['plan'] = dict(cohort=cohort, modes=ARMS[cohort], actual_input=ctx.READS[0], requests=ctx.RUNS,
        question='Compare staging all selected regions before DMA with preparing each region before its native append.',
        selection=SELECTOR, fixed='Native ordinal 1, early issue, first-reader drain, allocation, selected IDs, '
                                   'copy count/bytes, private slots, arithmetic and decode setup.',
        gates='Require actual stage-order, exact logical work, selected IDs, requested copies and MTP acceptance. '
              'First use is separate. Test three counterbalanced warmed pairs; advancement requires all three '
              'time ratios below one. Keep the 1000 tok/s milestone separate. Close on the first validation, '
              'ownership, logical-work or fault failure. No retiming or observers in the timing arm.',
        limits='Net change of staging order only. Wall-time gain alone does not qualify CPU/DMA overlap, '
               'physical PCIe saturation or sustained decode. Default all mode remains available.',
        retention='Keep compact qualified timed samples, logical work and fault/owner evidence.')
    ctx.save(row)


if __name__ == '__main__':
    require(sys.executable == basis.reference.model.PYTHON, 'use pinned Python')
    cohort, command = sys.argv[1:]
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        builder.guards.no_model_owners([])
        build.ROOT, build.KEY = build.STAGE_ROOT, build.STAGE_BUILD
        if cohort == 'build':
            require(command in ('prepare', 'configure', 'compile', 'host'), 'unexpected stage-order build command')
            build.build_step(command)
        elif (cohort, command) == ('fixture', 'smoke'):
            bank_smoke()
        else:
            ctx = model_context(cohort)
            if command == 'prepare':
                prepare(ctx, cohort)
            elif command == 'host':
                basis.comparison.host(ctx)
            else:
                basis.comparison.run(int(command), ctx)
