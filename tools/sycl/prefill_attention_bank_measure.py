#!/usr/bin/env python3
"""Compare attention store destinations through the actual long-prefill entrypoint."""

import fcntl
import importlib.util
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
import prefill_attention_bank_build as build
import prefill_prefix_expert_host_stage as stage

P, B, RESULT, basis = build.P, build.B, build.RESULT, build.basis
require, signature = build.require, build.signature
THIS = 'tools/sycl/prefill_attention_bank_measure.py'
SELECTOR = 'STRATA_SYCL_PREFILL_DIRECT_ATTENTION_STORE'
SMOKE = 'prefill_attention_bank_native_smoke_v1'
NUMERICAL = 'prefill_attention_bank_numerical_v1'
ORDER = {'numerical': ('copy', 'direct'),
         'timing': ('copy', 'direct', 'direct', 'copy', 'copy', 'direct')}


def check_store(log, plan):
    fields = stage.net.workspace.one(log, 'strata prefill attention bank: ')
    rows = plan['read'] - 1
    require(rows > 8192 and rows % 8192 == 0, 'this comparison uses complete 8192-row pieces')
    pieces = 48 * (rows // 8192)
    direct = plan['variant'] == 'direct'
    expected = dict(requested=str(int(direct)), pieces=str(pieces),
                    direct=str(pieces if direct else 0),
                    requested_D2D_bytes=str(0 if direct else 48 * rows * 10240 * 2))
    require(fields == expected, 'attention destination or actual requested D2D volume differs')
    return {k: int(v) for k, v in fields.items()}


def context(cohort):
    order = ORDER[cohort]
    traced = cohort == 'numerical'
    read, runs = (32769, 2) if traced else (65537, 3)
    stage_all = stage.model_context(cohort)
    loader = importlib.util.spec_from_file_location('prefill_attention_bank_cohort', basis.core.__file__)
    ctx = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(ctx)
    ctx.ROOT = B / f'prefill-attention-bank-{cohort}-v1'
    ctx.KEY, ctx.THIS = f'prefill_attention_bank_{cohort}_v1', THIS
    ctx.READS, ctx.ORDER, ctx.RUNS, ctx.CLEAN_FAST = (read,) * len(order), [(read, 1)] * len(order), runs, True
    base_current = ctx.current

    def reference():
        value, _ = build.native()
        image = value['binaries']['strata']
        smoke = basis.validation.committed(RESULT, (SMOKE,))
        require(smoke['passed'] and smoke['ownership_closed'] and smoke['native_exit_code'] == 0
                and smoke['native_build'] == signature(build.builder.compact(value)),
                'same-build bank smoke required')
        gates = {SMOKE: signature(smoke)}
        if not traced:
            gate = basis.validation.committed(RESULT, (NUMERICAL,))
            require(gate['stage'] == 'closed' and gate['passed'] and gate['complete']
                    and gate['ownership_closed'] and not gate['active'] and gate['decision']['matched_work']
                    and gate['binary'] == image, 'matched same-ELF attention numerical cohort required')
            gates[NUMERICAL] = signature(gate)
        _, _, speed = basis.basis()
        old = build.read(basis.reference.model.ROOT / 'record.json')
        require(old['models'] == speed['model_assets'] and old['probe'] == speed['native_validation_probe'],
                'exact coherent model assets and probe required')
        old['observer_admission'] = dict(gates=gates, native_build=signature(build.builder.compact(value)),
            fast_configuration_diagnostic=speed['observer_admission']['fast_configuration_diagnostic'])
        return dict(binaries=value['binaries'], native_source_commit=value['source_commit']), old, \
            build.read(Path(speed['runtime_manifest']['path'])), None

    def environment(row, plan):
        env = stage_all.environment(row, plan)
        env.update({stage.SELECTOR: 'all', SELECTOR: str(int(order[plan['index']] == 'direct'))})
        return env

    def current(row):
        require([p['variant'] for p in row['plans']] == list(order)
                and all((p['bank'], p['window'], p['relocate'], p['trace_progress'], p['upload'], p['read'])
                        == (512, 10, 0, traced, 'jit', read) for p in row['plans']),
                'exact attention-store comparison plan differs')
        return base_current(row)

    def route(log, plan):
        value = stage_all.request_route(log, dict(plan, variant='all'))
        value['attention_store'] = check_store(log, plan)
        return value

    def decision(processes):
        value = stage.net.decision(processes, order, traced)
        value['same_ELF'] = True
        if not traced:
            per_process = [statistics.median(s['actual_read_tokens'] * 1000 / s['prompt_ms']
                                            for s in p['samples'][1:]) for p in processes]
            value['warmed_pp_tok_s'] = {mode: [pp for pp, actual in zip(per_process, order, strict=True)
                                             if mode == actual] for mode in ('copy', 'direct')}
            value['all_direct_warmed_at_least_1000'] = all(pp >= 1000 for pp in value['warmed_pp_tok_s']['direct'])
        value['sustained_decode_qualified'] = value['full_context_qualified'] = False
        return value

    ctx.basis, ctx.environment, ctx.current = reference, environment, current
    ctx.request_route, ctx.check_sample = route, stage_all.check_sample
    ctx.after_execution, ctx.decision = stage.net.compare_work, decision
    return ctx


def prepare(ctx, cohort):
    ctx.prepare()
    row = build.read(ctx.ROOT / 'record.json')
    shared = []
    for plan, mode in zip(row['plans'], ORDER[cohort], strict=True):
        plan.update(variant=mode, bank=512, window=10, relocate=0,
                    trace_progress=cohort == 'numerical', upload='jit')
        env = ctx.environment(row, plan)
        require(plan['environment_sha256'] == basis.comparison.fingerprint(env), 'prepared environment differs')
        shared.append({k: v.replace(plan['root'], '<case>') for k, v in env.items() if k != SELECTOR})
    require(all(env == shared[0] for env in shared), 'the store selector must be the only runtime difference')
    row['plan'] = dict(cohort=cohort, modes=ORDER[cohort], actual_input=ctx.READS[0], requests=ctx.RUNS,
        question='Use the same ELF to compare copying the rounded attention residual with a direct BF16 store.',
        fixed='Stage all selected expert spans. Native copy ordinal 1, 512 slots, 8192 rows per GPU piece, '
              'same arithmetic, drain, KV and decode layout, thinking enabled and 24 emitted tokens.',
        gates='Require actual copy/direct piece counts and D2D receipt, exact prior route, checked retirement, '
              'matched selected experts, workload and MTP counts. Numerical cohort additionally compares router '
              'hashes. Keep first-use separate. For 64K advancement require the direct warm median to be faster '
              'in all three matched pairs; record TTFT and 1000 tok/s separately.',
        stopping='Stop on first fault, route, validation, equality, process, capability or budget failure. '
                 'No replacement of slower or ineligible speed samples.',
        limits='Small generation verifies comparable handoff work only. Full physical 262144 including RESTORE, '
               'agent, thinking-enabled long generation, and sustained decode qualification remain separate.',
        retention='Keep samples, route, owner and fault evidence. Retire raw output after closure.')
    ctx.save(row)


if __name__ == '__main__':
    require(sys.executable == basis.reference.model.PYTHON, 'use pinned Python')
    cohort, command = sys.argv[1:]
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        build.builder.guards.no_model_owners([])
        if (cohort, command) == ('fixture', 'smoke'):
            stage.schedule.bank_smoke(controller=THIS, key=SMOKE, arg='--fullfit-host-stage-order-smoke',
                marker='prefill_fullfit_host_stage_order_fixture PASS waits_retained=1',
                expected=[(512, 256, 2, 0, 4), (4, 512, 2, 0, 4), (4, 512, 2, 2, 4)] * 2,
                build_root=build.ROOT, native_reader=build.native)
        else:
            ctx = context(cohort)
            if command == 'prepare':
                prepare(ctx, cohort)
            elif command == 'host':
                basis.comparison.host(ctx)
            else:
                basis.comparison.run(int(command), ctx)
