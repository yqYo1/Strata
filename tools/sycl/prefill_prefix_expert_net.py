#!/usr/bin/env python3
"""Compare native copy scheduling with the arena FFN path, repository LICENSE."""

import fcntl
import importlib.util
import json
import re
import statistics
import sys

sys.dont_write_bytecode = True
import prefill_prefix_expert_measure as schedule

basis, builder = schedule.basis, schedule.builder
B, RESULT = schedule.B, schedule.RESULT
require, signature = schedule.require, schedule.signature
workspace = schedule.workspace
THIS = 'tools/sycl/prefill_prefix_expert_net.py'
SCHEDULE_TIMING = 'prefill_prefix_expert_schedule_timing_v1'
NUMERICAL = 'prefill_prefix_expert_net_numerical_v1'
REFERENCE, CANDIDATE = 'arena-ffn', 'prefix-deferred'
ORDER = {
    'numerical': (REFERENCE, CANDIDATE),
    'timing': (REFERENCE, CANDIDATE, CANDIDATE, REFERENCE, REFERENCE, CANDIDATE),
}
SOURCE = 'STRATA_SYCL_PREFILL_PREFIX_EXPERT_HOST_USM'
TRANSPORT = 'STRATA_SYCL_PREFILL_PREFIX_EXPERT_COPY'
SELECTORS = (SOURCE, TRANSPORT, schedule.ISSUE, schedule.DEFER)
ARENA = ('0', 'sycl', 'ffn', '0')
NATIVE = ('1', 'native1', 'prefix', '1')


def arena_route(log, plan):
    route = workspace.request_route(log, plan)
    require(workspace.one(log, 'strata MoE prefix router: ') ==
            dict(requested='early', effective='early', workspace_bytes=str(schedule.PREFIX_BYTES),
                 decode_changed='0')
            and workspace.one(log, 'strata MoE prefix router complete: ') ==
            dict(effective='early', layers='48', rows='8192', final_drain='checked'),
            'fixed prefix router not exercised')
    require(workspace.one(log, 'strata MoE prefix expert upload: ') ==
            dict(requested='ffn', effective='ffn', queue='compute')
            and workspace.one(log, 'strata MoE prefix upload storage: ') ==
            dict(host_usm_requested='0', host_usm_effective='0', allocated_bytes='0')
            and workspace.one(log, 'strata MoE prefix upload storage complete: ') ==
            dict(staged_bytes='0', queue='compute')
            and not workspace.records(log, 'strata MoE prefix upload transport: '),
            'arena source or same-queue route changed')
    layers = workspace.records(log, 'strata MoE prefix expert upload layer: ')
    require(len(layers) == 48 and route['permanent_hits'] == 0, 'expected one upload per demanded expert')
    experts = sum(int(layer['experts']) for layer in layers)
    payload = sum(int(layer['requested_H2D_bytes']) for layer in layers)
    require(workspace.one(log, 'strata MoE prefix expert upload complete: ') ==
            dict(effective='ffn', layers='48', experts=str(experts),
                 requested_H2D_bytes=str(payload), final_drain='checked')
            and 0 < experts <= route['uploads'] and 0 < payload <= route['requested_bank_H2D_bytes'],
            'arena copy receipt or checked retirement differs')
    # Each of the 48 selected layer receipts must subsequently match the
    # candidate parser's validated ID hash, stride, extent and readers.
    route['prefix_upload'] = dict(experts=experts, bytes=payload, selected=signature(layers))
    route['copy_transport'] = dict(mode='sycl', allocation=0, staged_bytes=0, issue='ffn', deferred=False)
    if plan['trace_progress']:
        frames = re.findall(r'^strata MoE router census: layer=(\d+) first=(\d+) rows=(\d+) '
            r'ids=([0-9a-f]{16}) prefix_weights=([0-9a-f]{16})$', log, re.M)
        require(len(frames) == 48 and all(tuple(map(int, row[:3])) == (i, 0, plan['read'] - 1)
                    for i, row in enumerate(frames)), 'exact router comparison evidence absent')
        route['route_digest'] = signature(frames)
    return route


def compare_work(outcome, row, _plan):
    def draft_counts(process):
        return [(s['drafts_offered'], s['drafts_accepted']) for s in process['samples']]

    for previous in row['processes']:
        require(schedule.work(outcome) == schedule.work(previous)
                and schedule.selected(outcome) == schedule.selected(previous)
                and draft_counts(outcome) == draft_counts(previous),
                'matched logical work, selected copies, routing or MTP handoff differs')


def decision(processes, order, traced):
    outcome = dict(matched_work=True, independent_processes=len(processes),
                   measured_modes=list(order), overlap_measured=False, model_adopted=False)
    if not traced:
        pairs = []
        for a in range(0, len(order), 2):
            times = {order[i]: schedule.timing(processes[i]) for i in (a, a + 1)}
            ratios = {metric: times[CANDIDATE][metric] / times[REFERENCE][metric]
                      for metric in times[REFERENCE]}
            pairs.append(dict(indices=[a, a + 1], times=times, ratios=ratios))
        warm_ratios = [p['ratios']['warm_prompt_ms'] for p in pairs]
        outcome.update(pairs=pairs, all_warmed_pairs_faster=all(r < 1 for r in warm_ratios),
                       warmed_prompt_time_geomean_ratio=statistics.geometric_mean(warm_ratios))
    return outcome


def model_context(cohort):
    order = ORDER[cohort]
    traced = cohort == 'numerical'
    read, runs = (32769, 2) if traced else (65537, 3)
    same_build = schedule.model_context('numerical' if traced else 'timing')
    spec = importlib.util.spec_from_file_location('prefix_expert_net', basis.core.__file__)
    ctx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ctx)
    ctx.ROOT = B / f'prefill-prefix-expert-net-{cohort}-v1'
    ctx.KEY, ctx.THIS = f'prefill_prefix_expert_net_{cohort}_v1', THIS
    ctx.READS, ctx.ORDER, ctx.RUNS, ctx.CLEAN_FAST = (read,) * len(order), [(read, 1)] * len(order), runs, True
    base_current = ctx.current

    def reference():
        native, prior, providers, extra = same_build.basis()
        image = native['binaries']['strata']
        timing = basis.validation.committed(RESULT, (SCHEDULE_TIMING,))
        require(timing['stage'] == 'closed' and timing['complete'] and timing['ownership_closed']
                and not timing['active'] and timing['binary'] == image
                and timing['decision']['comparisons']['combined_schedule']['all_warmed_pairs_faster'],
                'qualified same-ELF scheduling control required')
        gates = prior['observer_admission']['gates']
        gates['four_way_schedule'] = signature(timing)
        if not traced:
            numerical = basis.validation.committed(RESULT, (NUMERICAL,))
            require(numerical['stage'] == 'closed' and numerical['complete'] and numerical['ownership_closed']
                    and not numerical['active'] and numerical['binary'] == image
                    and numerical['decision']['matched_work'],
                    'matched same-ELF arena and native numerical processes required')
            gates['net_numerical'] = signature(numerical)
        return native, prior, providers, extra

    def environment(row, plan):
        env = same_build.environment(row, plan)
        values = ARENA if order[plan['index']] == REFERENCE else NATIVE
        env.update(zip(SELECTORS, values, strict=True))
        return env

    def current(row):
        require([p['variant'] for p in row['plans']] == list(order)
                and all((p['bank'], p['window'], p['relocate'], p['trace_progress'], p['upload'], p['read'])
                    == (512, 10, 0, traced, 'jit', read) for p in row['plans']), 'frozen net comparison differs')
        return base_current(row)

    def route(log, plan):
        if plan['variant'] == REFERENCE:
            return arena_route(log, plan)
        require(plan['variant'] == CANDIDATE, 'unexpected candidate')
        return schedule.request_route(log, plan)

    ctx.basis, ctx.environment, ctx.current = reference, environment, current
    ctx.request_route, ctx.check_sample = route, same_build.check_sample
    ctx.after_execution = compare_work
    ctx.decision = lambda processes: decision(processes, order, traced)
    return ctx


def prepare(ctx, cohort):
    ctx.prepare()
    row = json.loads((ctx.ROOT / 'record.json').read_text())
    fixed = []
    for plan, mode in zip(row['plans'], ORDER[cohort], strict=True):
        plan.update(variant=mode, bank=512, window=10, relocate=0, trace_progress=cohort == 'numerical', upload='jit')
        env = ctx.environment(row, plan)
        require(plan['environment_sha256'] == basis.comparison.fingerprint(env), 'prepared environment differs')
        fixed.append({k: v.replace(plan['root'], '<case>') for k, v in env.items() if k not in SELECTORS})
    require(all(env == fixed[0] for env in fixed), 'another environment variable changed between arms')
    row['plan'] = dict(cohort=cohort, modes=ORDER[cohort], actual_input=ctx.READS[0], requests=ctx.RUNS,
        question='Charge host staging, native copy ownership and scheduling together against the arena FFN route.',
        selectors={mode: dict(zip(SELECTORS, values, strict=True))
                   for mode, values in ((REFERENCE, ARENA), (CANDIDATE, NATIVE))},
        gates='Require same executable, actual selected expert IDs, bytes, request bodies, routing work and MTP handoff. '
              'Keep first use separate. Require all three warmed pairs to favor the candidate for advancement. '
              'Stop on validation, ownership, mismatch or fresh GPU fault. No retiming or observer in the clean arm.',
        limits='Net prefill timing only. The separate source, transport and four-way scheduling controls remain '
               'independent evidence. This does not measure device overlap, sustained TG or API/model qualification.',
        retention='Keep exact compact samples and route evidence. Review this extra executable at the next consumer boundary.')
    ctx.save(row)


if __name__ == '__main__':
    require(sys.executable == basis.reference.model.PYTHON, 'use pinned Python')
    cohort, command = sys.argv[1:]
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        builder.guards.no_model_owners([])
        ctx = model_context(cohort)
        if command == 'prepare':
            prepare(ctx, cohort)
        elif command == 'host':
            basis.comparison.host(ctx)
        else:
            basis.comparison.run(int(command), ctx)
