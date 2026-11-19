#!/usr/bin/env python3
"""Replay the unfenced 64K baseline after its mismatched pair."""

import fcntl
import json
import sys

sys.dont_write_bytecode = True
import prefill_prefix_expert_measure as measure


COHORT = 'unfenced-repeat'
FENCE = 'prefill_prefix_expert_transport_fence_v1'
HISTORY = 'prefill_prefix_expert_host_usm_timing_v1'
THIS = 'tools/sycl/prefill_prefix_expert_repeat.py'


def experiment():
    failure = measure.basis.validation.committed(measure.RESULT, (FENCE,))
    history = measure.basis.validation.committed(measure.RESULT, (HISTORY,))
    require = measure.require
    require(failure['stage'] == 'failed' and failure['ownership_closed']
            and not failure['active'] and len(failure['processes']) == 2,
            'closed original timing failure required')
    unfenced, fenced = failure['processes']
    require(unfenced['passed'] and not fenced['passed']
            and fenced['error'] == "RuntimeError('logical work differs between matched requests')"
            and all(p['native_exit_code'] == 0 and p['ownership_closed'] and not p['fresh_faults']
                    for p in (unfenced, fenced))
            and measure.selected(unfenced) == measure.selected(fenced),
            'expected later expert-work mismatch with matching prefix uploads')
    require(history['stage'] == 'closed' and history['request_hashes'] == failure['request_hashes']
            and all(measure.work(p) == measure.work(fenced) for p in history['processes']),
            'historical work reference differs')

    measure.COHORTS[COHORT] = ('sycl',)
    ctx = measure.model_context(COHORT)
    ctx.THIS = THIS
    base_current = ctx.current
    original = failure['plans'][0]
    plan = dict(cohort=COHORT, actual_input=65537, requests=3,
                question='Does the same unfenced native binary repeat the observed 64K expert-work excursion?',
                acceptance='Replay the exact baseline settings and three request bodies. Classify each '
                'request against the original unfenced and historical fenced work signatures. '
                'Stop on any ownership, safety, route, payload or request mismatch.',
                limits='One independent replay. Preserve the failed timing pair and make no speed or '
                'adoption claim. A matching reference does not rule out an intermittent excursion.',
                evidence={FENCE: measure.signature(failure), HISTORY: measure.signature(history)})

    def replay_inputs(row):
        replay, = row['plans']
        env = {key: value.replace(replay['root'], original['root'])
               for key, value in ctx.environment(row, replay).items()}
        require(row['binary'] == failure['binary'] and row['configuration'] == failure['configuration']
                and replay['variant'] == original['variant'] == 'sycl'
                and measure.basis.comparison.fingerprint(env) == original['environment_sha256'],
                'unfenced replay binary, configuration or settings differ')

    def current(row):
        require(row['plan'] == plan and row['request_hashes'] == failure['request_hashes'],
                'committed replay contract or request evidence differs')
        replay_inputs(row)
        return base_current(row)

    def after_execution(outcome, row, _plan):
        require(measure.selected(outcome) == measure.selected(unfenced)
                and [sample['request_sha256'] for sample in outcome['samples']]
                == failure['request_hashes']['65537'], 'replayed prefix payload or prompt differs')

    def decision(processes):
        subsequent, = processes
        return dict(speed_eligible=False, model_adopted=False, requests=[
            dict(index=index, matches_unfenced=observed == old, matches_reference=observed == reference)
            for index, (observed, old, reference) in enumerate(zip(
                measure.work(subsequent), measure.work(unfenced), measure.work(fenced), strict=True))])

    ctx.current, ctx.after_execution, ctx.decision = current, after_execution, decision
    return ctx, plan, replay_inputs, failure['request_hashes']


if __name__ == '__main__':
    measure.require(sys.executable == measure.basis.reference.model.PYTHON, 'use pinned Python')
    action, = sys.argv[1:]
    with (measure.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        measure.builder.guards.no_model_owners([])
        ctx, plan, replay_inputs, hashes = experiment()
        if action == 'prepare':
            measure.prepare(ctx, COHORT)
            row = json.loads((ctx.ROOT / 'record.json').read_text())
            replay_inputs(row)
            row.update(plan=plan, request_hashes=hashes)
            ctx.save(row)
        elif action == 'host':
            measure.basis.comparison.host(ctx)
        elif action == 'run':
            measure.basis.comparison.run(0, ctx)
        else:
            raise ValueError('use prepare, host or run')
