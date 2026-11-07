#!/usr/bin/env python3
"""Source-prepared32/96 comparison on the unchanged payload successor.

Main-authored; repository LICENSE. Reuse owned benchmark execution and its
independent phase decisions. Actual96positive/refusal qualification is mandatory
before preparation or execution; source preparation establishes no speed claim.
"""
import copy
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import check_adaptive_payload96_entry as entry96
import compare_adaptive_payload_generation as previous

clean = previous.clean
require = previous.require
SELF = Path(__file__).resolve()
ARMS = (('reference', 32), ('candidate', 96), ('candidate', 96),
        ('reference', 32), ('reference', 32), ('candidate', 96))


def prerequisites(full=False):
    ctx = entry96.refusal_context()
    entry96.prerequisite_rows(ctx.positive)
    data = clean.base.committed(clean.RESULT)
    positive = ctx.normal_entry()
    failed = json.loads((ctx.ROOT / 'record.json').read_text())
    require(data[entry96.KEY] == ctx.positive.compact(positive) and
        data[entry96.REFUSAL_KEY] == ctx.compact(failed) and
        failed['stage'] == 'closed' and failed['passed'] and failed['complete'] and
        failed['ownership_closed'] and not failed['active'] and
        failed['outcome']['native_exit_code'] == 1 and
        failed['outcome']['adaptive']['first_swaps'] == 96 and
        failed['outcome']['adaptive']['queued_second_swaps'] == 96 and
        failed['outcome']['adaptive']['stale_event_not_published'],
        'closed actual96positive and exact96second-record refusal required')
    ctx.positive.current(positive, full=full)
    ctx.current(failed, full=full)
    old, prior, contract, manifest, providers, refs = previous.prerequisites(full=full)
    require(positive['binary'] == failed['binary'] == old['binary'] and
        positive['native_source_commit'] == failed['native_source_commit'] == old['native_source_commit'] and
        positive['model_assets'] == failed['model_assets'] == old['model_assets'],
        'reuse the unchanged qualified successor and model assets')
    refs.update({name: dict(path=str(root / 'record.json'),
        sha256=clean.digest(root / 'record.json')) for name, root in
        (('positive96', ctx.positive.ROOT), ('record_failure96', ctx.ROOT))})
    return positive, prior, contract, manifest, providers, refs


def decision(rows, printed):
    result = previous.original.exact_decision(rows, printed)
    result['limits'] = ('Same-ELF adaptive32/96 focused comparison only. Exact TG is primary;'
        'printed decision retained. Judge prompt/TTFT independently. No other length,'
        'quality,agent,physical262144/RESTORE,ordinary teardown,adoption or70TG qualification.')
    return result


def context(workload):
    module = previous.template()
    module.ARMS, module.SELF, module.prerequisites = ARMS, SELF, prerequisites
    module.controls, module.prepare, module.exact_decision = controls, prepare, decision
    ctx = module.context(workload)
    ctx.ROOT = clean.B / ('adaptive-payload96-generation-' + workload + '-v1')
    ctx.KEY = 'adaptive_payload96_generation_' + workload.replace('-', '_') + '_v1'
    ctx.WORKLOAD_ID = 'adaptive_payload96_generation_20261107_' + workload
    prior_current = ctx.current

    def current(row):
        require(row['successor_positive_key'] == entry96.KEY and
            row['successor_refusal_key'] == entry96.REFUSAL_KEY,
            'exact96qualification keys changed')
        return prior_current(row)

    ctx.current = current
    return ctx


def controls():
    rows = [dict(kernel=arm, adapt_swaps=swaps, passed=True, complete=True,
        ownership_closed=True, samples=[dict(decode_tps=10.0, prompt_ms=1000.0,
            ttft_s=1.1, acceptance_rate=.7, generated_tokens=2048,
            decode_ms=204780.0 if arm == 'candidate' else 204800.0)])
        for arm, swaps in ARMS]
    ctx = context('regular-code')
    result = ctx.decision(rows)
    require(result['repeated_TG_gain'] and not result['printed_TG_decision']['repeated_TG_gain'],
        'retain a repeated exact gain hidden by rounding')
    slower = copy.deepcopy(rows)
    slower[1]['samples'][0]['decode_ms'] = 204801.0
    require(not ctx.decision(slower)['repeated_TG_gain'], 'retain a slower pair')
    phases = copy.deepcopy(rows)
    for row in phases:
        if row['kernel'] == 'candidate':
            row['samples'][0].update(prompt_ms=1010.0, ttft_s=1.2)
    result = ctx.decision(phases)
    require(result['repeated_TG_gain'] and not result['prompt_nonregression'] and
        not result['TTFT_nonregression'], 'TG gain cannot conceal a phase regression')
    prior = json.loads((clean.positive.width.SPEED_ROOT / 'record.json').read_text())
    env = clean.positive.width.generation_context(0).environment(prior, prior['plans'][0])
    for workload in clean.CAPS:
        ctx = context(workload)
        row = dict(configuration=copy.deepcopy(prior['configuration']),
            environment_template=dict(env), workload=workload)
        plans = [dict(index=i, kernel=arm, adapt_swaps=swaps, heads=1,
            root=str(ctx.ROOT / ('control-' + str(i)))) for i, (arm, swaps) in enumerate(ARMS[:2])]
        require(ctx.invariant(row, plans[0]) == ctx.invariant(row, plans[1]),
            'only budget and private paths may differ')
        for plan in plans:
            args = ctx.configuration(row, plan)['args']
            require(args[args.index('--adapt-swaps') + 1] == str(plan['adapt_swaps']) and
                args[args.index('--adapt-every') + 1] == '8' and
                args[args.index('--mtp-max-t') + 1] == str(clean.CAPS[workload]),
                'actual32/96budget,cadence and width selectors required')
    return dict(passed=True, small_exact_gain_retained=True, slower_pair_retained=True,
        phase_decisions_separate=True, immutable_configuration_controls=4, GPU_executed=False)


def prepare(module, ctx):
    module.prepare(ctx)
    row = json.loads((ctx.ROOT / 'record.json').read_text())
    row.update(successor_positive_key=entry96.KEY, successor_refusal_key=entry96.REFUSAL_KEY)
    row['plan'] = ('Six fresh owned processes adaptive32/96,96/32,32/96 on unchanged e1b011;'
        'every8verifier rounds,default decay0.7. Close both8/32cohorts and qualify actual96'
        'positive publication/dispatch/loan plus exact96second-record refusal before preparation. '
        'Four official warmups then one first long v5 thinking-on32K+/1K+generated request '
        'with an answer;code cap4/prose cap2 in separate cohorts,16K output allowance. '
        'Keep the existing source/body/token/route/head/clock/journal/validation/ownership/'
        'no-fault gates,all15asset hashes,placement,precision and runtime safety. '
        'Three exact paired TG ratios>1 establish focused repeatability without a gain floor;'
        'retain the printed decision and independently judge prompt/TTFT. Keep every sample. '
        'Stop first safety,ownership,eligibility or budget failure;no retries,exclusions or '
        'hash refresh. Commit each closed process before the next. No new engine generation,'
        'fixed4weak-MTP classification,physical-traffic claim or inference adoption.')
    ctx.save(row)


if __name__ == '__main__':
    args = sys.argv[1:]
    require(args == ['controls'] or len(args) == 2 and args[0] in clean.CAPS and
        (args[1] in ('prepare', 'host') or args[1].isdigit() and int(args[1]) < 6),
        'use controls | regular-code|varied-prose prepare|host|INDEX0..5')
    with (clean.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        clean.positive.fcntl.flock(lock, clean.positive.fcntl.LOCK_EX | clean.positive.fcntl.LOCK_NB)
        clean.base.guards.no_model_owners([])
        if args == ['controls']:
            revision = clean.base.basis.build.git(clean.P, 'rev-parse', 'HEAD').decode().strip()
            pins = clean.base.controls(revision, controller=SELF)
            print(json.dumps(dict(controls(), source_commit=revision,
                source_count=len(pins), source_signature_sha256=clean.signature(pins)), sort_keys=True))
        else:
            ctx = context(args[0])
            if args[1] == 'prepare': ctx.prepare_budget()
            elif args[1] == 'host': ctx.comparison.host(ctx)
            else: ctx.run(int(args[1]))
