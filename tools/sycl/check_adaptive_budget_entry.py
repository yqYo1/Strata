#!/usr/bin/env python3
"""Qualify a larger adaptive budget on the existing owned engine.

Main-authored; repository LICENSE. This is a diagnostic, not a speed claim.
Keep the active code/prose comparison inputs unchanged. No new native build.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import check_adaptive_native_entry as original

SELF = Path(__file__).resolve()
BUDGET = 32
PROFILE = 'data/expert-profile.bin'
require = original.require


def audit(text):
    """Reject incomplete/aliased publication and require both dispatch roles."""
    prefix = 'strata serve: ADAPT_AUDIT '
    patterns = [re.compile(prefix + pattern) for pattern in (
        r'phase=prepared generation=(\d+) incoming=(\d+) outgoing=(\d+) slot=(\d+)',
        r'phase=masked generation=(\d+) swaps=(\d+) identities=3',
        r'phase=published generation=(\d+) swaps=(\d+) identities=3 payload_match=1',
        r'phase=dispatch request=(\d+) published=(\d+) calls=(\d+) incoming_vram_entries=(\d+) outgoing_cpu_entries=(\d+) post_swap_refills=(\d+)',
        r'phase=loan-after-positive completed=(\d+) lent=(\d+) identities=3 payload_match=1')]
    prepared, masked, published, footers, loans = {}, {}, {}, [], []
    for line in text.splitlines():
        if not line.startswith(prefix):
            continue
        matches = [(kind, pattern.fullmatch(line)) for kind, pattern in enumerate(patterns)]
        matches = [(kind, match) for kind, match in matches if match]
        require(len(matches) == 1, 'malformed adaptive budget audit')
        kind, match = matches[0]
        require(all(len(value) <= 20 for value in match.groups()), 'oversized audit integer')
        values = tuple(map(int, match.groups()))
        require(all(value < 2**64 for value in values), 'audit integer bound')
        generation = values[0]
        if kind == 0:
            require(generation == len(published) + 1 and generation not in masked,
                    'prepare generation/order')
            prepared.setdefault(generation, []).append(values[1:])
        elif kind in (1, 2):
            target = masked if kind == 1 else published
            require(generation in prepared and generation not in target and
                    (kind == 1 or generation in masked), 'publication order/identity')
            target[generation] = values[1]
        elif kind == 3:
            require(generation == len(footers) + 1 and values[1] == len(published),
                    'request/footer generation order')
            footers.append(values)
        else:
            require(len(footers) == 1 and generation == len(published) > 0 and values[1] > 0,
                    'occupied loan must follow positive first-request dispatch')
            loans.append(values)
    require(prepared and set(prepared) == set(masked) == set(published) ==
            set(range(1, len(prepared) + 1)), 'unfinished replacement generation')
    for generation, rows in prepared.items():
        require(0 < len(rows) <= BUDGET and len(rows) == masked[generation] == published[generation],
                'replacement count/budget')
        require(all(len({row[column] for row in rows}) == len(rows) for column in range(3)),
                'duplicate incoming/outgoing/slot')
        require(not ({row[0] for row in rows} & {row[1] for row in rows}),
                'incoming/outgoing identity overlap')
        require(all(i != o and i < 24576 and o < 24576 and i // 512 == o // 512 and s < 1256
                    for i, o, s in rows), 'replacement geometry')
    require(len(footers) == 2 and 0 < footers[0][1] <= footers[1][1] == len(prepared) and
            0 < footers[0][2] < footers[1][2] and footers[0][3] > 0 and
            footers[1][3] > footers[0][3] and footers[1][4] > footers[0][4] > 0 and
            footers[0][5] == 0 and footers[1][5] == len(loans) > 0 and
            all(g >= footers[0][1] for g, _ in loans),
            'both replacement dispatch roles and post-positive refill required')
    require(max(map(len, prepared.values())) == BUDGET, 'full32-swap group was not exercised')
    return dict(generations=len(prepared), swaps=sum(map(len, prepared.values())),
        maximum_group=BUDGET, post_positive_loan_groups=len(loans), outgoing_CPU_observed=True,
        all_observed_payload_map_dispatch_checks_passed=True,
        request_coverage=[dict(request=r, published=g, dispatch_calls=c, incoming_GPU_entries=i,
            outgoing_CPU_entries=o, post_swap_refills=l) for r, g, c, i, o, l in footers])


def controls():
    p = 'strata serve: ADAPT_AUDIT '
    prepared = [p + f'phase=prepared generation=1 incoming={100+i} outgoing={i} slot={i}'
                for i in range(BUDGET)]
    good = prepared + [p + f'phase=masked generation=1 swaps={BUDGET} identities=3',
        p + f'phase=published generation=1 swaps={BUDGET} identities=3 payload_match=1',
        p + 'phase=dispatch request=1 published=1 calls=48 incoming_vram_entries=3 outgoing_cpu_entries=1 post_swap_refills=0',
        p + 'phase=loan-after-positive completed=1 lent=1124 identities=3 payload_match=1',
        p + 'phase=dispatch request=2 published=1 calls=96 incoming_vram_entries=8 outgoing_cpu_entries=2 post_swap_refills=1']
    require(audit('\n'.join(good))['swaps'] == BUDGET, 'valid32-group control')
    bad = [good[1:], good[:-1], good + [good[-1]],
        good[:BUDGET] + [good[BUDGET+1], good[BUDGET], *good[BUDGET+2:]],
        [good[-2], *good], [good[0], good[0], *good[2:]],
        [good[0].replace('slot=0', 'slot=1256'), *good[1:]],
        [good[0].replace('incoming=100', 'incoming=24576'), *good[1:]],
        [good[0].replace('incoming=100', 'incoming=512'), *good[1:]],
        [good[0].replace('incoming=100', 'incoming=0'), *good[1:]],
        [good[0].replace('incoming=100', 'incoming=1'), *good[1:]],
        [good[0].replace('generation=1', 'generation=2'), *good[1:]],
        [*good[:BUDGET], good[BUDGET].replace('swaps=32', 'swaps=31'), *good[BUDGET+1:]],
        [*good[:BUDGET+1], good[BUDGET+1].replace('swaps=32', 'swaps=31'), *good[BUDGET+2:]],
        [*good[:BUDGET+2], good[BUDGET+2].replace('outgoing_cpu_entries=1', 'outgoing_cpu_entries=0'), *good[BUDGET+3:]],
        [*good[:BUDGET+2], good[BUDGET+2].replace('calls=48', 'calls=0'), *good[BUDGET+3:]],
        [*good[:-2], good[-2].replace('lent=1124', 'lent=0'), good[-1]],
        [*good[:-1], good[-1].replace('calls=96', 'calls=48')],
        [*good[:-1], good[-1].replace('post_swap_refills=1', 'post_swap_refills=2')],
        [*good[:-1], good[-1] + ' unknown=1'],
        [good[0].replace('generation=1', 'generation=18446744073709551616'), *good[1:]],
        [*prepared, p + 'phase=prepared generation=1 incoming=132 outgoing=32 slot=32',
            *[line.replace('swaps=32', 'swaps=33') for line in good[BUDGET:]]]]
    for lines in bad:
        try:
            audit('\n'.join(lines))
        except RuntimeError:
            continue
        raise RuntimeError('corrupt32-group audit accepted')
    return dict(passed=True, complete_positive_fixture=True, maximum_group=BUDGET,
        negative=len(bad), GPU_executed=False,
        scope='Ordered reducer adversarial controls only;native32 groups unqualified.')


def prerequisite_rows(ctx):
    data = ctx.base.committed(ctx.RESULT)
    rows = [data[key] for key in ('adaptive_generation_regular_code_v2',
            'adaptive_generation_varied_prose_v2')]
    for row in rows:
        require(row['stage'] == 'closed' and row['passed'] and row['complete'] and
            row['ownership_closed'] and not row['active'] and len(row['processes']) == 6 and
            all(p['passed'] and p['complete'] and p['ownership_closed'] and
                p['native_exit_code'] == 0 and not p['fresh_faults'] for p in row['processes']),
            'close both frozen speed cohorts first;no concurrent diagnostic')
    positive = data[original.KEY]
    negative = data['adaptive_record_failure_native_v1']
    require(all(r['passed'] and r['complete'] and r['ownership_closed'] and not r['active']
                for r in (positive, negative)) and
            positive['outcome']['adaptive']['all_observed_payload_map_dispatch_checks_passed'] and
            positive['outcome']['adaptive']['outgoing_CPU_observed'] and
            negative['outcome']['adaptive']['stale_event_not_published'] and
            all(r['binary'] == positive['binary'] for r in (*rows, negative)),
            'same corrected-map positive/checked-record-negative engine required')


def profile_pin(ctx, row):
    raw = ctx.base.basis.build.git(ctx.P, 'show', row['source_commit'] + ':' + PROFILE)
    require((ctx.P / PROFILE).read_bytes() == raw, 'committed initial placement changed')
    args = row['configuration']['args']
    require(args.count('--expert-profile') == 1 and
        args[args.index('--expert-profile')+1] == str(ctx.P / PROFILE), 'initial profile selection')
    return dict(path=str(ctx.P / PROFILE), sha256=hashlib.sha256(raw).hexdigest())


def context():
    spec = importlib.util.spec_from_file_location('strata_budget32_native_entry', original.__file__)
    ctx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ctx)
    ctx.ROOT = ctx.B / 'adaptive-budget32-entry-v1'
    ctx.KEY = 'adaptive_budget32_native_entry_v1'
    ctx.SELF, ctx.audit, ctx.controls = SELF, audit, controls
    prior_configuration, prior_current = ctx.configuration, ctx.current

    def configuration():
        config, env, assets = prior_configuration()
        args = config['args']
        require(args[args.index('--adapt-swaps')+1] == '8', 'qualified8-entry basis changed')
        args[args.index('--adapt-swaps')+1] = str(BUDGET)
        return config, env, assets

    def current(row, full=False):
        prerequisite_rows(ctx)
        require(profile_pin(ctx, row) == row['placement_profile'], 'placement profile pin changed')
        return prior_current(row, full=full)

    ctx.configuration, ctx.current = configuration, current
    return ctx


def prepare(ctx):
    prerequisite_rows(ctx)
    ctx.prepare()
    row = json.loads((ctx.ROOT / 'record.json').read_text())
    try:
        row['placement_profile'] = profile_pin(ctx, row)
        require(row['plan'].count('positive8swaps/every8') == 1, 'native plan anchor changed')
        row['plan'] = row['plan'].replace('positive8swaps/every8', 'positive32swaps/every8')
        row['plan'] += ' Require an actual32-copy group and both incoming GPU/outgoing CPU observations. The initial expert profile is separately Git-pinned. No speed claim or new engine build.'
        row['limits'] = 'Synthetic32-budget payload/map/dispatch/loan correctness only. The larger-budget refusal control,clean performance,quality,prefill preservation,physical262144/RESTORE and ordinary teardown remain pending.'
    except BaseException as error:
        row.update(stage='failed-prepare', passed=False, complete=False, error=repr(error))
        raise
    finally:
        ctx.save(row)


if __name__ == '__main__':
    require(sys.argv[1:] in (['prepare'], ['run']), 'use prepare|run')
    ctx = context()
    with (ctx.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        ctx.fcntl.flock(lock, ctx.fcntl.LOCK_EX | ctx.fcntl.LOCK_NB)
        ctx.base.guards.no_model_owners([])
        if sys.argv[1] == 'prepare':
            prepare(ctx)
        else:
            ctx.run()
