#!/usr/bin/env python3
"""Refuse a second event-record failure after two full32-copy groups.

Main-authored; repository LICENSE. Reuse the owned failure supervisor and
existing engine. No GPU execution is eligible before the positive32 entry.
"""
import importlib.util
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import check_adaptive_budget_entry as budget
import check_adaptive_record_failure as original

SELF = Path(__file__).resolve()
require = original.require
BUDGET = budget.BUDGET


def audit(text):
    """Require a real successful record, then a failed record with no publish."""
    prefix = 'strata serve: ADAPT_AUDIT '
    patterns = [re.compile(pattern) for pattern in (
        prefix + r'phase=prepared generation=(\d+) incoming=(\d+) outgoing=(\d+) slot=(\d+)',
        prefix + r'phase=masked generation=(\d+) swaps=(\d+) identities=3',
        prefix + r'phase=published generation=(\d+) swaps=(\d+) identities=3 payload_match=1',
        r'strata: ADAPT_RECORD_DIAGNOSTIC ordinal=(\d+) injected=(\d+) status=(\d+)',
        r'strata serve: ADAPT_SUBMIT swaps=(\d+) H2D_requested_bytes=(\d+)',
        r'strata serve: cannot record the adaptive refill event')]
    rows, state, submitted_bytes = {1: [], 2: []}, 0, None
    for line in text.splitlines():
        if not line.startswith((prefix, 'strata: ADAPT_RECORD_DIAGNOSTIC ',
                'strata serve: ADAPT_SUBMIT ', 'strata serve: cannot record')):
            continue
        found = [(kind, p.fullmatch(line)) for kind, p in enumerate(patterns)
                 if p.fullmatch(line)]
        require(len(found) == 1, 'unknown failure-control progress')
        kind, match = found[0]
        require(all(len(value) <= 20 for value in match.groups()), 'oversized counter')
        values = tuple(map(int, match.groups()))
        require(all(value < 2**64 for value in values), 'unsigned counter bound')
        if kind == 0:
            generation = 1 if state == 0 else 2 if state == 4 else 0
            require(generation in rows and values[0] == generation and
                    len(rows[generation]) < BUDGET, 'prepare generation/order/budget')
            rows[generation].append(values[1:])
        elif kind == 1:
            generation = 1 if state == 0 else 2 if state == 4 else 0
            require(generation in rows and len(rows[generation]) == BUDGET and
                    values == (generation, BUDGET), 'full32 mask/count/order required')
            state += 1
        elif kind == 2:
            require(state == 3 and values == (1, BUDGET), 'stale publication')
            state = 4
        elif kind == 3:
            require((state == 1 and values == (1, 0, 0)) or
                    (state == 5 and values == (2, 1, 1)), 'actual record statuses/order')
            state += 1
        elif kind == 4:
            require(state == 2 and values[0] == BUDGET and values[1] > 0,
                    'first real positive-copy submission required')
            submitted_bytes, state = values[1], 3
        else:
            require(state == 6, 'refusal must follow actual second record failure')
            state = 7
    require(state == 7 and submitted_bytes is not None, 'incomplete refusal')
    for group in rows.values():
        require(len(group) == BUDGET and all(len({v[column] for v in group}) == BUDGET
                for column in range(3)), 'replacement tuple conservation')
        require(not ({v[0] for v in group} & {v[1] for v in group}) and all(
                i < 24576 and o < 24576 and i // 512 == o // 512 and s < 1256
                for i, o, s in group), 'replacement roles/geometry')
    return dict(first_record_status=0, second_record_status=1,
        successful_generation=1, first_swaps=BUDGET, failed_generation=2,
        queued_second_swaps=BUDGET, second_generation_masked=True,
        second_generation_published=False, first_H2D_requested_bytes=submitted_bytes,
        stale_event_not_published=True, maximum_group=BUDGET)


def controls():
    prefix = 'strata serve: ADAPT_AUDIT '
    groups = [[prefix + f'phase=prepared generation={g} incoming={100+g*40+i} '
               f'outgoing={i} slot={i}' for i in range(BUDGET)] for g in (1, 2)]
    first = [prefix + 'phase=masked generation=1 swaps=32 identities=3',
        'strata: ADAPT_RECORD_DIAGNOSTIC ordinal=1 injected=0 status=0',
        'strata serve: ADAPT_SUBMIT swaps=32 H2D_requested_bytes=67108864',
        prefix + 'phase=published generation=1 swaps=32 identities=3 payload_match=1']
    second = [prefix + 'phase=masked generation=2 swaps=32 identities=3',
        'strata: ADAPT_RECORD_DIAGNOSTIC ordinal=2 injected=1 status=1',
        'strata serve: cannot record the adaptive refill event']
    good = groups[0] + first + groups[1] + second
    require(audit('\n'.join(good))['queued_second_swaps'] == BUDGET,
            'valid full32 failure fixture')
    corrupt = [good[:i] + good[i+1:] for i in range(len(good))]
    corrupt += [good + [good[-1]], good + [first[-1].replace('generation=1', 'generation=2')],
        [good[0], *good],
        [good[0].replace('incoming=140', 'incoming=512'), *good[1:]],
        [good[0].replace('incoming=140', 'incoming=1'), *good[1:]],
        [good[0].replace('slot=0', 'slot=1256'), *good[1:]],
        [good[0].replace('generation=1', 'generation=18446744073709551616'), *good[1:]],
        groups[0] + first + groups[1] + [second[1], second[0], second[2]],
        groups[0] + first + groups[1] + [second[0], second[1].replace('status=1', 'status=0'), second[2]],
        groups[0] + first + groups[1] + [second[0], second[1].replace('injected=1', 'injected=0'), second[2]],
        groups[0] + [first[0], first[1].replace('status=0', 'status=1'), *first[2:]] + groups[1] + second,
        groups[0] + [*first[:2], first[2].replace('67108864', '0'), first[3]] + groups[1] + second,
        [line.replace('swaps=32', 'swaps=31') for line in good],
        groups[0] + [first[0], first[2], first[1], first[3]] + groups[1] + second,
        good[:-1] + [good[-1] + ' unknown=1']]
    for lines in corrupt:
        try:
            audit('\n'.join(lines))
        except RuntimeError:
            continue
        raise RuntimeError('corrupt full32 refusal accepted')
    return dict(passed=True, full_groups=2, maximum_group=BUDGET,
        negative=len(corrupt), GPU_executed=False,
        scope='CPU ordered-reader controls only;native32 refusal remains unqualified.')


def context():
    spec = importlib.util.spec_from_file_location('strata_budget32_record_failure', original.__file__)
    ctx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ctx)
    ctx.positive = budget.context()
    ctx.ROOT, ctx.KEY = ctx.B / 'adaptive-budget32-record-failure-v1', 'adaptive_budget32_record_failure_v1'
    ctx.SELF, ctx.audit, ctx.controls = SELF, audit, controls
    prior_current = ctx.current

    def normal_entry():
        row = json.loads((ctx.positive.ROOT / 'record.json').read_text())
        require(row['stage'] == 'closed' and row['passed'] and row['complete'] and
            row['ownership_closed'] and not row['active'] and
            row['outcome']['native_exit_code'] == 0 and
            row['outcome']['adaptive']['maximum_group'] == BUDGET and
            row['outcome']['adaptive']['all_observed_payload_map_dispatch_checks_passed'] and
            row['outcome']['adaptive']['outgoing_CPU_observed'] and
            row['outcome']['adaptive']['post_positive_loan_groups'] > 0,
            'closed same-engine full32-copy/map/dispatch/loan entry required')
        return row

    ctx.normal_entry = normal_entry
    def current(row, full=False):
        require(budget.profile_pin(ctx, row) == row['placement_profile'],
                'failure-control placement profile changed')
        return prior_current(row, full=full)

    ctx.current = current
    return ctx


def prepare(ctx):
    ctx.prepare()
    row = json.loads((ctx.ROOT / 'record.json').read_text())
    try:
        row['placement_profile'] = budget.profile_pin(ctx, row)
        row['plan'] += ' Both first and second generations must contain exactly32 accepted copies;generation2 must stay masked and unpublished. Same429b native engine and initial Git-pinned placement profile.'
        row['limits'] += ' This larger-budget test does not qualify clean adaptive32 performance.'
    except BaseException as error:
        row.update(stage='failed-prepare', passed=False, complete=False, error=repr(error))
        raise
    finally:
        ctx.save(row)


if __name__ == '__main__':
    require(sys.argv[1:] in (['controls'], ['prepare'], ['run']), 'use controls|prepare|run')
    ctx = context()
    with (ctx.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        ctx.fcntl.flock(lock, ctx.fcntl.LOCK_EX | ctx.fcntl.LOCK_NB)
        ctx.base.guards.no_model_owners([])
        if sys.argv[1] == 'controls':
            print(json.dumps(controls(), sort_keys=True))
        elif sys.argv[1] == 'prepare':
            prepare(ctx)
        else:
            ctx.run()
