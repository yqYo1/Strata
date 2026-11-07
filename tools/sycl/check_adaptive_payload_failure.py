#!/usr/bin/env python3
"""Require unpublished second-record refusal on the qualified32 successor.

Main-authored; repository LICENSE. The expected native exit is1. The failed
original cap trial and all prior controls keep their original outcomes.
"""
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import check_adaptive_budget_failure as original
import check_adaptive_payload_entry as positive

SELF = Path(__file__).resolve()
KEY = 'adaptive_payload32_record_failure_v1'
require = original.require


def context():
    ctx = original.context()
    ctx.positive = positive.context()
    ctx.ROOT, ctx.KEY, ctx.SELF = ctx.B / 'adaptive-payload32-record-failure-v1', KEY, SELF
    prior_current = ctx.current

    def current(row, full=False):
        require(row['successor_positive_key'] == positive.KEY and
            row['successor_build_key'] == ctx.positive.admission.build.KEY,
            'same successor positive/build binding required')
        return prior_current(row, full=full)

    ctx.current = current
    return ctx


def prepare(ctx):
    original.prepare(ctx)
    row = json.loads((ctx.ROOT / 'record.json').read_text())
    try:
        row.update(successor_positive_key=positive.KEY,
            successor_build_key=ctx.positive.admission.build.KEY)
        require(row['plan'].count('Same429b native engine') == 1, 'failure plan anchor changed')
        row['plan'] = row['plan'].replace('Same429b native engine',
            'Same host-qualified payload successor native engine')
        row['plan'] += (' The new positive32entry must close first. Preserve the original '
            'failed64MiB trial. Require real first completion/publication,exact32queued '
            'second copies,checked null-record refusal,second mask unpublished,normal '
            'expected1,closed owners and no fresh fault. No speed claim or retry.')
    except BaseException as error:
        row.update(stage='failed-successor-prepare', passed=False, complete=False, error=repr(error))
        raise
    finally: ctx.save(row)


if __name__ == '__main__':
    require(sys.argv[1:] in (['prepare'], ['run']), 'use prepare|run')
    ctx = context()
    with (ctx.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        ctx.fcntl.flock(lock, ctx.fcntl.LOCK_EX | ctx.fcntl.LOCK_NB)
        ctx.base.guards.no_model_owners([])
        if sys.argv[1] == 'prepare': prepare(ctx)
        else: ctx.run()
