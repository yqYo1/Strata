#!/usr/bin/env python3
"""Exercise full32-copy replacement/dispatch/loan on the payload successor.

Main-authored; repository LICENSE. Preserve the original failed64MiB trial.
Reuse the unchanged ordered reader and owned supervisor,not its failed result.
"""
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import adaptive_payload_admission as admission
import check_adaptive_budget_entry as budget

SELF = Path(__file__).resolve()
KEY = 'adaptive_payload32_native_entry_v1'
require = budget.require


def context():
    ctx = budget.context()
    ctx.admission = admission
    ctx.ROOT, ctx.KEY, ctx.SELF = ctx.B / 'adaptive-payload32-entry-v1', KEY, SELF
    prior_current = ctx.current

    def current(row, full=False):
        proof = admission.build.host_proof()
        require(row['successor_build_key'] == admission.build.KEY and
            row['host_qualified_header_sha256'] == proof['generated'][admission.build.source.HEADER] and
            row['predecessor_failed_key'] == 'adaptive_budget32_native_entry_v1',
            'successor header/provenance binding changed')
        return prior_current(row, full=full)

    ctx.current = current
    return ctx


def prepare(ctx):
    snapshot = ctx.base.committed(ctx.RESULT)
    failed = snapshot['adaptive_budget32_native_entry_v1']
    require(failed['stage'] == 'failed' and failed['passed'] is False and failed['complete'] is False,
        'preserve the failed original64MiB qualification')
    proof = admission.build.host_proof()
    budget.prepare(ctx)
    row = json.loads((ctx.ROOT / 'record.json').read_text())
    try:
        admission.build.check_publication(snapshot, ctx.KEY, ctx.compact(row),
            json.loads(ctx.RESULT.read_text()))
        row.update(successor_build_key=admission.build.KEY,
            host_qualified_header_sha256=proof['generated'][admission.build.source.HEADER],
            predecessor_failed_key='adaptive_budget32_native_entry_v1')
        row['plan'] = ('One fresh owned successor serving engine,two synthetic requests1024/256 '
            'then32769/256. Only its diagnostic header changed from429b;57host cases,fresh28+28 '
            'CPU pool parity and exact five-fixture/runtime component reuse must close first. '
            'Commit exact source,new binary,initial profile,15assets,configuration and safety '
            'before exec. Positive32swaps/every8,cap4,half1,INT8resident131072,1256slots,5workers,'
            '8192chunk,throughput BF16 HC/GDN16/MTP16,20ms pool spin and qualified runtime safety. '
            'Use immutable pre-exec progress/parameter-validation and payload/map/dispatch/loan '
            'audits,no full API poll trace. Require ordered INFO/READY/RESUME/PP/REUSED/T/DONE '
            'for both,real complete32-copy generations,checked completion and full payload/'
            'host-private-device publication,both incoming GPU and outgoing CPU dispatch roles,'
            'occupied post-positive loan/refill,actual T1/T4 Q6 windows,normalQUIT0,quiet '
            'validation,unchanged final15assets/boot/driver,no fresh fault and closed owners. '
            'Expert extent remains64MiB;cycle is256MiB. Any refusal or budget stop stays failed;'
            'no retry,pin refresh or narrower32group. Rare EOS controls extent,not quality. '
            'The original64MiB trial remains failed. New second-record32refusal,clean8/32speed,'
            'task quality,prefill preservation,ordinary teardown and physical262144remain open.')
        row['limits'] = ('Synthetic successor32payload/map/dispatch/loan only. No speed or broad '
            'model qualification;the original failed trial is not relabeled or reused as success.')
    except BaseException as error:
        row.update(stage='failed-successor-prepare', passed=False, complete=False, error=repr(error))
        raise
    finally: ctx.save(row)
    print('Prepared successor32entry;commit before native execution.', flush=True)


if __name__ == '__main__':
    require(sys.argv[1:] in (['prepare'], ['run']), 'use prepare|run')
    ctx = context()
    with (ctx.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        ctx.fcntl.flock(lock, ctx.fcntl.LOCK_EX | ctx.fcntl.LOCK_NB)
        ctx.base.guards.no_model_owners([])
        if sys.argv[1] == 'prepare': prepare(ctx)
        else: ctx.run()
