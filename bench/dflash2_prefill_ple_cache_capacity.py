"""Screen retained PLE rows on the same varied-text diagnostic build."""

import json
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
import dflash2_prefill_ple_corpus as corpus

bank, wait, geometry, validate, require = corpus.bank, corpus.wait, corpus.geometry, corpus.validate, corpus.require
THIS = str(Path(__file__).resolve().relative_to(bank.P))
KEY = 'dflash2_prefill_ple_cache_capacity_v1'
ROOT = bank.B / KEY.replace('_', '-')
WORKLOAD = 'technical-prose'
FILE = ROOT / 'corpus.txt'
DEFAULT_ROWS, LARGE_ROWS = 1 << 20, 4 << 20
FLAG = '--ple-row-cache'
ORDER = ('stock', 'large', 'large', 'stock')


def baseline():
    selected = corpus.baseline()
    reference = validate.committed(bank.RESULT, (corpus.campaign_key(WORKLOAD),))
    require(wait.retained.closed(reference, 'closed') and len(reference['processes']) == 2
            and reference['binary'] == selected['binary'] and FLAG not in selected['configuration']['args']
            and reference['plan']['corpus_sha256'] == corpus.CORPORA[WORKLOAD][0],
            'require the two closed stock prose processes on the retained observer build')
    for p in reference['processes']:
        require(p['passed'] and p['complete'] and p['ownership_closed'] and p['native_exit_code'] == 0
                and not p['fresh_faults'] and len(p['samples']) == len(p['routes']) == wait.speed.RUNS
                and all(s['prompt_tps'] < 1000 for s in p['samples'][1:])
                and all(r['record_touches'] > 0 for r in p['routes'][1:]),
                'warm prose must show repeatable storage work and a remaining throughput gap')
    hashes = [s['request_sha256'] for s in reference['processes'][0]['samples']]
    require(hashes == [s['request_sha256'] for s in reference['processes'][1]['samples']],
            'the closed stock prose request bodies differ')
    return dict(selected, request_hashes={str(wait.READ): hashes},
                closed_prose_receipt_sha256=bank.signature(reference))


def after_execution(outcome, row, plan):
    launch = json.loads((Path(plan['root']) / 'engine-start.json').read_text())
    args = launch['args']
    occurrences = [i for i, value in enumerate(args) if value == FLAG]
    expected = [str(LARGE_ROWS)] if plan['arm'] == 'large' else []
    require(launch['native'] == outcome['native_identity']
            and [args[i + 1] for i in occurrences if i + 1 < len(args)] == expected
            and len(occurrences) == len(expected), 'actual native row-cache option differs')
    corpus.after_execution(outcome, row, plan)
    outcome['row_cache_rows'] = LARGE_ROWS if expected else DEFAULT_ROWS


def decision(processes):
    require(tuple(p['arm'] for p in processes) == ORDER
            and all(p['passed'] and p['complete'] and p['ownership_closed']
                    and p['native_exit_code'] == 0 and not p['fresh_faults'] for p in processes)
            and len({(p['native_identity']['pid'], p['native_identity']['start_ticks'])
                     for p in processes}) == len(ORDER), 'four independently closed processes required')
    pairs = []
    for stock, large in ((processes[0], processes[1]), (processes[3], processes[2])):
        waits = [[r['ple_wait']['future_get_ms'] for r in p['routes'][1:]] for p in (stock, large)]
        records = [sum(r['record_touches'] for r in p['routes'][1:]) for p in (stock, large)]
        ttft = [statistics.median(s['ttft_s'] for s in p['samples'][1:]) for p in (stock, large)]
        pairs.append(dict(warm_read_record_totals=records, warm_wait_ms=waits,
                          warm_prompt_ratio=large['subsequent_median_ms'] / stock['subsequent_median_ms'],
                          warm_ttft_ratio=ttft[1] / ttft[0],
                          first_prefill_ratio=large['samples'][0]['prompt_ms'] / stock['samples'][0]['prompt_ms'],
                          first_ttft_ratio=large['samples'][0]['ttft_s'] / stock['samples'][0]['ttft_s']))
    eligible = all(pair['warm_read_record_totals'][1] < pair['warm_read_record_totals'][0]
                   and pair['warm_prompt_ratio'] < 1 and pair['warm_ttft_ratio'] < 1 for pair in pairs)
    return dict(pairs=pairs, warm_cache_mechanism_advances=eligible,
                warm_large_reaches_1000=all(s['prompt_tps'] >= 1000 for p in processes
                    if p['arm'] == 'large' for s in p['samples'][1:]), adopted=False)


def experiment():
    ctx = geometry.experiment()
    ctx.KEY, ctx.ROOT, ctx.THIS, ctx.ORDER = KEY, ROOT, THIS, ORDER
    base_configuration, base_command = ctx.configuration, ctx.command
    base_save, base_current = ctx.save, ctx.current
    digest, paths = corpus.CORPORA[WORKLOAD]

    def configuration(row, plan):
        cfg = base_configuration(row, plan)
        require(FLAG not in cfg['args'], 'stock PLE cache configuration differs')
        if plan['arm'] == 'large':
            cfg = dict(cfg, args=[*cfg['args'], FLAG, str(LARGE_ROWS)])
        return cfg

    def save(row):
        if row['stage'] == 'prepared':
            with FILE.open('xb') as target:
                target.write(corpus.corpus_payload(WORKLOAD))
            FILE.chmod(0o400)
        base_save(row)

    def current(row):
        require(bank.sha(FILE) == digest, 'corpus bytes differ')
        return base_current(row)

    ctx.baseline, ctx.configuration = baseline, configuration
    ctx.save, ctx.current = save, current
    ctx.command = lambda plan: [*base_command(plan), '--prefill-corpus', str(FILE)]
    ctx.environment = lambda row, plan: geometry.environment(row, dict(plan, arm='geometry'))
    ctx.request_route = lambda text, plan: geometry.route(text, dict(plan, arm='geometry'))
    ctx.after_execution, ctx.decision = after_execution, decision
    ctx.extra_sources = (*ctx.extra_sources, corpus.THIS, THIS)
    ctx.protocol = dict(order=list(ORDER), read=wait.READ, requests_per_process=wait.speed.RUNS,
        corpus=WORKLOAD, corpus_revision=corpus.CORPUS_REVISION, corpus_sha256=digest, corpus_paths=list(paths),
        capacity_rows=dict(stock=DEFAULT_ROWS, large=LARGE_ROWS),
        memory_addition_bytes=(LARGE_ROWS-DEFAULT_ROWS)*94 + (LARGE_ROWS-DEFAULT_ROWS)//8,
        scope='Use the identical instrumented early PLE binary and frozen prose requests. Change only '
              'the existing row-cache capacity option and verify it in the native launch receipt.',
        decision='Fewer subsequent read records and faster subsequent prefill and TTFT in both pairs '
                 'advance the mechanism. Keep first-use clocks and independent samples. Verify clean and '
                 'sustained decode separately before selecting a phase cache policy.',
        stop=geometry.PLAN['stop'])
    return ctx


if __name__ == '__main__':
    wait.speed.clock.main(experiment())
