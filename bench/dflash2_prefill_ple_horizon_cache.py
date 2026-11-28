"""Isolate the dedicated row cache on the whole-request prefill route."""

from pathlib import Path
import sys

sys.dont_write_bytecode = True
import dflash2_prefill_ple_horizon_clean as horizon
import dflash2_prefill_ple_record_window as grouped

bank, wait, require = grouped.bank, grouped.wait, grouped.require
capacity, corpus, phase = grouped.capacity, grouped.corpus, grouped.phase
THIS = str(Path(__file__).resolve().relative_to(bank.P))
KEY = 'dflash2_prefill_ple_horizon_cache_v1'
ROOT = bank.B / KEY.replace('_', '-')
ORDER = capacity.ORDER


def baseline(clean):
    selected = clean.baseline()
    result = grouped.validate.committed(bank.RESULT, (clean.KEY,))
    require(wait.retained.closed(result, 'closed')
            and result['plan'] == clean.protocol
            and result['decision'] == clean.decision(result['processes'])
            and result['decision']['repeated_first_gain']
            and all(result[k] == selected[k] for k in (
                'binary', 'native_source_commit', 'native_build_sha256', 'request_hashes',
                'constant_request_work')), 'require the matched clean two-versus-whole result')
    reference = grouped.validate.committed(bank.RESULT, (phase.KEY,))
    require(wait.retained.closed(reference, 'closed')
            and reference['decision']['warm_cache_mechanism_advances']
            and reference['request_hashes'] == selected['request_hashes']
            and reference['frozen_work'] == selected['constant_request_work'],
            'require the prior isolated-cache result and its frozen requests')
    return dict(selected, clean_horizon_sha256=bank.signature(result),
                dedicated_cache_reference_sha256=bank.signature(reference))


def route(text, plan):
    phase.check_first_cache(text, plan['arm'], 1)
    require(phase.TICKET not in text, 'per-ticket cache observer is enabled')
    return grouped.clean_route(text, dict(plan, arm='full'))


def after_execution(outcome, row, plan):
    text = (Path(plan['root']) / 'engine.stderr').read_text()
    rows = capacity.LARGE_ROWS if plan['arm'] == 'large' else 0
    starts = [line for line in text.splitlines() if line.startswith(phase.START)]
    expected = [f'{phase.START}long_prefill={rows} ordinary={capacity.DEFAULT_ROWS}'] if rows else []
    require(starts == expected, 'the actual dedicated cache allocation differs')
    grouped.clean_after_execution(outcome, row, plan)
    outcome['dedicated_cache_rows'] = rows


def decision(processes):
    require(tuple(p['arm'] for p in processes) == ORDER
            and all(p['passed'] and p['complete'] and p['ownership_closed']
                    and p['clean_speed'] and p['native_exit_code'] == 0 and not p['fresh_faults']
                    for p in processes)
            and len({(p['native_identity']['pid'], p['native_identity']['start_ticks'])
                     for p in processes}) == len(ORDER)
            and [p['dedicated_cache_rows'] for p in processes]
                == [0, capacity.LARGE_ROWS, capacity.LARGE_ROWS, 0],
            'require a closed pair in each order and the actual dedicated cache extents')
    pairs = [grouped.request_time_ratios(stock, large) for stock, large in (
        (processes[0], processes[1]), (processes[3], processes[2]))]
    return dict(pairs=pairs,
                warm_cache_advances=all(p['subsequent_pp_time'] < 1 and p['subsequent_ttft'] < 1
                                        for p in pairs),
                first_preserved_in_both=all(p['first_pp_time'] <= 1 and p['first_ttft'] <= 1
                                            for p in pairs),
                first_1000=[p['samples'][0]['prompt_ms'] <= wait.READ
                            for p in processes if p['arm'] == 'large'],
                subsequent_1000=[[s['prompt_ms'] <= wait.READ for s in p['samples'][1:]]
                                 for p in processes if p['arm'] == 'large'], adopted=False)


def experiment():
    clean = horizon.experiment()
    ctx = grouped.geometry.experiment()
    ctx.KEY, ctx.ROOT, ctx.THIS, ctx.ORDER = KEY, ROOT, THIS, ORDER
    payload = ROOT / 'corpus.txt'
    digest, paths = corpus.CORPORA[capacity.WORKLOAD]
    save_base, current_base, command_base = ctx.save, ctx.current, ctx.command

    def save(row):
        if row['stage'] == 'prepared':
            with payload.open('xb') as out:
                out.write(corpus.corpus_payload(capacity.WORKLOAD))
            payload.chmod(0o400)
        save_base(row)

    def current(row):
        require(bank.sha(payload) == digest, 'the frozen prose corpus changed')
        return current_base(row)

    def environment(row, plan):
        env = clean.environment(row, dict(plan, arm='full'))
        rows = capacity.LARGE_ROWS if plan['arm'] == 'large' else 0
        return dict(env, **{phase.SELECTOR: str(rows)})

    ctx.baseline, ctx.save, ctx.current = lambda: baseline(clean), save, current
    ctx.environment = environment
    ctx.command = lambda plan: [*command_base(plan), '--prefill-corpus', str(payload)]
    ctx.selector = phase.SELECTOR
    ctx.extra_sources = (*clean.extra_sources, phase.THIS, THIS)
    ctx.request_route, ctx.after_execution, ctx.decision = route, after_execution, decision
    ctx.run = lambda index: grouped.run_clean(ctx, index)
    ctx.protocol = dict(order=list(ORDER), read=wait.READ, requests_per_process=wait.speed.RUNS,
        corpus=capacity.WORKLOAD, corpus_sha256=digest, corpus_revision=corpus.CORPUS_REVISION,
        corpus_paths=list(paths), ordinary_cache_rows=capacity.DEFAULT_ROWS,
        dedicated_cache_rows=capacity.LARGE_ROWS, record_horizon='full',
        added_cache_array_bytes=capacity.LARGE_ROWS * 94 + capacity.LARGE_ROWS // 8,
        scope='Same prefill build, full record horizon, ordinary cache and input in both arms. '
              'Change only the dedicated prefill cache, with observers off and two reverse-order pairs.',
        decision='Require lower median revisit prefill time and TTFT in both pairs for the cache gain. '
                 'Report first-use effects and 1000 tok/s per request separately.',
        limits='One repeated technical-prose input. Vary the horizon at the selected cache size before '
               'claiming an interaction. Verify physical context, RESTORE and sustained decode separately.',
        stop=grouped.geometry.PLAN['stop'])
    return ctx


if __name__ == '__main__':
    require(sys.executable == bank.basis.reference.model.PYTHON, 'use the pinned controller interpreter')
    wait.speed.clock.main(experiment())
