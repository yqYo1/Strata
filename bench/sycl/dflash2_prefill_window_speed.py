"""Compare the admitted DFlash2 prefill windows with the official benchmark."""

from pathlib import Path
import re
import statistics
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
import dflash2_prefill_window_admission as window

bank, basis, require = window.bank, window.basis, window.require
speed, gate = window.store.speed, window.store.gate
READ, RUNS, HEADS = window.READ, speed.RUNS, 1
KEY = 'dflash2_prefill_window_speed_v1'
ROOT = bank.B / KEY.replace('_', '-')
THIS = str(Path(__file__).resolve().relative_to(bank.P))
ORDER = ('window10', 'window16', 'window16', 'window10', 'window10', 'window16')


def baseline():
    row = speed.baseline(gate.SPLIT_READ)
    admission = basis.validation.committed(bank.RESULT, (window.KEY,))
    plan = admission['plan']
    require(admission['stage'] == 'closed' and admission['complete'] and admission['passed']
            and admission['ownership_closed'] and not admission['active']
            and list(admission['outcomes']) == list(window.ARMS)
            and all(admission['comparison'][key] for key in ('independent_native_processes',
                    'short_state_and_outputs_match', 'kv_stage_matches'))
            and all(plan[key] == row[key] for key in ('binary', 'native_source_commit',
                    'native_build_sha256', 'boot', 'driver', 'runtime_manifest'))
            and plan['validation'] == row['native_validation_probe']
            and plan['drafter_assets'] == row['dflash2_assets'], 'require the closed same-build window admission')
    for arm, mode in window.ARMS.items():
        outcome = admission['outcomes'][arm]
        require(outcome['passed'] and outcome['complete'] and outcome['ownership_closed']
                and outcome['native_exit_code'] == 0 and not outcome['fresh_faults']
                and outcome['route']['windows'] == window.budget(mode)['windows']
                and plan['cases'][arm]['wire'][1]['read'] == READ, 'window admission scope differs')
    return dict(row, window_admission_sha256=bank.signature(admission))


def environment(row, plan):
    env = window.store.environment(row, Path(plan['root']), False)
    require(env[window.MODE] == '10' and env[window.UPLOAD] == 'batched'
            and env['STRATA_SYCL_PREFILL_EXPERT_BANK_SLOTS'] == str(window.SLOTS)
            and env['STRATA_SYCL_PREFILL_MOE_SPLIT_ACCUMULATOR'] == '1'
            and env.get(window.RELOCATE, '0') == '0'
            and env['STRATA_PREFILL_RELEASE_HEADS'] == str(HEADS),
            'unexpected inherited long-prefill policy')
    env.update({window.MODE: str(window.ARMS[plan['arm']]), window.UPLOAD: 'demand'})
    return env


def request_route(text, plan, *, direct=False, read=READ, slots=window.SLOTS):
    mode = window.ARMS[plan['arm']]
    rows = read - 1
    route = window.window_result(text, mode, direct=direct, slots=slots, rows=rows)
    layout = window.workspace.one(text, 'strata prefill: layer-major candidate, ')
    expected = dict(tokens=rows, chunk=window.store.CHUNK, residual_dtype='bf16',
                    residual_element_bytes=2, residual_gpu_rows=rows,
                    residual_gpu=rows * window.workspace.ROW, residual_host=0)
    require(all(layout[key] == str(value) for key, value in expected.items()),
            'quiet benchmark must execute the proven all-GPU BF16 layout')
    tap, = re.findall(r'^strata DFlash2: eager, prompt_tail_rows=2046 cold_prefix=0 '
                      r'vram_bytes=([1-9][0-9]*)$', text, re.M)
    require('strata loan audit:' not in text, 'unexpected cache-loan observer')
    pages = speed.KV.check_clean_request(text, 0, rows, window.store.CHUNK,
                                         reuse=True, tap_layers=gate.TAPPED_QSA_LAYERS)
    request, = re.findall(speed.generation.clock.footer.HEADER, text, re.M)
    return dict(route, kv=pages, tap_vram_bytes=int(tap), request_work=list(map(int, request)))


def after_execution(outcome, row, plan):
    require(outcome['clean_speed'] and outcome['native_identity'] not in
            [p['native_identity'] for p in row['processes']], 'require a fresh quiet process')
    for route, sample in zip(outcome['routes'], outcome['samples'], strict=True):
        require(route['request_work'] == [READ, 0, READ, sample['generated_tokens'],
                    sample['drafts_accepted'], sample['drafts_offered']], 'native and client request work differs')
    outcome['arm'] = plan['arm']


def decision(processes):
    require(tuple(p['arm'] for p in processes) == ORDER, 'expected three counterbalanced pairs')
    times = []
    for process in processes:
        first, *warm = process['samples']
        warm_ms = statistics.median(s['prompt_ms'] for s in warm)
        times.append(dict(arm=process['arm'], first_ms=first['prompt_ms'], first_ttft_s=first['ttft_s'],
                          warm_ms=warm_ms, warm_ttft_s=statistics.median(s['ttft_s'] for s in warm),
                          warm_pp=READ * 1000 / warm_ms))
    pairs = []
    for first in range(0, len(ORDER), 2):
        second = first + 1
        reference, candidate = (first, second) if ORDER[first] == 'window10' else (second, first)
        earlier, later = times[reference], times[candidate]
        base, trial = processes[reference], processes[candidate]
        routes = list(zip(base['routes'], trial['routes'], strict=True))
        pairs.append(dict(
            time_ratios={key: later[key] / earlier[key] for key in ('first_ms', 'first_ttft_s',
                         'warm_ms', 'warm_ttft_s')},
            matched_decode_work=all(x['request_work'] == y['request_work'] for x, y in routes),
            matched_kv=all(x['kv'] == y['kv'] for x, y in routes),
            requested_expert_H2D_bytes_removed=[x['requested_bank_H2D_bytes'] - y['requested_bank_H2D_bytes']
                                                for x, y in routes]))
    rates = {arm: [time['warm_pp'] for time in times if time['arm'] == arm] for arm in window.ARMS}
    return dict(
        process_times=times, pairs=pairs, warmed_PP=rates,
        median_warmed_PP={arm: statistics.median(values) for arm, values in rates.items()},
        repeated_warm_gain=all(pair['matched_kv'] and pair['matched_decode_work']
                               and pair['time_ratios']['warm_ms'] < 1 for pair in pairs),
        repeated_expert_H2D_reduction=all(all(v > 0 for v in pair['requested_expert_H2D_bytes_removed'])
                                          for pair in pairs),
        repeated_1000={arm: all(value >= 1000 for value in values) for arm, values in rates.items()},
        full_context_qualified=False, sustained_decode_qualified=False, adopted=False)


def experiment():
    reference = speed.experiment(gate.SPLIT_READ)
    ctx = speed.generation.experiment(synchronous_commit=True)
    ctx.KEY, ctx.ROOT, ctx.THIS, ctx.ORDER = KEY, ROOT, THIS, ORDER
    ctx.baseline, ctx.configuration, ctx.environment = baseline, speed.configuration, environment
    ctx.command = lambda plan: speed.command(plan, READ)
    ctx.protocol = dict(
        order=list(ORDER), read=READ, requests_per_process=RUNS,
        unit='Change the prefill window selector only. Hold demand uploads, bank128, copied attention, '
             'retained KV, head detachment and DFlash2 placement fixed.',
        acceptance='Use three counterbalanced fresh process pairs through the pinned bench.sh exact-prefill mode. '
                   'Verify each actual all-GPU route, VRAM headroom and retirement. Match counted request bodies. '
                   'Keep first-use and warmed PP/TTFT separate. Advance the warm gain only if all three paired '
                   'ratios are below one with matched KV and emitted verifier work. Record transfer reduction '
                   'and repeated 1000 tok/s separately.',
        stop='Stop on changed inputs, route mismatch, fallback, runtime, correctness, ownership or fault. '
             'Retain mixed timings and differing work. The short output exercises handoff.')
    old_current, old_save = ctx.current, ctx.save

    def pins(revision):
        values = reference.pins(revision)
        for name in (THIS, window.THIS, window.store.THIS,
                     str(Path(window.workspace.__file__).resolve().relative_to(bank.P))):
            path = bank.P / name
            require(path.read_bytes() == basis.validation.git('show', revision + ':' + name),
                    'changed DFlash2 window timing input: ' + name)
            values[name] = bank.sha(path)
        return values

    def save(row):
        if row['stage'] == 'prepared':
            row['request_hashes'] = {}
            for plan in row['plans']:
                plan.update(read=READ, heads=HEADS)
        old_save(row)

    def current(row):
        normalized = []
        for plan in row['plans']:
            require((plan['read'], plan['heads']) == (READ, HEADS), 'planned input extent changed')
            env = environment(row, plan)
            require(env.pop(window.MODE) == str(window.ARMS[plan['arm']]), 'unexpected window selector')
            normalized.append(bank.signature({key: value.replace(plan['root'], '<process>')
                                                for key, value in env.items()}))
        require(len(set(normalized)) == 1, 'settings differ outside the window selector')
        return old_current(row)

    ctx.pins, ctx.current, ctx.save = pins, current, save
    runner = SimpleNamespace(ROOT=ROOT, ORDER=[(READ, HEADS)] * len(ORDER), RUNS=RUNS,
        current=current, environment=environment, environment_fingerprint=speed.generation.suite.signature,
        save=save, request_route=request_route, after_execution=after_execution, decision=decision,
        validate_model_text=speed.validate_model_text, verify_model_assets=speed.verify_model_assets)
    ctx.run = lambda index: basis.comparison.run(index, runner)
    return ctx


if __name__ == '__main__':
    speed.generation.clock.main(experiment())
