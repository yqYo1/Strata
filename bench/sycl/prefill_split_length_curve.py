"""Measure prefill time and client TTFT across input lengths on one 128-slot split route."""

import fcntl
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
import prefill_kv_gather_speed as speed

bank, basis, require = speed.bank, speed.basis, speed.require
READS = (94209, 98305, 102401, 106497, 110593, 117489)
EDGE = 98306  # One more batched row than 12 complete 8192-row chunks.
FIRST = (106497, 98305, 98306, 117489, 94209, 110593, 102401)
ORDER = FIRST + FIRST[::-1]
RUNS = 3
KEY = 'prefill_split_length_curve_v1'
THIS = str(Path(__file__).resolve().relative_to(bank.P))


def select_arm(result):
    passed = result['contrast_passes']
    choices = ['base']
    choices.extend(arm for arm in ('reuse', 'vec8') if passed[arm + '/base'])
    if all(passed[name] for name in ('both/base', 'both/reuse', 'both/vec8')):
        choices.append('both')
    return max(choices, key=result['warmed_median_PP'].__getitem__)


def admission():
    native, prior, providers, _ = speed.admission(117489)
    original = basis.validation.committed(bank.RESULT, ('prefill_kv_gather_speed_117489_v1',))
    require(original['complete'] and original['passed'] and original['stage'] == 'closed'
            and original['ownership_closed'] and not original['active']
            and original['native_source_commit'] == native['native_source_commit']
            and original['binary'] == native['binaries']['strata']
            and original['decision'] == speed.decision(117489, original['processes']),
            'finish the balanced 117K comparison on this native engine first')
    prior['observer_admission'].update(curve_arm=select_arm(original['decision']),
        timing_result_sha256=bank.signature(original['decision']))
    return native, prior, providers, None


def summarize(processes):
    require(len(processes) == len(ORDER) and len({(p['native_identity']['pid'], p['native_identity']['start_ticks'])
            for p in processes}) == len(ORDER), 'every length needs independent processes')
    by_length = {n: [] for n in (*READS, EDGE)}
    for i, (p, n) in enumerate(zip(processes, ORDER, strict=True)):
        require(p['index'] == i and p['read'] == n and p['passed'] and p['complete'] and p['clean_speed']
                and p['ownership_closed'] and p['native_exit_code'] == 0 and not p['fresh_faults']
                and len(p['samples']) == RUNS, 'incomplete length or process')
        by_length[n].append(p)

    series = {}
    for phase in ('first', 'warm'):
        for key, factor in (('prompt_ms', 1), ('ttft_s', 1000)):
            points = []
            for n in sorted(by_length):
                times = []
                for process in by_length[n]:
                    samples = process['samples']
                    values = samples[:1] if phase == 'first' else samples[1:]
                    times.append(statistics.median(s[key] for s in values) * factor)
                require(len(times) == 2, 'missing reverse-order repeat')
                points.append(dict(actual_read_tokens=n, prompt_ms=statistics.median(times), process_ms=times))
            smooth = [p for p in points if p['actual_read_tokens'] != EDGE]
            fits = basis.core.approximation_models(smooth)
            require(set(fits['candidates']) == {'affine', 'quadratic', 'n_log_n'}, 'length fit is singular')
            edge_before, edge_after = (next(x for x in points if x['actual_read_tokens'] == n)
                                       for n in (EDGE - 1, EDGE))
            metric = 'prefill' if key == 'prompt_ms' else 'client_TTFT'
            series[phase + '_' + metric] = dict(unit='ms',
                points=[dict(read=p['actual_read_tokens'], median_ms=p['prompt_ms'], process_ms=p['process_ms'])
                        for p in points], smooth_models=fits['candidates'],
                adjacent_ms_per_token=[(b['prompt_ms'] - a['prompt_ms']) / (b['actual_read_tokens'] - a['actual_read_tokens'])
                                       for a, b in zip(smooth, smooth[1:])],
                extra_chunk_one_token_delta_ms=[b - a for a, b in
                    zip(edge_before['process_ms'], edge_after['process_ms'], strict=True)])
    boundary_work = []
    page_bytes = speed.state.kv.PAGE * speed.state.kv.CELL_BYTES
    for before, after in zip(by_length[EDGE - 1], by_length[EDGE], strict=True):
        differences = []
        for a, b in zip(before['routes'], after['routes'], strict=True):
            differences.append(dict(
                jobs=b['fast_window']['jobs'] - a['fast_window']['jobs'],
                fused_down_jobs=b['fast_window']['fused_down_jobs'] - a['fast_window']['fused_down_jobs'],
                bank_H2D_bytes=b['fast_window']['requested_bank_H2D_bytes'] - a['fast_window']['requested_bank_H2D_bytes'],
                prefix_H2D_bytes=page_bytes * (b['kv']['reloaded_pages'] - a['kv']['reloaded_pages'])))
        boundary_work.append(dict(process_indices=[before['index'], after['index']], sample_deltas=differences))
    return dict(series=series, boundary_work=boundary_work, adoption=False, sustained_decode_tested=False)


def experiment():
    ctx = basis.core.source.device.load_module('split_prefill_curve', basis.core.__file__)
    reference = basis.reference.context()
    ctx.THIS, ctx.KEY, ctx.ROOT = THIS, KEY, bank.B / 'prefill-split-length-curve-v1'
    ctx.READS, ctx.ORDER, ctx.RUNS, ctx.CLEAN_FAST = ORDER, [(n, 1) for n in ORDER], RUNS, True
    ctx.basis, ctx.check_sample, ctx.decision = admission, reference.check_sample, summarize
    old_save, old_current, old_pins = ctx.save, ctx.current, ctx.pins

    def pins(revision):
        values = old_pins(revision)
        require({THIS, speed.THIS, *speed.state.SCRIPTS, 'tools/prefill_scaling_bench.py'} <= values.keys(),
                'unfrozen curve controller, work checker or fit implementation')
        return values

    def environment(row, plan):
        env = speed.model.ordinary_environment(row, plan['root'], row['observer_admission']['ordinary_control'])
        reuse, vec8 = speed.state.ARMS[row['observer_admission']['curve_arm']]
        env[speed.model.INPUT] = env[speed.model.OUTPUT] = '0'
        env[speed.state.KV], env[speed.state.GATHER] = str(int(reuse)), str(int(vec8))
        require('STRATA_TRACE' not in env, 'clean curve must use the quiet provider')
        return env

    def route(text, plan):
        rows = plan['read'] - 1
        reuse, vec8 = speed.state.ARMS[plan['arm']]
        value = reference.request_route(text, plan)
        window = value['fast_window']
        require(window['split'] and window['staged'] and window['prefill_slots'] == 128 and
                window['maximum_window_rows'] == rows and window['split_guard_checked'] and
                value['residual_gpu_rows'] == rows and value['residual_host_bytes'] == 0 and
                'strata MoE prefix expert upload layer:' not in text,
                'input length changed the measured placement')
        store = speed.model.restore.require_store(text, dict(direct=False), plan)
        value['attention_store'] = store
        value['attention_input'] = speed.model.input_receipt(text, rows, store, False, trace_layers=False)
        value['kv'] = speed.state.kv.check_clean_request(text, 0, rows, 8192, reuse=reuse)
        value['gather'] = speed.state.gather_counts(text, vec8)
        return value

    def check_work(outcome, row, plan):
        def observed(p):
            return [speed.dispatch.signature(r, s) for r, s in zip(p['routes'], p['samples'], strict=True)]
        n = plan['read']
        work = observed(outcome)
        if n == 117489:
            require(work == row['observer_admission']['expected_work'][:RUNS], '117K anchor work changed')
        if plan['index'] >= len(FIRST):
            require(work == observed(row['processes'][FIRST.index(n)]), 'repeat changed workload or expert work')

    def save(row):
        if row['stage'] == 'prepared':
            for plan in row['plans']:
                plan['arm'] = row['observer_admission']['curve_arm']
            row.update(whole_bytes=len(ORDER) * row['budgets']['total_cap'] + (8 << 20),
                plan=dict(order=ORDER, requests=RUNS, arm=row['observer_admission']['curve_arm'],
                    analysis='Fit first and subsequent prompt time and client TTFT separately on six lengths. '
                             'Compare affine, quadratic and n log n errors with one length left out. '
                             'Report the signed 98305/98306 boundary difference separately in each order.',
                    stop='Stop first source, route, work, clock, owner, fault or budget failure. No speed exclusions.'))
        old_save(row)

    def current(row):
        require(row['plan']['order'] == list(ORDER) and row['plan']['requests'] == RUNS and
                [(p['read'], p['arm']) for p in row['plans']] ==
                [(n, row['observer_admission']['curve_arm']) for n in ORDER], 'changed curve order or selector')
        return old_current(row)

    ctx.pins, ctx.environment, ctx.request_route = pins, environment, route
    ctx.save, ctx.current, ctx.after_execution = save, current, check_work
    return ctx


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(sys.executable == basis.reference.model.PYTHON and
            (action in ('prepare', 'host') or action in map(str, range(len(ORDER)))), 'invalid curve stage')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        ctx = experiment()
        if action == 'prepare':
            ctx.prepare()
        elif action == 'host':
            basis.comparison.host(ctx)
        else:
            basis.comparison.run(int(action), ctx)
