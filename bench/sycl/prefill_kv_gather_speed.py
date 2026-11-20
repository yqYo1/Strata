"""Time KV prefix reuse and large expert gathers independently on the ordinary prefill route."""

import fcntl
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
import prefill_kv_capacity as capacity

dispatch, state, bank = capacity.dispatch, capacity.state, capacity.bank
model, basis, require = dispatch.model, dispatch.basis, dispatch.require
THIS = str(Path(__file__).resolve().relative_to(bank.P))
BLOCKS = (('base', 'reuse', 'both', 'vec8'), ('reuse', 'vec8', 'base', 'both'),
          ('vec8', 'both', 'reuse', 'base'), ('both', 'base', 'vec8', 'reuse'))
ORDER = tuple(arm for block in BLOCKS for arm in block)
CONTRASTS = ('reuse/base', 'vec8/base', 'both/vec8', 'both/reuse', 'both/base')


def admission(read):
    native, prior, providers, _ = dispatch.admission()
    stage = basis.validation.committed(bank.RESULT, (capacity.KEY,))
    require(stage['stage'] == 'closed' and stage['complete'] and stage['passed']
            and stage['ownership_closed'] and not stage['active']
            and stage['native_source_commit'] == native['native_source_commit']
            and stage['binary'] == native['binaries']['strata_full_context_diagnostic']
            and tuple(stage['outcomes']) == ('host', *capacity.ORDER)
            and all(p['passed'] and p['complete'] and p['ownership_closed'] and not p['fresh_faults']
                    for p in stage['outcomes'].values())
            and all(stage['outcomes'][name]['GPU_executed'] and stage['outcomes'][name]['native_exit_code'] == 0
                    for name in capacity.ORDER)
            and all(stage['outcomes'][name]['final_slot_consumed'] for name in ('full-reload', 'full-reuse')),
            'the same-build standalone final-slot controls must close first')
    traced = basis.validation.committed(bank.RESULT, (dispatch.KEY,))
    require(traced['stage'] == 'closed' and traced['complete'] and traced['passed']
            and traced['ownership_closed'] and not traced['active']
            and traced['binary'] == native['binaries']['strata']
            and traced['native_source_commit'] == native['native_source_commit']
            and traced['decision'] == dict(dispatch_verified=True, requests=16, speed_qualified=False)
            and stage['qualification']['dispatch'] == bank.signature(traced),
            'missing traced official dispatch or capacity qualification for this executable')
    revision, key = dispatch.HISTORY[read]
    parent = bank.previous.historic(revision, key)
    process = parent['processes'][0]
    work = [dispatch.signature(r, s) for r, s in zip(process['routes'], process['samples'], strict=True)]
    require([w['request'] for w in work] == parent['request_hashes'][str(read)]
            and len({w['request'] for w in work}) == len(work) == (3 if read == 65537 else 4)
            and all(w['generated'] == 24 for w in work)
            and work[:dispatch.RUNS] == prior['observer_admission']['parent_work'][str(read)],
            'the exact parent bodies and both traced dispatch bodies are required')
    prior['observer_admission'].update(capacity_sha256=bank.signature(stage), dispatch_sha256=bank.signature(traced),
                                      expected_work=work)
    return native, prior, providers, None


def decision(read, processes):
    timing = dispatch.split.bank.ordinary.store.stage.schedule.timing
    runs = 3 if read == 65537 else 4
    require(len(processes) == len(ORDER) and len({(p['native_identity']['pid'], p['native_identity']['start_ticks'])
            for p in processes}) == len(ORDER), 'need 16 independent processes at this length')
    times = {arm: [] for arm in state.ARMS}
    ratios = {name: [] for name in CONTRASTS}
    for index, (arm, process) in enumerate(zip(ORDER, processes, strict=True)):
        require(process['read'] == read and process['index'] == index and process['clean_speed']
                and process['complete'] and process['passed'] and process['ownership_closed']
                and process['native_exit_code'] == 0 and not process['fresh_faults']
                and len(process['samples']) == runs, 'ineligible or missing clean sample')
        times[arm].append(timing(process))
        if (index + 1) % len(state.ARMS) == 0:
            for name in CONTRASTS:
                a, b = name.split('/')
                ratios[name].append(times[a][-1]['warm_prompt_ms'] / times[b][-1]['warm_prompt_ms'])
    rates = {arm: [1000 * read / t['warm_prompt_ms'] for t in samples] for arm, samples in times.items()}
    return dict(process_times=times, warmed_process_PP=rates,
                warmed_median_PP={arm: statistics.median(v) for arm, v in rates.items()},
                paired_prompt_time_ratios=ratios, contrast_passes={
                    name: sum(t < 1 for t in v) >= 3 and statistics.median(v) < 1 for name, v in ratios.items()},
                repeated_1000={arm: min(v) >= 1000 for arm, v in rates.items()},
                sustained_decode_qualified=False, adopted=False)


def experiment(read):
    ctx = basis.core.source.device.load_module('kv_gather_clean_speed', basis.core.__file__)
    reference = basis.reference.context()
    ctx.THIS, ctx.KEY, ctx.ROOT = THIS, f'prefill_kv_gather_speed_{read}_v1', bank.B / f'prefill-kv-gather-speed-{read}-v1'
    ctx.READS, ctx.ORDER = (read,) * len(ORDER), [(read, 1)] * len(ORDER)
    ctx.RUNS, ctx.CLEAN_FAST = (3 if read == 65537 else 4), True
    ctx.basis, ctx.check_sample = lambda: admission(read), reference.check_sample
    ctx.decision = lambda processes: decision(read, processes)
    old_save, old_current, old_pins = ctx.save, ctx.current, ctx.pins

    def pins(revision):
        values = old_pins(revision)
        require({THIS, capacity.THIS, dispatch.THIS, dispatch.split.THIS, dispatch.split.bank.THIS,
                 *state.SCRIPTS} <= values.keys(), 'timing or qualification controller missing from Git pins')
        return values

    def environment(row, plan):
        env = model.ordinary_environment(row, plan['root'], row['observer_admission']['ordinary_control'])
        reuse, vec8 = state.ARMS[ORDER[plan['index']]]
        env[model.INPUT] = env[model.OUTPUT] = '0'
        env[state.KV], env[state.GATHER] = str(int(reuse)), str(int(vec8))
        require('STRATA_TRACE' not in env, 'use a clean native process')
        return env

    def route(text, plan):
        reuse, vec8 = state.ARMS[ORDER[plan['index']]]
        value = reference.request_route(text, plan)
        store = model.restore.require_store(text, dict(direct=False), plan)
        rows = read - 1
        require(store['gpu_rows'] == rows and store['host_rows'] == 0 and store['window_mode'] == 10
                and 'strata MoE prefix expert upload layer:' not in text, 'prefill placement changed')
        value['attention_store'] = store
        value['attention_input'] = model.input_receipt(text, rows, store, False, trace_layers=False)
        value['kv'] = state.kv.check_clean_request(text, 0, rows, model.restore.capacity.CHUNK, reuse=reuse)
        value['gather'] = state.gather_counts(text, vec8)
        if read == dispatch.split.READ:
            value['admission'] = dispatch.split.check_split_budget(text)
        return value

    def check_work(outcome, row, plan):
        work = [dispatch.signature(r, s) for r, s in zip(outcome['routes'], outcome['samples'], strict=True)]
        require(outcome['clean_speed'] and outcome['index'] == plan['index']
                and work == row['observer_admission']['expected_work'], 'work differs from the pinned parent')

    def save(row):
        if row['stage'] == 'prepared':
            for plan, arm in zip(row['plans'], ORDER, strict=True):
                plan['arm'] = arm
            row.update(whole_bytes=len(ORDER) * row['budgets']['total_cap'] + (8 << 20),
                       plan=dict(blocks=BLOCKS, requests=ctx.RUNS, contrasts=CONTRASTS,
                           acceptance='For each contrast at this length: at least three faster warmed prompt '
                                      'times in four matched blocks and a paired median time ratio below one.',
                           scope='First use and subsequent median PP and client TTFT separately. Exact original '
                                 'bodies, expert work, output count, MTP, KV and gather dispatch for each request.',
                           stop='First workload, route, owner, fault, runtime or budget failure. No speed-based '
                                'stopping or exclusion. Sustained decode and remaining state gates are separate.'))
        old_save(row)

    def current(row):
        require(row['plan']['blocks'] == list(map(list, BLOCKS)) and row['plan']['requests'] == ctx.RUNS
                and row['plan']['contrasts'] == list(CONTRASTS) and
                [(p['index'], p['arm'], p['read']) for p in row['plans']] ==
                [(i, arm, read) for i, arm in enumerate(ORDER)], 'frozen order or workload differs')
        return old_current(row)

    ctx.pins, ctx.environment, ctx.request_route = pins, environment, route
    ctx.save, ctx.current, ctx.after_execution = save, current, check_work
    return ctx


if __name__ == '__main__':
    length, action = sys.argv[1:]
    read = int(length)
    require(sys.executable == basis.reference.model.PYTHON and read in dispatch.HISTORY
            and (action in ('prepare', 'host') or action in map(str, range(len(ORDER)))), 'invalid length or stage')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        ctx = experiment(read)
        if action == 'prepare':
            ctx.prepare()
        elif action == 'host':
            basis.comparison.host(ctx)
        else:
            basis.comparison.run(int(action), ctx)
