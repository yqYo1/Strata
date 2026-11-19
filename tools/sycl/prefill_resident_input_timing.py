"""Time the input and output bank changes independently on the exact 64K prefill workload."""

import fcntl
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
import prefill_ordinary_bank_compare as ordinary
import prefill_resident_input_model as model

build, basis, require = model.build, model.basis, model.require
READ = ordinary.READ
THIS = 'tools/sycl/prefill_resident_input_timing.py'
BLOCKS = (('copy', 'input', 'output', 'both'), ('input', 'both', 'copy', 'output'),
          ('output', 'copy', 'both', 'input'), ('both', 'output', 'input', 'copy'))
ORDER = tuple(arm for block in BLOCKS for arm in block)


def admission():
    native, speed, identity, fixture = model.inputs()
    state = basis.validation.committed(model.RESULT, (model.KEY,))
    require(state['stage'] == 'closed' and state['passed'] and state['complete'] and state['ownership_closed']
            and not state['active'] and state['binary'] == native['binaries']['strata']
            and state['native_source_commit'] == native['native_source_commit']
            and state['build_sha256'] == identity and state['device_sha256'] == fixture
            and list(state['outcomes']) == list(model.ORDER)
            and all(p['passed'] and p['complete'] and p['ownership_closed'] and not p['fresh_faults']
                    and p['native_exit_code'] == 0 for p in state['outcomes'].values()),
            'same-engine 32K, mixed-host, switch and actual RESTORE required for all four arms')
    prior = build.previous.read(basis.reference.model.ROOT / 'record.json')
    require(prior['models'] == speed['model_assets'] and prior['probe'] == speed['native_validation_probe'],
            'benchmark source model or validation provider differs')
    prior['observer_admission'] = dict(fast_configuration_diagnostic=speed['observer_admission']['fast_configuration_diagnostic'],
        model_state_sha256=build.signature(state), build_sha256=identity, ordinary_control=model.ordinary_control(),
        historical_requests=speed['request_hashes'][str(READ)])
    return (dict(binaries=native['binaries'], native_source_commit=native['native_source_commit']), prior,
            build.previous.read(Path(speed['runtime_manifest']['path'])), None)


def decision(processes):
    require(len(processes) == len(ORDER) and all(p['passed'] and p['complete'] for p in processes),
            'complete all four counterbalanced blocks')
    timings = {arm: [] for arm in model.ARMS}
    ratios = {'input/copy': [], 'output/copy': [], 'both/copy': [], 'both/output': [], 'both/input': []}
    for offset in range(0, len(ORDER), 4):
        block = {}
        for index in range(offset, offset + 4):
            arm = ORDER[index]
            require(processes[index]['index'] == index and arm not in block, 'unexpected process order')
            sample = ordinary.store.stage.schedule.timing(processes[index])
            timings[arm].append(sample)
            block[arm] = sample['warm_prompt_ms']
        for contrast, values in ratios.items():
            candidate, reference = contrast.split('/')
            values.append(block[candidate] / block[reference])
    pp = {arm: [READ * 1000 / s['warm_prompt_ms'] for s in samples] for arm, samples in timings.items()}
    return dict(process_times=timings, warmed_process_PP=pp,
                warmed_median_PP={arm: statistics.median(rates) for arm, rates in pp.items()},
                paired_prompt_time_ratios=ratios,
                faster_in_every_block={contrast: max(values) < 1 for contrast, values in ratios.items()},
                repeated_1000={arm: min(rates) >= 1000 for arm, rates in pp.items()},
                sustained_decode_qualified=False, physical_final_slot_qualified=False, adopted=False)


def experiment():
    ctx = basis.core.source.device.load_module('resident_input_clean_cohort', basis.core.__file__)
    historical = basis.reference.context()
    ctx.THIS, ctx.KEY = THIS, 'prefill_resident_input_timing_v1'
    ctx.ROOT = model.B / 'prefill-resident-input-timing-v1'
    ctx.READS, ctx.ORDER = (READ,) * len(ORDER), [(READ, 1)] * len(ORDER)
    ctx.RUNS, ctx.CLEAN_FAST = 3, True
    ctx.basis, ctx.check_sample = admission, historical.check_sample
    ctx.after_execution, ctx.decision = ordinary.check_work, decision
    base_current, base_save = ctx.current, ctx.save

    def environment(row, plan):
        borrow, direct = model.ARMS[ORDER[plan['index']]]
        env = model.ordinary_environment(row, plan['root'], row['observer_admission']['ordinary_control'])
        env[model.INPUT], env[model.OUTPUT] = str(int(borrow)), str(int(direct))
        return env

    def route(text, plan):
        value = historical.request_route(text, plan)
        borrow, direct = model.ARMS[ORDER[plan['index']]]
        store = model.restore.require_store(text, dict(direct=direct), plan)
        require(store['gpu_rows'] == READ - 1 and store['window_mode'] == 10 and store['host_rows'] == 0
                and 'strata MoE prefix expert upload layer:' not in text, 'unexpected prefill placement')
        value['attention_store'] = store
        value['attention_input'] = model.input_receipt(text, READ - 1, store, borrow, trace_layers=False)
        return value

    def current(row):
        require([p['variant'] for p in row['plans']] == list(ORDER) and row['plan']['blocks'] == list(map(list, BLOCKS))
                and row['read_order'] == list(ctx.READS), 'frozen arms, blocks or read lengths differ')
        return base_current(row)

    def save(row):
        if row['stage'] == 'prepared':
            shared = None
            for i, p in enumerate(row['plans']):
                p['variant'] = ORDER[i]
                env = environment(row, p)
                require(basis.comparison.fingerprint(env) == p['environment_sha256'], 'prepared arm environment differs')
                normalized = {k: v.replace(p['root'], '<case>') for k, v in env.items() if k not in (model.INPUT, model.OUTPUT)}
                if shared is not None:
                    require(normalized == shared, 'unplanned environment difference')
                shared = normalized
            row.update(read_order=list(ctx.READS), whole_bytes=len(ORDER) * row['budgets']['total_cap'] + (8 << 20),
                       plan=dict(blocks=BLOCKS, read=READ, runs=3, unit='Change BF16 bank read and write independently.',
                                 acceptance='Compare second/third prompt-time medians in four complete blocks. '
                                 'Separate input-only, output-only, combined and both incremental contrasts. '
                                 'A contrast requires four ratios below one. Keep TTFT and 1000 PP separate.',
                                 fixed='Same native binary and ordinary384; exact historical bodies, MoE work, '
                                 'draft counts, copy byte totals and placement; 24 tokens with thinking on.',
                                 stop='First workload, route, validation, ownership, fault or budget failure. '
                                 'Retain every eligible time; no speed-based stopping or cohort replacement.'))
        base_save(row)

    ctx.environment, ctx.request_route, ctx.current, ctx.save = environment, route, current, save
    return ctx


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(sys.executable == basis.reference.model.PYTHON and
            (action in ('prepare', 'host') or action in map(str, range(len(ORDER)))), 'invalid benchmark invocation')
    with (model.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        build.builder.guards.no_model_owners([])
        ctx = experiment()
        if action == 'prepare':
            ctx.prepare()
        elif action == 'host':
            basis.comparison.host(ctx)
        else:
            basis.comparison.run(int(action), ctx)
