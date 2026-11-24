"""Measure the separate bank-capacity and input-length changes on DFlash2."""

from pathlib import Path
import statistics
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
import dflash2_prefill_window_bank as admission
import dflash2_prefill_window_speed as window_speed

bank, basis, require = window_speed.bank, window_speed.basis, window_speed.require
window, speed = admission.window, window_speed.speed
READ, LONG_READ = window.READ, admission.LONG_READ
KEY = 'dflash2_prefill_window_bank_speed_v1'
ROOT = bank.B / KEY.replace('_', '-')
THIS = str(Path(__file__).resolve().relative_to(bank.P))
ARMS = {'bank128': (READ, window.SLOTS), 'bank32': (READ, admission.SLOTS),
        'longer': (LONG_READ, admission.SLOTS)}
BLOCKS = (('bank128', 'bank32', 'longer'), ('bank32', 'longer', 'bank128'),
          ('longer', 'bank128', 'bank32'))
ORDER = tuple(arm for block in BLOCKS for arm in block)
COMPARABLE_WORK = ('logical_copy_bytes', 'requested_bank_H2D_bytes', 'uploads', 'jobs',
                   'dequants', 'windows', 'maximum_rows', 'kv', 'tap_vram_bytes', 'request_work')
BUILD_KEYS = ('binary', 'native_source_commit', 'native_build_sha256', 'boot', 'driver', 'runtime_manifest')


def baseline():
    row = window_speed.baseline()
    fixed_key, long_key = admission.RESULT_KEYS[READ], admission.RESULT_KEYS[LONG_READ]
    fixed = basis.validation.committed(bank.RESULT, (fixed_key,))
    longer = basis.validation.committed(bank.RESULT, (long_key,))
    require(fixed == bank.previous.historic(admission.FIXED_BANK_TRACE, fixed_key)
            and fixed['comparison']['matched_work'] and fixed['comparison']['matched_state']
            and longer['plan']['fixed_bank_sha256'] == bank.signature(fixed)
            and longer['comparison'] == dict(matched_short_controls=True, matched_decode_placement=True,
                                            distinct_process=True, prefix_rows_added=LONG_READ - READ)
            and longer['outcome']['first_verifier_window'] == [LONG_READ - 1, 1],
            'require the independently admitted bank and longer-input routes')
    for read, case in ((READ, fixed), (LONG_READ, longer)):
        plan, outcome = case['plan'], case['outcome']
        arm = plan['cases'][admission.CASE]
        require(case['stage'] == 'closed' and case['complete'] and case['passed'] and case['ownership_closed']
                and not case['active'] and outcome['native_exit_code'] == 0 and not outcome['fresh_faults']
                and all(plan[k] == row[k] for k in BUILD_KEYS)
                and plan['validation'] == row['native_validation_probe'] and plan['drafter_assets'] == row['dflash2_assets']
                and [wire['read'] for wire in arm['wire']] == [82, read, 82]
                and arm['budget'] == window.budget(16, slots=admission.SLOTS, rows=read - 1)
                and outcome['route']['windows'] == 48 and outcome['route']['maximum_rows'] == read - 1,
                'capacity receipt is from a different model extent, placement or build')
    return dict(row, bank32_admission_sha256=bank.signature(fixed),
                longer_admission_sha256=bank.signature(longer))


def environment(row, plan):
    env = window_speed.environment(row, dict(plan, arm='window16'))
    require(env[admission.SLOT_OPTION] == str(window.SLOTS) and env[window.store.SELECTOR] == '0',
            'require the original copied-writer policy')
    env[admission.SLOT_OPTION] = str(ARMS[plan['arm']][1])
    return env


def request_route(text, plan):
    read, slots = ARMS[plan['arm']]
    require(read == plan['read'], 'quiet request extent differs')
    return window_speed.request_route(text, dict(plan, arm='window16'), read=read, slots=slots)


def after_execution(outcome, row, plan):
    require(outcome['clean_speed'] and outcome['native_identity'] not in
            [p['native_identity'] for p in row['processes']], 'require a fresh quiet process')
    for route, sample in zip(outcome['routes'], outcome['samples'], strict=True):
        require(route['request_work'] == [plan['read'], 0, plan['read'], sample['generated_tokens'],
                                         sample['drafts_accepted'], sample['drafts_offered']],
                'native and client request work differs')
    comparable = [{key: route[key] for key in COMPARABLE_WORK} for route in outcome['routes']]
    for other in row['processes']:
        if other['read'] == plan['read']:
            require(comparable == [{key: route[key] for key in COMPARABLE_WORK} for route in other['routes']],
                    'same-input work changed across processes or banks')
    outcome['arm'] = plan['arm']


def decision(processes):
    require(tuple(p['arm'] for p in processes) == ORDER, 'expected all counterbalanced blocks')
    times = []
    for process in processes:
        first, *warm = process['samples']
        ms = statistics.median(sample['prompt_ms'] for sample in warm)
        times.append(dict(arm=process['arm'], first_ms=first['prompt_ms'], first_ttft_s=first['ttft_s'],
                          warm_ms=ms, warm_ttft_s=statistics.median(sample['ttft_s'] for sample in warm),
                          warm_pp=process['read'] * 1000 / ms))
    ratios, slopes = [], []
    metric_keys = ('first_ms', 'first_ttft_s', 'warm_ms', 'warm_ttft_s')
    added = LONG_READ - READ
    for start in range(0, len(times), len(ARMS)):
        block = {point['arm']: point for point in times[start:start + len(ARMS)]}
        original, compact, longer = (block[arm] for arm in ARMS)
        ratios.append({key: compact[key] / original[key] for key in metric_keys})
        slopes.append(dict(first_ms_per_token=(longer['first_ms'] - compact['first_ms']) / added,
                           warm_ms_per_token=(longer['warm_ms'] - compact['warm_ms']) / added,
                           first_ttft_ms_per_token=1000 * (longer['first_ttft_s'] - compact['first_ttft_s']) / added,
                           warm_ttft_ms_per_token=1000 * (longer['warm_ttft_s'] - compact['warm_ttft_s']) / added))
    rates = {arm: [point['warm_pp'] for point in times if point['arm'] == arm] for arm in ARMS}
    all_warm_at_target = {
        arm: all(process['read'] * 1000 / sample['prompt_ms'] >= 1000
                 for process in processes if process['arm'] == arm for sample in process['samples'][1:])
        for arm in ARMS
    }
    return dict(process_times=times, bank32_over_bank128=ratios, length_adjacent_slopes=slopes,
                warmed_PP=rates, median_warmed_PP={arm: statistics.median(rates[arm]) for arm in ARMS},
                repeated_bank_nonregression=all(r['warm_ms'] <= 1 for r in ratios),
                repeated_bank_gain=all(r['warm_ms'] < 1 for r in ratios),
                repeated_1000={arm: all(rate >= 1000 for rate in group) for arm, group in rates.items()},
                all_warm_requests_1000=all_warm_at_target,
                full_context_qualified=False, sustained_decode_qualified=False, adopted=False)


def experiment():
    previous = window_speed.experiment()
    ctx = speed.generation.experiment(synchronous_commit=True)
    ctx.KEY, ctx.ROOT, ctx.THIS, ctx.ORDER = KEY, ROOT, THIS, ORDER
    ctx.baseline, ctx.configuration, ctx.environment = baseline, speed.configuration, environment
    ctx.command = lambda plan: speed.command(plan, ARMS[plan['arm']][0])
    ctx.protocol = dict(order=list(ORDER), blocks=[list(block) for block in BLOCKS],
        arms={arm: dict(read=read, bank_slots=slots) for arm, (read, slots) in ARMS.items()},
        requests_per_process=window_speed.RUNS,
        unit='Compare bank slots at 118785 inputs and input extent at bank32. '
             'Hold copied attention, the mode16 scheduler, demand uploads, retained KV and DFlash2 fixed.',
        acceptance='Use three complete blocks, rotating order so every arm occupies every position once. '
                   'Require exact counted bodies and matched work at each length. Keep first-use and subsequent '
                   'PP and client TTFT separate. Keep every eligible process. Report each bank ratio and each '
                   'adjacent length slope. Record 1000 tok/s at the process-median and individual-warm-request levels.',
        stop='Stop on changed source, route, runtime, ownership, budget or fault. This comparison has short output.')
    old_current, old_save = ctx.current, ctx.save

    def pins(revision):
        files = previous.pins(revision)
        for name in (THIS, admission.THIS, admission.MANIFEST):
            require((bank.P / name).read_bytes() == basis.validation.git('show', revision + ':' + name),
                    'changed bank and input-length comparison source: ' + name)
            files[name] = bank.sha(bank.P / name)
        return files

    def save(row):
        if row['stage'] == 'prepared':
            row['request_hashes'] = {}
            for plan in row['plans']:
                plan.update(read=ARMS[plan['arm']][0], heads=window_speed.HEADS)
        old_save(row)

    def current(row):
        fingerprints = []
        for plan, arm in zip(row['plans'], ORDER, strict=True):
            read, slots = ARMS[arm]
            require((plan['arm'], plan['read'], plan['heads']) == (arm, read, window_speed.HEADS),
                    'changed frozen arm or input extent')
            env = environment(row, plan)
            require(env.pop(admission.SLOT_OPTION) == str(slots), 'unexpected bank selector')
            fingerprints.append(bank.signature({k: v.replace(plan['root'], '<process>') for k, v in env.items()}))
        require(len(set(fingerprints)) == 1, 'environment differs outside the bank selector')
        return old_current(row)

    ctx.pins, ctx.current, ctx.save = pins, current, save
    runner = SimpleNamespace(ROOT=ROOT, ORDER=[(ARMS[arm][0], window_speed.HEADS) for arm in ORDER],
        RUNS=window_speed.RUNS, current=current, environment=environment,
        environment_fingerprint=speed.generation.suite.signature, save=save, request_route=request_route,
        after_execution=after_execution, decision=decision, validate_model_text=speed.validate_model_text,
        verify_model_assets=speed.verify_model_assets)
    ctx.run = lambda index: basis.comparison.run(index, runner)
    return ctx


if __name__ == '__main__':
    speed.generation.clock.main(experiment())
