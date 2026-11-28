"""Measure exposed PLE gather wait and screen Linux DirectFile worker count."""

from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
import dflash2_prefill_ple_early_speed as timing

bank, wait, require = timing.bank, timing.wait, timing.require
ple = timing.build.ple
THIS = str(Path(__file__).resolve().relative_to(bank.P))
KEY = 'dflash2_prefill_ple_early_wait_v1'
ROOT = bank.B / KEY.replace('_', '-')
ORDER = ('stock', 'early', 'early', 'stock')
ARMS = ('stock', 'early')
WARM_OBSERVER_RANGE = (0.95, 1.05)
PLAN = dict(order=list(ORDER), read=wait.READ, requests_per_process=wait.speed.RUNS,
    scope='Two opposite-order process pairs on the rebuilt clean-timing executable. Reuse its pinned bodies, '
          'private cache policy and settings. Enable only the PLE wait clock in both arms.',
    accounting='Require one direct I/O marker with all layer-1 gathers, actual first-gather scheduling, '
               'matching native work and the inclusive PLE time bounded by the completed prompt.',
    decision='Report first and subsequent intervals separately. Compare each subsequent median with its '
             'same-arm clean reference. Attribute the wait only if every ratio is in [0.95, 1.05]. '
             'Compare the aggregate exposed wait with the clean first-prefill gap to 1000 tok/s.',
    limits='Relevant to the repeated-text workload. The sum includes every layer-1 future wait and does not '
           'isolate the first chunk or measure physical SSD traffic. First-use observer overhead is unknown. '
           'Collect throughput and qualification evidence separately.',
    stop=timing.PLAN['stop'])
IO_WORKERS = 'STRATA_IO_THREADS'
WORKER_KEY = 'dflash2_prefill_ple_io_workers_v1'
WORKER_ORDER = ('16', '32', '32', '16')
WORKER_PLAN = dict(order=list(WORKER_ORDER), read=wait.READ, requests_per_process=wait.speed.RUNS,
    scope='Keep early issue enabled, observer clocks off and all other settings and requests fixed. '
          'Compare 16 and 32 Linux DirectFile workers in two opposite-order process pairs.',
    eligibility='Require a closed clean timing gain and a closed paired attribution with a positive '
                'first-prefill gap to 1000 tok/s and more exposed first than subsequent wait in both early processes.',
    decision='Require faster first prefill and TTFT in both pairs, and subsequent median time ratios at most one.',
    limits='Measures the whole request on one repeated-text workload. Physical I/O traffic is unmeasured. '
           'The setting also affects decode; sustained decode and model qualification are separate.',
    stop=timing.PLAN['stop'])


def baseline(native_context):
    row = timing.baseline(native_context)
    raw = bank.previous.read(timing.ROOT / 'record.json')
    clean = wait.basis.compact(raw)
    require(wait.retained.closed(raw, 'closed')
            and clean == wait.basis.validation.committed(bank.RESULT, (timing.KEY,))
            and all(raw[k] == v for k, v in row.items())
            and clean['plan'] == timing.PLAN
            and clean['decision'] == timing.decision(clean['processes']),
            'require the closed clean measurement with the same engine, runtime and request bodies')
    for plan in clean['plans']:
        require(bank.signature(timing.environment(raw, plan)) == plan['environment_sha256'],
                'the clean arm must use the same settings before enabling the observer')
    controls = {}
    for arm in ARMS:
        matches = [p for p in clean['processes'] if p['arm'] == arm]
        controls[arm] = dict(
            subsequent_median_ms=statistics.median(p['subsequent_median_ms'] for p in matches),
            first_gap_ms=[max(0, p['samples'][0]['prompt_ms'] - wait.READ) for p in matches])
    return dict(row, clean_timing_sha256=bank.signature(clean), clean_controls=controls)


def environment(row, plan):
    return dict(timing.environment(row, plan), **{ple.SELECTOR: '1'})


def route(text, plan):
    lines = text.splitlines(keepends=True)
    clean = ''.join(line for line in lines if not line.startswith(ple.PREFIX))
    frame = timing.route(clean, plan)
    return dict(frame, ple_wait=ple.gather_accounting(text, plan['read'] - 1, int(frame['work']['operator_max'])))


def after_execution(outcome, row, plan):
    for frame, sample in zip(outcome['routes'], outcome['samples'], strict=True):
        require(frame['ple_wait']['host_ple_ms'] <= sample['prompt_ms'] + 0.001,
                'PLE clock exceeds the enclosing prompt interval')
    bare = [{k: v for k, v in frame.items() if k != 'ple_wait'} for frame in outcome['routes']]
    timing.after_execution(dict(outcome, routes=bare), row, plan)
    control = row['clean_controls'][plan['arm']]
    outcome.update(arm=plan['arm'], clean_speed=False,
                   warm_relative_to_clean=outcome['subsequent_median_ms'] / control['subsequent_median_ms'],
                   clean_first_gap_ms=control['first_gap_ms'])


def decision(processes):
    wait.decision(processes, ORDER)  # Check independent owned process closure.
    remaining = {}
    for arm in ARMS:
        members = [p for p in processes if p['arm'] == arm]
        waits = [[r['ple_wait']['future_get_ms'] for r in p['routes']] for p in members]
        remaining[arm] = dict(first_wait_ms=[values[0] for values in waits],
                              subsequent_wait_ms=[values[1:] for values in waits],
                              clean_first_gap_ms=members[0]['clean_first_gap_ms'])
    ratios = [p['warm_relative_to_clean'] for p in processes]
    low, high = WARM_OBSERVER_RANGE
    return dict(remaining=remaining, warm_relative_to_clean=ratios,
                attribution_eligible=all(low <= ratio <= high for ratio in ratios), adopted=False)


def experiment():
    native = timing.build.context()
    ctx = timing.experiment()
    ctx.ROOT, ctx.KEY, ctx.THIS, ctx.ORDER = ROOT, KEY, THIS, ORDER
    ctx.baseline, ctx.protocol, ctx.environment = lambda: baseline(native), PLAN, environment
    ctx.extra_sources = (*ctx.extra_sources, THIS)
    ctx.request_route, ctx.after_execution, ctx.decision = route, after_execution, decision
    return ctx


def worker_baseline(native):
    row = baseline(native)
    clean = wait.basis.validation.committed(bank.RESULT, (timing.KEY,))
    raw = bank.previous.read(ROOT / 'record.json')
    profiled = wait.basis.compact(raw)
    require(sys.platform.startswith('linux') and clean['decision']['advance']
            and wait.retained.closed(raw, 'closed')
            and profiled == wait.basis.validation.committed(bank.RESULT, (KEY,))
            and all(raw[k] == v for k, v in row.items()) and profiled['plan'] == PLAN
            and profiled['decision'] == decision(profiled['processes'])
            and profiled['decision']['attribution_eligible'],
            'require the closed same-build clean gain and eligible PLE attribution on Linux')
    early = profiled['decision']['remaining']['early']
    extra_wait = [first - statistics.median(later) for first, later in
                  zip(early['first_wait_ms'], early['subsequent_wait_ms'], strict=True)]
    require(all(ms > 0 for ms in extra_wait) and statistics.median(early['clean_first_gap_ms']) > 0,
            'require repeatable extra exposed first-use wait and a remaining first-prefill gap')
    return dict(row, attribution_sha256=bank.signature(profiled), early_extra_wait_ms=extra_wait)


def worker_environment(row, plan):
    env = timing.environment(row, dict(plan, arm='early'))
    require(IO_WORKERS not in env and plan['arm'] in WORKER_ORDER, 'unexpected DirectFile worker setting')
    return dict(env, **{IO_WORKERS: plan['arm']})


def worker_after_execution(outcome, row, plan):
    timing.after_execution(outcome, row, dict(plan, arm='early'))
    outcome['arm'] = plan['arm']


def worker_decision(processes):
    require(all(p['clean_speed'] and len(p['samples']) == wait.speed.RUNS for p in processes),
            'require all clean worker comparisons')
    prompt_ratios = wait.decision(processes, WORKER_ORDER)['observed_over_clean']
    pairs = []
    for x, y, prompt in zip(processes[::2], processes[1::2], prompt_ratios, strict=True):
        control, candidate = (x, y) if x['arm'] == '16' else (y, x)
        a, *later_a = control['samples']
        b, *later_b = candidate['samples']
        pairs.append(dict(first_prefill=prompt['first'], first_ttft=b['ttft_s'] / a['ttft_s'],
                          subsequent_prefill=prompt['subsequent'],
                          subsequent_ttft=statistics.median(s['ttft_s'] for s in later_b)
                          / statistics.median(s['ttft_s'] for s in later_a)))
    first_gain = all(p['first_prefill'] < 1 and p['first_ttft'] < 1 for p in pairs)
    subsequent_preserved = all(statistics.median(p[key] for p in pairs) <= 1
                               for key in ('subsequent_prefill', 'subsequent_ttft'))
    return dict(paired_32_over_16=pairs, repeat_first_gain=first_gain,
                subsequent_preserved=subsequent_preserved, advance=first_gain and subsequent_preserved,
                first_1000={arm: [p['samples'][0]['prompt_ms'] <= wait.READ for p in processes if p['arm'] == arm]
                            for arm in ('16', '32')}, adopted=False)


def worker_experiment():
    native = timing.build.context()
    ctx = timing.experiment()
    ctx.ROOT, ctx.KEY, ctx.THIS, ctx.ORDER = bank.B / WORKER_KEY.replace('_', '-'), WORKER_KEY, THIS, WORKER_ORDER
    ctx.baseline, ctx.protocol, ctx.environment = lambda: worker_baseline(native), WORKER_PLAN, worker_environment
    ctx.selector, ctx.extra_sources = IO_WORKERS, (*ctx.extra_sources, THIS)
    ctx.request_route = lambda text, plan: timing.route(text, dict(plan, arm='early'))
    ctx.after_execution, ctx.decision = worker_after_execution, worker_decision
    return ctx


if __name__ == '__main__':
    if sys.argv[1:2] == ['--io-workers']:
        sys.argv.pop(1)
        wait.speed.clock.main(worker_experiment())
    else:
        wait.speed.clock.main(experiment())
