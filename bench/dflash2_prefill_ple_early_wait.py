"""Measure the remaining exposed gather wait after early PLE scheduling."""

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


if __name__ == '__main__':
    wait.speed.clock.main(experiment())
