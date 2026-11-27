"""Measure exposed PLE row-gather waits inside the cold prefill layer."""

from pathlib import Path
import re
import statistics
import sys

sys.dont_write_bytecode = True
import dflash2_cold_window_measure as window

bank, wait, require = window.bank, window.wait, window.require
THIS = str(Path(__file__).resolve().relative_to(bank.P))
KEY = 'dflash2_cold_ple_wait_v1'
ROOT = bank.B / KEY.replace('_', '-')
SELECTOR = 'STRATA_SYCL_PREFILL_PLE_WAIT_PROFILE'
PREFIX = 'strata prefill PLE wait: '
MARKER = re.compile(
    r'position=0 rows=([1-9][0-9]*) mode=direct gathers=([1-9][0-9]*) '
    r'future_get_ms=([0-9]+\.[0-9]{3}) host_ple_ms=([0-9]+\.[0-9]{3})'
)
ORDER = ('ple', 'ple')
PLAN = dict(
    order=list(ORDER), read=wait.READ, requests_per_process=wait.speed.RUNS,
    scope='Two independent processes on the closed layer-window binary. Keep all prior observers active '
          'and enable only the existing PLE wait clock before exec.',
    accounting='Require every Direct I/O gather in the layer-1 slice. Bound the future wait inside the '
               'unclassified remainder and the inclusive PLE clock inside the layer remainder.',
    decision='Compare first and subsequent waits within each process. Check subsequent duration against '
             'the two closed window-observed processes as a historical reference. Use the nested waits '
             'to select prefill scheduling work. Collect clean performance in a separate comparison.',
    stop='Stop on changed work, incomplete accounting, invalid owner exit or new device fault.'
)


def reference():
    row = wait.basis.validation.committed(bank.RESULT, (window.KEY,))
    require(wait.retained.closed(row, 'closed') and row['decision'] == window.decision(row['processes'])
            and row['decision']['warm_observer_gate'], 'require the closed layer-window comparison')
    deltas = row['decision']['first_minus_subsequent']
    require(len(deltas) == 2 and all(
        v['other']['unclassified_wall_s'] > 0
        and v['other']['unclassified_wall_s'] > v['moe']['unclassified_wall_s'] + v['handoff']['unclassified_wall_s']
        for v in deltas.values()), 'require repeated first-use excess outside the MoE window and handoff')
    return row


def baseline():
    evidence = reference()
    base = window.baseline()
    require(all(evidence[k] == base[k] for k in (
        'binary', 'native_source_commit', 'native_build_sha256', 'request_hashes', 'frozen_prefill_work'
    )), 'layer-window source, executable or workload differs')
    return dict(base, window_evidence_sha256=bank.signature(evidence))


def environment(row, plan):
    return dict(window.environment(row, dict(plan, arm='windows')), **{SELECTOR: '1'})


def accounting(text, rows, chunk, enclosing):
    lines = [line.removeprefix(PREFIX) for line in text.splitlines() if line.startswith(PREFIX)]
    require(len(lines) == 1 and rows > 0 and chunk > 0, 'require one PLE interval for the full layer slice')
    fields = MARKER.fullmatch(lines[0])
    require(fields is not None, 'unexpected PLE position, read mode or clock fields')
    extent, gathers, exposed, inclusive = fields.groups()
    require(int(extent) == rows and int(gathers) == (rows + chunk - 1) // chunk,
            'PLE gather coverage differs from the executed pieces')
    exposed, inclusive = float(exposed), float(inclusive)
    # Native clocks are printed to the nearest microsecond.
    require(exposed <= inclusive + 0.001
            and exposed <= enclosing['unclassified_wall_s'] * 1000 + 0.001
            and inclusive <= enclosing['wall_s'] * 1000 + 0.001,
            'PLE clocks exceed their enclosing intervals')
    return dict(gathers=int(gathers), future_get_ms=exposed, host_ple_ms=inclusive)


def route(text, plan):
    frame = window.route(text, dict(plan, arm='windows'))
    frame['ple_wait'] = accounting(text, plan['read'] - 1, int(frame['work']['operator_max']),
                                   frame['cold_windows']['other'])
    return frame


def after_execution(outcome, row, plan):
    require(all('ple_wait' in frame for frame in outcome['routes']), 'missing PLE attribution')
    routes = [{k: v for k, v in frame.items() if k != 'ple_wait'} for frame in outcome['routes']]
    window.after_execution(dict(outcome, routes=routes), row, dict(plan, arm='windows'))
    outcome.update(arm=plan['arm'], clean_speed=False)


def decision(processes):
    controls = [p for p in reference()['processes'] if p['arm'] == 'windows']
    history = wait.decision([controls[0], *processes, controls[1]], ('windows', *ORDER, 'windows'))
    result = {}
    for p in processes:
        first, *later = p['routes']
        ple = first['ple_wait']['future_get_ms'] - statistics.median(
            frame['ple_wait']['future_get_ms'] for frame in later)
        other = first['cold_windows']['other']['unclassified_wall_s'] - statistics.median(
            frame['cold_windows']['other']['unclassified_wall_s'] for frame in later)
        result[str(p['index'])] = dict(exposed_wait_s=ple / 1000, unclassified_remainder_s=other,
                                       unclassified_after_wait_s=other - ple / 1000)
    return dict(observer_over_historical_window=history['observed_over_clean'],
                historical_warm_gate=history['warm_observer_gate'], first_minus_subsequent=result,
                adopted=False)


def experiment():
    ctx = window.experiment()
    ctx.ROOT, ctx.KEY, ctx.THIS, ctx.ORDER = ROOT, KEY, THIS, ORDER
    ctx.baseline, ctx.protocol, ctx.environment = baseline, PLAN, environment
    ctx.selector, ctx.extra_sources = SELECTOR, (*ctx.extra_sources, THIS)
    ctx.request_route, ctx.after_execution, ctx.decision = route, after_execution, decision
    return ctx


if __name__ == '__main__':
    window.select_build()
    wait.speed.clock.main(experiment())
