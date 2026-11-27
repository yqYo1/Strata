"""Locate first-use host costs on the isolated DFlash2 prefill build."""

import math
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
import dflash2_cold_host_build as build
import dflash2_primary_prefill_cpu_wait as wait

bank, require = wait.bank, wait.require
KEY = 'dflash2_cold_host_phases_v1'
ROOT = bank.B / KEY.replace('_', '-')
THIS = str(Path(__file__).resolve().relative_to(bank.P))
ORDER = ('wait', 'phases', 'phases', 'wait')
SELECTOR = 'STRATA_PREFILL_COLD_HOST_PHASES'
PREFIX = 'strata prefill cold host phases: '
PHASES = ('admission', 'provision', 'layers', 'retire')
TIMES = ('wall_s', 'cpu_s', 'wait_wall_s', 'wait_cpu_s', 'host_wall_s', 'host_cpu_s',
         'unclassified_wall_s', 'unclassified_cpu_s')
FAULTS = ('minor_faults', 'major_faults')
TOLERANCE_S = 1e-7   # Native prints nine decimals for each independent interval.
PLAN = dict(order=list(ORDER), read=wait.READ, requests_per_process=wait.speed.RUNS,
    scope='One fresh diagnostic executable. Keep V4 CPU waits enabled and switch only the host phase ledger.',
    acceptance='Require identical previous request bodies and native work, all four phases per observed request, '
               'and conservation inside V4. Require both paired subsequent duration ratios within five percent.',
    decision='Record first-to-subsequent deltas independently by observed process and phase. '
             'Retain the first-use run spread and keep provider/backing and machine-wide faults unclaimed.',
    stop='Stop on changed work, incomplete accounting, validation error, device fault or owner failure.')


def baseline():
    reference = wait.baseline()
    record = bank.previous.read(build.ROOT / 'record.json')
    require(wait.retained.closed(record, 'built') and record['base_binary'] == reference['binary']
            and record['base_build_sha256'] == reference['native_build_sha256']
            and record['native_base_commit'] == reference['native_source_commit'],
            'require the exact isolated cold-host observer build')
    build.check(record)
    receipt = build.builder.compact(record)
    reference['base_physical_context_sha256'] = reference.pop('physical_context_sha256')
    return dict(reference, binary=record['binaries']['strata'],
                native_source_commit=record['overlay_commit'], native_base_commit=record['native_base_commit'],
                native_overlay_sha256=bank.signature(record['overlays']),
                native_build_sha256=bank.signature(receipt), native_context_qualified=False)


def environment(row, plan):
    return dict(wait.curve.environment(row, plan), **{
        wait.SELECTOR: '1', SELECTOR: str(int(plan['arm'] == 'phases')),
    })


def phase_accounting(lines, v4, *, enabled):
    if not enabled:
        require(not lines, 'unexpected cold-host observer')
        return None
    require(len(lines) == 1, 'missing or duplicate cold-host observer')
    pairs = [piece.split('=', 1) for piece in lines[0].split()]
    require(all(len(pair) == 2 for pair in pairs), 'malformed cold-host observer')
    fields = dict(pairs)
    expected = {'version', 'available', 'phases'}
    expected.update(f'{p}_{k}' for p in PHASES for k in (*TIMES, *FAULTS))
    require(len(fields) == len(pairs) and fields.keys() == expected and
            (fields['version'], fields['available'], fields['phases']) == ('1', '1', '4'),
            'incomplete cold-host phase partition')
    phases = {}
    for name in PHASES:
        phase = {k: float(fields[f'{name}_{k}']) for k in TIMES}
        counts = {k: fields[f'{name}_{k}'] for k in FAULTS}
        require(all(math.isfinite(v) and v >= 0 for v in phase.values())
                and all(v.isascii() and v.isdecimal() for v in counts.values()), 'invalid cold-host counters')
        for metric in ('wall_s', 'cpu_s'):
            parts = sum(phase[f'{kind}_{metric}'] for kind in ('wait', 'host', 'unclassified'))
            require(abs(phase[metric] - parts) <= TOLERANCE_S, 'cold-host phase does not conserve')
        phases[name] = dict(phase, **{k: int(v) for k, v in counts.items()})
    for index, metric in enumerate(('cpu_s', 'wall_s')):
        totals = {k: sum(phase[k] for phase in phases.values())
                  for k in (metric, f'wait_{metric}', f'host_{metric}', f'unclassified_{metric}')}
        uncovered = v4[metric] - totals[metric]
        host = sum(v[index] for v in v4['outside_wait'].values())
        require(abs(totals[f'wait_{metric}'] - v4[f'wait_{metric}']) <= TOLERANCE_S
                and abs(totals[f'host_{metric}'] - host) <= TOLERANCE_S
                and uncovered >= -TOLERANCE_S
                and abs(v4['unclassified'][index] - totals[f'unclassified_{metric}'] - uncovered)
                    <= TOLERANCE_S,
                'cold-host phases do not nest in the V4 ledger')
    return phases


def route(text, plan):
    frame = wait.route(text, plan)
    lines = [line.removeprefix(PREFIX) for line in text.splitlines() if line.startswith(PREFIX)]
    phases = phase_accounting(lines, frame['cpu_wait'], enabled=plan['arm'] == 'phases')
    if phases is not None:
        frame['cold_host'] = phases
    return frame


def after_execution(outcome, row, plan):
    observed = plan['arm'] == 'phases'
    routes = outcome['routes']
    require(all(('cold_host' in frame) is observed for frame in routes), 'unexpected phase observer state')
    work = [{k: v for k, v in frame.items() if k != 'cold_host'} for frame in routes]
    wait.after_execution(dict(outcome, routes=work), row, dict(plan, arm='cpu-wait'))
    outcome.update(arm=plan['arm'], clean_speed=False)


def decision(processes):
    pairs = wait.decision(processes, ORDER)
    pairs['phases_over_wait'] = pairs.pop('observed_over_clean')
    relative = {}
    for process in processes:
        if process['arm'] == 'phases':
            first, *later = [frame['cold_host'] for frame in process['routes']]
            relative[str(process['index'])] = {
                phase: {k: first[phase][k] - statistics.median(sample[phase][k] for sample in later)
                        for k in ('wall_s', 'cpu_s', 'unclassified_wall_s', 'unclassified_cpu_s', *FAULTS)}
                for phase in PHASES}
    return dict(pairs, first_minus_subsequent=relative)


def experiment():
    ctx = wait.experiment()
    ctx.ROOT, ctx.KEY, ctx.THIS, ctx.ORDER = ROOT, KEY, THIS, ORDER
    ctx.baseline, ctx.protocol, ctx.environment = baseline, PLAN, environment
    ctx.selector, ctx.extra_sources = SELECTOR, (THIS, build.THIS, 'bench/test_dflash2_primary_prefill_cpu_wait.py')
    ctx.request_route, ctx.after_execution, ctx.decision = route, after_execution, decision
    return ctx


if __name__ == '__main__':
    wait.speed.clock.main(experiment())
