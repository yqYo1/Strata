"""Attribute first-use prefill time inside the measured hot layer's execution."""

import math
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
import dflash2_cold_layer_measure as layer

bank, build, wait, require = layer.bank, layer.build, layer.wait, layer.require
THIS = str(Path(__file__).resolve().relative_to(bank.P))
TEST = 'tests/sycl/prefill_cold_host_phases.cpp'
KEY = 'dflash2_cold_layer_windows_v1'
ROOT = bank.B / KEY.replace('_', '-')
BUILD_KEY = 'dflash2_cold_layer_windows_native_v1'
BUILD_ROOT = bank.B / BUILD_KEY.replace('_', '-')
SOURCE = '8f1f62a25a1f94e05aacf8d53a6242706e595e31'
PRIOR = '200ec7292232630cc73131ca4a7483a7bbaef2e2'
SELECTOR = 'STRATA_PREFILL_COLD_LAYER_WINDOWS'
PREFIX = 'strata prefill cold layer windows: '
ORDER = ('layers', 'windows', 'windows', 'layers')
SEGMENTS = ('run', 'moe', 'handoff')
METRICS = (*layer.TIMES, *layer.FAULTS)
PLAN = dict(order=list(ORDER), read=wait.READ, requests_per_process=wait.speed.RUNS,
            scope='Same diagnostic engine, with layer accounting in both arms. Change only the selected layer '
                  'window observer before exec. Compare four independent processes with the same native work.',
            accounting='Cover the single full-extent mode16 window at layer index 1. Keep MoE, handoff and their complement within '
                       'the enclosing run interval. Record caller wall, CPU and faults with no added queue drain.',
            decision='Require both paired subsequent-duration ratios within five percent. Measure first minus '
                     'subsequent by segment in both windows processes. First-use ratios remain descriptive.',
            stop='Stop on changed work, incomplete accounting, invalid owner exit or new device fault.')


def prior():
    outcome = bank.previous.historic(PRIOR, layer.KEY)
    record = bank.previous.read(layer.BUILD_ROOT / 'record.json')
    require(wait.retained.closed(outcome, 'closed') and wait.retained.closed(record, 'built')
            and outcome['native_build_sha256'] == bank.signature(build.builder.compact(record))
            and outcome['native_source_commit'] == layer.LAYER_SOURCE and outcome['decision']['warm_observer_gate'],
            'require the closed layer-detail cohort and its original build receipt')
    detailed = [p for p in outcome['processes'] if p['arm'] == 'layers']
    require(len(detailed) == 2 and all(p['routes'][0]['cold_layers']['run']['peak_layer'] == 1
                and p['routes'][0]['cold_layers']['run']['peak_unclassified_wall_s'] > 5 for p in detailed),
            'require the repeated first-use excess at layer index 1')
    return outcome, record


def inputs(revision):
    previous, record = prior()
    fields, payload, original, base = layer.layer_inputs(revision)
    require(bank.stage.hashed(payload) == record['source_files'],
            'recover the exact previously measured engine')
    before = dict(build.source.blobs(bank.P, SOURCE + '^', build.OVERLAYS))
    after = dict(build.source.blobs(bank.P, SOURCE, build.OVERLAYS))
    require(all(before[p] == payload[p] and after[p] != before[p] for p in build.OVERLAYS),
            'the window observer must apply to the exact layer-detail preimages')
    payload.update(after)
    hashes = bank.stage.hashed(payload)
    fields.update(overlay_commit=SOURCE, overlays={p: hashes[p] for p in build.OVERLAYS},
                  source_files=hashes, closed_layer_sha256=bank.signature(previous))
    return fields, payload, original, base


def select_build():
    layer.select_layer_build()
    build.KEY, build.ROOT, build.TREE = BUILD_KEY, BUILD_ROOT, BUILD_ROOT / 'source'
    build.CONTROLLERS = (*build.CONTROLLERS, THIS, TEST)
    build.inputs = inputs


def baseline():
    previous, _ = prior()
    reference = wait.baseline()
    native = bank.previous.read(BUILD_ROOT / 'record.json')
    require(wait.retained.closed(native, 'built') and native['base_binary'] == reference['binary']
            and native['base_build_sha256'] == reference['native_build_sha256']
            and native['overlay_commit'] == SOURCE and native['closed_layer_sha256'] == bank.signature(previous),
            'require the exact layer-window observer build')
    build.check(native)
    reference['base_physical_context_sha256'] = reference.pop('physical_context_sha256')
    return dict(reference, binary=native['binaries']['strata'], native_source_commit=SOURCE,
                native_build_sha256=bank.signature(build.builder.compact(native)), native_context_qualified=False)


def environment(row, plan):
    env = layer.environment(row, dict(plan, arm='layers'))
    return dict(env, **{SELECTOR: str(int(plan['arm'] == 'windows'))})


def accounting(text, enclosing, rows, windows, *, enabled):
    lines = [line.removeprefix(PREFIX) for line in text.splitlines() if line.startswith(PREFIX)]
    if not enabled:
        require(not lines, 'unexpected window observer')
        return None
    require(len(lines) == 1 and rows > 0 and windows > 0, 'require one complete layer-window record')
    pairs = [field.split('=', 1) for field in lines[0].split()]
    require(all(len(pair) == 2 for pair in pairs), 'malformed window record')
    fields = dict(pairs)
    actual = dict(version='1', available='1', layer='1', windows=str(windows), rows=str(rows))
    columns = {f'{segment}_{k}' for segment in SEGMENTS for k in METRICS}
    require(len(fields) == len(pairs) and fields.keys() == set(actual) | columns
            and all(fields[k] == v for k, v in actual.items()), 'incomplete layer-window coverage')
    values = {}
    for segment in SEGMENTS:
        times = {k: float(fields[f'{segment}_{k}']) for k in layer.TIMES}
        faults = {k: fields[f'{segment}_{k}'] for k in layer.FAULTS}
        require(all(math.isfinite(v) and v >= 0 for v in times.values())
                and all(v.isascii() and v.isdecimal() for v in faults.values())
                and times['unclassified_wall_s'] <= times['wall_s']
                and times['unclassified_cpu_s'] <= times['cpu_s'], 'invalid layer-window clocks or faults')
        values[segment] = dict(times, **{k: int(v) for k, v in faults.items()})
    other = {}
    for k in METRICS:
        tolerance = layer.coarse.TOLERANCE_S if k in layer.TIMES else 0
        run, moe, handoff = (values[s][k] for s in SEGMENTS)
        require(moe + handoff <= run + tolerance and run <= enclosing[k] + tolerance,
                'window subtotal exceeds its enclosing run')
        other[k] = max(0, run - moe - handoff)
    return dict(values, other=other)


def route(text, plan):
    frame = layer.route(text, dict(plan, arm='layers'))
    rows = plan['read'] - 1
    require(frame['work']['windows'] == '48' and frame['work']['maximum_rows'] == str(rows),
            'require one full-extent window per layer')
    observed = accounting(text, frame['cold_layers']['run'], rows, 1,
                          enabled=plan['arm'] == 'windows')
    if observed is not None:
        frame['cold_windows'] = observed
    return frame


def after_execution(outcome, row, plan):
    enabled = plan['arm'] == 'windows'
    require(all(('cold_windows' in frame) is enabled for frame in outcome['routes']),
            'unexpected window observer state')
    unobserved = [{k: v for k, v in frame.items() if k != 'cold_windows'} for frame in outcome['routes']]
    layer.after_execution(dict(outcome, routes=unobserved), row, dict(plan, arm='layers'))
    outcome.update(arm=plan['arm'], clean_speed=False)


def decision(processes):
    result = wait.decision(processes, ORDER)
    result['windows_over_layers'] = result.pop('observed_over_clean')
    differences = {}
    for p in processes:
        if p['arm'] == 'windows':
            first, *later = [frame['cold_windows'] for frame in p['routes']]
            differences[str(p['index'])] = {
                phase: {k: v - statistics.median(other[phase][k] for other in later)
                        for k, v in first[phase].items()} for phase in first}
    return dict(result, first_minus_subsequent=differences)


def experiment():
    ctx = layer.experiment()
    ctx.ROOT, ctx.KEY, ctx.THIS, ctx.ORDER = ROOT, KEY, THIS, ORDER
    ctx.baseline, ctx.protocol, ctx.environment = baseline, PLAN, environment
    ctx.selector, ctx.extra_sources = SELECTOR, (*ctx.extra_sources, THIS, TEST)
    ctx.request_route, ctx.after_execution, ctx.decision = route, after_execution, decision
    return ctx


if __name__ == '__main__':
    action, = sys.argv[1:]
    select_build()
    if action.startswith('build-'):
        build.main(action.removeprefix('build-'))
    else:
        wait.speed.clock.main(experiment())
