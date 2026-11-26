"""Localize the matched first-use DFlash2 host excess within a layer pass."""

import math
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
import dflash2_cold_host_build as build
import dflash2_cold_host_measure as coarse

wait, bank, require = coarse.wait, coarse.bank, coarse.require
THIS = str(Path(__file__).resolve().relative_to(bank.P))
TEST = 'bench/test_dflash2_cold_layer_measure.py'
KEY = 'dflash2_cold_layer_phases_v1'
ROOT = bank.B / KEY.replace('_', '-')
BUILD_KEY = 'dflash2_cold_layer_native_v1'
BUILD_ROOT = bank.B / BUILD_KEY.replace('_', '-')
LAYER_SOURCE = '8f379bcaa9dcdd08fcbeb011d8a215ee526b637a'
LAYER_FILES = ('src/prefill/layer_major.inc', 'src/prefill/cold_host_phases.hpp')
ORDER = ('coarse', 'layers', 'layers', 'coarse')
SELECTOR = 'STRATA_PREFILL_COLD_LAYER_PHASES'
PREFIX = 'strata prefill cold layer phases: '
SEGMENTS = ('prepare', 'run', 'retire')
TIMES = ('wall_s', 'cpu_s', 'unclassified_wall_s', 'unclassified_cpu_s')
FAULTS = ('minor_faults', 'major_faults')
PLAN = dict(order=list(ORDER), read=wait.READ, requests_per_process=wait.speed.RUNS,
    scope='One diagnostic executable. Enable V4 and the coarse host-phase ledger in both arms.',
    work='Use the same 118785-token bodies and native work as the prior closed comparison.',
    checks='Demand positive layer geometry, matched markers, nested clock and fault totals, and quiet validation.',
    decision='Require both paired subsequent duration ratios within five percent. Compare first minus '
             'subsequent medians by layer segment and process. Record peak layer locations and count faults.',
    stop='Stop on changed work, invalid accounting, ownership or fresh device failure.')

original_inputs = build.inputs


def prior_measurement():
    row = coarse.wait.basis.validation.committed(bank.RESULT, (coarse.KEY,))
    require(wait.retained.closed(row, 'closed') and row['decision']['warm_observer_gate']
            and row['native_source_commit'] == build.OVERLAY,
            'finish the coarse phase comparison and observer-cost gate before the layer follow-up')
    deltas = row['decision']['first_minus_subsequent']
    require(len(deltas) == 2 and all(
        d['layers']['unclassified_wall_s'] > 0 and
        d['layers']['unclassified_wall_s'] == max(v['unclassified_wall_s'] for v in d.values())
        for d in deltas.values()), 'require repeated excess within the layer loop')
    return row


def layer_inputs(revision):
    prior = prior_measurement()
    fields, payload, original, base = original_inputs(revision)
    source = build.source
    before = dict(source.blobs(bank.P, LAYER_SOURCE + '^', LAYER_FILES))
    updated = dict(source.blobs(bank.P, LAYER_SOURCE, LAYER_FILES))
    require(all(before[p] == payload[p] and updated[p] != before[p] for p in LAYER_FILES),
            'the layer change must apply to the exact coarse observer source')
    payload.update(updated)
    hashes = bank.stage.hashed(payload)
    fields.update(overlay_commit=LAYER_SOURCE, overlays={p: hashes[p] for p in build.OVERLAYS},
                  source_files=hashes, coarse_observation_sha256=bank.signature(prior))
    return fields, payload, original, base


def select_layer_build():
    # The caller is a private CLI process for this measurement series.
    build.KEY, build.ROOT, build.TREE = BUILD_KEY, BUILD_ROOT, BUILD_ROOT / 'source'
    build.CONTROLLERS = (*build.CONTROLLERS, coarse.THIS, wait.THIS, THIS, TEST)
    build.inputs = layer_inputs


def baseline():
    reference = wait.baseline()
    record = bank.previous.read(BUILD_ROOT / 'record.json')
    require(wait.retained.closed(record, 'built') and record['base_binary'] == reference['binary']
            and record['base_build_sha256'] == reference['native_build_sha256']
            and record['native_base_commit'] == reference['native_source_commit']
            and record['overlay_commit'] == LAYER_SOURCE,
            'require the exact isolated layer-detail observer build')
    build.check(record)
    reference['base_physical_context_sha256'] = reference.pop('physical_context_sha256')
    return dict(reference, binary=record['binaries']['strata'], native_source_commit=LAYER_SOURCE,
                native_base_commit=record['native_base_commit'],
                native_overlay_sha256=bank.signature(record['overlays']),
                native_build_sha256=bank.signature(build.builder.compact(record)), native_context_qualified=False)


def environment(row, plan):
    return dict(coarse.environment(row, plan), **{
        coarse.SELECTOR: '1', SELECTOR: str(int(plan['arm'] == 'layers')),
    })


def layer_accounting(lines, enclosing, count, *, enabled):
    if not enabled:
        require(not lines, 'unexpected layer-detail observer')
        return None
    require(len(lines) == 1 and count > 0, 'missing, duplicate or unbounded layer-detail observer')
    pairs = [field.split('=', 1) for field in lines[0].split()]
    require(all(len(pair) == 2 for pair in pairs), 'malformed layer-detail observer')
    fields = dict(pairs)
    expected = {'version', 'available', 'layers'}
    expected.update(f'{name}_{k}' for name in SEGMENTS
                    for k in (*TIMES, *FAULTS, 'peak_unclassified_wall_s', 'peak_layer'))
    require(len(fields) == len(pairs) and fields.keys() == expected and
            (fields['version'], fields['available'], fields['layers']) == ('1', '1', str(count)),
            'incomplete layer-detail partition')
    segments = {}
    for n in SEGMENTS:
        values = {k: float(fields[f'{n}_{k}']) for k in (*TIMES, 'peak_unclassified_wall_s')}
        integers = {k: fields[f'{n}_{k}'] for k in (*FAULTS, 'peak_layer')}
        require(all(math.isfinite(v) and v >= 0 for v in values.values())
                and all(v.isascii() and v.isdecimal() for v in integers.values()), 'invalid layer-detail counters')
        segment = dict(values, **{k: int(v) for k, v in integers.items()})
        require(segment['peak_layer'] < count
                and segment['peak_unclassified_wall_s'] <= segment['unclassified_wall_s']
                and all(segment[f'unclassified_{k}'] <= segment[k] for k in ('cpu_s', 'wall_s')),
                'layer-detail peak or unclassified time exceeds its extent')
        segments[n] = segment
    for metric in (*TIMES, *FAULTS):
        tolerance = coarse.TOLERANCE_S if metric in TIMES else 0
        require(sum(p[metric] for p in segments.values()) <= enclosing[metric] + tolerance,
                'layer-detail counters exceed their enclosing interval')
    return segments


def route(text, plan):
    frame = coarse.route(text, dict(plan, arm='phases'))
    lines = [line.removeprefix(PREFIX) for line in text.splitlines() if line.startswith(PREFIX)]
    details = layer_accounting(lines, frame['cold_host']['layers'], int(frame['work']['windows']),
                               enabled=plan['arm'] == 'layers')
    if details is not None:
        frame['cold_layers'] = details
    return frame


def after_execution(outcome, row, plan):
    expected = plan['arm'] == 'layers'
    require(all(('cold_layers' in frame) is expected for frame in outcome['routes']),
            'unexpected layer-detail observer state')
    coarse_routes = [{k: v for k, v in frame.items() if k != 'cold_layers'} for frame in outcome['routes']]
    coarse.after_execution(dict(outcome, routes=coarse_routes), row, dict(plan, arm='phases'))
    outcome.update(arm=plan['arm'], clean_speed=False)


def decision(processes):
    value = wait.decision(processes, ORDER)
    value['layer_detail_over_coarse'] = value.pop('observed_over_clean')
    differences = {}
    for proc in processes:
        if proc['arm'] == 'layers':
            first, *later = [frame['cold_layers'] for frame in proc['routes']]
            differences[str(proc['index'])] = {
                n: {k: first[n][k] - statistics.median(v[n][k] for v in later)
                    for k in (*TIMES, *FAULTS)} for n in SEGMENTS}
    return dict(value, first_minus_subsequent=differences)


def experiment():
    ctx = coarse.experiment()
    ctx.ROOT, ctx.KEY, ctx.THIS, ctx.ORDER = ROOT, KEY, THIS, ORDER
    ctx.baseline, ctx.protocol, ctx.environment = baseline, PLAN, environment
    ctx.selector, ctx.extra_sources = SELECTOR, (*ctx.extra_sources, THIS, TEST)
    ctx.request_route, ctx.after_execution, ctx.decision = route, after_execution, decision
    return ctx


if __name__ == '__main__':
    action, = sys.argv[1:]
    select_layer_build()
    if action.startswith('build-'):
        build.main(action.removeprefix('build-'))
    else:
        wait.speed.clock.main(experiment())
