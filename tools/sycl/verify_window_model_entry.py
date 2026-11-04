#!/usr/bin/env python3
"""Bounded phase/Q6 model entry through this fork's bench.sh; repository LICENSE.

Reuse the maintained owned frontend/model supervisor in a private context. Both
arms use the same new engine; only the immutable decode-column selector differs.
These traced 32K checks qualify routes and handoff, not sustained decode speed.
"""
import fcntl
import importlib.util
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import check_verify_window_phase as device
import check_prefill_strategy_model as template

build, base = device.build, device.base
P, B, RESULT = device.P, device.B, device.RESULT
SELF = Path(__file__).resolve()
require, digest, write = device.require, device.digest, device.write
ARMS = {'reference': '0', 'columns': '1'}


def proof():
    row = json.loads((device.ROOT / 'record.json').read_text())
    device.unchanged(row)
    native = device.built()
    require(row['stage'] == 'closed' and row['passed'] and row['complete'] and
            row['ownership_closed'] and not row['active'] and row['GPU_executed'] and
            row['fresh_faults'] == [] and row['boot_unchanged'] and
            [(r['name'], r['native_exit'], r['checks']) for r in row['results']] ==
            [('submission', 0, 10), ('phase-cache', 0, 16)],
            'closed same-build small GPU qualification required')
    # The builder verifies the exact composed source and actual compiler/link
    # inputs, including the numerical component's two Q6 source files. The
    # synthetic phase fixture supplies no arithmetic/model qualification.
    native = dict(native, environment=native['recipe']['environment'],
                  boot=row['boot'], driver=row['driver'])
    providers = json.loads(Path(row['runtime_manifest']['path']).read_text())
    prior = base.committed(RESULT)['cache_map_model_entry_zero_v1']
    require(prior['passed'] and prior['complete'] and prior['ownership_closed'] and
            not prior['active'] and prior['binary'] == native['selected_binaries']['strata'],
            'closed selected-engine model environment required')
    driver = prior['environment']['ZE_ENABLE_ALT_DRIVERS']
    require(providers.get(driver) == digest(driver),
            'selected model driver must belong to the current GPU-qualified provider inventory')
    environment = dict(row['environment'], ZE_ENABLE_ALT_DRIVERS=driver)
    return native, dict(source_commit=row['source_commit'], environment=environment), providers


def pinned_sources(revision):
    pins = template.pinned_sources(revision)
    pins.update(base.controls(revision, controller=SELF))
    return pins


def captures(text, arm):
    require(arm in ARMS, 'unknown Q6 arm')
    result = []
    prefix = 'strata verify: phase='
    pattern = (r'strata verify: phase=(prompt|decode) tokens=([1-4]) '
               r'q6_reference_calls=(\d+) q6_columns_calls=(\d+)')
    for line in text.splitlines():
        if not line.startswith(prefix):
            continue
        match = re.fullmatch(pattern, line)
        require(match is not None, 'malformed/out-of-plan phase capture')
        phase, tokens, reference, columns = match.groups()
        sample = dict(phase=phase, tokens=int(tokens), reference=int(reference), columns=int(columns))
        require(not sample['columns'] or (phase == 'decode' and arm == 'columns'),
                'Q6 columns leaked into prompt/reference')
        result.append(sample)
    decoded = [r for r in result if r['phase'] == 'decode']
    require(decoded and all(r['reference'] + r['columns'] > 0 for r in decoded),
            'missing actual decoded Q6 capture')
    if arm == 'columns':
        require(any(r['columns'] > 0 for r in decoded), 'candidate never selected Q6 columns')
    return result


def parser_check():
    prompt = 'strata verify: phase=prompt tokens=4 q6_reference_calls=2 q6_columns_calls=0\n'
    reference = 'strata verify: phase=decode tokens=4 q6_reference_calls=2 q6_columns_calls=0\n'
    columns = 'strata verify: phase=decode tokens=4 q6_reference_calls=1 q6_columns_calls=1\n'
    captures(prompt + reference, 'reference')
    captures(prompt + columns, 'columns')
    bad = [(prompt + columns, 'reference'),
           (prompt.replace('q6_columns_calls=0', 'q6_columns_calls=1') + columns, 'columns'),
           (prompt + reference, 'columns'), (prompt, 'reference'),
           (prompt + reference.replace('tokens=4', 'tokens=8'), 'reference'),
           (prompt + reference + 'strata verify: phase=invalid tokens=1 q6_reference_calls=2 q6_columns_calls=0', 'reference')]
    for text, arm in bad:
        try:
            captures(text, arm)
        except RuntimeError:
            continue
        raise RuntimeError('invalid phase/selection evidence accepted')
    return dict(passed=True, positive=2, negative=len(bad), GPU_executed=False)


def context(arm):
    require(arm in ARMS, 'choose reference|columns')
    spec = importlib.util.spec_from_file_location('strata_phase_q6_entry_context', template.__file__)
    ctx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ctx)
    ctx.configure('adaptive-throughput')
    ctx.ROOT = build.ROOT / ('model-entry-' + arm + '-v1')
    ctx.KEY = 'verify_window_model_entry_' + arm + '_v1'
    ctx.proof, ctx.pinned_sources = proof, pinned_sources
    original_current, original_route = ctx.current, ctx.profile_route

    def current(row):
        require(row['environment'].get('STRATA_SYCL_DECODE_Q6_COLUMNS') == ARMS[arm] and
                row['environment'].get('STRATA_SYCL_ADAPT_SYNC') == '1' and
                row['configuration']['args'][row['configuration']['args'].index('--spec') + 1] == '4' and
                row['configuration']['args'][row['configuration']['args'].index('--short-read') + 1] == '0',
                'freeze Q6 arm,checked adaptation,MTP4 and selected short-read0 before startup')
        return original_current(row)

    def route(text):
        result = original_route(text)
        result['Q6_request_captures'] = captures(text, arm)
        # The selected short-read0 configuration bypasses prompt verify windows.
        # Check any that are observed without claiming this route was exercised.
        # The complete bounded log also includes the benchmark's short warmup.
        full = (ctx.ROOT / 'engine.stderr').read_text()
        all_captures = captures(full, arm)
        result['Q6_prompt_window_observed'] = any(r['phase'] == 'prompt' for r in all_captures)
        return result

    ctx.current, ctx.profile_route = current, route
    return ctx


def prepare(ctx, arm):
    controls = parser_check()
    ctx.prepare()
    row = json.loads((ctx.ROOT / 'record.json').read_text())
    row['environment'].update(STRATA_SYCL_DECODE_Q6_COLUMNS=ARMS[arm], STRATA_SYCL_ADAPT_SYNC='1')
    row['native_component_source'] = build.phase.Q6
    row['qualification_basis'] = dict(build_result_key=build.KEY, device_result_key=device.KEY,
        Q6_component_result_key='q6_byte_load_component_v2', qualified_shapes=30,
        unchanged_selected_reference='479a6fdac8cbaa35a9bf3a41d2b6e23be5260e54',
        driver_environment_basis='Closed cache_map_model_entry_zero_v1 ALT driver;exact current GPU-qualified provider hash required.',
        limits='Only the eight declared phase/Q6 source differences. Same-build small payload lifecycle and standalone Q6 numerics qualify entry,not full model arithmetic or adoption. No unchanged-fixture ELF claim.')
    row['Q6_arm'] = arm
    row['Q6_parser_controls'] = controls
    row['criteria'] = (
        'Two frozen current-source Python frontend probes,then two fresh32769-input bench.sh requests '
        'with thinking enabled,zero prefix reuse and1..24outputs. Keep default5workers,1256decode slots,'
        'spec4,8192prefill chunk,throughput,GDN16/MTP16 and immutable runtime safety. '
        'Keep selected short-read0;this entry does not qualify the model prompt-verify route. '
        'Any observed prompt graph must use reference Q6;require per-long-request decoded Q6 captures with only '
        'the declared arm selected,actual GDN/native16/MTP/dense/head restore markers,valid counts/clocks,'
        'normal streams/native exit,quiet final parameter validation,unchanged inputs/providers and no fresh fault. '
        'Stop on the first failure;no retry,hash refresh or post-result criterion change. '
        'This is traced model-entry/handoff qualification,not a clean paired speed comparison,sustained TG,'
        'task quality,ordinary provider teardown,physical262144 or RESTORE qualification.')
    row['retention'] = (
        'Main-owned bounded32K entry;64MiB maximum raw output per native owner. Keep compact samples,'
        'actual route/capture evidence and original failures. Retire successful logs/JIT after source '
        'recovery and outcome commit. Preserve selected479a and this one candidate only for the imminent '
        'same-binary selector and selected-engine long/agent/thinking-on strong/weak-MTP comparisons.')
    ctx.save(row)
    print('Prepared fixed-Q6 ' + arm + ' model entry;commit before frontend probes.', flush=True)


if __name__ == '__main__':
    require(len(sys.argv) == 3 and sys.argv[1] in ARMS and sys.argv[2] in ('prepare', 'host', 'run'),
            'use reference|columns prepare|host|run')
    ctx = context(sys.argv[1])
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base.guards.no_model_owners([])
        if sys.argv[2] == 'prepare':
            prepare(ctx, sys.argv[1])
        elif sys.argv[2] == 'host':
            ctx.host()
        else:
            ctx.run()
