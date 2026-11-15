#!/usr/bin/env python3
"""Admit centering0/1 through bench.sh with actual phase/dispatch receipts.

Main-authored, repository LICENSE. These thinking-on32K short-output checks
qualify routes and lifecycle, not sustained decode speed or model adoption.
"""
import fcntl
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import q6_centering_checks as checks
import verify_window_model_entry as entry
from iq3_model_entry import dispatches

P, B, RESULT, base = checks.P, checks.B, checks.RESULT, checks.base
SELF = Path(__file__).resolve()
PREFIX = 'strata verify centering:'
require, digest = checks.require, entry.digest


def proof():
    gpu = checks.device_context()
    row = json.loads((checks.DEVICE_ROOT / 'record.json').read_text())
    gpu.unchanged(row)
    native = gpu.built()
    require(row['stage'] == 'closed' and row['passed'] and row['complete']
            and row['ownership_closed'] and not row['active'] and row['GPU_executed']
            and row['fresh_faults'] == [] and row['boot_unchanged']
            and [(r['name'], r['native_exit'], r['checks']) for r in row['results']] ==
                [('submission', 0, 10), ('phase-cache', 0, 16)],
            'fresh same-build host and small traced device gates required')
    providers = json.loads(Path(row['runtime_manifest']['path']).read_text())
    prior = base.committed(RESULT)['cache_map_model_entry_zero_v1']
    require(prior['passed'] and prior['complete'] and prior['ownership_closed']
            and not prior['active'] and prior['binary']['sha256'] ==
                '552b1616d21ad1b9aa6a317cb940d375c003246b896a255a935e09292804c96a',
            'retain the qualified fallback model environment')
    driver = prior['environment']['ZE_ENABLE_ALT_DRIVERS']
    require(providers.get(driver) == digest(driver), 'qualified model ALT provider required')
    native = dict(native, environment=native['recipe']['environment'],
                  boot=row['boot'], driver=row['driver'])
    return native, dict(source_commit=row['source_commit'],
        environment=dict(row['environment'], ZE_ENABLE_ALT_DRIVERS=driver)), providers


def pinned_sources(revision):
    pins = entry.pinned_sources(revision)
    pins.update(base.controls(revision, controller=SELF))
    return pins


def captures(text, arm):
    original = entry.captures(text, 'columns')
    centered = []
    pattern = (r'strata verify centering: phase=(prompt|decode) tokens=([1-4]) '
               r'enabled=([01]) centered_calls=(\d+) columns_calls=(\d+)')
    for line in text.splitlines():
        if not line.startswith(PREFIX):
            continue
        match = re.fullmatch(pattern, line)
        require(match is not None, 'malformed centering capture receipt')
        phase, tokens, enabled, selected, columns = match.groups()
        sample = dict(phase=phase, tokens=int(tokens), enabled=int(enabled),
                      centered=int(selected), columns=int(columns))
        require(sample['enabled'] == arm and
                0 <= sample['centered'] <= sample['columns'] < 2**64 and
                (not sample['centered'] or (arm == 1 and phase == 'decode')),
                'centering selector/count/prompt exclusion mismatch')
        centered.append(sample)
    require(len(original) == len(centered) and all(
        (old['phase'], old['tokens'], old['columns']) ==
        (new['phase'], new['tokens'], new['columns'])
        for old, new in zip(original, centered)), 'join every centering receipt to its phase capture')
    decoded = [r for r in centered if r['phase'] == 'decode']
    require(decoded and (arm == 0 or any(r['centered'] > 0 for r in decoded)),
            'missing actual selected decode centering')
    return centered


def parser_controls():
    def sample(arm):
        return ('strata verify: phase=prompt tokens=4 q6_reference_calls=2 q6_columns_calls=0\n'
            f'strata verify centering: phase=prompt tokens=4 enabled={arm} centered_calls=0 columns_calls=0\n'
            'strata verify: phase=decode tokens=4 q6_reference_calls=1 q6_columns_calls=2\n'
            f'strata verify centering: phase=decode tokens=4 enabled={arm} centered_calls={arm} columns_calls=2\n')
    for arm in (0, 1): captures(sample(arm), arm)
    good = sample(1)
    bad = [(good, 0), (sample(0).replace('centered_calls=0 columns_calls=2',
            'centered_calls=1 columns_calls=2'), 0),
        (good.replace('centered_calls=0 columns_calls=0', 'centered_calls=1 columns_calls=0'), 1),
        (good.replace('centered_calls=1 columns_calls=2', 'centered_calls=3 columns_calls=2'), 1),
        (good.replace('centered_calls=1 columns_calls=2', 'centered_calls=0 columns_calls=2'), 1),
        (good.replace('centered_calls=1 columns_calls=2', 'centered_calls=1 columns_calls=3'), 1),
        (good.replace('tokens=4 enabled=1 centered_calls=1', 'tokens=8 enabled=1 centered_calls=1'), 1),
        (good.replace('centered_calls=1 columns_calls=2', 'centered_calls=-1 columns_calls=2'), 1),
        (good.replace('centered_calls=1 columns_calls=2', f'centered_calls={2**64} columns_calls={2**64}'), 1),
        ('\n'.join(x for x in good.splitlines() if not x.startswith(PREFIX)), 1),
        (good + 'strata verify centering: invalid\n', 1)]
    for text, arm in bad:
        try: captures(text, arm)
        except RuntimeError: continue
        raise RuntimeError('invalid centering/phase dispatch receipt accepted')
    return dict(passed=True, positive=2, negative=len(bad), GPU_executed=False)


def context(arm):
    require(arm in (0, 1), 'choose centering0 or1')
    ctx = entry.context('columns')
    ctx.ROOT = checks.model.ROOT / ('model-entry-centering' + str(arm) + '-v1')
    ctx.KEY = 'q6_centering_model_entry_arm' + str(arm) + '_v1'
    ctx.proof, ctx.pinned_sources = proof, pinned_sources
    original_current, original_route = ctx.current, ctx.profile_route

    def current(row):
        env, args = row['environment'], row['configuration']['args']
        require(env['STRATA_SYCL_DECODE_Q6_CENTERING'] == str(arm)
            and env['STRATA_SYCL_DECODE_Q6_COLUMNS'] == env['STRATA_IQ3_PAIR'] ==
                env['STRATA_IQ3_PAIR_AUDIT'] == env['STRATA_SYCL_ADAPT_SYNC'] == '1'
            and env['STRATA_IQ_SINGLE_EXACT'] == env['STRATA_CPU_SERVICE'] ==
                env['STRATA_CPU_WORK_CENSUS'] == env['STRATA_DECODE_TIMING'] == '0'
            and args[args.index('--adapt-swaps') + 1] == '0'
            and args[args.index('--expert-cache') + 1] == '1256'
            and args[args.index('--spec') + 1] == '4'
            and row['read_tokens'] == 32769 and row['runs'] == 2,
            'freeze actual engine/configuration and immutable centering-only arm')
        return original_current(row)

    def route(text):
        result = original_route(text)
        result['centering_captures'] = captures(text, arm)
        rows = dispatches(text, 1, positive=True)
        require(len(rows) == 1, 'one actual IQ3 footer per long request')
        full = dispatches((ctx.ROOT / 'engine.stderr').read_text(), 1, positive=False)
        require([r['request'] for r in full] == [1, 2, 3]
                and rows[0]['request'] in (2, 3), 'bind actual dispatch to warmup and both requests')
        result['IQ3_dispatch'] = rows[0]
        return result

    ctx.current, ctx.profile_route = current, route
    return ctx


def prepare(ctx, arm):
    controls = parser_controls()
    ctx.prepare()
    row = json.loads((ctx.ROOT / 'record.json').read_text())
    row['environment'].update(STRATA_SYCL_DECODE_Q6_CENTERING=str(arm),
        STRATA_SYCL_DECODE_Q6_COLUMNS='1', STRATA_SYCL_ADAPT_SYNC='1',
        STRATA_IQ3_PAIR='1', STRATA_IQ3_PAIR_AUDIT='1', STRATA_IQ_SINGLE_EXACT='0',
        STRATA_CPU_SERVICE='0', STRATA_CPU_WORK_CENSUS='0', STRATA_DECODE_TIMING='0')
    row['centering_arm'], row['centering_parser_controls'] = arm, controls
    row['qualification_basis'] = dict(build_key=checks.model.KEY, host_key=checks.HOST_KEY,
        device_key=checks.DEVICE_KEY, component_commit=checks.model.source.COMPONENT_EVIDENCE,
        policy_commit=checks.model.source.POLICY_EVIDENCE, eligible_shapes=18)
    row['criteria'] += (' Require a joined centering receipt for every actual phase/T capture;'
        'centering0 selects none,centering1 selects positive decode calls,prompt selects none,'
        'selected calls never exceed column calls. Common IQ3_PAIR1/AUDIT1 footer IDs1/2/3'
        'must show eligible and selected rows equal. This diagnostic does not qualify'
        'the unexercised model prompt-verify route or sustained/clean TG.')
    ctx.save(row)
    print('Prepared immutable centering' + str(arm) + '32K route admission;commit before probes.', flush=True)


if __name__ == '__main__':
    require(len(sys.argv) == 3 and sys.argv[1] in ('0', '1')
            and sys.argv[2] in ('prepare', 'host', 'run'), 'use0|1 prepare|host|run')
    arm = int(sys.argv[1]); ctx = context(arm)
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base.guards.no_model_owners([])
        {'prepare': lambda: prepare(ctx, arm), 'host': ctx.host, 'run': ctx.run}[sys.argv[2]]()
