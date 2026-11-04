#!/usr/bin/env python3
"""Same-build IQ3 selector admission through the maintained bench.sh owner.

Repository LICENSE. Keep Q6 columns and all placement/runtime settings fixed;
only the immutable decode IQ3 selector differs. This is not a speed comparison.
"""
import fcntl
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import iq3_device as device
import verify_window_model_entry as entry

P, B, RESULT, base = device.P, device.B, device.RESULT, device.base
SELF = Path(__file__).resolve()
require = device.require
PREFIX = 'strata IQ3_GU_DELTA '


def proof():
    gpu = device.context()
    row = json.loads((device.ROOT / 'record.json').read_text())
    gpu.unchanged(row)
    native = gpu.built()
    require(row['stage'] == 'closed' and row['passed'] and row['complete']
        and row['ownership_closed'] and not row['active'] and row['GPU_executed']
        and row['fresh_faults'] == [] and row['boot_unchanged']
        and [(r['name'], r['native_exit'], r['checks']) for r in row['results']] ==
            [('submission', 0, 10), ('phase-cache', 0, 16)],
        'closed same-build IQ3 host-v2 and small traced GPU prerequisites required')
    providers = json.loads(Path(row['runtime_manifest']['path']).read_text())
    prior = base.committed(RESULT)['cache_map_model_entry_zero_v1']
    require(prior['passed'] and prior['complete'] and prior['ownership_closed']
        and not prior['active'] and prior['binary']['sha256'] ==
            '552b1616d21ad1b9aa6a317cb940d375c003246b896a255a935e09292804c96a',
        'retain the qualified fallback model environment')
    driver = prior['environment']['ZE_ENABLE_ALT_DRIVERS']
    require(providers.get(driver) == entry.digest(driver),
        'model ALT driver must match the current GPU-qualified provider inventory')
    native = dict(native, environment=native['recipe']['environment'],
        boot=row['boot'], driver=row['driver'])
    gate = dict(source_commit=row['source_commit'],
        environment=dict(row['environment'], ZE_ENABLE_ALT_DRIVERS=driver))
    return native, gate, providers


def pinned_sources(revision):
    pins = entry.pinned_sources(revision)
    pins.update(base.controls(revision, controller=SELF))
    return pins


def dispatches(text, arm, *, positive):
    result = []
    for line in text.splitlines():
        if not line.startswith(PREFIX.rstrip()):
            continue
        match = re.fullmatch(PREFIX + r'request=(\d+) arm=([01]) calls=(\d+) '
            r'nt1_rows=(\d+) pair_rows=(\d+)', line)
        require(match is not None, 'malformed IQ3 request dispatch evidence')
        request, actual, calls, rows, paired = map(int, match.groups())
        require(actual == arm and 0 < request < 2**64
            and all(0 <= v < 2**64 for v in (calls, rows, paired))
            and paired == arm * rows and (calls > 0 or rows == 0)
            and (not positive or (calls > 0 and rows > 0)),
            'IQ3 actual selector/eligible/selected counter mismatch')
        result.append(dict(request=request, arm=actual, calls=calls,
            nt1_rows=rows, pair_rows=paired))
    require(result, 'missing IQ3 dispatch footer')
    return result


def parser_controls():
    def line(arm, request=2, calls=4, rows=128, paired=None):
        if paired is None:
            paired = arm * rows
        return (PREFIX + f'request={request} arm={arm} calls={calls} '
            f'nt1_rows={rows} pair_rows={paired}\n')
    for arm in (0, 1):
        require(len(dispatches(line(arm), arm, positive=True)) == 1,
            'valid IQ3 dispatch control failed')
    bad = [('', 0), (line(1), 0), (line(0, paired=1), 0),
        (line(1, paired=127), 1), (line(0, rows=0), 0),
        (line(1, calls=0), 1), (line(0, request=0), 0),
        (line(1, request=2**64), 1), (line(1).replace('calls=4', 'calls=-1'), 1),
        (line(1).rstrip().replace('pair_rows=128', 'pair_rows='), 1),
        (line(1).replace('IQ3_GU_DELTA ', 'IQ3_GU_DELTA'), 1)]
    for text, arm in bad:
        try:
            dispatches(text, arm, positive=True)
        except RuntimeError:
            continue
        raise RuntimeError('invalid IQ3 dispatch control accepted')
    return dict(passed=True, positive=2, negative=len(bad), GPU_executed=False)


def contract(row, arm):
    env, args = row['environment'], row['configuration']['args']
    require(env['STRATA_IQ3_PAIR'] == str(arm)
        and env['STRATA_IQ3_PAIR_AUDIT'] == '1'
        and env['STRATA_IQ_SINGLE_EXACT'] == env['STRATA_CPU_SERVICE'] == '0'
        and env['STRATA_CPU_WORK_CENSUS'] == env['STRATA_DECODE_TIMING'] == '0'
        and env['STRATA_SYCL_DECODE_Q6_COLUMNS'] == env['STRATA_SYCL_ADAPT_SYNC'] == '1'
        and args[args.index('--adapt-swaps') + 1] == '0'
        and args[args.index('--expert-cache') + 1] == '1256'
        and args[args.index('--spec') + 1] == '4' and '--no-split-rows' not in args
        and row['read_tokens'] == 32769 and row['runs'] == 2,
        'freeze IQ3-only selector, common audit, Q6, observers and model extent')


def context(arm):
    require(arm in (0, 1), 'choose IQ3 selector 0 or 1')
    ctx = entry.context('columns')
    ctx.ROOT = B / ('decode-iq3-model-entry-arm' + str(arm) + '-v1')
    ctx.KEY = 'decode_iq3_model_entry_arm' + str(arm) + '_v1'
    ctx.proof, ctx.pinned_sources = proof, pinned_sources
    old_current, old_route = ctx.current, ctx.profile_route

    def current(row):
        contract(row, arm)
        return old_current(row)

    def route(text):
        result = old_route(text)
        selected = dispatches(text, arm, positive=True)
        require(len(selected) == 1, 'one IQ3 footer per long request required')
        full = dispatches((ctx.ROOT / 'engine.stderr').read_text(), arm, positive=False)
        require([r['request'] for r in full] == [1, 2, 3]
            and selected[0]['request'] in (2, 3),
            'bind IQ3 evidence to one warmup and two actual long requests')
        result['IQ3_dispatch'] = selected[0]
        return result

    ctx.current, ctx.profile_route = current, route
    return ctx


def prepare(ctx, arm):
    controls = parser_controls()
    ctx.prepare()
    row = json.loads((ctx.ROOT / 'record.json').read_text())
    row['environment'].update(STRATA_IQ3_PAIR=str(arm), STRATA_IQ3_PAIR_AUDIT='1',
        STRATA_IQ_SINGLE_EXACT='0', STRATA_CPU_SERVICE='0', STRATA_CPU_WORK_CENSUS='0',
        STRATA_DECODE_TIMING='0', STRATA_SYCL_DECODE_Q6_COLUMNS='1', STRATA_SYCL_ADAPT_SYNC='1')
    contract(row, arm)
    row['IQ3_arm'] = arm
    row['IQ3_parser_controls'] = controls
    row['qualification_basis'] = dict(build_result_key='decode_iq3_pair_model_build_v1',
        host_result_key='decode_iq3_host_v2', device_result_key=device.KEY,
        component_result_key='cpu_iq3_pair_dispatch_v3',
        reuse='Exact unchanged five-ELF model generation;all-eight CPU production-entry wins,bitwise384/80 arithmetic,28-case pool/phase parity,submission10 and phase16windows/12captures. No inherited model qualification.')
    row['criteria'] = ('Two frozen current-source frontend probes,then one thinking-on warmup '
        'and two fresh32769-input/1..24-output requests through maintained bench.sh. '
        'Only immutable IQ3 arm0/1 differs;Q6columns1,static1256cache,split,spec4,short-read0,'
        'throughput,GDN16/MTP16 and runtime safety remain identical. Require existing '
        'source/build/ELF/provider/token/body/route/clock/validation/fault/owner gates, '
        'one exact IQ3 footer per long request with positive actual calls/NT1 rows and '
        'pair_rows=arm*NT1_rows,and exact warmup/long request IDs1/2/3. Stop on first '
        'failure;no retry or hash refresh. No sustained TG,clean speed,quality,agent,'
        'ordinary-provider-teardown,physical262144,RESTORE or adoption qualification.')
    row['retention'] = ('Keep compact actual routes,request/selector counters,source/ELF identities '
        'and correctness/fault/owner outcomes;retire successful logs/journals/JIT after commit. '
        'Preserve479a fallback and the same five candidate outputs for the imminent '
        'six-process thinking-on strong/weak-MTP IQ3-only comparison;no new native generation.')
    ctx.save(row)
    print('Prepared IQ3 arm' + str(arm) + ' short-model admission;commit before frontend probes.', flush=True)


if __name__ == '__main__':
    require(len(sys.argv) == 3 and sys.argv[1] in ('0', '1') and
        sys.argv[2] in ('prepare', 'host', 'run'), 'use iq3_model_entry.py 0|1 prepare|host|run')
    arm = int(sys.argv[1])
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base.guards.no_model_owners([])
        ctx = context(arm)
        {'prepare': lambda: prepare(ctx, arm), 'host': ctx.host, 'run': ctx.run}[sys.argv[2]]()
