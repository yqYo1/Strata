"""Check the independent score and decay implementations on fresh and restored server requests."""

import fcntl
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import prefill_kv_gather_restore as control
import prefill_score_decay_model as build

bank, require = build.bank, build.require
PARENT = 'b633f7d694cec941c0df25c0fb9edea650d41ca3'
KEY = 'prefill_score_decay_restore_v1'
ROOT = bank.B / 'prefill-score-decay-restore-v1'
ARMS = {'base': (False, False), 'keys': (True, False), 'decay': (False, True), 'both': (True, True)}
ORDER = tuple(f'{size}-{arm}' for size in ('short', 'near') for arm in ARMS)
SCRIPTS = [str(Path(__file__).resolve().relative_to(bank.P)), *control.SCRIPTS,
           str(Path(build.__file__).resolve().relative_to(bank.P))]


def original():
    return bank.previous.historic(PARENT, control.KEY)


def inputs():
    native = bank.previous.read(build.ROOT / 'record.json')
    build.context(native['source_commit']).check_build(native)
    require(native['stage'] == 'built' and native['complete'] and native['passed']
            and native['ownership_closed'] and not native['active']
            and native['host_control']['passed'] and native['host_control']['ownership_closed']
            and native['projection']['source_sha256'] == build.IMAGE_HASH,
            'complete the coherent model build, component prerequisites and host fixture')
    parent = original()
    measured = bank.previous.historic(build.PARENT_BUILD, control.device.coherent_model.KEY)
    speed = bank.previous.read(bank.basis.reference.ROOT / 'record.json')
    require(parent['stage'] == 'closed' and parent['complete'] and parent['passed']
            and parent['ownership_closed'] and not parent['active']
            and parent['binary'] == measured['binaries']['strata'] and parent['build_sha256'] == bank.signature(measured)
            and all(parent[k] == speed[k] for k in ('configuration', 'native_validation_probe', 'runtime_manifest'))
            and parent['environment_template_sha256'] == bank.signature(speed['environment_template'])
            and parent['model_assets_sha256'] == bank.signature(speed['model_assets'])
            and all(parent['outcomes'][size + '-base']['passed'] for size in ('short', 'near')),
            'need the original model with the same configuration and assets at both lengths')
    return native, speed, bank.signature(bank.builder.compact(native)), bank.signature(parent)


def phase_counts(log, requests, keys, decay):
    frames = re.split(r'^strata serve: STATE_HASH L=.*$', log, flags=re.M)
    require(len(frames) == len(requests) + 1, 'expected one state publication per request')
    counts = []
    for i, frame in enumerate(frames):
        if i == len(requests) or requests[i]['prefill_route'] == 'cache-window':
            require('strata prefill score packs:' not in frame and 'strata prefill GDN decay:' not in frame,
                    'dedicated prefill ran outside a layer-major request')
        else:
            req = requests[i]
            require(req['prefill_route'] == 'layer-major', 'unknown prompt route')
            counts.append(dict(request=i, **build.check_dispatch(frame, req['reused'], req['read'] - 1,
                                                                 retain_keys=keys, store_decay=decay)))
    return counts


def check_reader():
    checks = control.model.check_reader()
    # Independent sums of the 32 full slices and 16-row tail at position
    # 253919. Reusing the last score row of the previous chunk is invalid.
    requests = [dict(prefill_route='cache-window'), dict(prefill_route='layer-major', reused=253919, read=8209)]
    marker = 'strata serve: STATE_HASH L=0'
    rejected = 0
    for keys, decay in ARMS.values():
        packed, reused = (131060, 1999624) if keys else (2130684, 0)
        ref, stored = (0, 8208) if decay else (8208, 0)
        layers = [
            f'strata prefill score packs: layer={layer} slices=33 packed_live_rows={packed} reused_live_rows={reused}'
            if layer % 4 == 3 else
            f'strata prefill GDN decay: layer={layer} chunks=2 reference_rows={ref} stored_rows={stored}'
            for layer in range(48)]
        payload = '\n'.join([marker, *layers, marker])
        counts = phase_counts(payload, requests, keys, decay)
        require(len(counts) == 1 and counts[0]['request'] == 1, 'restored segment was not counted')
        for invalid in (payload.replace(f'packed_live_rows={packed}', f'packed_live_rows={packed - 1}', 1),
                        layers[0] + '\n' + payload, payload + '\n' + layers[-1]):
            try:
                phase_counts(invalid, requests, keys, decay)
            except RuntimeError:
                rejected += 1
            else:
                raise RuntimeError('accepted stale score extent or prefill dispatch outside the restored request')
    return dict(**checks, score_decay_arms=len(ARMS), invalid_dispatch_rejected=rejected)


def experiment():
    ctx = bank.basis.core.source.device.load_module('score_decay_restore', control.model.__file__)
    ctx.KEY, ctx.ROOT, ctx.ORDER, ctx.inputs = KEY, ROOT, ORDER, inputs
    ctx.check_reader = check_reader
    previous_inspect, previous_save, previous_current = ctx.inspect, ctx.save, ctx.current

    def environment(row, case):
        env = control.model.environment(row, case)
        env[control.KV] = env[control.GATHER] = '0'
        env['STRATA_SYCL_PREFILL_RETAIN_SCORE_KEYS'] = str(int(case['keys']))
        env['STRATA_SYCL_PREFILL_GDN_PRECOMPUTE_DECAY'] = str(int(case['decay']))
        return env

    def plan(row, name):
        size, arm = name.split('-')
        keys, decay = ARMS[arm]
        case = dict(name=name, near=size == 'near', root=str(ROOT / name), borrow=False, direct=False,
                    keys=keys, decay=decay, command=control.restore.argv(row),
                    requests=control.restore.input_plan(size == 'near'))
        case['environment_sha256'] = bank.signature(environment(row, case))
        return case

    def inspect(log, row, case, requests, pid):
        outcome = previous_inspect(log, row, case, requests, pid)
        size, arm = case['name'].split('-')
        parent = original()['outcomes'][size + '-base']
        control.restore.compare(outcome, parent)
        if arm != 'base':
            control.restore.compare(outcome, row['outcomes'][size + '-base'])
        require(control.check_frames(log, requests, False, False) == parent['dispatch']
                and all(all(a.get(k) == b.get(k) for k in ('attention_store', 'attention_input', 'expert_bank'))
                        for a, b in zip(requests, parent['requests'], strict=True)),
                'KV transfers, MoE gather work or placement changed')
        outcome['score_decay_dispatch'] = phase_counts(log, requests, case['keys'], case['decay'])
        return outcome

    def save(row):
        if row['stage'] == 'prepared':
            pins = bank.stage.source.pin(bank.P, SCRIPTS)['source_sha256']
            require(set(pins) == set(SCRIPTS), 'commit all restore controllers')
            row['sources'].update(pins)
            row['plan'] = dict(order=ORDER, parent=PARENT,
                gates='Check both lengths against their original controls and each arm against the same-build '
                      'reference. Match model state, published and restored prefix, decode handoff, KV transfers, '
                      'gather workload and placements. Verify score and decay dispatch on fresh and resumed rows; '
                      'no dedicated dispatch on the short decode-layout request. Stop at the first error.',
                scope='Instrumented model state and dispatch. Clean performance and final-slot tests are separate.')
        previous_save(row)

    def current(row):
        require(row['plan']['parent'] == PARENT and row['plan']['order'] == list(ORDER)
                and set(SCRIPTS) <= row['sources'].keys(), 'changed state comparison')
        previous_current(row)
        bank.basis.comparison.shared.models.model_pins(row['configuration'], row['model_assets'])

    ctx.environment, ctx.plan, ctx.inspect, ctx.save, ctx.current = environment, plan, inspect, save, current
    return ctx


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(sys.executable == bank.basis.reference.model.PYTHON and action in ('prepare', *ORDER),
            'use the pinned interpreter and one state comparison stage')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        ctx = experiment()
        if action == 'prepare':
            ctx.prepare()
        else:
            ctx.run(action)
