"""Exercise KV prefix reuse and the large MoE gather on one native server build."""

import fcntl
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import prefill_mixed_gather_measure as device
import prefill_kv_stage_trace as kv

model, bank, require = device.model, device.bank, device.require
restore = model.restore
PARENT = '4775de3b03d73734270b3c8049e4845e345e3a6d'
KEY = 'prefill_kv_gather_restore_v1'
ROOT = bank.B / 'prefill-kv-gather-restore-v1'
ARMS = {'base': (False, False), 'reuse': (True, False),
        'vec8': (False, True), 'both': (True, True)}
ORDER = tuple(f'{size}-{arm}' for size in ('short', 'near') for arm in ARMS)
KV = 'STRATA_SYCL_PREFILL_KEEP_KV_STAGE'
GATHER = 'STRATA_SYCL_PREFILL_MOE_GATHER_VEC8'
SCRIPTS = [str(Path(p).resolve().relative_to(bank.P)) for p in
           (__file__, kv.__file__, device.__file__, device.fixture.__file__,
            device.coherent_model.__file__, device.coherent_model.kv.__file__)]


def reference_workloads(native, speed):
    build_sha = bank.signature(bank.builder.compact(native))
    gate = device.validation.committed(bank.RESULT, (device.COHERENT_KEY,))
    require(gate['build_sha256'] == build_sha and gate['binary'] == native['binaries'][device.fixture.TARGET]
            and gate['native_source_commit'] == native['native_source_commit']
            and [s['stage'] for s in gate['stages']] == ['contiguous', 'ragged']
            and all(s['passed'] and s['ownership_closed'] and s['native_exit_code'] == 0
                    and not s['fresh_faults'] for s in gate['stages']),
            'the two same-build device data/guard cases must pass first')
    parent = bank.previous.historic(PARENT, model.KEY)
    require(parent['passed'] and parent['complete'] and parent['ownership_closed']
            and native['projection']['parent_binary'] == parent['binary']
            and parent['native_source_commit'] == bank.INPUT_SOURCE
            and all(parent[k] == speed[k] for k in ('configuration', 'runtime_manifest', 'native_validation_probe'))
            and parent['model_assets_sha256'] == bank.signature(speed['model_assets'])
            and parent['environment_template_sha256'] == bank.signature(speed['environment_template'])
            and all(parent['outcomes'][arm]['passed'] for arm in model.ORDER),
            'the original measured model/state/RESTORE control is required')
    return build_sha, bank.signature(gate)


def inputs():
    native, speed, _, _ = device.coherent_inputs()
    build_sha, device_sha = reference_workloads(native, speed)
    return native, speed, build_sha, device_sha


def gather_counts(frame, vec8, *, historical=False):
    # In the fixed large-only GU route every job of at least 2334 rows also
    # submits one fused GU on the same rows. Its independent counters establish
    # actual eligible work for both gather arms, including partial windows.
    def fields(label, pattern):
        return tuple(map(int, restore.capacity.one(frame, '^strata MoE ' + label + ': ' + pattern + '$', label)))

    fused, rows, small = fields('GU fusion', r'fused_jobs=(\d+) fused_rows=(\d+) separate_jobs=(\d+) min_rows=2334')
    jobs = int(restore.capacity.one(frame, r'^strata MoE accumulator window: windows=\d+ dequants=\d+ '
        r'jobs=(\d+) source_H2D_bytes=\d+ GPU_resident_windows_only=1 operator_max=8192 maximum_rows=\d+$',
        'accumulator dispatch'))
    require(fused + small == jobs and 0 < fused and 2334 * fused <= rows <= 8192 * fused
            and frame.count('strata MoE mixed banks: storage=scratch ') == 1, 'unexpected mixed gather geometry')
    if historical:
        require(not vec8 and 'strata MoE prefill gather:' not in frame, 'expected original scalar parent')
    else:
        counts = fields('prefill gather', r'vec8_jobs=(\d+) vec8_rows=(\d+) scalar_jobs=(\d+) min_rows=2334')
        require(counts == ((fused, rows, small) if vec8 else (0, 0, jobs)), 'actual gather dispatch differs')
    return dict(jobs=jobs, eligible_jobs=fused, eligible_rows=rows)


def check_frames(log, requests, reuse, vec8, *, historical=False):
    require(log.count('[native16-gu] fusion=large min_rows=2334') == 1, 'expected large-only GU fusion')
    frames = re.split(r'^strata serve: STATE_HASH L=.*$', log, flags=re.M)
    require(len(frames) == len(requests) + 1, 'each completed request needs one state publication')
    results = []
    for index, frame in enumerate(frames):
        if index == len(requests) or requests[index]['prefill_route'] == 'cache-window':
            require(not any(marker in frame for marker in ('strata prefill KV stage:', 'strata MoE prefill gather:',
                    'strata trace: layer-major layer ', 'strata MoE accumulator window:')),
                    'unexpected prefill dispatch outside a layer-major request')
            continue
        req = requests[index]
        results.append(dict(request=index, gather=gather_counts(frame, vec8, historical=historical),
            kv=kv.check_request(frame, req['reused'], req['read'] - 1, restore.capacity.CHUNK,
                                reuse=reuse, historical=historical)))
    return results


def experiment():
    ctx = bank.basis.core.source.device.load_module('kv_gather_restore_server', model.__file__)
    ctx.KEY, ctx.ROOT, ctx.ORDER, ctx.inputs = KEY, ROOT, ORDER, inputs
    original_environment, original_inspect = ctx.environment, ctx.inspect
    original_save, original_current = ctx.save, ctx.current

    def environment(row, case):
        env = original_environment(row, case)
        require(env['STRATA_SYCL_PREFILL_GU_SWIGLU'] == 'large', 'require the qualified large-only GU selector')
        env[KV], env[GATHER] = str(int(case['reuse'])), str(int(case['vec8']))
        return env

    def plan(row, name):
        size, arm = name.split('-')
        reuse, vec8 = ARMS[arm]
        case = dict(name=name, root=str(ROOT / name), near=size == 'near', borrow=False, direct=False,
                    reuse=reuse, vec8=vec8, command=restore.argv(row), requests=restore.input_plan(size == 'near'))
        case['environment_sha256'] = bank.signature(environment(row, case))
        return case

    def inspect(log, row, case, requests, pid):
        outcome = original_inspect(log, row, case, requests, pid)
        counts = check_frames(log, requests, case['reuse'], case['vec8'])
        size, arm = case['name'].split('-')
        parent = bank.previous.historic(PARENT, model.KEY)['outcomes'][size + '-copy']
        restore.compare(outcome, parent)
        for a, b in zip(requests, parent['requests'], strict=True):
            require(all(a.get(k) == b.get(k) for k in ('attention_store', 'attention_input', 'expert_bank')),
                    'the fixed bank and copy placements must match the original engine')
        if arm != 'base':
            base = row['outcomes'][size + '-base']
            restore.compare(outcome, base)
            for a, b in zip(counts, base['dispatch'], strict=True):
                require(a['request'] == b['request'] and a['gather'] == b['gather']
                        and all(a['kv'][k] == b['kv'][k] for k in ('mirror_D2H_calls', 'mirror_D2H_bytes'))
                        and a['kv']['reloaded_pages'] + a['kv']['reused_pages'] == b['kv']['reloaded_pages'],
                        'changed expert work, prefix extent or host mirror')
        outcome['dispatch'] = counts
        return outcome

    def save(row):
        if row['stage'] == 'prepared':
            pins = bank.stage.source.pin(bank.P, SCRIPTS)['source_sha256']
            require(set(pins) == set(SCRIPTS), 'commit the model controllers before preparation')
            row['sources'].update(pins)
            row['plan'] = dict(order=ORDER, parent=PARENT, unit='KV prefix reuse and large expert gathers independently.',
                gates='Check original and same-build state, decode handoff and cache restore. Check per-layer '
                      'prefix copies and actual gather jobs for fresh inputs and nonaligned resumed suffixes.',
                scope='Traced state and dispatch. Cancellation, final physical slot, speed and sustained TG are separate.')
        original_save(row)

    def current(row):
        require(set(SCRIPTS) <= row['sources'].keys() and row['plan']['parent'] == PARENT
                and row['plan']['order'] == list(ORDER), 'incomplete controller pins or wrong comparison order')
        original_current(row)
        bank.basis.comparison.shared.models.model_pins(row['configuration'], row['model_assets'])

    ctx.environment, ctx.plan, ctx.inspect, ctx.save, ctx.current = environment, plan, inspect, save, current
    return ctx


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(sys.executable == bank.basis.reference.model.PYTHON and action in ('prepare', *ORDER),
            'use the pinned interpreter and one model stage')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        ctx = experiment()
        if action == 'prepare':
            ctx.prepare()
        else:
            ctx.run(action)
