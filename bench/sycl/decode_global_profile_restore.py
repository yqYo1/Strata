"""Verify byte-matched global decode admissions on the ordinary prefill/RESTORE route."""

import fcntl
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import decode_profile_restore as per_layer
import decode_profile_tail as tail

bank, model, restore = per_layer.bank, per_layer.model, per_layer.restore
state, census, footer, require = per_layer.state, per_layer.census, per_layer.footer, per_layer.require
ROOT = bank.B / 'decode-global-profile-restore-v1'
KEY = 'decode_global_profile_restore_v1'
ORDER = per_layer.ORDER
# Actual-capacity predictions. Both rank files preserve the 8192-row scratch tail.
GEOMETRY = {
    'stock': (1241, 2573849600, '6c1702ba8d9facd59e5acb265dfc14f9148e24f0749b7c28b2ffe604b196bd2d'),
    'intel': (1244, 2573772800, '10af810479ba51bdd152c26add79dd6c402539eb4e920cda369287b2d4be1266'),
}
SOURCES = [*per_layer.SOURCES, 'bench/sycl/decode_global_profile_restore.py',
           'bench/sycl/decode_profile_tail.py', 'tools/make_profile.py']


def controls():
    row = bank.basis.validation.committed(bank.RESULT, (per_layer.KEY,))
    require(row['stage'] == 'closed' and row['complete'] and row['passed'] and row['ownership_closed']
            and not row['active'] and tuple(row['outcomes']) == ORDER
            and all(p['complete'] and p['passed'] and p['native_exit_code'] == 0 and p['ownership_closed']
                    and not p['fresh_faults'] for p in row['outcomes'].values()), 'need both per-layer ranking controls')
    return row


def manifest(row):
    args = row['configuration']['args']
    return Path(args[args.index('--pack') + 1]) / 'native_experts.txt'


def check_state(log, requests, case, reference, slots, payload_bytes):
    """Bind each native loan and its payload readback to the fresh or restored request."""
    frames = re.split(r'^strata serve: STATE_HASH (L=.*)$', log, flags=re.M)
    require(len(frames) == 7 and len(requests) == 3 and log.count('strata loan audit: phase=') == 8,
            'expected three state publications and two complete loans')
    for i, (req, ref) in enumerate(zip(requests, reference['requests'], strict=True)):
        frame = frames[i*2]
        state_hash = dict(re.findall(r'(\w+)=([0-9a-f,-]+)', frames[i*2+1]))
        require(set(state_hash) == set(restore.ACTIVE_STATE + restore.STALE_STATE)
                and int(state_hash['L']) == req['prompt'], 'wrong state domains or position')
        require(all(req[k] == ref[k] for k in ('prompt', 'read', 'reused', 'publications')),
                'prompt extent, cache reuse or publications differ from same-ranking control')
        store = restore.require_store(frame, case, req)
        require(store == ref['attention_store']
                and model.input_receipt(frame, req['read'] - 1, store, False) == ref['attention_input'],
                'prefill store or read path differs')
        phases = re.findall(r'^strata loan audit: phase=(\w+)', frame, re.M)
        if i == 1:
            require(not phases and req['read'] == req['prompt'] == 33 and req['publications'] == [32]
                    and 'strata prefill policy: requested=throughput route=cache-window tokens=32 chunk=8192'
                    in frame and 'strata serve: DRAFT_PREFILL ' not in frame, 'short switch changed route')
            req['prefill_route'] = 'cache-window'
            continue
        req['prefill_route'] = 'layer-major'
        require(model.expert_bank_receipt(frame, store, case['near'] and i == 0) == ref['expert_bank'],
                'prefill expert bank changed')
        ends = []
        position = req['reused']
        for count in re.findall(r'^strata serve: DRAFT_PREFILL path=(?:batched|token) mode=[012] cells=(\d+)$', frame, re.M):
            require(0 < int(count) <= 8192, 'draft publication exceeds chunk')
            position += int(count)
            ends.append(position)
        lent = slots - 132
        require(ends == req['publications'] and phases == ['layout', 'lend', 'payload', 'refill']
                and f'strata loan audit: phase=layout chunk=8192 first=132 slots={slots} bytes=2321664000 '
                    f'need=2319958528 lent={lent} mask_match=1' in frame.splitlines()
                and f'strata loan audit: phase=payload rows={slots} bytes={payload_bytes} lent={lent} '
                    'same_slots=1 payload_match=1' in frame.splitlines(), 'wrong scratch extent or expert payload')
        for phase in ('lend', 'refill'):
            restore.capacity.one(frame, rf'^strata loan audit: phase={phase} table_entries=24576 table_match=1 '
                                 r'free_bytes=\d+ total_bytes=\d+$', phase)
    parked = requests[0]['prompt']
    restored, = re.findall(r'^strata serve: conversation cache: restored (\d+) tokens \(live\)', log, re.M)
    snap, = re.findall(r'^strata serve: SNAPSHOT_VERIFY (.*)$', log, re.M)
    snapshot = dict(s.split('=', 1) for s in snap.split())
    require(int(restored) == parked == int(snapshot['upto']) and snapshot['result'] == 'PASS'
            and snapshot['source'] == 'ram' and 'strata serve: conversation cache: skip parking' not in log
            and all(snapshot[k] == reference['restore_payload'][k] for k in
                    ('cells', 'mode', 'resident', 'main_layers', 'main_bytes', 'draft_bytes', 'running_bytes')),
            'live conversation restore or verified byte extent differs')
    return snapshot


def inspect(log, row, case, requests, pid):
    arm = case['arm']
    slots, payload_bytes, _ = GEOMETRY[arm]
    proof = restore.validation.preload_evidence(log, pid)
    require(proof['loader'] == row['native_validation_probe']['loader']['path']
            and not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|teardown uncertain|cleanup failed', log),
            'native validation failed')
    path = ROOT / (arm + '.bin')
    require(log.splitlines().count(f'strata generate: profile {path}: 24576 ranked pairs, built for 24576 slots') == 1
            and log.count('strata generate: profile ') == 1 and log.count('strata generate: pre-filled ') == 1
            and f'strata generate: pre-filled {slots} of {slots} slots from the profile; slot 0 verified' in log.splitlines()
            and log.count('PROFILE, ranked by routing frequency, no eviction.') == 1 and 'R4.2g PER-LAYER:' not in log,
            'expected the full computed global cache')
    reference = controls()['outcomes'][case['name']]
    snapshot = check_state(log, requests, case, reference, slots, payload_bytes)
    dispatch = state.check_frames(log, requests, False, False)
    ordinary = per_layer.same_build_controls()['outcomes'][case['name'].split('-')[0] + '-base']
    require(dispatch == ordinary['dispatch'], 'fixed-prompt expert work or KV transfer count changed')
    windows = re.findall(r'^strata trace: window (\d+) (\d+)$', log, re.M)
    require([(int(p), int(t)) for p, t in windows] == [(r['prompt'] - 1, 1) for r in requests],
            'expected a single T1 verifier at each absolute prompt end')
    bindings = []
    for req, head, clock, copies in zip(requests, re.findall(footer.HEADER, log, re.M),
            re.findall(footer.TIMING, log, re.M), re.findall(footer.COPY, log, re.M), strict=True):
        times = dict(zip(footer.LABELS, map(float, clock[1:]), strict=True))
        require(tuple(map(int, head)) == (req['prompt'], req['reused'], req['read'], 1, 0, 0)
                and int(clock[0]) == 1 and times['average_T'] == times['tokens_per_window'] == 1
                and times['PCIe'] == 0 and not any(map(int, copies[:6])), 'verifier or expert copy work differs')
        bindings.append(dict(prompt_tokens=req['prompt'], windows=1, rows=1, cpu_jobs=int(copies[-2]),
                             cpu_entries=int(copies[-1]), pool_ms=times['CPU_pool_run']))
    # Bind actual membership to the original ranked prefix, independently of
    # the permutation used to size and loan the physical cache slots.
    original = bank.P / per_layer.PROFILES[arm]
    masks = [0] * 48
    for layer, expert in tail.make_profile.read_profile(original)[:slots]:
        masks[layer] |= 1 << expert
    seen = census.extract(log, bindings, census.manifest_formats(manifest(row)), masks)
    return dict(validation=proof, restored_tokens=requests[0]['prompt'], restore_payload=snapshot,
                dispatch=dispatch,
                decode_placement=[{k: s[k] for k in ('request', 'membership_sha256', 'cpu_jobs', 'cpu_entries',
                                      'vram_entries', 'logical_job_blob_bytes')} for s in seen])


def experiment(action):
    ctx = bank.basis.core.source.device.load_module('global_decode_state', model.__file__)
    ctx.KEY, ctx.ROOT, ctx.ORDER = KEY, ROOT, ORDER
    parent = per_layer.experiment()
    original_save, original_current = ctx.save, ctx.current

    def inputs():
        native, speed, build, fixture = parent.inputs()
        prior = controls()
        require(prior['binary'] == native['binaries']['strata'] and prior['native_source_commit'] == native['native_source_commit']
                and prior['build_sha256'] == build and prior['device_sha256'] == fixture, 'per-layer control ELF differs')
        return native, speed, build, fixture

    def plan(row, name):
        case = parent.plan(row, name)
        arm = case['arm']
        command = case['command']
        command.remove('--expert-cache-per-layer')
        command[command.index('--expert-cache') + 1] = '967'
        command[command.index('--expert-profile') + 1] = str(ROOT / (arm + '.bin'))
        case['root'] = str(ROOT / name)
        case['environment_sha256'] = bank.signature(parent.environment(row, case))
        return case

    def save(row):
        if row['stage'] == 'prepared':
            row['sources'].update(bank.stage.source.pin(bank.P, SOURCES)['source_sha256'])
            for arm, source in per_layer.PROFILES.items():
                ranked, slots, size = tail.profile_for_loan(bank.P / source, manifest(row))
                path = ROOT / (arm + '.bin')
                require(slots == GEOMETRY[arm][0] and size == 2321664000 and not path.exists(), 'unexpected slot layout')
                tail.make_profile.write_profile(path, ranked)
                require(bank.sha(path) == GEOMETRY[arm][2], 'generated ranking differs')
            row['plan'] = dict(order=list(ORDER), profile_sha256={a: g[2] for a, g in GEOMETRY.items()},
                control_sha256=bank.signature(controls()), scope='Global vs same-ranking per-layer on the same ELF. '
                'Require identical prefill work, physical scratch extent, verified snapshot size and restored member masks '
                'predicted for each placement. Stop on first failed state, runtime or ownership check. Diagnostic only.')
        original_save(row)

    def current(row):
        require(set(SOURCES) <= row['sources'].keys() and row['plan']['order'] == list(ORDER)
                and row['plan']['control_sha256'] == bank.signature(controls())
                and all(bank.sha(ROOT / (a + '.bin')) == row['plan']['profile_sha256'][a] == g[2]
                        for a, g in GEOMETRY.items()), 'changed comparison inputs')
        original_current(row)
        bank.basis.comparison.shared.models.model_pins(row['configuration'], row['model_assets'])

    if action in ORDER:
        # A private protocol reader has one fixed expected INFO throughout
        # this process. Both the original reader and raw native output stay intact.
        ctx.restore = bank.basis.core.source.device.load_module('global_decode_protocol', restore.__file__)
        slots, size, _ = GEOMETRY[action.split('-')[1]]
        ctx.restore.INFO = dict(restore.INFO, expert_slots=str(slots), expert_slots_primary=str(slots),
                                expert_cache_mib=str(size >> 20), expert_cache_primary_mib=str(size >> 20))
    ctx.inputs, ctx.environment, ctx.plan = inputs, parent.environment, plan
    ctx.inspect, ctx.save, ctx.current = inspect, save, current
    return ctx


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(sys.executable == bank.basis.reference.model.PYTHON and action in ('prepare', *ORDER),
            'use the pinned interpreter and a planned stage')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        ctx = experiment(action)
        if action == 'prepare':
            ctx.prepare()
        else:
            ctx.run(action)
