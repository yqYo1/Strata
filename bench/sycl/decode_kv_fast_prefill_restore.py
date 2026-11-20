"""Check the isolated decode KV routes on the new binary before timing them."""

import fcntl
from pathlib import Path
import re
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
import decode_kv_fast_prefill_build as build
import prefill_kv_gather_restore as state
import decode_footer as footer
import decode_kv_bench as previous

model, bank, restore, require = state.model, state.bank, state.restore, state.require
ROOT = bank.B / 'decode-kv-fast-prefill-restore-v1'
KEY = 'decode_kv_fast_prefill_restore_v1'
THIS = str(Path(__file__).resolve().relative_to(bank.P))
ORDER = [f'{size}-{width}-{arm}' for size in ('short', 'near')
         for width in (2, 4) for arm in ('off', 'on')]
OUTPUTS = (1, 1, 8)
SELECTOR = 'STRATA_SYCL_DECODE_KV_COALESCED'
WINDOW = r'^strata trace: window (\d+) (\d+)$'
STATE = r'^strata serve: STATE_HASH (.+)$'
PLAN = dict(order=ORDER, output_caps=list(OUTPUTS),
    scope='Identical fresh prefix, unrelated short switch and live RESTORE on both arms. '
          'Mask the decode resolver at process startup. Prove the first T1 and the actual T2 or T4 '
          'after RESTORE, including the capped final output and committed state extent.',
    stop='Stop on a wrong route, state, workload, fault, source, or ownership check. No speed claim.')


def inputs():
    _, speed, _, reference_device = state.inputs()
    candidate = bank.previous.read(build.ROOT / 'record.json')
    build.check(candidate)
    require(candidate['stage'] == 'built' and candidate['passed'] and candidate['complete']
            and candidate['ownership_closed'] and not candidate['active'],
            'commit the complete coherent KV candidate build first')
    return candidate, speed, bank.signature(build.builder.compact(candidate)), reference_device


def requests(near):
    cases = []
    for (ids, reused), limit in zip(restore.workloads(near), OUTPUTS, strict=True):
        payload = restore.protocol.wire(ids, limit)
        cases.append(dict(prompt=len(ids), max_new=limit, reused=reused,
                          wire_bytes=len(payload), wire_sha256=restore.signature_bytes(payload)))
    return cases


def environment(row, case):
    return dict(model.environment(row, case),
                **{state.KV: '0', state.GATHER: '0', SELECTOR: str(int(case['enabled'])),
                   'STRATA_SYCL_KV_COPY_GROUPS': '0'})


def plan(row, name):
    size, width, arm = name.split('-')
    width = int(width)
    command = restore.argv(row)
    require('--mtp-max-t' not in command and '--spec-min-p' not in command
            and command.count('--suffix-draft') == 1 and command[command.index('--suffix-draft')+1] == '0',
            'require fixed-width MTP without a suffix proposal')
    command.extend(('--mtp-max-t', str(width)))
    case = dict(name=name, root=str(ROOT / name), near=size == 'near', borrow=False, direct=False,
                width=width, enabled=arm == 'on', requests=requests(size == 'near'), command=command)
    case['environment_sha256'] = bank.signature(environment(row, case))
    return case


def interact(io, case, vocab_size, *, native):
    info = {}
    while True:
        line = io.line(600)
        require(not line.startswith('ERR'), line)
        if line.startswith('INFO '):
            info.update(t.split('=', 1) for t in line.split()[1:] if '=' in t)
        if line.startswith('READY '):
            require(line == 'READY 262144 stop', 'wrong physical context')
            break
    require(native and all(info.get(k) == v for k, v in restore.INFO.items())
            and info.get('mtp_max') == str(case['width']), 'wrong native placement, KV format or width cap')
    result = []
    for (ids, reused), receipt in zip(restore.workloads(case['near']), case['requests'], strict=True):
        limit = receipt['max_new']
        payload = restore.protocol.wire(ids, limit)
        require(len(payload) == receipt['wire_bytes'] and restore.signature_bytes(payload) == receipt['wire_sha256'],
                'generated workload differs from the committed plan')
        for start in range(0, len(payload), 256 << 10):
            io.send(payload[start:start+(256 << 10)])
        emitted, pp = [], []
        while True:
            line = io.line(1800)
            require(not line.startswith('ERR'), line)
            marker, *fields = line.split()
            if marker == 'PP':
                pp.append(int(fields[0]))
            elif marker == 'T':
                require(len(fields) == 1 and len(emitted) < limit and 0 <= int(fields[0]) < vocab_size,
                        'out-of-range output ID or count')
                emitted.append(int(fields[0]))
            elif marker == 'DONE':
                break
        require(len(fields) == 14 and fields[4] in ('length', 'stop'), 'native DONE protocol or termination differs')
        generated, prompt, accepted, offered, actual_reused, read = map(int, (
            *fields[:2], *fields[5:8], fields[-1]))
        require((prompt, actual_reused, read) == (len(ids), reused, len(ids)-reused)
                and 0 <= accepted <= offered and 1 <= generated == len(emitted) <= limit
                and (generated == limit or fields[4] == 'stop') and (limit == 1 or generated > 1)
                and pp and reused < pp[0] and pp[-1] == prompt - 1
                and all(a < b for a, b in zip(pp, pp[1:])), 'incomplete prompt, handoff or output')
        result.append(dict(prompt=prompt, read=read, reused=reused, generated=generated, finish=fields[4], accepted=accepted,
                           offered=offered, output_sha256=bank.signature(emitted), publications=pp))
    io.send(b'QUIT\n')
    io.exit(90)
    return result


def window_evidence(frame, request, case):
    windows = [tuple(map(int, pair)) for pair in re.findall(WINDOW, frame, re.M)]
    require(windows and windows[0] == (request['prompt']-1, 1)
            and all(0 < b[0]-a[0] <= a[1] for a, b in zip(windows, windows[1:]))
            and all(pos + width <= 262144 for pos, width in windows)
            and sum(width - 1 for _, width in windows) == request['offered'],
            'actual verifier windows, draft count or physical bounds differ')
    capture = previous.kv_captures(frame, int(case['enabled']))
    if request['generated'] > 1:
        target = case['width']
        require(len(windows) > 1 and all(width == target for _, width in windows[1:]),
                'the planned post-RESTORE width was not exercised')
        require(any(r['phase'] == 'decode' and r['width'] == target for r in capture),
                'no capture for the completed post-RESTORE verifier width')
    else:
        require(len(windows) == 1 and not any(r['coalesced_calls'] for r in capture),
                'cooperative resolver ran in the single-row controls')
    marker = restore.capacity.one(frame, STATE, 'committed session state')
    fields = dict(t.split('=', 1) for t in marker.split())
    extent = request['prompt'] - 1 + len(windows) + request['accepted']
    last_position, last_width = windows[-1]
    require(set(fields) == set(restore.ACTIVE_STATE + restore.STALE_STATE)
            and request['prompt'] - 1 + request['generated'] <= int(fields['L']) == extent <= 262144
            and last_position < extent <= last_position + last_width,
            'wrong committed extent including accepted draft rows')
    return dict(windows=[list(window) for window in windows], captures=capture,
                active_state={k: fields[k] for k in restore.ACTIVE_STATE})


def check_reader():
    request = dict(prompt=262128, generated=8, accepted=6, offered=6)
    seen = ('strata trace: window 262127 1\n'
            'strata trace: window 262128 4\n'
            'strata verify KV_PLAN: phase=decode tokens=4 enabled=1 scalar_calls=0 coalesced_calls=12\n'
            'strata trace: window 262132 4\n'
            'strata serve: STATE_HASH L=262136 gdn=0 ple=0 tail=0 pooled=0 kv=0 mtp=0 '
            'stale=0 dead=0 pooled_full=0 ple_prev=0,0\n')
    case = dict(width=4, enabled=True)
    out = window_evidence(seen, request, case)
    require(out['windows'][-1] == [262132, 4] and out['active_state']['L'] == '262136',
            'cap-truncated output must preserve the additional committed row')
    off = seen.replace('enabled=1 scalar_calls=0 coalesced_calls=12',
                       'enabled=0 scalar_calls=12 coalesced_calls=0')
    window_evidence(off, request, dict(case, enabled=False))
    bad = (seen.replace('phase=decode', 'phase=prompt'), seen.replace('tokens=4', 'tokens=2'),
           seen.replace('262132 4', '262133 4'), seen.replace('L=262136', 'L=262135'))
    for log in bad:
        try:
            window_evidence(log, request, case)
        except (ValueError, RuntimeError):
            continue
        raise RuntimeError('invalid verifier trace or committed state was accepted')
    return dict(positive=2, negative=len(bad), passed=True)


def loan_evidence(frame, request):
    if request['prompt'] == request['read'] == 33:
        require(not any(marker in frame for marker in ('strata loan audit: phase=lend',
                                                       'strata loan audit: phase=payload')),
                'short switch unexpectedly borrowed the expert cache')
        return 0
    for phase in ('lend', 'refill'):
        restore.capacity.one(frame, r'^strata loan audit: phase=' + phase +
            r' table_entries=24576 table_match=1 free_bytes=\d+ total_bytes=\d+$', phase + ' table')
    lent = int(restore.capacity.one(frame, r'^strata loan audit: phase=payload rows=1256 bytes=2575206400 '
        r'lent=([1-9]\d*) same_slots=1 payload_match=1$', 'restored expert cache payload'))
    require(lent <= 1256, 'loan exceeds the admitted expert cache')
    return lent


def inspect(log, row, case, requests_seen, pid):
    proof = restore.validation.preload_evidence(log, pid)
    require(proof['loader'] == row['native_validation_probe']['loader']['path']
            and not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|teardown uncertain|cleanup failed', log),
            'unexpected native runtime error')
    heads = list(re.finditer(footer.HEADER, log, re.M))
    previous_end, observed = 0, []
    for head, request in zip(heads, requests_seen, strict=True):
        require(tuple(map(int, head.groups())) == tuple(request[k] for k in
                ('prompt', 'reused', 'read', 'generated', 'accepted', 'offered')),
                'native footer differs from the actual protocol reply')
        frame = log[previous_end:head.start()]
        observed.append(dict(window_evidence(frame, request, case), loan_slots=loan_evidence(frame, request)))
        previous_end = head.end()
    require(not re.search(WINDOW + '|' + STATE + '|^strata verify KV_PLAN:', log[previous_end:], re.M),
            'trailing unaccounted windows or state')
    restored, kind = restore.capacity.one(log,
        r'^strata serve: conversation cache: restored (\d+) tokens \((live|checkpoint)\)', 'live RESTORE')
    require(int(restored) == requests_seen[0]['prompt'] and kind == 'live', 'previous conversation did not restore')
    snapshot = restore.capacity.one(log, r'^strata serve: SNAPSHOT_VERIFY (.*)$', 'restored payload')
    snapshot = dict(t.split('=', 1) for t in snapshot.split())
    require(snapshot['result'] == 'PASS' and snapshot['source'] == 'ram'
            and int(snapshot['upto']) == int(restored) and int(snapshot['main_layers']) == 12
            and all(int(snapshot[k]) > 0 for k in ('main_bytes', 'draft_bytes', 'running_bytes')),
            'incomplete live cache snapshot verification')
    out = dict(validation=proof, requests=requests_seen, snapshot=snapshot, verification=observed)
    if case['enabled']:
        off = row['outcomes'][case['name'].removesuffix('-on') + '-off']
        require(off['passed'] and off['complete'] and off['ownership_closed'] and not off['fresh_faults']
                and out['requests'] == off['requests'] and out['snapshot'] == off['snapshot']
                and all(all(a[k] == b[k] for k in ('windows', 'active_state', 'loan_slots'))
                        for a, b in zip(observed, off['verification'], strict=True)),
                'enabled resolver changed the controlled work or active state')
    return out


def experiment():
    ctx = bank.basis.core.source.device.load_module('isolated_decode_kv_restore', model.__file__)
    ctx.ROOT, ctx.KEY, ctx.ORDER = ROOT, KEY, ORDER
    ctx.inputs, ctx.plan, ctx.environment, ctx.inspect = inputs, plan, environment, inspect
    ctx.restore = SimpleNamespace(**{**vars(restore), 'interact': interact})
    base_save, base_current = ctx.save, ctx.current
    base_check_reader = ctx.check_reader
    controllers = [THIS, build.THIS, *state.SCRIPTS]

    def checks():
        return dict(common=base_check_reader(), decode=check_reader())

    def save(row):
        if row['stage'] == 'prepared':
            row['sources'].update(bank.stage.source.pin(bank.P, controllers)['source_sha256'])
            row['plan'] = PLAN
        base_save(row)

    def current(row):
        require(row['plan'] == PLAN and set(controllers) <= row['sources'].keys(), 'commit the complete gate inputs')
        base_current(row)

    ctx.save, ctx.current, ctx.check_reader = save, current, checks
    return ctx


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(sys.executable == bank.basis.reference.model.PYTHON and action in ('prepare', *ORDER),
            'use the pinned interpreter and one committed model stage')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        ctx = experiment()
        if action == 'prepare':
            ctx.prepare()
        else:
            ctx.run(action)
