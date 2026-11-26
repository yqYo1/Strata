"""Check expert placement through DFlash2 live RESTORE and cancellation."""

import fcntl
from pathlib import Path
import re
import struct
import sys
import time

sys.dont_write_bytecode = True
import dflash2_decode_ranked_refill_checks as fixtures
import dflash2_decode_stock_global_restore as stock
from make_profile import N_EXPERT, N_LAYER

build, primary, quota = fixtures.build, stock.primary, stock.quota
bank, basis, require = primary.bank, primary.basis, primary.require
validation = basis.validation
THIS = str(Path(__file__).resolve().relative_to(bank.P))
KEY = 'dflash2_decode_ranked_refill_restore_v1'
ROOT = bank.B / KEY.replace('_', '-')
CANCEL = 'cancel-ranked'
ORDER = ('host', 'short-stock', 'short-ranked', 'near-stock', 'near-ranked', CANCEL)
SELECTOR = 'STRATA_SYCL_DFLASH2_DECODE_REFILL_PROFILE'
SHORT = 33
CANCEL_REQUESTS = ((32769, 0, True), (SHORT, 128, False), (32769, 256, False),
                   (32769, 512, True), (SHORT, 768, False))
PLAN = dict(order=list(ORDER),
            scope='Reuse parked 32769/253919, unrelated 33 and restored 8209-fresh-token requests. '
                  'Start both arms with the same stock global profile; enable only final ranked refill in the candidate.',
            gates='Check the masked table before each batched segment, the full table for the short prefill, '
                  'the tables and payload after each handoff, and the live RAM snapshot and suffix read. '
                  'Require actual load, segment extents and successful closure.',
            cancel='Send STOP after the mask and first QSA KV stage, first with stock and then blended residency. '
                   'Require partial read with zero output, the original payload and map, and a fresh short request.',
            stop='Stop on the first failed comparison. Physical full fill, quality and speed are separate.')

PHASE = 'strata DFlash2 expert phase: '
PUBLISH = 'strata DFlash2 decode refill: published '
PAYLOAD = 'strata loan audit: phase=payload '
READ = re.compile(r'strata trace: read (\d+) tokens \((windows|batched)\) in [\d.]+ ms')
FIRST_KV_STAGE = re.compile(rb'strata prefill KV stage: layer=3 reloaded_pages=\d+ reused_pages=\d+\n')


def map_digest(ranks, first_masked):
    slots = [-1] * (N_LAYER * N_EXPERT)
    for index, (layer, expert) in enumerate(ranks[:first_masked]):
        slots[layer * N_EXPERT + expert] = index
    # The native audit hashes the complete little-endian int32 residency table.
    digest = 1469598103934665603
    for byte in struct.pack(f'<{len(slots)}i', *slots):
        digest = ((digest ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return f'{digest:016x}'


def phase_record(index, phase, begin, end, boundary, digest):
    return (f'{PHASE}request={index} phase={phase} from={begin} to={end} '
            f'first_masked={boundary} entries={N_LAYER * N_EXPERT} map={digest}')


def expected_trace(requests, digests, ranked):
    protected, slots = quota.PROTECTED, quota.EXPECTED_SLOTS
    installed = 'stock'
    trace = []
    for index, request in enumerate(requests, 1):
        begin, end = request['reused'], request['prompt'] - 1
        window = end - begin == SHORT - 1
        cancelled = request.get('cancelled', False)
        phase, boundary = ('prefill-cache', slots) if window else ('prefill-batched', protected)
        trace.append(phase_record(index, phase, begin, end, boundary, digests[installed if window else 'masked']))
        trace.append(f'read {end - begin} {"windows" if window else "batched"}')
        if not window:
            switched = ranked and not cancelled
            trace.append(f'{PAYLOAD}rows={slots} bytes={quota.EXPECTED_BYTES} lent={slots - protected} '
                         f'same_slots={int(not switched)} payload_match=1')
            if switched:
                trace.append(f'{PUBLISH}first={protected} slots={slots - protected} payload_bytes={quota.EXPECTED_TAIL}')
                installed = 'blend'
        final = ('cancel-refill', begin) if cancelled else ('decode-handoff', end)
        trace.append(phase_record(index, final[0], begin, final[1], slots, digests[installed]))
    return trace


def check_trace(log, requests, digests, ranked):
    actual = []
    for line in log.splitlines():
        if line.startswith((PHASE, PUBLISH, PAYLOAD)):
            actual.append(line)
        elif match := READ.fullmatch(line):
            actual.append(f'read {match[1]} {match[2]}')
    expected = expected_trace(requests, digests, ranked)
    require(actual == expected, 'expert phase trace differs from the planned ordering, ranges, residency or payloads')
    return dict(phases=[line.removeprefix(PHASE) for line in actual if line.startswith(PHASE)],
                ranked_tail_refills=sum(line.startswith(PUBLISH) for line in actual),
                verified_payloads=sum(line.startswith(PAYLOAD) for line in actual))


def cancel_workloads(ctx):
    for n, offset, cancelled in CANCEL_REQUESTS:
        payload = ctx.protocol.wire([(i + offset) % 4096 for i in range(n)], 1)
        yield payload, dict(prompt=n, reused=0, max_new=1, cancelled=cancelled,
                            wire_bytes=len(payload), wire_sha256=ctx.signature_bytes(payload))


def stop_after_kv_stage(io, diagnostic, phase):
    deadline = time.monotonic() + 180
    masked = False
    expected = phase.encode() + b'\n'
    while time.monotonic() < deadline:
        io.poll()
        require(not io.exit_watch.poll(0), 'model exited before the cancellation trigger')
        position = diagnostic.tell()
        line = diagnostic.readline(8193)
        require(len(line) <= 8192, 'stderr line exceeds the cancellation reader limit')
        if not line.endswith(b'\n'):
            diagnostic.seek(position)
            time.sleep(.02)
        elif line.startswith(PHASE.encode()):
            require(line == expected and not masked, 'unexpected expert phase before STOP')
            masked = True
        elif masked and FIRST_KV_STAGE.fullmatch(line):
            io.send(b'STOP\n')
            return line.decode().strip()
    raise TimeoutError('no first-layer KV stage after the checked expert loan')


def interact_cancel(ctx, io, case, vocab_size):
    info = {}
    while True:
        line = io.line(600)
        require(not line.startswith('ERR'), line)
        if line.startswith('INFO '):
            info.update(t.split('=', 1) for t in line.split()[1:] if '=' in t)
        if line.startswith('READY '):
            require(line == 'READY 262144 stop', 'native STOP protocol unavailable')
            break
    require(all(info.get(k) == v for k, v in ctx.INFO.items()), 'native diagnostics changed layout')
    results = []
    stderr = Path(case['root']) / 'owned' / f'{case["name"]}.stderr'
    with stderr.open('rb') as diagnostic:
        for index, ((payload, request), pinned) in enumerate(zip(cancel_workloads(ctx), case['requests'], strict=True), 1):
            require(request == pinned, 'cancel/recovery fixture bytes changed')
            # Avoid consuming progress from the completed request when waiting for the next loan.
            diagnostic.seek(0, 2)
            io.send(payload)
            n, cancelled = request['prompt'], request['cancelled']
            triggered = None
            if cancelled:
                phase = phase_record(index, 'prefill-batched', 0, n - 1, quota.PROTECTED, case['masked_digest'])
                triggered = stop_after_kv_stage(io, diagnostic, phase)
            tokens, pp = [], []
            while True:
                line = io.line(1800)
                require(not line.startswith('ERR'), line)
                fields = line.split()
                if line.startswith('PP '):
                    pp.append(int(fields[1]))
                if line.startswith('T '):
                    tokens.append(int(fields[1]))
                if line.startswith('DONE '):
                    break
            require(len(fields) == 15, 'native DONE format differs')
            output, prompt, accepted, offered, reused, read = map(int, (fields[1], fields[2], *fields[6:9], fields[14]))
            require((prompt, accepted, offered, reused) == (n, 0, 0, 0)
                    and output == len(tokens) and all(0 <= token < vocab_size for token in tokens)
                    and all(0 <= a < b < n for a, b in zip([0, *pp[:-1]], pp)), 'native request work differs')
            reached = pp[-1] if pp else 0
            if cancelled:
                require(fields[5] == 'cancel' and output == 0 and read == reached < n - 1,
                        'STOP did not cancel before prompt completion')
            else:
                require(fields[5] in ('length', 'stop') and output == 1 and read == n and reached == n - 1,
                        'recovery did not read and decode the complete fresh request')
            results.append(dict(prompt=prompt, reused=reused, read=read, cancelled=cancelled, finish=fields[5],
                                output_hash=ctx.signature(tokens), publications=pp, stop_after=triggered))
    io.send(b'QUIT\n')
    io.exit(90)
    return results


def qualified_build():
    native = bank.previous.read(build.ROOT / 'record.json')
    require(build.retained.closed(native, 'built'), 'finish the ranked refill build')
    original = build.check(native)
    envelope = primary.physical.checks.parameters(native, original['native_validation_probe'])
    qualification = validation.committed(bank.RESULT, (fixtures.KEY,))
    fields = ('build_sha256', 'native_source_commit', 'boot', 'driver', 'runtime_manifest', 'native_validation_probe')
    require(build.retained.closed(qualification, 'closed')
            and all(qualification[k] == envelope[k] for k in fields)
            and all(qualification['outcomes'][name]['passed'] for name in fixtures.ORDER),
            'require the lease and payload fixtures on this exact build')
    return native, envelope, qualification


def experiment():
    ctx = primary.experiment()
    base_environment, base_inspect, base_interact = ctx.environment, ctx.inspect, ctx.interact
    ctx.ROOT, ctx.KEY, ctx.ORDER = ROOT, KEY, ORDER

    def environment(row, case):
        env = dict(base_environment(row, case), STRATA_CPU_WORK_CENSUS='1', STRATA_DECODE_TIMING='1')
        require(SELECTOR not in env, 'starting template must use stock placement')
        if case['arm'] == 'ranked':
            env[SELECTOR] = row['profiles']['blend']['path']
        return env

    def inputs(revision):
        native, envelope, qualification = qualified_build()
        prior = bank.previous.read(stock.ROOT / 'record.json')
        saved = validation.committed(bank.RESULT, (stock.KEY,))
        require(ctx.compact(prior) == saved and build.retained.closed(prior, 'closed')
                and prior['binary'] == native['base_binary'] and prior['native_build_sha256'] == native['base_build_sha256']
                and all(prior[k] == envelope[k] for k in ('boot', 'driver', 'native_validation_probe', 'runtime_manifest')),
                'require the exact closed stock RESTORE control and runtime')
        basis.comparison.shared.models.model_pins(prior['configuration'], prior['model_assets'], full=False)
        require(all(bank.sha(p) == h for p, h in prior['drafter_assets'].items()), 'draft assets changed')

        profiles = native['profiles']
        ranks = {arm: quota.read_profile(profile['path']) for arm, profile in profiles.items()}
        protected, slots = quota.PROTECTED, quota.EXPECTED_SLOTS
        require(profiles['stock']['path'] == str(stock.GLOBAL)
                and profiles['stock']['sha256'] == prior['profile_sha256']
                and ranks['stock'][:protected] == ranks['blend'][:protected]
                and ranks['stock'][protected] != ranks['blend'][protected], 'the admitted 132-slot prefix changed')
        digests = {arm: map_digest(rank, slots) for arm, rank in ranks.items()}
        require(digests['stock'] != digests['blend'], 'the frozen placement maps are identical')
        digests['masked'] = map_digest(ranks['stock'], protected)
        pins = ctx.sources(revision)
        files = (THIS, fixtures.THIS, build.THIS, stock.THIS, quota.THIS, primary.THIS, primary.physical.THIS,
                 primary.physical.checks.THIS, 'tools/make_profile.py')
        for name in files:
            path = bank.P / name
            require(path.read_bytes() == validation.git('show', revision + ':' + name),
                    'commit the exact phase-check input: ' + name)
            pins[name] = bank.sha(path)
        fields = ('configuration', 'native_validation_probe', 'boot', 'driver', 'vocab_size', 'drafter_assets',
                  'ordinary_control', 'python_selected', 'environment_template', 'model_assets', 'runtime_manifest')
        return dict({k: prior[k] for k in fields}, binary=native['binaries']['strata'],
                    native_build_sha256=envelope['build_sha256'], native_source_commit=native['native_source_commit'],
                    profiles=profiles, digests=digests, source_commit=revision, sources=pins, plan=PLAN,
                    stock_restore_sha256=bank.signature(saved), fixtures_sha256=bank.signature(qualification)), prior

    def plans(row, prior):
        previous = {case['name']: case for case in prior['plans']}
        result = []
        for name in ORDER[1:-1]:
            extent, arm = name.split('-')
            old = previous[extent + '-global']
            require(old['requests'] == ctx.input_plan(old['near']) and not old['direct']
                    and old['command'][old['command'].index('--expert-profile') + 1] == row['profiles']['stock']['path']
                    and bank.signature(environment(row, old)) == old['environment_sha256'],
                    'the checked stock request or provider envelope changed')
            case = dict(old, name=name, arm=arm, root=str(ROOT / name),
                        command=[row['binary']['path'], *old['command'][1:]])
            case['environment_sha256'] = bank.signature(environment(row, case))
            result.append(case)
        base = next(case for case in result if case['name'] == 'short-ranked')
        case = dict(base, name=CANCEL, root=str(ROOT / CANCEL), masked_digest=row['digests']['masked'],
                    requests=[request for _, request in cancel_workloads(ctx)])
        case['environment_sha256'] = bank.signature(environment(row, case))
        result.append(case)
        return result

    def prepare():
        require(not ROOT.exists() and KEY not in validation.committed(bank.RESULT), 'require one fresh model cohort')
        row, prior = inputs(validation.git('rev-parse', 'HEAD').decode().strip())
        cases = plans(row, prior)
        ROOT.mkdir(mode=0o700)
        (ROOT / 'host').mkdir(mode=0o700)
        for case in cases:
            for folder in ('tmp', 'cache'):
                (Path(case['root']) / folder).mkdir(mode=0o700, parents=True)
        row.update(stage='prepared', outcomes={}, plans=cases, passed=False, complete=False,
                   ownership_closed=True, active=False, speed_eligible=False)
        ctx.save(row)

    def current(row):
        require(ctx.compact(row) == validation.committed(bank.RESULT, (KEY,)), 'commit the preceding model gate')
        expected, prior = inputs(row['source_commit'])
        require(all(row[k] == v for k, v in expected.items()) and row['plans'] == plans(row, prior),
                'the committed build, runtime or RESTORE plan changed')

    def interact(io, case, vocab_size, *, native):
        if case.get('name') == CANCEL:
            return interact_cancel(ctx, io, case, vocab_size)
        return base_interact(io, case, vocab_size, native=native)

    def inspect(log, row, case, requests, pid):
        if case['name'] == CANCEL:
            proof = ctx.validation.preload_evidence(log, pid)
            require(proof['loader'] == row['native_validation_probe']['loader']['path']
                    and not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|teardown uncertain', log),
                    'runtime error while cancelling or recovering')
            result = dict(validation=proof)
        else:
            extent = case['name'].split('-')[0]
            result = base_inspect(log, row, dict(case, name=extent + '-copy'), requests, pid)
        stock.check_loader(log, row['profiles']['stock']['path'], per_layer=False)
        ranked = case['arm'] == 'ranked'
        available = f'strata DFlash2 decode refill: available slots={quota.EXPECTED_SLOTS} first_changed={quota.PROTECTED}'
        require([line for line in log.splitlines() if line.startswith('strata DFlash2 decode refill: available ')]
                == ([available] if ranked else []), 'ranked tail startup does not match the arm')
        result['expert_placement'] = check_trace(log, requests, row['digests'], ranked)
        return result

    ctx.prepare, ctx.current, ctx.environment, ctx.inspect, ctx.interact = prepare, current, environment, inspect, interact
    return ctx


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(sys.executable == basis.reference.model.PYTHON and action in ('prepare', *ORDER), 'select a RESTORE stage')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        runner = experiment()
        if action == 'prepare':
            runner.prepare()
        else:
            runner.run(action)
