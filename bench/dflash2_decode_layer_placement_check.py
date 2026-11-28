"""Check decode cache publication while keeping the measured per-layer stock placement."""

import fcntl
from hashlib import sha256
from pathlib import Path
import re
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import dflash2_decode_layer_placement_build as build
import dflash2_prefill_ple_horizon_continuation as horizon
import dflash2_decode_stock_global_restore as stock
from check_dflash2_model import request_ids, wire
from engine_once import native_environment
from native_request_protocol import RequestProtocol
from run_q6_columns import validation_log

bank, builder, require, validation = build.bank, build.builder, build.require, build.validate
model = horizon.primary.speed.generation.model
models = bank.basis.comparison.shared.models
KEY = 'dflash2_decode_layer_placement_check_v1'
ROOT = bank.B / KEY.replace('_', '-')
THIS = 'bench/dflash2_decode_layer_placement_check.py'
ORDER = ('stock', 'blend', 'copy-failure')
PREFILL_CACHE = 'stock'
GEOMETRY = (979, 2_010_419_200)
SELECTOR = 'STRATA_SYCL_DFLASH2_DECODE_PHASE_PROFILE'
AUDIT = 'STRATA_SYCL_DFLASH2_DECODE_PHASE_AUDIT'
CLOCK = 'STRATA_SYCL_DFLASH2_DECODE_PHASE_CLOCK'
FAULT = 'STRATA_SYCL_DFLASH2_DECODE_PHASE_TEST_COPY_FAILURE'
FAULT_ERROR = 'ERR decode placement test: stopped after one queued copy'
FAULT_PREFIX = 'strata decode placement copy failure test:'
FAULT_EVENT = re.compile(r'^strata decode placement copy failure test: emitted=(\d+) '
                         r'queued=1 first_slot=(\d+) h2d_bytes=(\d+) drained=1 masked=1$', re.M)
FIRST_COPY = (132, 1_971_200)
EVENT = re.compile(r'^strata decode placement: profile=(blend|stock) emitted=(\d+) slots=(\d+) '
                   r'h2d_bytes=(\d+) start_ns=(\d+) wall_ms=(\d+\.\d+) audit=1$', re.M)
PREPARE = re.compile(r'^strata decode placement prepare: emitted=(\d+) slots=(\d+) '
                     r'start_ns=(\d+) wall_ms=(\d+\.\d+)$', re.M)
WORK_PREFIX = 'strata decode placement work:'
WORK = re.compile(r'^strata decode placement work: windows=(\d+) verified_rows=(\d+) '
                  r'w1=(\d+) w2=(\d+) w3=(\d+) w4=(\d+) other=(\d+)$', re.M)
STOCK_PROFILE = bank.P / build.PROFILES_FILE
STATE = re.compile(r'^strata serve: STATE_HASH L=.*$', re.M)
LIMITS = dict(wall=720, cpu=1800, rss_cap=104 << 30, text_cap=12 << 20,
              total_cap=512 << 20, file_cap=12 << 20)


def inputs(revision):
    native = bank.previous.read(build.ROOT / 'record.json')
    require(build.grouped.wait.retained.closed(native, 'built'), 'finish the coherent placement build')
    build.context().check(native)
    base = horizon.baseline()
    require(base['native_build_sha256'] == native['prefill_build_sha256'],
            'same measured prefill basis required')
    models.model_pins(base['configuration'], base['model_assets'], full=False)
    profiles = native['profiles']
    controllers = (THIS, build.THIS, stock.THIS, stock.quota.THIS,
                   'tools/make_profile.py', 'tools/strata_tokenizer.py',
                   'tools/sycl/check_dflash2_model.py', 'tools/sycl/engine_once.py',
                   'tools/sycl/native_request_protocol.py', 'tools/sycl/run_q6_columns.py',
                   *(Path(x.__file__).resolve().relative_to(bank.P).as_posix()
                     for x in (models, validation, builder.ownership)))
    own_pins = build.source.pin(bank.P, controllers)['source_sha256']
    require(set(own_pins) == set(controllers)
            and bank.stage.hashed(dict(build.source.blobs(bank.P, revision, own_pins))) == own_pins,
            'commit the exact model checker, tokenizer and protocol')
    pins = horizon.experiment().pins(revision) | own_pins
    args = list(base['validated_dflash2_args'])
    locked = {'--expert-cache': '1256', '--expert-profile': str(STOCK_PROFILE), '--adapt-swaps': '0',
              '--prompt-cache': '1', '--prompt-cache-every': '0', '--pcie-frac': '0', '--spec': '4',
              '--draft': 'dflash2', '--draft-fallback': 'none', '--max-context': '262144'}
    require(all(args.count(flag) == 1 and args[args.index(flag) + 1] == value for flag, value in locked.items())
            and args.count('--expert-cache-per-layer') == 1 and '--mmap-experts' not in args
            and bank.sha(STOCK_PROFILE) == native['controller_sources'][build.PROFILES_FILE],
            'preserve the measured per-layer profile, geometry and stock arguments')
    ids = request_ids(Path(base['configuration']['tokenizer']))['ids']
    require(len(ids) == 82, 'the short thinking prompt changed')
    requests = [dict(ids=[i, *ids], cap=n, sha256=sha256(wire([i, *ids], n)).hexdigest())
                for i, n in enumerate((32, 640, 32, 640, 32), 1)]
    cases, environments = {}, {}
    binary = native['binaries']['strata']
    for index, arm in enumerate(ORDER):
        directory = ROOT / arm
        env = horizon.environment(base, dict(root=str(directory), index=index, arm=PREFILL_CACHE))
        require(not any(name in env for name in (CLOCK, AUDIT, SELECTOR, FAULT)),
                'unexpected placement setting in the measured prefill template')
        env = native_environment(base, env)
        env.update(STRATA_TRACE='1', STRATA_STATE_HASH='1', LD_DEBUG='libs',
                   UR_LOG_VALIDATION=f'level:warning;flush:warning;output:file,{directory / "validation.log"}',
                   **{CLOCK: '1'})
        if arm != 'stock':
            env.update({SELECTOR: profiles['blend']['path'], AUDIT: '1'})
        if arm == 'copy-failure':
            require('STRATA_REFILL_BLOCKING' not in env, 'copy failure test needs the measured queued copies')
            env[FAULT] = '1'
        cases[arm] = dict(binary=binary, command=[binary['path'], '--serve', *args],
                          environment_sha256=bank.signature(env))
        environments[arm] = env
    plan = dict(source_commit=revision, sources=pins, order=list(ORDER), cases=cases, requests=requests,
                prefill_cache=PREFILL_CACHE,
                boot=base['boot'], driver=base['driver'], native_source_commit=native['native_source_commit'],
                build_sha256=bank.signature(builder.compact(native)), profiles=profiles,
                prefill_build_sha256=native['prefill_build_sha256'],
                model_assets_sha256=bank.signature(base['model_assets']),
                drafter_assets_sha256=bank.signature(base['dflash2_assets']),
                scope='Use the actual 1,256-slot per-layer stock placement in both processes. After the '
                      'first emitted tokens, finish one switch and cancel a second just after audited '
                      'publication. Compare fresh capped controls and all active-state hashes after restoration. '
                      'In a separate owner, stop after one queued copy and require the audited masked error exit.',
                first_copy=list(FIRST_COPY), limits=dict.fromkeys(ORDER, LIMITS))
    return plan, base, environments


def cancel_after_install(io, log):
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        io.poll()
        require(not io.exit_watch.poll(0), 'native process exited before the switch')
        position = log.tell()
        line = log.readline(16385)
        require(len(line) <= 16384, 'placement log line bound')
        if not line.endswith(b'\n'):
            log.seek(position)
            time.sleep(0.02)
        elif line.startswith(b'strata decode placement:'):
            match = EVENT.fullmatch(line.decode().strip())
            require(match and match[1] == 'blend', 'STOP trigger is not the checked install')
            io.send(b'STOP\n')
            return int(match[5])
    raise TimeoutError('no completed decode placement before STOP deadline')


def ready(io):
    info = {}
    while (line := io.line(600)).startswith('INFO '):
        info.update(field.split('=', 1) for field in line.split()[1:])
    require(line == 'READY 262144 stop' and all(info.get(k) == v for k, v in model.INFO.items()),
            'unexpected context, drafter or actual cache size')


def interact(io, plan, arm):
    ready(io)
    observed = []
    with (ROOT / arm / 'owned' / f'{arm}.stderr').open('rb') as log:
        for index, req in enumerate(plan['requests']):
            ids, cap = req['ids'], req['cap']
            data = wire(ids, cap)
            require(sha256(data).hexdigest() == req['sha256'], 'prompt wire changed')
            cancel = index == 3 and arm == 'blend'
            log.seek(0, 2)
            io.send(data)
            trigger = cancel_after_install(io, log) if cancel else None
            reader = RequestProtocol(len(ids), cap)
            first = None
            while reader.done is None:
                line = io.line(180)
                if line.startswith('T ') and first is None:
                    first = time.monotonic_ns()
                if line.startswith('DONE '):
                    fields = line.split()
                    require(len(fields) == 15, 'native DONE format changed')
                    finish = fields[5]
                    require(finish == 'cancel' if cancel else finish in ('length', 'stop'), 'unexpected finish reason')
                    require(0 < len(reader.tokens) <= cap and (finish != 'length' or len(reader.tokens) == cap),
                            'unexpected output extent')
                    reader.output_tokens = len(reader.tokens)
                    fields[5] = 'length'
                    line = ' '.join(fields)
                reader.feed(line)
            observed.append(dict(generated=len(reader.tokens), finish=finish, first_ns=first, stop_trigger_ns=trigger,
                                 output_sha256=bank.signature(reader.tokens),
                                 prefixes={str(n): bank.signature(reader.tokens[:n]) for n in range(32, 36)
                                           if n <= len(reader.tokens)},
                                 drafts=[int(reader.done[i]) for i in (6, 7)]))
    io.send(b'QUIT\n')
    io.exit(90)
    return observed


def interact_copy_failure(io, plan):
    ready(io)
    request = plan['requests'][1]
    wire_data = wire(request['ids'], request['cap'])
    require(sha256(wire_data).hexdigest() == request['sha256'], 'fault fixture prompt changed')
    io.send(wire_data)
    reader = RequestProtocol(len(request['ids']), request['cap'])
    while True:
        line = io.line(180)
        if line.startswith('ERR '):
            require(line == FAULT_ERROR and reader.reused and 32 <= len(reader.tokens) <= 35,
                    'required partial copy error after the ordinary prefix')
            break
        require(not line.startswith('DONE '), 'fault fixture completed the request')
        reader.feed(line)
    io.exit(90)
    return dict(emitted=len(reader.tokens), prefix_sha256=bank.signature(reader.tokens), error=FAULT_ERROR)


def inspect_copy_failure(text, row, sample):
    stock.check_loader(text, str(STOCK_PROFILE), per_layer=True)
    faults = list(FAULT_EVENT.finditer(text))
    require(len(faults) == text.count(FAULT_PREFIX) == 1, 'require one explicit masked-copy failure')
    fault, = faults
    emitted, slot, copied = map(int, fault.groups())
    require(emitted == sample['emitted'] and [slot, copied] == row['first_copy'],
            'copy failure geometry differs from the measured profile')
    before, after = text[:fault.start()], text[fault.end():]
    require(text.count('strata DFlash2: eager, ') == 1
            and before.count('strata trace: window ') > 0
            and not any(marker in after for marker in ('strata trace: window ', 'strata DFlash2: eager, '))
            and not any(marker in text for marker in ('strata decode placement:', WORK_PREFIX))
            and not STATE.search(text), 'publication, another window or request completion after failure')
    prepare = PREPARE.findall(before)
    require(len(prepare) == before.count('strata decode placement prepare:') == 1
            and tuple(map(int, prepare[0][:2])) == (emitted, GEOMETRY[0]),
            'require the one prepared extent before the fault')
    reference = row['outcomes']['stock']['requests'][1]
    require(reference['prefixes'][str(emitted)] == sample['prefix_sha256'],
            'pre-copy output changed before the injected failure')
    sample.update(queued_copies=1, copied_bytes=copied, first_slot=slot, drained=True, masked=True)


def verification_work(frame, emitted, accepted, offered):
    matches = WORK.findall(frame)
    require(len(matches) == frame.count(WORK_PREFIX) == 1, 'require one native verification work total')
    windows, rows, *widths, other = map(int, matches[0])
    require(windows > 0 and rows == windows + offered and windows <= rows <= 4 * windows
            and 0 <= accepted <= offered and 0 <= windows + accepted - emitted <= 3,
            'native verification work does not reconcile with draft and output counts')
    require(other == 0 and sum(widths) == windows
            and sum(width * count for width, count in enumerate(widths, 1)) == rows,
            'native target widths do not reconcile with completed verifier work')
    return dict(verify_windows=windows, verified_rows=rows, windows_by_width=widths)


def inspect_model(text, row, arm, requests):
    stock.check_loader(text, str(STOCK_PROFILE), per_layer=True)
    states = list(STATE.finditer(text))
    require(len(states) == len(requests) and 'decode phase profile inactive' not in text
            and not re.search(r'^strata mtp: .*loaded', text, re.M), 'unexpected model placement or state record')
    start = 0
    for index, (state, req) in enumerate(zip(states, requests, strict=True)):
        frame = text[start:state.start()]
        start = state.end()
        require(frame.count('strata DFlash2: eager, ') == 1, 'each request must exercise DFlash2')
        req.update(verification_work(frame, req['generated'], *req['drafts']))
        events = EVENT.findall(frame)
        prepare = PREPARE.findall(frame)
        require(len(events) == frame.count('strata decode placement:')
                and len(prepare) == frame.count('strata decode placement prepare:')
                and len(prepare) == int(arm == 'blend' and index == 1),
                'unexpected, duplicate or malformed placement preparation record')
        req['active_state'] = {k: v for k, v in re.findall(r'(\w+)=([0-9a-f,-]+)', state[0])
                               if k in model.restore.ACTIVE_STATE}
        require(set(req['active_state']) == set(model.restore.ACTIVE_STATE), 'missing active state field')
        if index not in (1, 3) or arm == 'stock':
            require(not events, 'unexpected switch for the stock or capped request')
            continue
        require([e[0] for e in events] == ['blend', 'stock'], 'each long diagnostic must switch and restore once')
        install, restored = [int(e[1]) for e in events]
        install_ns, restore_ns = [int(e[4]) for e in events]
        require(32 <= install <= 35 and install <= restored == req['generated'] and install_ns < restore_ns
                and all(tuple(map(int, e[2:4])) == GEOMETRY for e in events), 'unexpected switch position or payload')
        require(req['first_ns'] < install_ns if index == 1 else req['stop_trigger_ns'] == install_ns,
                'first native output or the STOP trigger differs')
        if prepare:
            (emitted, slots, prepare_ns, elapsed), = prepare
            require(int(emitted) == install and int(slots) == GEOMETRY[0]
                    and req['first_ns'] < int(prepare_ns) < install_ns and float(elapsed) > 0,
                    'profile preparation must begin after first native output and before installation')
            req['placement_preparation'] = dict(emitted=int(emitted), slots=int(slots),
                                                wall_ms=float(elapsed),
                                                after_first_ms=(int(prepare_ns) - req['first_ns']) / 1e6)
        control = row['outcomes']['stock']['requests'][index]
        require(req['prefixes'][str(install)] == control['prefixes'][str(install)], 'pre-switch tokens differ')
        req['transitions'] = [dict(profile=e[0], emitted=int(e[1]), wall_ms=float(e[5])) for e in events]
    require(not any(prefix in text[start:] for prefix in ('strata decode placement:',
                                                          'strata decode placement prepare:', WORK_PREFIX)),
            'placement data after the final request')
    if arm == 'blend':
        fields = ('generated', 'finish', 'output_sha256', 'active_state', 'drafts', 'verify_windows',
                  'verified_rows', 'windows_by_width')
        controls = row['outcomes']['stock']['requests']
        require(all(all(requests[i][f] == controls[i][f] for f in fields) for i in (0, 2, 4)),
                'stock output or state changed after the placement restore')
    for req in requests:
        require(req['first_ns'] is not None, 'first output missing')
        req.pop('first_ns')
        req.pop('stop_trigger_ns')


def compact(row):
    sources = row['sources']
    return {k: v for k, v in row.items() if k != 'sources'} | dict(
        source_count=len(sources), source_set_sha256=bank.signature(sources))


def save(row):
    bank.stage.write(ROOT / 'record.json', row)
    data = bank.previous.read(bank.RESULT)
    data[KEY] = compact(row)
    bank.stage.write(bank.RESULT, data)


def run(row, base, arm, environment):
    directory = ROOT / arm
    case = row['cases'][arm]
    require(not (directory / 'execution.json').exists() and bank.sha(case['binary']['path']) == case['binary']['sha256'],
            'require the fresh case and the exact executable')
    cursor = next(line.removeprefix('-- cursor: ') for line in subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15).splitlines()
        if line.startswith('-- cursor: '))
    owner = builder.ownership.Owner(directory / 'owned', lambda: bank.stage.write(
        directory / 'execution.json', dict(active=owner.active is not None, commands=owner.commands)))
    outcome = dict(arm=arm, host_probe_accepted=False)

    def interact_case(io, _entry):
        if arm == 'copy-failure':
            outcome['copy_failure'] = interact_copy_failure(io, row)
        else:
            outcome['requests'] = interact(io, row, arm)

    try:
        entry, stdout, stderr = owner.run(arm, case['command'], environment, bank.P,
            interaction=interact_case, **row['limits'][arm])
        outcome['native_exit_code'] = entry['exit_code']
        expected_exit = 1 if arm == 'copy-failure' else 0
        require(builder.ownership.closed(entry) and entry['exit_code'] == expected_exit, 'unexpected process exit')
        text = stderr.read_text()
        require(not any(line.startswith(horizon.phase.START) for line in text.splitlines()),
                'placement verification must retain the ordinary PLE row cache')
        loader = base['native_validation_probe']['loader']['path']
        outcome['validation'] = validation_log(text, entry['pid'], loader, traced=False)
        layer = directory / 'validation.log'
        require(not layer.exists() or layer.stat().st_size == 0, 'runtime validation warnings')
        outcome['providers'] = bank.device.loaded_providers(text, dict(base, environment=environment))
        if arm == 'copy-failure':
            inspect_copy_failure(text, row, outcome['copy_failure'])
        else:
            inspect_model(text, row, arm, outcome['requests'])
        models.model_pins(base['configuration'], base['model_assets'], full=True)
        require(all(bank.sha(path) == digest for path, digest in base['dflash2_assets'].items()),
                'DFlash2 model assets changed during the placement check')
        outcome['final_assets_verified'] = True
        outcome['host_probe_accepted'] = True
    except Exception as error:
        outcome['error'] = repr(error)
    finally:
        validation.finish(outcome, owner, cursor, row['boot'])
        outcome['passed'] &= outcome.get('driver_after') == row['driver']
        row['outcomes'][arm] = outcome
        complete = outcome['passed'] and arm == ORDER[-1]
        row.update(stage='closed' if complete else 'running' if outcome['passed'] else 'failed',
                   complete=complete, passed=complete, active=owner.active is not None,
                   ownership_closed=outcome['ownership_closed'])
        save(row)
    require(outcome['passed'], outcome.get('error', 'placement check failed validation or ownership'))


def main(action):
    require(sys.executable == bank.basis.reference.model.PYTHON and action in ('prepare', *ORDER),
            'use the pinned Python and one placement-check stage')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        builder.guards.no_model_owners([])
        if action == 'prepare':
            require(not ROOT.exists() and KEY not in validation.committed(bank.RESULT), 'inspect any prior check first')
            plan, _, _ = inputs(validation.git('rev-parse', 'HEAD').decode().strip())
            for arm in ORDER:
                for child in ('tmp', 'cache'):
                    (ROOT / arm / child).mkdir(mode=0o700, parents=True)
            save(dict(plan, stage='prepared', outcomes={}, complete=False, passed=False, ownership_closed=True, active=False))
            return
        row = bank.previous.read(ROOT / 'record.json')
        plan, base, env = inputs(row['source_commit'])
        require(compact(row) == validation.committed(bank.RESULT, (KEY,)) and all(row[k] == v for k, v in plan.items())
                and row['stage'] in ('prepared', 'running') and row['ownership_closed'] and not row['active']
                and list(row['outcomes']) == list(ORDER[:ORDER.index(action)]), 'commit the exact next case')
        run(row, base, action, env[action])


if __name__ == '__main__':
    main(*sys.argv[1:])
