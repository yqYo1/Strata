"""Check copy retirement and live decode-only placement on the same build."""

import fcntl
from hashlib import sha256
from pathlib import Path
import re
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import dflash2_decode_phase_placement_build as build
import dflash2_decode_stock_global_restore as stock
import dflash2_primary_checks as fixtures
from check_dflash2_model import request_ids, wire
from engine_once import native_environment
from native_request_protocol import RequestProtocol
from run_q6_columns import validation_log

bank, builder, require, validation = build.bank, build.builder, build.require, build.validation
model = build.retained.speed.generation.model
KEY = 'dflash2_decode_phase_placement_check_v1'
ROOT = bank.B / KEY.replace('_', '-')
THIS = Path(__file__).resolve().relative_to(bank.P).as_posix()
RANKED = 'a57499a15ed571fa3c2456409fb41c6b371c3e39'
ORDER = ('host', 'arc', 'stock', 'blend')
GEOMETRY = (979, 2_010_419_200)
SELECTOR = 'STRATA_SYCL_DFLASH2_DECODE_PHASE_PROFILE'
AUDIT = 'STRATA_SYCL_DFLASH2_DECODE_PHASE_AUDIT'
CLOCK = 'STRATA_SYCL_DFLASH2_DECODE_PHASE_CLOCK'
EVENT = re.compile(r'^strata decode placement: profile=(blend|stock) emitted=(\d+) slots=(\d+) '
                   r'h2d_bytes=(\d+) start_ns=(\d+) wall_ms=(\d+\.\d+) audit=1$', re.M)
STATE = re.compile(r'^strata serve: STATE_HASH L=.*$', re.M)
LIMITS = dict(wall=720, cpu=1800, rss_cap=104 << 30, text_cap=12 << 20,
              total_cap=512 << 20, file_cap=12 << 20)


def inputs(revision):
    native = bank.previous.read(build.ROOT / 'record.json')
    require(build.retained.closed(native, 'built'), 'finish the coherent placement build')
    original = build.check(native)
    base = build.retained.baseline()
    prior = bank.previous.historic(RANKED, 'dflash2_decode_ranked_refill_speed_v1')
    profiles = prior['qualified_profiles']
    require(build.retained.closed(prior, 'closed') and set(profiles) == {'stock', 'blend'}
            and all(bank.sha(p['path']) == p['sha256'] for p in profiles.values()),
            'recover both qualified 1256-slot profiles')
    controllers = (THIS, stock.THIS, stock.primary.THIS, stock.quota.THIS,
                   'tools/make_profile.py', 'tools/strata_tokenizer.py')
    pins = bank.stage.source.pin(bank.P, controllers)['source_sha256']
    require(bank.stage.hashed(dict(build.source.blobs(bank.P, revision, pins))) == pins,
            'commit the exact model checker and tokenizer')
    args = list(base['validated_dflash2_args'])
    for flag, original_value, value in (('--expert-cache', '1256', '968'),
            ('--expert-profile', base['configuration']['args'][base['configuration']['args'].index('--expert-profile') + 1],
             profiles['stock']['path'])):
        require(args.count(flag) == 1 and args[args.index(flag) + 1] == original_value,
                'unexpected qualified CLI: ' + flag)
        args[args.index(flag) + 1] = value
    require('--expert-cache-per-layer' in args and '--mmap-experts' not in args
            and args[args.index('--adapt-swaps') + 1] == '0' and args[args.index('--prompt-cache') + 1] == '1',
            'require the fixed global cache, full RAM arena and live-state hashing')
    args.remove('--expert-cache-per-layer')
    ids = request_ids(Path(base['configuration']['tokenizer']))['ids']
    require(len(ids) == 82, 'the short thinking prompt changed')
    requests = [dict(ids=[i, *ids], cap=n, sha256=sha256(wire([i, *ids], n)).hexdigest())
                for i, n in enumerate((32, 640, 32, 640, 32), 1)]
    envelope = fixtures.parameters(native, original['native_validation_probe'])
    cases, environments = {}, {}
    for index, arm in enumerate(ORDER):
        directory = ROOT / arm
        if arm in ('host', 'arc'):
            target = 'prefill_scratch_host_test' if arm == 'host' else 'sycl_prefill_scratch_test'
            binary = native['binaries'][target]
            command = [binary['path']] if arm == 'host' else [binary['path'], '--case', 'loan']
            env = dict(native['recipe']['environment'] if arm == 'host' else envelope['environment'],
                       TMPDIR=str(directory / 'tmp'), XDG_CACHE_HOME=str(directory / 'cache'))
        else:
            binary = native['binaries']['strata']
            command = [binary['path'], '--serve', *args]
            env = build.retained.speed.environment(base, dict(root=str(directory), index=index, arm='retain'))
            env = native_environment(base, env)
            env.update(STRATA_TRACE='1', STRATA_STATE_HASH='1', STRATA_SYCL_DECODE_T4_SCHEDULE='stock',
                       STRATA_SYCL_DECODE_T4_PREFIX_HELP='0', **{CLOCK: '1'})
            if arm == 'blend':
                env.update({SELECTOR: profiles['blend']['path'], AUDIT: '1'})
        if arm != 'host':
            env['UR_LOG_VALIDATION'] = f'level:warning;flush:warning;output:file,{directory / "validation.log"}'
        cases[arm] = dict(binary=binary, command=command, environment_sha256=bank.signature(env))
        environments[arm] = env
    plan = dict(source_commit=revision, sources=pins, order=list(ORDER), cases=cases, requests=requests,
                boot=base['boot'], driver=base['driver'], native_source_commit=native['native_source_commit'],
                build_sha256=bank.signature(builder.compact(native)), profiles=profiles,
                scope='Host lease and traced Arc copies, then ordered native fresh requests. Complete one switch, '
                      'cancel one after installation, and check stock output/state after each restore. '
                      'Both arms use stock during prefill and the original stock T4 schedule.',
                limits=dict(host=bank.device.HOST, arc=bank.device.SMALL, stock=LIMITS, blend=LIMITS))
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


def interact(io, plan, arm):
    info = {}
    while (line := io.line(600)).startswith('INFO '):
        info.update(field.split('=', 1) for field in line.split()[1:])
    require(line == 'READY 262144 stop' and all(info.get(k) == v for k, v in model.INFO.items()),
            'unexpected context, drafter or actual cache size')
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


def inspect_model(text, row, arm, requests):
    stock.check_loader(text, row['profiles']['stock']['path'], per_layer=False)
    states = list(STATE.finditer(text))
    require(len(states) == len(requests) and 'decode phase profile inactive' not in text
            and not re.search(r'^strata mtp: .*loaded', text, re.M), 'unexpected model placement or state record')
    start = 0
    for index, (state, req) in enumerate(zip(states, requests, strict=True)):
        frame = text[start:state.start()]
        start = state.end()
        require(frame.count('strata DFlash2: eager, ') == 1, 'each request must exercise DFlash2')
        events = EVENT.findall(frame)
        require(len(events) == frame.count('strata decode placement:'), 'malformed placement record')
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
        control = row['outcomes']['stock']['requests'][index]
        require(req['prefixes'][str(install)] == control['prefixes'][str(install)], 'pre-switch tokens differ')
        req['transitions'] = [dict(profile=e[0], emitted=int(e[1]), wall_ms=float(e[5])) for e in events]
    require('strata decode placement:' not in text[start:], 'placement after the final request')
    if arm == 'blend':
        fields = ('generated', 'finish', 'output_sha256', 'active_state', 'drafts')
        controls = row['outcomes']['stock']['requests']
        require(all(all(requests[i][f] == controls[i][f] for f in fields) for i in (0, 2, 4)),
                'stock output or state changed after the placement restore')
    for req in requests:
        req.pop('first_ns')
        req.pop('stop_trigger_ns')


def save(row):
    bank.stage.write(ROOT / 'record.json', row)
    data = bank.previous.read(bank.RESULT)
    data[KEY] = row
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
        outcome['requests'] = interact(io, row, arm)

    try:
        entry, stdout, stderr = owner.run(arm, case['command'], environment, bank.P,
            interaction=interact_case if arm in ('stock', 'blend') else None, **row['limits'][arm])
        outcome['native_exit_code'] = entry['exit_code']
        require(builder.ownership.closed(entry) and entry['exit_code'] == 0, 'unexpected process exit')
        text = stderr.read_text()
        if arm == 'host':
            require(re.search(r'^prefill_scratch_fixture cases=\d+ status=PASS$', stdout.read_text(), re.M) and not text,
                    'incomplete host lease fixture')
        else:
            loader = base['native_validation_probe']['loader']['path']
            outcome['validation'] = validation_log(text, entry['pid'], loader, traced=arm == 'arc')
            layer = directory / 'validation.log'
            require(not layer.exists() or layer.stat().st_size == 0, 'runtime validation warnings')
            outcome['providers'] = bank.device.loaded_providers(text, dict(base, environment=environment))
            if arm == 'arc':
                require('prefill_cache_loan_fixture cases=20 cycles=3 model_PP=UNRUN status=PASS' in stdout.read_text(),
                        'missing the checked lease fixture coverage')
            else:
                inspect_model(text, row, arm, outcome['requests'])
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
        require(row == validation.committed(bank.RESULT, (KEY,)) and all(row[k] == v for k, v in plan.items())
                and row['stage'] in ('prepared', 'running') and row['ownership_closed'] and not row['active']
                and list(row['outcomes']) == list(ORDER[:ORDER.index(action)]), 'commit the exact next case')
        run(row, base, action, env[action])


if __name__ == '__main__':
    main(*sys.argv[1:])
