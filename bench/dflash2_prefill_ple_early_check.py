"""Qualify the early PLE schedule on the actual DFlash2 serving path."""

import fcntl
import math
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import dflash2_prefill_ple_early_build as build
import dflash2_primary_tap_payload as tap

bank, require = build.bank, build.require
validation, model, owners = tap.validation, tap.model, tap.owners
THIS = str(Path(__file__).resolve().relative_to(bank.P))
KEY = 'dflash2_prefill_ple_early_check_v3'
ROOT = bank.B / KEY.replace('_', '-')
REFERENCE = '63c0c6353737c2fb4dc5ade7b1d6aa68da59531d'
ORDER = ('cancel', 'direct')
SELECTOR = 'STRATA_SYCL_PREFILL_PLE_EARLY_READ'
LAYER0 = 'strata trace: layer-major layer 0 of 48: loading 512 experts'
LAYER1 = 'strata trace: layer-major layer 1 of 48: loading 512 experts'
ISSUED = 'strata trace: prefill PLE early read issued layer=0'
CONSUMED = 'strata prefill PLE early read: rows=8192 issue_layer=0 consume_layer=1'
STOP_DEADLINE = 150


def inputs(revision, ctx):
    native = bank.previous.read(build.ROOT / 'record.json')
    require(build.wait.retained.closed(native, 'built'), 'require the complete early PLE native build')
    ctx.check(native)
    reference = bank.previous.historic(REFERENCE, tap.KEY)
    require(build.wait.retained.closed(reference, 'closed')
            and reference['outcomes']['direct']['passed'], 'recover the original direct-writer parity')
    expected = reference['outcomes']['direct']
    old, base, envs, requests = tap.inputs(revision, base=build.wait.retained.baseline())
    require(all(reference['plan'][k] == old[k] for k in (
        'binary', 'command', 'cases', 'driver', 'boot', 'runtime_manifest',
        'native_validation_probe', 'dflash2_assets', 'model_assets_sha256')),
        'the original native request, model or runtime inputs differ')
    pins = build.source.pin(bank.P, [*native['controller_sources'], tap.THIS, THIS])['source_sha256']
    require(bank.stage.hashed(dict(build.source.blobs(bank.P, revision, pins))) == pins,
            'commit the exact early-read checker inputs')
    environments = {}
    cases = {}
    direct = old['cases']['direct']
    for name in ORDER:
        env = {k: v.replace(str(tap.ROOT / 'direct'), str(ROOT / name)) for k, v in envs['direct'].items()}
        env[SELECTOR] = '1'
        environments[name] = env
        cases[name] = dict(direct, environment_sha256=bank.signature(env))
    plan = {k: old[k] for k in (
        'boot', 'driver', 'runtime_manifest', 'native_validation_probe', 'dflash2_assets', 'model_assets_sha256')}
    binary = native['binaries']['strata']
    plan.update(source_commit=revision, source_inputs_sha256=bank.signature(pins), reference=REFERENCE,
                native_source_commit=native['native_source_commit'],
                native_build_sha256=bank.signature(build.builder.compact(native)),
                binary=binary, command=[binary['path'], *old['command'][1:]], cases=cases, order=list(ORDER),
                stop_deadline_s=STOP_DEADLINE,
                scope='Send STOP after the traced layer-0 gather issue, then repeat the short '
                      'request on the same server. A fresh process completes the original direct-writer '
                      'short/long/short requests and checks target state, taps and token IDs.')
    return plan, base, environments, requests, expected


def stop_at_layer_zero(io, log, ids, output):
    offset = log.stat().st_size
    data = model.wire(ids, output)
    for p in range(0, len(data), 256 << 10):
        io.send(data[p:p + (256 << 10)])
    marker = (ISSUED + '\n').encode()
    deadline = time.monotonic() + STOP_DEADLINE
    last = b''
    with log.open('rb') as stream:
        stream.seek(offset)
        while marker not in last:
            io.poll()
            require(time.monotonic() < deadline and not io.exit_watch.poll(0),
                    'did not observe the first gather issue before cancellation')
            block = stream.read(64 << 10)
            if block:
                last = last[-len(marker):] + block
            else:
                time.sleep(0.02)
    io.send(b'STOP\n')
    require(io.line(90) == 'RESUME 0' and io.line(90) == 'REUSED 0',
            'expected a fresh request cancelled before completed prefill')
    done = io.line(90).split()
    require(len(done) == 15 and done[:3] == ['DONE', '0', str(len(ids))]
            and done[5:9] == ['cancel', '0', '0', '0'] and done[14] == '0'
            and all(math.isfinite(float(done[i])) and float(done[i]) >= 0 for i in (3, 4, 13))
            and all(done[i].isascii() and done[i].isdecimal() for i in range(9, 13)),
            'invalid cancellation extent, clock or output')
    return dict(finish='cancel', generated=0, prompt=len(ids), read=0, prompt_ms=float(done[3]))


def inspect(log, pid, plan, name, responses, reference):
    requests = list(tap.taps.REQUEST.finditer(log))
    require([tuple(map(int, match.groups())) for match in requests] == [(p, 0) for p in tap.taps.PROMPTS],
            'unexpected actual native diagnostic prompts')
    middle = log[requests[1].start():requests[2].start()]
    require(log.count('strata generate: PLE on, table ') == 1 and middle.count(LAYER0) == 1
            and middle.count(ISSUED) == log.count(ISSUED) == 1 and middle.index(ISSUED) < middle.index(LAYER0)
            and 'strata prefill dense: phase=bank residency=Bank activations=0 ' in middle
            and 'strata MoE window configuration: requested_mode=16 effective_mode=16 maximum_rows=122880 ' in middle,
            'the long request did not issue the read on the dense-bank mode-16 route')
    fields = ('requests', 'states', 'tap_payloads')
    if name == 'direct':
        require(middle.count(LAYER1) == 1 and middle.count(CONSUMED) == log.count(CONSUMED) == 1
                and middle.index(LAYER1) < middle.index(CONSUMED)
                and 'strata prefill dense: phase=restored residency=Full activations=48 ' in middle,
                'first gather did not reach layer 1 or the dense phase did not restore')
        evidence = tap.inspect(log, pid, plan, 'direct', responses)
        outcome = dict(evidence, requests=responses)
        require(all(outcome[k] == reference[k] for k in (*fields, 'route')),
                'the scheduling change altered output, active state, draft taps or work')
        return outcome
    require(LAYER1 not in middle and CONSUMED not in log and '(cancelled)' in middle,
            'expected cancellation before consuming the first gather')
    wire = plan['cases'][name]['wire']
    shortened = dict(plan, cases={'smoke': {'wire': [wire[0], wire[2]]}})
    evidence = tap.inspect(log[:requests[1].start()] + log[requests[2].start():], pid, shortened,
                           'smoke', [responses[0], responses[2]])
    outcome = dict(evidence, requests=[responses[0], responses[2]], cancelled=responses[1])
    require(all(outcome[k] == [reference[k][0], reference[k][2]] for k in fields),
            'the server did not recover the original short request after STOP')
    return outcome


def save(row):
    bank.stage.write(ROOT / 'record.json', row)
    result = bank.previous.read(bank.RESULT)
    result[KEY] = row
    bank.stage.write(bank.RESULT, result)


def run(row, base, name, env, requests, expected):
    plan = row['plan']
    root = ROOT / name
    receipt = root / 'execution.json'
    require(not receipt.exists(), 'existing owned early-read diagnostic')
    cursor = next(line.removeprefix('-- cursor: ') for line in subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'],
        text=True, timeout=15).splitlines() if line.startswith('-- cursor: '))
    outcome = dict(complete=False, host_probe_accepted=False)
    responses = []

    def persist():
        bank.stage.write(receipt, dict(active=bool(owner.active), commands=owner.commands))

    def interact(io, _entry):
        info = {}
        line = io.line(600)
        while line.startswith('INFO '):
            info.update(word.split('=', 1) for word in line.split()[1:])
            line = io.line(600)
        require(line == 'READY 262144 stop' and all(info.get(k) == v for k, v in model.INFO.items()),
                'native context or placement changed')
        for i, (ids, n) in enumerate(requests):
            if name == 'cancel' and i == 1:
                responses.append(stop_at_layer_zero(io, root / 'owned' / (name + '.stderr'), ids, n))
            else:
                responses.append(model.request(io, ids, n))
        io.send(b'QUIT\n')
        io.exit(90)

    owner = owners.Owner(root / 'owned', persist)
    try:
        entry, _, err = owner.run(name, plan['command'], env, bank.P, interaction=interact, **plan['cases'][name]['limits'])
        outcome['native_exit_code'] = entry['exit_code']
        require(owners.closed(entry) and entry['exit_code'] == 0
                and (root / 'validation.log').stat().st_size == 0, 'native or validation failure')
        outcome['native_identity'] = {k: entry['direct_identity'][k] for k in ('pid', 'start_ticks')}
        outcome.update(inspect(err.read_text(), entry['pid'], plan, name, responses, expected))
        require(all(bank.sha(p) == h for p, h in plan['dflash2_assets'].items()), 'draft assets changed')
        tap.basis.comparison.shared.models.model_pins(base['configuration'], base['model_assets'], full=name == 'direct')
        outcome.update(complete=True, host_probe_accepted=True)
    except BaseException as error:
        outcome['error'] = repr(error)
    finally:
        validation.finish(outcome, owner, cursor, plan['boot'])
        outcome['passed'] = outcome['passed'] and outcome.get('driver_after') == plan['driver']
        row['outcomes'][name] = outcome
        closed = outcome['passed'] and name == ORDER[-1]
        row.update(stage='closed' if closed else 'partial' if outcome['passed'] else 'failed',
                   complete=closed, passed=closed, active=bool(owner.active),
                   ownership_closed=outcome['ownership_closed'])
        save(row)
    print({k: outcome.get(k) for k in ('passed', 'native_exit_code', 'fresh_faults', 'error')}, flush=True)
    require(outcome['passed'], outcome.get('error') or 'early-read diagnostic did not pass')


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(sys.executable == tap.basis.reference.model.PYTHON and action in ('prepare', *ORDER),
            'select a planned early-read stage with the pinned interpreter')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        ctx = build.context()
        if action == 'prepare':
            require(not ROOT.exists() and KEY not in validation.committed(bank.RESULT), 'fresh diagnostic required')
            revision = validation.git('rev-parse', 'HEAD').decode().strip()
            plan, *_ = inputs(revision, ctx)
            for name in ORDER:
                for folder in ('cache', 'tmp'):
                    (ROOT / name / folder).mkdir(parents=True, mode=0o700)
            save(dict(plan=plan, stage='prepared', outcomes={}, active=False, complete=False, passed=False,
                      ownership_closed=True, speed_eligible=False))
        else:
            row = bank.previous.read(ROOT / 'record.json')
            require(row == validation.committed(bank.RESULT, (KEY,)) and not row['active']
                    and row['ownership_closed'] and tuple(row['outcomes']) == ORDER[:ORDER.index(action)]
                    and row['stage'] in ('prepared', 'partial'), 'commit the preceding successful stage')
            plan, base, environments, requests, expected = inputs(row['plan']['source_commit'], ctx)
            require(plan == row['plan'], 'frozen early-read diagnostic inputs changed')
            run(row, base, action, environments[action], requests, expected)
