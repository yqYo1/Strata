"""Compare the primary DFlash2 tap payloads with the two attention writers."""

import fcntl
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
import dflash2_primary_prefill_curve as curve
import dflash2_primary_prefill_speed as speed
import dflash2_v2_model as model
import dflash2_tap_payloads as taps
from engine_once import native_environment

bank, basis, require = speed.bank, speed.basis, speed.require
validation, owners = basis.validation, model.owner_module
KEY = 'dflash2_primary_tap_payload_v2'
ROOT = bank.B / KEY.replace('_', '-')
THIS = str(Path(__file__).resolve().relative_to(bank.P))
REFERENCE = '321c5776ab4c73d502529d4a4f23f7175f39bfd6'
REFERENCE_KEY = 'dflash2_tap_payload_model_v1'
PREVIOUS_ATTEMPT = '1cfd63cd0a3d5e847b5c96c4b5b3e9411fbc2964'
WRITER = 'STRATA_SYCL_PREFILL_DIRECT_ATTENTION_STORE'
ORDER = ('smoke', 'copy', 'direct')
LIMITS = dict(model.LIMITS)


def inputs(revision):
    row = curve.baseline()
    prior = bank.previous.historic(REFERENCE, REFERENCE_KEY)
    stopped = bank.previous.historic(PREVIOUS_ATTEMPT, 'dflash2_primary_tap_payload_v1')
    require(stopped['stage'] == 'failed' and stopped['ownership_closed'] and not stopped['active']
            and stopped['plan']['binary'] == row['binary']
            and stopped['outcomes']['smoke']['error'] == "RuntimeError('missing target state or DFlash2 verifier work')",
            'preserve the original smoke rejection')
    require(row['selected_arm'] == 'retain' and prior['stage'] == 'closed' and prior['passed']
            and prior['complete'] and prior['ownership_closed'] and not prior['active']
            and not prior['comparison']['tap_differences'] and prior['comparison']['correct_writer'],
            'recover the selected primary KV arm and original tap payload control')
    short = model.request_ids(Path(row['configuration']['tokenizer']))['ids']
    long = [i % 4096 for i in range(speed.READ - len(short))] + short
    requests = [(ids, 32) for ids in (short, long, short)]
    wire = [dict(read=len(ids), output=n, sha256=model.restore.signature_bytes(model.wire(ids, n)))
            for ids, n in requests]
    require(tuple(x['read'] for x in wire) == taps.PROMPTS
            and wire == prior['plan']['cases']['copy']['wire'], 'recover the original request bytes')
    args = row['validated_dflash2_args']
    expected = prior['plan']['cases']['copy']['command'][2:]
    rank = str(bank.P / 'data/expert-profile.bin')
    require(args == [rank if arg == expected[expected.index('--expert-profile') + 1] else arg
                     for arg in expected], 'native inference flags changed')
    command = [row['binary']['path'], '--serve', *args]
    sources = speed.experiment().pins(revision)
    for module in (sys.modules[__name__], curve, model, taps):
        path = Path(module.__file__).resolve()
        name = path.relative_to(bank.P).as_posix()
        require(path.read_bytes() == validation.git('show', revision + ':' + name),
                'commit exact tap diagnostic input: ' + name)
        sources[name] = bank.sha(path)

    environments, cases = {}, {}
    for name in ORDER:
        root = str(ROOT / name)
        env = speed.environment(row, dict(root=root, index=0, arm='retain'))
        env.update({WRITER: str(int(name == 'direct')), 'STRATA_DFLASH2_TAP_PAYLOAD_AUDIT': '1',
                    'STRATA_TRACE': '1', 'STRATA_STATE_HASH': '1'})
        environments[name] = native_environment(row, env)
        limits = dict(LIMITS, wall=300, cpu=1200, text_cap=4 << 20) if name == 'smoke' else LIMITS
        cases[name] = dict(wire=wire[:1] if name == 'smoke' else wire, limits=limits,
                           environment_sha256=bank.signature(environments[name]))
    normalized = [{k: v.replace(str(ROOT / name), '<case>') for k, v in env.items() if k != WRITER}
                  for name, env in environments.items()]
    require(all(value == normalized[0] for value in normalized), 'another native setting differs')
    plan = {k: row[k] for k in ('native_source_commit', 'native_build_sha256', 'binary', 'boot', 'driver',
                               'runtime_manifest', 'native_validation_probe', 'dflash2_assets', 'selection_sha256')}
    plan.update(source_commit=revision, source_inputs=dict(count=len(sources), sha256=bank.signature(sources)),
                reference=REFERENCE, previous_attempt=PREVIOUS_ATTEMPT, order=list(ORDER), command=command, cases=cases,
                model_assets_sha256=bank.signature(row['model_assets']),
                scope='Short observer smoke, then fresh copy and direct processes. Require matching five tap '
                      'digests, active target state, expert/KV work and output. Diagnostic clocks only. '
                      'Preserve the first failed outcome and stop.')
    return plan, row, environments, requests


def inspect(log, pid, plan, name, responses):
    proof = validation.preload_evidence(log, pid)
    require(proof['loader'] == plan['native_validation_probe']['loader']['path']
            and log.count('strata native head DPAS: ready=1 columns=4 layout=canonical') == 1
            and not re.search(r'(?m)^strata mtp: .*loaded|UR_RESULT_ERROR_|ZE_RESULT_ERROR_|teardown uncertain', log),
            'unexpected MTP or runtime validation')
    prompts = tuple(item['read'] for item in plan['cases'][name]['wire'])
    payloads = taps.read_fresh_tap_payloads(log, prompts)
    for payload in payloads:
        payload['layer_digests'] = {str(k): v for k, v in payload['layer_digests'].items()}
    states = []
    for line in re.findall(r'^strata serve: STATE_HASH L=.*$', log, re.M):
        state = dict(re.findall(r'(\w+)=([0-9a-f,-]+)', line))
        states.append({k: state[k] for k in model.restore.ACTIVE_STATE})
    # The native STATE_HASH is emitted after committing the generated tokens.
    require([r['prompt'] for r in responses] == list(prompts)
            and [int(state['L']) for state in states] == [r['prompt'] + r['generated'] for r in responses]
            and all(r['offered'] > 0 for r in responses), 'missing target state or DFlash2 verifier work')
    evidence = dict(tap_payloads=payloads, states=states, validation=proof)
    if name != 'smoke':
        require(payloads[0] == payloads[-1] and states[0] == states[-1] and responses[0] == responses[-1],
                'the final short request did not reproduce the first')
        _, start, end = [match.start() for match in taps.REQUEST.finditer(log)]
        frame = log[start:end]
        route = speed.route(frame, dict(arm='retain'), direct=name == 'direct')
        first = re.search(r'^strata trace: window (\d+) (\d+)$', frame, re.M)
        require(first and tuple(map(int, first.groups())) == (speed.READ - 1, 1)
                and frame.index('strata DFlash2: eager, ') < first.start(),
                'tap ingestion or verifier handoff changed')
        long = responses[1]
        require(route['native_request'] == [speed.READ, 0, speed.READ, long['generated'],
                                               long['accepted'], long['offered']], 'client/native work differs')
        evidence['route'] = route
    return evidence


def compare(outcomes, name, outcome):
    smoke = outcomes['smoke']
    require(all(result['native_identity'] != outcome['native_identity'] for result in outcomes.values()),
            'expected independent native executables')
    fields = ('requests', 'states', 'tap_payloads')
    require(all(smoke[k] == outcome[k][:1] for k in fields), 'fresh short observer control changed')
    if name == 'direct':
        copy = outcomes['copy']
        require(all(copy[k] == outcome[k] for k in fields), 'copy/direct tap bytes, state or output differ')
        keep = lambda route: {k: v for k, v in route.items() if k != 'direct_pieces'}
        require(keep(copy['route']) == keep(outcome['route']), 'copy/direct expert, KV or verifier work differs')


def save(row):
    bank.stage.write(ROOT / 'record.json', row)
    result = bank.previous.read(bank.RESULT)
    result[KEY] = row
    bank.stage.write(bank.RESULT, result)


def run(row, base, name, env, requests):
    plan = row['plan']
    root = ROOT / name
    require(not (root / 'execution.json').exists(), 'existing owner receipt')
    cursor = next(line.removeprefix('-- cursor: ') for line in subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'],
        text=True, timeout=15).splitlines() if line.startswith('-- cursor: '))
    outcome = dict(complete=False, host_probe_accepted=False, requests=[])

    def persist():
        bank.stage.write(root / 'execution.json', dict(active=bool(owner.active), commands=owner.commands))

    def interact(io, _entry):
        info = {}
        line = io.line(600)
        while line.startswith('INFO '):
            info.update(word.split('=', 1) for word in line.split()[1:])
            line = io.line(600)
        require(line == 'READY 262144 stop' and all(info.get(k) == v for k, v in model.INFO.items()),
                'native context or placement changed')
        for ids, n in requests:
            outcome['requests'].append(model.request(io, ids, n))
        io.send(b'QUIT\n')
        io.exit(90)

    owner = owners.Owner(root / 'owned', persist)
    try:
        entry, _, err = owner.run(name, plan['command'], env, bank.P, interaction=interact, **plan['cases'][name]['limits'])
        outcome['native_exit_code'] = entry['exit_code']
        require(owners.closed(entry) and entry['exit_code'] == 0, 'native process failed')
        require((root / 'validation.log').stat().st_size == 0, 'UR validation output was not quiet')
        outcome['native_identity'] = {k: entry['direct_identity'][k] for k in ('pid', 'start_ticks')}
        outcome.update(inspect(err.read_text(), entry['pid'], plan, name, outcome['requests']))
        if name != 'smoke':
            compare(row['outcomes'], name, outcome)
        require(all(bank.sha(p) == h for p, h in plan['dflash2_assets'].items()), 'draft assets changed')
        basis.comparison.shared.models.model_pins(base['configuration'], base['model_assets'], full=name == 'direct')
        outcome.update(complete=True, host_probe_accepted=True)
    except BaseException as error:
        outcome['error'] = repr(error)
    finally:
        validation.finish(outcome, owner, cursor, plan['boot'])
        outcome['passed'] = outcome['passed'] and outcome['driver_after'] == plan['driver']
        row['outcomes'][name] = outcome
        closed = outcome['passed'] and name == 'direct'
        row.update(stage='closed' if closed else 'partial' if outcome['passed'] else 'failed',
                   complete=closed, passed=closed, active=bool(owner.active),
                   ownership_closed=outcome['ownership_closed'])
        save(row)
    print({k: outcome.get(k) for k in ('passed', 'native_exit_code', 'fresh_faults', 'error')}, flush=True)
    require(outcome['passed'], outcome.get('error') or 'tap observer check failed')


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(sys.executable == basis.reference.model.PYTHON and action in ('prepare', *ORDER),
            'select a planned tap diagnostic stage with the pinned interpreter')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        if action == 'prepare':
            require(not ROOT.exists() and KEY not in validation.committed(bank.RESULT), 'fresh tap diagnostic plan required')
            revision = validation.git('rev-parse', 'HEAD').decode().strip()
            plan, *_ = inputs(revision)
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
            plan, base, environments, requests = inputs(row['plan']['source_commit'])
            require(plan == row['plan'], 'frozen tap diagnostic inputs changed')
            run(row, base, action, environments[action], requests[:1] if action == 'smoke' else requests)
