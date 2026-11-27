"""Check the stock and two split schedules on the same DFlash2 model build."""

import fcntl
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
import dflash2_decode_t4_check as arc
import dflash2_primary_tap_payload as tap

build, bank, require, validation = arc.build, arc.bank, arc.require, arc.validation
KEY = 'dflash2_decode_t4_model_v1'
ROOT = bank.B / KEY.replace('_', '-')
THIS = str(Path(__file__).resolve().relative_to(bank.P))
ORDER = ('stock', 'after-cpu', 'early')
T4 = re.compile(r'^DECODE_T4 mode=(\S+) position=(\d+) layers=(\d+) pool_ready=(\d+) early=(\d+)$', re.M)
Q6_CONTROL = False
V2_OBSERVATION = 'e36f293d5'


def inputs(revision):
    native = bank.previous.read(build.ROOT / 'record.json')
    build.check(native)
    device = validation.committed(bank.RESULT, (arc.KEY,))
    signature = bank.signature(build.builder.compact(native))
    require(build.retained.closed(device, 'closed') and device['native_build_sha256'] == signature
            and device['binary'] == native['binaries'][build.fixture.TARGET], 'pass the same-build Arc fixture first')
    old, base, envs, requests = tap.inputs(revision, base=build.retained.baseline())
    pins = bank.stage.source.pin(bank.P, [THIS, tap.THIS, arc.THIS, build.THIS])['source_sha256']
    require(bank.stage.hashed(dict(build.source.blobs(bank.P, revision, pins))) == pins,
            'commit the exact T4 model checker')
    binary = native['binaries']['strata']
    args = old['command'][1:]
    require(args[args.index('--pcie-frac') + 1] == '0' and args[args.index('--spec') + 1] == '4'
            and '--spec-split' not in args and '--no-host-worker' not in args,
            'require the measured single-card DFlash2 dispatch flags')
    env, cases = {}, {}
    for name in ORDER:
        env[name] = {k: v.replace(str(tap.ROOT / 'smoke'), str(ROOT / name))
                     for k, v in envs['smoke'].items()}
        env[name].update(STRATA_SYCL_DECODE_T4_SCHEDULE=name, STRATA_SYCL_DECODE_T4_AUDIT='1')
        if Q6_CONTROL:
            require(env[name]['STRATA_SYCL_DECODE_Q6_DPAS_DENSE'] == '1', 'require the original dense DPAS setting')
            env[name]['STRATA_SYCL_DECODE_Q6_DPAS_DENSE'] = '0'
        cases[name] = dict(old['cases']['smoke'], environment_sha256=bank.signature(env[name]))
    plan = {k: old[k] for k in ('boot', 'driver', 'runtime_manifest', 'native_validation_probe',
                                'dflash2_assets', 'model_assets_sha256')}
    plan.update(source_commit=revision, inputs_sha256=bank.signature(pins), arc_evidence_sha256=bank.signature(device),
                native_source_commit=native['native_source_commit'], native_build_sha256=signature,
                binary=binary, command=[binary['path'], *args], cases=cases, order=list(ORDER),
                scope='Bounded actual DFlash2 decode with no per-request option overrides. Use the original '
                      '82-input/32-output request. Verify actual T4 dispatch, active target state, '
                      'tap payloads and output before collecting integrated performance.')
    if Q6_CONTROL:
        original = bank.previous.historic(V2_OBSERVATION, 'dflash2_decode_t4_model_v1')
        require(original['stage'] == 'closed' and original['ownership_closed']
                and all(original['outcomes'][k]['passed'] for k in ORDER)
                and original['plan']['binary'] == binary
                and all(old['cases']['smoke']['wire'] == original['plan']['cases'][k]['wire'] for k in ORDER),
                'reproduce the two original arm inputs')
        legacy = bank.B / 'dflash2-decode-t4-model-v1'
        for name in ORDER:
            same_case = {k: v.replace(str(ROOT / name), str(legacy / name)) for k, v in env[name].items()}
            same_case['STRATA_SYCL_DECODE_Q6_DPAS_DENSE'] = '1'
            require(bank.signature(same_case) == original['plan']['cases'][name]['environment_sha256'],
                    'the Q6 intervention must change only dense DPAS')
        plan.update(dense_dpas='0', reference=V2_OBSERVATION,
                    scope='Compare the stock and deferred schedules with dense DPAS disabled in both. '
                          'Keep the same executable, request, runtime and other kernel selections. '
                          'Compare the short-request state digests with each other and their original DPAS-on arms.')
    return plan, base, env, requests[0]


def save(row):
    bank.stage.write(ROOT / 'record.json', row)
    results = bank.previous.read(bank.RESULT)
    results[KEY] = row
    bank.stage.write(bank.RESULT, results)


def run(row, model_base, name, env, request):
    plan = row['plan']
    case = ROOT / name
    receipt = case / 'execution.json'
    require(not receipt.exists(), 'existing model owner receipt')
    cursor, = [line.removeprefix('-- cursor: ') for line in subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True,
        timeout=15).splitlines() if line.startswith('-- cursor: ')]
    observed = dict(requests=[], host_probe_accepted=False)

    def interact(io, _):
        info = {}
        line = io.line(180)
        while line.startswith('INFO '):
            info.update(word.split('=', 1) for word in line.split()[1:])
            line = io.line(180)
        require(line == 'READY 262144 stop' and all(info.get(k) == v for k, v in tap.model.INFO.items()),
                'native context or placement changed')
        observed['requests'].append(tap.model.request(io, *request))
        io.send(b'QUIT\n')
        io.exit(60)

    owner = tap.owners.Owner(case / 'owned', lambda: bank.stage.write(
        receipt, dict(active=bool(owner.active), commands=owner.commands)))
    try:
        entry, _, err = owner.run(name, plan['command'], env, bank.P, interaction=interact,
                                  **plan['cases'][name]['limits'])
        observed['native_exit_code'] = entry['exit_code']
        require(tap.owners.closed(entry) and entry['exit_code'] == 0
                and (case / 'validation.log').stat().st_size == 0, 'native or UR validation failure')
        log = err.read_text()
        observed.update(tap.inspect(log, entry['pid'], dict(plan, cases={'smoke': plan['cases'][name]}),
                                    'smoke', observed['requests']))
        widths = [tuple(map(int, m)) for m in re.findall(r'^strata trace: window (\d+) (\d+)$', log, re.M)]
        t4 = [(mode, int(pos), int(layers), int(ready), int(early))
              for mode, pos, layers, ready, early in T4.findall(log)]
        positions = [pos for pos, width in widths if width == 4]
        require(positions and (not t4 if name == 'stock' else
                [p for _, p, _, _, _ in t4] == positions
                and all(mode == name and layers == 48 and 0 <= early <= ready <= layers
                        for mode, _, layers, ready, early in t4)
                and sum(ready for _, _, _, ready, _ in t4) > 0
                and (sum(early for _, _, _, _, early in t4) > 0) == (name == 'early')),
                'the actual width-four schedule was not exercised')
        require(all(bank.sha(p) == h for p, h in plan['dflash2_assets'].items()), 'draft assets changed')
        tap.basis.comparison.shared.models.model_pins(model_base['configuration'], model_base['model_assets'],
                                                     full=not Q6_CONTROL or name == ORDER[-1])
        observed.update(host_probe_accepted=True, widths=[list(item) for item in widths],
                        publications=[list(item) for item in t4])
    except BaseException as error:
        observed['error'] = repr(error)
    finally:
        validation.finish(observed, owner, cursor, plan['boot'])
        observed['passed'] &= observed.get('driver_after') == plan['driver']
        row['outcomes'][name] = observed
        row.update(active=bool(owner.active), ownership_closed=observed['ownership_closed'],
                   stage='partial' if observed['passed'] else 'failed')
        if name == ORDER[-1] and observed['passed']:
            checks = ('requests', 'states', 'tap_payloads', 'widths')
            pairs = zip(ORDER, ORDER[1:])
            row['comparison'] = {f'{a}/{b}': {k: row['outcomes'][a][k] == row['outcomes'][b][k] for k in checks}
                                 for a, b in pairs}
            row['passed'] = all(all(checks.values()) for checks in row['comparison'].values())
            if Q6_CONTROL:
                on = bank.previous.historic(V2_OBSERVATION, 'dflash2_decode_t4_model_v1')['outcomes']
                row['changed_state_keys'] = {
                    'stock/split with DPAS off': state_differences(row['outcomes']['stock'], observed),
                    'stock on/off': state_differences(on['stock'], row['outcomes']['stock']),
                    'split on/off': state_differences(on['after-cpu'], observed),
                }
            row.update(stage='closed', complete=True)
        save(row)
    print(name, observed['passed'], observed.get('error'), row.get('comparison', {}), flush=True)
    require(observed['passed'], observed.get('error', 'model owner or device gate failed'))


def state_differences(a, b):
    left, = a['states']
    right, = b['states']
    require(left.keys() == right.keys(), 'model state schemas differ')
    return [key for key in left if left[key] != right[key]]


if __name__ == '__main__':
    if sys.argv[1:2] == ['--q6-control']:
        sys.argv.pop(1)
        Q6_CONTROL = True
        KEY = 'dflash2_decode_t4_q6_control_v1'
        ROOT = bank.B / KEY.replace('_', '-')
        ORDER = ('stock', 'after-cpu')
    action, = sys.argv[1:]
    require(sys.executable == tap.basis.reference.model.PYTHON and action in ('prepare', *ORDER),
            'one pinned T4 model check stage')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        build.builder.guards.no_model_owners([])
        if action == 'prepare':
            require(not ROOT.exists() and KEY not in validation.committed(bank.RESULT), 'one fresh model plan')
            plan, *_ = inputs(validation.git('rev-parse', 'HEAD').decode().strip())
            for name in ORDER:
                for part in ('tmp', 'cache'):
                    (ROOT / name / part).mkdir(mode=0o700, parents=True)
            save(dict(plan=plan, outcomes={}, stage='prepared', active=False, ownership_closed=True,
                      complete=False, passed=False))
        else:
            row = bank.previous.read(ROOT / 'record.json')
            require(row == validation.committed(bank.RESULT, (KEY,)) and not row['active']
                    and row['ownership_closed'] and tuple(row['outcomes']) == ORDER[:ORDER.index(action)]
                    and row['stage'] in ('prepared', 'partial'), 'commit the preceding model result')
            plan, base, env, request = inputs(row['plan']['source_commit'])
            require(plan == row['plan'], 'planned model inputs changed')
            run(row, base, action, env[action], request)
