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
KEY = 'dflash2_decode_t4_model_v2'
ROOT = bank.B / KEY.replace('_', '-')
THIS = str(Path(__file__).resolve().relative_to(bank.P))
ORDER = ('stock', 'after-cpu', 'early')
MATCHED_CONTROL, MATCHED_KEY = '4a36122ad', KEY
DPAS_STOCK_REFERENCE = 'e36f293d5'
PRODUCTION_DPAS = False
T4 = re.compile(r'^DECODE_T4 mode=(\S+) position=(\d+) layers=(\d+) split=(\d+) pool_ready=(\d+) early=(\d+)$',
                re.M)
DENSE_CAPTURE = re.compile(r'^strata Q6_DPAS_DENSE request=(\d+) device=0 phase=decode tokens=4 '
                           r'capture_id=(\d+) ni=(\d+) no=(\d+) nc=4 layout=canonical extra_weight_bytes=0 '
                           r'canonical_table_excludes_dpas=1 calls=(\d+) complete=(\d+)$', re.M)
STATE_FIELDS = ('requests', 'states', 'tap_payloads', 'widths')


def dense_capture(log):
    samples = [tuple(map(int, m)) for m in DENSE_CAPTURE.findall(log)]
    require(len(samples) == 4 and len({(r, c) for r, c, *_ in samples}) == 1
            and len({(ni, no) for _, _, ni, no, _, _ in samples}) == 4
            and all(c > 0 and complete == 1 for _, c, _, _, _, complete in samples)
            and any(calls > 0 for _, _, _, _, calls, _ in samples),
            'require a complete actual width-four dense DPAS capture')
    return sorted([ni, no, calls] for _, _, ni, no, calls, _ in samples)


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
    previous = bank.previous.historic(build.Q6_EVIDENCE, build.Q6_KEY)
    ready = {row[3] for row in previous['outcomes']['after-cpu']['publications']}
    require(build.retained.closed(previous, 'closed') and ready == {23}
            and old['cases']['smoke']['wire'] == previous['plan']['cases']['stock']['wire'],
            'require the original short request and observed publication scope')
    env, cases = {}, {}
    for name in ORDER:
        env[name] = {k: v.replace(str(tap.ROOT / 'smoke'), str(ROOT / name))
                     for k, v in envs['smoke'].items()}
        env[name].update(STRATA_SYCL_DECODE_T4_SCHEDULE=name, STRATA_SYCL_DECODE_T4_AUDIT='1')
        env[name]['STRATA_SYCL_DECODE_Q6_DPAS_DENSE'] = str(int(PRODUCTION_DPAS))
        cases[name] = dict(old['cases']['smoke'], environment_sha256=bank.signature(env[name]))
    plan = {k: old[k] for k in ('boot', 'driver', 'runtime_manifest', 'native_validation_probe',
                                'dflash2_assets', 'model_assets_sha256')}
    plan.update(source_commit=revision, inputs_sha256=bank.signature(pins), arc_evidence_sha256=bank.signature(device),
                native_source_commit=native['native_source_commit'], native_build_sha256=signature,
                binary=binary, command=[binary['path'], *args], cases=cases, order=list(ORDER),
                split_layers=next(iter(ready)), dense_dpas='0', numerical_reference=build.Q6_EVIDENCE,
                scope='Replay the original 82-input/32-output DFlash2 request. Use matched dense DPAS off '
                      'to check mixed stock and split dispatch, active state, prompt taps and output.')
    if PRODUCTION_DPAS:
        control = bank.previous.historic(MATCHED_CONTROL, MATCHED_KEY)
        require(build.retained.closed(control, 'closed') and control['plan']['native_build_sha256'] == signature
                and control['plan']['binary'] == binary, 'pass the same-build DPAS-off model control first')
        original_root = bank.B / MATCHED_KEY.replace('_', '-')
        for name in ORDER:
            normalized = {k: v.replace(str(ROOT / name), str(original_root / name)) for k, v in env[name].items()}
            normalized['STRATA_SYCL_DECODE_Q6_DPAS_DENSE'] = '0'
            original = control['plan']['cases'][name]
            require(bank.signature(normalized) == original['environment_sha256']
                    and cases[name]['wire'] == original['wire'] and cases[name]['limits'] == original['limits'],
                    'reproduce the exact diagnostic inputs before enabling production DPAS')
            env[name]['STRATA_SYCL_Q6_ROUTE_AUDIT'] = '1'
            cases[name]['environment_sha256'] = bank.signature(env[name])
        plan.update(dense_dpas='1', matched_control=MATCHED_CONTROL, stock_reference=DPAS_STOCK_REFERENCE,
                    scope='Same short request and executable with the production dense DPAS selector restored. '
                          'Count the actual width-four DPAS calls. Check exact early/deferred parity with shared '
                          'arithmetic and preserve stock prompt taps. Record stock/split differences.')
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
        t4 = [(mode, int(pos), int(layers), int(split), int(ready), int(early))
              for mode, pos, layers, split, ready, early in T4.findall(log)]
        positions = [pos for pos, width in widths if width == 4]
        expected = plan['split_layers']
        require(positions and (not t4 if name == 'stock' else
                [p for _, p, *_ in t4] == positions
                and all(mode == name and layers == 48 and split == ready == expected
                        and early == (expected if name == 'early' else 0)
                        for mode, _, layers, split, ready, early in t4)),
                'the actual width-four schedule was not exercised')
        if PRODUCTION_DPAS:
            observed['dense_capture'] = dense_capture(log)
            if name == 'stock':
                reference = bank.previous.historic(DPAS_STOCK_REFERENCE, 'dflash2_decode_t4_model_v1')
                current = dict(observed, widths=[list(item) for item in widths])
                require(all(reference['outcomes']['stock'][k] == current[k] for k in STATE_FIELDS),
                        'the production stock control did not reproduce the original short execution')
        require(all(bank.sha(p) == h for p, h in plan['dflash2_assets'].items()), 'draft assets changed')
        tap.basis.comparison.shared.models.model_pins(model_base['configuration'], model_base['model_assets'],
                                                     full=name == ORDER[-1])
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
            pairs = zip(ORDER, ORDER[1:])
            row['comparison'] = {f'{a}/{b}': {k: row['outcomes'][a][k] == row['outcomes'][b][k] for k in STATE_FIELDS}
                                 for a, b in pairs}
            if PRODUCTION_DPAS:
                stock = row['outcomes']['stock']['dense_capture']
                deferred = row['outcomes']['after-cpu']['dense_capture']
                early = row['outcomes']['early']['dense_capture']
                retained = all(a[:2] == b[:2] and b[2] <= a[2] for a, b in zip(stock, deferred, strict=True))
                stock_calls = sum(item[2] for item in stock)
                split_calls = sum(item[2] for item in early)
                row['production_dpas_preserved'] = (retained and deferred == early and 0 < split_calls < stock_calls)
                row['passed'] = (row['production_dpas_preserved']
                    and row['comparison']['stock/after-cpu']['tap_payloads']
                    and all(row['comparison']['after-cpu/early'].values()))
            else:
                row['passed'] = all(all(checks.values()) for checks in row['comparison'].values())
            row.update(stage='closed', complete=True)
        save(row)
    print(name, observed['passed'], observed.get('error'), row.get('comparison', {}), flush=True)
    require(observed['passed'], observed.get('error', 'model owner or device gate failed'))


if __name__ == '__main__':
    if sys.argv[1:2] == ['--production-dpas']:
        sys.argv.pop(1)
        PRODUCTION_DPAS = True
        KEY = 'dflash2_decode_t4_production_dpas_v1'
        ROOT = bank.B / KEY.replace('_', '-')
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
