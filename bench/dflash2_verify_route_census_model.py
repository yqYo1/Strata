"""Check the bounded DFlash2 route census in the production native server."""

import fcntl
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
import dflash2_primary_prefill_speed as speed
import dflash2_verify_route_census as census
import dflash2_decode_route_census_build as build
import dflash2_v2_model as model
from engine_once import native_environment

bank, basis, require = speed.bank, speed.basis, speed.require
validation, owners = basis.validation, model.owner_module
KEY = 'dflash2_verify_route_census_model_v1'
ROOT = bank.B / KEY.replace('_', '-')
THIS = str(Path(__file__).resolve().relative_to(bank.P))
ORDER = ('disabled', 'enabled')
DUMP = 'STRATA_SYCL_DFLASH2_DECODE_ROUTE_DUMP'
LIMITS = dict(model.LIMITS, wall=1200, cpu=4800, total_cap=96 << 20)
LONG_QUESTION = (
    'Design a crash-consistent persistent B+ tree in Python. Reason through page '
    'allocation, split and merge, copy-on-write updates, checksums, atomic root '
    'publication, free-list recovery and concurrent readers. Then give a complete '
    'implementation with a deterministic simulated power-loss test for each step '
    'of a write. Explain the invariants and the asymptotic costs.'
)


def inputs(revision):
    native = bank.previous.read(build.ROOT / 'record.json')
    measured = bank.previous.read(speed.ROOT / 'record.json')
    reference = bank.previous.historic(build.PREFILL_EVIDENCE, build.PREFILL_KEY)
    require(build.primary.builder.compact(native) == validation.committed(bank.RESULT, (build.KEY,))
            and native['stage'] == 'built' and native['passed'] and native['complete']
            and native['ownership_closed'] and not native['active']
            and basis.compact(measured) == reference and reference['binary'] == native['retained_references']['prefill']
            and measured['boot'] == speed.checks.BOOT.read_text().strip()
            and measured['driver'] == native['driver']
            and native['foundation']['runtime_manifest']['sha256'] == measured['runtime_manifest']['sha256'],
            'require the built census executable and its exact measured primary runtime')
    build.configure_controller()
    build.primary.check(native)
    runtime = measured['runtime_manifest']
    require(bank.sha(runtime['path']) == runtime['sha256']
            and all(bank.sha(p) == h for p, h in bank.previous.read(Path(runtime['path'])).items())
            and bank.sha(measured['template']['path']) == measured['template']['sha256']
            and all(bank.sha(p) == h for p, h in measured['dflash2_assets'].items()),
            'changed runtime, template or draft input')
    basis.comparison.shared.models.model_pins(measured['configuration'], measured['model_assets'], full=False)
    contract, manifest = basis.core.frontend.python_manifest()
    require(contract == measured['python_contract'] and manifest['selected'] == measured['python_selected'],
            'use the measured Python and tokenizer providers')

    args = list(measured['validated_dflash2_args'])
    rank = bank.P / 'data/expert-profile.bin'
    require(bank.sha(rank) == measured['sources']['data/expert-profile.bin']
            and '--greedy' in args and '--adapt-swaps' in args and args[args.index('--adapt-swaps') + 1] == '0'
            and args[args.index('--spec') + 1] == '4' and args[args.index('--draft') + 1] == 'dflash2'
            and args[args.index('--prompt-cache-every') + 1] == '0' and '--dump-routing' not in args,
            'expected exact greedy, fixed-placement DFlash2 spec4 settings')
    args[args.index('--expert-profile') + 1] = str(rank)
    args[args.index('--prompt-cache') + 1] = '0'
    tokenizer = Path(measured['configuration']['tokenizer'])
    short = model.request_ids(tokenizer)['ids']
    longer = model.request_ids(tokenizer, content=LONG_QUESTION)['ids']
    require(short != longer, 'the sustained request must differ from the short controls')
    work = [(short, 32), (longer, 1025), (short, 32)]

    names = (THIS, 'bench/dflash2_verify_route_census.py', 'bench/dflash2_decode_residency_replay.py',
             'bench/sycl/dflash2_v2_model.py', 'bench/dflash2_primary_prefill_speed.py')
    sources = speed.experiment().pins(revision)
    for name in names:
        path = bank.P / name
        require(path.read_bytes() == validation.git('show', revision + ':' + name),
                'commit exact model check input: ' + name)
        sources[name] = bank.sha(path)
    engine = native['binaries']['strata']
    cases, environments = {}, {}
    for name in ORDER:
        root = ROOT / name
        env = speed.environment(measured, dict(root=str(root), index=0, arm='retain'))
        require(DUMP not in env, 'census must be selected only by this plan')
        env.update(STRATA_TRACE='1', STRATA_STATE_HASH='1')
        if name == 'enabled':
            env[DUMP] = str(root / 'census.dat')
        environments[name] = native_environment(measured, env)
        cases[name] = dict(command=[engine['path'], '--serve', *args, '--dump-routing', str(root / 'routing.dat')],
                           environment_sha256=bank.signature(environments[name]))
    normalized = [{k: v.replace(str(ROOT / name), '<case>') for k, v in env.items() if k != DUMP}
                  for name, env in environments.items()]
    require(normalized[0] == normalized[1], 'change only the route census observer')
    plan = dict(source_commit=revision, sources_sha256=bank.signature(sources), source_count=len(sources),
                build_commit=native['native_source_commit'], binary=engine,
                build_sha256=bank.signature(build.primary.builder.compact(native)), order=list(ORDER),
                cases=cases, limits=LIMITS, boot=measured['boot'], driver=measured['driver'],
                requests=[dict(read=len(ids), generated=count, wire_sha256=census.replay.sha(model.wire(ids, count)))
                          for ids, count in work],
                scope='Two fresh trace-enabled diagnostic processes. Count the first 256 decode windows, '
                'compare full raw routes and work, and repeat the short request after the cap. '
                'Cancellation, RESTORE and observer overhead use separate comparisons.')
    return plan, measured, environments, work


def inspect(log_path, root, name, pid, base, requests):
    route_path = root / 'routing.dat'
    count_path = root / 'census.dat'
    capture = (log_path, route_path, count_path) if name == 'enabled' else (log_path, route_path)
    data = census.read_capture(capture)
    log, routes = data[0].decode(), data[1]
    proof = validation.preload_evidence(log, pid)
    require(proof['loader'] == base['native_validation_probe']['loader']['path']
            and log.count('strata native head DPAS: ready=1 columns=4 layout=canonical') == 1
            and len(re.findall(r'^strata DFlash2: eager, ', log, re.M)) == len(requests)
            and not re.search(r'(?m)^strata mtp: .*loaded|UR_RESULT_ERROR_|ZE_RESULT_ERROR_|teardown uncertain', log),
            'expected the validated DFlash2 route and canonical target head')
    footer = speed.clock.footer
    headers = [list(map(int, m.groups())) for m in re.finditer(footer.HEADER, log, re.M)]
    workers = [list(map(int, m.groups())) for m in re.finditer(footer.COPY, log, re.M)]
    require(headers == [[r['prompt'], 0, r['prompt'], r['generated'], r['accepted'], r['offered']] for r in requests]
            and len(workers) == len(requests) and all(r['offered'] > 0 for r in requests),
            'fresh native output, draft counts or expert work differs')
    states = []
    for line in re.findall(r'^strata serve: STATE_HASH L=.*$', log, re.M):
        state = dict(re.findall(r'(\w+)=([0-9a-f,-]+)', line))
        states.append({k: state[k] for k in model.restore.ACTIVE_STATE})
    require(len(states) == len(requests) and states[0] == states[-1] and requests[0] == requests[-1]
            and [int(s['L']) for s in states] == [r['prompt'] + r['generated'] for r in requests]
            and len(routes) > 0, 'repeat short control or active target state changed')
    result = dict(requests=requests, expert_work=workers, states=states, validation=proof,
                  raw_routing=dict(bytes=len(routes), sha256=census.replay.sha(routes)))
    if name == 'enabled':
        rows = census.verify(log, routes, data[2])
        require(len(rows) == len(requests) and rows[1]['windows'] == census.MAX_WINDOWS
                and rows[1]['request_windows'] > census.MAX_WINDOWS
                and 0 < rows[0]['windows'] < census.MAX_WINDOWS
                and all(rows[0][k] == rows[-1][k] for k in ('windows', 'offered_rows', 'counts_sha256')),
                'full cap and fresh repeat count checks are required')
        result['census'] = rows
    else:
        require(not count_path.exists() and census.MARKER not in log, 'observer ran in its disabled arm')
    return result


def save(row):
    bank.stage.write(ROOT / 'record.json', row)
    result = bank.previous.read(bank.RESULT)
    result[KEY] = row
    bank.stage.write(bank.RESULT, result)


def run(row, measured, name, env, work):
    root = ROOT / name
    require(not any((root / f).exists() for f in ('execution.json', 'routing.dat', 'census.dat')),
            'this arm already has an owner or a capture')
    cursor, = (s.removeprefix('-- cursor: ') for s in subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'],
        text=True, timeout=15).splitlines() if s.startswith('-- cursor: '))
    owner = owners.Owner(root / 'owned', lambda: bank.stage.write(
        root / 'execution.json', dict(active=bool(owner.active), commands=owner.commands)))
    plan = row['plan']
    requests = []
    outcome = dict(complete=False, host_probe_accepted=False, requests=requests)

    def interact(io, _entry):
        info = {}
        line = io.line(600)
        while line.startswith('INFO '):
            info.update(word.split('=', 1) for word in line.split()[1:])
            line = io.line(600)
        require(line == 'READY 262144 stop' and all(info.get(k) == v for k, v in model.INFO.items()),
                'native context or expert placement changed')
        for ids, count in work:
            requests.append(model.request(io, ids, count))
        io.send(b'QUIT\n')
        io.exit(90)

    try:
        entry, _, stderr = owner.run(name, plan['cases'][name]['command'], env, bank.P,
                                     interaction=interact, **plan['limits'])
        outcome['native_exit_code'] = entry['exit_code']
        require(owners.closed(entry) and entry['exit_code'] == 0
                and (root / 'validation.log').stat().st_size == 0, 'native exit or validation failed')
        outcome['native_identity'] = entry['direct_identity']
        outcome.update(inspect(stderr, root, name, entry['pid'], measured, requests))
        if name == 'enabled':
            control = row['outcomes']['disabled']
            require(control['native_identity'] != outcome['native_identity']
                    and all(control[k] == outcome[k] for k in ('requests', 'expert_work', 'states', 'raw_routing')),
                    'counting changed model work or results')
        require(all(bank.sha(p) == h for p, h in measured['dflash2_assets'].items()), 'drafter inputs changed')
        basis.comparison.shared.models.model_pins(measured['configuration'], measured['model_assets'], full=name == 'enabled')
        outcome.update(complete=True, host_probe_accepted=True)
    except BaseException as error:
        outcome['error'] = repr(error)
    finally:
        validation.finish(outcome, owner, cursor, plan['boot'])
        outcome['passed'] = outcome['passed'] and outcome['driver_after'] == plan['driver']
        row['outcomes'][name] = outcome
        complete = outcome['passed'] and name == 'enabled'
        row.update(stage='closed' if complete else 'partial' if outcome['passed'] else 'failed',
                   complete=complete, passed=complete, active=bool(owner.active),
                   ownership_closed=outcome['ownership_closed'])
        save(row)
    print({k: outcome.get(k) for k in ('passed', 'native_exit_code', 'fresh_faults', 'error')}, flush=True)
    require(outcome['passed'], outcome.get('error') or 'route census failed')


def main(action):
    require(sys.executable == basis.reference.model.PYTHON and action in ('prepare', *ORDER),
            'use the pinned interpreter and select prepare, disabled or enabled')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        if action == 'prepare':
            require(not ROOT.exists() and KEY not in validation.committed(bank.RESULT), 'fresh census plan required')
            revision = validation.git('rev-parse', 'HEAD').decode().strip()
            plan, *_ = inputs(revision)
            for name in ORDER:
                for folder in ('cache', 'tmp'):
                    (ROOT / name / folder).mkdir(parents=True, mode=0o700)
            save(dict(plan=plan, stage='prepared', outcomes={}, active=False, complete=False, passed=False,
                      ownership_closed=True, speed_eligible=False))
            return
        row = bank.previous.read(ROOT / 'record.json')
        require(row == validation.committed(bank.RESULT, (KEY,)) and not row['active']
                and row['ownership_closed'] and tuple(row['outcomes']) == ORDER[:ORDER.index(action)]
                and row['stage'] in ('prepared', 'partial'), 'commit the preceding successful stage')
        plan, measured, environments, work = inputs(row['plan']['source_commit'])
        require(plan == row['plan'], 'frozen route census inputs changed')
        run(row, measured, action, environments[action], work)


if __name__ == '__main__':
    main(*sys.argv[1:])
