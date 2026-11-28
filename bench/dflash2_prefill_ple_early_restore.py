"""Compare PLE scheduling after restoring a live prefix on the same engine."""

import fcntl
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
import dflash2_prefill_ple_early_capacity as capacity
import dflash2_primary_restore as original
from engine_once import native_environment

bank, basis, require = capacity.bank, capacity.basis, capacity.require
timing, build = capacity.timing, capacity.build
THIS = str(Path(__file__).resolve().relative_to(bank.P))
KEY = 'dflash2_prefill_ple_early_restore_v2'
ROOT = bank.B / KEY.replace('_', '-')
ORDER = ('host', 'stock', 'early')
PREFIX = 16385
SUFFIX = timing.wait.READ
PLAN = dict(
    order=list(ORDER), prefix=PREFIX, fresh_suffix=SUFFIX,
    scope='Park the first conversation, switch to a 33-token prompt, and restore the original prefix before '
          'reading the long fresh suffix. Use the same native executable, prompts and direct writer in both arms.',
    gates='Require live target-only restoration from RAM, a mode-16 suffix, actual early-read consumption '
          'only for the enabled suffix, and identical outputs and active states. Check every KV chunk at its '
          'absolute suffix position, including unaligned heads, page tails and the published byte count.',
    limits='Traced live RESTORE at a nonzero position. The standalone 262144-input result and the '
           'first-prefill performance comparison are separate receipts.',
)


def workloads(_near):
    first = [i % 4096 for i in range(PREFIX)]
    other = list(range(128, 161))
    resumed = first + [256 + i % 16 for i in range(SUFFIX)]
    return [(first, 0), (other, 0), (resumed, PREFIX)]


def environment(row, case):
    base = timing.environment(row, dict(root=case['root'], index=0, arm=case['name']))
    require(base['STRATA_SYCL_PREFILL_KEEP_KV_STAGE'] == '1', 'restore KV stage policy changed')
    base.update(STRATA_TRACE='1', STRATA_STATE_HASH='1', STRATA_SNAPSHOT_VERIFY='1')
    return native_environment(row, base)


def inspect(peer, log, pid, name, requests, row):
    proof = peer.validation.preload_evidence(log, pid)
    require(proof['loader'] == row['native_validation_probe']['loader']['path']
            and not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|teardown uncertain|'
                              r'conversation cache: (skip parking|discard invalid)|^strata mtp: .*loaded', log, re.M),
            'unexpected runtime, snapshot or drafter failure')
    restored = peer.capacity.one(log, r'^strata serve: conversation cache: restored (\d+) tokens \(live\)',
                                 'live restore')
    require(int(restored) == PREFIX and requests[2]['reused'] == PREFIX and requests[2]['read'] == SUFFIX,
            'native restore or fresh suffix extent changed')
    snapshot = dict(x.split('=', 1) for x in peer.capacity.one(log,
        r'^strata serve: SNAPSHOT_VERIFY (.*)$', 'target snapshot readback').split())
    require(snapshot['result'] == 'PASS' and snapshot['source'] == 'ram' and int(snapshot['main_layers']) == 12
            and int(snapshot['upto']) == PREFIX
            and int(snapshot['main_bytes']) >= PREFIX * 12 * original.physical.kv.CELL_BYTES
            and int(snapshot['running_bytes']) > 0 and int(snapshot['draft'], 16) == 0
            and all(int(snapshot[k]) == 0 for k in ('cells', 'mode', 'resident', 'draft_bytes')),
            'incomplete target-only restore payload')

    states = list(re.finditer(r'^strata serve: STATE_HASH L=.*$', log, re.M))
    require(len(states) == len(requests) == 3, 'expected three request state digests')
    start = 0
    for index, (seen, request) in enumerate(zip(states, requests, strict=True)):
        frame = log[start:seen.start()]
        start = seen.end()
        state = dict(re.findall(r'(\w+)=([0-9a-f,-]+)', seen[0]))
        require(set(state) == set(peer.ACTIVE_STATE + peer.STALE_STATE) and int(state['L']) == request['prompt'],
                'committed state digest domains or position differ')
        request['active_state'] = {k: state[k] for k in peer.ACTIVE_STATE}
        taps, cold, vram = map(int, peer.capacity.one(frame,
            r'^strata DFlash2: eager, prompt_tail_rows=(\d+) cold_prefix=([01]) vram_bytes=(\d+)$', 'DFlash2 tail'))
        require((taps, cold) == (min(request['read'] - 1, 2046), 0) and vram > 0,
                'DFlash2 tail placement differs')
        consumed = [line for line in frame.splitlines() if line.startswith(timing.MARKER)]
        expected = [timing.check.CONSUMED] if index == 2 and name == 'early' else []
        require(consumed == expected, 'unexpected early-read execution')
        if index == 2:
            rows = request['read'] - 1
            require(f'strata MoE window configuration: requested_mode=16 effective_mode=16 maximum_rows={rows} '
                    in frame and f'residual_gpu_rows={rows} ' in frame, 'the restored suffix missed mode-16 admission')
            line = peer.capacity.one(frame, r'^strata prefill attention bank: requested=1 pieces=(\d+) direct=(\d+) '
                                      r'requested_D2D_bytes=0$', 'direct writer')
            chunk = peer.capacity.CHUNK
            pieces = 48 * ((rows + chunk - 1) // chunk)
            require(tuple(map(int, line)) == (pieces, pieces), 'the restored suffix did not use direct stores')
            suffix_kv = original.physical.kv.check_request(frame, request['reused'], rows, chunk, reuse=True)
    require(log.count(timing.check.ISSUED) == int(name == 'early')
            and log.count(timing.check.CONSUMED) == int(name == 'early'), 'unexpected early reads outside the suffix')
    return dict(validation=proof, restored_tokens=PREFIX, snapshot=snapshot, requests=requests,
                suffix_kv=suffix_kv, early_reads=int(name == 'early'))


def experiment(*, key=KEY, root=ROOT, order=ORDER, plan=PLAN, physical_key=capacity.KEY,
               selector=capacity.SELECTOR, prerequisites=None, launch_environment=environment,
               inspect_result=inspect, extra_sources=()):
    require(len(order) == 3 and order[0] == 'host', 'require one host check and two restore arms')
    peer = basis.core.source.device.load_module('early_ple_live_peer', original.restore.__file__)
    peer.workloads = workloads
    timing_pins = timing.experiment().pins
    if prerequisites is None:
        native_context = build.context()
        prerequisites = lambda: capacity.prerequisites(native_context)

    def inputs(revision):
        native, baseline, measured_sha = prerequisites()
        physical = basis.validation.committed(bank.RESULT, (physical_key,))
        require(timing.wait.retained.closed(physical, 'closed')
                and physical['outcomes']['full-copy']['final_slot_consumed']
                and physical['native_build_sha256'] == baseline['native_build_sha256']
                and physical['binary'] == native['binaries']['strata_full_context_diagnostic'],
                'require the final-slot proof from the same build')
        pins = peer.sources(revision) | timing_pins(revision)
        kv_reader = Path(original.physical.kv.__file__).resolve().relative_to(bank.P).as_posix()
        for name in (THIS, capacity.THIS, original.THIS, original.physical.THIS, kv_reader, *extra_sources):
            require((bank.P / name).read_bytes() == basis.validation.git('show', revision + ':' + name),
                    'commit the exact restore input: ' + name)
            pins[name] = bank.sha(bank.P / name)
        fields = ('boot', 'driver', 'runtime_manifest', 'native_validation_probe', 'configuration', 'model_assets',
                  'environment_template', 'binary', 'native_source_commit', 'native_build_sha256',
                  'selected_arm', 'ordinary_control', 'python_selected', 'dflash2_assets')
        row = {k: baseline[k] for k in fields}
        command = peer.argv(dict(row, configuration=dict(row['configuration'], args=baseline['validated_dflash2_args'])))
        request_plan = peer.input_plan(False)
        require(all(p['prompt'] + p['max_new'] + 8 <= 262144 for p in request_plan), 'restore input exceeds context')
        cases = [dict(name=name, root=str(root / name), near=False, requests=request_plan, command=command)
                 for name in order[1:]]
        launch = [launch_environment(row, case) for case in cases]
        normalized = [{k: v.replace(case['root'], '<case>') for k, v in env.items() if k != selector}
                      for case, env in zip(cases, launch, strict=True)]
        require(normalized[0] == normalized[1], 'another runtime setting changed between restore arms')
        for case, env in zip(cases, launch, strict=True):
            case['environment_sha256'] = bank.signature(env)
        return dict(row, source_commit=revision, sources=pins, plans=cases, plan=plan,
                    vocab_size=physical['vocab_size'], physical_sha256=bank.signature(physical), timing_sha256=measured_sha)

    def save(row):
        bank.stage.write(root / 'record.json', row)
        output = bank.previous.read(bank.RESULT)
        output[key] = basis.compact(row)
        bank.stage.write(bank.RESULT, output)

    def prepare():
        require(not root.exists() and key not in basis.validation.committed(bank.RESULT), 'fresh restore attempt required')
        row = inputs(basis.validation.git('rev-parse', 'HEAD').decode().strip())
        root.mkdir(mode=0o700)
        for name in order:
            parent = root / name
            parent.mkdir(mode=0o700)
            for folder in ('tmp', 'cache'):
                (parent / folder).mkdir(mode=0o700)
        save(dict(row, stage='prepared', outcomes={}, complete=False, passed=False, active=False, ownership_closed=True))

    def run(name):
        row = bank.previous.read(root / 'record.json')
        require(basis.compact(row) == basis.validation.committed(bank.RESULT, (key,))
                and row['ownership_closed'] and not row['active']
                and tuple(row['outcomes']) == order[:order.index(name)]
                and all(v['passed'] for v in row['outcomes'].values()), 'commit the preceding restore outcome')
        require(all(row[k] == v for k, v in inputs(row['source_commit']).items()), 'restore inputs changed')
        case_root = root / name
        receipt = case_root / 'execution.json'
        require(not receipt.exists(), 'inspect the existing owner receipt')
        cursor = next(s.removeprefix('-- cursor: ') for s in subprocess.check_output(
            ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15).splitlines()
             if s.startswith('-- cursor: '))
        owner = peer.owner_module.Owner(case_root / 'owned', lambda: bank.stage.write(
            receipt, dict(active=bool(owner.active), commands=owner.commands)))
        outcome = dict(complete=False, host_probe_accepted=False)
        try:
            requests = []
            if name == 'host':
                case = dict(near=False, requests=row['plans'][0]['requests'])
                cmd = [sys.executable, '-B', '-c', peer.host_peer(), json.dumps(case['requests'])]
                env = dict(PATH='/usr/bin:/bin', LANG='C.UTF-8')
                limits = dict(wall=50, cpu=40, rss_cap=512 << 20, text_cap=128 << 10, file_cap=1 << 20, total_cap=4 << 20)
            else:
                case = next(c for c in row['plans'] if c['name'] == name)
                cmd, env = case['command'], launch_environment(row, case)
                require(bank.signature(env) == case['environment_sha256'], 'the pre-exec runtime changed')
                limits = dict(wall=1800, cpu=18000, rss_cap=104 << 30, text_cap=16 << 20,
                              file_cap=32 << 20, total_cap=256 << 20)

            def exchange(io, _entry):
                requests.extend(peer.interact(io, case, row['vocab_size'], native=name != 'host'))

            entry, _, err = owner.run(name, cmd, env, bank.P, interaction=exchange, **limits)
            require(peer.owner_module.closed(entry) and entry['exit_code'] == 0, 'restore process failed or did not close')
            outcome['native_exit_code'] = entry['exit_code']
            if name != 'host':
                require((case_root / 'validation.log').stat().st_size == 0, 'runtime parameter validation output is not quiet')
                outcome.update(inspect_result(peer, err.read_text(), entry['pid'], name, requests, row))
            if name == order[-1]:
                reference = row['outcomes'][order[1]]
                fields = ('restored_tokens', 'snapshot', 'requests', 'suffix_kv')
                require(all(outcome[k] == reference[k] for k in fields), 'live restore or continuation differs between arms')
                basis.comparison.shared.models.model_pins(row['configuration'], row['model_assets'], full=True)
            outcome.update(complete=True, host_probe_accepted=True)
        except Exception as error:
            outcome['error'] = repr(error)
        finally:
            peer.validation.finish(outcome, owner, cursor, row['boot'])
            passed = bool(outcome['passed'] and outcome['complete'] and outcome['driver_after'] == row['driver'])
            outcome['passed'] = passed
            row['outcomes'][name] = outcome
            done = passed and name == order[-1]
            row.update(stage='closed' if done else 'running' if passed else 'failed', complete=done, passed=done,
                       active=bool(owner.active), ownership_closed=outcome['ownership_closed'])
            save(row)
        print({k: outcome.get(k) for k in ('passed', 'native_exit_code', 'restored_tokens', 'fresh_faults', 'error')}, flush=True)
        require(passed, 'restore gate stopped; preserve the outcome')

    return prepare, run


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(sys.executable == basis.reference.model.PYTHON and action in ('prepare', *ORDER),
            'select one planned stage with the pinned interpreter')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        prepare, run = experiment()
        prepare() if action == 'prepare' else run(action)
