"""Check DFlash2 on the retained B570 build's all-GPU split-prefill path."""

import fcntl
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
import dflash2_v2_generation as generation
import prefill_workspace_admission as workspace

model, bank, basis, require = generation.model, generation.bank, generation.basis, generation.require
KEY = 'dflash2_v2_split_admission_v1'
ROOT = bank.B / KEY.replace('_', '-')
THIS = str(Path(__file__).resolve().relative_to(bank.P))
CASES = {'boundary': 90114, 'long': 117489}
OUTPUT = 32


def inputs(revision, root=ROOT):
    base = generation.sync_baseline()
    pins = generation.experiment(synchronous_commit=True).pins(revision)
    for name in (THIS, str(Path(workspace.__file__).relative_to(bank.P))):
        require((bank.P / name).read_bytes() == basis.validation.git('show', revision + ':' + name),
                'changed split admission input: ' + name)
        pins[name] = bank.sha(bank.P / name)
    plans, launches = {}, {}
    for index, (case, count) in enumerate(CASES.items()):
        plan = dict(root=str(root / case), index=index, arm='candidate')
        config = generation.configuration(base, plan)
        args = config['args']
        # Keep the prior session's live state for fingerprinting. The native
        # wire reader requires a fresh prompt on all three requests.
        args[args.index('--prompt-cache') + 1] = '1'
        env = generation.v2.launch_environment(base, plan, split=True)
        env.update(STRATA_SYCL_DECODE_Q6_DPAS_HEAD='1', STRATA_PREFILL_RELEASE_MTP_EXPERTS='0',
                   STRATA_COMMIT_SYNC='1', STRATA_TRACE='1', STRATA_SYCL_CACHE_LOAN_AUDIT='1',
                   STRATA_STATE_HASH='1', STRATA_WATCHDOG_S='240')
        env[generation.v2.state.KV] = '0'
        env = model.native_environment(base, env)
        require(env.get('STRATA_SYCL_PREFILL_MOE_SPLIT_ACCUMULATOR') == '1'
                and env.get('STRATA_SYCL_PREFILL_EXPERT_BANK_SLOTS') == '128'
                and env.get(generation.v2.UPLOAD) == 'batched',
                'require the selected V2 split and upload settings')
        short = model.request_ids(Path(config['tokenizer']))['ids']
        require(len(short) == 82, 'the verified short control changed')
        long = [i % 4096 for i in range(count - len(short))] + short
        requests = [(ids, OUTPUT) for ids in (short, long, short)]
        wire = [dict(read=len(ids), output=budget, sha256=model.restore.signature_bytes(model.wire(ids, budget)))
                for ids, budget in requests]
        plans[case] = dict(command=[base['binary']['path'], '--serve', *args],
                           environment_sha256=bank.signature(env), wire=wire)
        launches[case] = env, requests
    require(plans['boundary']['command'] == plans['long']['command']
            and {k: v.replace(str(root / 'boundary'), '<case>') for k, v in launches['boundary'][0].items()}
            == {k: v.replace(str(root / 'long'), '<case>') for k, v in launches['long'][0].items()},
            'keep the launch settings fixed across input extents')
    plan = dict(source_commit=revision, inputs=dict(files=len(pins), sha256=bank.signature(pins)),
                native_source_commit=base['native_source_commit'], native_build_sha256=base['native_build_sha256'],
                binary=base['binary'], driver=base['driver'], boot=base['boot'],
                runtime_manifest=base['runtime_manifest'], validation=base['native_validation_probe'],
                drafter_assets=base['dflash2_assets'], speed_reference=base['speed_reference'],
                commit_order_gate=base['commit_order_gate'], cases=plans, order=list(CASES), limits=model.LIMITS,
                scope='Trace the first split row and the original 117489-input geometry in fresh processes. '
                      'Keep 8192-token chunks, the canonical head, DFlash2 spec4, synchronized commit, '
                      'batched expert uploads and copied attention. Check actual GPU rows, tail guards, '
                      'expert-cache refill, the DFlash2 tap and the short reset. Stop at the first failure. '
                      'The direct wire inputs provide route and handoff evidence.')
    return plan, launches


def split_route(frame, count):
    rows = count - 1
    store = model.restore.require_store(frame, dict(direct=False), dict(read=count))
    require(store['gpu_rows'] == store['maximum_rows'] == rows and store['host_rows'] == 0
            and store['window_mode'] == 10, 'split prefill must hold every BF16 residual row in VRAM')
    admission = workspace.one(frame, 'strata MoE split admission: ')
    completion = workspace.one(frame, 'strata MoE split complete: ')
    bank_record = workspace.one(frame, 'strata MoE expert bank: ')
    upload = workspace.one(frame, 'strata MoE split expert upload: ')
    window = workspace.one(frame, 'strata MoE accumulator window: ')
    expected = workspace.workspace(rows, split=True, relocated=False)
    require(admission['active'] == '1' and int(admission['rows']) == int(completion['rows']) == rows
            and admission['primary_rows'] == completion['primary_rows'] == '65536'
            and int(admission['tail_payload_bytes']) == int(completion['tail_bytes']) == expected['tail_payload']
            and int(admission['extra_allocation_bytes']) == expected['extra']
            and int(admission['max_single_allocation_bytes']) >= rows * workspace.ROW
            and admission['prefill_slots'] == bank_record['effective_slots'] == '128'
            and bank_record['staged'] == '1' and int(bank_record['residual_gpu_rows']) == rows
            and completion['guard'] == completion['final_drain'] == upload['final_drain'] == 'checked'
            and completion['decode_changed'] == '0' and upload['windows'] == window['windows'] == '48'
            and int(upload['copy_commands']) == int(upload['experts']) == int(window['dequants']) > 0
            and int(window['maximum_rows']) == rows, 'actual split scratch, upload or retirement differs')
    return dict(store=store, tail_bytes=expected['tail_payload'], extra_bytes=expected['extra'],
                prefill_slots=128, split_guard_checked=True, uploads=int(upload['experts']))


def committed_state(frame, request):
    state = workspace.one(frame, 'strata serve: STATE_HASH ')
    windows = [tuple(map(int, x)) for x in re.findall(r'^strata trace: window (\d+) (\d+)$', frame, re.M)]
    first = request['prompt'] - 1
    require(windows and windows[0][0] == first and all(1 <= width <= 4 for _, width in windows)
            and all(1 <= next_pos - pos <= width for (pos, width), (next_pos, _) in zip(windows, windows[1:]))
            and sum(width - 1 for _, width in windows) == request['offered'], 'incomplete verifier trace')
    extent = first + len(windows) + request['accepted']
    pos, width = windows[-1]
    require(pos < extent <= pos + width and extent == int(state['L'])
            and first + OUTPUT <= extent <= first + OUTPUT + 3, 'unexpected committed state after decode')
    return {k: state[k] for k in model.restore.ACTIVE_STATE}


def inspect(log, plan, pid, count, requests):
    proof = basis.validation.preload_evidence(log, pid)
    require(proof['loader'] == plan['validation']['loader']['path']
            and log.count('strata native head DPAS: ready=1 columns=4 layout=canonical') == 1
            and not re.search(r'(?m)^strata mtp: .*loaded|UR_RESULT_ERROR_|ZE_RESULT_ERROR_|teardown uncertain', log),
            'unexpected target head, MTP load or runtime validation')
    markers = list(re.finditer(r'^strata trace: request (\d+) 0$', log, re.M))
    require([int(mark[1]) for mark in markers] == [82, count, 82], 'unexpected native request extents')
    starts = [mark.start() for mark in markers] + [len(log)]
    frames = [log[start:end] for start, end in zip(starts, starts[1:])]
    middle = frames[1]
    route = split_route(middle, count)
    taps = re.findall(r'^strata DFlash2: eager, prompt_tail_rows=(\d+) cold_prefix=0 vram_bytes=([1-9]\d*)$', log, re.M)
    require([int(rows) for rows, _ in taps] == [81, 2046, 81]
            and requests[0] == requests[2] and all(r['offered'] > 0 for r in requests)
            and log.count('strata prefill policy: requested=throughput route=cache-window') == 2,
            'DFlash2 tap coverage, draft work or short reset differs')
    states = [committed_state(frame, request) for frame, request in zip(frames, requests, strict=True)]
    require(states[0] == states[2], 'short reset changed active state')
    masked = re.findall(r'^strata cache loan: phase=masked generation=(\d+) chunk=8192 first=(\d+) lent=(\d+) decode_ready=0$', middle, re.M)
    ready = re.findall(r'^strata cache loan: phase=ready generation=(\d+) chunk=8192 first=(\d+) lent=(\d+) decode_ready=1$', middle, re.M)
    require(len(ready) == 1 and masked == ready and int(ready[0][2]) > 0
            and sum(map(int, ready[0][1:])) == 1256 and 'payload_match=1' in middle, 'expert-cache refill differs')
    return dict(validation=proof, split=route, states=states, loan=list(map(int, ready[0])),
                tap_rows=[int(rows) for rows, _ in taps])


def save(row):
    bank.stage.write(ROOT / 'record.json', row)
    result = bank.previous.read(bank.RESULT)
    result[KEY] = row
    bank.stage.write(bank.RESULT, result)


def run(row, case, env, requests):
    plan, path = row['plan'], ROOT / case
    cursor, = (s.removeprefix('-- cursor: ') for s in subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15).splitlines()
        if s.startswith('-- cursor: '))
    owner = model.owner_module.Owner(path / 'owned', lambda: bank.stage.write(
        path / 'execution.json', dict(active=bool(owner.active), commands=owner.commands)))
    outcome = dict(complete=False, host_probe_accepted=False, requests=[])

    def interact(io, _entry):
        info = {}
        while (line := io.line(600)).startswith('INFO '):
            info.update(x.split('=', 1) for x in line.split()[1:] if '=' in x)
        require(line == 'READY 262144 stop' and all(info.get(k) == v for k, v in model.INFO.items()),
                'unexpected context, drafter or decode placement')
        for ids, budget in requests:
            outcome['requests'].append(model.request(io, ids, budget))
            print(case, 'completed input', len(ids), flush=True)
        io.send(b'QUIT\n')
        io.exit(90)

    try:
        entry, _, stderr = owner.run('model', plan['cases'][case]['command'], env, bank.P,
                                     interaction=interact, **plan['limits'])
        outcome['native_exit_code'] = entry.get('exit_code')
        outcome['native_identity'] = {k: entry['direct_identity'][k] for k in ('pid', 'start_ticks')}
        require(model.owner_module.closed(entry) and entry['exit_code'] == 0,
                entry.get('error') or 'native owner or exit differs')
        outcome.update(inspect(stderr.read_text(), plan, entry['pid'], CASES[case], outcome['requests']))
        require(all(bank.sha(p) == h for p, h in plan['drafter_assets'].items()), 'draft assets changed')
        outcome.update(complete=True, host_probe_accepted=True)
    except Exception as error:
        outcome['error'] = repr(error)
    finally:
        basis.validation.finish(outcome, owner, cursor, plan['boot'])
        outcome['passed'] = outcome['passed'] and outcome.get('driver_after') == plan['driver']
        row['outcomes'][case] = outcome
        complete = len(row['outcomes']) == len(CASES) and outcome['passed']
        row.update(stage=('closed' if complete else 'partial') if outcome['passed'] else 'failed',
                   complete=complete, passed=complete, active=False, ownership_closed=outcome['ownership_closed'])
        save(row)
    print({k: outcome.get(k) for k in ('passed', 'native_exit_code', 'fresh_faults', 'error')}, flush=True)
    require(outcome['passed'], outcome.get('error', 'DFlash2 split admission failed'))


def main(action, case=None):
    require(sys.executable == basis.reference.model.PYTHON
            and ((action == 'prepare' and case is None) or (action == 'run' and case in CASES)),
            'use the pinned Python with prepare or run boundary/long')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        if action == 'prepare':
            require(not ROOT.exists() and KEY not in basis.validation.committed(bank.RESULT), 'new split plan required')
            plan, _ = inputs(basis.validation.git('rev-parse', 'HEAD').decode().strip())
            for name in CASES:
                for folder in ('cache', 'tmp'):
                    (ROOT / name / folder).mkdir(parents=True, mode=0o700)
            save(dict(plan=plan, stage='prepared', complete=False, passed=False, active=False,
                      ownership_closed=True, outcomes={}))
            return
        row = bank.previous.read(ROOT / 'record.json')
        plan, launches = inputs(row['plan']['source_commit'])
        earlier = list(CASES)[:list(CASES).index(case)]
        require(row == basis.validation.committed(bank.RESULT, (KEY,)) and row['plan'] == plan
                and row['stage'] == ('partial' if earlier else 'prepared') and row['ownership_closed']
                and list(row['outcomes']) == earlier and all(row['outcomes'][i]['passed'] for i in earlier)
                and not (ROOT / case / 'owned').exists() and not (ROOT / case / 'execution.json').exists(),
                'require the committed plan and closed prior cases without retries')
        run(row, case, *launches[case])


if __name__ == '__main__':
    main(*sys.argv[1:])
