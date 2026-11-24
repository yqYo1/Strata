"""Bounded DFlash2 commit-order check on the retained Arc B570 V2 engine."""

import fcntl
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
import dflash2_v2_generation as generation

model, bank, basis, require = generation.model, generation.bank, generation.basis, generation.require
KEY = 'dflash2_v2_commit_order_v1'
ROOT = bank.B / KEY.replace('_', '-')
THIS = str(Path(__file__).resolve().relative_to(bank.P))
FAILURE = '1a418e7bb'
RECOVERY = 'dflash2_v2_post_reset_v1'
CASES = {'near': 11793, 'long': 32769}
OUTPUT = 32
OBSERVER_ENV = dict(STRATA_COMMIT_SYNC='1', STRATA_TRACE='1',
                    STRATA_SYCL_CACHE_LOAN_AUDIT='1', STRATA_WATCHDOG_S='240')


def inputs(revision):
    base = generation.baseline()
    failed = bank.previous.historic(FAILURE, generation.KEY)
    wait = failed['first_process_stall']['wait']
    recovered = basis.validation.committed(bank.RESULT, (RECOVERY,))
    require(failed['stage'] == 'failed' and failed['ownership_closed'] and not failed['active']
            and failed['binary'] == base['binary'] and failed['native_build_sha256'] == base['native_build_sha256']
            and failed['runtime_manifest'] == base['runtime_manifest'] and failed['driver'] == base['driver']
            and failed['boot'] == base['boot'] and wait['width'] == 4 and wait['position'] == 11801
            and failed['first_process_stall']['request']['prompt_tokens'] == CASES['near'],
            'require the exact closed DFlash2 failure and its verifier geometry')
    result, recovery_plan = recovered['result'], recovered['plan']
    require(recovered['passed'] and recovered['complete'] and recovered['ownership_closed'] and not recovered['active']
            and recovery_plan['native_build_sha256'] == base['native_build_sha256']
            and recovery_plan['boot'] == base['boot'] and result['driver_after'] == base['driver']
            and result['fresh_faults'] == [] and result['native_exit_code'] == 0,
            'require the closed same-boot post-reset device proof')
    pins = generation.experiment().pins(revision)
    require((bank.P / THIS).read_bytes() == basis.validation.git('show', revision + ':' + THIS),
            'controller source changed')
    pins[THIS] = bank.sha(bank.P / THIS)

    first = failed['plans'][0]
    prior_config = Path(first['root']) / 'config.json'
    prior_env = generation.environment(base, first)
    require(bank.sha(prior_config) == first['generated']['config.json']
            and generation.suite.signature(prior_env) == first['environment_sha256'],
            'require the original failed process configuration and environment')
    prior_args = bank.previous.read(prior_config)['args']
    plans, launches = {}, {}
    for case, n in CASES.items():
        path = ROOT / case
        config = generation.configuration(base, dict(root=str(path)))
        command = [base['binary']['path'], '--serve', *config['args']]
        require(config['args'] == prior_args, 'native arguments differ from the failed process')
        env = generation.environment(base, dict(root=str(path)))
        require({k: v.replace(str(path), first['root']) for k, v in env.items()} == prior_env
                and 'STRATA_COMMIT_SYNC' not in env and env.get('STRATA_SYCL_VERIFY_OVERLAP', '1') != '0',
                'keep the original environment and overlap before setting the bounded diagnostic controls')
        env = model.native_environment(base, dict(env, **OBSERVER_ENV))
        short = model.request_ids(Path(config['tokenizer']))['ids']
        require(len(short) == 82, 'the checked short request changed')
        long = [i % 4096 for i in range(n - len(short))] + short
        requests = [(ids, OUTPUT) for ids in (short, long, short)]
        plans[case] = dict(command=command, environment_sha256=bank.signature(env),
            wire=[dict(prompt=len(ids), output=count, sha256=model.restore.signature_bytes(model.wire(ids, count)))
                  for ids, count in requests])
        launches[case] = (env, requests)

    plan = dict(source_commit=revision, inputs=dict(files=len(pins), sha256=bank.signature(pins)),
        native_source_commit=base['native_source_commit'], native_build_sha256=base['native_build_sha256'],
        binary=base['binary'], boot=base['boot'], driver=base['driver'], runtime_manifest=base['runtime_manifest'],
        drafter_assets=base['dflash2_assets'], failure=FAILURE, recovery_sha256=bank.signature(recovered['result']),
        validation=base['native_validation_probe'], cases=plans, order=list(CASES),
        environment_changes=OBSERVER_ENV, limits=model.LIMITS,
        scope='Keep verifier overlap enabled and retire each commit before the next draft. At 11793 and then '
              '32769 fresh synthetic IDs, require 32 outputs, a width-four window at or beyond the previous '
              'fault position, all-GPU BF16 residuals, checked loan refill, repeatable short output, and '
              'normal release without a new Xe fault. Use independent processes and stop on the first failure. '
              'The agent history is different and the traced runs are for diagnosis.')
    return plan, launches


def inspect(log, plan, pid, n, requests):
    proof = basis.validation.preload_evidence(log, pid)
    require(proof['loader'] == plan['validation']['loader']['path']
            and not re.search(r'(?m)^strata mtp: .*loaded|UR_RESULT_ERROR_|ZE_RESULT_ERROR_|teardown uncertain', log)
            and log.count('strata native head DPAS: ready=1 columns=4 layout=canonical') == 1,
            'unexpected validation, target head or MTP load')
    taps = re.findall(r'^strata DFlash2: eager, prompt_tail_rows=(\d+) cold_prefix=0 vram_bytes=(\d+)$', log, re.M)
    require(len(taps) == 3 and [int(rows) for rows, _ in taps] == [81, 2046, 81]
            and all(int(size) > 0 for _, size in taps)
            and requests[0] == requests[2] and all(r['offered'] > 0 for r in requests)
            and log.count('strata prefill policy: requested=throughput route=cache-window') == 2,
            'unexpected DFlash2 taps, draft work or reset output')
    windows = [list(map(int, v)) for v in re.findall(r'^strata trace: window (\d+) (\d+)$', log, re.M)]
    middle = [v for v in windows if n - 1 <= v[0] < n + OUTPUT]
    require(any(pos >= 11801 and width == 4 for pos, width in middle), 'the middle request missed the width-four gate')
    store = model.restore.require_store(log, dict(direct=False), dict(read=n))
    banks = re.findall(r'^strata MoE expert bank: requested_slots=(\d+) effective_slots=(\d+) '
                       r'staged=(\d+) host_slab_bytes=\d+ device_bytes=(\d+) ', log, re.M)
    masks = re.findall(r'^strata cache loan: phase=masked generation=(\d+) chunk=8192 first=(\d+) lent=(\d+) decode_ready=0$', log, re.M)
    ready = re.findall(r'^strata cache loan: phase=ready generation=(\d+) chunk=8192 first=(\d+) lent=(\d+) decode_ready=1$', log, re.M)
    require(banks == [('384', '384', '1', '1022361600')] and store['gpu_rows'] == n - 1
            and store['window_mode'] == 10 and len(masks) == 1 and masks == ready
            and ready[0][1:] == ('132', '1124') and 'payload_match=1' in log, 'unexpected residual route or cache refill')
    return dict(validation=proof, placement=store, loan=list(map(int, ready[0])),
                tap_rows=[int(rows) for rows, _ in taps], middle_windows=middle)


def save(row):
    bank.stage.write(ROOT / 'record.json', row)
    result = bank.previous.read(bank.RESULT)
    result[KEY] = row
    bank.stage.write(bank.RESULT, result)


def run_case(plan, path, case, env, requests, inspect_case, *, request_reader=model.request,
             limits=model.LIMITS):
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
                'unexpected context, drafter or placement')
        outcome['info'] = model.INFO
        reuse = plan['cases'][case].get('reused', [0] * len(requests))
        for (ids, budget), reused_tokens in zip(requests, reuse, strict=True):
            done = request_reader(io, ids, budget, reused_tokens=reused_tokens)
            outcome['requests'].append(done)
            print(case, done, flush=True)
        io.send(b'QUIT\n')
        io.exit(90)

    try:
        entry, _, stderr = owner.run('model', plan['cases'][case]['command'], env, bank.P,
                                     interaction=interact, **limits)
        outcome['native_exit_code'] = entry.get('exit_code')
        identity = entry.get('direct_identity', {})
        outcome['native_identity'] = {k: identity.get(k) for k in ('pid', 'start_ticks')}
        require(model.owner_module.closed(entry) and entry['exit_code'] == 0,
                entry.get('error') or 'native owner or exit differs')
        outcome.update(inspect_case(stderr.read_text(), entry['pid'], outcome['requests']))
        require(all(bank.sha(p) == h for p, h in plan['drafter_assets'].items()), 'draft assets changed')
        outcome.update(complete=True, host_probe_accepted=True)
    except Exception as error:
        outcome['error'] = repr(error)
    finally:
        basis.validation.finish(outcome, owner, cursor, plan['boot'])
        outcome['passed'] = outcome['passed'] and outcome.get('driver_after') == plan['driver']
    return outcome


def run(row, case, env, requests):
    plan = row['plan']
    outcome = run_case(plan, ROOT / case, case, env, requests,
                       lambda log, pid, responses: inspect(log, plan, pid, CASES[case], responses))
    row['outcomes'][case] = outcome
    complete = len(row['outcomes']) == len(CASES) and outcome['passed']
    row.update(stage=('closed' if complete else 'partial') if outcome['passed'] else 'failed',
               complete=complete, passed=complete, active=False, ownership_closed=outcome['ownership_closed'])
    save(row)
    print({k: outcome.get(k) for k in ('passed', 'native_exit_code', 'fresh_faults', 'error')}, flush=True)
    require(outcome['passed'], outcome.get('error', 'DFlash2 commit-order check failed'))


def main(action, case=None):
    require(sys.executable == basis.reference.model.PYTHON
            and ((action == 'prepare' and case is None) or (action == 'run' and case in CASES)),
            'use the pinned Python with prepare or run near/long')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        if action == 'prepare':
            require(not ROOT.exists() and KEY not in basis.validation.committed(bank.RESULT), 'new plan required')
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
