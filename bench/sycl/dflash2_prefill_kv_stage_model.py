"""Check the aligned DFlash2 KV-stage change on an owned V2 server."""

import fcntl
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import dflash2_v2_commit_order as device
import prefill_kv_upload_build as build

generation, model = device.generation, device.model
bank, basis, require = device.bank, device.basis, device.require
THIS = str(Path(__file__).resolve().relative_to(bank.P))
KEY = 'dflash2_prefill_kv_stage_model_v1'
ROOT = bank.B / KEY.replace('_', '-')
ARMS = {'reload': '0', 'retain': '1'}
READ, OUTPUT = 32769, 32
RAGGED_PREFIX, RAGGED_READ = 83, 8209
TAPPED_QSA_LAYERS = (15, 35)
EXPECTED_H2D_REDUCTION = 467140608


def inputs(revision):
    base = generation.sync_baseline()
    native = bank.previous.read(build.DFLASH2_KV_ROOT / 'record.json')
    build.context('dflash2-kv-stage').check(native)
    require(native['stage'] == 'built' and native['complete'] and native['passed']
            and native['ownership_closed'] and not native['active'] and native['host_control']['passed']
            and native['projection']['dflash2_kv_base_binary'] == base['binary']
            and native['projection']['dflash2_kv_source'] == build.DFLASH2_KV_SOURCE
            and native['driver'] == base['driver'], 'require the coherent candidate and its host fixtures')
    proof = bank.previous.historic(generation.COMMIT_ORDER_GATE, device.KEY)
    positive = proof['outcomes']['long']
    require(positive['passed'] and positive['native_exit_code'] == 0 and positive['fresh_faults'] == [],
            'recover the real DFlash2 32769-token model gate')
    paths = [THIS, device.THIS, build.THIS]
    pins = generation.experiment().pins(revision)
    pins.update(bank.stage.source.pin(bank.P, paths)['source_sha256'])
    require(all((bank.P / name).read_bytes() == basis.validation.git('show', revision + ':' + name)
                for name in paths), 'changed committed model controller')

    plans, launches = {}, {}
    binary = native['binaries']['strata']
    for arm, value in ARMS.items():
        path = str(ROOT / arm)
        cfg = generation.configuration(base, dict(root=path))
        args = cfg['args']
        require(['--serve', *args] == proof['plan']['cases']['long']['command'][1:],
                'keep the previous model and placement arguments')
        # Cache the last short request so the next one can exercise an unaligned
        # fresh suffix from its live prefix. Prefix checkpoints stay disabled.
        args[args.index('--prompt-cache') + 1] = '1'
        command = [binary['path'], '--serve', *args]
        env = generation.sync_environment(base, dict(root=path))
        env.update(STRATA_TRACE='1', STRATA_SYCL_CACHE_LOAN_AUDIT='1', STRATA_STATE_HASH='1')
        env[generation.v2.state.KV] = value
        env = model.native_environment(base, env)

        short = model.request_ids(Path(cfg['tokenizer']))['ids']
        long = [i % 4096 for i in range(READ - len(short))] + short
        prefix = [64] + short
        require(len(prefix) == RAGGED_PREFIX and RAGGED_PREFIX % 4 != 0, 'unaligned live prefix')
        extended = prefix + [i % 4096 for i in range(RAGGED_READ - len(short))] + short
        requests = [(ids, OUTPUT) for ids in (short, long, short)] + [(prefix, 1), (extended, 1)]
        wire = [dict(prompt=len(ids), output=count, sha256=model.restore.signature_bytes(model.wire(ids, count)))
                for ids, count in requests]
        require(wire[:3] == proof['plan']['cases']['long']['wire'], 'use the checked short/long/short inputs')
        plans[arm] = dict(command=command, environment_sha256=bank.signature(env), wire=wire,
                          reused=[0, 0, 0, 0, RAGGED_PREFIX])
        launches[arm] = env, requests
    ignored = {generation.v2.state.KV}
    normalized = [{k: v.replace(str(ROOT / a), '<case>') for k, v in launches[a][0].items() if k not in ignored}
                  for a in ARMS]
    require(normalized[0] == normalized[1], 'only select the KV retention mode between processes')
    plan = dict(source_commit=revision, inputs=dict(files=len(pins), sha256=bank.signature(pins)),
        native_source_commit=native['native_source_commit'], build_sha256=bank.signature(build.builder.compact(native)),
        binary=binary, boot=base['boot'], driver=base['driver'], runtime_manifest=base['runtime_manifest'],
        drafter_assets=base['dflash2_assets'], validation=base['native_validation_probe'],
        cases=plans, order=list(ARMS), limits=model.LIMITS, model_reference=generation.COMMIT_ORDER_GATE,
        tap_layers=list(TAPPED_QSA_LAYERS), expected_prefix_H2D_bytes_removed=EXPECTED_H2D_REDUCTION,
        scope='Independent reload and retain processes. Check the matched 32769-token and unaligned continuation requests. '
              'Compare per-layer KV counts, retained state and generated work, with tap and cache-loan checks. '
              'Preserve the exact candidate and both original references. These traced requests test correctness.')
    return plan, launches


def inspect(log, pid, plan, arm, requests):
    pattern = r'^strata trace: request (\d+) 0$'
    markers = list(re.finditer(pattern, log, re.M))
    require([int(mark[1]) for mark in markers] == [82, READ, 82, RAGGED_PREFIX, RAGGED_PREFIX + RAGGED_READ],
            'the native trace must cover all five wire requests')
    result = device.inspect(log[:markers[3].start()], plan, pid, READ, requests[:3])
    trace = generation.v2.state.kv
    retain = ARMS[arm] == '1'
    result['kv_stage'] = trace.check_request(log[markers[1].end():markers[2].start()],
        0, READ - 1, 8192, reuse=retain, tap_layers=plan['tap_layers'])
    ragged = log[markers[4].end():]
    result['ragged_kv_stage'] = trace.check_request(ragged, RAGGED_PREFIX, RAGGED_READ - 1, 8192,
                                                  reuse=retain, tap_layers=plan['tap_layers'])
    result['ragged_store'] = model.restore.require_store(ragged, dict(direct=False), dict(read=RAGGED_READ))
    require(result['ragged_kv_stage']['reused_pages'] == 0 and
            result['ragged_store']['gpu_rows'] == RAGGED_READ - 1, 'unaligned chunks must reload every prefix')
    lines = re.findall(r'^strata serve: STATE_HASH L=.*$', log, re.M)
    states = [{k: v for k, v in re.findall(r'(\w+)=([0-9a-f,-]+)', line) if k in model.restore.ACTIVE_STATE}
              for line in lines]
    require(len(states) == 5 and all(set(state) == set(model.restore.ACTIVE_STATE) for state in states)
            and states[0] == states[2], 'the restored short request state must agree')
    result['states'] = states
    return result


def save(row):
    bank.stage.write(ROOT / 'record.json', row)
    result = bank.previous.read(bank.RESULT)
    result[KEY] = row
    bank.stage.write(bank.RESULT, result)


def run(row, arm, env, requests):
    plan = row['plan']
    outcome = device.run_case(plan, ROOT / arm, arm, env, requests,
                              lambda log, pid, responses: inspect(log, pid, plan, arm, responses))
    row['outcomes'][arm] = outcome
    if arm == 'retain' and outcome['passed']:
        before = row['outcomes']['reload']
        fields = ('states', 'requests', 'tap_rows', 'placement', 'loan', 'middle_windows',
                  'ragged_kv_stage', 'ragged_store')
        matched = all(before[k] == outcome[k] for k in fields)
        removed = before['kv_stage']['requested_prefix_H2D_bytes'] - outcome['kv_stage']['requested_prefix_H2D_bytes']
        row['comparison'] = dict(matched=matched, requested_prefix_H2D_bytes_removed=removed)
        if not matched or removed != plan['expected_prefix_H2D_bytes_removed']:
            row['comparison']['error'] = 'candidate did not preserve target state/work and remove the expected prefix reloads'
            outcome['passed'] = False
    complete = arm == 'retain' and outcome['passed']
    row.update(stage=('closed' if complete else 'partial') if outcome['passed'] else 'failed',
               complete=complete, passed=complete, active=not outcome['ownership_closed'],
               ownership_closed=outcome['ownership_closed'])
    save(row)
    print({k: outcome.get(k) for k in ('passed', 'native_exit_code', 'fresh_faults', 'error')}, flush=True)
    require(outcome['passed'], 'preserve the original DFlash2 result')


def main(action, arm=None):
    require(sys.executable == basis.reference.model.PYTHON
            and ((action == 'prepare' and arm is None) or (action == 'run' and arm in ARMS)),
            'use the pinned Python with prepare or run reload/retain')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        if action == 'prepare':
            require(not ROOT.exists() and KEY not in basis.validation.committed(bank.RESULT), 'new plan required')
            plan, _ = inputs(basis.validation.git('rev-parse', 'HEAD').decode().strip())
            for name in ARMS:
                for folder in ('cache', 'tmp'):
                    (ROOT / name / folder).mkdir(parents=True, mode=0o700)
            save(dict(plan=plan, stage='prepared', complete=False, passed=False, active=False,
                      ownership_closed=True, outcomes={}))
            return
        row = bank.previous.read(ROOT / 'record.json')
        plan, launches = inputs(row['plan']['source_commit'])
        earlier = list(ARMS)[:list(ARMS).index(arm)]
        require(row == basis.validation.committed(bank.RESULT, (KEY,)) and row['plan'] == plan
                and row['stage'] == ('partial' if earlier else 'prepared') and row['ownership_closed']
                and not row['active'] and list(row['outcomes']) == earlier
                and all(row['outcomes'][a]['passed'] for a in earlier)
                and not (ROOT / arm / 'owned').exists() and not (ROOT / arm / 'execution.json').exists(),
                'require the committed plan and closed prior arms without retries')
        run(row, arm, *launches[arm])


if __name__ == '__main__':
    main(*sys.argv[1:])
