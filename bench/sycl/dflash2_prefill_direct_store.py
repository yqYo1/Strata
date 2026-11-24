"""Check direct BF16 attention output in the retained DFlash2 split route."""

import fcntl
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import dflash2_prefill_kv_stage_speed as speed
import prefill_split_store as fixture
from engine_once import native_environment
from prefill_gather_store import validate_traced_fixture

gate, bank, basis, require = speed.gate, speed.bank, speed.basis, speed.require
model, split = gate.model, gate.split
THIS = str(Path(__file__).resolve().relative_to(bank.P))
FIXTURE_KEY = 'dflash2_prefill_direct_store_component_v1'
FIXTURE_ROOT = bank.B / FIXTURE_KEY.replace('_', '-')
KEY = 'dflash2_prefill_direct_store_admission_v1'
ROOT = bank.B / KEY.replace('_', '-')
READ, CHUNK = 84721, 8192
SPLIT_TIMING = '7af266dcf73937d0f78ac0ecc0c88eec55cc2211'
ARMS = {'copy': False, 'direct': True}
SELECTOR = 'STRATA_SYCL_PREFILL_DIRECT_ATTENTION_STORE'
ATTENTION_READER = 'STRATA_SYCL_PREFILL_READ_ATTENTION_BANK'
PIECES = 48 * ((READ - 1 + CHUNK - 1) // CHUNK)
COPY_BYTES = 48 * (READ - 1) * split.workspace.ROW


def reference():
    row = speed.baseline(gate.SPLIT_READ)
    measured = bank.previous.historic(SPLIT_TIMING, speed.SPLIT_KEY)
    require(measured['stage'] == 'closed' and measured['complete'] and measured['passed']
            and measured['ownership_closed'] and not measured['active']
            and measured['decision']['repeated_warm_gain']
            and all(measured[name] == row[name] for name in
                    ('binary', 'native_source_commit', 'native_build_sha256', 'driver', 'boot', 'runtime_manifest')),
            'recover the matching measured retained-KV split route')
    runtime = row['runtime_manifest']
    providers = bank.previous.read(Path(runtime['path']))
    require(bank.sha(runtime['path']) == runtime['sha256'] and len(providers) == runtime['files']
            and all(bank.sha(path) == sha for path, sha in providers.items())
            and Path('/proc/sys/kernel/random/boot_id').read_text().strip() == row['boot']
            and bank.builder.guards.driver_guard() == row['driver'], 'runtime, driver or boot changed')
    native = bank.previous.read(speed.build.DFLASH2_KV_ROOT / 'record.json')
    return row, native, providers


def inputs(revision, paths):
    pins = speed.experiment(gate.SPLIT_READ).pins(revision)
    for name in [THIS, *paths]:
        path = bank.P / name
        require(path.read_bytes() == basis.validation.git('show', revision + ':' + name),
                'changed DFlash2 attention-store input: ' + name)
        pins[name] = bank.sha(path)
    return dict(files=len(pins), sha256=bank.signature(pins))


def environment(row, path, direct):
    env = speed.environment(row, dict(root=str(path), index=0, arm='retain'), gate.SPLIT_READ)
    require(env.get(SELECTOR, '0') == env.get(ATTENTION_READER, '0') == '0',
            'expected the original copied attention writer and scratch reader')
    env.update({SELECTOR: str(int(direct)), ATTENTION_READER: '0'})
    return env


def fixture_inputs(revision):
    row, native, providers = reference()
    old = basis.validation.committed(bank.RESULT, ('prefill_kv_upload_store_component_v1',))
    binary = native['binaries'][fixture.FIXTURE]
    require(old['complete'] and old['passed'] and old['ownership_closed'] and not old['active']
            and old['result']['cases'] == 4 and not old['result']['fresh_faults']
            and all(binary[k] == old['plan']['fixture'][k] for k in ('sha256', 'bytes'))
            and Path(binary['path']).stat().st_size == binary['bytes']
            and bank.sha(binary['path']) == binary['sha256'],
            'verify the current build contains the previously checked attention fixture')
    paths = (fixture.THIS, fixture.attention.CONTROLLER, 'bench/sycl/prefill_gather_store.py')
    pins = inputs(revision, paths)
    env = environment(row, FIXTURE_ROOT, False)
    env.update(TMPDIR=str(FIXTURE_ROOT / 'tmp'), XDG_CACHE_HOME=str(FIXTURE_ROOT / 'cache'))
    env = native_environment(row, env)
    env.update(UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
               UR_LOG_TRACING='level:info;flush:info;output:stderr')
    plan = dict(source_commit=revision, sources_count=pins['files'], sources_sha256=pins['sha256'],
                native_source_commit=row['native_source_commit'], native_build_sha256=row['native_build_sha256'],
                engine=row['binary'], fixture=binary, command=[binary['path'], 'correctness'],
                budgets=fixture.attention.BUDGET, environment_sha256=bank.signature(env),
                providers_sha256=bank.signature(providers), split_timing=SPLIT_TIMING,
                boot=row['boot'], driver=row['driver'],
                scope='Check exact BF16 store bits, a partial chunk, bounds and checked cancellation.')
    return plan, env, row['native_validation_probe']


def fixture_context():
    ctx = basis.core.source.device.load_module('dflash2_attention_store_fixture', fixture.__file__)
    ctx.THIS, ctx.KEY, ctx.ROOT, ctx.inputs = THIS, FIXTURE_KEY, FIXTURE_ROOT, fixture_inputs
    ctx.validate_runtime = validate_traced_fixture
    return ctx


def model_inputs(revision):
    row, native, providers = reference()
    component = basis.validation.committed(bank.RESULT, (FIXTURE_KEY,))
    proof, observed = component['plan'], component['result']
    require(component['stage'] == 'closed' and component['complete'] and component['passed']
            and component['ownership_closed'] and not component['active']
            and observed['cases'] == 4 and observed['native_exit_code'] == 0 and not observed['fresh_faults']
            and proof['fixture'] == native['binaries'][fixture.FIXTURE] and proof['engine'] == row['binary']
            and proof['providers_sha256'] == bank.signature(providers) and proof['split_timing'] == SPLIT_TIMING
            and all(proof[k] == row[k] for k in ('native_source_commit', 'native_build_sha256', 'driver', 'boot')),
            'complete the same-build traced attention-store check')
    pins = inputs(revision, [gate.device.THIS])
    plans, launches = {}, {}
    for arm, direct in ARMS.items():
        path = ROOT / arm
        config = speed.configuration(row, dict(root=str(path)))
        args = config['args']
        args[args.index('--prompt-cache') + 1] = '1'
        env = environment(row, path, direct)
        env.update(STRATA_TRACE='1', STRATA_SYCL_CACHE_LOAN_AUDIT='1', STRATA_STATE_HASH='1')
        env = native_environment(row, env)
        short = model.request_ids(Path(config['tokenizer']))['ids']
        require(len(short) == 82 and (READ - 1) % CHUNK == 2800, 'expected the checked short input and split tail')
        long = [i % 4096 for i in range(READ - len(short))] + short
        requests = [(ids, split.OUTPUT) for ids in (short, long, short)]
        wire = [dict(read=len(ids), output=budget, sha256=model.restore.signature_bytes(model.wire(ids, budget)))
                for ids, budget in requests]
        plans[arm] = dict(command=[row['binary']['path'], '--serve', *args],
                          environment_sha256=bank.signature(env), wire=wire, direct=direct)
        launches[arm] = env, requests
    normal = [{k: v.replace(str(ROOT / arm), '<case>') for k, v in launches[arm][0].items()
               if k != SELECTOR} for arm in ARMS]
    require(normal[0] == normal[1] and plans['copy']['command'] == plans['direct']['command']
            and plans['copy']['wire'] == plans['direct']['wire'], 'only vary the attention writer')
    plan = dict(source_commit=revision, inputs=pins, pieces=PIECES, copy_bytes=COPY_BYTES,
                native_source_commit=row['native_source_commit'], native_build_sha256=row['native_build_sha256'],
                binary=row['binary'], boot=row['boot'], driver=row['driver'], runtime_manifest=row['runtime_manifest'],
                validation=row['native_validation_probe'], drafter_assets=row['dflash2_assets'],
                order=list(ARMS), cases=plans, limits=model.LIMITS, component_sha256=bank.signature(component),
                scope='Copy then direct in independent traced processes. Require matching all-GPU split geometry, '
                      'per-layer retained KV, cache refill, active state and short/long/short DFlash2 outputs. '
                      'Stop on any changed input, ownership, route, runtime error, or fault. This is admission; '
                      'the subsequent clean benchmark measures prompt speed.')
    return plan, launches, row


def inspect(log, plan, pid, arm, requests):
    result = split.inspect(log, plan, pid, READ, requests, direct=ARMS[arm])
    markers = list(re.finditer(r'^strata trace: request (\d+) 0$', log, re.M))
    frame = log[markers[1].end():markers[2].start()]
    kv = speed.KV.check_request(frame, 0, READ - 1, CHUNK, reuse=True, tap_layers=gate.TAPPED_QSA_LAYERS)
    store = result['split']['store']
    require(kv['reused_pages'] > 0 and store['pieces'] == PIECES and
            store['direct'] == (PIECES if ARMS[arm] else 0) and
            store['copy_bytes'] == (0 if ARMS[arm] else COPY_BYTES), 'requested attention work differs')
    return dict(result, kv_stage=kv)


def work(outcome):
    route = dict(outcome['split'])
    route['store'] = {k: v for k, v in route['store'].items() if k not in ('direct', 'copy_bytes')}
    return dict(split=route, **{k: outcome[k] for k in ('requests', 'states', 'loan', 'tap_rows', 'kv_stage')})


def save(row):
    gate.save(row, ROOT, KEY)


def prepare():
    require(not ROOT.exists() and KEY not in basis.validation.committed(bank.RESULT), 'fresh model plan required')
    plan, _, _ = model_inputs(basis.validation.git('rev-parse', 'HEAD').decode().strip())
    for arm in ARMS:
        for name in ('cache', 'tmp'):
            (ROOT / arm / name).mkdir(mode=0o700, parents=True)
    save(dict(plan=plan, stage='prepared', complete=False, passed=False, active=False,
              ownership_closed=True, outcomes={}))


def run(arm):
    row = bank.previous.read(ROOT / 'record.json')
    earlier = list(ARMS)[:list(ARMS).index(arm)]
    case = ROOT / arm
    require(row == basis.validation.committed(bank.RESULT, (KEY,)) and row['ownership_closed'] and not row['active']
            and list(row['outcomes']) == earlier and all(row['outcomes'][old]['passed'] for old in earlier)
            and row['stage'] == ('partial' if earlier else 'prepared')
            and not (case / 'owned').exists() and not (case / 'execution.json').exists(),
            'require the committed model plan and closed earlier cases')
    plan, launches, reference_row = model_inputs(row['plan']['source_commit'])
    require(plan == row['plan'], 'model plan or inputs changed')
    env, requests = launches[arm]

    def inspect_case(log, pid, done):
        result = inspect(log, plan, pid, arm, done)
        speed.verify_model_assets(reference_row, {'args': plan['cases'][arm]['command'][2:]})
        return result

    outcome = gate.device.run_case(plan, case, arm, env, requests, inspect_case)
    row['outcomes'][arm] = outcome
    if arm == 'direct' and outcome['passed']:
        copy = row['outcomes']['copy']
        matched = work(copy) == work(outcome) and copy['native_identity'] != outcome['native_identity']
        removed = copy['split']['store']['copy_bytes'] - outcome['split']['store']['copy_bytes']
        row['comparison'] = dict(matched=matched, requested_D2D_bytes_removed=removed)
        if not matched or removed != COPY_BYTES:
            outcome.update(passed=False, error='direct store changed native work or committed state')
    complete = arm == 'direct' and outcome['passed']
    row.update(stage=('closed' if complete else 'partial') if outcome['passed'] else 'failed',
               complete=complete, passed=complete, active=not outcome['ownership_closed'],
               ownership_closed=outcome['ownership_closed'])
    save(row)
    print({k: outcome.get(k) for k in ('passed', 'native_exit_code', 'fresh_faults', 'error')}, flush=True)
    require(outcome['passed'], outcome.get('error', 'DFlash2 split store did not qualify'))


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(sys.executable == basis.reference.model.PYTHON
            and action in ('prepare-fixture', 'run-fixture', 'prepare', *ARMS),
            'use the pinned Python with one planned stage')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        if action == 'prepare-fixture':
            fixture_context().prepare()
        elif action == 'run-fixture':
            fixture_context().run()
        elif action == 'prepare':
            prepare()
        else:
            run(action)
