"""Check one-window DFlash2 expert bank capacity with its existing executable."""

import fcntl
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import dflash2_prefill_window_admission as window

bank, basis, require = window.bank, window.basis, window.require
KEY = 'dflash2_prefill_window_bank32_v1'
ROOT = bank.B / KEY.replace('_', '-')
THIS = str(Path(__file__).resolve().relative_to(bank.P))
MANIFEST = 'tests/data/native_experts/iq3_s.txt'
COPY_TRACE = '4b25af35d4ff9f7f080b0ac562d0f39347a2656c'
CONTROL, CASE, SLOTS = 'window16', 'bank32', 32
SLOT_OPTION = 'STRATA_SYCL_PREFILL_EXPERT_BANK_SLOTS'


def packed_layers(manifest, capacity):
    layers = []
    for line in manifest.decode().splitlines():
        if not line or line.startswith('#'):
            continue
        fields = line.split()
        require(len(fields) >= 8 and int(fields[0]) == len(layers), 'native manifest order')
        stride = int(fields[4])
        require(0 < stride <= window.workspace.STRIDE and stride % 256 == 0, 'native bank stride')
        count = min(512, capacity // stride)
        layers.append(dict(layer=str(len(layers)), slots=str(count), stride=str(stride),
                           used_bytes=str(count * stride), backing_bytes=str(capacity), full_fit=str(int(count == 512))))
    require(len(layers) == 48 and all(SLOTS <= int(row['slots']) < 512 for row in layers),
            'expected the 48 private partial banks')
    return layers


def inputs(revision):
    trace = basis.validation.committed(bank.RESULT, (window.KEY,))
    require(trace == bank.previous.historic(COPY_TRACE, window.KEY) and trace['stage'] == 'closed'
            and trace['passed'] and trace['complete'] and trace['ownership_closed'] and not trace['active'],
            'require the original mode16 copy admission')
    control, original = trace['outcomes'][CONTROL], trace['plan']
    source, settings, requests, reference = window.inputs(revision)
    case = source['cases'][CONTROL]
    require(control['passed'] and control['native_exit_code'] == 0 and not control['fresh_faults']
            and all(source[k] == original[k] for k in ('binary', 'native_source_commit', 'native_build_sha256',
                    'boot', 'driver', 'runtime_manifest', 'validation', 'drafter_assets'))
            and case == original['cases'][CONTROL], 'same-build bank control changed')
    env = {k: v.replace(str(window.ROOT / CONTROL), str(ROOT / CASE)) for k, v in settings[CONTROL].items()}
    require(env[SLOT_OPTION] == str(window.SLOTS) and env[window.MODE] == '16'
            and env[window.UPLOAD] == 'demand' and env[window.store.SELECTOR] == '0',
            'keep the observed one-window demand and copy policy')
    env[SLOT_OPTION] = str(SLOTS)

    reader_pins = {}
    for name in (THIS, MANIFEST):
        path = bank.P / name
        require(path.read_bytes() == basis.validation.git('show', revision + ':' + name),
                'changed source or expert geometry: ' + name)
        reader_pins[name] = bank.sha(path)
    manifest = (bank.P / MANIFEST).read_bytes()
    args = case['command'][2:]
    require((Path(args[args.index('--pack') + 1]) / 'native_experts.txt').read_bytes() == manifest,
            'installed packed layer geometry differs')
    budget = window.budget(16, slots=SLOTS)
    layers = packed_layers(manifest, budget['incremental_allocations']['bank'])
    plan = dict(source, order=[CASE], cases={CASE: dict(case, budget=budget, environment_sha256=bank.signature(env))},
        bank_reader_sha256=reader_pins, reference=COPY_TRACE, packed_layers_sha256=bank.signature(layers),
        scope='Hold the 82 / 118785 / 82 wire sequence and mode16 copied writer fixed. '
              'Change only the private bank extent, then verify the per-layer packing, expert work, '
              'target state, cache refill, tap coverage and closed owners before considering a longer shape.')
    return plan, env, requests, reference, control


def save(row):
    window.store.gate.save(row, ROOT, KEY)


def prepare():
    require(not ROOT.exists() and KEY not in basis.validation.committed(bank.RESULT), 'fresh 32-slot case required')
    revision = basis.validation.git('rev-parse', 'HEAD').decode().strip()
    plan, _, _, _, _ = inputs(revision)
    root = ROOT / CASE
    for folder in ('cache', 'tmp'):
        (root / folder).mkdir(mode=0o700, parents=True)
    save(dict(plan=plan, stage='prepared', complete=False, passed=False, ownership_closed=True, active=False))


def run():
    row = bank.previous.read(ROOT / 'record.json')
    root = ROOT / CASE
    require(row == basis.validation.committed(bank.RESULT, (KEY,)) and row['stage'] == 'prepared'
            and row['ownership_closed'] and not row['active'] and not (root / 'owned').exists()
            and not (root / 'execution.json').exists(), 'require the committed unused bank plan')
    plan, env, requests, reference, control = inputs(row['plan']['source_commit'])
    require(plan == row['plan'], 'frozen bank inputs changed')

    def inspect(log, pid, answers):
        result = window.inspect(log, plan, pid, CONTROL, answers, slots=SLOTS)
        start = log.index('strata trace: request 118785 0')
        end = log.index('strata trace: request 82 0', start)
        middle = log[start:end]
        layers = window.workspace.records(middle, 'strata packed expert bank: ')
        require(len(layers) == 48 and bank.signature(layers) == plan['packed_layers_sha256'],
                'observed per-layer packed bank geometry differs')
        summary = window.workspace.one(middle, 'strata MoE expert bank complete: ')
        require(summary['permanent_hits'] == '0' and result['route']['uploads'] == result['route']['dequants'],
                'unexpected expert slot reuse')
        window.store.speed.verify_model_assets(reference, {'args': plan['cases'][CASE]['command'][2:]})
        return dict(result, packed_layers_sha256=bank.signature(layers),
                    usable_slots_by_layer=[int(layer['slots']) for layer in layers])

    outcome = window.device.run_case(plan, root, CASE, env, requests, inspect)
    if outcome['passed']:
        fields = ('logical_copy_bytes', 'requested_bank_H2D_bytes', 'uploads', 'jobs', 'dequants', 'windows', 'maximum_rows')
        matched = all(outcome[k] == control[k] for k in ('requests', 'info', 'states', 'kv_stage', 'tap_rows', 'loan'))
        work = all(outcome['route'][k] == control['route'][k] for k in fields)
        released = control['route']['fixed_with_headroom'] - outcome['route']['fixed_with_headroom']
        distinct = outcome['native_identity'] != control['native_identity']
        row['comparison'] = dict(matched_work=work, matched_state=matched, distinct_process=distinct,
                                 backing_bytes_removed=released)
        if not (matched and work and distinct and released == (window.SLOTS - SLOTS) * window.workspace.STRIDE):
            outcome.update(passed=False, error='bank geometry, expert work, state or ownership differs')
    passed = bool(outcome['passed'])
    row.update(outcome=outcome, stage='closed' if passed else 'failed', passed=passed, complete=passed,
               ownership_closed=outcome['ownership_closed'], active=not outcome['ownership_closed'])
    save(row)
    print({k: outcome.get(k) for k in ('passed', 'native_exit_code', 'fresh_faults', 'error')}, flush=True)
    require(passed, outcome.get('error') or 'bank32 validation failed')


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(sys.executable == basis.reference.model.PYTHON and action in ('prepare', 'run'), 'use prepare or run')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        {'prepare': prepare, 'run': run}[action]()
