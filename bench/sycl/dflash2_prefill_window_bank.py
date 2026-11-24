"""Check DFlash2 private bank capacity and a longer input on the existing executable."""

import fcntl
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import dflash2_prefill_window_admission as window

bank, basis, require = window.bank, window.basis, window.require
KEY = 'dflash2_prefill_window_bank32_v1'
LONG_READ = 122881
RESULT_KEYS = {window.READ: KEY, LONG_READ: 'dflash2_prefill_window_bank32_122881_v1'}
THIS = str(Path(__file__).resolve().relative_to(bank.P))
MANIFEST = 'tests/data/native_experts/iq3_s.txt'
COPY_TRACE = '4b25af35d4ff9f7f080b0ac562d0f39347a2656c'
FIXED_BANK_TRACE = '7e32088f5585bcffbd21fef63da08b3227f47aa4'
NATIVE_IDENTITY = ('binary', 'native_source_commit', 'native_build_sha256', 'boot', 'driver',
                   'runtime_manifest', 'validation', 'drafter_assets', 'limits')
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


def fixed_bank_reference(source, env, case, geometry):
    result = basis.validation.committed(bank.RESULT, (KEY,))
    require(result == bank.previous.historic(FIXED_BANK_TRACE, KEY) and result['stage'] == 'closed'
            and result['complete'] and result['passed'] and result['ownership_closed'] and not result['active'],
            'require the closed fixed-input 32-slot admission')
    original = result['plan']
    fixed_case = original['cases'][CASE]
    control = result['outcome']
    require(all(source[k] == original[k] for k in NATIVE_IDENTITY)
            and all(case[k] == fixed_case[k] for k in ('command', 'wire'))
            and fixed_case['environment_sha256'] == bank.signature(env)
            and fixed_case['budget'] == window.budget(16, slots=SLOTS)
            and original['packed_layers_sha256'] == control['packed_layers_sha256'] == geometry
            and len(control['usable_slots_by_layer']) == 48
            and result['comparison']['matched_work'] and result['comparison']['matched_state']
            and control['passed'] and control['native_exit_code'] == 0 and not control['fresh_faults'],
            'the fixed-input control has different owners, inputs, or private bank geometry')
    return result


def inputs(revision, read, root):
    trace = basis.validation.committed(bank.RESULT, (window.KEY,))
    require(trace == bank.previous.historic(COPY_TRACE, window.KEY) and trace['stage'] == 'closed'
            and trace['passed'] and trace['complete'] and trace['ownership_closed'] and not trace['active'],
            'require the original mode16 copy admission')
    control, original = trace['outcomes'][CONTROL], trace['plan']
    source, settings, requests, reference = window.inputs(revision)
    case = source['cases'][CONTROL]
    require(control['passed'] and control['native_exit_code'] == 0 and not control['fresh_faults']
            and all(source[k] == original[k] for k in NATIVE_IDENTITY)
            and case == original['cases'][CONTROL], 'same-build bank control changed')
    fixed_root = bank.B / KEY.replace('_', '-') / CASE
    env = {k: v.replace(str(window.ROOT / CONTROL), str(fixed_root)) for k, v in settings[CONTROL].items()}
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
    budget = window.budget(16, slots=SLOTS, rows=read - 1)
    layers = packed_layers(manifest, budget['incremental_allocations']['bank'])
    geometry = bank.signature(layers)
    scope = ('Hold the 82 / 118785 / 82 wire sequence and mode16 copied writer fixed. '
             'Change only the private bank extent, then verify the per-layer packing, expert work, '
             'target state, cache refill, tap coverage and closed owners before considering a longer shape.')
    extra = {}
    if read == LONG_READ:
        fixed = fixed_bank_reference(source, env, case, geometry)
        control = fixed['outcome']
        rows = read - 1
        require(rows == 15 * window.store.CHUNK and read - window.READ == 4096,
                'the planned longer input must add one complete token period')
        added = list(range(read - window.READ))
        tokens, output = requests[1]
        requests = [requests[0], (added + tokens, output), requests[2]]
        wire = [dict(read=len(ids), output=count, sha256=window.model.restore.signature_bytes(window.model.wire(ids, count)))
                for ids, count in requests]
        require([x['read'] for x in wire] == [82, read, 82] and wire[0] == case['wire'][0] and wire[2] == case['wire'][2],
                'longer case must keep both short controls byte-identical')
        case = dict(case, wire=wire)
        prior = control['route']
        margin = prior['free_bytes'] - budget['fixed_with_headroom']
        require(margin > 0 and budget['shared_arena'] <= prior['arena_bytes']
                and budget['largest_allocation'] <= prior['allocation_limit'],
                'projected extent misses the measured free, shared, or single-allocation bound')
        scope = ('Add one 4096-token period to the long input. Keep the 32-slot allocation and '
                 'mode16 copied-writer demand route fixed. Require 15 full GPU prefix chunks, '
                 'the last prompt token in a width-one verifier window, matching short controls '
                 'and closed owners. Record the measured capacity and state without speed selection.')
        extra = dict(fixed_bank_trace=FIXED_BANK_TRACE, fixed_bank_sha256=bank.signature(fixed),
                     projected_free_margin_bytes=margin, requested_read=read)
    env = {k: v.replace(str(fixed_root), str(root / CASE)) for k, v in env.items()}
    plan = dict(source, order=[CASE], cases={CASE: dict(case, budget=budget, environment_sha256=bank.signature(env))},
        bank_reader_sha256=reader_pins, reference=COPY_TRACE, packed_layers_sha256=geometry, scope=scope, **extra)
    return plan, env, requests, reference, control


def prepare(read, root, key):
    require(not root.exists() and key not in basis.validation.committed(bank.RESULT), 'fresh 32-slot case required')
    revision = basis.validation.git('rev-parse', 'HEAD').decode().strip()
    plan, _, _, _, _ = inputs(revision, read, root)
    for folder in ('cache', 'tmp'):
        (root / CASE / folder).mkdir(mode=0o700, parents=True)
    window.store.gate.save(dict(plan=plan, stage='prepared', complete=False, passed=False,
                                 ownership_closed=True, active=False), root, key)


def inspect(log, plan, pid, answers, read, reference, *, direct=False):
    result = window.inspect(log, plan, pid, CONTROL, answers, direct=direct, slots=SLOTS, read=read)
    start = log.index(f'strata trace: request {read} 0\n')
    end = log.index('strata trace: request 82 0', start)
    middle = log[start:end]
    layers = window.workspace.records(middle, 'strata packed expert bank: ')
    require(len(layers) == 48 and bank.signature(layers) == plan['packed_layers_sha256'],
            'observed per-layer packed bank geometry differs')
    summary = window.workspace.one(middle, 'strata MoE expert bank complete: ')
    require(summary['permanent_hits'] == '0' and result['route']['uploads'] == result['route']['dequants'],
            'unexpected expert slot reuse')
    if read == LONG_READ:
        windows = [(int(pos), int(width))
                   for pos, width in re.findall(r'^strata trace: window (\d+) (\d+)$', middle, re.M)]
        require(windows and windows[0] == (read - 1, 1)
                and middle.index('strata DFlash2: eager, ') < middle.index(f'strata trace: window {read - 1} 1\n'),
                'last prompt token or DFlash2 verifier handoff differs')
        result['first_verifier_window'] = list(windows[0])
    window.store.speed.verify_model_assets(reference, {'args': plan['cases'][CASE]['command'][2:]})
    return dict(result, packed_layers_sha256=bank.signature(layers),
                usable_slots_by_layer=[int(layer['slots']) for layer in layers])

def run(read, job_root, key):
    row = bank.previous.read(job_root / 'record.json')
    root = job_root / CASE
    require(row == basis.validation.committed(bank.RESULT, (key,)) and row['stage'] == 'prepared'
            and row['ownership_closed'] and not row['active'] and not (root / 'owned').exists()
            and not (root / 'execution.json').exists(), 'require the committed unused bank plan')
    plan, env, requests, reference, control = inputs(row['plan']['source_commit'], read, job_root)
    require(plan == row['plan'], 'frozen bank inputs changed')

    outcome = window.device.run_case(plan, root, CASE, env, requests,
        lambda log, pid, answers: inspect(log, plan, pid, answers, read, reference))
    if outcome['passed'] and read == LONG_READ:
        matched = all(outcome[key][i] == control[key][i] for key in ('requests', 'states') for i in (0, 2))
        persistent = all(outcome[key] == control[key] for key in ('info', 'loan', 'tap_rows', 'usable_slots_by_layer'))
        distinct = outcome['native_identity'] != control['native_identity']
        row['comparison'] = dict(matched_short_controls=matched, matched_decode_placement=persistent,
                                 distinct_process=distinct,
                                 prefix_rows_added=outcome['route']['maximum_rows'] - control['route']['maximum_rows'])
        if not (matched and persistent and distinct and row['comparison']['prefix_rows_added'] == read - window.READ):
            outcome.update(passed=False, error='longer extent changed the short controls, cache refill or ownership')
    elif outcome['passed']:
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
    window.store.gate.save(row, job_root, key)
    print({k: outcome.get(k) for k in ('passed', 'native_exit_code', 'fresh_faults', 'error')}, flush=True)
    require(passed, outcome.get('error') or 'bank32 validation failed')


if __name__ == '__main__':
    action, *extent = sys.argv[1:]
    require(sys.executable == basis.reference.model.PYTHON and action in ('prepare', 'run')
            and extent in ([], [str(LONG_READ)]), 'use prepare or run, optionally followed by 122881')
    read = LONG_READ if extent else window.READ
    key = RESULT_KEYS[read]
    root = bank.B / key.replace('_', '-')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        {'prepare': prepare, 'run': run}[action](read, root, key)
