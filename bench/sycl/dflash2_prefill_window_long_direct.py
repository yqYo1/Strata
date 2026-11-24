"""Check either attention copy change on the admitted 122,881-input bank32 route."""

import fcntl
from hashlib import sha256
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import dflash2_prefill_window_bank as extent

bank, basis, require = extent.bank, extent.basis, extent.require
window, read = extent.window, extent.LONG_READ
KEY = 'dflash2_prefill_window_122881_direct_v1'
INPUT_KEY = 'dflash2_prefill_window_122881_input_v1'
THIS = str(Path(__file__).resolve().relative_to(bank.P))
COPY_KEY = extent.RESULT_KEYS[read]
COPY_TRACE = 'bb974ff52765fb084dd790b2f109a7532b953b30'
COPY_ATTENTION_TRACE_SHA256 = 'e6dde097d693ed2043c23ed707adcde296385b41ad337b5ae2819aeb601a74d5'
CASE = extent.CASE


def attention_input(log, plan, *, reader):
    from prefill_resident_input_model import input_receipt

    markers = list(re.finditer(r'^strata trace: request (\d+) 0$', log, re.M))
    require([int(marker[1]) for marker in markers] == [82, read, 82], 'unexpected reader request frames')
    prefix, start, end = (marker.start() for marker in markers)
    rows = read - 1
    budget = plan['cases'][CASE]['budget']
    chunk = window.store.CHUNK
    pieces = 48 * ((rows + chunk - 1) // chunk)
    layout = dict(gpu_rows=rows, maximum_rows=budget['maximum_rows'], pieces=pieces)
    require(input_receipt(log[prefix:start], 81, dict(pieces=0), reader) is None
            and input_receipt(log[end:], 81, dict(pieces=0), reader) is None,
            'the short path borrowed a prefill input')
    return input_receipt(log[start:end], rows, layout, reader)


def copy_input_evidence(control, plan, revision):
    from prefill_resident_input_model import __file__ as reader_source

    source = Path(reader_source).resolve().relative_to(bank.P)
    require((bank.P / source).read_bytes() == basis.validation.git('show', revision + ':' + str(source)),
            'commit the exact input evidence reader')
    trace_root = bank.B / COPY_KEY.replace('_', '-') / CASE
    receipt = trace_root / 'execution.json'
    original = bank.previous.read(receipt)
    command, = original['commands']
    identity = command['direct_identity']
    require(not original['active'] and command['exit_code'] == 0 and command['session_empty']
            and command['direct_child_reaped']
            and {k: identity[k] for k in ('pid', 'start_ticks')} == control['native_identity'],
            'input copy trace has different or live ownership')
    log = trace_root / 'owned/model.stderr'
    require(0 < log.stat().st_size <= plan['limits']['text_cap'], 'invalid retained input copy capture')
    captured = log.read_bytes()
    require(sha256(captured).hexdigest() == COPY_ATTENTION_TRACE_SHA256, 'reference attention trace changed')
    counters = attention_input(captured.decode(), plan, reader=False)
    require(counters == dict(eligible=690, borrowed=0, borrowed_bytes=0, copy_bytes=115762790400,
                             checked_layers=48), 'original input copy counts changed')
    return dict(counters=counters, trace_sha256=sha256(captured).hexdigest(),
                execution_sha256=bank.sha(receipt), reader_source=str(source),
                reader_sha256=bank.sha(bank.P / source))


def inputs(revision, job_root, *, reader):
    original = basis.validation.committed(bank.RESULT, (COPY_KEY,))
    require(original == bank.previous.historic(COPY_TRACE, COPY_KEY) and original['stage'] == 'closed'
            and original['complete'] and original['passed'] and original['ownership_closed']
            and not original['active'], 'require the closed 122881-input copy control')
    control = original['outcome']
    require(control['first_verifier_window'] == [read - 1, 1] and control['native_exit_code'] == 0
            and not control['fresh_faults'] and original['comparison']['matched_short_controls'],
            'require the completed same-route copy process')
    plan, env, requests, reference, _ = extent.inputs(revision, read, job_root)
    previous = original['plan']
    copy = previous['cases'][CASE]
    case = plan['cases'][CASE]
    lhs = {k: v.replace(str(job_root / CASE), str(bank.B / COPY_KEY.replace('_', '-') / CASE))
           for k, v in env.items()}
    require(all(plan[k] == previous[k] for k in (*extent.NATIVE_IDENTITY, 'fixed_bank_sha256',
                                                 'packed_layers_sha256', 'projected_free_margin_bytes'))
            and all(case[k] == copy[k] for k in ('command', 'wire', 'budget'))
            and bank.signature(lhs) == copy['environment_sha256']
            and env[window.store.SELECTOR] == '0' and env[window.store.ATTENTION_READER] == '0',
            'the copy control has different native inputs or policy')
    source = bank.P / THIS
    require(source.read_bytes() == basis.validation.git('show', revision + ':' + THIS),
            'changed longer direct-writer controller')
    selector = window.store.ATTENTION_READER if reader else window.store.SELECTOR
    env[selector] = '1'
    scope = ('Require 690 borrowed GPU input spans and keep copied output.' if reader
             else 'Require direct output for all 720 stores and keep the scratch input reader.')
    plan.update(cases={CASE: dict(case, environment_sha256=bank.signature(env))},
                copy_trace_commit=COPY_TRACE, copy_trace_sha256=bank.signature(original),
                direct_reader_sha256=bank.sha(source),
                scope='Repeat the admitted 82 / 122881 / 82 inputs with the same bank32 and demand policy. '
                      + scope + ' Require identical expert work, target state, retained KV, DFlash2 taps, '
                      'cache refill and verifier handoff.')
    plan['copy_attention_input'] = copy_input_evidence(control, plan, revision)
    return plan, env, requests, reference, control


def prepare(job_root, key, *, reader):
    require(not job_root.exists() and key not in basis.validation.committed(bank.RESULT), 'fresh attention case required')
    revision = basis.validation.git('rev-parse', 'HEAD').decode().strip()
    plan, _, _, _, _ = inputs(revision, job_root, reader=reader)
    for name in ('cache', 'tmp'):
        (job_root / CASE / name).mkdir(mode=0o700, parents=True)
    window.store.gate.save(dict(plan=plan, stage='prepared', complete=False, passed=False,
                                ownership_closed=True, active=False), job_root, key)


def run(job_root, key, *, reader):
    row = bank.previous.read(job_root / 'record.json')
    root = job_root / CASE
    require(row == basis.validation.committed(bank.RESULT, (key,)) and row['stage'] == 'prepared'
            and row['ownership_closed'] and not row['active'] and not (root / 'owned').exists()
            and not (root / 'execution.json').exists(), 'require the committed unused attention plan')
    plan, env, requests, reference, control = inputs(row['plan']['source_commit'], job_root, reader=reader)
    require(plan == row['plan'], 'frozen attention inputs changed')

    def inspect(log, pid, answers):
        result = extent.inspect(log, plan, pid, answers, read, reference, direct=not reader)
        result['attention_input'] = attention_input(log, plan, reader=reader)
        return result

    outcome = window.device.run_case(plan, root, CASE, env, requests, inspect)
    if outcome['passed']:
        fields = ('requests', 'states', 'info', 'loan', 'kv_stage', 'tap_rows', 'packed_layers_sha256',
                  'usable_slots_by_layer', 'first_verifier_window')
        work_fields = ('fixed_with_headroom', 'requested_bank_H2D_bytes', 'uploads', 'jobs', 'dequants',
                       'windows', 'maximum_rows')
        state_matches = all(outcome[k] == control[k] for k in fields)
        work_matches = all(outcome['route'][k] == control['route'][k] for k in work_fields)
        distinct = outcome['native_identity'] != control['native_identity']
        old_output = control['route']['logical_copy_bytes']
        remaining_output = outcome['route']['logical_copy_bytes']
        output_ok = old_output == 48 * (read - 1) * window.workspace.ROW
        old_input = plan['copy_attention_input']['counters']
        observed = outcome['attention_input']
        if reader:
            removed = old_input['copy_bytes']
            copy_ok = (output_ok and remaining_output == old_output and observed['copy_bytes'] == 0
                       and observed['borrowed'] == old_input['eligible']
                       and observed['borrowed_bytes'] == removed)
        else:
            removed = old_output
            copy_ok = output_ok and remaining_output == 0 and observed == old_input
        row['comparison'] = dict(matched_state=state_matches, matched_work=work_matches,
                                 distinct_process=distinct, requested_D2D_bytes_removed=removed)
        if not (state_matches and work_matches and distinct and copy_ok):
            outcome.update(passed=False, error='the attention copy change failed its state, work, or ownership check')
    passed = bool(outcome['passed'])
    row.update(outcome=outcome, stage='closed' if passed else 'failed', complete=passed, passed=passed,
               ownership_closed=outcome['ownership_closed'], active=not outcome['ownership_closed'])
    window.store.gate.save(row, job_root, key)
    print({k: outcome.get(k) for k in ('passed', 'native_exit_code', 'fresh_faults', 'error')}, flush=True)
    require(passed, outcome.get('error') or 'long attention copy validation failed')


if __name__ == '__main__':
    action, *options = sys.argv[1:]
    require(sys.executable == basis.reference.model.PYTHON and action in ('prepare', 'run')
            and options in ([], ['--reader']), 'use prepare or run, optionally followed by --reader')
    reader = bool(options)
    key = INPUT_KEY if reader else KEY
    job_root = bank.B / key.replace('_', '-')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        {'prepare': prepare, 'run': run}[action](job_root, key, reader=reader)
