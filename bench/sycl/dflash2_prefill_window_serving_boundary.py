"""Check the serving boundary of the long-window policy on the retained DFlash2 engine."""

import fcntl
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import dflash2_prefill_window_bank as extent

bank, basis, require = extent.bank, extent.basis, extent.require
window, model = extent.window, extent.window.model
KEY = 'dflash2_prefill_window_serving_boundary_v1'
ROOT = bank.B / KEY.replace('_', '-')
THIS = str(Path(__file__).resolve().relative_to(bank.P))
CAPACITY, OUTPUT, RESERVED = 262144, 1, 8
BOUNDARY = CAPACITY - RESERVED - OUTPUT
CASE = extent.CASE
COPY_RECORD = 'bb974ff52765fb084dd790b2f109a7532b953b30'
STANDALONE_KEY = 'dflash2_prefill_physical_context_v1'
WRITER_KEY = 'dflash2_prefill_window_122881_writer_speed_v1'


def input_sequence(short, short_output):
    def tokens(read):
        return [i % 4096 for i in range(read - len(short))] + short

    return [(short, short_output), (tokens(BOUNDARY + 1), OUTPUT),
            (tokens(BOUNDARY), OUTPUT), (short, short_output)]


def request(io, ids, output, *, reused_tokens=0):
    n = len(ids)
    if n not in (BOUNDARY, BOUNDARY + 1):
        return model.request(io, ids, output, reused_tokens=reused_tokens)
    require(reused_tokens == 0 and output == OUTPUT, 'unexpected native boundary request')
    data = model.wire(ids, output)
    for offset in range(0, len(data), 256 << 10):
        io.send(data[offset:offset + (256 << 10)])
    if n > BOUNDARY:
        expected = f'ERR prompt ({n} tokens) + max_new ({output}) exceeds the context ({CAPACITY})'
        require(io.line(60) == expected, 'one-token-over-limit GEN request was not rejected')
        return dict(prompt=n, rejected=True, expected=expected)
    parser = model.RequestProtocol(n, output, reused_tokens=0)
    while parser.done is None:
        parser.feed(io.line(900))
    require(parser.position == n - 1 and len(parser.tokens) == output
            and int(parser.done[6]) == int(parser.done[7]) == 0, 'incomplete last serving position')
    return dict(prompt=n, generated=len(parser.tokens), output_sha256=bank.signature(parser.tokens),
                accepted=0, offered=0, finish=parser.done[5], prefill_last=parser.position)


def inputs(revision):
    control = basis.validation.committed(bank.RESULT, (extent.RESULT_KEYS[extent.LONG_READ],))
    require(control == bank.previous.historic(COPY_RECORD, extent.RESULT_KEYS[extent.LONG_READ])
            and control['stage'] == 'closed' and control['complete'] and control['passed']
            and control['ownership_closed'] and not control['active'], 'recover the closed mode16/32-slot control')
    plan, env, old_requests, reference, _ = extent.inputs(revision, extent.LONG_READ, ROOT)
    base, case = control['plan'], plan['cases'][CASE]
    normalized = {k: v.replace(str(ROOT / CASE), str(bank.B / extent.RESULT_KEYS[extent.LONG_READ].replace('_', '-') / CASE))
                  for k, v in env.items()}
    require(all(plan[key] == base[key] for key in extent.NATIVE_IDENTITY)
            and all(case[key] == base['cases'][CASE][key] for key in ('command', 'wire', 'budget'))
            and bank.signature(normalized) == base['cases'][CASE]['environment_sha256'],
            'another native configuration input changed')
    timing = basis.validation.committed(bank.RESULT, (WRITER_KEY,))
    require(timing['stage'] == 'closed' and timing['passed'] and timing['complete']
            and timing['ownership_closed'] and not timing['active']
            and all(timing[k] == plan[k] for k in ('native_source_commit', 'native_build_sha256', 'binary'))
            and timing['decision']['repeated_warm_gain'] and timing['decision']['all_warm_requests_1000']['direct']
            and env[window.store.SELECTOR] == env[window.store.ATTENTION_READER] == '0',
            'recover the qualified writer timing and the original release/cache policy')
    env[window.store.SELECTOR] = '1'
    physical = basis.validation.committed(bank.RESULT, (STANDALONE_KEY,))
    full = physical['outcomes']['full-copy']
    require(physical['stage'] == 'closed' and physical['complete'] and physical['passed']
            and physical['ownership_closed'] and not physical['active']
            and physical['native_source_commit'] == plan['native_source_commit']
            and physical['native_build_sha256'] == plan['native_build_sha256']
            and full['final_slot_consumed'] and full['batched_rows'] == CAPACITY - 1
            and full['generated_ids'] == 0 and full['native_exit_code'] == 0 and not full['fresh_faults'],
            'recover the separate same-source physical full-context diagnostic')
    for name in (THIS, window.device.THIS):
        p = bank.P / name
        require(p.read_bytes() == basis.validation.git('show', revision + ':' + name),
                'changed boundary reader: ' + name)
    fixtures = input_sequence(*old_requests[0])
    wire = [dict(read=len(ids), output=count, sha256=model.restore.signature_bytes(model.wire(ids, count)))
            for ids, count in fixtures]
    require([r['read'] for r in wire] == [82, BOUNDARY + 1, BOUNDARY, 82]
            and wire[0] == base['cases'][CASE]['wire'][0] and wire[3] == base['cases'][CASE]['wire'][2],
            'boundary requests changed the short controls')
    approved_case = dict(case, wire=wire, environment_sha256=bank.signature(env))
    approved_case.pop('budget')   # The actual ordinary allocator reports the fallback layout.
    plan.update(cases={CASE: approved_case}, limits=dict(plan['limits'], wall=1200, cpu=5000),
                input_context=CAPACITY, output_reserved=RESERVED,
                physical_reference_sha256=bank.signature(physical), copy_reference=COPY_RECORD,
                writer_reference_sha256=bank.signature(timing),
                boundary_readers={name: bank.sha(bank.P / name) for name in (THIS, window.device.THIS)},
                scope='Request the first input extent refused by the reserved-output guard, then the largest '
                      'legal GEN with one output ID. Select the proven writer with the mode16 private-bank32 policy. '
                      'Require ordinary fallback, state handoff and matching short controls. The standalone '
                      'all-slots result and live conversation RESTORE are separate gates.')
    return plan, env, fixtures, reference, control['outcome']


def inspect(log, plan, pid, answers, reference):
    proof = basis.validation.preload_evidence(log, pid)
    markers = list(re.finditer(r'^strata trace: request (\d+) 0$', log, re.M))
    require(proof['loader'] == plan['validation']['loader']['path']
            and [int(marker[1]) for marker in markers] == [82, BOUNDARY, 82]
            and not re.search(r'(?m)^strata mtp: .*loaded|UR_RESULT_ERROR_|ZE_RESULT_ERROR_|teardown uncertain', log),
            'server rejection reached model execution or runtime changed')
    require(len(answers) == 4 and answers[1]['rejected'] and answers[1]['prompt'] == BOUNDARY + 1,
            'missing checked context refusal')
    frames = [log[a.start():b.start()] for a, b in zip(markers, markers[1:])] + [log[markers[-1].start():]]
    middle = frames[1]
    rows = BOUNDARY - 1
    workspace = window.workspace
    fallback = workspace.one(middle, 'strata MoE long workspace: ')
    require(fallback['requested_window'] == '16' and fallback['relocated_active'] == '0'
            and fallback['effective_window'] == '10' and fallback['extra_bytes'] == '0'
            and fallback['admission'] == 'ordinary_fallback' and fallback['decode_changed'] == '0',
            'the long input must use the ordinary checked fallback')
    execution_window = workspace.one(middle, 'strata MoE window configuration: ')
    require(execution_window['requested_mode'] == '16'
            and execution_window['effective_mode'] in ('10', '2')
            and 0 < int(execution_window['maximum_rows']) <= 10 * window.store.CHUNK,
            'the fallback executor did not select a bounded ordinary window')
    layout = workspace.one(middle, 'strata prefill: layer-major candidate, ')
    gpu_rows = int(layout['residual_gpu_rows'])
    require(int(layout['tokens']) == rows and layout['residual_dtype'] == 'bf16'
            and layout['chunk'] == '8192' and 0 < gpu_rows < rows
            and int(layout['residual_gpu']) == gpu_rows * workspace.ROW
            and int(layout['residual_host']) == (rows - gpu_rows) * workspace.ROW,
            'unexpected BF16 device/host extents for fallback')
    bank_layout = workspace.one(middle, 'strata MoE expert bank: ')
    cache_bytes = int(bank_layout['device_bytes'])
    allocation_limit = int(bank_layout['allocation_limit_bytes'])
    require(bank_layout['requested_slots'] == str(extent.SLOTS) and bank_layout['effective_slots'] == '512'
            and bank_layout['staged'] == '0' and int(bank_layout['residual_gpu_rows']) == gpu_rows
            and cache_bytes == int(layout['layer_cache']) == 512 * workspace.STRIDE
            and max(cache_bytes, int(layout['residual_gpu'])) <= allocation_limit,
            'ordinary layer cache or single-allocation extent differs')
    store = workspace.one(middle, 'strata prefill attention bank: ')
    require(store['requested'] == '1' and int(store['direct']) == int(store['pieces']) > 0
            and store['requested_D2D_bytes'] == '0', 'the fallback attention writer did not execute')
    state = workspace.one(middle, 'strata serve: STATE_HASH ')
    windows = [(int(p), int(w)) for p, w in re.findall(r'^strata trace: window (\d+) (\d+)$', middle, re.M)]
    activation = 'strata DFlash2: eager, prompt_tail_rows=2046 cold_prefix=0 '
    require(windows == [(rows, 1)] and int(state['L']) == BOUNDARY
            and answers[2]['prefill_last'] == rows and answers[2]['generated'] == OUTPUT
            and middle.count(activation) == 1
            and middle.index(activation) < middle.index(f'strata trace: window {rows} 1\n'),
            'unverified final admitted server token or DFlash2 handoff')
    mask_marker = 'strata cache loan: phase=masked '
    ready_marker = 'strata cache loan: phase=ready '
    mask, ready = workspace.one(middle, mask_marker), workspace.one(middle, ready_marker)
    require(mask['decode_ready'] == '0' and ready['decode_ready'] == '1'
            and all(mask[key] == ready[key] for key in ('generation', 'chunk', 'first', 'lent'))
            and ready['chunk'] == '8192' and int(ready['first']) + int(ready['lent']) == 1256
            and middle.index(mask_marker) < middle.index(ready_marker) < middle.index(activation)
            and 'payload_match=1' in middle, 'decode cache did not return to the fixed placement')
    states = [window.split.committed_state(frames[i], answers[j]) for i, j in ((0, 0), (2, 3))]
    require(states == [reference['states'][0], reference['states'][2]]
            and [answers[0], answers[3]] == [reference['requests'][0], reference['requests'][2]],
            'out-of-limit refusal or fallback changed the short controls')
    kv = window.store.speed.KV.check_request(middle, 0, rows, window.store.CHUNK, reuse=True,
                                              tap_layers=window.store.gate.TAPPED_QSA_LAYERS)
    return dict(validation=proof, state=state, short_states=states, fallback=fallback,
                execution_window=execution_window, kv=kv, physical_prefix_rows=rows,
                device_residual_rows=gpu_rows, host_residual_rows=rows - gpu_rows,
                fallback_bank_bytes=cache_bytes, allocation_limit_bytes=allocation_limit, attention_store=store)


def save(row):
    window.store.gate.save(row, ROOT, KEY)


def prepare():
    require(not ROOT.exists() and KEY not in basis.validation.committed(bank.RESULT), 'fresh boundary plan required')
    revision = basis.validation.git('rev-parse', 'HEAD').decode().strip()
    plan, _, _, _, _ = inputs(revision)
    for name in ('cache', 'tmp'):
        (ROOT / CASE / name).mkdir(mode=0o700, parents=True)
    save(dict(plan=plan, stage='prepared', complete=False, passed=False, ownership_closed=True, active=False))


def run():
    row = bank.previous.read(ROOT / 'record.json')
    root = ROOT / CASE
    require(row == basis.validation.committed(bank.RESULT, (KEY,)) and row['stage'] == 'prepared'
            and row['ownership_closed'] and not row['active']
            and not (root / 'execution.json').exists() and not (root / 'owned').exists(),
            'require the committed unused boundary plan')
    plan, environment, fixtures, assets, reference = inputs(row['plan']['source_commit'])
    require(plan == row['plan'], 'frozen boundary inputs changed')

    def inspection(log, pid, answers):
        result = inspect(log, plan, pid, answers, reference)
        window.store.speed.verify_model_assets(assets, {'args': plan['cases'][CASE]['command'][2:]})
        return result

    outcome = window.device.run_case(plan, root, CASE, environment, fixtures, inspection,
                                     request_reader=request, limits=plan['limits'])
    passed = bool(outcome['passed'])
    row.update(outcome=outcome, stage='closed' if passed else 'failed', passed=passed, complete=passed,
               ownership_closed=outcome['ownership_closed'], active=not outcome['ownership_closed'])
    save(row)
    print({k: outcome.get(k) for k in ('passed', 'native_exit_code', 'fresh_faults', 'error')}, flush=True)
    require(passed, outcome.get('error') or 'DFlash2 serving boundary validation failed')


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(sys.executable == basis.reference.model.PYTHON and action in ('prepare', 'run'), 'use prepare or run')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        {'prepare': prepare, 'run': run}[action]()
