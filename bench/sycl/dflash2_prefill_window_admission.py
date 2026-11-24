"""Check the 120,001-token DFlash2 window at a fixed 128-slot expert bank."""

import fcntl
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import dflash2_prefill_direct_store as store

bank, basis, require = store.bank, store.basis, store.require
device, split, model = store.gate.device, store.split, store.model
workspace = split.workspace

KEY = 'dflash2_prefill_window_admission_v1'
ROOT = bank.B / KEY.replace('_', '-')
THIS = str(Path(__file__).resolve().relative_to(bank.P))
ROWS, SLOTS, HEADROOM = 120000, 128, 512 << 20
READ = ROWS + 1
ARMS = {'window10': 10, 'window16': 16}
MODE = 'STRATA_SYCL_PREFILL_MOE_WINDOW'
RELOCATE = 'STRATA_SYCL_PREFILL_MOE_RELOCATE_JOB_BUFFERS'
UPLOAD = store.speed.generation.v2.UPLOAD
HEAD_BYTES = 521472000
HEAD_DETACH = (f'strata prefill heads: target detached bytes={HEAD_BYTES} source=owned-RAM '
               f'D2H={HEAD_BYTES} H2D-on-restore={HEAD_BYTES}')


def budget(mode):
    split_active = mode == 16
    maximum = ROWS if split_active else 81920
    scratch = workspace.workspace(maximum, split=split_active, relocated=split_active)
    prefix = ((ROWS + 3) // 4 * 4) * 2056 + 256
    allocations = dict(bank=SLOTS * workspace.STRIDE, residual=ROWS * workspace.ROW,
                       prefix_and_status=prefix, relocated_owner=scratch['extra'])
    return dict(maximum_rows=maximum, windows=48 * ((ROWS + maximum - 1) // maximum),
                extra_bytes=scratch['extra'], tail_bytes=scratch['tail_payload'],
                incremental_allocations=allocations, reserve_bytes=HEADROOM,
                fixed_with_headroom=sum(allocations.values()) + HEADROOM,
                shared_arena=scratch['region'] + scratch['mixed_tail'],
                largest_allocation=max(allocations.values()))


def inputs(revision):
    source, launches, reference = store.model_inputs(revision)
    admission = basis.validation.committed(bank.RESULT, (store.KEY,))
    same_launch = ('binary', 'native_source_commit', 'native_build_sha256', 'boot', 'driver',
                   'runtime_manifest', 'validation', 'drafter_assets', 'cases', 'limits',
                   'split_boundary_reference', 'component_sha256')
    require(admission['complete'] and admission['passed'] and admission['ownership_closed']
            and admission['stage'] == 'closed' and not admission['active']
            and all(admission['plan'][key] == source[key] for key in same_launch),
            'recover the closed DFlash2 split and its native launch')
    copy = source['cases']['copy']
    original, original_requests = launches['copy']
    require(original.get(MODE) == '10' and original.get(RELOCATE, '0') == '0'
            and original.get('STRATA_SYCL_PREFILL_MOE_SPLIT_ACCUMULATOR') == '1'
            and original.get('STRATA_SYCL_PREFILL_EXPERT_BANK_SLOTS') == str(SLOTS)
            and original.get(store.SELECTOR) == '0' and original.get(UPLOAD) == 'batched'
            and original.get('STRATA_PREFILL_RELEASE_HEADS') == '1'
            and original.get('STRATA_PREFILL_HEAD_HOST_BACKING', '0') == '0'
            and original.get('STRATA_SYCL_DECODE_Q6_DPAS_HEAD') == '1'
            and copy['command'][copy['command'].index('--draft') + 1] == 'dflash2'
            and '--mtp' not in copy['command'], 'unexpected inherited head, cache, draft or prefill configuration')
    short, output = original_requests[0]
    long = [i % 4096 for i in range(READ - len(short))] + short
    requests = [(short, output), (long, output), (short, output)]
    wire = [dict(read=len(ids), output=count, sha256=model.restore.signature_bytes(model.wire(ids, count)))
            for ids, count in requests]

    cases, envs = {}, {}
    for arm, mode in ARMS.items():
        root = ROOT / arm
        env = {k: v.replace(str(store.ROOT / 'copy'), str(root)) for k, v in original.items()}
        env.update({MODE: str(mode), UPLOAD: 'demand'})
        cases[arm] = dict(command=copy['command'], environment_sha256=bank.signature(env),
                          wire=wire, budget=budget(mode))
        envs[arm] = env
    normalized = [{k: v.replace(str(ROOT / arm), '<case>') for k, v in env.items() if k != MODE}
                  for arm, env in envs.items()]
    require(normalized[0] == normalized[1], 'the only independent launch input is the window mode')
    workspace_source = str(Path(workspace.__file__).resolve().relative_to(bank.P))
    reader_sources = {}
    for path in (THIS, split.THIS, workspace_source):
        expected = basis.validation.git('show', revision + ':' + path)
        require((bank.P / path).read_bytes() == expected, 'changed window-admission reader: ' + path)
        reader_sources[path] = bank.sha(bank.P / path)
    plan = {key: source[key] for key in ('native_source_commit', 'native_build_sha256', 'binary',
                                        'driver', 'boot', 'runtime_manifest', 'validation',
                                        'drafter_assets', 'limits')}
    plan.update(source_commit=revision, inputs=dict(launch=source['inputs'], readers=reader_sources),
                preceding_admission_sha256=bank.signature(admission), cases=cases, order=list(ARMS),
                scope='Check actual 120001-input DFlash2 window admission in independent traced processes. '
                      'Hold expert bank128, H2D demand policy, copied attention, core placement, canonical '
                      'released target head and KV retention fixed. Require all-GPU BF16 rows, 96/48 '
                      'windows, checked retirement and unaltered short reset; record long fingerprints. '
                      'This is capacity and state-handoff evidence. Use separate clean process repetitions '
                      'and appropriate numerical/task checks before making speed or adoption decisions.')
    return plan, envs, requests, reference


def window_result(frame, mode):
    predicted = budget(mode)
    maximum = predicted['maximum_rows']
    wide = mode == 16
    admission = workspace.one(frame, 'strata MoE accumulator admission: ')
    split_owner = workspace.one(frame, 'strata MoE split admission: ')
    bank_owner = workspace.one(frame, 'strata MoE expert bank: ')
    work = workspace.one(frame, 'strata MoE accumulator window: ')
    uploads = workspace.one(frame, 'strata MoE expert bank complete: ')
    window = workspace.one(frame, 'strata MoE window configuration: ')
    free, arena, limit = (int(admission['free_gpu_bytes']), int(admission['shared_arena_bytes']),
                         int(split_owner['max_single_allocation_bytes']))
    require(int(admission['rows']) == int(window['maximum_rows']) == int(work['maximum_rows']) == maximum
            and admission['requested_mode'] == window['requested_mode'] == window['effective_mode'] == str(mode)
            and int(admission['gpu_prefix_rows']) == int(bank_owner['residual_gpu_rows']) == ROWS
            and int(admission['fixed_including_headroom_bytes']) == predicted['fixed_with_headroom'] <= free
            and predicted['shared_arena'] <= arena and predicted['largest_allocation'] <= limit
            and int(work['windows']) == predicted['windows'] and work['GPU_resident_windows_only'] == '1'
            and bank_owner['requested_slots'] == bank_owner['effective_slots'] == str(SLOTS)
            and bank_owner['staged'] == '1' and int(bank_owner['device_bytes']) == SLOTS * workspace.STRIDE
            and bank_owner['general_map'] == 'private_scratch_never_published',
            'window request did not admit the expected bank, memory bounds, or complete GPU windows')
    require(int(split_owner['rows']) == ROWS and split_owner['active'] == str(int(wide))
            and int(split_owner['extra_allocation_bytes']) == predicted['extra_bytes']
            and int(split_owner['tail_payload_bytes']) == predicted['tail_bytes']
            and split_owner['decode_changed'] == '0', 'unexpected split owner')
    long_work = workspace.records(frame, 'strata MoE long workspace complete: ')
    split_work = workspace.records(frame, 'strata MoE split complete: ')
    require(len(long_work) == len(split_work) == int(wide), 'incorrect split/relocated completion count')
    if wide:
        require(long_work[0] == dict(rows=str(ROWS), extra_bytes=str(predicted['extra_bytes']),
                    guards='3_checked', final_drain='checked', decode_changed='0')
                and split_work[0] == dict(rows=str(ROWS), primary_rows='65536', tail_bytes=str(predicted['tail_bytes']),
                    guard='checked', final_drain='checked', decode_changed='0'), 'split/relocated owner did not retire')
    attention = workspace.one(frame, 'strata prefill attention bank: ')
    pieces, copy_bytes = model.restore.capacity.expected_store(ROWS, ROWS, maximum)
    head = frame.find(HEAD_DETACH + '\n')
    require(attention == dict(requested='0', pieces=str(pieces), direct='0', requested_D2D_bytes=str(copy_bytes))
            and 0 <= frame.find('strata prefill heads: draft_reader=absent-dflash2') < head
            < frame.index('strata MoE accumulator admission: '),
            'attention writer or head retirement differs from the fixed policy')
    require('strata MoE split expert upload: ' not in frame and uploads['final_drain'] == 'checked'
            and int(uploads['uploads']) + int(uploads['permanent_hits']) == int(work['dequants']) > 0,
            'demand upload or retirement differs')
    return dict(free_bytes=free, arena_bytes=arena, allocation_limit=limit, logical_copy_bytes=copy_bytes,
                fixed_with_headroom=predicted['fixed_with_headroom'],
                free_bytes_beyond_reserve=free - predicted['fixed_with_headroom'],
                requested_bank_H2D_bytes=int(uploads['requested_H2D_bytes']), uploads=int(uploads['uploads']),
                jobs=int(work['jobs']), dequants=int(work['dequants']), windows=int(work['windows']),
                maximum_rows=maximum)


def inspect(log, plan, pid, arm, responses):
    proof = basis.validation.preload_evidence(log, pid)
    markers = list(re.finditer(r'^strata trace: request (\d+) 0$', log, re.M))
    require(proof['loader'] == plan['validation']['loader']['path']
            and log.count('strata native head DPAS: ready=1 columns=4 layout=canonical') == 1
            and not re.search(r'(?m)^strata mtp: .*loaded|UR_RESULT_ERROR_|ZE_RESULT_ERROR_|teardown uncertain', log)
            and [int(m[1]) for m in markers] == [82, READ, 82], 'invalid owner, runtime validation or request extent')
    starts = [m.start() for m in markers] + [len(log)]
    frames = [log[a:b] for a, b in zip(starts, starts[1:])]
    middle = frames[1]
    require(log.count('strata prefill heads: draft_reader=absent-dflash2') == 1
            and log.count('strata prefill heads: target detached ') == 1
            and 'strata prefill heads: draft detached ' not in log,
            'only the long request may detach the target head')
    observed = window_result(middle, ARMS[arm])
    kv = store.speed.KV.check_request(middle, 0, ROWS, store.CHUNK, reuse=True,
                                      tap_layers=store.gate.TAPPED_QSA_LAYERS)
    masked = workspace.records(middle, 'strata cache loan: phase=masked ')
    ready = workspace.records(middle, 'strata cache loan: phase=ready ')
    taps = [int(t) for t in re.findall(r'^strata DFlash2: eager, prompt_tail_rows=(\d+) cold_prefix=0 vram_bytes=[1-9]\d*$', log, re.M)]
    states = [split.committed_state(frame, answer) for frame, answer in zip(frames, responses, strict=True)]
    require(len(masked) == len(ready) == 1 and masked[0]['decode_ready'] == '0' and ready[0]['decode_ready'] == '1'
            and all(masked[0][k] == ready[0][k] for k in ('generation', 'chunk', 'first', 'lent'))
            and ready[0]['chunk'] == '8192' and ready[0]['first'] == '132' and ready[0]['lent'] == '1124'
            and 'payload_match=1' in middle
            and middle.index('strata cache loan: phase=masked ') < middle.index('strata cache loan: phase=ready ')
                < middle.index('strata DFlash2: eager, ')
            and taps == [81, 2046, 81] and responses[0] == responses[2] and states[0] == states[2]
            and all(a['offered'] > 0 for a in responses), 'target/draft handoff or short reset differs')
    return dict(validation=proof, route=observed, states=states, kv_stage=kv, tap_rows=taps, loan=ready[0])


def save(row):
    store.gate.save(row, ROOT, KEY)


def main(action):
    if action == 'prepare':
        require(not ROOT.exists() and KEY not in basis.validation.committed(bank.RESULT), 'fresh window plan needed')
        plan, _, _, _ = inputs(basis.validation.git('rev-parse', 'HEAD').decode().strip())
        for arm in ARMS:
            for name in ('cache', 'tmp'):
                (ROOT / arm / name).mkdir(mode=0o700, parents=True)
        save(dict(plan=plan, stage='prepared', complete=False, passed=False, active=False,
                  ownership_closed=True, outcomes={}))
        return

    row = bank.previous.read(ROOT / 'record.json')
    order = list(ARMS)
    earlier = order[:order.index(action)]
    root = ROOT / action
    require(row == basis.validation.committed(bank.RESULT, (KEY,)) and row['ownership_closed'] and not row['active']
            and list(row['outcomes']) == earlier and all(row['outcomes'][old]['passed'] for old in earlier)
            and row['stage'] == ('partial' if earlier else 'prepared')
            and not (root / 'owned').exists() and not (root / 'execution.json').exists(),
            'require the committed window plan and closed previous process')
    plan, envs, requests, reference = inputs(row['plan']['source_commit'])
    require(plan == row['plan'], 'window-admission inputs changed')

    def inspect_case(log, pid, responses):
        result = inspect(log, plan, pid, action, responses)
        store.speed.verify_model_assets(reference, {'args': plan['cases'][action]['command'][2:]})
        return result

    observed = device.run_case(plan, root, action, envs[action], requests, inspect_case)
    row['outcomes'][action] = observed
    if action == order[-1] and observed['passed']:
        control = row['outcomes'][order[0]]
        wide_budget = plan['cases'][action]['budget']['fixed_with_headroom']
        row['comparison'] = dict(independent_native_processes=control['native_identity'] != observed['native_identity'],
            wide_budget_slack_at_control_free=control['route']['free_bytes'] - wide_budget,
            actual_wide_budget_slack=observed['route']['free_bytes_beyond_reserve'],
            short_state_and_outputs_match=(control['states'][0], control['requests'][0])
                                          == (observed['states'][0], observed['requests'][0]),
            long_state_fingerprint_matches=control['states'][1] == observed['states'][1],
            long_generated_work_matches=control['requests'][1] == observed['requests'][1],
            kv_stage_matches=control['kv_stage'] == observed['kv_stage'],
            requested_bank_H2D_bytes_removed=control['route']['requested_bank_H2D_bytes']
                                             - observed['route']['requested_bank_H2D_bytes'])
        comparison = row['comparison']
        if not all(comparison[k] for k in ('independent_native_processes', 'short_state_and_outputs_match',
                                           'kv_stage_matches')):
            observed.update(passed=False, error='native ownership, short decode or retained KV differs')
    complete = action == order[-1] and observed['passed']
    row.update(stage=('closed' if complete else 'partial') if observed['passed'] else 'failed',
               complete=complete, passed=complete, active=not observed['ownership_closed'],
               ownership_closed=observed['ownership_closed'])
    save(row)
    print({k: observed.get(k) for k in ('passed', 'native_exit_code', 'fresh_faults', 'error')}, flush=True)
    require(observed['passed'], observed.get('error', 'window admission failed'))


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(sys.executable == basis.reference.model.PYTHON and action in ('prepare', *ARMS),
            'use the pinned Python with prepare, window10 or window16')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        main(action)
