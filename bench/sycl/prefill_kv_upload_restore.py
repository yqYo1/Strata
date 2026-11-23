"""Qualify conversation restore and the first decode token on the measured V2 engine."""

import fcntl
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
import prefill_kv_upload_compare as upload

from engine_once import native_environment
import prefill_kv_gather_restore as state

bank, basis, require = upload.bank, upload.basis, upload.require
restore = state.restore
workspace = upload.previous.workspace
KEY = 'prefill_kv_upload_restore_v2'
ROOT = bank.B / KEY.replace('_', '-')
THIS = str(Path(__file__).resolve().relative_to(bank.P))
PHYSICAL = '6d5debd9cbcb9011d0fc7aa7374ec1700d9e6ccd'
FIRST_RESTORE = '74ad105368ed894fecd2b3e06d76509547544268'
FIRST_KEY = 'prefill_kv_upload_restore_v1'
CONTEXT, CHUNK, SLOTS = 262144, 8192, 128
CASES = {'split-demand': 'demand', 'split-batched': 'batched'}
ORDER = ('host', *CASES)


def required_basis():
    native = bank.previous.read(upload.build.ROOT / 'record.json')
    upload.build.context().check(native)
    key = upload.KEYS['time']
    previous = bank.previous.historic(upload.UPLOAD_TIMING, key)
    speed = bank.previous.read(upload.experiment('time').ROOT / 'record.json')
    decision = previous['decision']
    require(upload.gather.closed(previous) and basis.compact(speed) == previous
            and speed['binary'] == native['binaries']['strata']
            and speed['native_source_commit'] == native['native_source_commit']
            and decision['repeated_warm_gain']['batched/demand']
            and decision['repeated_1000']['batched'],
            'require the same executable and the completed three-pair batched-upload comparison')

    physical = bank.previous.historic(PHYSICAL, 'prefill_kv_upload_capacity_v1')
    full = physical['outcomes']['full-copy']
    require(upload.gather.closed(physical)
            and physical['binary'] == native['binaries']['strata_full_context_diagnostic']
            and physical['configuration'] == speed['configuration']
            and physical['native_source_commit'] == native['native_source_commit']
            and full['batched_rows'] == CONTEXT - 1 and full['final_slot_consumed']
            and full['generated_ids'] == 0 and full['passed'] and not full['fresh_faults'],
            'require the same-build copied-output physical-context diagnostic')

    manifest = speed['runtime_manifest']
    providers = bank.previous.read(Path(manifest['path']))
    require(speed['boot'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            and speed['driver'] == native['driver'] == bank.builder.guards.driver_guard()
            and bank.sha(manifest['path']) == manifest['sha256'] and len(providers) == manifest['files']
            and all(bank.sha(p) == h for p, h in providers.items()), 'the measured runtime changed')
    basis.comparison.shared.models.model_pins(speed['configuration'], speed['model_assets'])

    first = bank.previous.historic(FIRST_RESTORE, FIRST_KEY)
    require(first == basis.validation.committed(bank.RESULT, (FIRST_KEY,))
            and first['stage'] == 'failed' and first['ownership_closed'] and not first['active']
            and first['plan']['binary'] == native['binaries']['strata']
            and first['plan']['runtime_manifest'] == manifest and first['plan']['boot'] == speed['boot']
            and first['plan']['driver'] == speed['driver']
            and all(first['outcomes'][k]['passed'] for k in ('host', 'ordinary', 'near-fallback'))
            and first['outcomes']['split-demand']['error'] == "RuntimeError('split expert bank did not retire')"
            and first['outcomes']['split-demand']['native_exit_code'] == 0
            and not first['outcomes']['split-demand']['fresh_faults'],
            'require the original V2 ordinary and fallback controls and preserved failed split reader')
    return native, speed, physical, first


def request_inputs():
    prefix, suffix = 172012, 90116
    initial = [i % 4096 for i in range(prefix)]
    return [(initial, 0), (list(range(128, 161)), 0),
            (initial + [256 + i % 16 for i in range(suffix)], prefix)]


def environment(speed, case):
    ctx = upload.experiment('time')
    env = ctx.environment(speed, dict(root=case['root'], index=upload.MODES.index(case['mode'])))
    env = native_environment(speed, env)
    env.update(STRATA_TRACE='1', STRATA_STATE_HASH='1', STRATA_SNAPSHOT_VERIFY='1',
               STRATA_SYCL_CACHE_LOAN_AUDIT='1', TMPDIR=case['root'] + '/tmp',
               XDG_CACHE_HOME=case['root'] + '/cache')
    require(env['STRATA_SYCL_PREFILL_EXPERT_BANK_SLOTS'] == str(SLOTS)
            and env[upload.previous.UPLOAD] == case['mode']
            and env['STRATA_SYCL_PREFILL_MOE_SPLIT_ACCUMULATOR'] == '1'
            and env['STRATA_SYCL_CACHE_SCRATCH_LEASE'] == env['STRATA_SYCL_PREFILL_SAME_QUEUE'] == '1'
            and env['UR_ENABLE_LAYERS'] == 'UR_LAYER_PARAMETER_VALIDATION'
            and 'UR_LOG_TRACING' not in env, 'require the qualified queue and diagnostic scope')
    return env


def failed_stderr(first):
    return Path(first['plan']['cases']['split-demand']['root']) / 'owned' / 'split-demand.stderr'


def make_plan(revision, native, speed, physical, first):
    pins = upload.experiment('time').pins(revision)
    pins.update(bank.stage.source.pin(bank.P, [THIS])['source_sha256'])
    cases = {}
    for name, mode in CASES.items():
        inputs = request_inputs()
        requests = restore.input_plan(False, inputs=inputs)
        require(all(r['prompt'] + r['max_new'] + 8 <= CONTEXT for r in requests),
                'restore sequence must leave the native server room to generate')
        root = ROOT / name
        case = dict(root=str(root), near=True, direct=False, mode=mode,
                    command=restore.argv(dict(speed, binary=native['binaries']['strata'])),
                    requests=requests, wall=1800)
        previous = first['plan']['cases'][name]
        require(all(case[k] == previous[k] for k in ('requests', 'command', 'mode', 'wall'))
                and bank.signature(environment(speed, previous)) == previous['environment_sha256'],
                'the new reader must retain the original workload and launch')
        case['environment_sha256'] = bank.signature(environment(speed, case))
        cases[name] = case
    a, b = cases['split-demand'], cases['split-batched']
    normalize = lambda case: {k: v.replace(case['root'], '<case>') for k, v in environment(speed, case).items()
                              if k != upload.previous.UPLOAD}
    require(a['requests'] == b['requests'] and a['command'] == b['command'] and normalize(a) == normalize(b),
            'split restore arms must differ by scheduling alone')
    return dict(source_commit=revision, source_inputs=dict(files=len(pins), sha256=bank.signature(pins)),
                native_source_commit=native['native_source_commit'], binary=native['binaries']['strata'],
                serial_order=list(ORDER), cases=cases, budgets=restore.capacity.BUDGET,
                speed_reference=upload.UPLOAD_TIMING, capacity_reference=PHYSICAL, previous_restore=FIRST_RESTORE,
                failed_stderr_sha256=bank.sha(failed_stderr(first)),
                physical_receipt_sha256=bank.signature(physical), runtime_manifest=speed['runtime_manifest'],
                model_assets_sha256=bank.signature(speed['model_assets']),
                boot=speed['boot'], driver=speed['driver'],
                scope='Instrumented native pipe GEN, parked live RESTORE, exact state and first verifier handoff.',
                acceptance='Check the original split stderr reader on the host, then run a fresh pair. Match '
                           'state, work, snapshot and expert-cache bytes. Require bounded permanent mappings, '
                           'upload and dequant conservation, checked scratch and bank retirement, and 48 '
                           'batched upload windows with batching.',
                stop='Stop at the first input, path, state, output, runtime, ownership, fault or budget failure.')


def split_evidence(frame, request, *, split, batched):
    admission = workspace.one(frame, 'strata MoE split admission: ')
    rows = request['read'] - 1
    require((int(admission['active']), int(admission['rows'])) == (int(split), rows)
            and admission['decode_changed'] == '0', 'split placement differs from the frozen request')
    done = workspace.records(frame, 'strata MoE split complete: ')
    uploads = workspace.records(frame, 'strata MoE split expert upload: ')
    if not split:
        require(not done and not uploads, 'fallback unexpectedly ran split upload')
        return dict(split=False)
    completion, = done
    store = request['attention_store']
    require(int(admission['prefill_slots']) == SLOTS
            and int(admission['max_single_allocation_bytes']) >= rows * restore.capacity.ROW_BYTES
            and store['gpu_rows'] == store['maximum_rows'] == rows and store['host_rows'] == 0
            and int(completion['rows']) == rows and completion['guard'] == 'checked'
            and completion['final_drain'] == 'checked' and completion['decode_changed'] == '0',
            'incomplete all-GPU split or checked scratch retirement')
    bank_end = workspace.one(frame, 'strata MoE expert bank complete: ')
    experts = int(bank_end['uploads'])
    permanent, total = int(bank_end['permanent_slots']), int(bank_end['total_slots'])
    work = workspace.one(frame, 'strata MoE accumulator window: ')
    h2d_bytes = int(bank_end['requested_H2D_bytes'])
    require(experts > 0 and h2d_bytes > 0 and int(bank_end['permanent_hits']) == 0
            and experts == int(work['dequants']) and 0 <= permanent < total == SLOTS
            and bank_end['final_drain'] == 'checked', 'split expert bank work or retirement differs')
    if batched:
        record, = uploads
        require(permanent == 0 and (int(record['windows']), int(record['experts']), int(record['copy_commands']))
                == (48, experts, experts) and record['final_drain'] == 'checked',
                'missing one or more checked batched upload windows')
    else:
        require(not uploads, 'demand unexpectedly ran batched upload')
    return dict(split=True, slots=SLOTS, uploads=experts, permanent_slots=permanent,
                requested_h2d_bytes=h2d_bytes, batched_windows=48 if batched else 0)


def inspect(log, speed, case, name, requests, pid, outcomes):
    outcome = restore.inspect(log, speed, case, requests, pid)
    frames = re.split(r'^strata serve: STATE_HASH L=.*$', log, flags=re.M)
    traces = state.check_frames(log, requests, reuse=True, vec8=False)
    require([t['request'] for t in traces] == [0, 2], 'expected two long reads and one short cache-window switch')
    for t in traces:
        i = t['request']
        request = requests[i]
        t['input'] = state.model.input_receipt(frames[i], request['read'] - 1, request['attention_store'], False)
        t['split'] = split_evidence(frames[i], request, split=i == 2,
                                    batched=case['mode'] == 'batched')
    if name == 'split-batched':
        control = outcomes['split-demand']
        restore.compare(outcome, control)
        for actual, prior in zip(traces, control['dispatch'], strict=True):
            mode_fields = ('batched_windows', 'permanent_slots')
            actual_split = {k: v for k, v in actual['split'].items() if k not in mode_fields}
            prior_split = {k: v for k, v in prior['split'].items() if k not in mode_fields}
            require(actual_split == prior_split and all(actual[k] == prior[k] for k in ('kv', 'gather', 'input')),
                    'actual gathered rows, KV, or split upload work changed')
    outcome['dispatch'] = traces
    return outcome


def reader_control(first):
    log = failed_stderr(first).read_text()
    frames = re.split(r'^strata serve: STATE_HASH L=.*$', log, flags=re.M)
    require(len(frames) == 4, 'require the original completed request frames')
    frame, request = frames[2], first['outcomes']['split-demand']['requests'][2]
    observed = split_evidence(frame, request, split=True, batched=False)
    require(observed['permanent_slots'] == 127 and observed['uploads'] == 19993,
            'unexpected original demand-bank counters')
    marker = 'permanent_hits=0 permanent_slots=127 total_slots=128 final_drain=checked'
    require(frame.count(marker) == 1, 'unexpected bank retirement observation')
    bad = (
        marker.replace('permanent_slots=127', 'permanent_slots=128'),
        marker.replace('permanent_hits=0', 'permanent_hits=1'),
        marker.replace('final_drain=checked', 'final_drain=unverified'),
    )
    for replacement in bad:
        try:
            split_evidence(frame.replace(marker, replacement), request, split=True, batched=False)
        except RuntimeError:
            continue
        raise RuntimeError('accepted an invalid bank counter or drain')
    return dict(accepted_demand_slots=127, invalid_bank_receipts_rejected=len(bad))


def save(row):
    bank.stage.write(ROOT / 'record.json', row)
    result = bank.previous.read(bank.RESULT)
    result[KEY] = row
    bank.stage.write(bank.RESULT, result)


def prepare():
    require(not ROOT.exists() and KEY not in basis.validation.committed(bank.RESULT), 'inspect an existing trial first')
    revision = basis.validation.git('rev-parse', 'HEAD').decode().strip()
    plan = make_plan(revision, *required_basis())
    ROOT.mkdir(mode=0o700)
    save(dict(plan=plan, outcomes={}, stage='prepared', active=False, ownership_closed=True))


def run(name):
    row = bank.previous.read(ROOT / 'record.json')
    require(row == basis.validation.committed(bank.RESULT, (KEY,))
            and row['stage'] in ('prepared', 'running') and row['ownership_closed'] and not row['active']
            and list(row['outcomes']) == list(ORDER[:ORDER.index(name)]),
            'commit the frozen inputs and verify completion of the preceding owner')
    plan = row['plan']
    native, speed, physical, first = required_basis()
    require(plan == make_plan(plan['source_commit'], native, speed, physical, first), 'frozen inputs differ')
    root = ROOT / name
    require(not root.exists(), 'inspect the existing case owner before running again')
    for folder in ('tmp', 'cache'):
        (root / folder).mkdir(mode=0o700, parents=True)
    cursor, = (s.removeprefix('-- cursor: ') for s in subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'],
        text=True, timeout=15).splitlines() if s.startswith('-- cursor: '))
    outcome = dict(complete=False, host_probe_accepted=False)
    owner = restore.owner_module.Owner(root / 'owned',
        lambda: bank.stage.write(root / 'execution.json', dict(active=bool(owner.active), commands=owner.commands)))
    try:
        vocab = json.loads((Path(speed['configuration']['tokenizer']) / 'vocab.json').read_bytes())
        require(set(range(4096)) <= set(vocab.values()), 'generated fixture IDs are outside the pinned vocabulary')
        vocab_size = max(vocab.values()) + 1
        if name == 'host':
            peer = restore.host_peer()
            case = plan['cases']['split-demand']
            argv = [sys.executable, '-B', '-c', peer, json.dumps(case['requests'])]
            entry, _, _ = owner.run('split-wire', argv, dict(PATH='/usr/bin:/bin', LANG='C.UTF-8'), bank.P,
                interaction=lambda io, _: restore.interact(io, case, vocab_size, native=False, inputs=request_inputs()),
                wall=60, cpu=40, rss_cap=512 << 20, text_cap=128 << 10, file_cap=1 << 20, total_cap=4 << 20)
            require(restore.owner_module.closed(entry) and entry['exit_code'] == 0, 'host peer failed')
            outcome.update(host_cases=['split-demand'], reader_control=reader_control(first), gpu_executed=False)
        else:
            case = plan['cases'][name]
            requests = []

            def exchange(io, _):
                requests.extend(restore.interact(io, case, vocab_size, native=True, inputs=request_inputs()))

            print(f'Native restore case {name} starts', flush=True)
            entry, _, stderr = owner.run(name, case['command'], environment(speed, case), bank.P,
                interaction=exchange, wall=case['wall'], **plan['budgets'])
            outcome.update(native_exit_code=entry['exit_code'], gpu_executed=True, requests=requests)
            require(restore.owner_module.closed(entry) and entry['exit_code'] == 0, 'native server or teardown failed')
            outcome.update(inspect(stderr.read_text(), speed, case, name, requests, entry['pid'], row['outcomes']))
            if name == ORDER[-1]:
                basis.comparison.shared.models.model_pins(speed['configuration'], speed['model_assets'], full=True)
        require(plan == make_plan(plan['source_commit'], native, speed, physical, first), 'inputs changed during run')
        outcome.update(complete=True, host_probe_accepted=True)
    except Exception as error:
        outcome['error'] = repr(error)
    finally:
        basis.validation.finish(outcome, owner, cursor, plan['boot'])
        outcome['passed'] = bool(outcome['passed'] and outcome['driver_after'] == plan['driver'])
        row['outcomes'][name] = outcome
        done = outcome['passed'] and name == ORDER[-1]
        row.update(stage='closed' if done else 'running' if outcome['passed'] else 'failed',
                   passed=done, complete=done, active=bool(owner.active), ownership_closed=outcome['ownership_closed'])
        save(row)
    print(json.dumps({k: outcome.get(k) for k in ('passed', 'restored_tokens', 'native_exit_code', 'fresh_faults', 'error')}),
          flush=True)
    require(outcome['passed'], outcome.get('error', 'restore or ownership check failed'))


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(sys.executable == basis.reference.model.PYTHON and action in ('prepare', *ORDER),
            'use the pinned benchmark interpreter and one planned action')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        prepare() if action == 'prepare' else run(action)
