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
KEY = 'prefill_kv_upload_restore_v1'
ROOT = bank.B / KEY.replace('_', '-')
THIS = str(Path(__file__).resolve().relative_to(bank.P))
PHYSICAL = '6d5debd9cbcb9011d0fc7aa7374ec1700d9e6ccd'
OLDER_RESTORE = 'b633f7d694cec941c0df25c0fb9edea650d41ca3'
CONTEXT, CHUNK, SLOTS = 262144, 8192, 128
CASES = {
    'ordinary': (32769, 8209, 'batched'),
    'near-fallback': (253919, 8209, 'batched'),
    'split-demand': (172012, 90116, 'demand'),
    'split-batched': (172012, 90116, 'batched'),
}
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

    older = bank.previous.historic(OLDER_RESTORE, state.KEY)
    require(upload.gather.closed(older)
            and all(older['outcomes'][k]['passed'] for k in ('short-reuse', 'near-reuse')),
            'require the earlier ordinary and near-capacity copied-output controls')
    return native, speed, physical, older


def request_inputs(name):
    prefix, suffix, _ = CASES[name]
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


def make_plan(revision, native, speed, physical, older):
    pins = upload.experiment('time').pins(revision)
    pins.update(bank.stage.source.pin(bank.P, [THIS])['source_sha256'])
    cases = {}
    for name, (_, _, mode) in CASES.items():
        inputs = request_inputs(name)
        requests = restore.input_plan(False, inputs=inputs)
        require(all(r['prompt'] + r['max_new'] + 8 <= CONTEXT for r in requests),
                'restore sequence must leave the native server room to generate')
        if name in ('ordinary', 'near-fallback'):
            ref = 'short-reuse' if name == 'ordinary' else 'near-reuse'
            historic, = (p for p in older['plans'] if p['name'] == ref)
            require(requests == historic['requests'], 'ordinary or near-fallback wire inputs changed')
        root = ROOT / name
        case = dict(root=str(root), near=name != 'ordinary', direct=False, mode=mode,
                    command=restore.argv(dict(speed, binary=native['binaries']['strata'])),
                    requests=requests, wall=1200 if name == 'ordinary' else 1800)
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
                speed_reference=upload.UPLOAD_TIMING, capacity_reference=PHYSICAL, previous_restore=OLDER_RESTORE,
                physical_receipt_sha256=bank.signature(physical), runtime_manifest=speed['runtime_manifest'],
                model_assets_sha256=bank.signature(speed['model_assets']),
                boot=speed['boot'], driver=speed['driver'],
                scope='Instrumented native pipe GEN, parked live RESTORE, exact state and first verifier handoff.',
                acceptance='First run a host peer for each geometry. Check exact input and state extents, '
                           'snapshot and expert-cache bytes. Require matched original ordinary state/output. '
                           'The aligned 90115-row split pair must match state and work and exercise 48 '
                           'checked upload windows only with batching.',
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
    require(experts > 0 and int(bank_end['permanent_hits']) == int(bank_end['permanent_slots']) == 0
            and bank_end['final_drain'] == 'checked', 'split expert bank did not retire')
    if batched:
        record, = uploads
        require((int(record['windows']), int(record['experts']), int(record['copy_commands']))
                == (48, experts, experts) and record['final_drain'] == 'checked',
                'missing one or more checked batched upload windows')
    else:
        require(not uploads, 'demand unexpectedly ran batched upload')
    return dict(split=True, slots=SLOTS, uploads=experts,
                requested_h2d_bytes=int(bank_end['requested_H2D_bytes']), batched_windows=48 if batched else 0)


def inspect(log, speed, case, name, requests, pid, older, outcomes):
    outcome = restore.inspect(log, speed, case, requests, pid)
    frames = re.split(r'^strata serve: STATE_HASH L=.*$', log, flags=re.M)
    traces = state.check_frames(log, requests, reuse=True, vec8=False)
    require([t['request'] for t in traces] == [0, 2], 'expected two long reads and one short cache-window switch')
    for t in traces:
        i = t['request']
        request = requests[i]
        t['input'] = state.model.input_receipt(frames[i], request['read'] - 1, request['attention_store'], False)
        t['split'] = split_evidence(frames[i], request, split=i == 2 and name.startswith('split-'),
                                    batched=case['mode'] == 'batched')
    if name in ('ordinary', 'near-fallback'):
        prior = older['outcomes']['short-reuse' if name == 'ordinary' else 'near-reuse']
        restore.compare(outcome, prior)
    elif name == 'split-batched':
        control = outcomes['split-demand']
        restore.compare(outcome, control)
        for actual, prior in zip(traces, control['dispatch'], strict=True):
            actual_split = {k: v for k, v in actual['split'].items() if k != 'batched_windows'}
            prior_split = {k: v for k, v in prior['split'].items() if k != 'batched_windows'}
            require(actual_split == prior_split and all(actual[k] == prior[k] for k in ('kv', 'gather', 'input')),
                    'actual gathered rows, KV, or split upload work changed')
    outcome['dispatch'] = traces
    return outcome


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
    native, speed, physical, older = required_basis()
    require(plan == make_plan(plan['source_commit'], native, speed, physical, older), 'frozen inputs differ')
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
            checked = []
            for geometry in ('ordinary', 'near-fallback', 'split-demand'):
                case = plan['cases'][geometry]
                argv = [sys.executable, '-B', '-c', peer, json.dumps(case['requests'])]
                entry, _, _ = owner.run(geometry, argv, dict(PATH='/usr/bin:/bin', LANG='C.UTF-8'), bank.P,
                    interaction=lambda io, _: restore.interact(io, case, vocab_size, native=False,
                                                               inputs=request_inputs(geometry)),
                    wall=60, cpu=40, rss_cap=512 << 20, text_cap=128 << 10,
                    file_cap=1 << 20, total_cap=4 << 20)
                require(restore.owner_module.closed(entry) and entry['exit_code'] == 0, 'host peer failed')
                checked.append(geometry)
            outcome.update(host_cases=checked, gpu_executed=False)
        else:
            case = plan['cases'][name]
            requests = []

            def exchange(io, _):
                requests.extend(restore.interact(io, case, vocab_size, native=True, inputs=request_inputs(name)))

            print(f'Native restore case {name} starts', flush=True)
            entry, _, stderr = owner.run(name, case['command'], environment(speed, case), bank.P,
                interaction=exchange, wall=case['wall'], **plan['budgets'])
            outcome.update(native_exit_code=entry['exit_code'], gpu_executed=True, requests=requests)
            require(restore.owner_module.closed(entry) and entry['exit_code'] == 0, 'native server or teardown failed')
            outcome.update(inspect(stderr.read_text(), speed, case, name, requests, entry['pid'], older, row['outcomes']))
            if name == ORDER[-1]:
                basis.comparison.shared.models.model_pins(speed['configuration'], speed['model_assets'], full=True)
        require(plan == make_plan(plan['source_commit'], native, speed, physical, older), 'inputs changed during run')
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
