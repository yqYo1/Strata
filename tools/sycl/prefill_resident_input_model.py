"""Check the four independent BF16 prefill input/output choices on the ordinary queue."""

import fcntl
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
import prefill_attention_bank_restore as restore
import prefill_resident_input_build as build

basis, require = build.basis, build.require
P, B, RESULT = build.P, build.B, build.RESULT
ROOT = B / 'prefill-resident-input-model-v1'
KEY = 'prefill_resident_input_model_v1'
DEVICE = 'prefill_resident_input_device_v1'
INPUT = 'STRATA_SYCL_PREFILL_READ_ATTENTION_BANK'
OUTPUT = restore.SELECTOR
ARMS = {'copy': (False, False), 'input': (True, False),
        'output': (False, True), 'both': (True, True)}
ORDER = tuple(f'{size}-{arm}' for size in ('short', 'near') for arm in ARMS)
FIELDS = ('configuration', 'environment_template', 'model_assets', 'native_validation_probe',
          'runtime_manifest', 'boot', 'driver')


def inputs():
    native = build.previous.read(build.ROOT / 'record.json')
    build.check(native)
    signature = build.signature(build.builder.compact(native))
    fixture = basis.validation.committed(RESULT, (DEVICE,))
    require(native['stage'] == 'built' and native['passed'] and native['complete'] and native['host_control']['passed']
            and native['ownership_closed'] and not native['active'] and fixture['passed']
            and fixture['ownership_closed'] and fixture['native_exit_code'] == 0 and fixture['fresh_faults'] == []
            and fixture['source'] == native['native_source_commit'] == build.INPUT_SOURCE
            and fixture['native_build_sha256'] == signature and fixture['binary'] == native['binaries'][build.TARGET]
            and fixture['arms'] == 4 and fixture['resident_window'] == [8, 3],
            'require the coherent engine and its actual four-way device fixture')
    # build.check already verifies this original reference through basis.basis.
    speed = build.previous.read(basis.reference.ROOT / 'record.json')
    return native, speed, signature, build.signature(fixture)


def ordinary_environment(row, root, prior):
    env = basis.reference.context().environment(row, dict(index=0, root=root, heads=1))
    # Match the completed ordinary control before introducing the two selectors and diagnostics.
    base_env = {k: v.replace(root, prior['root']) for k, v in env.items()}
    base_env[OUTPUT] = '0'
    require(basis.comparison.fingerprint(base_env) == prior['environment_sha256']
            and env['STRATA_SYCL_PREFILL_EXPERT_BANK_SLOTS'] == '384'
            and env['STRATA_SYCL_CACHE_SCRATCH_LEASE'] == env['STRATA_SYCL_PREFILL_SAME_QUEUE'] == '1'
            and env.get('STRATA_SYCL_PREFILL_PREFIX_EXPERT_COPY', 'sycl') == 'sycl'
            and env.get('STRATA_SYCL_PREFILL_PREFIX_EXPERT_UPLOAD', 'off') == 'off'
            and env.get('STRATA_SYCL_PREFILL_PREFIX_EXPERT_DEFER_COMPLETION', '0') == '0',
            'ordinary 384-slot server placement changed')
    return env


def environment(row, case):
    root = case['root']
    env = ordinary_environment(row, root, row['ordinary_control'])
    probe = row['native_validation_probe']
    env.update(TMPDIR=root + '/tmp', XDG_CACHE_HOME=root + '/cache', STRATA_TRACE='1',
               STRATA_STATE_HASH='1', STRATA_SNAPSHOT_VERIFY='1', STRATA_DECODE_TIMING='1',
               STRATA_SYCL_CACHE_LOAN_AUDIT='1', LD_PRELOAD=probe['library']['path'],
               STRATA_UR_PROBE_EXPECTED_LOADER=probe['loader']['path'], UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION')
    env.pop('UR_LOG_TRACING', None)
    for name in ('UR_LOG_VALIDATION', 'UR_LOG_LOADER', 'UR_LOG_LEVEL_ZERO'):
        env[name] = 'level:warning;flush:warning;output:stderr'
    env[INPUT], env[OUTPUT] = str(int(case['borrow'])), str(int(case['direct']))
    return env


def plan(row, name):
    size, arm = name.split('-')
    borrow, direct = ARMS[arm]
    case = dict(name=name, near=size == 'near', borrow=borrow, direct=direct, root=str(ROOT / name))
    case.update(command=restore.argv(row), requests=restore.input_plan(case['near']),
                environment_sha256=build.signature(environment(row, case)))
    return case


def ordinary_control():
    prior = basis.validation.committed(RESULT, (build.COMPARISON,))
    require(prior['stage'] == 'closed' and prior['passed'] and prior['ownership_closed']
            and not prior['active'] and len(prior['processes']) == 9 and prior['plans'][1]['variant'] == 'copy',
            'completed ordinary 64K comparison required')
    return {k: prior['plans'][1][k] for k in ('root', 'environment_sha256')}


def prepare():
    require(not ROOT.exists() and KEY not in basis.validation.committed(RESULT), 'fresh comparison required')
    native, speed, identity, fixture = inputs()
    pin = build.stage.source.pin(P, ['tools/sycl', *basis.POLICIES])
    row = {k: speed[k] for k in FIELDS}
    vocab = json.loads((Path(row['configuration']['tokenizer']) / 'vocab.json').read_bytes())
    require(set(range(4096)) <= set(vocab.values()), 'model fixture IDs missing')
    row.update(stage='prepared', source_commit=pin['revision'], sources=pin['source_sha256'],
               binary=native['binaries']['strata'], native_source_commit=native['native_source_commit'],
               build_sha256=identity, device_sha256=fixture, ordinary_control=ordinary_control(),
               vocab_size=max(vocab.values()) + 1, outcomes={}, passed=False, complete=False,
               ownership_closed=True, active=False,
               plan=dict(unit='Independent read/write selectors, same ordinary server layout and ELF.',
                         order=ORDER, gates='Both sizes must match their same-size copied-input/output control '
                         'on active state, one greedy fixture ID, publications, restored snapshot and decode cache. '
                         'Require exact eligible device rows and bytes for the PLE model, mixed-host near case, '
                         'short cache-window switch and ragged 8208-row resumed suffix. Stop at first failed gate.',
                         scope='Instrumented native correctness. Clean throughput and physical final slot tested separately.'))
    row['plans'] = [plan(row, name) for name in ORDER]
    row['host_input_checks'] = check_reader()
    for case in row['plans']:
        require(all(r['prompt'] + r['max_new'] + 8 <= restore.capacity.CAPACITY for r in case['requests']),
                'request exceeds native server admission')
        for folder in ('tmp', 'cache'):
            (Path(case['root']) / folder).mkdir(parents=True, mode=0o700)
    save(row)


def save(row):
    build.stage.write(ROOT / 'record.json', row)
    result = build.previous.read(RESULT)
    result[KEY] = basis.compact(row)
    build.stage.write(RESULT, result)


def current(row):
    require(basis.validation.committed(RESULT, (KEY,)) == basis.compact(row)
            and build.stage.source.pin(P, list(row['sources']))['source_sha256'] == row['sources'],
            'commit and verify all model controller inputs')
    native, speed, identity, fixture = inputs()
    require(row['binary'] == native['binaries']['strata'] and row['native_source_commit'] == native['native_source_commit']
            and row['build_sha256'] == identity and row['device_sha256'] == fixture
            and all(row[k] == speed[k] for k in FIELDS) and row['ordinary_control'] == ordinary_control()
            and row['plans'] == [plan(row, name) for name in ORDER], 'frozen model inputs differ')


def input_receipt(frame, rows, store, borrow, *, trace_layers=True):
    label = 'strata prefill attention input: '
    traced = re.findall(r'^strata trace: prefill attention input after layer (\d+): '
                        r'eligible=(\d+) borrowed=(\d+) borrowed_bytes=(\d+) requested_D2D_bytes=(\d+)$',
                        frame, re.M)
    if rows < restore.capacity.CHUNK:
        require(not traced and label not in frame and store['pieces'] == 0,
                'short decode-cache path accessed the prefill bank')
        return None
    pieces, volume = restore.capacity.expected_store(rows, store['gpu_rows'], store['maximum_rows'])
    layers = restore.capacity.LAYERS
    require(pieces == store['pieces'] and pieces % layers == volume % layers == 0,
            'attention input and output window geometry differs')
    # All 48 layers use the same resident windows. Layer 0 embeds; layer 1 runs PLE in scratch.
    eligible, nbytes = pieces // layers * (layers - 2), volume // layers * (layers - 2)
    values = restore.capacity.one(frame, '^' + label + r'requested=(\d+) eligible=(\d+) borrowed=(\d+) '
                                  r'borrowed_bytes=(\d+) requested_D2D_bytes=(\d+)$', 'BF16 attention inputs')
    requested, actual, used, borrowed_bytes, copied_bytes = map(int, values)
    require(eligible > 0 and (requested, actual, used, borrowed_bytes, copied_bytes) ==
            (int(borrow), eligible, eligible if borrow else 0, nbytes if borrow else 0, 0 if borrow else nbytes),
            'wrong resident input selection, piece count or D2D byte count')
    require(len(traced) == (layers if trace_layers else 0), 'unexpected layer trace count')
    for layer, counts in enumerate(traced):
        completed = max(0, layer - 1)
        count, size = (pieces // layers) * completed, (volume // layers) * completed
        require(tuple(map(int, counts)) == (layer, count, count if borrow else 0,
                size if borrow else 0, 0 if borrow else size), f'wrong input execution at layer {layer}')
    return dict(eligible=actual, borrowed=used, borrowed_bytes=borrowed_bytes, copy_bytes=copied_bytes,
                checked_layers=len(traced))


def expert_bank_receipt(frame, store, mixed):
    slots, effective, staged, gpu_rows = map(int, restore.capacity.one(frame,
        r'^strata MoE expert bank: requested_slots=(\d+) effective_slots=(\d+) staged=(\d+) '
        r'host_slab_bytes=\d+ device_bytes=\d+ residual_gpu_rows=(\d+) ', 'ordinary expert bank'))
    # ExpertBankPlan only stages 384 slots for an all-device residual.
    # A host-backed tail needs the full 512-slot cache on the same SYCL queue.
    expected = (512, 0) if mixed else (384, 1)
    require((slots, gpu_rows) == (384, store['gpu_rows']) and (effective, staged) == expected
            and (store['host_rows'] > 0) == mixed, 'server expert bank or residual placement differs')
    return dict(requested=slots, effective=effective, staged=bool(staged))


def check_reader():
    """Catch totals that look correct even when PLE or the host tail borrowed the wrong input."""
    checked, rejected = 0, 0
    # Fixed model extents: ragged 8192+16 and fourteen resident chunks before a host tail.
    for rows, gpu, maximum, pieces in ((8208, 8208, 8208, 2), (253918, 114688, 81920, 14)):
        store = dict(gpu_rows=gpu, maximum_rows=maximum, pieces=48 * pieces)
        for borrow in (False, True):
            lines = []
            count = size = 0
            for layer in range(48):
                if layer >= 2:
                    count += pieces
                    size += gpu * 20480
                stats = (f'eligible={count} borrowed={count if borrow else 0} '
                         f'borrowed_bytes={size if borrow else 0} requested_D2D_bytes={0 if borrow else size}')
                lines.append(f'strata trace: prefill attention input after layer {layer}: {stats}')
            lines.append(f'strata prefill attention input: requested={int(borrow)} {stats}')
            result = input_receipt('\n'.join(lines), rows, store, borrow)
            require(result['eligible'] == count and result['borrowed_bytes'] + result['copy_bytes'] == size,
                    'wrong two-piece or mixed-host control')
            checked += 1
            # Publish a layer-2-sized prefix at layer 1 while retaining the exact final totals.
            corrupt = lines.copy()
            corrupt[1] = lines[2].replace('after layer 2:', 'after layer 1:')
            try:
                input_receipt('\n'.join(corrupt), rows, store, borrow)
            except RuntimeError as error:
                require(str(error) == 'wrong input execution at layer 1', 'unexpected rejection')
                rejected += 1
            else:
                raise RuntimeError('incorrect PLE-layer execution was accepted')
    require(input_receipt('', 32, dict(pieces=0), False) is None, 'short control failed')
    rejected_banks = 0
    for mixed, gpu, host in ((False, 8208, 0), (True, 114688, 139230)):
        for effective, staged in ((384, 1), (512, 0)):
            frame = (f'strata MoE expert bank: requested_slots=384 effective_slots={effective} staged={staged} '
                     f'host_slab_bytes=0 device_bytes=1022361600 residual_gpu_rows={gpu} allocation_limit_bytes=0')
            try:
                expert_bank_receipt(frame, dict(gpu_rows=gpu, host_rows=host), mixed)
            except RuntimeError as error:
                require(str(error) == 'server expert bank or residual placement differs' and (effective == 512) != mixed,
                        'unexpected expert placement rejection')
                rejected_banks += 1
            else:
                require((effective == 512) == mixed, 'incorrect expert bank admitted')
    require(rejected_banks == 2, 'missing expert bank rejection')
    return dict(long_cases=checked, incorrect_layer_rejected=rejected, short_cache_case=True,
                incorrect_expert_bank_rejected=rejected_banks)


def inspect(log, row, case, requests, pid):
    result = restore.inspect(log, row, case, requests, pid)
    require(len(re.findall(r'^strata generate: PLE on, table \d+ rows of ', log, re.M)) == 1
            and 'strata MoE prefix expert upload' not in log, 'unexpected PLE or upload route')
    frames = re.split(r'^strata serve: STATE_HASH L=.*$', log, flags=re.M)
    require(len(frames) == 4 and 'strata prefill attention input: ' not in frames[-1],
            'unassigned input counters after final publication')
    for frame, request in zip(frames[:-1], requests, strict=True):
        store = request['attention_store']
        request['attention_input'] = input_receipt(frame, request['read'] - 1, store, case['borrow'])
        if request['prefill_route'] == 'layer-major':
            mixed = case['near'] and request is requests[0]
            request['expert_bank'] = expert_bank_receipt(frame, store, mixed)
    return result


def run(name):
    row = build.previous.read(ROOT / 'record.json')
    current(row)
    index = ORDER.index(name)
    require(row['stage'] in ('prepared', 'running') and not row['active'] and row['ownership_closed']
            and list(row['outcomes']) == list(ORDER[:index]), 'inspect previous model receipts')
    case = row['plans'][index]
    root = Path(case['root'])
    receipt = root / 'execution.json'
    require(not receipt.exists(), 'existing attempt must be inspected')
    cursor, = (line.removeprefix('-- cursor: ') for line in subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15).splitlines()
        if line.startswith('-- cursor: '))
    outcome = dict(complete=False, host_probe_accepted=False)
    owner = restore.owner_module.Owner(root / 'owned', lambda: build.stage.write(receipt, owner.commands))
    requests = []
    try:
        def exchange(io, entry):
            requests.extend(restore.interact(io, case, row['vocab_size'], native=True))
        entry, _, err = owner.run(name, case['command'], environment(row, case), P, interaction=exchange,
                                  wall=2400 if case['near'] else 1200, **restore.capacity.BUDGET)
        outcome.update(native_exit_code=entry['exit_code'], requests=requests)
        require(restore.owner_module.closed(entry) and entry['exit_code'] == 0, 'native server or teardown failed')
        outcome.update(inspect(err.read_text(), row, case, requests, entry['pid']))
        if case['borrow'] or case['direct']:
            restore.compare(outcome, row['outcomes'][name.split('-')[0] + '-copy'])
        current(row)
        if name == ORDER[-1]:
            basis.comparison.shared.models.model_pins(row['configuration'], row['model_assets'], full=True)
        outcome.update(complete=True, host_probe_accepted=True)
    except Exception as exc:
        outcome['error'] = repr(exc)
    finally:
        basis.validation.finish(outcome, owner, cursor, row['boot'])
        outcome['passed'] = outcome['passed'] and outcome.get('driver_after') == row['driver']
        row['outcomes'][name] = outcome
        done = outcome['passed'] and name == ORDER[-1]
        row.update(stage='closed' if done else 'running' if outcome['passed'] else 'failed',
                   complete=done, passed=done, ownership_closed=outcome['ownership_closed'], active=bool(owner.active))
        save(row)
    require(outcome['passed'], outcome.get('error', 'model gate failed'))


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(sys.executable == basis.reference.model.PYTHON and action in ('prepare', *ORDER),
            'use the pinned interpreter and a planned model action')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        build.builder.guards.no_model_owners([])
        if action == 'prepare':
            prepare()
        else:
            run(action)
