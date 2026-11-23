"""Exercise the leased terminal verifier and reuse of a full parked conversation."""

import fcntl
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
import prefill_score_decay_restore as state

bank, require, control = state.bank, state.require, state.control
restore = control.restore
KEY = 'prefill_leased_terminal_v1'
ROOT = bank.B / 'prefill-leased-terminal-v1'
ORDER = ('fresh-a', 'fresh-b', *control.ARMS)
CAPACITY, PREFIX = 262144, 253919
TURN = (PREFIX - 1) % 4096
SCRIPTS = [str(Path(__file__).resolve().relative_to(bank.P)), *state.SCRIPTS]
PREFIX_ERROR = 'ERR CONSUME_FULL reused prefix leaves no batched loan before the verifier'


def requests(arm):
    """label, wire bytes, (prompt length, output limit, reuse), or exact rejection."""
    def gen(label, ids, count, reuse):
        return label, restore.protocol.wire(ids, count), (len(ids), count, reuse)

    a = list(range(512, 545))
    b = list(range(768, 801))
    if arm in ('fresh-a', 'fresh-b'):
        if arm == 'fresh-a':
            yield 'ordinary-zero', restore.protocol.wire([2, 17], 0), 'ERR bad request: max_new'
            yield 'ordinary-terminal', b'CONSUME_FULL 2,17\n', (
                'ERR expected: GEN <max_new> <id,id,...> or GENI <max_new> <file> <id,id,...>')
        else:
            payload = b'GEN 1 ' + b'2,' * (CAPACITY - 1) + b'2\n'
            yield 'ordinary-reserve', payload, 'ERR prompt (262144 tokens) + max_new (1) exceeds the context (262144)'
        ids = a if arm == 'fresh-a' else b
        yield gen(arm[-1], ids, 16, 0)
        return
    prefix, switch, _ = (ids for ids, _ in restore.workloads(True))
    require(len(prefix) == PREFIX and prefix[-1] == TURN and TURN > 512, 'original prefix changed')
    # The marker is already the last original token. A second marker at the
    # final verifier position creates an n-1 checkpoint without a split inside
    # the 8224-row batched suffix. Keep the earlier 253918 checkpoint as well.
    full = prefix + [256 + i % 16 for i in range(CAPACITY - PREFIX - 1)] + [TURN]
    terminal = b'CONSUME_FULL ' + ','.join(map(str, full)).encode() + b'\n'
    yield 'zero', restore.protocol.wire([2, 17], 0), 'ERR bad request: max_new'
    yield 'reserve', restore.protocol.wire(full, 1), 'ERR prompt (262144 tokens) + max_new (1) exceeds the context (262144)'
    yield 'short-terminal', b'CONSUME_FULL 2,17\n', 'ERR CONSUME_FULL requires exactly 262144 input tokens and physical context'
    yield gen('prefix', prefix, 1, 0)
    yield gen('park-prefix', switch, 1, 0)
    yield 'fill', terminal, (CAPACITY, 0, PREFIX)
    yield gen('a', a, 16, 0)
    yield 'reject-parked', terminal, PREFIX_ERROR
    # The earlier checkpoint matches; the live prefix and n-1 checkpoint do
    # not. The snapshot verifier must identify the *262144*-token parked image.
    yield gen('restore-full', prefix[:-1] + [401, 402, 403], 1, PREFIX - 1)
    yield gen('b', b, 16, 0)


def plan(arm):
    return [dict(label=label, wire_bytes=len(wire), sha256=restore.signature_bytes(wire),
                 expected=list(expected) if isinstance(expected, tuple) else expected)
            for label, wire, expected in requests(arm)]


def command(row, arm):
    args = restore.argv(row)
    args[0] = row['binary' if arm.startswith('fresh-') else 'diagnostic']['path']
    for name, val in (('--turn-token', TURN), ('--prompt-cache-root', 0), ('--tail-role-token', -1)):
        require(args.count(name) <= 1, 'ambiguous checkpoint option')
        if name in args:
            args[args.index(name) + 1] = str(val)
        else:
            args.extend((name, str(val)))
    return args


def exchange(io, arm, expected_plan, vocab):
    info = {}
    while True:
        line = io.line(600)
        if line.startswith('INFO '):
            info.update(pair.split('=', 1) for pair in line.split()[1:] if '=' in pair)
        else:
            require(line == 'READY 262144 stop', 'unexpected native startup: ' + line[:150])
            break
    require(all(info.get(k) == v for k, v in restore.INFO.items()), 'server layout differs')
    results = {}
    for (label, payload, expected), frozen in zip(requests(arm), expected_plan, strict=True):
        require(frozen['label'] == label and len(payload) == frozen['wire_bytes']
                and restore.signature_bytes(payload) == frozen['sha256']
                and frozen['expected'] == (list(expected) if isinstance(expected, tuple) else expected),
                'request wire or expected protocol changed: ' + label)
        for offset in range(0, len(payload), 256 << 10):
            io.send(payload[offset:offset + (256 << 10)])
        if isinstance(expected, str):
            require(io.line(60) == expected, 'request was not rejected before execution: ' + label)
            continue
        n, limit, reuse = expected
        require(io.line(600) == f'RESUME {reuse}', 'unexpected selected prefix: ' + label)
        tokens, progress, reported_reuse, generation = [], [], [], 0
        while True:
            line = io.line(1800)
            words = line.split()
            require(bool(words), 'empty server response')
            if words[0] == 'PP':
                progress.append(int(words[1]))
            elif words[0] == 'REUSED':
                reported_reuse.append(int(words[1]))
            elif words[0] == 'T':
                tokens.append(int(words[1]))
                require(0 <= tokens[-1] < vocab and len(tokens) <= limit, 'invalid emitted token')
            elif words[0] == 'FULL_CONTEXT_SERVER':
                m = re.fullmatch(r'FULL_CONTEXT_SERVER consumed_position=262144 output_tokens=0 window_width=1 '
                                 r'commit_count=1 commit_wait=PASS refill_drain=PASS generation=([1-9]\d*)', line)
                require(label == 'fill' and not generation and m, 'unexpected terminal verifier publication')
                generation = int(m[1])
            elif words[0] == 'DONE':
                require(len(words) == 15 and reported_reuse == [reuse] and progress and progress[-1] == n - 1
                        and reuse < progress[0] and all(a < b for a, b in zip(progress, progress[1:])),
                        'incomplete request publications: ' + label)
                produced, prompt, accepted, offered, old, read = map(int, (words[1], words[2], *words[6:9], words[14]))
                require((produced, prompt, old, read) == (len(tokens), n, reuse, n - reuse)
                        and 0 <= accepted <= offered and words[5] in ('length', 'stop')
                        and bool(generation) == (limit == 0), 'prompt/output accounting mismatch')
                if limit == 16:
                    require(produced > 1 and offered > 0, 'reset comparison needs actual speculative decode')
                else:
                    require(produced == limit and offered == 0, 'unexpected draft or output')
                results[label] = dict(prompt=n, reused=reuse, generated=produced, output_sha256=restore.signature(tokens),
                                      offered=offered, accepted=accepted, publications=progress,
                                      finish=words[5], generation=generation)
                break
            else:
                raise RuntimeError('unexpected server response: ' + line[:150])
    io.send(b'QUIT\n')
    io.exit(90)
    return results


def inspect(log, arm, responses, previous, pid, probe, *, v2_prefix=None):
    proof = restore.validation.preload_evidence(log, pid)
    require(proof['loader'] == probe['loader']['path'] and not re.search(
        r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|cleanup failed|teardown uncertain|conversation cache: skip parking', log),
        'runtime validation or parking failed')
    matches = list(re.finditer(r'^strata serve: STATE_HASH L=.*$', log, re.M))
    require(len(matches) == len(responses), 'missing or extra server state fingerprint')
    start = 0
    earlier_generation = 0
    for (label, res), marker in zip(responses.items(), matches, strict=True):
        frame = log[start:marker.start()]
        start = marker.end()
        fields = dict(re.findall(r'(\w+)=([0-9a-f,-]+)', marker[0]))
        res['state'] = {k: fields[k] for k in restore.ACTIVE_STATE}
        require(set(fields) == set((*restore.ACTIVE_STATE, *restore.STALE_STATE)), 'unexpected fingerprint schema')
        draft_batches = list(map(int, re.findall(
            r'^strata serve: DRAFT_PREFILL path=(?:batched|token) mode=[012] cells=(\d+)$', frame, re.M)))
        if label in ('prefix', 'fill'):
            upto = res['reused']
            published = []
            for count in draft_batches:
                require(0 < count <= restore.capacity.CHUNK, 'invalid draft publication size')
                upto += count
                published.append(upto)
            require(published == res['publications'], 'draft callback extents differ from prompt publications')
            if label == 'prefix':
                original = (state.original()['outcomes']['near-base']['requests'][0]
                            if v2_prefix is None else v2_prefix)
                require(res['state'] == original['active_state'] and res['output_sha256'] == original['output_hash'],
                        'initial prefix diverged from the measured ordinary route')
            reuse, vec8 = control.ARMS[arm] if v2_prefix is None else (True, False)
            n, old = res['prompt'], res['reused']
            require(int(fields['L']) == n and res['generated'] <= 1, 'wrong final committed position')
            rows = n - old - 1
            res['gather'] = control.gather_counts(frame, vec8)
            res['kv'] = control.kv.check_request(frame, old, rows, restore.capacity.CHUNK, reuse=reuse)
            if v2_prefix is None:
                state.build.check_dispatch(frame, old, rows, retain_keys=False, store_decay=False)
            else:
                require('strata prefill score packs:' not in frame and 'strata prefill GDN decay:' not in frame,
                        'unexpected score or GDN experiment in the retained V2 route')
                res['attention_store'] = restore.require_store(frame, dict(direct=False), dict(read=rows + 1))
            generations = []
            for phase, ready in (('masked', 0), ('ready', 1)):
                gen, first, lent = map(int, restore.capacity.one(frame, rf'^strata cache loan: phase={phase} '
                    rf'generation=(\d+) chunk=8192 first=(\d+) lent=(\d+) decode_ready={ready}$', phase))
                require(first + lent == 1256 and lent > 0, 'wrong expert-cache lease extent')
                generations.append((gen, first, lent))
            gen, _, lent = generations[0]
            require(generations[1] == generations[0] and gen > earlier_generation, 'incorrect loan publication')
            earlier_generation = gen
            required = [f'strata prefill cache binding: generation={gen} parent_match=1 execution_distinct=1',
                        f'strata loan audit: phase=payload rows=1256 bytes=2575206400 lent={lent} same_slots=1 payload_match=1']
            lines = frame.splitlines()
            require(all(lines.count(line) == 1 for line in required), 'missing cache binding or expert payload proof')
            for phase in ('lend', 'refill'):
                restore.capacity.one(frame, rf'^strata loan audit: phase={phase} table_entries=24576 table_match=1 '
                                     r'free_bytes=\d+ total_bytes=\d+$', phase + ' table publication')
            if label == 'fill':
                require(res['generation'] == gen, 'terminal verifier observed another loan')
        else:
            rows = int(restore.capacity.one(frame, r'^strata prefill policy: requested=throughput route=cache-window '
                                            r'tokens=(\d+) chunk=8192$', 'short verify-window route'))
            require(rows == res['prompt'] - res['reused'] - 1 and not draft_batches
                    and res['publications'] == [res['prompt'] - 1], 'short prompt used an unexpected MTP path')
            require('strata cache loan: phase=masked' not in frame and 'strata prefill score packs:' not in frame
                    and 'strata prefill GDN decay:' not in frame and 'strata MoE prefill gather:' not in frame,
                    'short-cache branch unexpectedly ran long prefill')
        if label in ('fill', 'restore-full'):
            saved, reuse, kind = (PREFIX, PREFIX, 'live') if label == 'fill' else (CAPACITY, PREFIX - 1, 'checkpoint')
            restore.capacity.one(frame, rf'^strata serve: conversation cache: restored {reuse} tokens \({kind}\) in '
                                 r'[\d.]+ ms; parked=\d+ bytes=\d+$', 'selected snapshot')
            fields = dict(pair.split('=', 1) for pair in restore.capacity.one(
                frame, r'^strata serve: SNAPSHOT_VERIFY (.*)$', 'verified snapshot').split())
            require(fields['result'] == 'PASS' and fields['source'] == 'ram' and int(fields['upto']) == saved
                    and int(fields['main_layers']) == 12
                    and all(int(fields[k]) > 0 for k in ('main_bytes', 'draft_bytes', 'running_bytes')),
                    'wrong restored image or missing verified state')
            res['snapshot_tokens'] = saved
        else:
            require('strata serve: SNAPSHOT_VERIFY' not in frame, 'unexpected restored session')
        if label in ('a', 'b') and not arm.startswith('fresh-'):
            require(res == previous['fresh-' + label]['requests'][label],
                    'reset state, MTP work or output differs from the independent fresh server: ' + label)
        if arm not in ('base', 'fresh-a', 'fresh-b'):
            reference = previous['base']['requests'][label]
            for field in ('state', 'generated', 'output_sha256', 'offered', 'accepted', 'finish'):
                require(res[field] == reference[field], 'candidate changed ' + label + ' ' + field)
    return dict(validation=proof, requests=responses)


def runtime():
    native, speed, identity, _ = state.inputs()
    gate = bank.basis.validation.committed(bank.RESULT, (state.KEY,))
    require(gate['build_sha256'] == identity and gate['binary'] == native['binaries']['strata']
            and not gate['active'] and gate['ownership_closed']
            and all(gate['outcomes'].get(k, {}).get('passed') for k in ('short-base', 'near-base')),
            'need original-equivalent same-build off controls at both lengths')
    return native, speed, identity


def environment(row, case):
    env = control.model.environment(row, case)
    reuse, gather = control.ARMS.get(case['arm'], (False, False))
    env.update({control.KV: str(int(reuse)), control.GATHER: str(int(gather)),
                'STRATA_SYCL_PREFILL_RETAIN_SCORE_KEYS': '0', 'STRATA_SYCL_PREFILL_GDN_PRECOMPUTE_DECAY': '0'})
    return env


def case_plan(row, arm):
    case = dict(arm=arm, root=str(ROOT / arm), borrow=False, direct=False,
                command=command(row, arm), requests=plan(arm))
    case['environment_sha256'] = bank.signature(environment(row, case))
    return case


def compact(row):
    small = {k: v for k, v in row.items() if k not in ('sources', 'model_assets', 'plans', 'environment_template')}
    for k in ('sources', 'model_assets', 'plans', 'environment_template'):
        small[k + '_sha256'] = bank.signature(row[k])
    return small


def save(row):
    bank.stage.write(ROOT / 'record.json', row)
    data = bank.previous.read(bank.RESULT)
    data[KEY] = compact(row)
    bank.stage.write(bank.RESULT, data)


def prepare():
    require(not ROOT.exists() and KEY not in bank.basis.validation.committed(bank.RESULT), 'inspect previous execution')
    native, speed, identity = runtime()
    sources = bank.stage.source.pin(bank.P, [*SCRIPTS, 'tools/sycl', *bank.basis.POLICIES])
    row = {k: speed[k] for k in control.model.FIELDS}
    row.update(stage='prepared', source_commit=sources['revision'], sources=sources['source_sha256'],
               native_source_commit=native['native_source_commit'], build_sha256=identity,
               binary=native['binaries']['strata'],
               diagnostic=native['binaries']['strata_full_context_diagnostic'],
               ordinary_control=control.model.ordinary_control(), outcomes={}, passed=False, complete=False,
               ownership_closed=True, active=False, order=ORDER, timing_eligible=False)
    row['plans'] = [case_plan(row, arm) for arm in ORDER]
    for case in row['plans']:
        for subdir in ('tmp', 'cache'):
            (Path(case['root']) / subdir).mkdir(mode=0o700, parents=True, exist_ok=False)
    save(row)


def current(row):
    native, speed, identity = runtime()
    require(bank.basis.validation.committed(bank.RESULT, (KEY,)) == compact(row)
            and bank.stage.source.pin(bank.P, list(row['sources']))['source_sha256'] == row['sources']
            and all(row[k] == speed[k] for k in control.model.FIELDS)
            and row['order'] == list(ORDER) and row['build_sha256'] == identity
            and row['native_source_commit'] == native['native_source_commit']
            and row['binary'] == native['binaries']['strata']
            and row['diagnostic'] == native['binaries']['strata_full_context_diagnostic']
            and row['ordinary_control'] == control.model.ordinary_control()
            and row['plans'] == [case_plan(row, arm) for arm in ORDER], 'terminal comparison inputs changed')
    bank.basis.comparison.shared.models.model_pins(row['configuration'], row['model_assets'])


def run(arm):
    row = bank.previous.read(ROOT / 'record.json')
    current(row)
    idx = ORDER.index(arm)
    case = row['plans'][idx]
    root = Path(case['root'])
    receipt = root / 'execution.json'
    require(row['stage'] in ('prepared', 'running') and row['ownership_closed']
            and not row['active'] and list(row['outcomes']) == list(ORDER[:idx]) and not receipt.exists(), 'inspect original case first')
    vocab = set(json.loads((Path(row['configuration']['tokenizer']) / 'vocab.json').read_bytes()).values())
    require(set(range(4096)) <= vocab, 'some diagnostic prompt IDs are outside this vocabulary')
    cursor, = (line.removeprefix('-- cursor: ') for line in subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15).splitlines()
        if line.startswith('-- cursor: '))
    owner = restore.owner_module.Owner(root / 'owned', lambda: bank.stage.write(receipt, owner.commands))
    outcome = dict(complete=False, host_probe_accepted=False)
    try:
        outputs = {}

        def interact(io, entry):
            outputs.update(exchange(io, arm, case['requests'], max(vocab) + 1))

        entry, _, err = owner.run(arm, case['command'], environment(row, case), bank.P, interaction=interact,
                                  wall=2400, **restore.capacity.BUDGET)
        outcome['native_exit_code'] = entry['exit_code']
        require(restore.owner_module.closed(entry) and entry['exit_code'] == 0, 'server or teardown failed')
        parsed = inspect(err.read_text(), arm, outputs, row['outcomes'], entry['pid'], row['native_validation_probe'])
        current(row)
        if arm == ORDER[-1]:
            bank.basis.comparison.shared.models.model_pins(row['configuration'], row['model_assets'], full=True)
        outcome.update(**parsed, complete=True, host_probe_accepted=True)
    except Exception as exc:
        outcome['error'] = repr(exc)
    finally:
        restore.validation.finish(outcome, owner, cursor, row['boot'])
        outcome['passed'] = outcome['passed'] and outcome.get('driver_after') == row['driver']
        row['outcomes'][arm] = outcome
        done = outcome['passed'] and arm == ORDER[-1]
        row.update(stage='closed' if done else 'running' if outcome['passed'] else 'failed',
                   passed=done, complete=done, ownership_closed=outcome['ownership_closed'], active=bool(owner.active))
        save(row)
    require(outcome['passed'], outcome.get('error', 'terminal qualification failed'))


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(sys.executable == bank.basis.reference.model.PYTHON and action in ('layout', 'prepare', *ORDER),
            'use the pinned Python and a planned terminal stage')
    if action == 'layout':
        pins = bank.stage.source.pin(bank.P, SCRIPTS)
        require(set(pins['source_sha256']) == set(SCRIPTS), 'commit the entire terminal workload first')
        for name in ('fresh-a', 'fresh-b', 'base'):
            print(name, [(p['label'], p['expected']) for p in plan(name)])
    else:
        with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            bank.builder.guards.no_model_owners([])
            if action == 'prepare':
                prepare()
            else:
                run(action)
