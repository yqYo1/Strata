"""Check live MTP catch-up dispatch and restored state on the fixed engine."""

import fcntl
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
import decode_kv_fast_prefill_restore as server
import decode_mtp_context_fixed as physical

model, bank, restore, require = server.model, server.bank, server.restore, server.require
sys.path[:0] = [str(bank.P), str(bank.P / 'tools')]
from collect_profile_traces import load_tokenizer
from long_generation_bench import WORKLOADS
from serve.frontend import ChatTemplate

ROOT = bank.B / 'decode-mtp-prefix-restore-v2'
KEY = 'decode_mtp_prefix_restore_v2'
FIRST_TRIAL = '7a479bc5b9652e4aa5c6a8fb1ec88d21766b9ace'
FIRST_KEY = 'decode_mtp_prefix_restore_v1'
THIS = str(Path(__file__).resolve().relative_to(bank.P))
ORDER = ('short-off', 'short-on', 'near-off', 'near-on')
FRESH = 8209
GENERATION = 64
CONTEXT_RESERVE = 16
OUTPUTS = (1, 1, GENERATION, 1)
TOKENIZER_FILES = ('vocab.json', 'merges.txt', 'token_type.json', 'tokenizer.json', 'chat_template.jinja')
PROMPT_SOURCES = ('tools/collect_profile_traces.py', 'tools/long_generation_bench.py',
                  'tools/strata_tokenizer.py', 'serve/frontend.py')
CATCHUP = (r'^strata mtp: decode catchup p=(\d+) verified=(\d+) accepted=(\d+) '
           r'rows=(\d+) coupled=(\d+)$')
PLAN = dict(order=list(ORDER), original_failure=FIRST_TRIAL, output_caps=list(OUTPUTS),
    scope='Fixed-width uncoupled MTP4. Use one binary and unchanged prefill, head, '
          'cache and math settings. Vary only whether catch-up consumes the committed prefix.',
    workload='Keep 8209 fresh tokens after live RESTORE. Append the thinking-enabled varied-prose chat '
             'suffix in place of the end of the synthetic pattern. Give it 64 output slots. Adjust '
             'the near-capacity prefix to keep 16 additional slots for the server.',
    acceptance='Verify the emitted width of every nonterminal draft call against its verifier '
               'position and the next commit. Require actual shortening on both input sizes. '
               'Compare matched fixed greedy requests, committed state and live RESTORE. '
               'A divergence requires fixed-history numerical examination before timing.',
    limits='Traced correctness only. Full-model quality, output distributions and clean speed '
           'need separate validation; no equality policy for ordinary generated text.')


def inputs():
    values = physical.inputs()
    record = bank.previous.read(physical.ROOT / 'record.json')
    closed = bank.basis.validation.committed(bank.RESULT, (physical.KEY,))
    near, full = (closed['outcomes'][name] for name in physical.ORDER)
    require(closed == bank.basis.compact(record)
            and closed['stage'] == 'closed' and closed['complete'] and closed['passed']
            and closed['ownership_closed'] and not closed['active']
            and closed['plan'] == physical.PLAN
            and all(record[k] == v for k, v in values.items())
            and all(v['passed'] and v['GPU_executed'] and v['native_exit_code'] == 0
                    and v['ownership_closed'] and not v['fresh_faults'] for v in (near, full))
            and (near['batched_rows'], near['outputs']) == (262141, 2)
            and full['final_slot_consumed'], 'complete the fixed width-four physical gate')
    first = bank.previous.historic(FIRST_TRIAL, FIRST_KEY)
    rejected = first['outcomes']['short-off']
    require(first['stage'] == 'failed' and first['ownership_closed'] and not first['active']
            and first['binary'] == values['binary'] and first['build_sha256'] == values['build_sha256']
            and not rejected['passed'] and rejected['native_exit_code'] == 0 and not rejected['fresh_faults']
            and (rejected['requests'][2]['accepted'], rejected['requests'][2]['offered']) == (6, 6),
            'preserve the original fully accepted restore baseline')
    binary = dict(binaries={'strata': values['binary']}, native_source_commit=values['native_source_commit'])
    return binary, values, values['build_sha256'], bank.signature(closed)


def tokenizer_fixture(row):
    folder = Path(row['configuration']['tokenizer'])
    return dict(folder=str(folder), sha256={name: row['model_assets'][str(folder / name)]['sha256']
                                            for name in TOKENIZER_FILES})


def prose_tokens(fixture):
    folder = Path(fixture['folder'])
    require(set(fixture['sha256']) == set(TOKENIZER_FILES)
            and all(bank.sha(folder / name) == expected for name, expected in fixture['sha256'].items()),
            'restore tokenizer inputs changed')
    config = json.loads((folder / 'tokenizer.json').read_text())
    require(config['pre'] == 'qwen35', 'the fixture uses the Qwen chat tokenizer')
    template = ChatTemplate(folder / 'chat_template.jinja')
    text = template.render([{'role': 'user', 'content': WORKLOADS['varied-prose']['instruction']}],
                           enable_thinking=True)
    tokens = load_tokenizer(folder).encode(text, parse_special=True)
    require(0 < len(tokens) <= FRESH and '<think>' in text, 'missing thinking prompt or suffix capacity')
    return tokens


def workload(near, tail):
    initial = 32769
    if near:
        initial = physical.capacity.CAPACITY - FRESH - GENERATION - CONTEXT_RESERVE
    prefix = [i % 4096 for i in range(initial)]
    padding = [256 + i % 16 for i in range(FRESH - len(tail))]
    return [(prefix, 0), (list(range(128, 161)), 0),
            (prefix + padding + tail, initial), (list(range(256, 289)), 0)]


def plan(row, name):
    near = name.startswith('near-')
    tokenizer = tokenizer_fixture(row)
    body = workload(near, prose_tokens(tokenizer))
    requests = server.requests(near, inputs=body, limits=OUTPUTS)
    case = server.plan(row, name.replace('-', '-4-', 1), input_plan=requests)
    case.update(name=name, root=str(ROOT / name), tokenizer=tokenizer)
    case['environment_sha256'] = bank.signature(environment(row, case))
    return case


def interact(io, case, vocab_size, *, native):
    body = workload(case['near'], prose_tokens(case['tokenizer']))
    return server.interact(io, case, vocab_size, native=native, inputs=body)


def environment(row, case):
    env = server.environment(row, case)
    require(physical.SELECTOR not in env and env['STRATA_SYCL_DECODE_Q6_DPAS_HEAD'] == '2',
            'require the original packed head and an independent MTP selector')
    env[physical.SELECTOR] = str(int(case['enabled']))
    return env


def catchup_evidence(frame, request, enabled):
    windows = [tuple(map(int, m)) for m in re.findall(server.WINDOW, frame, re.M)]
    calls = [tuple(map(int, m)) for m in re.findall(CATCHUP, frame, re.M)]
    require(windows and windows[0] == (request['prompt'] - 1, 1)
            and len(calls) + 1 == len(windows)
            and all(1 <= width <= 4 and 0 <= pos <= 262144 - width for pos, width in windows)
            and sum(width - 1 for _, width in windows) == request['offered'],
            'missing verifier windows, terminal skip or offered drafts')
    for (pos, width, accepted, rows, coupled), expected, (following, _) in zip(
            calls, windows[:-1], windows[1:], strict=True):
        require((pos, width) == expected and coupled == 0 and 0 <= accepted < width
                and following == pos + accepted + 1 and rows == (accepted + 1 if enabled else width),
                'draft catch-up does not match the accepted verifier prefix')
    terminal_pos, terminal_width = windows[-1]
    terminal_accept = request['accepted'] - sum(call[2] for call in calls)
    require(0 <= terminal_accept < terminal_width, 'terminal accept count is inconsistent')
    state = restore.capacity.one(frame, server.STATE, 'committed decoder state')
    state = dict(part.split('=', 1) for part in state.split())
    require(set(state) == set(restore.ACTIVE_STATE + restore.STALE_STATE)
            and int(state['L']) == terminal_pos + terminal_accept + 1, 'committed decoder extent changed')
    return dict(windows=[list(window) for window in windows],
                accepted=[call[2] for call in calls] + [terminal_accept],
                eligible=sum(width == 4 and a < width - 1 for _, width, a, _, _ in calls),
                catchup_rows=sum(call[3] for call in calls),
                active_state={k: state[k] for k in restore.ACTIVE_STATE})


def inspect(log, row, case, requests, pid):
    proof = restore.validation.preload_evidence(log, pid)
    require(proof['loader'] == row['native_validation_probe']['loader']['path']
            and not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|teardown uncertain|cleanup failed', log),
            'native runtime proof or validation differs')
    frames = list(re.finditer(server.footer.HEADER, log, re.M))
    observed = []
    previous_end = 0
    for frame, request in zip(frames, requests, strict=True):
        require(tuple(map(int, frame.groups())) == tuple(request[k] for k in
                ('prompt', 'reused', 'read', 'generated', 'accepted', 'offered')), 'request/footer mismatch')
        content = log[previous_end:frame.start()]
        observed.append(dict(catchup_evidence(content, request, case['enabled']),
                             loan_slots=server.loan_evidence(content, request)))
        previous_end = frame.end()
    require(not re.search(server.WINDOW + '|' + CATCHUP + '|' + server.STATE, log[previous_end:], re.M),
            'unassigned trace after the last request')
    require(sum(o['eligible'] for o in observed) > 0, 'no shortened width-four opportunity')
    restored, kind = restore.capacity.one(log,
        r'^strata serve: conversation cache: restored (\d+) tokens \((live|checkpoint)\)', 'live RESTORE')
    snapshot = restore.capacity.one(log, r'^strata serve: SNAPSHOT_VERIFY (.*)$', 'restored payload')
    snapshot = dict(p.split('=', 1) for p in snapshot.split())
    require(kind == 'live' and int(restored) == requests[0]['prompt'] == int(snapshot['upto'])
            and snapshot['result'] == 'PASS' and snapshot['source'] == 'ram' and snapshot['main_layers'] == '12'
            and all(int(snapshot[k]) > 0 for k in ('main_bytes', 'draft_bytes', 'running_bytes')),
            'the draft and target snapshot were not restored')
    if case['enabled']:
        off = row['outcomes'][case['name'].removesuffix('-on') + '-off']
        require(off['passed'] and off['ownership_closed'] and not off['fresh_faults']
                and off['requests'] == requests and off['snapshot'] == snapshot
                and all({k: v for k, v in a.items() if k != 'catchup_rows'} ==
                        {k: v for k, v in b.items() if k != 'catchup_rows'}
                        for a, b in zip(observed, off['verification'], strict=True))
                and sum(a['catchup_rows'] for a in observed) < sum(b['catchup_rows'] for b in off['verification']),
                'committed catch-up changed the fixed request, active state or RESTORE')
    return dict(validation=proof, verification=observed, snapshot=snapshot)


def experiment():
    ctx = bank.basis.core.source.device.load_module('mtp_prefix_restore', model.__file__)
    ctx.ROOT, ctx.KEY, ctx.ORDER = ROOT, KEY, ORDER
    ctx.inputs, ctx.plan, ctx.environment, ctx.inspect = inputs, plan, environment, inspect
    ctx.restore = SimpleNamespace(**{**vars(restore), 'interact': interact})
    save, current = ctx.save, ctx.current
    controllers = [THIS, physical.THIS, physical.build.THIS, physical.reference.THIS, server.THIS,
                   *server.state.SCRIPTS, *physical.previous.SOURCES, *PROMPT_SOURCES]

    def write(row):
        if row['stage'] == 'prepared':
            row['sources'].update(bank.stage.source.pin(bank.P, controllers)['source_sha256'])
            row['plan'] = PLAN
        save(row)

    def verify(row):
        require(row['plan'] == PLAN and set(controllers) - {'tools/sycl'} <= row['sources'].keys(),
                'freeze the catch-up and source checks before launch')
        current(row)

    ctx.save, ctx.current = write, verify
    return ctx


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(sys.executable == bank.basis.reference.model.PYTHON and action in ('prepare', *ORDER),
            'use the pinned interpreter and one committed model stage')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        ctx = experiment()
        ctx.prepare() if action == 'prepare' else ctx.run(action)
