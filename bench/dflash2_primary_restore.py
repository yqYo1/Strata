"""Verify live DFlash2 conversation restoration on the primary engine."""

import fcntl
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import dflash2_primary_context as physical
import dflash2_v2_generation as generation
from engine_once import native_environment
import prefill_attention_bank_restore as restore

bank, basis, require = physical.bank, physical.basis, physical.require
KEY = 'dflash2_primary_live_restore_v1'
ROOT = bank.B / KEY.replace('_', '-')
THIS = str(Path(__file__).resolve().relative_to(bank.P))
REFERENCE = '962acee66e7e3ff4969370f8fadb28e673fba006'
REFERENCE_KEY = 'dflash2_target_snapshot_restore_v2'
OLD_RECEIPT = bank.B / REFERENCE_KEY.replace('_', '-') / 'record.json'
ORDER = ('host', 'short-copy', 'near-copy')
PLAN = dict(order=list(ORDER),
            scope='Repeat the checked native GEN A/B/A+suffix on the primary binary.',
            cases='Park 32769 or 253919 tokens, switch to an unrelated 33-token prompt, then read 8209 new tokens. '
                  'Keep one output token per request, the original queue envelope and the copy writer.',
            gates='Require live target-only restore, checked RAM payload for all 12 attention layers and running state, '
                  'expected server extents and progress, DFlash2 tail, KV staging and no runtime or ownership failure.',
            limits='Instrumented native correctness. HTTP, direct writer and clean throughput are separate.')


def reference():
    committed = bank.previous.historic(REFERENCE, REFERENCE_KEY)
    receipt = bank.previous.read(OLD_RECEIPT)
    require(basis.compact(receipt) == committed and committed['stage'] == 'closed' and committed['passed']
            and committed['complete'] and committed['ownership_closed'] and not committed['active'],
            'recover the closed live-restore reference')
    for name, tokens in zip(ORDER[1:], (32769, 253919), strict=True):
        row = committed['outcomes'][name]
        require(row['passed'] and row['ownership_closed'] and not row['fresh_faults']
                and row['restored_tokens'] == tokens and row['restore_payload']['result'] == 'PASS'
                and row['requests'][-1]['prompt'] == tokens + 8209, 'recover the original live request boundaries')
    return committed, receipt


def suffix_kv(frame, request, expected):
    start, rows, chunk = request['reused'], request['read'] - 1, restore.capacity.CHUNK
    positions = range(start, start + rows, chunk)
    kv = physical.kv
    planned = [(l, *kv.expected_prefix(positions, l, reuse=l not in physical.TAPPED_QSA_LAYERS)[:2])
               for l in range(3, 48, 4)]
    stages, mirrors = [], []
    for line in frame.splitlines():
        if line.startswith('strata prefill KV stage:'):
            match = kv.STAGE.fullmatch(line)
            require(match, 'malformed suffix stage record')
            stages.append(tuple(map(int, match.groups())))
        elif line.startswith('strata QSA mirror bulk:'):
            match = kv.MIRROR.fullmatch(line)
            require(match, 'malformed suffix host mirror record')
            mirrors.append(tuple(map(int, match.groups())))
    require(stages == planned and [(m[0], m[1]) for m in mirrors] == [
        (min(chunk, start + rows - pos), pos) for pos in positions] * 12,
        'restored suffix lost or reordered a KV segment')
    result = dict(reloaded_pages=sum(v[1] for v in stages), reused_pages=sum(v[2] for v in stages),
                  mirror_D2H_calls=sum(v[3] for v in mirrors), mirror_D2H_bytes=sum(v[4] for v in mirrors))
    require(all(expected[k] == v for k, v in result.items()), 'restored suffix KV extent or traffic changed')
    return result


def experiment():
    ctx = basis.core.source.device.load_module('primary_live_restore', restore.__file__)
    ctx.ROOT, ctx.KEY, ctx.ORDER = ROOT, KEY, ORDER
    base_argv = ctx.argv

    def environment(row, case):
        env = generation.sync_environment(row, dict(root=case['root'], index=0))
        env.update(STRATA_TRACE='1', STRATA_STATE_HASH='1', STRATA_SNAPSHOT_VERIFY='1',
                   STRATA_SYCL_CACHE_LOAN_AUDIT='1', STRATA_SYCL_PREFILL_KEEP_KV_STAGE='1')
        env[ctx.SELECTOR] = '0'
        return native_environment(row, env)

    def argv(row):
        return base_argv(dict(row, configuration=dict(row['configuration'], args=row['draft_arguments'])))

    def inputs(revision):
        prior, seed = reference()
        native, probe = physical.checks.build()
        envelope = physical.checks.parameters(native, probe)
        capacity = basis.validation.committed(bank.RESULT, (physical.KEY,))
        full = capacity.get('outcomes', {}).get('full-copy', {})
        require(capacity['stage'] == 'closed' and capacity['passed'] and capacity['complete']
                and capacity['ownership_closed'] and not capacity['active'] and full.get('final_slot_consumed')
                and capacity['native_build_sha256'] == envelope['build_sha256']
                and capacity['binary'] == native['binaries']['strata_full_context_diagnostic'],
                'complete the primary standalone physical context first')
        require(all(seed[k] == envelope[k] for k in ('boot', 'driver', 'runtime_manifest', 'native_validation_probe'))
                and all(seed[k] == capacity[k] for k in ('configuration', 'vocab_size'))
                and all(bank.sha(p) == h for p, h in seed['drafter_assets'].items()),
                'restore model or runtime inputs changed')
        basis.comparison.shared.models.model_pins(seed['configuration'], seed['model_assets'], full=False)
        pins = ctx.sources(revision)
        for name in (THIS, physical.THIS, physical.checks.THIS, generation.THIS, 'data/expert-profile.bin'):
            require((bank.P / name).read_bytes() == basis.validation.git('show', revision + ':' + name),
                    'commit exact restore/controller input: ' + name)
            pins[name] = bank.sha(bank.P / name)
        rank = 'data/expert-profile.bin'
        args = list(seed['draft_arguments'])
        old_path = args[args.index('--expert-profile') + 1]
        require(old_path == capacity['ranking_paths']['reference'] and pins[rank] ==
                bank.previous.historic(physical.checks.primary.PREFILL,
                    'dflash2_snapshot_direct_prefill_v1')['ranking_profile']['sha256'],
                'restore ranking does not match the primary physical fixture')
        fields = ('configuration', 'environment_template', 'model_assets', 'native_validation_probe', 'boot',
                  'driver', 'runtime_manifest', 'vocab_size', 'drafter_assets', 'ordinary_control', 'python_selected')
        row = {k: seed[k] for k in fields}
        row.update(native_source_commit=native['native_source_commit'], binary=native['binaries']['strata'],
                   native_build_sha256=envelope['build_sha256'], sources=pins, source_commit=revision,
                   physical_context_sha256=bank.signature(capacity), prior_evidence=REFERENCE,
                   draft_arguments=[str(bank.P / rank) if a == old_path else a for a in args],
                   reference_profile_path=old_path, plan=PLAN)
        return row, prior

    def cases(row, prior):
        result = []
        for name, old in zip(ORDER[1:], prior['plans'], strict=True):
            case = ctx.make_case(row, name)
            command = [str(bank.P / 'data/expert-profile.bin') if a == row['reference_profile_path'] else a
                       for a in old['command'][1:]]
            require(old['name'] == name and case['requests'] == old['requests'] and case['command'][1:] == command
                    and bank.signature(environment(row, old)) == old['environment_sha256'],
                    'live restore protocol, argument or environment differs from the reference')
            result.append(case)
        return result

    def inspect(log, row, case, requests, pid):
        proof = ctx.validation.preload_evidence(log, pid)
        require(proof['loader'] == row['native_validation_probe']['loader']['path']
                and not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|teardown uncertain|'
                                  r'conversation cache: (skip parking|discard invalid)|^strata mtp: .*loaded', log, re.M),
                'runtime or snapshot error')
        parked = requests[0]['prompt']
        restored, source = ctx.capacity.one(log, r'^strata serve: conversation cache: restored '
            r'(\d+) tokens \((live|checkpoint)\)', 'live restore')
        require(int(restored) == parked and source == 'live' and requests[2]['reused'] == parked
                and requests[2]['read'] == 8209, 'failed to restore the live prefix')
        snapshot = dict(x.split('=', 1) for x in ctx.capacity.one(log,
                        r'^strata serve: SNAPSHOT_VERIFY (.*)$', 'target snapshot readback').split())
        require(snapshot['result'] == 'PASS' and snapshot['source'] == 'ram' and int(snapshot['main_layers']) == 12
                and int(snapshot['upto']) == parked and int(snapshot['main_bytes']) >= parked * 12 * physical.kv.CELL_BYTES
                and int(snapshot['running_bytes']) > 0 and int(snapshot['draft'], 16) == 0
                and all(int(snapshot[k]) == 0 for k in ('cells', 'mode', 'resident', 'draft_bytes')),
                'incomplete target-only KV and recurrent-state readback')
        prior, _ = reference()
        previous = prior['outcomes'][case['name']]['requests']
        states = list(re.finditer(r'^strata serve: STATE_HASH L=.*$', log, re.M))
        start = 0
        for i, (state, request, old) in enumerate(zip(states, requests, previous, strict=True)):
            frame = log[start:state.start()]
            start = state.end()
            hashes = dict(re.findall(r'(\w+)=([0-9a-f,-]+)', state[0]))
            require(set(hashes) == set(ctx.ACTIVE_STATE + ctx.STALE_STATE) and int(hashes['L']) == request['prompt'],
                    'incomplete state hashes or final target position')
            request['active_state'] = {k: hashes[k] for k in ctx.ACTIVE_STATE}
            taps, cold, size = map(int, ctx.capacity.one(frame,
                r'^strata DFlash2: eager, prompt_tail_rows=(\d+) cold_prefix=([01]) vram_bytes=(\d+)$', 'DFlash2 tail'))
            require(taps == (32 if i == 1 else 2046) and cold == 0 and size > 0, 'DFlash2 tail changed')
            request['draft_tail'] = dict(rows=taps, cold_prefix=cold, vram_bytes=size)
            request['attention_store'] = ctx.require_store(frame, case, request)
            if i == 0:
                request['kv_stage'] = physical.verify_kv(frame, request['read'] - 1, old['kv_stage'])
            if i == 2:
                request['kv_stage'] = suffix_kv(frame, request, old['kv_stage'])
        return dict(validation=proof, restored_tokens=parked, restore_payload=snapshot, requests=requests)

    def prepare():
        require(not ROOT.exists() and KEY not in basis.validation.committed(bank.RESULT), 'inspect prior restore attempt')
        row, prior = inputs(basis.validation.git('rev-parse', 'HEAD').decode().strip())
        ROOT.mkdir(mode=0o700)
        (ROOT / 'host').mkdir(mode=0o700)
        plans = cases(row, prior)
        for case in plans:
            for folder in ('tmp', 'cache'):
                (Path(case['root']) / folder).mkdir(mode=0o700, parents=True)
        row.update(stage='prepared', outcomes={}, plans=plans, complete=False, passed=False,
                   active=False, ownership_closed=True, speed_eligible=False)
        ctx.save(row)

    def current(row):
        require(basis.validation.committed(bank.RESULT, (KEY,)) == ctx.compact(row), 'commit the preceding restore stage')
        expected, prior = inputs(row['source_commit'])
        require(all(row[k] == v for k, v in expected.items()) and row['plans'] == cases(row, prior),
                'native restore inputs changed')

    ctx.argv, ctx.environment, ctx.inspect, ctx.prepare, ctx.current = argv, environment, inspect, prepare, current
    return ctx


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(sys.executable == basis.reference.model.PYTHON and action in ('prepare', *ORDER), 'select a restore stage')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        ctx = experiment()
        if action == 'prepare':
            ctx.prepare()
        else:
            ctx.run(action)
