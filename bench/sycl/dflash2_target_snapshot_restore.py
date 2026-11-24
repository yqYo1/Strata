"""Check DFlash2 server conversation reuse on the target-only snapshot build."""

import fcntl
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import dflash2_prefill_physical_context as physical
import dflash2_target_snapshot_build as build
from engine_once import native_environment

bank, basis, require = build.bank, physical.basis, build.require
KEY = 'dflash2_target_snapshot_restore_v1'
ROOT = bank.B / KEY.replace('_', '-')
THIS = str(Path(__file__).resolve().relative_to(bank.P))
ORDER = ('host', 'short-copy', 'near-copy')
CLEAN_REFERENCE = '4e68b1ac7'
PLAN = dict(order=list(ORDER),
            scope='Owned native GEN conversation switch and restore on the target-only DFlash2 build.',
            work='Send A, unrelated B, and A plus 8209 input tokens. Park 32769 and 253919 tokens '
                 'in independent processes. The near-capacity continuation has 262128 input tokens.',
            gates='Require the original request bytes, actual live snapshot restoration, exact resumed '
                  'and fresh extents, all 12 target attention layers and running-state byte readback. '
                  'Check the DFlash2 tap tail, the KV stage and the BF16 store for the resumed suffix. '
                  'One generated token and normal ownership/fault checks qualify each request.',
            limits='The server cases preserve output room. Full physical fill, public HTTP and clean '
                   'prefill/decode performance have their own gates.',
            stop='Stop on the first source, route, input, verification or ownership failure.')


def experiment():
    original = physical.speed.gate.model.restore
    ctx = basis.core.source.device.load_module('target_snapshot_restore', original.__file__)
    ctx.ROOT, ctx.KEY, ctx.ORDER = ROOT, KEY, ORDER
    original_case, original_argv = ctx.make_case, ctx.argv
    models = basis.comparison.shared.models

    def environment(row, case):
        env = physical.speed.generation.sync_environment(row, dict(root=case['root'], index=0))
        env.update(STRATA_TRACE='1', STRATA_STATE_HASH='1', STRATA_SNAPSHOT_VERIFY='1',
                   STRATA_SYCL_CACHE_LOAN_AUDIT='1')
        env[ctx.SELECTOR] = '0'
        env[physical.speed.SELECTOR] = physical.speed.gate.ARMS['retain']
        return native_environment(row, env)

    def argv(row):
        return original_argv(dict(row, configuration=dict(row['configuration'],
                                args=list(row['draft_arguments']))))

    def inputs(revision):
        native = bank.previous.read(build.ROOT / 'record.json')
        build.context().check(native)
        require(native['stage'] == 'built' and native['host_control']['passed']
                and native['passed'] and native['complete'] and native['ownership_closed']
                and not native['active'], 'finish the coherent engine and its host controls')
        prior = bank.previous.read(physical.ROOT / 'record.json')
        parent = bank.previous.read(build.PARENT_ROOT / 'record.json')
        require(basis.compact(prior) == basis.validation.committed(bank.RESULT, (physical.KEY,))
                and prior['stage'] == 'closed' and prior['passed'] and prior['complete']
                and prior['ownership_closed'] and not prior['active']
                and prior['outcomes']['full-copy']['final_slot_consumed']
                and prior['native_build_sha256'] == bank.signature(bank.builder.compact(parent))
                and native['projection']['target_snapshot_parent_hash'] == parent['binaries']['strata']['sha256'],
                'recover the exact physical reference for these weights and runtime')
        manifest = prior['runtime_manifest']
        libraries = bank.previous.read(Path(manifest['path']))
        require(len(libraries) == manifest['files'] and bank.sha(manifest['path']) == manifest['sha256']
                and all(bank.sha(p) == h for p, h in libraries.items())
                and Path('/proc/sys/kernel/random/boot_id').read_text().strip() == prior['boot'],
                'runtime provider inventory or boot changed')
        binary = native['binaries']['strata']
        names = ('configuration', 'environment_template', 'model_assets', 'native_validation_probe',
                 'boot', 'driver', 'runtime_manifest', 'vocab_size', 'draft_arguments')
        row = {k: prior[k] for k in names}
        row['drafter_assets'] = prior['dflash2_assets']
        selected = bank.previous.historic(CLEAN_REFERENCE, 'dflash2_prefill_handoff_timing_v1')
        control, = selected['processes']
        require(control['passed'] and control['ownership_closed'] and control['clean_speed']
                and selected['native_source_commit'] == parent['native_source_commit']
                and selected['environment_template_sha256'] == bank.signature(row['environment_template']),
                'recover the same-build DFlash2 server settings')
        row.update({k: selected[k] for k in ('ordinary_control', 'python_selected')})
        args = row['draft_arguments']
        require('--mtp' not in args and args[args.index('--draft') + 1] == 'dflash2'
                and args[args.index('--draft-fallback') + 1] == 'none'
                and native['driver'] == row['driver'], 'expected the validated isolated SYCL drafter')
        require(all(bank.sha(p) == h for p, h in row['drafter_assets'].items()), 'draft assets changed')
        models.model_pins(row['configuration'], row['model_assets'], full=False)
        speed = physical.speed
        paths = [THIS, build.THIS, physical.THIS, speed.THIS, speed.gate.THIS,
                 speed.generation.THIS, speed.generation.v2.THIS,
                 str(Path(speed.KV.__file__).relative_to(bank.P)), 'tools/sycl', *bank.basis.POLICIES]
        pins = bank.stage.source.pin(bank.P, paths)['source_sha256']
        require(all(bank.sha(bank.P / name) == h for name, h in pins.items())
                and bank.stage.hashed(dict(bank.stage.source.blobs(bank.P, revision, pins))) == pins,
                'commit all restore and protocol inputs')
        return dict(row, native_source_commit=native['native_source_commit'], binary=binary,
                    native_build_sha256=bank.signature(bank.builder.compact(native)), sources=pins,
                    source_commit=revision, plan=PLAN)

    def inspect(log, row, case, requests, pid):
        proof = ctx.validation.preload_evidence(log, pid)
        require(proof['loader'] == row['native_validation_probe']['loader']['path']
                and not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|teardown uncertain|'
                                  r'conversation cache: (skip parking|discard invalid)|'
                                  r'^strata mtp: .*loaded', log, re.M), 'runtime or snapshot error')
        parked = requests[0]['prompt']
        extent, source = ctx.capacity.one(log, r'^strata serve: conversation cache: restored '
            r'(\d+) \((live|checkpoint)\)', 'the exact live snapshot')
        require(int(extent) == parked and source == 'live' and requests[2]['reused'] == parked
                and requests[2]['read'] == 8209, 'prompt fell through to a fresh read')
        line = ctx.capacity.one(log, r'^strata serve: SNAPSHOT_VERIFY (.*)$', 'target snapshot readback')
        snapshot = dict(x.split('=', 1) for x in line.split())
        require(snapshot['result'] == 'PASS' and snapshot['source'] == 'ram'
                and int(snapshot['main_layers']) == 12 and int(snapshot['upto']) == parked
                and int(snapshot['main_bytes']) >= parked * 12 * physical.speed.KV.CELL_BYTES
                and int(snapshot['running_bytes']) > 0
                and all(int(snapshot[k], 16 if k == 'draft' else 10) == 0
                        for k in ('draft', 'cells', 'mode', 'resident', 'draft_bytes')),
                'the complete target snapshot must be verified without an MTP payload')
        states = list(re.finditer(r'^strata serve: STATE_HASH L=.*$', log, re.M))
        require(len(states) == len(requests) == 3, 'complete request state publications')
        start = 0
        for i, (state, req) in enumerate(zip(states, requests, strict=True)):
            frame = log[start:state.start()]
            start = state.end()
            hashes = dict(re.findall(r'(\w+)=([0-9a-f,-]+)', state[0]))
            require(set(hashes) == set(ctx.ACTIVE_STATE + ctx.STALE_STATE)
                    and int(hashes['L']) == req['prompt'], 'wrong final position or state fingerprint')
            req['active_state'] = {k: hashes[k] for k in ctx.ACTIVE_STATE}
            taps, cold, size = map(int, ctx.capacity.one(frame,
                r'^strata DFlash2: eager, prompt_tail_rows=(\d+) cold_prefix=([01]) vram_bytes=(\d+)$',
                'actual DFlash2 tail'))
            require(taps == (32 if i == 1 else 2046) and size > 0, 'unexpected captured tail')
            req['draft_tail'] = dict(rows=taps, cold_prefix=cold, vram_bytes=size)
            req['attention_store'] = ctx.require_store(frame, case, req)
            if i != 1:
                req['kv_stage'] = physical.speed.KV.check_request(frame, req['reused'],
                    req['read'] - 1, 8192, reuse=True, tap_layers=physical.speed.gate.TAPPED_QSA_LAYERS)
        return dict(validation=proof, restored_tokens=parked, restore_payload=snapshot, requests=requests)

    def prepare():
        require(not ROOT.exists() and KEY not in basis.validation.committed(bank.RESULT),
                'fresh target-only conversation restore plan required')
        revision = basis.validation.git('rev-parse', 'HEAD').decode().strip()
        row = inputs(revision)
        vocab = json.loads((Path(row['configuration']['tokenizer']) / 'vocab.json').read_bytes())
        require(set(range(4096)) <= set(vocab.values()) and max(vocab.values()) + 1 == row['vocab_size'],
                'fixture tokenizer differs')
        ROOT.mkdir(mode=0o700)
        (ROOT / 'host').mkdir(mode=0o700)
        plans = [original_case(row, name) for name in ORDER[1:]]
        for case in plans:
            for folder in ('tmp', 'cache'):
                (Path(case['root']) / folder).mkdir(mode=0o700, parents=True)
        row.update(stage='prepared', outcomes={}, plans=plans, complete=False, passed=False,
                   active=False, ownership_closed=True, speed_eligible=False)
        ctx.save(row)

    def current(row):
        require(basis.validation.committed(bank.RESULT, (KEY,)) == ctx.compact(row)
                and all(row[k] == v for k, v in inputs(row['source_commit']).items())
                and row['plans'] == [original_case(row, name) for name in ORDER[1:]],
                'frozen target-only restore inputs changed')

    ctx.argv, ctx.environment, ctx.inspect, ctx.current = argv, environment, inspect, current
    ctx.prepare = prepare
    return ctx


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(sys.executable == basis.reference.model.PYTHON and action in ('prepare', *ORDER),
            'select one target-only restore action with the pinned interpreter')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        build.builder.guards.no_model_owners([])
        ctx = experiment()
        if action == 'prepare':
            ctx.prepare()
        else:
            ctx.run(action)
