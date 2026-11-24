"""Check the closed training capture with the traced KV reader."""

import fcntl
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import dflash2_long_routing as capture

bank, basis, require = capture.bank, capture.basis, capture.require
KEY = 'dflash2_long_routing_train_replay_v1'
ORIGINAL = '9f21303b1a90fe1ab4b39b4a20fd3d5b01c80d3d'
THIS = str(Path(__file__).resolve().relative_to(bank.P))
KV = 'bench/sycl/prefill_kv_stage_trace.py'
ORIGINAL_ERROR = "ValueError('clean request contains the diagnostic trace')"


def save(value):
    record = bank.previous.read(bank.RESULT)
    record[KEY] = value
    bank.stage.write(bank.RESULT, record)


def inputs(revision):
    ctx = capture.experiment('train')
    original = bank.previous.historic(ORIGINAL, ctx.KEY)
    live = bank.previous.read(ctx.ROOT / 'record.json')
    prior, = original['processes']
    require(original == ctx.compact(live) == basis.validation.committed(bank.RESULT, (ctx.KEY,))
            and original['stage'] == 'failed' and not original['active']
            and original['ownership_closed'] and prior['ownership_closed'] and prior['boot_unchanged']
            and not prior['fresh_faults'] and prior['driver_after'] == original['driver']
            and prior['error'] == ORIGINAL_ERROR, 'recover the exact closed checker failure')
    require(bank.sha(original['binary']['path']) == original['binary']['sha256'],
            'original routing build identity changed')
    for name, digest in live['sources'].items():
        if name == capture.THIS:
            continue
        path = bank.P / name
        require((not path.exists() if digest is None else bank.sha(path) == digest),
                'original routing input changed: ' + name)
    plan, = original['plans']
    root = Path(plan['root'])
    require(root == ctx.ROOT / '0', 'original capture path differs')
    files = ['config.json', 'plan.json', 'execution.json', 'engine-start.json', 'engine-exit.json',
             'engine.stderr', 'validation.log', capture.ROUTING_FILE]
    suite, = (root / 'bench').glob('*/results.json')
    files.extend(str((suite.parent / leaf).relative_to(root)) for leaf in
                 ('results.json', 'requests.jsonl', 'journal-state.json'))
    pins = {}
    for name in files:
        path = root / name
        require(path.is_file() and not path.is_symlink(), 'missing original routing input: ' + name)
        pins[name] = bank.sha(path)
    require(all(pins[n] == h for n,h in plan['generated'].items())
            and (root / capture.ROUTING_FILE).stat().st_size <= original['budgets']['file_cap']
            and (root / 'engine.stderr').stat().st_size <= original['budgets']['text_cap'],
            'original generated inputs or output envelope differ')
    terminal = bank.previous.read(root / 'execution.json')
    owner, = terminal['commands']
    require(not terminal['active'] and ctx.OWNER.closed(owner) and owner['exit_code'] == 0
            and terminal['outcome'] == prior, 'original benchmark owner or evidence differs')
    cfg = bank.previous.read(root / 'config.json')
    exit_ = ctx.gen.native_end(root, original, dict(config=cfg), original['boot'], owner['pid'])
    require(exit_['native'] == prior['native_identity'], 'original native identity differs')
    sources = {}
    for name in (THIS, capture.THIS, KV, *capture.probe.READERS):
        path = bank.P / name
        require(path.read_bytes() == basis.validation.git('show', revision + ':' + name),
                'commit the traced reader and routing replay first')
        sources[name] = bank.sha(path)
    frozen = dict(source_commit=revision, sources=sources, original=ORIGINAL,
                  original_source_commit=original['source_commit'],
                  original_source_inputs=original['source_inputs'], original_record_sha256=bank.signature(original),
                  original_error=ORIGINAL_ERROR, original_native_identity=exit_['native'], native_exit_code=0,
                  capture_sha256=pins, root=str(root), binary=original['binary'],
                  scope='Replay exact captured training work through the traced KV, CPU census and routing readers. '
                        'Preserve the original failed checker and use the separate replay for training eligibility.')
    return frozen, live, root, cfg


def main(action):
    require(sys.executable == basis.reference.model.PYTHON and action in ('prepare', 'run'),
            'use the pinned Python with prepare or run')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        if action == 'prepare':
            require(KEY not in basis.validation.committed(bank.RESULT), 'original replay case already exists')
            revision = basis.validation.git('rev-parse', 'HEAD').decode().strip()
            frozen, _, root, _ = inputs(revision)
            require(not (root / capture.USAGE_FILE).exists(), 'unexpected derived file before the replay')
            save(dict(stage='prepared', passed=False, plan=frozen))
            return
        result = basis.validation.committed(bank.RESULT, (KEY,))
        frozen, original, root, cfg = inputs(result['plan']['source_commit'])
        require(result['stage'] == 'prepared' and result['plan'] == frozen, 'replay inputs changed')
        try:
            evidence = capture.evidence(root, capture.WORKLOAD_ID, 'train')
            prior, = original['processes']
            require(evidence == prior['evidence'], 'original benchmark journal or request work changed')
            text = (root / 'engine.stderr').read_text()
            validation = basis.validation.preload_evidence(text, frozen['original_native_identity']['pid'])
            require(validation['loader'] == original['native_validation_probe']['loader']['path']
                    and not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|teardown uncertain', text),
                    'unexpected original engine/runtime validation')
            basis.comparison.require_quiet_validation(root)
            capture.leaves.speed.verify_model_assets(original, cfg)
            result['routes'] = capture.routes(text, evidence['samples'], root.parent)
            result.update(stage='closed', passed=True, native_executed=False)
        except Exception as error:
            result.update(stage='failed', passed=False, native_executed=False, error=repr(error))
        save(result)
        print({k: result.get(k) for k in ('stage', 'passed', 'error')}, flush=True)
        require(result['passed'], 'preserve the original capture and replay outcomes')


if __name__ == '__main__':
    action, = sys.argv[1:]
    main(action)
