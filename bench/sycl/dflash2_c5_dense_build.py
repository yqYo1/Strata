"""Rebuild the measured width-five head engine with one dense Q6 selector."""

import fcntl
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import dflash2_c5_head_build as head
from official_model_sources import merge_delta

bank, builder, require = head.bank, head.builder, head.require
KEY = 'dflash2_c5_dense_native_v1'
ROOT = bank.B / KEY.replace('_', '-')
THIS = str(Path(__file__).resolve().relative_to(bank.P))
DENSE = '063ec7946329400d2887e5a0696f97be5341c477'
HEAD_EVIDENCE = 'b71522878920fbf548ba70f8a806c27731efe0f4'
HEAD_KEY = 'dflash2_c5_head_generation_v1'
PATHS = ('include/strata/core/verify.hpp', 'include/strata/core/verify_window_policy.hpp',
         'src/core/verify.cpp', head.POLICY_TEST)


def parent_inputs(expected=None):
    native, foundation, source, recipe, proof = head.project(expected)
    parent = bank.previous.read(head.ROOT / 'record.json')
    compact = builder.compact(parent)
    require(parent['stage'] == 'built' and parent['passed'] and parent['complete']
            and parent['ownership_closed'] and not parent['active'] and parent['host_control']['passed']
            and compact == bank.basis.validation.committed(bank.RESULT, (head.KEY,))
            and parent['source_files'] == bank.stage.hashed(source) and parent['recipe'] == recipe
            and parent['projection'] == proof and parent['foundation'] == foundation
            and parent['external_inputs'] == native['external_inputs'],
            'recover the exact head build source and recipe from Git')
    timing = bank.previous.historic(HEAD_EVIDENCE, HEAD_KEY)
    require(timing['stage'] == 'closed' and timing['complete'] and timing['passed']
            and timing['ownership_closed'] and not timing['active']
            and timing['decision']['head_gain_qualified'] and not timing['decision']['adopted']
            and timing['c5_head_arm']['binary'] == parent['binaries']['strata']
            and timing['c5_head_arm']['native_source_commit'] == parent['native_source_commit']
            and timing['c5_head_arm']['native_build_sha256'] == bank.signature(compact),
            'the measured width-five head must bind the recovered parent')
    return native, foundation, source, recipe, proof, parent


def project(expected=None):
    native, foundation, source, recipe, proof, parent = parent_inputs(expected)
    component = bank.previous.historic(head.C5_EVIDENCE, 'q6_canonical_c5_component_check_v2')
    shape = component['decision']['cells']['2560x10240']
    ratios = shape['candidate_over_reference_process_ratios']
    require(shape['component_gain_qualified'] and len(ratios) == 3 and all(r < 1 for r in ratios),
            'require the repeated win for the exact dense projection')
    changed = {}
    for path in PATHS:
        previous = source[path]
        raw = merge_delta(previous, path, DENSE + '^', DENSE)
        require(raw != previous and merge_delta(raw, path, DENSE, DENSE + '^') == previous,
                'dense projection changes unrelated bytes: ' + path)
        changed[path] = raw
    source.update(changed)
    old, new = str(head.ROOT), str(ROOT)
    recipe = dict(configure=[arg.replace(old, new) for arg in recipe['configure']],
                  environment={k: v.replace(old, new) for k, v in recipe['environment'].items()},
                  targets=recipe['targets'])
    proof = dict(proof, dense_source=DENSE, dense_head_evidence=HEAD_EVIDENCE,
                 dense_parent_build_sha256=bank.signature(builder.compact(parent)),
                 overlays=proof['overlays'] | bank.stage.hashed(changed))
    return native, foundation, source, recipe, proof


def context():
    ctx = head.context()
    head_check = ctx.check
    ctx.KEY, ctx.ROOT, ctx.project = KEY, ROOT, project
    ctx.scope = ('One coherent build of the width-five head plus the decode-only Q6 2560x10240 selector. '
                 'Rebuild all verifier consumers and keep the prefill sources, placement and compiler unchanged. '
                 'The model experiment toggles only the dense selector in independent processes.')
    ctx.retention = ('Replace the closed width-five head generation after exact source and binary recovery. '
                     'Keep the width-four speed and physical-context/RESTORE references through the isolated '
                     'dense comparison. Select a retained generation at its first failed gate or closure.')

    def check(row, committed=True):
        if not committed and row['stage'] == 'prepared':
            pins = bank.stage.source.pin(bank.P, [THIS])
            require(pins['revision'] == row['source_commit'], 'pin the dense build controller')
            row['controller_sources'].update(pins['source_sha256'])
        require(THIS in row['controller_sources'], 'missing dense build controller pin')
        return head_check(row, committed)

    ctx.check = check
    return ctx


def policy(ctx):
    row = bank.previous.read(ROOT / 'record.json')
    ctx.check(row)
    require(row['stage'] == 'built' and row['host_control']['passed'] and 'dense_policy_control' not in row,
            'complete the same-build prefix host gate before the route policy check')
    scope = ROOT / 'dense-policy'
    scope.mkdir(mode=0o700)
    owner = builder.ownership.Owner(scope / 'owned', lambda: bank.stage.write(
        scope / 'execution.json', dict(active=bool(owner.active), commands=owner.commands)))
    binary = row['binaries'][head.POLICY_TARGET]
    try:
        entry, out, err = owner.run('verify-dense-policy', [binary['path']], row['recipe']['environment'], bank.P,
            wall=90, cpu=60, rss_cap=1 << 30, text_cap=128 << 10, file_cap=1 << 20, total_cap=4 << 20)
        passed = builder.ownership.closed(entry) and entry['exit_code'] == 0 and not err.read_bytes()
        passed = passed and out.read_text() == head.POLICY_OUT
        row['dense_policy_control'] = dict(passed=passed, binary=binary, exit_code=entry.get('exit_code'),
                                           ownership_closed=builder.ownership.closed(entry), GPU_executed=False)
        require(passed, 'dense route host policy fixture failed')
    except BaseException:
        row.update(stage='failed', passed=False, complete=False)
        raise
    finally:
        row['ownership_closed'] = not owner.active and all(builder.ownership.closed(e) for e in owner.commands)
        ctx.save(row)


def main(action):
    require(sys.argv[1:] == [action] and action in
            ('verify-parent', 'projection', 'prepare', 'configure', 'compile', 'host', 'policy'),
            'select a dense build stage')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        builder.guards.no_model_owners([])
        if action == 'verify-parent':
            *_, parent = parent_inputs()
            head.context().check(parent)
            print({'sources': len(parent['source_files']), 'objects': parent['actual_inputs']['fresh_objects'],
                   'binary': parent['binaries']['strata'], 'build_receipt': bank.signature(builder.compact(parent))})
            return
        ctx = context()
        builder.ROOT, builder.KEY, builder.check, builder.save = ROOT, KEY, ctx.check, ctx.save
        if action == 'projection':
            _, _, source, _, proof = project()
            print({'sources': len(source), 'dense_delta': {p: proof['overlays'][p] for p in PATHS}})
        elif action == 'prepare':
            require(not (head.ROOT / 'build').exists(), 'retire the recovered closed head build first')
            head.base.prepare(ctx)
        elif action == 'host':
            bank.stage.host(root_path=ROOT, check_build=ctx.check, save_record=ctx.save)
        elif action == 'policy':
            policy(ctx)
        else:
            row = bank.previous.read(ROOT / 'record.json')
            builder.TARGETS = row['recipe']['targets']
            builder.execute(action)


if __name__ == '__main__':
    main(*sys.argv[1:])
