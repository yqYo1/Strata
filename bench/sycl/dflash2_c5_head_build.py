"""Project just the width-five Q6 head onto the measured DFlash2 engine."""

import fcntl
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import dflash2_target_snapshot_build as snapshot
from official_model_sources import merge_delta, once

base, bank, builder, require = snapshot.base, snapshot.bank, snapshot.builder, snapshot.require
KEY = 'dflash2_c5_head_native_v1'
ROOT = bank.B / KEY.replace('_', '-')
THIS = str(Path(__file__).resolve().relative_to(bank.P))
C5_EVIDENCE = 'f282d174d61ddd3162de74f106420974d8903377'
C5_INITIAL = 'ac3ba9d8501ecf88181fe6ec883120f532ec34d2'
C5_SOURCE = 'b06064eadf1331104f48883c21425a296648392e'
POLICY_TARGET = 'sycl_verify_window_policy_test'
POLICY_TEST = 'tests/core/verify_window_policy_test.cpp'
VERIFY = 'src/core/verify.cpp'
KERNEL_INPUTS = ('include/strata/core/verify_window_policy.hpp',
                 'src/kernels/sycl/q6_dpas_component.cpp',
                 'src/kernels/sycl/q6_dpas_component.hpp')


def region(source, start, end):
    require(source.count(start) == 1, 'ambiguous C5 source region')
    first = source.index(start)
    return source[first:source.index(end, first)]


def project_c5(files):
    """Apply the exact owned C5 changes while keeping the older verifier base."""
    def blob(revision, path):
        return bank.stage.source.git(bank.P, 'show', revision + ':' + path)

    changed = {}
    for path in KERNEL_INPUTS:
        selected = merge_delta(files[path], path, C5_INITIAL, C5_SOURCE)
        require(merge_delta(selected, path, C5_SOURCE, C5_INITIAL) == files[path],
                'C5 would change unrelated source: ' + path)
        changed[path] = selected

    old, new = (blob(rev, VERIFY) for rev in (C5_INITIAL, C5_SOURCE))
    helper = region(new, b'bool decode_q6_c5_head_enabled()', b'bool decode_clock_geometry(')
    anchor = b'#ifdef STRATA_ENABLE_SYCL\nusing CacheReader = '
    patches = [(anchor, b'#ifdef STRATA_ENABLE_SYCL\n' + helper + b'using CacheReader = ')]
    for old_start, new_start, end in (
        (b'            (window.q6_dpas_head_calls==0', b'            (window.q6_dpas_head_calls == 0',
         b'            (window.q6_packed_head_calls'),
        (b'                } else if(verify_q6_dpas_head(', b'                } else if (verify_q6_dpas_head(',
         b'                    ++graph_window.q6_dpas_head_calls;'),
    ):
        patches.append((region(old, old_start, end), region(new, new_start, end)))
    selected = files[VERIFY]
    for before, after in patches:
        selected = once(selected, before, after)
    reverted = selected
    for before, after in reversed(patches):
        reverted = once(reverted, after, before)
    require(reverted == files[VERIFY], 'C5 would change unrelated verifier source')
    changed[VERIFY] = selected

    old_test, new_test = (blob(rev, POLICY_TEST) for rev in (C5_INITIAL, C5_SOURCE))
    end = b'    // Independent expected set from e44575158'
    old_test = region(old_test, b'    // Only the independently measured winning head', end)
    new_test = region(new_test, b'    for (bool opt_in_c5', end)
    changed[POLICY_TEST] = once(files[POLICY_TEST], old_test, new_test)
    require(once(changed[POLICY_TEST], new_test, old_test) == files[POLICY_TEST],
            'C5 would change unrelated host controls')
    return changed


def project(expected=None):
    native, foundation, files, recipe, proof = snapshot.project(expected)
    parent = bank.previous.read(snapshot.ROOT / 'record.json')
    parent_files = bank.stage.hashed(files)
    require(parent['stage'] == 'built' and parent['passed'] and parent['complete'] and not parent['active']
            and parent['ownership_closed'] and parent['host_control']['passed']
            and parent['source_files'] == parent_files and parent['recipe'] == recipe
            and parent['projection'] == proof and parent['foundation'] == foundation
            and builder.compact(parent) == bank.basis.validation.committed(bank.RESULT, (snapshot.KEY,)),
            'recover the exact compiled DFlash2 target-snapshot source and recipe')
    component = bank.previous.historic(C5_EVIDENCE, 'q6_canonical_c5_component_check_v2')
    head = component['decision']['cells']['2560x248320']
    require(component['stage'] == 'closed' and not component['active'] and head['component_gain_qualified']
            and len(head['candidate_over_reference_process_ratios']) == 3
            and all(v < 1 for v in head['candidate_over_reference_process_ratios'])
            and all(r['passed'] and r['ownership_closed'] and not r['fresh_faults'] for r in component['outcomes']),
            'the repeated canonical C5 full-head component gate must pass')

    changed = project_c5(files)
    require(all(raw != files[path] for path, raw in changed.items()), 'empty width-five source change')
    files.update(changed)
    objects = {Path(p).relative_to(snapshot.ROOT / 'build').as_posix() for p in parent['actual_inputs']['objects']}
    policy_object = f'CMakeFiles/{POLICY_TARGET}.dir/{POLICY_TEST}.o'
    require(len(objects) == proof['object_count'] and bank.signature(sorted(objects)) == proof['object_set_sha256']
            and policy_object not in objects and POLICY_TARGET not in recipe['targets'],
            'recover the full original compiler object set')
    objects.add(policy_object)
    old, new = str(snapshot.ROOT), str(ROOT)
    recipe = dict(configure=[arg.replace(old, new) for arg in recipe['configure']],
                  environment={name: value.replace(old, new) for name, value in recipe['environment'].items()},
                  targets=[*recipe['targets'], POLICY_TARGET])
    proof = dict(proof, c5_source=C5_SOURCE, c5_component_evidence=C5_EVIDENCE,
                 c5_parent_source_sha256=bank.signature(parent_files),
                 c5_parent_build_sha256=bank.signature(builder.compact(parent)),
                 overlays=proof['overlays'] | bank.stage.hashed(changed),
                 object_count=len(objects), object_set_sha256=bank.signature(sorted(objects)))
    return native, foundation, files, recipe, proof


def context():
    ctx = base.context('dflash2-kv-stage')
    check_build = ctx.check
    ctx.KEY, ctx.ROOT, ctx.project = KEY, ROOT, project
    ctx.scope = ('Compile the existing DFlash2 target-snapshot projection plus the width-five decode head and '
                 'its policy controls. Use the same prefill source, weights, placement and compiler configuration. '
                 'Rebuild every translation unit in the original targets and the added host policy fixture.')
    ctx.retention = ('Keep the validated width-four incumbent and exact target-snapshot controls through the '
                     'first head checks and matched width-five model trial. Review the candidate and superseded '
                     'snapshot at the trial decision. Keep original source and decisions in Git.')

    def check(row, committed=True):
        if not committed and row['stage'] == 'prepared':
            controllers = {THIS, snapshot.THIS}
            for mod in tuple(sys.modules.values()):
                filename = getattr(mod, '__file__', None)
                if filename and (path := Path(filename).resolve()).is_relative_to(bank.P / 'bench' / 'sycl'):
                    controllers.add(str(path.relative_to(bank.P)))
            pins = bank.stage.source.pin(bank.P, sorted(controllers))
            require(pins['revision'] == row['source_commit'] and controllers <= pins['source_sha256'].keys(),
                    'commit all loaded build controllers and helpers')
            row['controller_sources'].update(pins['source_sha256'])
        require(THIS in row['controller_sources'] and snapshot.THIS in row['controller_sources'],
                'pin both the C5 projection and its target-snapshot basis')
        return check_build(row, committed)

    ctx.check = check
    return ctx


def head_policy(ctx):
    row = bank.previous.read(ROOT / 'record.json')
    ctx.check(row)
    require(row['stage'] == 'built' and row['host_control']['passed'] and 'head_policy_control' not in row,
            'commit the full build and its preserved prefill host controls')
    owner = builder.ownership.Owner(ROOT / 'owned-policy')
    binary = row['binaries'][POLICY_TARGET]
    try:
        entry, out, err = owner.run('verify-five-column-policy', [binary['path']], row['recipe']['environment'],
                                     bank.P, wall=90, cpu=60, rss_cap=1 << 30, text_cap=128 << 10,
                                     file_cap=1 << 20, total_cap=4 << 20)
        require(builder.ownership.closed(entry) and entry['exit_code'] == 0 and not err.read_bytes()
                and out.read_text().startswith('verify-window-policy: passed '), 'head policy fixture failed')
        row['head_policy_control'] = dict(passed=True, exit_code=0, binary=binary,
                                         result=out.read_text().strip(), GPU_executed=False)
    except BaseException as error:
        row['head_policy_control'] = dict(passed=False, error=repr(error))
        raise
    finally:
        row['head_policy_control']['ownership_closed'] = (not owner.active and bool(owner.commands)
            and all(builder.ownership.closed(entry) for entry in owner.commands))
        ctx.save(row)


def main(action):
    require(action in ('projection', 'prepare', 'configure', 'compile', 'host', 'head-policy'), 'select a build stage')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        builder.guards.no_model_owners([])
        ctx = context()
        builder.ROOT, builder.KEY, builder.check, builder.save = ROOT, KEY, ctx.check, ctx.save
        if action == 'projection':
            _, _, files, recipe, proof = ctx.project()
            print({'sources': len(files), 'targets': recipe['targets'], 'object_count': proof['object_count']})
        elif action == 'prepare':
            base.prepare(ctx)
        elif action == 'host':
            bank.stage.host(root_path=ROOT, check_build=ctx.check, save_record=ctx.save)
        elif action == 'head-policy':
            head_policy(ctx)
        else:
            record = bank.previous.read(ROOT / 'record.json')
            builder.TARGETS = record['recipe']['targets']
            builder.execute(action)


if __name__ == '__main__':
    main(*sys.argv[1:])
