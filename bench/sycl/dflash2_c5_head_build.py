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
POLICY_KEY = 'dflash2_c5_head_policy_v2'
POLICY_ROOT = ROOT / 'head-policy-v2'
POLICY_REFUSAL = '1315025aaefe4af60709eec49fd8afca588a2169'
POLICY_OUT = ('verify-window-policy: passed selections=30 centering=18 native-half=26 dpas-head=1 '
              'dpas-dense=4 phase-roundtrip=1 invalid-keys=4 retirement=1 route-budget=64\n')
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
    ctx.retention = ('Retain 96832af6f as the validated width-four speed reference and 4b4ac554c for '
                     'the separate physical-context, RESTORE and settled fixed-width controls. The third '
                     'comparison build is for the head probe and matched width-five trial. At that decision '
                     'select the validated generation and retire the superseded comparison build with its '
                     'original sources in Git. The historical 14508acbf lineage reader still consumes its '
                     'original build inputs and carries its existing provenance hold separately.')

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


def save_policy(row):
    bank.stage.write(POLICY_ROOT / 'record.json', row)
    data = bank.previous.read(bank.RESULT)
    data[POLICY_KEY] = row
    bank.stage.write(bank.RESULT, data)


def head_policy(ctx, prepare):
    compiled = bank.previous.read(ROOT / 'record.json')
    ctx.check(compiled)
    earlier = bank.previous.historic(POLICY_REFUSAL, KEY)
    compact = builder.compact(compiled)
    require(compact == earlier and not earlier['head_policy_control']['passed']
            and compiled['stage'] == 'built' and compiled['host_control']['passed'],
            'keep the original failed head check and successful prefill control')
    binary = compiled['binaries'][POLICY_TARGET]
    identity = dict(binary=binary, native_source_commit=compiled['native_source_commit'],
                    build_sha256=bank.signature(compact), environment_sha256=bank.signature(compiled['recipe']['environment']))
    if prepare:
        require(not POLICY_ROOT.exists() and POLICY_KEY not in bank.basis.validation.committed(bank.RESULT),
                'fresh host-policy supervision plan required')
        pins = bank.stage.source.pin(bank.P, [THIS])
        prior_budget_scope = builder.ownership.artifact_bytes(ROOT)
        require(prior_budget_scope > (4 << 20), 'recover the original whole-build output-budget conflict')
        plan = dict(**identity, source_commit=pins['revision'], controller_sha256=pins['source_sha256'][THIS],
                    original=POLICY_REFUSAL, old_scope_bytes_at_review=prior_budget_scope,
                    output_scope=str(POLICY_ROOT), expected=POLICY_OUT,
                    budgets=dict(wall=90, cpu=60, rss_cap=1 << 30, text_cap=128 << 10,
                                 file_cap=1 << 20, total_cap=4 << 20))
        POLICY_ROOT.mkdir(mode=0o700)
        save_policy(dict(stage='prepared', plan=plan, passed=False, ownership_closed=True))
        return

    row = bank.previous.read(POLICY_ROOT / 'record.json')
    plan = row['plan']
    execution = POLICY_ROOT / 'execution.json'
    path = bank.P / THIS
    require(row == bank.basis.validation.committed(bank.RESULT, (POLICY_KEY,)) and row['stage'] == 'prepared'
            and all(plan[k] == v for k, v in identity.items())
            and plan['output_scope'] == str(POLICY_ROOT) and plan['expected'] == POLICY_OUT
            and bank.sha(path) == plan['controller_sha256']
            and path.read_bytes() == bank.stage.source.git(bank.P, 'show', plan['source_commit'] + ':' + THIS)
            and not execution.exists(), 'commit the independent host-policy check and preserve its inputs')
    owner = builder.ownership.Owner(POLICY_ROOT / 'owned', lambda: bank.stage.write(
        execution, dict(active=bool(owner.active), commands=owner.commands)))
    try:
        entry, out, err = owner.run('verify-five-column-policy', [binary['path']], compiled['recipe']['environment'],
                                    bank.P, **plan['budgets'])
        row.update(exit_code=entry.get('exit_code'), supervisor_error=entry.get('error'))
        require(builder.ownership.closed(entry) and entry['exit_code'] == 0 and not err.read_bytes()
                and out.read_text() == plan['expected'], 'head policy fixture failed')
        row.update(stage='closed', passed=True, result=out.read_text().strip(), GPU_executed=False)
    except BaseException as error:
        row.update(stage='failed', passed=False, error=repr(error))
        raise
    finally:
        row['ownership_closed'] = (not owner.active and bool(owner.commands)
            and all(builder.ownership.closed(cmd) for cmd in owner.commands))
        save_policy(row)


def main(action):
    require(action in ('projection', 'prepare', 'configure', 'compile', 'host', 'prepare-head-policy', 'head-policy'),
            'select a build stage')
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
        elif action in ('prepare-head-policy', 'head-policy'):
            head_policy(ctx, prepare=action == 'prepare-head-policy')
        else:
            record = bank.previous.read(ROOT / 'record.json')
            builder.TARGETS = record['recipe']['targets']
            builder.execute(action)


if __name__ == '__main__':
    main(*sys.argv[1:])
