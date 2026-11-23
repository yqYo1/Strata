"""Verify prefill-only index-key retention on the measured V2 SYCL engine."""

import fcntl
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import prefill_moe_gate_clock_build as build
import prefill_moe_gate_clock_measure as model
import prefill_retained_score_keys as keys
from official_model_sources import merge_delta

bank, require = build.bank, build.require
THIS = str(Path(__file__).resolve().relative_to(bank.P))
CONTROLS = (THIS, build.THIS, model.THIS,
            *(str(Path(m.__file__).resolve().relative_to(bank.P)) for m in (keys, keys.device)))
BUILD_KEY = 'prefill_score_keys_native_v2'
BUILD_ROOT = bank.B / BUILD_KEY.replace('_', '-')
COMPONENT_KEY = 'prefill_score_keys_component_v2'
COMPONENT_ROOT = bank.B / COMPONENT_KEY.replace('_', '-')
COUNTS = 'e88604bca4535fac04d4f8f61a3dc4469634ddaf'
FIXTURE_OBJECT = f'CMakeFiles/{keys.TARGET}.dir/{keys.FIXTURE}.o'
ROLES = ('smoke', 'check', 'boundary', '0', '1', '2')
PLAN = dict(
    scope='Repack only newly active FP16 index keys inside one prefill SELECT loop. '
          'Reset the extent for every chunk, layer and restored request.',
    roles=list(ROLES),
    correctness='Compare every score, selected ID and guard on both epochs. Require a '
                'detectable stale extent after overwriting the borrowed scratch.',
    performance='Three independent processes, five alternating pairs of 12 chunks '
                'of 2800 queries. Require exact IDs, at least 100 ms per batch, '
                'timer control at most 1%, and a lower paired median in every process.',
    stopping='Stop on changed build or launch inputs, output mismatch, new GPU fault, '
             'ineligible interval, owner failure or finite budget exhaustion.',
)


def apply_delta(source, path, before, after):
    original = source[path]
    changed = merge_delta(original, path, before, after)
    require(merge_delta(changed, path, after, before) == original,
            'score-key projection changes unrelated source: ' + path)
    source[path] = changed


def build_context(revision):
    ctx = build.upload.context()
    ctx.ROOT, ctx.KEY = BUILD_ROOT, BUILD_KEY
    ctx.scope = 'Score-key retention and its checked same-build indexer fixture on the exact V2 source.'
    ctx.retention = ('Keep the V2 prefill and coherent decode references through the component and '
                     'eligible model comparisons. Review this third generation after their closure.')

    def project(expected=None):
        native, foundation, source, recipe, proof, parent = build.retained_v2_projection(expected)
        controls = bank.stage.source.blobs(bank.P, revision, CONTROLS)
        require(all((bank.P / name).read_bytes() == raw for name, raw in controls),
                'use the committed score-key controller and helpers')
        paths = (keys.HEADER, keys.ADMISSION, *keys.UNITS)
        for path in paths:
            apply_delta(source, path, keys.BASE, keys.SOURCE)
        apply_delta(source, keys.CALLER, COUNTS + '^', COUNTS)
        apply_delta(source, keys.FIXTURE, keys.BOUNDARY_SOURCE + '^', keys.BOUNDARY_SOURCE)

        objects = sorted(Path(p).relative_to(build.upload.ROOT / 'build').as_posix()
                         for p in parent['actual_inputs']['objects'])
        require(bank.signature(objects) == proof['object_set_sha256'] and FIXTURE_OBJECT not in objects
                and keys.TARGET not in recipe['targets'], 'unexpected V2 objects or score fixture target')
        recipe = build.recipe_at(recipe, BUILD_ROOT)
        recipe['targets'] = [*recipe['targets'], keys.TARGET]
        overlays = bank.stage.hashed({p: source[p] for p in paths})
        proof = dict(proof, score_key_source=keys.SOURCE, indexer_boundary_source=keys.BOUNDARY_SOURCE,
                     score_dispatch_source=COUNTS, score_parent_build_sha256=bank.signature(bank.builder.compact(parent)),
                     object_count=len(objects) + 1, object_set_sha256=bank.signature(sorted([*objects, FIXTURE_OBJECT])),
                     overlays=proof['overlays'] | overlays)
        return native, foundation, source, recipe, proof

    ctx.project = project
    return ctx


def component_inputs():
    native = bank.previous.read(BUILD_ROOT / 'record.json')
    build_context(native['source_commit']).check(native)
    require(native['stage'] == 'built' and native['passed'] and native['complete']
            and native['ownership_closed'] and not native['active'] and native['host_control']['passed'],
            'complete the same-build score-key engine, fixture and host controls')
    parent, prior, _ = model.retained_v2_model()
    timeline = bank.previous.read(bank.B / model.TIMELINE_KEY.replace('_', '-') / 'record.json')
    recorded = bank.previous.historic(model.TIMELINE, model.TIMELINE_KEY)
    require(model.basis.compact(timeline) == recorded and native['driver'] == parent['driver'] == prior['driver']
            and native['projection']['score_parent_build_sha256'] == bank.signature(bank.builder.compact(parent)),
            'use the verified V2 launch settings and providers')
    binary = native['binaries'][keys.TARGET]
    inputs = dict(native_source_commit=native['native_source_commit'],
                  build_sha256=bank.signature(bank.builder.compact(native)), binary=binary,
                  timeline=model.TIMELINE, timeline_sha256=bank.signature(recorded),
                  environment_template_sha256=recorded['environment_template_sha256'],
                  runtime_manifest=recorded['runtime_manifest'], boot=recorded['boot'])
    require(bank.signature(timeline['environment_template']) == inputs['environment_template_sha256'],
            'local V2 environment template changed')
    launches = {role: bank.signature(keys.fixture_launch(role, binary, timeline, COMPONENT_ROOT)) for role in ROLES}
    return native, timeline, inputs, launches


def component(action):
    native, timeline, inputs, launches = component_inputs()
    result = bank.basis.validation.committed(bank.RESULT)
    if action == 'prepare':
        require(COMPONENT_KEY not in result and not COMPONENT_ROOT.exists(), 'inspect the existing score-key plan')
        result[COMPONENT_KEY] = dict(plan=PLAN, inputs=inputs, launches=launches, stages=[], stage='ready')
        bank.stage.write(bank.RESULT, result)
        return

    row = result[COMPONENT_KEY]
    require(row['stage'] == 'ready' and row['plan'] == PLAN and row['inputs'] == inputs and row['launches'] == launches
            and [s['role'] for s in row['stages']] == list(ROLES[:ROLES.index(action)])
            and all(s['passed'] and s['ownership_closed'] for s in row['stages'])
            and not (COMPONENT_ROOT / ('run-' + action)).exists(), 'wrong score-key stage or changed plan')
    previous = bank.signature(row)
    stage = dict(role=action, passed=False)
    try:
        keys.measure(action, inputs['binary'], native, timeline, stage, COMPONENT_ROOT,
                     launch_sha256=launches[action])
        require(stage['passed'] and stage['native_exit_code'] == 0, 'score-key fixture device gate failed')
    except BaseException as error:
        stage['error'] = repr(error)
        row['stage'] = 'failed'
        raise
    finally:
        row['stages'].append(stage)
        if action == ROLES[-1] and row['stage'] != 'failed':
            ratios = [s['median_paired_time_ratio'] for s in row['stages'][-3:]]
            row.update(stage='closed', decision=dict(paired_time_ratios=ratios,
                        component_gain=all(ratio < 1 for ratio in ratios), model_tested=False))
        current = bank.previous.read(bank.RESULT)
        require(bank.signature(current[COMPONENT_KEY]) == previous, 'score-key record changed during execution')
        current[COMPONENT_KEY] = row
        bank.stage.write(bank.RESULT, current)


if __name__ == '__main__':
    phase, action = sys.argv[1:]
    require(sys.executable == bank.basis.reference.model.PYTHON, 'use the pinned benchmark interpreter')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        if phase == 'build':
            require(action in ('projection', 'prepare', 'configure', 'compile', 'host'), 'select a build action')
            record = BUILD_ROOT / 'record.json'
            revision = (bank.previous.read(record)['source_commit'] if record.exists() else
                        bank.basis.validation.git('rev-parse', 'HEAD').decode().strip())
            build.run_stage(build_context(revision), revision, action, CONTROLS)
        else:
            require(phase == 'component' and action in ('prepare', *ROLES), 'select a component action')
            component(action)
