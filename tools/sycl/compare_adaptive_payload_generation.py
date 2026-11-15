#!/usr/bin/env python3
"""Compare8/32replacement budgets on the qualified one-header successor.

Main-authored; repository LICENSE. Reuse benchmark/decision semantics,not
physical predecessor binaries. The new positive/refusal controls must pass.
"""
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import check_adaptive_payload_failure as refusal
import compare_adaptive_budget_generation as original

clean = original.clean
SELF = Path(__file__).resolve()
require = original.require


def prerequisites(full=False):
    ctx = refusal.context()
    original.refusal.budget.prerequisite_rows(ctx)
    data = clean.base.committed(clean.RESULT)
    port = data['upstream_streamed_tool_tags_host_v2']
    require(port['stage'] == 'closed' and port['passed'] and port['complete'] and
        port['ownership_closed'] and not port['active'] and not port['GPU_executed'] and
        port['origin_commit'] == '3bbf0f399e2a1b3d38642eca4342721fc7994201' and
        all(port['sources'][n] == clean.digest(clean.P / n) for n in original.FRONTEND_FILES) and
        len(port['tests']) >= 3 and all(t['exit_code'] == 0 for t in port['tests']),
        'required upstream frontend behavior must stay qualified')
    for name in original.FRONTEND_FILES:
        require((clean.P / name).read_bytes() == clean.base.basis.build.git(clean.P, 'show',
            port['source_commit'] + ':' + name), 'recover exact tested frontend: ' + name)
    normal = ctx.normal_entry()
    failed = json.loads((ctx.ROOT / 'record.json').read_text())
    require(data[ctx.positive.KEY] == ctx.positive.compact(normal) and
        data[ctx.KEY] == ctx.compact(failed) and failed['stage'] == 'closed' and
        failed['passed'] and failed['complete'] and failed['ownership_closed'] and
        not failed['active'] and failed['outcome']['native_exit_code'] == 1 and
        failed['outcome']['adaptive']['first_swaps'] == 32 and
        failed['outcome']['adaptive']['queued_second_swaps'] == 32 and
        failed['outcome']['adaptive']['stale_event_not_published'] and
        normal['binary'] == failed['binary'] and
        normal['native_source_commit'] == failed['native_source_commit'],
        'closed successor32positive/checked-refusal pair required')
    providers = dict(ctx.positive.current(normal, full=full))
    ctx.current(failed)
    # Reuse settings and the qualified frontend interpreter only. Native
    # qualification comes from the successor above,not the old width engine.
    prior = json.loads((ctx.positive.width.SPEED_ROOT / 'record.json').read_text())
    contract, manifest = ctx.positive.width.context(4).python_manifest()
    require(contract == prior['python_contract'] and manifest['selected'] == prior['python_selected'] and
        normal['model_assets'] == prior['model_assets'], 'frontend runtime/model assets changed')
    probe = normal['native_validation_probe']['library']; providers[probe['path']] = probe['sha256']
    refs = {k: dict(path=str(root / 'record.json'), sha256=clean.digest(root / 'record.json'))
        for k, root in (('positive32', ctx.positive.ROOT), ('record_failure32', ctx.ROOT))}
    refs['frontend'] = dict(key='upstream_streamed_tool_tags_host_v2', sha256=clean.signature(port))
    return normal, prior, contract, manifest, providers, refs


def template():
    spec = importlib.util.spec_from_file_location('strata_payload_budget_generation', original.__file__)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def controls():
    module = template()
    module.context = context
    return module.controls()


def context(workload):
    module = template()
    module.SELF, module.prerequisites, module.controls, module.prepare = SELF, prerequisites, controls, prepare
    ctx = module.context(workload)
    ctx.ROOT = clean.B / ('adaptive-payload-generation-' + workload + '-v1')
    ctx.KEY = 'adaptive_payload_generation_' + workload.replace('-', '_') + '_v1'
    ctx.WORKLOAD_ID = 'adaptive_payload_generation_20261107_' + workload
    prior_current = ctx.current

    def current(row):
        require(row['successor_positive_key'] == refusal.positive.KEY and
            row['successor_refusal_key'] == refusal.KEY,
            'successor qualification keys changed')
        return prior_current(row)

    ctx.current = current
    return ctx


def prepare(module, ctx):
    original.prepare(module, ctx)
    row = json.loads((ctx.ROOT / 'record.json').read_text())
    try:
        require(row['plan'].count('on unchanged429b') == 1, 'budget speed plan anchor changed')
        row['plan'] = row['plan'].replace('on unchanged429b', 'on the same qualified payload successor')
        row['plan'] += (' Native qualification is the successor32positive/refusal pair;the old '
            'settings/frontend contract supplies no old-engine qualification. Preserve native '
            'build/positive/refusal receipt bytes consumed by later gates;record raw retirement '
            'separately rather than changing those hashes. Retire429b after consumer migration '
            'and verified source recovery,before these clean trials. No heavy resource observers.')
        row.update(successor_positive_key=refusal.positive.KEY, successor_refusal_key=refusal.KEY)
    except BaseException as error:
        row.update(stage='failed-successor-prepare', passed=False, complete=False, error=repr(error))
        raise
    finally: ctx.save(row)


if __name__ == '__main__':
    args = sys.argv[1:]
    require(args == ['controls'] or len(args) == 2 and args[0] in clean.CAPS and
        (args[1] in ('prepare', 'host') or args[1].isdigit() and int(args[1]) < 6),
        'use controls | regular-code|varied-prose prepare|host|INDEX0..5')
    with (clean.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        clean.positive.fcntl.flock(lock, clean.positive.fcntl.LOCK_EX | clean.positive.fcntl.LOCK_NB)
        clean.base.guards.no_model_owners([])
        if args == ['controls']:
            revision = clean.base.basis.build.git(clean.P, 'rev-parse', 'HEAD').decode().strip()
            pins = clean.base.controls(revision, controller=SELF)
            result = dict(controls(), source_commit=revision, source_count=len(pins),
                source_signature_sha256=clean.signature(pins))
            print(json.dumps(result, sort_keys=True))
        else:
            ctx = context(args[0])
            if args[1] == 'prepare': ctx.prepare_budget()
            elif args[1] == 'host': ctx.comparison.host(ctx)
            else: ctx.run(int(args[1]))
