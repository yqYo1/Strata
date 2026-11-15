#!/usr/bin/env python3
"""Fresh full-engine build for the qualified cooperative KV component.

Main-authored from this fork's coherent build controller; repository LICENSE.
Builds every selected object again; does not execute GPU or model workloads.
"""
import copy
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shlex
import sys

sys.dont_write_bytecode = True
import check_qsa16_stream as base
import build_coherent_prefill as coherent
import decode_window_bench as reference
import fixture_build as fixture

P, B, RESULT = base.P, base.B, base.RESULT
ROOT, KEY = B / 'decode-kv-native-v1', 'decode_kv_native_v1'
SEED, SEED_KEY = reference.reuse.ROOT / 'record.json', reference.reuse.KEY
TARGETS = [*coherent.TARGETS, 'sycl_expert_dispatch_bounds_test']
HEADERS = {
    'include/strata/prefill/gemm.hpp': ('STRATA_ENABLE_SYCL',),
    'include/strata/core/verify.hpp': ('STRATA_ENABLE_SYCL',),
    'include/strata/core/verify_window_policy.hpp': (),
    'include/strata/kernels/cpu/pool.hpp': ('STRATA_CPU_SERVICE_PROFILE',),
    'include/strata/core/expert_source.hpp': ('STRATA_ENABLE_SYCL',),
}
require, sha, signature = base.require, base.sha, fixture.signature


def prerequisite():
    data = json.loads(base.common.source.git(P, 'show', 'HEAD:' + str(RESULT.relative_to(P))))
    component = data['kv_planner_speed_v1']
    host = data['decode_kv_policy_host_v2']
    for row in (component, host):
        require(row['stage'] == 'closed' and not row['active']
            and all(row[k] for k in ('passed', 'complete', 'ownership_closed')),
            'closed component and actual-policy host evidence required')
    require(component['decision']['eligible_widths'] == [2, 4]
        and len(component['decision']['cells']) == 6
        and all(c['reproducible_gain'] and len(c['paired_time_ratios']) == 3
            and all(0 < x < 1 for x in c['paired_time_ratios'])
            for c in component['decision']['cells'])
        and not component['decision']['production_adopted'],
        'preserve the actual six-cell component performance gate')
    image = component['binary']
    require(Path(image['path']).stat().st_size == image['bytes']
        and sha(image['path']) == image['sha256'], 'retained component ELF changed')
    require(len(host['outcomes']) == 4 and all(c['exit_code'] == 0
        and c['ownership_closed'] for c in host['outcomes'])
        and not host['GPU_executed'] and not host['model_executed'],
        'actual policy and Verifier syntax evidence required')
    unchanged = {}
    for source, paths in (
        (component['native_source_commit'], (
            'src/kernels/sycl/kv_stream.cpp',
            'include/strata/kernels/kv_stream_copy_groups.hpp')),
        (host['project']['revision'], (
            'src/core/verify.cpp', 'include/strata/core/verify.hpp',
            'include/strata/core/verify_window_policy.hpp',
            'tests/core/verify_window_policy_test.cpp')),
    ):
        for path, raw in base.common.source.blobs(P, source, paths):
            h = hashlib.sha256(raw).hexdigest()
            require(sha(P / path) == h, 'tested kernel/policy source changed: ' + path)
            unchanged[path] = h
    retirement = component['production_preparation']['build_retirement']
    require(retirement['state'] == 'retired' and retirement['complete']
        and retirement['inventory_retired'] and len(retirement['entries']) == 10,
        'close normal ended build consumers before the successor generation')
    for name in retirement['entries']:
        require(sorted(p.name for p in (B / name).iterdir())
            == sorted(p.name for p in (B / name).iterdir()
                if p.name in ('record.json', 'actual-inputs.json', 'external-inputs.json')),
            'ended products reappeared: ' + name)
    return dict(component_source=component['native_source_commit'], component_binary=image,
        component_decision_sha256=signature(component['decision']),
        host_source=host['project']['revision'], host_outcomes=host['outcomes'],
        tested_native_inputs=unchanged, normal_build_retirement_complete=True,
        legacy_provenance_holds=retirement['legacy_remaining'],
        inherited_engine_GPU_model_speed_or_full_context_qualification=False)


def definition_census(row, output):
    database = json.loads((ROOT / 'build/compile_commands.json').read_text())
    commands = {}
    for item in database:
        args = shlex.split(item['command'])
        if '-o' in args:
            p = Path(args[args.index('-o') + 1])
            commands[str((p if p.is_absolute() else Path(item['directory']) / p).resolve())] = args
    headers = {str(P / name): name for name in HEADERS}
    consumers = {name: {} for name in HEADERS}
    obj = None
    for line in output.read_text().splitlines():
        if line and not line[0].isspace():
            obj = str((ROOT / 'build' / line.split(': #deps ', 1)[0]).resolve())
        elif obj and line.startswith('    '):
            p = Path(line.strip())
            name = headers.get(str((p if p.is_absolute() else ROOT / 'build' / p).resolve()))
            if name is None:
                continue
            require(obj in commands, 'actual class includer missing from compile database')
            args = commands[obj]
            conditions = {}
            for macro in HEADERS[name]:
                definitions = [x for x in args if x == '-D' + macro or x.startswith('-D' + macro + '=')]
                require(definitions == ['-D' + macro + '=1']
                    and '-U' + macro not in args, 'inconsistent class condition: ' + name)
                conditions[macro] = '1'
            require(not any(x.startswith('-D_WIN32') for x in args), 'unexpected Windows class condition')
            consumers[name][str(Path(obj).relative_to(ROOT / 'build'))] = conditions
    result = {}
    for name, entries in consumers.items():
        text = (P / name).read_text()
        directives = '\n'.join(re.findall(r'^\s*#\s*(?:if|ifdef|ifndef|elif)\b([^\n]*)', text, re.M))
        require(set(re.findall(r'\bSTRATA_[A-Z0-9_]+\b', directives)) == set(HEADERS[name])
            and entries, 'review the actual header class conditions: ' + name)
        result[name] = dict(header_sha256=sha(P / name), includers=len(entries),
            includers_sha256=signature(entries), conditions={m: '1' for m in HEADERS[name]},
            all_objects_from_fresh_build=True)
    require(result['include/strata/prefill/gemm.hpp']['includers'] >= 5,
        'actual Gemm implementation, wrapper and fixture closure required')
    return dict(headers=result, partial_archive_reuse=False, fresh_complete_build=True)


def context():
    spec = importlib.util.spec_from_file_location('decode_kv_build_context', base.__file__)
    ctx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ctx)
    ctx.ROOT, ctx.KEY, ctx.TARGETS, ctx.MODEL_BUILD = ROOT, KEY, TARGETS, True
    ctx.SEED, ctx.SEED_KEY, ctx.component_reference = SEED, SEED_KEY, prerequisite
    ctx.common.PREFIXES = fixture.PREFIXES
    original = ctx.context
    def native_context():
        native = original()
        parse, census = native.planned_compilation, native.census
        native.planned_compilation = lambda text: parse(text, allowed_duplicate_targets=coherent.DUPLICATES)
        def checked_census(row, dep, link, **kwargs):
            value = census(row, dep, link, **kwargs)
            value['class_definition_coherence'] = definition_census(row, dep)
            return value
        native.census = checked_census
        return native
    ctx.context = native_context
    return ctx


def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()), 'fresh engine attempt only')
    native, proof = reference.retained_native()
    project = base.common.source.pin(P, fixture.PREFIXES)
    recipe = copy.deepcopy(native['recipe'])
    old = str(SEED.parent)
    recipe['configure'] = [str(ROOT) + v[len(old):] if v.startswith(old) else v
        for v in recipe['configure']]
    recipe['configure'][recipe['configure'].index('-S') + 1] = str(P)
    recipe['environment'] = {k: v.replace(old, str(ROOT)) for k, v in recipe['environment'].items()}
    recipe['environment']['STRATA_SYCL_DECODE_KV_COALESCED'] = '0'
    recipe['targets'] = recipe['build_targets'] = TARGETS
    require(recipe['configure'][recipe['configure'].index('-B') + 1] == str(ROOT / 'build'),
        'all objects must be freshly compiled in the new root')
    probe, validation = base.probe_qualification()
    row = dict(stage='prepared', project=project, dependencies=copy.deepcopy(native['dependencies']),
        recipe=recipe, recipe_seed=dict(path=str(SEED), sha256=sha(SEED)),
        toolchain=native['toolchain'], external_inputs=native['external_inputs'],
        native_source_commit=project['revision'], preserved_binaries=list(native['binaries'].values()),
        native_validation_probe=probe, validation_qualification=validation,
        build_prerequisite=prerequisite(), retained_reference=proof,
        duplicate_source_admission=coherent.DUPLICATES,
        plan='Commit every stage before configure/compile. Fresh full engine and13selected targets,zero project object/archive reuse. Preserve the qualified14508engine and12same-build ELFs. Recover exact source/dependencies/SDK/compiler/recipe;require complete actual compiler/link census and consistent class conditions across Gemm,Verifier,Window,Pool,ExpertSource includers. No GPU/model execution or inherited qualification. Next commit traced/numerical native and production-route checks,then a separate matched decode-only model plan preserving prefill and physical262144/RESTORE obligations.',
        retention='One validated14508reference and one imminent decode-KVcandidate generation;standalone1e776component retained only through this model decision. Ten normal ended groups are retired;three explicitly recorded special-format provenance holds remain separate and must be reconciled at build closure. Never overwrite the reference;retire rejected candidate/component after exact source/recipe recovery.',
        budgets=dict(build=dict(wall=2400, cpu=7200, rss_cap=12 << 30,
            text_cap=32 << 20, file_cap=512 << 20, total_cap=4 << 30)),
        commands=[], outcomes={}, passed=False, complete=False, active=False,
        ownership_closed=True, GPU_executed=False, model_qualified=False,
        physical262144_model_qualified=False, performance_qualified=False)
    ROOT.mkdir(mode=0o700)
    for name in ('tmp', 'cache'):
        (ROOT / name).mkdir(mode=0o700)
    ctx = context()
    ctx.current(row, committed=False)
    ctx.save(row)
    print('Prepared fresh full-engine KVcandidate;commit before configure. No GPU executed.', flush=True)


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'configure', 'compile'),
        'use prepare|configure|compile')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base.guards.no_model_owners([])
        if sys.argv[1] == 'prepare':
            prepare()
        else:
            context().build(sys.argv[1])
