#!/usr/bin/env python3
"""Fresh full-engine build after the mixed Gemm definition finding.

Main-authored from this fork's original full-build recipe; repository LICENSE.
No native archive/object is reused and no GPU execution is performed here.
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
import build_moe_census_model as previous

P, B, RESULT = base.P, base.B, base.RESULT
ROOT, KEY = B/'coherent-prefill-native-v1', 'coherent_prefill_native_v1'
SEED, SEED_KEY = previous.ROOT/'record.json', previous.KEY
TARGETS = [*previous.TARGETS, 'strata_full_context_diagnostic',
           'full_context_diagnostic_host_test', 'sycl_gu_policy_test']
DUPLICATES = {
    str(P/'src/prefill/sycl/qsa_native16.cpp'):
        ['strata_prefill', 'sycl_native16_qsa_test'],
    **{str(P/('src/program/'+name)):
       ['strata', 'strata_full_context_diagnostic']
       for name in ('generate.cpp', 'dense_prefill_nll.cpp')},
}
CORRECTION = '32bde1b0dba0b5c7f83529a0e182e8d83497afc6'
HOLD = 'gu_class_definition_coherence_20261112'
HEADER = P/'include/strata/prefill/gemm.hpp'
require, sha = base.require, base.sha


def prerequisite():
    original = json.loads(base.common.source.git(P, 'show',
        CORRECTION+':'+str(RESULT.relative_to(P))))[HOLD]
    require(original == base.validation.committed(RESULT, (HOLD,))
        and not original['partial_relink_runtime_eligible'],
        'preserve the mixed-definition finding and original failed observations')
    text = HEADER.read_text()
    conditions = re.findall(r'^\s*#\s*(?:if|elif|ifdef|ifndef)\b([^\n]*)', text, re.M)
    require(conditions and all(re.fullmatch(r'\s*defined\(STRATA_ENABLE_SYCL\)\s*', x)
        for x in conditions), 'review new Gemm preprocessing conditions before building')
    return dict(correction_commit=CORRECTION, key=HOLD,
        header_sha256=sha(HEADER), class_condition='STRATA_ENABLE_SYCL',
        purpose='Coherent reconstruction,not compact-GU gain/adoption qualification',
        inherited_GPU_model_speed_or_context_qualification=False)


def definition_census(row, output):
    """Check the real header includers against their fresh compile commands."""
    database = json.loads((ROOT/'build/compile_commands.json').read_text())
    commands = {}
    for item in database:
        args = shlex.split(item['command'])
        if '-o' not in args:
            continue
        obj = Path(args[args.index('-o')+1])
        obj = (obj if obj.is_absolute() else Path(item['directory'])/obj).resolve()
        commands[str(obj)] = args
    obj, consumers = None, []
    for line in output.read_text().splitlines():
        if line and not line[0].isspace():
            name = line.split(': #deps ', 1)[0]
            obj = str((ROOT/'build'/name).resolve())
        elif line.startswith('    ') and obj:
            path = Path(line.strip())
            path = (path if path.is_absolute() else ROOT/'build'/path).resolve()
            if path != HEADER:
                continue
            require(obj in commands, 'Gemm includer missing from actual compile database')
            args = commands[obj]
            require('-DSTRATA_ENABLE_SYCL=1' in args and
                not any(x.startswith('-USTRATA_ENABLE_SYCL') for x in args),
                'Gemm includer uses a different class preprocessing condition')
            consumers.append(str(Path(obj).relative_to(ROOT/'build')))
    required = {'CMakeFiles/strata_prefill.dir/src/prefill/sycl/gemm.cpp.o',
        'CMakeFiles/strata_prefill.dir/src/prefill/sycl/gemm16.cpp.o',
        'CMakeFiles/sycl_gu_policy_test.dir/bench/sycl/gu_compact.cpp.o'}
    require(required <= set(consumers) and len(consumers) == len(set(consumers)),
        'fresh Gemm wrapper,implementation and fixture includers required')
    return dict(header_sha256=sha(HEADER), includer_count=len(consumers),
        includer_objects=sorted(consumers), same_SYCL_class_condition=True,
        all_objects_from_fresh_build=True, partial_archive_reuse=False)


def context():
    spec = importlib.util.spec_from_file_location('coherent_prefill_build_context', base.__file__)
    ctx = importlib.util.module_from_spec(spec); spec.loader.exec_module(ctx)
    ctx.ROOT, ctx.KEY, ctx.TARGETS, ctx.MODEL_BUILD = ROOT, KEY, TARGETS, True
    ctx.SEED, ctx.SEED_KEY, ctx.component_reference = SEED, SEED_KEY, prerequisite
    original_context = ctx.context
    def native_context():
        native = original_context()
        parse, census = native.planned_compilation, native.census
        native.planned_compilation = lambda text: parse(text, allowed_duplicate_targets=DUPLICATES)
        def checked_census(row, dep, link, **kwargs):
            result = census(row, dep, link, **kwargs)
            result['class_definition_coherence'] = definition_census(row, dep)
            return result
        native.census = checked_census
        return native
    ctx.context = native_context
    return ctx


def dependencies(seed):
    """Recover original nested GGML inputs, rather than refreshing their hashes."""
    actual = json.loads(Path(seed['actual_inputs']['path']).read_text())
    require(sha(seed['actual_inputs']['path']) == seed['actual_inputs']['sha256'],
        'original compiler census changed')
    pins = copy.deepcopy(seed['dependencies'])
    for pin in pins:
        root = Path(pin['root'])
        extra = {str(Path(p).relative_to(root)): h for p, h in actual['compiled'].items()
            if Path(p).is_relative_to(root) and str(Path(p).relative_to(root)) not in pin['source_sha256']}
        recovered = {p: hashlib.sha256(raw).hexdigest() for p, raw in
            base.common.source.blobs(root, pin['revision'], extra)}
        require(recovered == extra, 'nested original dependency inputs are not recoverable')
        pin['source_sha256'].update(recovered)
        pin['count'] = len(pin['source_sha256'])
        pin['aggregate_sha256'] = hashlib.sha256(json.dumps(pin['source_sha256'],
            sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return pins


def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()), 'fresh build root required')
    seed = json.loads(SEED.read_text())
    ctx = context()
    require(base.validation.committed(RESULT, (SEED_KEY,)) == ctx.compact(seed)
        and seed['stage'] == 'built' and seed['passed'] and seed['complete']
        and seed['ownership_closed'] and not seed['active'], 'committed coherent seed recipe required')
    project = base.common.source.pin(P, base.common.PREFIXES)
    recipe = copy.deepcopy(seed['recipe']); old = str(SEED.parent)
    recipe['configure'] = [str(ROOT)+v[len(old):] if v.startswith(old) else v
        for v in recipe['configure']]
    recipe['environment'] = {k: v.replace(old, str(ROOT)) for k, v in recipe['environment'].items()}
    recipe['targets'] = recipe['build_targets'] = TARGETS
    require(recipe['configure'][recipe['configure'].index('-B')+1] == str(ROOT/'build')
        and not (ROOT/'build').exists(), 'all native outputs must be newly compiled')
    preserved = list(seed['binaries'].values())
    for name, key in (('qsa16-bulk-mirror-model-native-v1', 'qsa16_bulk_mirror_model_native_v1'),
                      ('q6-dense-head-backing-native-v3', 'q6_dense_head_backing_build_v3')):
        prior = json.loads((B/name/'record.json').read_text())
        public = base.validation.committed(RESULT, (key,))
        require(public['binaries'] == prior['binaries'] and prior['passed']
            and prior['complete'] and prior['ownership_closed'], 'preserve prefill/decode references')
        preserved.extend(prior['binaries'].values())
    probe, proof = ctx.probe_qualification()
    row = dict(stage='prepared', project=project, dependencies=dependencies(seed), recipe=recipe,
        recipe_seed=dict(path=str(SEED), sha256=sha(SEED)), toolchain=seed['toolchain'],
        external_inputs=seed['external_inputs'], native_source_commit=project['revision'],
        preserved_binaries=preserved, native_validation_probe=probe, validation_qualification=proof,
        build_prerequisite=prerequisite(), duplicate_source_admission=DUPLICATES,
        plan='Commit each stage before configure/compile. Fresh full engine,existing nine targets,'
             'physical full-context diagnostic and host fixture,and actual GU fixture. Complete '
             'source/generated/compiler/link census and same-header/SYCL-class condition for every '
             'actual Gemm includer required. No partial archives,source/SDK refresh or GPU execution. '
             'Then commit bounded traced native/numerical checks and a fresh model/repeated-speed '
             'plan. GUlarge is the selected next model policy;compact gain gate remains failed. '
             'No model,task,decode,ordinary teardown or262144/RESTORE qualification inherited.',
        retention='One temporary coherent successor replaces the ineligible partial generation,without '
             'overwriting97fda fallback/reader fixtures,56cad9 imminent matched-prefill reference or '
             'a347 independent-decode reference. Retire partial engine/archive/fixtures and obsolete '
             'readers after exact source/recipe recovery and consumer migration. Review at fresh '
             'native/model closure before another generation;legacy source holds remain separate.',
        budgets=dict(build=dict(wall=2400,cpu=7200,rss_cap=12<<30,text_cap=32<<20,
            file_cap=512<<20,total_cap=4<<30)), commands=[], outcomes={}, passed=False,
        complete=False, active=False, ownership_closed=True, GPU_executed=False,
        model_qualified=False, physical262144_model_qualified=False, performance_qualified=False)
    ROOT.mkdir(mode=0o700)
    for name in ('tmp', 'cache'):
        (ROOT/name).mkdir(mode=0o700)
    ctx.current(row, committed=False); ctx.save(row)
    print('Prepared fresh coherent full build;commit before configure. No GPU execution.', flush=True)


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'configure', 'compile'),
        'use prepare|configure|compile')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base.guards.no_model_owners([])
        if sys.argv[1] == 'prepare':
            prepare()
        else:
            context().build(sys.argv[1])
