#!/usr/bin/env python3
"""Fresh CPU/GPU expert-bank fixtures only; Main-owned,repository LICENSE.

No engine build,partial archive relink or inherited GPU qualification. Preserve
the current coherent engine; old engine-generation retirement remains pending.
"""
import copy
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import check_qsa16_stream as base
import check_expert_copy_runs as cpu

P, B, RESULT = base.P, base.B, base.RESULT
ROOT, KEY = B/'expert-copy-size-component-v2', 'expert_copy_size_component_v2'
SEED, SEED_KEY = B/'coherent-prefill-native-v1/record.json', 'coherent_prefill_native_v1'
TARGETS = ['sycl_expert_bank_test','prefill_expert_bank_plan_test']
UNITS = {'tests/sycl/expert_bank.cpp','tests/expert_bank_plan.cpp','src/core/expert_cache.cpp',
         'src/sycl/runtime.cpp','src/sycl/compat.cpp','src/sycl/command_profile.cpp'}
HEADER = P/'include/strata/prefill/expert_bank_plan.hpp'
require, sha = base.require, base.sha


def prerequisite():
    row = json.loads((cpu.ROOT/'record.json').read_text())
    public = base.validation.committed(RESULT,(cpu.KEY,))
    require(public == {k:v for k,v in row.items() if k!='commands'} and
        row['stage']=='closed' and row['passed'] and row['complete'] and
        row['ownership_closed'] and not row['active'] and not row['GPU_executed'],
        'closed committed CPU planner gate required')
    checks = row['checks']
    require(checks['configurations']==513 and checks['windows_each']==8 and
        checks['negative_cases']==22 and checks['sequential_ticket_slot_generation_and_bytes_equal']
        and row['compiler_census']['uncovered']==0,'complete CPU slot/failure contract required')
    recovered = {p:hashlib.sha256(raw).hexdigest() for p,raw in
        base.common.source.blobs(P,row['source_commit'],row['source_sha256'])}
    require(recovered==row['source_sha256'] and all(sha(P/p)==h for p,h in recovered.items()),
        'CPU-tested source/controller changed')
    require(all(cpu.owner.closed(c) and c['exit_code']==0 for c in row['commands']),
        'CPU gate ownership changed')
    return dict(key=cpu.KEY,source_commit=row['source_commit'],
        record_sha256=sha(cpu.ROOT/'record.json'),cpp_and_planner_unchanged=True,
        GPU_speed_model_or_context_inherited=False)


def definition_census(output):
    current, users = None, []
    for line in output.read_text().splitlines():
        if line and not line[0].isspace(): current=line.split(': #deps ',1)[0]
        elif line.startswith('    ') and current:
            path=Path(line.strip());path=(path if path.is_absolute() else ROOT/'build'/path).resolve()
            if path==HEADER:
                users.append(current)
    expected={'CMakeFiles/sycl_expert_bank_test.dir/tests/sycl/expert_bank.cpp.o',
              'CMakeFiles/prefill_expert_bank_plan_test.dir/tests/expert_bank_plan.cpp.o'}
    require(set(users)==expected and len(users)==len(expected),
        'fresh CPU/device planner includers required')
    return dict(header_sha256=sha(HEADER),includer_objects=sorted(users),
        all_six_objects_fresh=True,partial_archive_reuse=False,
        class_has_no_backend_conditional_members=True)


def context():
    spec=importlib.util.spec_from_file_location('copy_run_component_context',base.__file__)
    ctx=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctx)
    ctx.ROOT,ctx.KEY,ctx.SEED,ctx.SEED_KEY=ROOT,KEY,SEED,SEED_KEY
    ctx.TARGETS,ctx.TARGET,ctx.UNITS=TARGETS,TARGETS[0],UNITS
    old_current,old_context=ctx.current,ctx.context
    def current(row,**kwargs):
        old_current(row,**kwargs)
        require(prerequisite()==row['build_prerequisite'],'copy-run CPU evidence changed')
    def native_context():
        native=old_context();census=native.census
        def checked_census(row,dep,link,**kwargs):
            result=census(row,dep,link,**kwargs)
            result['class_definition_coherence']=definition_census(dep)
            return result
        native.census=checked_census
        return native
    ctx.current,ctx.context=current,native_context
    return ctx




def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()),'fresh fixture build required')
    gate=prerequisite();seed=json.loads(SEED.read_text())
    public=base.validation.committed(RESULT,(SEED_KEY,))
    require(public==base.compact(seed) and seed['stage']=='built' and seed['passed'] and
        seed['complete'] and seed['ownership_closed'] and not seed['active'],
        'closed coherent full-build recipe required')
    for pin in [seed['project'],*seed['dependencies']]:
        recovered={p:hashlib.sha256(raw).hexdigest() for p,raw in
            base.common.source.blobs(Path(pin['root']),pin['revision'],pin['source_sha256'])}
        require(recovered==pin['source_sha256'],'original coherent source/recipe recovery failed')
    recipe=copy.deepcopy(seed['recipe']);old=str(SEED.parent)
    recipe['configure']=[str(ROOT)+v[len(old):] if v.startswith(old) else v for v in recipe['configure']]
    recipe['configure'][recipe['configure'].index('-S')+1]=str(P)
    recipe['environment']={k:v.replace(old,str(ROOT)) for k,v in recipe['environment'].items()}
    recipe['targets']=recipe['build_targets']=TARGETS
    require(recipe['configure'][recipe['configure'].index('-B')+1]==str(ROOT/'build'),
        'isolated fresh fixture output required')
    probe,proof=context().probe_qualification()
    row=dict(stage='prepared',project=base.common.source.pin(P,base.common.PREFIXES),
        dependencies=copy.deepcopy(seed['dependencies']),recipe=recipe,
        recipe_seed=dict(path=str(SEED),sha256=sha(SEED)),toolchain=seed['toolchain'],
        external_inputs=seed['external_inputs'],preserved_binaries=[seed['binaries']['strata']],
        build_prerequisite=gate,native_validation_probe=probe,validation_qualification=proof,
        plan='Compile only CPU planner and actual production bank-upload GPU fixture with bounded '
            'copy sizes and disjoint host accounting,including all '
            'six fresh translation units. Preserve qualified compiler/SDK and strict-FP recipe. '
            'Require source/compiler/generated/link census with zero uncovered files and both '
            'unconditional planner-class includers freshly built. Commit each stage before execution. '
            'Then commit a separate CPU/traced bounded-size correctness gate and attribution/clean '
            'comparison plan. No previous fixture GPU or performance qualification is inherited. '
            'No engine,weights,GPU execution,decode or model speed in this build.',
        retention='One temporary fixture generation for the imminent transfer-size/host-accounting '
            'diagnostic. The previous exact-copy fixture generation is already retired after source '
            'recovery;its failed outcome stays in Git. Existing coherent engine remains the fallback. '
            'Old engine-generation/provenance/reader holds remain open; '
            'resolve them before another engine build. Retire rejected candidate exports/fixture modes '
            'after the decision;keep original measured commits reachable.',
        budgets=dict(build=dict(wall=2400,cpu=7200,rss_cap=12<<30,text_cap=32<<20,
            file_cap=512<<20,total_cap=2<<30)),commands=[],outcomes={},passed=False,complete=False,
        active=False,ownership_closed=True,GPU_executed=False,model_qualified=False,
        physical262144_model_qualified=False,performance_qualified=False)
    row['native_source_commit']=row['project']['revision']
    ROOT.mkdir(mode=0o700)
    for name in ('tmp','cache'):(ROOT/name).mkdir(mode=0o700)
    ctx=context();ctx.current(row,committed=False);ctx.save(row)
    print('Prepared six-TU/two-fixture copy-run build;commit before configure.',flush=True)


if __name__=='__main__':
    require(sys.argv[1:] in (['prepare'],['configure'],['compile']),'prepare|configure|compile')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB);base.guards.no_model_owners([])
        if sys.argv[1]=='prepare':prepare()
        else:
            ctx=context();ctx.build(sys.argv[1])
            if sys.argv[1]=='compile':
                row=json.loads((ROOT/'record.json').read_text());row['complete']=True;ctx.save(row)
