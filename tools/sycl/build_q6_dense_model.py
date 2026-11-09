#!/usr/bin/env python3
"""Build independent dense-C4 and immutable host-head trials; Main only.

Reuse the maintained owned build/census implementation in an isolated Python
module context. No old build is overwritten and no GPU code is executed here.
Main-authored from this fork's measured components; repository LICENSE applies.
"""
import ast
import copy
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import build_q6_head_model as primary
import measurement_guards as guards
import measurement_owner as owner

P, B = primary.P, primary.B
ROOT, KEY = B / 'q6-dense-head-backing-native-v3', 'q6_dense_head_backing_build_v3'
REFERENCE_ROOT, REFERENCE_KEY = primary.ROOT, primary.KEY
COMPONENT = 'q6_dense_c4_component_v2'
SHAPES = ('6144x2560', '2560x6144', '2560x10240', '2560x12288')
POLICY = 'sycl_verify_window_policy_test'
POLICY_OUT = 'verify-window-policy: passed selections=30 centering=18 native-half=26 dpas-head=1 dpas-dense=4 phase-roundtrip=1 invalid-keys=4 retirement=1 route-budget=64\n'
require, sha, write, signature = guards.require, primary.digest, primary.write, primary.signature


def prerequisites():
    data = primary.public()
    component = data[COMPONENT]
    require(component['stage']=='closed' and component['passed'] and component['complete'] and
        component['ownership_closed'] and not component['active'] and len(component['timing'])==3 and
        sum(len(v['samples']) for v in component['timing'])==180 and
        all(component['decision']['cells'][s]['component_gain_qualified'] for s in SHAPES) and
        not component['decision']['cells']['2560x512']['component_gain_qualified'] and
        not component['decision']['inference_adopted'], 'closed shape-specific component evidence required')
    reference = json.loads((REFERENCE_ROOT/'record.json').read_text())
    require(data[REFERENCE_KEY]==primary.compact(reference) and reference['stage']=='built' and
        reference['passed'] and reference['complete'] and reference['ownership_closed'] and
        reference['actual_inputs']['uncovered']==0, 'preserve the current qualified native reference')
    cohort = data['q6_packed_head_generation_speed_v1']
    require(cohort['stage']=='closed' and cohort['passed'] and cohort['complete'] and
        cohort['ownership_closed'] and len(cohort['processes'])==6 and
        all(v['passed'] and v['complete'] and v['native_exit_code']==0 and v['fresh_faults']==[]
            for v in cohort['processes']) and cohort['binary']==reference['binaries']['strata'],
        'preserve the current same-build thinking-generation reference')
    original = reference['project']
    recovered = {p:hashlib.sha256(raw).hexdigest() for p,raw in
        primary.source.blobs(P,original['revision'],original['source_sha256'])}
    require(recovered==original['source_sha256'], 'original current native inputs must remain recoverable')
    name='tools/sycl/q6_head_owner.py'
    before=ast.parse(primary.source.git(P,'show',primary.READER+':'+name))
    after=ast.parse((P/name).read_text())
    functions=lambda tree:{v.name:ast.dump(v,include_attributes=False) for v in tree.body if isinstance(v,ast.FunctionDef)}
    a,b=functions(before),functions(after)
    require(set(b)=={'parse_native','finish_gpu_probe'} and a['finish_gpu_probe']==b['finish_gpu_probe'],
        'retain the exact qualified fault-finalizer AST')
    primary.source.git(P,'merge-base','--is-ancestor',original['revision'],'origin/feature/sycl')
    for binary in reference['binaries'].values():
        require(sha(binary['path'])==binary['sha256'] and Path(binary['path']).stat().st_size==binary['bytes'],
            'current fallback engine or same-build fixture changed')
    return dict(reference_record=dict(path=str(REFERENCE_ROOT/'record.json'),sha256=sha(REFERENCE_ROOT/'record.json')),
        reference_source=original['revision'],reference_binaries=reference['binaries'],
        component_key=COMPONENT,eligible_non_head_shapes={s:component['decision']['cells'][s] for s in SHAPES},
        excluded_small_shape=component['decision']['cells']['2560x512'],
        packed_head_preserved=True,host_backing_source='2ca460fa3ec74f6be7c9760216ac4b4044b987ab',
        fault_finalizer_AST_identical=True,numerical_or_GPU_result_inherited=False,model_adopted=False)


def context():
    spec=importlib.util.spec_from_file_location('dense_native_build',primary.__file__)
    ctx=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctx)
    ctx.ROOT,ctx.KEY=ROOT,KEY
    ctx.TARGETS=[*primary.TARGETS,POLICY]
    ctx.HOST_CASES=50
    ctx.prerequisites=prerequisites
    original_current=ctx.current
    def current(row):
        original_current(row)
        if 'actual_inputs' in row:
            actual=row['actual_inputs']
            require(sha(actual['path'])==actual['sha256'],'fresh compiler census changed')
            census=json.loads(Path(actual['path']).read_text())
            require(actual['source_commit']==row['project']['revision'] and actual['uncovered']==0 and
                all(sha(p)==h for field in ('compiled','generated','links') for p,h in census[field].items()),
                'fresh actual native/fixture inputs changed')
            require(all(sha(v['path'])==v['sha256'] and Path(v['path']).stat().st_size==v['bytes']
                for v in row['binaries'].values()),'fresh engine or required fixture changed')
    ctx.current=current
    return ctx


def prepare(ctx):
    require(not ROOT.exists() and KEY not in ctx.public(),'fresh candidate build only;no overwrite/retry')
    guards.no_model_owners([])
    proof=prerequisites();reference=json.loads((REFERENCE_ROOT/'record.json').read_text())
    project=ctx.source.pin(P,ctx.PREFIXES)
    _,identity=ctx.selected.basis()
    recipe=copy.deepcopy(reference['recipe'])
    recipe['configure']=[v.replace(str(REFERENCE_ROOT),str(ROOT)) for v in recipe['configure']]
    recipe['environment']={k:v.replace(str(REFERENCE_ROOT),str(ROOT)) for k,v in recipe['environment'].items()}
    recipe['targets']=recipe['build_targets']=ctx.TARGETS
    recipe['environment'].update(STRATA_SYCL_DECODE_Q6_DPAS_HEAD='0',
        STRATA_SYCL_DECODE_Q6_DPAS_DENSE='0',STRATA_PREFILL_HEAD_HOST_BACKING='0')
    require(recipe['configure'][recipe['configure'].index('-S')+1]==str(P) and
        recipe['configure'][recipe['configure'].index('-B')+1]==str(ROOT/'build'), 'fresh primary build path required')
    row=dict(stage='prepared',project=project,native_source_commit=project['revision'],
        dependencies=copy.deepcopy(reference['dependencies']),recipe=recipe,toolchain=reference['toolchain'],
        external_inputs=reference['external_inputs'],fallbacks=reference['fallbacks']|
            dict(current_qualified=reference['binaries']['strata']),
        selected_native_sha256=signature(identity),prerequisites=proof,
        host_python=ctx.HOST_PYTHON,host_python_sha256=sha(ctx.HOST_PYTHON),budgets=ctx.BUDGET,
        commands=[],passed=False,complete=False,active=False,ownership_closed=True,GPU_executed=False,
        model_qualified=False,production_Verifier_qualified=False,
        plan=dict(host='50 committed CPU parser/census/protocol controls;all original cases plus three backing-policy controls.',
            configure='Owned finite configure and exact selected Ninja commands;commit generated inputs before compile.',
            compile='Fresh engine,three production-linked GPU fixtures and CPU phase/shape/cache-policy fixture;'
                    'complete compiler/generated/link census and selected engine/head kernel identity.',
            policy='Run the compiled CPU fixture:four independently winning non-head C4 shapes only;'
                   'disabled/prompt/other types and widths/small/full-head stay on incumbents;phase-cache retirement.',
            subsequent='No GPU result inherited. Fresh bounded traced submission/phase/head0/1/2 checks under'
                ' both backing policies,then same-engine model handoff,thinking strong/weak and long/agent checks.'
                ' Compare dense0/1 with backing fixed0;compare backing0/1 with decode selection fixed.'
                ' Keep all individual speed samples and require repeatable per-mechanism gains;do not trade'
                ' prefill for decode. Physical262144/RESTORE and task gates remain independent.'),
        retention='One new candidate for the imminent four-shape decode and host-backed prefill comparisons.'
            ' Preserve current1db/four same-build fixtures without overwrite;retire the losing/superseded'
            ' generation after qualification,exact original source/recipe recovery and consumer updates.'
            ' Existing5a6/90f/552b/e1b consumers and56legacy provenance holds remain separately recorded.')
    ctx.build.check_inputs(row)
    ROOT.mkdir(mode=0o700)
    for name in ('tmp','cache'):(ROOT/name).mkdir(mode=0o700)
    ctx.save(row)
    print('Prepared independent dense/backing build;commit before configure.',flush=True)


def policy(ctx):
    row=json.loads((ROOT/'record.json').read_text());ctx.current(row)
    require(row['stage']=='built' and row['passed'] and row['complete'] and 'policy_check' not in row,
        'fresh compiled CPU policy check only')
    scope=None
    def persist():write(ROOT/'policy-execution.json',dict(active=bool(scope and scope.active),commands=scope.commands if scope else []))
    try:
        scope=owner.Owner(ROOT/'owned-policy',persist)
        e,out,err=scope.run('phase-shape-policy',[row['binaries'][POLICY]['path']],row['recipe']['environment'],
            P,wall=120,cpu=120,rss_cap=1<<30,text_cap=1<<20,file_cap=4<<20,total_cap=64<<20)
        require(owner.closed(e) and e['exit_code']==0 and out.read_text()==POLICY_OUT,
            'complete native phase/shape/cache policy controls required')
        row['policy_check']=dict(passed=True,GPU_executed=False,native_exit_code=0,
            selections=4,prompt_and_width_isolation=True,cache_retirement=True,source_commit=row['project']['revision'])
    except BaseException as error:row.update(stage='failed',passed=False,complete=False,error=repr(error))
    finally:
        row['commands']+=scope.commands if scope else []
        row['active']=bool(scope and scope.active)
        row['ownership_closed']=not row['active'] and all(owner.closed(v) for v in row['commands'])
        row['passed']=bool(row['passed'] and row['ownership_closed']);ctx.save(row)
    require(row['passed'] and row['ownership_closed'],row.get('error','native policy failed'))
    print('Closed native CPU policy check;GPU/model qualification remains separate.',flush=True)


if __name__=='__main__':
    require(sys.argv[1:] in (['prepare'],['configure'],['compile'],['policy']),
        'build_q6_dense_model.py prepare|configure|compile|policy')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([])
        ctx=context();mode=sys.argv[1]
        if mode=='prepare':prepare(ctx)
        elif mode=='policy':policy(ctx)
        else:ctx.execute(mode)
