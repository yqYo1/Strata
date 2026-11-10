#!/usr/bin/env python3
"""Fresh same-build native checks before the streamed-QSA16 model trial.

Main-authored from this fork's owned supervision; repository LICENSE (MIT).
No new executable or numerical implementation. Main serializes every stage.
"""
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.dont_write_bytecode=True
import build_q6_head_model as common
import build_q6_dense_model as incumbent
import check_q6_dense_model as previous
import check_qsa16_stream as candidate
import measurement_guards as guards
import measurement_owner as owner
import ur_validation as validation
from q6_canonical_component import image as canonical_image

P,B,RESULT=common.P,common.B,common.RESULT
NATIVE=B/'qsa16-stream-model-native-v2'
NATIVE_KEY='qsa16_stream_model_native_v2'
ROOT,KEY=B/'qsa16-stream-model-device-v1','qsa16_stream_model_device_v1'
THIS='tools/sycl/check_qsa16_stream_model.py'
PRIOR='q6_dense_head_backing_device_v1'
STAGES=('host','submission','phase','head0-backing0','head2-backing0')
require,sha,write=guards.require,common.digest,common.write


def native(expected=None):
    row=json.loads((NATIVE/'record.json').read_text())
    require(validation.committed(RESULT,(NATIVE_KEY,))==candidate.compact(row) and
        row['stage']=='built' and row['passed'] and row['complete'] and
        row['ownership_closed'] and not row['active'] and not row['GPU_executed'],
        'closed exact full-engine build required')
    original=row['project'];current=common.source.pin(P,common.PREFIXES)
    a,b=original['source_sha256'],current['source_sha256']
    require(a=={p:hashlib.sha256(raw).hexdigest() for p,raw in
        common.source.blobs(P,original['revision'],a)} and
        set(b)-set(a)=={THIS} and not set(a)-set(b) and
        all(b[p]==h for p,h in a.items()),
        'only this committed observer may extend the exact native build')
    require(row['native_source_commit']==original['revision']==row['actual_inputs']['source_commit']
        and row['actual_inputs']['translation_units']==128 and row['actual_inputs']['uncovered']==0,
        'original native source and complete compiler census required')
    for pin in row['dependencies']:
        require(pin['source_sha256']=={p:hashlib.sha256(raw).hexdigest() for p,raw in
            common.source.blobs(Path(pin['root']),pin['revision'],pin['source_sha256'])} and
            all(sha(Path(pin['root'])/p)==h for p,h in pin['source_sha256'].items()),
            'original dependency source changed')
    require(sha(candidate.SEED)==row['recipe_seed']['sha256'] and
        sha(row['external_inputs']['path'])==row['external_inputs']['sha256'] and
        all(sha(p)==h for p,h in row['toolchain'].items()) and
        all(sha(p)==h for p,h in json.loads(Path(row['external_inputs']['path']).read_text()).items()),
        'compiler/SDK/recipe seed changed')
    seed=json.loads(candidate.SEED.read_text())
    require(common.HOST_PYTHON==seed['host_python'] and
        sha(common.HOST_PYTHON)==seed['host_python_sha256'],
        'qualified host-control interpreter changed')
    for item in (*row['preserved_binaries'],*row['binaries'].values()):
        require(Path(item['path']).stat().st_size==item['bytes'] and
            sha(item['path'])==item['sha256'],'native/reference executable changed')
    require(candidate.probe_qualification()==(row['native_validation_probe'],row['validation_qualification']),
        'qualified pre-main loader/runtime proof changed')
    require(all(sha(p)==h for p,h in (row['configured_files']|row['generated_inputs']).items()) and
        sha(row['actual_inputs']['path'])==row['actual_inputs']['sha256'],
        'original configured/generated/compiler census changed')
    census=json.loads(Path(row['actual_inputs']['path']).read_text())
    require(all(sha(p)==h for field in ('compiled','generated','links')
        for p,h in census[field].items()),'actual compiled/header/link inputs changed')
    proof=dict(native_source_commit=row['native_source_commit'],original_source_files=len(a),
        observer_source_files=len(b),observer_source_sha256=common.signature(b),
        added_observers={THIS:b[THIS]},exact_five_ELFs_census_recipe=True,
        native_recompiled=False,GPU_result_inherited=False)
    require(expected is None or proof==expected,'native observer admission changed')
    return dict(row,observer_admission=proof)


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def context(expected=None):
    ctx=load_module('qsa16_stream_native_head_context',previous.previous.__file__)
    ctx.ROOT,ctx.KEY=ROOT,KEY
    ctx.build=SimpleNamespace(**{**vars(common),'ROOT':NATIVE,'KEY':NATIVE_KEY})
    ctx.native=lambda:native(expected)
    return ctx


def host(device,ctx,row,scope,outcome):
    device.host(ctx,row,scope,outcome)
    env=dict(row['environment']);env.pop('LD_PRELOAD',None)
    e,out,err=scope.run('phase-shape-policy',
        [row['binaries'][incumbent.POLICY]['path']],env,P,**row['host_budget'])
    require(owner.closed(e) and e['exit_code']==0 and out.read_text()==incumbent.POLICY_OUT,
        'fresh CPU phase/shape/cache policy required')
    outcome['policy_check']=dict(passed=True,GPU_executed=False,native_exit_code=0)
    n=native(row['native_observer_admission'])
    reference=json.loads(candidate.SEED.read_text())
    actual={}
    for label,binary in (('engine',n['binaries']['strata']),
                         ('fixture',n['binaries']['sycl_q6_head_owner_test']),
                         ('reference',reference['binaries']['strata'])):
        image=canonical_image(Path(binary['path']).read_bytes())
        actual[label]=common.structure(binary,image)
    require(actual['engine']==actual['fixture']==actual['reference'],
        'selected head SPIR-V changed against the current decode reference')
    outcome['selected_head_IR']=dict(actual['engine'],
        same_as_current_reference=True,engine_fixture_identical=True,
        scope='Exact selected linked-module structure,including literals/order;not final ISA,GPU correctness or speed.')


def prepare():
    require(not ROOT.exists() and KEY not in common.public(),'fresh device plan;no overwrite/retry')
    n=native();ctx=context(n['observer_admission'])
    prior=validation.committed(RESULT,(PRIOR,))
    require(prior['stage']=='closed' and prior['passed'] and prior['complete'] and
        prior['ownership_closed'] and not prior['active'],'qualified fixture environment required')
    old=str(B/'q6-dense-head-backing-device-v1')
    env={k:v.replace(old,str(ROOT)) for k,v in prior['environment'].items()}
    require(env['STRATA_PREFILL_RELEASE_HEADS']=='1' and env['ONEAPI_DEVICE_SELECTOR']=='level_zero:gpu' and
        env['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD']=='1' and env['EnableDirectSubmission']=='0' and
        env['STRATA_SYCL_VERIFY_NATIVE_CAPTURE']=='0' and env['SYCL_CACHE_PERSISTENT']=='0' and
        env['STRATA_PREFILL_HEAD_HOST_BACKING']=='0','preserve qualified immutable safety/head settings')
    require(sha(ctx.RUNTIME)==ctx.RUNTIME_SHA and
        guards.driver_guard()==prior['driver'] and
        Path('/proc/sys/kernel/random/boot_id').read_text().strip()==prior['boot'],
        'qualified runtime/driver/boot changed')
    runtime=json.loads(ctx.RUNTIME.read_text())
    require(all(sha(p)==h for p,h in runtime.items()) and
        n['native_validation_probe']==prior['native_validation_probe'],'runtime/probe bytes changed')
    row=dict(stage='prepared',source_commit=common.source.git(P,'rev-parse','HEAD').decode().strip(),
        controller=common.source.pin(P,['tools/sycl']),native_source_commit=n['native_source_commit'],
        native_record=dict(path=str(NATIVE/'record.json'),sha256=sha(NATIVE/'record.json')),
        host_python=dict(path=common.HOST_PYTHON,sha256=sha(common.HOST_PYTHON)),
        native_observer_admission=n['observer_admission'],binary=n['binaries']['sycl_q6_head_owner_test'],
        binaries=n['binaries'],prior_environment_source=PRIOR,environment=env,assets=prior['assets'],
        runtime_manifest=dict(path=str(ctx.RUNTIME),sha256=ctx.RUNTIME_SHA),runtime_pins=runtime,
        native_validation_probe=prior['native_validation_probe'],boot=prior['boot'],driver=prior['driver'],
        archived_gate=dict(commit=ctx.ORIGINAL,path=ctx.ORIGINAL_PATH,sha256=ctx.archived_gate()[1]),
        native_budget=prior['native_budget'],host_budget=prior['host_budget'],
        plan=dict(order=list(STAGES),host='50existing CPU parser/census/protocol cases,11fresh empty/switch/backing refusals,the fresh CPU phase/shape/cache policy,and exact selected head-module structure against current a347 and the production-linked fixture.',
            small='Fresh full-trace submission10 and phase-cache16 checks before real-head GPU work.',
            head='Two fresh real-shard processes:default head0/backing0 and selected packed head2/backing0. Existing numeric cases,retire/overwrite/restore cycles,graph replays and mode2partial-column checks. Full shard hashes before/after each process,pre-main/post-device owner/provider gates,immutable environment,quiet validation,normal0,closed owners and no fresh faults.',
            stop='Commit every closed stage;stop first failure,no retries or changed criteria.',
            next='Then prepare a separately committed actual-mode1-session and streamed typed-attention/prefill-to-decode plan. No model or speed follows from these fixtures.'),
        limits='Only default/current packed heads with backing0 are screened here;canonical head1 and backing1 are not newly qualified. No dense model routing,attention/model/task/physical262144/RESTORE,PP/TG/TTFT,ordinary provider teardown or inference adoption.',
        retention='Same-build native prerequisite for the imminent streamed-prefill model admission. Keep compact numerical/transition outcomes,source/ELF/runtime identity and faults. Retire successful full traces,polling and caches after closure. No new engine;candidate plus named a347/552b references and separate legacy provenance holds remain.',
        results=[],commands=[],passed=False,complete=False,active=False,ownership_closed=True,
        GPU_executed=False,model_executed=False)
    ROOT.mkdir(mode=0o700)
    for name in ('tmp','cache'):(ROOT/name).mkdir(mode=0o700)
    ctx.save(row);print('Prepared same-build streamed-QSA16 native checks;commit before host.',flush=True)


def execute(stage):
    device=load_module('qsa16_stream_native_device_context',previous.__file__)
    device.ROOT,device.KEY,device.STAGES=ROOT,KEY,STAGES
    device.context=context
    original_host=device.host
    device.host=lambda ctx,row,scope,outcome:host(SimpleNamespace(host=original_host),ctx,row,scope,outcome)
    device.execute(stage)


if __name__=='__main__':
    require(len(sys.argv)==2 and sys.argv[1] in ('prepare',*STAGES),
        'check_qsa16_stream_model.py prepare|'+'|'.join(STAGES))
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([])
        if sys.argv[1]=='prepare':prepare()
        else:execute(sys.argv[1])
