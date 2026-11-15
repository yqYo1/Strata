#!/usr/bin/env python3
"""Fresh GPU/host ownership checks for the dense/backing candidate; Main only.

Reuse our maintained supervision and strict numerical/trace readers. Native
source, five binaries and build recipe remain at their original revisions;
the only added input is this committed observer. No inference adoption here.
Main-authored; repository LICENSE applies.
"""
import copy
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode=True
import build_q6_dense_model as build
import check_q6_head_model as previous
import measurement_guards as guards
import measurement_owner as owner
from q6_canonical_component import validated_trace
from q6_head_owner import parse_native
import ur_validation as validation

P,B,RESULT=build.P,build.B,build.primary.RESULT
ROOT,KEY=B/'q6-dense-head-backing-device-v1','q6_dense_head_backing_device_v1'
THIS='tools/sycl/check_q6_dense_model.py'
PRIOR='q6_head_packed_device_v1'
STAGES=('host','submission','phase',*(f'head{mode}-backing{backing}'
    for mode in (0,1,2) for backing in (0,1)))
require,sha,write=guards.require,build.sha,build.write


def native(expected=None):
    ctx=build.context();n=json.loads((build.ROOT/'record.json').read_text())
    require(validation.committed(RESULT,(build.KEY,))==ctx.compact(n) and n['stage']=='built' and
        n['passed'] and n['complete'] and n['ownership_closed'] and not n['active'] and
        n['policy_check']['passed'] and not n['GPU_executed'], 'closed fresh build and compiled CPU policy required')
    before=n['admission_project'];project=ctx.source.pin(P,ctx.PREFIXES)
    a,b=before['source_sha256'],project['source_sha256']
    require(a=={p:hashlib.sha256(raw).hexdigest() for p,raw in
        ctx.source.blobs(P,before['revision'],a)} and set(b)-set(a)=={THIS} and not set(a)-set(b) and
        all(b[p]==h for p,h in a.items()), 'only this new observer may differ from the exact build admission')
    require(n['native_source_commit']==n['project']['revision']==n['actual_inputs']['source_commit'] and
        build.prerequisites()==n['prerequisites'], 'native source/fallback proof changed')
    ctx.build.check_inputs(dict(n,project=project))
    require(all(sha(p)==h for p,h in (n['generated_inputs']|n['configured_files']).items()),
        'configured/generated recipe changed')
    actual=n['actual_inputs'];require(sha(actual['path'])==actual['sha256'] and actual['uncovered']==0,
        'exact native compiler census changed')
    census=json.loads(Path(actual['path']).read_text())
    require(all(sha(p)==h for field in ('compiled','generated','links') for p,h in census[field].items()) and
        all(sha(v['path'])==v['sha256'] and Path(v['path']).stat().st_size==v['bytes']
            for v in n['binaries'].values()), 'actual native inputs/five binaries changed')
    proof=dict(native_source_commit=n['native_source_commit'],build_admission_source_commit=before['revision'],
        original_source_files=before['count'],observer_source_files=project['count'],
        observer_source_sha256=build.signature(b),added_observers={THIS:b[THIS]},
        exact_five_ELFs_and_census_and_recipe=True,native_recompiled=False,GPU_result_inherited=False)
    require(expected is None or proof==expected,'observer admission changed during the device plan')
    n['observer_admission']=proof
    return n


def context(expected=None):
    spec=importlib.util.spec_from_file_location('dense_head_device',previous.__file__)
    ctx=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctx)
    ctx.ROOT,ctx.KEY=ROOT,KEY
    ctx.build=build.context();ctx.native=lambda:native(expected)
    return ctx


def prepare():
    require(not ROOT.exists() and KEY not in build.primary.public(),'fresh device plan;no overwrite/retry')
    n=native();ctx=context(n['observer_admission'])
    prior=validation.committed(RESULT,(PRIOR,))
    require(prior['stage']=='closed' and prior['passed'] and prior['complete'] and prior['ownership_closed'],
        'qualified immutable runtime/fixture environment required')
    env={k:v.replace(str(previous.ROOT),str(ROOT)) for k,v in prior['environment'].items()}
    env.update(STRATA_SYCL_DECODE_Q6_DPAS_HEAD='0',STRATA_SYCL_DECODE_Q6_DPAS_DENSE='0',
        STRATA_PREFILL_HEAD_HOST_BACKING='0')
    require(env['STRATA_PREFILL_RELEASE_HEADS']=='1' and env['ONEAPI_DEVICE_SELECTOR']=='level_zero:gpu' and
        env['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD']=='1' and env['EnableDirectSubmission']=='0' and
        env['STRATA_SYCL_VERIFY_NATIVE_CAPTURE']=='0' and env['SYCL_CACHE_PERSISTENT']=='0',
        'preserve qualified immutable safety settings')
    require(sha(ctx.RUNTIME)==ctx.RUNTIME_SHA and guards.driver_guard()==prior['driver'] and
        Path('/proc/sys/kernel/random/boot_id').read_text().strip()==prior['boot'],'qualified runtime/driver/boot required')
    runtime=json.loads(ctx.RUNTIME.read_text());require(all(sha(p)==h for p,h in runtime.items()),'runtime bytes changed')
    row=dict(stage='prepared',source_commit=ctx.build.source.git(P,'rev-parse','HEAD').decode().strip(),
        controller=ctx.build.source.pin(P,['tools/sycl']),native_source_commit=n['native_source_commit'],
        native_record=dict(path=str(build.ROOT/'record.json'),sha256=sha(build.ROOT/'record.json')),
        native_observer_admission=n['observer_admission'],binary=n['binaries']['sycl_q6_head_owner_test'],
        binaries=n['binaries'],prior_environment_source=PRIOR,environment=env,assets=prior['assets'],
        runtime_manifest=dict(path=str(ctx.RUNTIME),sha256=ctx.RUNTIME_SHA),runtime_pins=runtime,
        native_validation_probe=prior['native_validation_probe'],boot=prior['boot'],driver=prior['driver'],
        archived_gate=dict(commit=ctx.ORIGINAL,path=ctx.ORIGINAL_PATH,sha256=ctx.archived_gate()[1]),
        native_budget=prior['native_budget'],host_budget=prior['host_budget'],
        plan=dict(order=list(STAGES),host='50 existing CPU reader cases,then11fresh native CPU-only empty/switch/backing refusal cases.',
            small='Fresh full-trace submission10 and phase-cache16 checks before real-head GPU work.',
            head='Six independent full-trace NativeHead processes:mode0/1/2 each with backing0then1.'
                 ' Existing four numeric cases,two retire/overwrite/restore cycles,12graph replays;mode2'
                 ' additionally checks eight partial-column cases. Full real-shard hashes before/after'
                 ' each process;two actual native/provider/env gates;quiet validation,normal0,closed,no fault.',
            stop='Commit every closed stage. Stop first failure;no retries or criterion changes.',
            scope='Native fixture checks only;dense dispatcher still needs actual model/route evidence.'),
        limits='No model throughput,physical traffic/saturation,task/agent,262144/RESTORE,ordinary-provider teardown or default adoption.',
        retention='Keep compact numerical/transition samples,actual RAM/device extents,source/ELF/runtime and fault/closure results.'
            ' Retire successful full traces/polling/caches after closure;keep current a347 candidate and1db reference.'
            ' Separate legacy phase/feature consumers and provenance holds remain unchanged.',
        results=[],commands=[],passed=False,complete=False,active=False,ownership_closed=True,
        GPU_executed=False,model_executed=False)
    ROOT.mkdir(mode=0o700)
    for name in ('tmp','cache'):(ROOT/name).mkdir(mode=0o700)
    ctx.save(row);print('Prepared fresh dense/backing device plan;commit before host.',flush=True)


def host(ctx,row,scope,outcome):
    env=dict(row['environment']);env.pop('LD_PRELOAD',None)
    e,out,err=scope.run('reader-controls',[ctx.build.HOST_PYTHON,'-B','-m','unittest','-v',
        'tools.sycl.test_q6_dpas_component','tools.sycl.test_compile_census','tools.sycl.test_q6_head_generation_speed'],
        env,P,**row['host_budget'])
    text=out.read_text()+err.read_text()
    require(owner.closed(e) and e['exit_code']==0 and 'Ran 50 tests' in text and '\nOK\n' in text,
        'all50existing CPU reader controls required')
    cases=[('empty',['--host'],{},'HOST owner_negative_cases=3 GPU=0\nPACKED_HOST owner_negative_cases=1 GPU=0')]
    reject='REJECT_SWITCH passed=1 GPU=0'
    for label,changes,message in (
        ('flag3',{'STRATA_SYCL_DECODE_Q6_DPAS_HEAD':'3'},'must be 0 or 1'),
        ('empty-flag',{'STRATA_SYCL_DECODE_Q6_DPAS_HEAD':''},'must be 0 or 1'),
        ('release0',{'STRATA_SYCL_DECODE_Q6_DPAS_HEAD':'1','STRATA_PREFILL_RELEASE_HEADS':'0'},'requires STRATA_PREFILL_RELEASE_HEADS=1'),
        ('release-absent',{'STRATA_SYCL_DECODE_Q6_DPAS_HEAD':'1','STRATA_PREFILL_RELEASE_HEADS':None},'requires STRATA_PREFILL_RELEASE_HEADS=1'),
        ('packed-release0',{'STRATA_SYCL_DECODE_Q6_DPAS_HEAD':'2','STRATA_PREFILL_RELEASE_HEADS':'0'},'requires STRATA_PREFILL_RELEASE_HEADS=1'),
        ('packed-release-absent',{'STRATA_SYCL_DECODE_Q6_DPAS_HEAD':'2','STRATA_PREFILL_RELEASE_HEADS':None},'requires STRATA_PREFILL_RELEASE_HEADS=1'),
        ('backing3',{'STRATA_PREFILL_HEAD_HOST_BACKING':'3'},'STRATA_PREFILL_HEAD_HOST_BACKING must be 0 or 1'),
        ('backing-empty',{'STRATA_PREFILL_HEAD_HOST_BACKING':''},'STRATA_PREFILL_HEAD_HOST_BACKING must be 0 or 1'),
        ('backing-release0',{'STRATA_PREFILL_HEAD_HOST_BACKING':'1','STRATA_PREFILL_RELEASE_HEADS':'0'},'host backing requires STRATA_PREFILL_RELEASE_HEADS=1'),
        ('backing-release-absent',{'STRATA_PREFILL_HEAD_HOST_BACKING':'1','STRATA_PREFILL_RELEASE_HEADS':None},'host backing requires STRATA_PREFILL_RELEASE_HEADS=1')):
        cases.append((label,['--reject-switch',message],changes,reject))
    for label,args,changes,expected in cases:
        child=dict(env)
        for k,v in changes.items():
            if v is None:child.pop(k,None)
            else:child[k]=v
        e,out,err=scope.run(label,[row['binary']['path'],*args],child,ROOT,**row['host_budget'])
        require(owner.closed(e) and e['exit_code']==0 and out.read_text().strip()==expected and
            not re.search(r'--->\s*ur[A-Z]|<---\s*ur[A-Z]',err.read_text()), 'CPU owner/switch refusal:'+label)
    outcome.update(passed=True,native_exit_code=0,GPU_executed=False,reader_cases=50,native_cases=len(cases))


def small(ctx,row,scope,stage,outcome):
    target,args,expected=(('sycl_submission_test',['--submission'],ctx.SUBMISSION) if stage=='submission'
        else ('sycl_verifier_retirement_test',['--phase-cache'],[ctx.PHASE,'phase_fixture_status=PASS cases=1']))
    env=dict(row['environment'],LD_PRELOAD=row['native_validation_probe']['library']['path'],
        STRATA_UR_PROBE_EXPECTED_LOADER=row['native_validation_probe']['loader']['path'])
    cursor=next(v.split(': ',1)[1] for v in subprocess.check_output(
        ['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15).splitlines() if v.startswith('-- cursor: '))
    outcome['native_execution_attempted']=True
    try:
        e,out,err=scope.run(stage,[row['binaries'][target]['path'],*args],env,ROOT,**row['native_budget'])
        outcome['native_exit_code']=e.get('exit_code')
        require(owner.closed(e) and e['exit_code']==0,'small owned device fixture failed')
        text=out.read_text();proof,body=validated_trace(err.read_text(),e['pid'])
        require(proof['loader']==env['STRATA_UR_PROBE_EXPECTED_LOADER'] and
            (not (ROOT/'validation.log').exists() or (ROOT/'validation.log').stat().st_size==0), 'quiet actual validation required')
        require(([v for v in text.splitlines() if v.startswith('case ')]==expected and
            text.count('fixture_status=PASS mode=submission')==1) if stage=='submission' else text.splitlines()==expected,
            'complete small native fixture extent required')
        require('urEnqueue' in body,'actual traced native device submissions required')
        outcome.update(passed=True,native_exit_code=0,validation=proof,host_probe_accepted=True,
            GPU_executed=True,cases=10 if stage=='submission' else 16)
    finally:
        validation.finish(outcome,scope,cursor,row['boot'])
        if outcome.get('driver_after')!=row['driver']:outcome['passed']=False


def execute(stage):
    row=json.loads((ROOT/'record.json').read_text());ctx=context(row['native_observer_admission']);ctx.verify(row)
    require(row['stage'] in ('prepared','in_progress') and len(row['results'])==STAGES.index(stage) and
        all(v['passed'] for v in row['results']), 'fresh ordered stage;no retry')
    scope=None;outcome=dict(mode=stage,passed=False,GPU_executed=False)
    def persist():write(ROOT/(stage+'-execution.json'),dict(active=bool(scope and scope.active),commands=scope.commands if scope else []))
    try:
        scope=owner.Owner(ROOT/('owned-'+stage),persist)
        if stage=='host':host(ctx,row,scope,outcome)
        elif stage in ('submission','phase'):small(ctx,row,scope,stage,outcome)
        else:
            match=re.fullmatch(r'head([012])-backing([01])',stage);mode,backing=map(int,match.groups())
            local=dict(row,GPU_executed=False,
                environment=dict(row['environment'],STRATA_PREFILL_HEAD_HOST_BACKING=str(backing)))
            ctx.parse_native=lambda text,arm:parse_native(text,arm,host_backed=bool(backing))
            original_assets=ctx.model_assets
            def assets():
                print('Checking full real-shard hashes for '+stage,flush=True)
                result=original_assets();print('Real-shard hashes verified for '+stage,flush=True);return result
            ctx.model_assets=assets
            try:ctx.run_gpu(local,scope,('reference','canonical','sole-packed')[mode],outcome,label=stage)
            finally:
                outcome['GPU_executed']=bool(local.get('GPU_executed'))
            e=scope.commands[-1]
            validated_trace((scope.output/('native-'+stage+'.stderr')).read_text(),e['pid'])
            outcome.update(backing=backing,head_mode=mode,trace_errors_checked=True)
        require(outcome['passed'],'native stage predicates failed')
    except BaseException as error:outcome.update(passed=False,host_probe_accepted=False,error=repr(error))
    finally:
        outcome['ownership_closed']=not scope or scope.active is None and all(owner.closed(v) for v in scope.commands)
        outcome['passed']=bool(outcome['passed'] and outcome['ownership_closed'])
        row['results'].append(outcome);row['commands']+=scope.commands if scope else []
        row['active']=bool(scope and scope.active);row['ownership_closed']=not row['active'] and all(owner.closed(v) for v in row['commands'])
        row['GPU_executed']=bool(row['GPU_executed'] or outcome['GPU_executed'])
        row['stage']='failed' if not outcome['passed'] else 'closed' if stage==STAGES[-1] else 'in_progress'
        row['complete']=row['stage']=='closed';row['passed']=bool(row['complete'] and row['ownership_closed'])
        ctx.save(row)
    require(outcome['passed'],outcome.get('error','device stage failed'))
    print('Closed '+stage+';commit before the next stage. Model qualification remains separate.',flush=True)


if __name__=='__main__':
    require(len(sys.argv)==2 and sys.argv[1] in ('prepare',*STAGES),'fresh dense device stage')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([])
        if sys.argv[1]=='prepare':prepare()
        else:execute(sys.argv[1])
