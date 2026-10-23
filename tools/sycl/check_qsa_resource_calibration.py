#!/usr/bin/env python3
"""First small traced/codegen qualification of the isolated QSA calibration.

Main-authored,repository LICENSE. No timing or model execution in this step.
Commit prepare before run;first failure stops without retry or settings changes.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode=True
import qsa_resource_calibration as build
import generation_engine as gen
import measurement_guards as guards
import measurement_owner as supervision
import profile_attention
import resource_engine as resource
import ur_validation as validation
from measure_down_scatter import postchecks

P,B,A=gen.P,gen.B,gen.A
ROOT=build.ROOT/'check-smoke-v1'
RESULT=A/'qsa-resource-calibration-execution-current.json'


def write(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def frontend_IR():
    text=(build.ROOT/'calibration.ll').read_text()
    bodies=re.findall(r'(?ms)^(define[^\n]*spir_kernel[^\n]*\{\n.*?^})',text)
    result={}
    for body in bodies:
        name=body.splitlines()[0]
        if not name.startswith('define') or '@_ZTSZZN' not in name:continue
        match=re.search(r'ELi([012])EEE',name)
        guards.require(match is not None,'unknown calibration kernel')
        mode=int(match[1]);mad=len(re.findall(r'call[^\n]*CooperativeMatrixMulAddKHR',body))
        kv_stores=len(re.findall(r'store i16 [^\n]*ptr addrspace\(3\)',body))
        global_loads=len(re.findall(r'load [^\n]*ptr addrspace\(1\)',body))
        barriers=len(re.findall(r'call[^\n]*__spirv_ControlBarrier',body))
        guards.require(str(mode) not in result and global_loads>0,'distinct observable runtime-load kernels required')
        guards.require((mad>0) if mode==0 else (mad==0),'matrix arm codegen mismatch')
        if mode==1:guards.require(kv_stores==0 and barriers>0,'L must retain row publication,not KV SLM stores')
        if mode==2:guards.require(kv_stores>0 and barriers>=5,'S must retain KV SLM stores/five barriers')
        result[str(mode)]=dict(matrix_mad_call_sites=mad,kv_i16_SLM_store_sites=kv_stores,
            global_load_sites=global_loads,barrier_sites=barriers)
    guards.require(set(result)=={'0','1','2'},'all three frontend kernels required')
    return dict(sha256=build.digest(build.ROOT/'calibration.ll'),kernels=result,
        limit='Frontend IR only;final ISA/resource review remains required before timing.')


def current(record):
    built=gen.committed(build.RESULT);build.admission(built,binaries=True)
    guards.require(built['passed'] and built['complete'] and built['ownership_closed'] and not built['active']
        and not built['GPU_executed'] and build.digest(build.RESULT)==record['build_sha256']
        and built['binary']==record['binary'],'closed matching calibration build required')
    guards.require(gen.benchmark_pins(record['controller_source_commit'])==record['controller_inputs'],
                   'qualification controller inputs changed')
    guards.require(frontend_IR()==record['frontend_IR'] and build.runtime_lineage()==record['runtime_lineage'],
                   'runtime/source evidence changed')
    guards.require(built['device_IR']['calibration']['sha256']==record['frontend_IR']['sha256'],
                   'frontend IR differs from the recorded compiler output')


def prepare():
    guards.require(not ROOT.exists() and not RESULT.exists(),'fresh bounded smoke only')
    built=gen.committed(build.RESULT);build.admission(built,binaries=True)
    guards.require(built['stage']=='built' and built['passed'] and built['ownership_closed'] and not built['active'],
                   'complete native compile before smoke')
    _,_,setup=resource.qualification();_,restored,_=resource.modules()
    ROOT.mkdir(mode=0o700)
    for name in ('tmp','cache','isa'):(ROOT/name).mkdir()
    env=restored.environment(setup,ROOT,'0');probe=validation.preload_qualification()
    env.update(LD_PRELOAD=probe['library']['path'],STRATA_UR_PROBE_EXPECTED_LOADER=probe['loader']['path'],
        UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
        UR_LOG_TRACING='level:info;flush:info;output:stderr',
        IGC_DumpZEInfoToConsole='1',IGC_ShaderDumpEnable='1',IGC_DumpToCustomDir=str(ROOT/'isa'))
    revision=gen.git('rev-parse','HEAD').decode().strip()
    record=dict(stage='prepared',mode='small-traced-QSA-resource-calibration',controller_source_commit=revision,
        controller_inputs=gen.benchmark_pins(revision),native_source_commit=built['source_commit'],
        build_sha256=build.digest(build.RESULT),binary=built['binary'],runtime_lineage=build.runtime_lineage(),
        frontend_IR=frontend_IR(),environment=env,command=[built['binary']['path'],'--smoke'],
        prospective_gates='One fresh97-cell/3-query/cap33 process. Actual production/control full output bits,all producer checksums, '
            'runtime V perturbation,guards,input immutability,same-PID pre-main validation,normal owned exit, '
            'actual executed kernel/resource join,unchanged source/runtime/selected13assets and no fresh fault. '
            'First failure stops;no retry. Preserve final ISA for current DCE/resource review only;no timing before that review.',
        budgets=dict(wall=180,cpu=900,rss_cap=8<<30,text_cap=64<<20,file_cap=64<<20,total_cap=512<<20),
        retention='Successful API traces and JIT dumps only through immediate final-codegen/resource review. '
            'Retire after compact extraction/verification. No tensors,model or realID replay.',
        active=False,ownership_closed=True,passed=False,complete=False,GPU_executed=False,
        timing_launch_permitted=False,model_adoption_qualified=False,hardware_saturation_qualified=False)
    current(record);write(ROOT/'plan.json',record);write(RESULT,record)
    print('Prepared one traced small calibration check;commit before execution.',flush=True)


def run():
    record=gen.committed(RESULT)
    guards.require(record==json.loads((ROOT/'plan.json').read_text()) and record['stage']=='prepared'
        and not (ROOT/'record.json').exists(),'fresh committed smoke required')
    current(record)
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    before=subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15)
    cursor=next(x.split(': ',1)[1] for x in before.splitlines() if x.startswith('-- cursor: '))
    owner=None
    def persist():
        record['active']=bool(owner and owner.active is not None)
        write(ROOT/'record.json',dict(record,owned_commands=owner.commands if owner else []))
    owner=supervision.Owner(ROOT/'owned',persist)
    try:
        entry,out,err=owner.run('smoke',record['command'],record['environment'],P,**record['budgets'])
        record['GPU_executed']=True
        guards.require(supervision.closed(entry) and entry['exit_code']==0,'small fixture did not close normally')
        text=out.read_text();trace=err.read_text()
        guards.require(text.count('case cells=97 queries=3 cap=33 pattern=gather control_bits=PASS '
            'producer_checksums=PASS perturbation=PASS guards=PASS immutable=PASS')==1 and
            text.count('fixture=PASS model=UNRUN physical_counters=UNMEASURED')==1,'missing exact fixture checks')
        activation=validation.preload_evidence(trace,entry['pid'])
        body=trace.split(f'strata UR validation: pid={entry["pid"]} phase=before-main',1)[1]
        guards.require(not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|Strata SYCL cleanup:',body),
                       'native runtime error')
        emitted=profile_attention.emitted_resources(trace+text,trace)
        executed=[x for x in emitted if x['observed_launches']]
        guards.require(len(executed)==8 and all('prefix_batch_impl' in x['kernel'] for x in executed),
                       'actual eight preparation/attention/probe kernels required')
        resources=[]
        for item in executed:
            name=item['kernel'];mode=re.search(r'ELi([012])EEE',name)
            label=('control' if mode[1]=='0' else 'load' if mode[1]=='1' else 'stage') if mode else 'production'
            resources.append(dict(arm=label,phase='attention' if name.startswith('_ZTSZZN') else 'prefix',**item))
        current(record)
        files=[p for p in (ROOT/'isa').rglob('*') if p.is_file()]
        guards.require(files,'final native codegen dumps missing')
        record.update(passed=True,complete=True,validation_activation=activation,native_identity=entry['direct_identity'],
            checks=dict(control_bits=True,producer_checksums=True,perturbation=True,guards=True,immutable=True),
            resources=resources,codegen_files=dict(files=len(files),bytes=sum(p.stat().st_size for p in files)),
            final_ISA_review_pending=True)
    except BaseException as error:record.update(error=repr(error),passed=False,complete=False)
    finally:
        record['ownership_closed']=owner.active is None and all(supervision.closed(c) for c in owner.commands)
        postchecks(record,boot,cursor);record['stage']='closed';persist();write(RESULT,record)
    print(json.dumps({k:record.get(k) for k in ('passed','complete','ownership_closed','fresh_faults','error')}),flush=True)
    return 0 if record['passed'] else 1


if __name__=='__main__':
    guards.require(len(sys.argv)==2 and sys.argv[1] in ('prepare','run'),'use prepare|run')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([]);guards.driver_guard()
        sys.exit(prepare() if sys.argv[1]=='prepare' else run())
