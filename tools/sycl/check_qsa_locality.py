#!/usr/bin/env python3
"""Small traced qualification of Main's equal-operand locality diagnostic.

Repository LICENSE. Ownership/validation pattern follows17a703e60 project code.
No timing or model execution. Prepare and commit before the single-use run.
"""
import fcntl
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode=True
import qsa_locality_build as build
import generation_engine as gen
import measurement_guards as guards
import measurement_owner as supervision
import profile_attention
import resource_engine as resource
import ur_validation as validation
from measure_down_scatter import postchecks

P,B,A=gen.P,gen.B,gen.A
ROOT=build.ROOT/'smoke-v1'
RESULT=A/'qsa-locality-check-current.json'
SCOPE='qsa_samecode_equal_operand_locality_audit_20261023'


def write(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def research_qualification(built):
    path=A/'research-index.json'
    current=json.loads(path.read_text())[SCOPE]
    committed=json.loads(gen.git('show','HEAD:'+str(path.relative_to(P))))[SCOPE]
    # Only this verified subtree is consumed;unrelated research rows are not
    # frozen by a locality run. Changed/missing/type-altered evidence still fails.
    guards.require(current==committed,'commit exact locality research qualification')
    guards.require(current['status']=='reviewed_decision_support' and current['worker_ownership_closed']
        and current['main_resolution']['small_GPU_admission'] is True,
        'closed report/native Luna-max review and Main resolution required before GPU')
    review=current['review']
    guards.require(build.digest(current['report'])==current['report_sha256'] and
        build.digest(P/review['path'])==review['sha256'] and review['model']=='native Codex gpt-6-luna max' and
        current['source_fixture_sha256']==built['inputs']['bench/sycl/qsa_locality.cpp'],
        'research report/review/source changed')
    return {k:current[k] for k in ('report','report_sha256','review','source_fixture_sha256','main_resolution')}


def current(record):
    built=gen.committed(build.RESULT);build.admission(built,binaries=True)
    guards.require(built['passed'] and built['complete'] and built['ownership_closed'] and not built['active']
        and not built['GPU_executed'] and build.digest(build.RESULT)==record['build_sha256']
        and built['binary']==record['binary'],'closed matching native build required')
    guards.require(gen.benchmark_pins(record['controller_source_commit'])==record['controller_inputs'],
                   'qualification controller inputs changed')
    guards.require(build.digest(build.ROOT/'production.ll')==record['production_IR_sha256'] and
        build.runtime_lineage()==record['runtime_lineage'],'IR/runtime evidence changed')
    guards.require(research_qualification(built)==record['research_qualification'],
                   'reviewed locality admission changed')


def prepare():
    guards.require(not ROOT.exists() and not RESULT.exists(),'fresh bounded smoke only')
    built=gen.committed(build.RESULT);build.admission(built,binaries=True)
    guards.require(built['stage']=='built' and built['passed'] and built['ownership_closed'] and not built['active'],
                   'complete native compile before smoke')
    research=research_qualification(built)
    _,_,setup=resource.qualification();_,restored,_=resource.modules()
    ROOT.mkdir(mode=0o700)
    for name in ('tmp','cache','isa'):(ROOT/name).mkdir()
    env=restored.environment(setup,ROOT,'0');probe=validation.preload_qualification()
    env.update(LD_PRELOAD=probe['library']['path'],STRATA_UR_PROBE_EXPECTED_LOADER=probe['loader']['path'],
        UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
        UR_LOG_TRACING='level:info;flush:info;output:stderr',
        IGC_DumpZEInfoToConsole='1',IGC_ShaderDumpEnable='1',IGC_DumpToCustomDir=str(ROOT/'isa'))
    revision=gen.git('rev-parse','HEAD').decode().strip()
    record=dict(stage='prepared',controller_source_commit=revision,controller_inputs=gen.benchmark_pins(revision),
        native_source_commit=built['source_commit'],build_sha256=build.digest(build.RESULT),binary=built['binary'],
        runtime_lineage=build.runtime_lineage(),production_IR_sha256=build.digest(build.ROOT/'production.ll'),
        research_qualification=research,
        environment=env,command=[built['binary']['path'],'--smoke'],
        prospective_gates='One97/3/33 mixed-width traced process. All output bits match across common/shifted/scattered '
            'unique valid masks;V perturbation changes output identically;guards/input immutability. Same-PID pre-main '
            'validation and normal owned exit. Exactly two actual production prefix/attention kernels. Attention '
            'SIMD16/GRF128/dynamicSLM20032/spill12480/rawSLM32 match closed calibration;inspect final executed ISA. '
            'Unchanged source/runtime/selected13assets,no fresh fault. First failure stops;no retry or timing.',
        budgets=dict(wall=180,cpu=900,rss_cap=8<<30,text_cap=64<<20,file_cap=64<<20,total_cap=512<<20),
        retention='Successful trace/JIT only through immediate codegen/ownership review;retire after compact evidence. '
            'No model,tensor capture or actual-ID replay.',
        active=False,ownership_closed=True,passed=False,complete=False,GPU_executed=False,
        timing_launch_permitted=False,model_adoption_qualified=False,hardware_saturation_qualified=False)
    current(record);write(ROOT/'plan.json',record);write(RESULT,record)
    print('Prepared one small traced locality check;commit before execution.',flush=True)


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
        cases=re.findall(r'^case cells=97 queries=3 cap=33 uniform=0 output_bits=PASS perturbation=PASS '
            r'guards=PASS immutable=PASS unique_ids=PASS unions=(\d+),(\d+),(\d+)$',text,re.M)
        guards.require(len(cases)==1 and cases[0][0]=='33' and text.count('fixture=PASS model=UNRUN physical_counters=UNMEASURED')==1,
                       'exact fixture checks missing')
        activation=validation.preload_evidence(trace,entry['pid'])
        body=trace.split(f'strata UR validation: pid={entry["pid"]} phase=before-main',1)[1]
        guards.require(not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|Strata SYCL cleanup:',body),
                       'native runtime error')
        emitted=profile_attention.emitted_resources(trace+text,trace)
        executed=[x for x in emitted if x['observed_launches']]
        guards.require(len(executed)==2 and all('prefix_batch_implILi33ELb0ELb0ELb1ELb1ELb1ELb1ELb0EEE' in x['kernel'] for x in executed),
                       'only actual selected production prefix/attention kernels required')
        attention=[x for x in executed if x['kernel'].startswith('_ZTSZZN')]
        guards.require(len(attention)==1,'one production attention image required')
        env=attention[0]['execution_env'];launch=attention[0]['observed_launches']
        guards.require(all(env.get(k)==v for k,v in dict(simd_size='16',grf_count='128',spill_size='12480',slm_size='32').items())
            and launch['dynamic_local_bytes']==20032,'production resource descriptors changed')
        current(record)
        files=[p for p in (ROOT/'isa').rglob('*') if p.is_file()];guards.require(files,'executed native dumps absent')
        record.update(passed=True,complete=True,validation_activation=activation,resources=executed,
            checks=dict(output_bits=True,perturbation=True,guards=True,immutable=True,unique_ids=True,logical_unions=list(map(int,cases[0]))),
            codegen_files=dict(files=len(files),bytes=sum(p.stat().st_size for p in files)),final_ISA_review_pending=True)
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
