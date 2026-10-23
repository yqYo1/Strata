#!/usr/bin/env python3
"""Whole-production-path locality sensitivity;not model speed or saturation.

Main-authored,repository LICENSE. Owned cohort pattern follows1e027ed29.
"""
import fcntl
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys

sys.dont_write_bytecode=True
import check_qsa_locality as checked
import qsa_locality_build as build
import generation_engine as gen
import measurement_guards as guards
import measurement_owner as supervision
import ur_validation as validation
from measure_down_scatter import postchecks

P,B,A=gen.P,gen.B,gen.A
ROOT=build.ROOT/'measure-v1'
RESULT=A/'qsa-locality-timing-current.json'


def write(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def admission(record):
    small=gen.committed(checked.RESULT);checked.current(small)
    guards.require(small['passed'] and small['complete'] and small['ownership_closed'] and
        not small['active'] and not small['fresh_faults'] and small['codegen_review']['passed'] and
        not small['final_ISA_review_pending'] and build.digest(checked.RESULT)==record['small_sha256'],
        'closed small/native-ISA qualification required')
    guards.require(gen.benchmark_pins(record['controller_source_commit'])==record['controller_inputs'],
                   'measurement controller inputs changed')


def prepare():
    guards.require(not ROOT.exists() and not RESULT.exists(),'fresh single-use cohort only')
    small=gen.committed(checked.RESULT);checked.current(small)
    guards.require(small['codegen_review']['passed'] and not small['final_ISA_review_pending'],
                   'final codegen review before full cohort')
    ROOT.mkdir(mode=0o700);jobs=[]
    for i in range(4):
        run=ROOT/str(i);run.mkdir()
        for name in ('tmp','cache'):(run/name).mkdir()
        env=dict(small['environment'])
        for key in list(env):
            if key.startswith(('UR_LOG_','IGC_')) or key in ('UR_ENABLE_LAYERS','LD_PRELOAD','STRATA_UR_PROBE_EXPECTED_LOADER'):
                del env[key]
        env.update(TMPDIR=str(run/'tmp'),XDG_CACHE_HOME=str(run/'cache'),STRATA_TRACE='0')
        if i==0:
            probe=small['runtime_lineage']['native_validation_probe']
            env.update(LD_PRELOAD=probe['library']['path'],STRATA_UR_PROBE_EXPECTED_LOADER=probe['loader']['path'],
                UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION',
                UR_LOG_VALIDATION='level:warning;flush:warning;output:stderr',
                UR_LOG_LOADER='level:warning;flush:warning;output:stderr',
                UR_LOG_LEVEL_ZERO='level:warning;flush:warning;output:stderr',STRATA_TRACE='1')
        for key,value in dict(NEOReadDebugKeys='1',EnableDirectSubmission='0',UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1',SYCL_CACHE_PERSISTENT='0').items():
            guards.require(env.get(key)==value,'runtime safety setting changed')
        command=[small['binary']['path'],'--validate'] if i==0 else [small['binary']['path'],'--timing',str(i-1)]
        jobs.append(dict(index=i,command=command,environment=env))
    cells,queries,cap=117488,512,2051;blocks=cells//32
    unions=[cap,len({((b*11+c//32)%blocks)*32+c%32 for b in range(queries) for c in range(cap)}),
        len({((b*8191+c*7919)%blocks)*32+c%32 for b in range(queries) for c in range(cap)})]
    revision=gen.git('rev-parse','HEAD').decode().strip()
    record=dict(stage='prepared',controller_source_commit=revision,controller_inputs=gen.benchmark_pins(revision),
        native_source_commit=small['native_source_commit'],binary=small['binary'],small_sha256=build.digest(checked.RESULT),
        jobs=jobs,logical_union_rows=unions,
        prospective_gates='One parameter-validated117488/512/2051 cohort covering mixed and uniform widths;then three clean '
            'processes,15cyclically balanced128-call batches each. All output bits and V perturbation match across '
            'three valid unique-ID masks;guards/immutability pass. Batch>=100ms and>=1000median timer overhead; '
            'equal two-call warmups. Same native/source/runtime and selected13assets,no fresh fault before next process. '
            'First failure stops,no retry/adaptive enlargement. For each process,scattered/common median>=1.10 '
            'qualifies a substantial whole-path address/working-set sensitivity only;otherwise unresolved. '
            'No cache-residency,physical-traffic,saturation or model-adoption claim.',
        budgets=dict(wall=300,cpu=1500,rss_cap=8<<30,text_cap=32<<20,file_cap=128<<20,total_cap=256<<20),
        results=[],samples=[],active=False,ownership_closed=True,passed=False,complete=False,GPU_executed=False,
        clean_model_speed_qualified=False,hardware_saturation_qualified=False,model_adoption_qualified=False,
        retention='Keep compact individual timings/configuration/decision. Retire successful raw output/reports and '
            'extra fixture at closure after exact source/recipe recovery;selected6e40/decode unchanged.')
    admission(record);write(ROOT/'plan.json',record);write(RESULT,record)
    print('Prepared full validation/three clean locality processes;commit before execution.',flush=True)


def run():
    record=gen.committed(RESULT)
    guards.require(record==json.loads((ROOT/'plan.json').read_text()) and record['stage']=='prepared'
        and not (ROOT/'record.json').exists(),'fresh committed plan required')
    admission(record);boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    journal=subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15)
    cursor=next(x.split(': ',1)[1] for x in journal.splitlines() if x.startswith('-- cursor: '))
    owner=None
    def persist():
        record['active']=bool(owner and owner.active is not None)
        write(ROOT/'record.json',dict(record,owned_commands=owner.commands if owner else []))
    owner=supervision.Owner(ROOT/'owned',persist)
    try:
        for job in record['jobs']:
            print('Running validated cohort' if job['index']==0 else f'Running clean locality process {job["index"]}/3',flush=True)
            entry,out,err=owner.run('cohort-'+str(job['index']),job['command'],job['environment'],P,**record['budgets'])
            record['GPU_executed']=True;text=out.read_text();trace=err.read_text();samples=[]
            for line in text.splitlines():
                if line.startswith('sample '):
                    row=dict(x.split('=',1) for x in line.split()[1:])
                    for k in ('process','arm','round','order','calls','elapsed_ns','timer_pair_ns','eligible'):row[k]=int(row[k])
                    samples.append(row)
            record['samples'].extend(samples);persist()
            guards.require(supervision.closed(entry) and entry['exit_code']==0,'cohort did not close normally')
            cases=re.findall(r'^case cells=117488 queries=512 cap=2051 uniform=([01]) output_bits=PASS '
                r'perturbation=PASS guards=PASS immutable=PASS unique_ids=PASS unions=(\d+),(\d+),(\d+)$',text,re.M)
            guards.require([int(c[0]) for c in cases]==([0,1] if job['index']==0 else [1]) and
                all(list(map(int,c[1:]))==record['logical_union_rows'] for c in cases) and
                text.count('fixture=PASS model=UNRUN physical_counters=UNMEASURED')==1,'full cohort checks missing')
            if job['index']==0:
                guards.require(not samples,'validation must not time')
                activation=validation.preload_evidence(trace,entry['pid'])
                body=trace.split(f'strata UR validation: pid={entry["pid"]} phase=before-main',1)[1]
                guards.require(not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|Strata SYCL cleanup:',body),'validation error')
            else:
                activation=None
                guards.require(len(samples)==15 and all(x['process']==job['index']-1 and x['calls']==128 and x['eligible']==1
                    and x['elapsed_ns']>=100000000 and x['elapsed_ns']>=1000*x['timer_pair_ns'] and x['checks']=='PASS'
                    for x in samples),'ineligible timing batches')
                guards.require([(x['round'],x['order'],x['arm']) for x in samples]==
                    [(r,o,(job['index']-1+r+o)%3) for r in range(5) for o in range(3)],'timing order differs')
            postchecks(record,boot,cursor)
            guards.require(record['postchecks_complete'] and record['boot_unchanged'] and not record['fresh_faults'],
                           'fault/boot gate before next process')
            admission(record);guards.driver_guard()
            record['results'].append(dict(index=job['index'],pid=entry['pid'],exit_code=entry['exit_code'],
                normal_owned_closure=True,validation_activation=activation,cases=len(cases),samples=len(samples)))
            persist()
        medians=[{a:statistics.median(s['elapsed_ns']/s['calls'] for s in record['samples'] if s['process']==p and s['arm']==a)
            for a in range(3)} for p in range(3)]
        ratios=[m[2]/m[0] for m in medians]
        record.update(passed=True,complete=True,process_medians_ns=medians,scattered_over_common=ratios,
            scattered_over_common_geomean=math.exp(statistics.mean(map(math.log,ratios))),
            substantial_address_sensitivity=all(r>=1.10 for r in ratios))
    except BaseException as error:record.update(error=repr(error),passed=False,complete=False)
    finally:
        record['ownership_closed']=owner.active is None and all(supervision.closed(c) for c in owner.commands)
        postchecks(record,boot,cursor);record['stage']='closed';persist();write(RESULT,record)
    print(json.dumps({k:record.get(k) for k in ('passed','complete','ownership_closed','fresh_faults','substantial_address_sensitivity','error')}),flush=True)
    return 0 if record['passed'] else 1


if __name__=='__main__':
    guards.require(len(sys.argv)==2 and sys.argv[1] in ('prepare','run'),'use prepare|run')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([]);guards.driver_guard()
        sys.exit(prepare() if sys.argv[1]=='prepare' else run())
