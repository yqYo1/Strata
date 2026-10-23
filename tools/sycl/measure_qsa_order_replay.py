#!/usr/bin/env python3
"""Actual-mask/synthetic-operand whole-query order comparison; repository LICENSE.

Main-owned. One full parameter-validated process then three clean processes.
This selects a mechanism worth implementing on GPU,not model adoption.
"""
import fcntl
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys

sys.dont_write_bytecode=True
import capture_qsa_locality as capture
import check_qsa_order_replay as checked
import generation_engine as gen
import measurement_guards as guards
import measurement_owner as supervision
import ur_validation as validation
from measure_down_scatter import postchecks

P,B,A=gen.P,gen.B,gen.A
ROOT=checked.replay.ROOT/'actual-v1'
RESULT=A/'qsa-actual-order-current.json'
SCOPE='actual_qsa_order_cohort_controller_audit_20261023'


def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def review():
    row=gen.committed(A/'research-index.json')[SCOPE]
    guards.require(row.get('source_gate_passed') is True,'actual replay cohort review required')
    for item in row['review'].values():guards.require(gen.common.digest(item['path'])==item['sha256'],'cohort review changed')
    return row['review']


def captured():
    origin=gen.committed(capture.RESULT)
    guards.require(origin['stage']=='closed' and origin['passed'] and origin['complete'] and origin['ownership_closed'] and
        not origin['active'] and origin['fresh_faults']==[] and origin['boot_unchanged'] and origin['postchecks_complete'] and
        origin['actual_host_check_passed'] and origin['warmup_capture_absent'] and origin['effective_environment_verified'],
        'closed provenance-bound actual capture and host check required')
    capture.current(origin)
    ref=origin['completion'];guards.require(gen.common.digest(ref['path'])==ref['sha256'],'completion changed')
    completed=json.loads(Path(ref['path']).read_text())
    expected=dict(state='complete',transaction_id=origin['transaction_id'],native_source=origin['build_source_commit'],
        binary=origin['binary'],native_identity=origin['native_identity'],boot_id=origin['machine_identity']['boot_id'],
        request_sha256=origin['request']['request_sha256'],journal_sha256=origin['journal_sha256'],
        config_sha256=origin['generated_inputs'][str(capture.ROOT/'config.json')],tail=origin['tail'],files=origin['files'],
        normal_owned_exit=True,actual_host_check_passed=True)
    guards.require(completed==expected,'capture/session/request completion binding differs')
    for suffix,item in origin['files'].items():
        guards.require(item['path']==str(capture.BASE)+suffix and Path(item['path']).stat().st_size==item['bytes'] and
            gen.common.digest(item['path'])==item['sha256'],'captured file bytes changed')
    guards.require(capture.bind_tail(origin['files'],origin['tail']['cells'])==origin['tail'],'actual request extent differs')
    return origin


def admission(record):
    origin=captured();built=checked.native()
    guards.require(gen.common.digest(capture.RESULT)==record['capture_sha256'] and origin['completion']==record['completion'] and
        origin['tail']==record['tail'] and built['binary']==record['binary'],'actual replay provenance changed')
    guards.require(gen.benchmark_pins(record['controller_source_commit'])==record['controller_inputs'],
        'measurement controller inputs changed')
    guards.require(review()==record['review'],'cohort review qualification changed')


def prepare():
    guards.require(not ROOT.exists() and not RESULT.exists(),'fresh single-use actual replay cohort')
    origin=captured();built=checked.native();ROOT.mkdir(mode=0o700);jobs=[]
    for i in range(4):
        root=ROOT/str(i);root.mkdir()
        for name in ('cache','tmp'):(root/name).mkdir()
        env=dict(built['environment'],XDG_CACHE_HOME=str(root/'cache'),TMPDIR=str(root/'tmp'),STRATA_TRACE='0')
        guards.require(not any(k in env for k in ('LD_PRELOAD','UR_ENABLE_LAYERS')),'clean base environment required')
        if i==0:
            probe=origin['diagnostic_qualification']['runtime_lineage']['native_validation_probe']
            env.update(LD_PRELOAD=probe['library']['path'],STRATA_UR_PROBE_EXPECTED_LOADER=probe['loader']['path'],
                UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION',STRATA_TRACE='1',
                UR_LOG_VALIDATION='level:warning;flush:warning;output:stderr',
                UR_LOG_LOADER='level:warning;flush:warning;output:stderr',UR_LOG_LEVEL_ZERO='level:warning;flush:warning;output:stderr')
        for k,v in dict(UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1',NEOReadDebugKeys='1',EnableDirectSubmission='0',SYCL_CACHE_PERSISTENT='0').items():
            guards.require(env.get(k)==v,'runtime safety setting changed')
        command=[built['binary']['path'],'--validate',str(capture.BASE)] if i==0 else [built['binary']['path'],'--timing',str(i-1),str(capture.BASE)]
        jobs.append(dict(index=i,command=command,environment=env))
    revision=gen.git('rev-parse','HEAD').decode().strip()
    record=dict(stage='prepared',controller_source_commit=revision,controller_inputs=gen.benchmark_pins(revision),
        review=review(),
        native_source_commit=built['source_commit'],binary=built['binary'],capture_sha256=gen.common.digest(capture.RESULT),
        completion=origin['completion'],tail=origin['tail'],jobs=jobs,processes=[],samples=[],
        sample_columns=['layer','round','order','arm','elapsed_ns','timer_pair_ns'],sample_calls=32,
        prospective_gates='Require exact committed actual capture/session/request/file hashes and passed actual-file host check. '
            'One full parameter-validated12-layer replay then three clean independent processes. '
            'Each process:five cyclically rotated rounds,all three arms,32 production-wrapper calls/batch,>=100ms '
            'and>=1000median timer pair overhead,equal two-call warmups. Original-row output bits,V perturbation, '
            'guards and immutable inputs pass every layer/arm.540timing samples total. '
            'For each method separately,all three processes must have sum_of_layer_medians(method)/sum_of_layer_medians(natural)<=0.97 '
            'and no per-layer/process median ratio>1.05 to justify a future GPU-local implementation experiment. '
            'Reverse is a control and eligible mechanism by the same rule. No model trial from this CPU-built-order result. '
            'Stop on first source/runtime/file/validation/owner/fault/budget failure;no retry or adaptive batches.',
        limits='Actual last512-query masks/steps with synthetic finite Q/K/V,identity pages and full prefix pack. '
            'Same per-query operands/order/output ownership and same device allocations across arms. CPU grouping/uploads/restore '
            'excluded;SYCL does not guarantee relative workgroup order/co-residency/cache behavior. '
            'Sum of isolated layer medians is not end-to-end model time;no physical traffic or saturation measurement. '
            'Actual model final chunk may have more than512queries. Selected6e40/decode1256slots unchanged.',
        budgets=dict(wall=600,cpu=2400,rss_cap=8<<30,text_cap=32<<20,file_cap=32<<20,total_cap=128<<20),
        complete=False,passed=False,active=False,ownership_closed=True,GPU_executed=False,
        clean_model_speed_qualified=False,hardware_saturation_qualified=False,model_adoption_qualified=False,
        retention='Keep useful individual samples/criteria/source/provenance. Retire actual paired files,successful raw '
            'traces and additional diagnostic generation after mechanism decision and exact source/recipe recovery.')
    admission(record);write(ROOT/'plan.json',record);write(RESULT,record)


def output(text,index,tail):
    cases=re.findall(r'^case layer=(\d+) first=(\d+) cells=(\d+) queries=512 cap=2051 cpu_order_ns=(\d+) '
        r'output_bits=PASS perturbation=PASS guards=PASS immutable=PASS$',text,re.M)
    guards.require(len(cases)==12 and [int(x[0]) for x in cases]==list(range(12)) and
        all(int(x[1])==tail['first'] and int(x[2])==tail['cells'] for x in cases) and
        text.count('fixture=PASS model=UNRUN physical_counters=UNMEASURED cpu_order=EXCLUDED')==1,
        'full actual-mask replay checks missing')
    samples=[]
    for line in text.splitlines():
        if not line.startswith('sample '):continue
        fields=line.split()[1:];row=dict(s.split('=',1) for s in fields)
        guards.require(len(fields)==len(row)==10 and row.pop('checks')=='PASS','sample checks/fields')
        row={k:int(v) for k,v in row.items()}
        guards.require(set(row)=={'layer','process','arm','round','order','calls','elapsed_ns','timer_pair_ns','eligible'} and
            row['process']==index-1 and row['calls']==32 and row['eligible']==1 and row['timer_pair_ns']>=0 and row['elapsed_ns']>=100000000 and
            row['elapsed_ns']>=1000*row['timer_pair_ns'] and row['arm']==(row['process']+row['round']+row['order'])%3,
            'ineligible timing batch')
        samples.append(row)
    if index==0:guards.require(not samples,'validation must not time');return [],None
    keys={(r['layer'],r['round'],r['order']) for r in samples}
    guards.require(len(samples)==180 and keys=={(l,r,o) for l in range(12) for r in range(5) for o in range(3)},'timing conservation')
    medians=[[statistics.median(r['elapsed_ns']/32 for r in samples if r['layer']==l and r['arm']==a) for a in range(3)] for l in range(12)]
    sums=[sum(row[a] for row in medians) for a in range(3)]
    worst=[max(row[a]/row[0] for row in medians) for a in range(3)]
    ratio=[v/sums[0] for v in sums]
    summary=dict(process=index-1,sum_of_isolated_layer_medians_ns=sums,ratios=ratio,worst_layer_ratios=worst,
        passing_methods=[a for a in (1,2) if ratio[a]<=.97 and worst[a]<=1.05])
    return [[r[k] for k in ['layer','round','order','arm','elapsed_ns','timer_pair_ns']] for r in samples],summary


def run():
    record=gen.committed(RESULT)
    guards.require(record==json.loads((ROOT/'plan.json').read_text()) and record['stage']=='prepared' and
        not (ROOT/'record.json').exists(),'fresh committed actual cohort')
    admission(record);boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    before=subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15)
    cursor=next(x.split(': ',1)[1] for x in before.splitlines() if x.startswith('-- cursor: '));owner=None
    def persist():
        record['active']=bool(owner and owner.active is not None)
        write(ROOT/'record.json',dict(record,owned_commands=owner.commands if owner else []))
    owner=supervision.Owner(ROOT/'owned',persist);persist()
    try:
        for job in record['jobs']:
            entry,out,err=owner.run('process-'+str(job['index']),job['command'],job['environment'],P,**record['budgets'])
            record['GPU_executed']=True
            guards.require(supervision.closed(entry) and entry['exit_code']==0,'actual replay did not close normally')
            samples,summary=output(out.read_text(),job['index'],record['tail']);record['samples'].append(samples)
            if job['index']==0:
                trace=err.read_text();activation=validation.preload_evidence(trace,entry['pid'])
                body=trace.split(f'strata UR validation: pid={entry["pid"]} phase=before-main',1)[1]
                guards.require(not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|Strata SYCL cleanup:',body),'runtime validation error')
                record['activation']=activation
            record['processes'].append(dict(index=job['index'],exit_code=0,summary=summary))
            persist();admission(record);probe=dict(passed=True,ownership_closed=True);postchecks(probe,boot,cursor)
            guards.require(probe['passed'],'fresh fault/incomplete observation;stop before next process')
        record['passing_methods']=[a for a in (1,2) if all(a in r['summary']['passing_methods'] for r in record['processes'][1:])]
        record.update(passed=True,complete=True,decision='GPU-local implementation experiment justified' if record['passing_methods'] else 'Neither order meets prospective mechanism gate;no model trial')
    except BaseException as error:record['error']=repr(error)
    finally:
        record['ownership_closed']=owner.active is None and all(supervision.closed(e) for e in owner.commands)
        postchecks(record,boot,cursor);record['stage']='closed';persist();write(RESULT,record)
    print(json.dumps({k:record.get(k) for k in ('passed','complete','ownership_closed','passing_methods','fresh_faults','error')}),flush=True)
    return 0 if record['passed'] else 1


if __name__=='__main__':
    guards.require(len(sys.argv)==2 and sys.argv[1] in ('prepare','run'),'prepare|run')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([]);guards.driver_guard()
        sys.exit(prepare() if sys.argv[1]=='prepare' else run())
