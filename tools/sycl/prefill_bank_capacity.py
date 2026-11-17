#!/usr/bin/env python3
"""One-variable 384/512 prefill-bank comparison on the coherent14508 engine.

Main-authored from owned benchmark controls; repository LICENSE. No native
rebuild, queue/math/decode change, XeStrata input or model adoption.
"""
import copy
import fcntl
import json
import math
from pathlib import Path
import re
import socket
import statistics
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
import coherent_original_reuse as original
import coherent_prefill_bench as reference
import compare_official_prefill as comparison
import qsa16_stream_bench as core
import ur_validation as validation

P, B, RESULT = reference.P, reference.B, reference.RESULT
ROOT, KEY = B/'prefill-bank-capacity-v1', 'prefill_bank_capacity_v1'
THIS = 'tools/sycl/prefill_bank_capacity.py'
PYTHON, RUNS = reference.model.PYTHON, 3
# First establish actual512 admission at the larger residual size. Then three
# adjacent, alternating pairs per length; no queue or scheduling change.
PAIR_READS = (32769, 65537, 65537, 32769, 32769, 65537)
CASES = [(65537,512,True)] + [(n,s,False) for i,n in enumerate(PAIR_READS)
    for s in ((512,384) if i in (2,3) else (384,512))]
ORDER = [(n,1) for n,_,_ in CASES]
STRIDE = 2662400
POLICIES = ('AGENTS.md','docs/ARTIFACT_RETENTION.md','docs/BENCHMARKS.md')
require, sha, signature = original.require, original.sha, original.fingerprint
compact = core.compact


def save(row):
    core.write(ROOT/'record.json',row)
    values=json.loads(RESULT.read_text());values[KEY]=compact(row);core.write(RESULT,values)


def basis(expected=None):
    native,proof=original.verify_original(expected=expected)
    prior=json.loads((reference.ROOT/'record.json').read_text())
    require(validation.committed(RESULT,(reference.KEY,))==compact(prior)
        and prior['stage']=='closed' and not prior['active']
        and all(prior[k] for k in ('passed','complete','ownership_closed'))
        and prior['native_source_commit']==original.SOURCE
        and prior['binary']==native['binaries']['strata'], 'original closed speed reference required')
    runtime=prior['runtime_manifest'];files=json.loads(Path(runtime['path']).read_text())
    require(sha(runtime['path'])==runtime['sha256'] and all(sha(p)==h for p,h in files.items())
        and comparison.build.guards.driver_guard()==prior['driver']
        and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==prior['boot'],
        'original runtime,boot or driver changed')
    comparison.shared.models.model_pins(prior['configuration'],prior['model_assets'],full=False)
    return native,proof,prior,files


def environment(row,plan):
    env=comparison.environment(row,plan)
    env.update(STRATA_SYCL_PREFILL_EXPERT_BANK_SLOTS=str(plan['slots']),
        STRATA_SYCL_PREFILL_MOE_CENSUS='0',STRATA_PREFILL_LAYER_MAJOR_PROFILE='0')
    if plan['trace_progress']:env['STRATA_TRACE']='1'
    require(env['STRATA_SYCL_PREFILL_SAME_QUEUE']=='1'
        and env['STRATA_SYCL_PREFILL_EXPERT_UPLOAD_STAGING']=='0'
        and env['STRATA_SYCL_PREFILL_EXPERT_BANK_PACKED']=='1', 'fixed transfer/bank policy required')
    return env


def command(plan):
    root=Path(plan['root'])
    return ['/bin/sh',str(P/'bench.sh'),'--config',str(root/'config.json'),'--port',str(plan['port']),
        '--prefill-exact-tokens',str(plan['read']),'--think','--runs',str(RUNS),
        '--profile-request-gates','--workload-id','native_gdn_20261018',
        '--fail-fast-engine-error','--ready-timeout','600','--out',str(root/'bench')]


def invariant(row,plan):
    env=environment(row,plan);env.pop('STRATA_TRACE',None)
    env['STRATA_SYCL_PREFILL_EXPERT_BANK_SLOTS']='<slots>'
    return signature(json.dumps(dict(config=row['configuration'],env=env),sort_keys=True)
        .replace(plan['root'],'<process>').replace(str(ROOT),'<campaign>'))


def current(row):
    require(validation.committed(RESULT,(KEY,))==compact(row)
        and core.pins(row['source_commit'])==row['sources']
        and all(sha(P/p)==h for p,h in row['policies'].items()), 'commit/freeze exact controller and plan')
    native,proof,prior,files=basis(row['original_native_proof'])
    require(row['reference_evidence_sha256']==signature(compact(prior))
        and row['binary']==native['binaries']['strata']
        and row['native_source_commit']==original.SOURCE
        and all(row[k]==prior[k] for k in ('configuration','model_assets','template','boot',
            'driver','native_validation_probe','python_contract','python_selected'))
        and row['environment_template']==prior['environment_template'], 'original model/configuration changed')
    contract,python=core.frontend.python_manifest()
    require(contract==row['python_contract'] and python['selected']==row['python_selected']
        and sha(row['template']['path'])==row['template']['sha256']
        and comparison.build.supervision.artifact_bytes(ROOT)<=row['whole_bytes']
        and len(row['plans'])==len(CASES), 'frontend/template/output budget changed')
    for i,plan in enumerate(row['plans']):
        root=Path(plan['root'])
        require(plan['index']==i and (plan['read'],plan['slots'],plan['trace_progress'])==CASES[i]
            and plan['heads']==1 and plan['command']==command(plan)
            and invariant(row,plan)==row['invariant_sha256']
            and json.loads((root/'config.json').read_text())==dict(row['configuration'],log=str(root/'engine.stderr'))
            and json.loads((root/'plan.json').read_text())==dict(binary=row['binary'],native_validation_probe=row['native_validation_probe'])
            and all(sha(root/p)==h for p,h in plan['generated'].items())
            and comparison.fingerprint(environment(row,plan))==plan['environment_sha256'],
            'unplanned configuration,environment,workload or native image')
    return files,python


def request_route(text,plan):
    result=core.request_route(text,plan)  # CLEAN_FAST=False: no legacy384-only gate.
    bank=comparison.route(text,plan['read'],1,expected_rows=plan['read']-1,
        requested_slots=plan['slots'],draft_bytes=0)
    require(bank['residual_gpu_rows']==plan['read']-1 and not bank['split']
        and bank['staged'] and bank['prefill_slots']==plan['slots']
        and result['layer_cache_bytes']==plan['slots']*STRIDE,
        'actual allocation/all-GPU residual/ordinary route differs')
    require('softmax=parallel-exp-ascending-sum' in text and 'softmax=subgroup' not in text,
        'preserve actual attention math')
    phases=re.findall(r'^strata mtp: prompt expert phase=(\S+) ',text,re.M)
    require(phases==['before-detach','after-detach','after-prompt-before-restore','after-restore']
        and re.findall(r'^strata mtp: experts restored bytes=(\d+),',text,re.M)==['707788800'],
        'preserve actual MTP detach/restore handoff')
    if plan['trace_progress']:
        geometry=re.findall(r'^strata packed expert bank: layer=(\d+) slots=(\d+) stride=(\d+) used_bytes=(\d+) backing_bytes=(\d+) full_fit=([01])$',text,re.M)
        require(len(geometry)==48 and [int(g[0]) for g in geometry]==list(range(48))
            and all(int(g[1])==512 and g[5]=='1' and int(g[3])<=int(g[4])==512*STRIDE
                for g in geometry), 'actual48 full-fit512 layer banks required')
        result['actual_full_fit_layers']=48
    result['bank']=bank
    result['actual_GU_dispatch']=reference.dispatch_reader.dispatch(text,'large')
    return result


def check_sample(sample,count,read,minimum_ms=10000):
    comparison.check_sample(sample,count,read,minimum_ms)
    require(sample['generated_tokens']==24,'exact24thinking outputs required')


def decision(processes):
    require(len(processes)==len(CASES) and all(p['passed'] and p['complete']
        and p['native_exit_code']==0 and p['ownership_closed'] and not p['fresh_faults']
        and p['read']==CASES[i][0] and len(p['samples'])==RUNS
        for i,p in enumerate(processes)), 'all13ordered native processes required')
    require(all(r['actual_full_fit_layers']==48 for r in processes[0]['routes']), '512 admission absent')
    points=[]
    for read in (32769,65537):
        pairs=[processes[1+2*i:3+2*i] for i,n in enumerate(PAIR_READS) if n==read]
        ratios=[];cells={}
        for slots in (384,512):
            group=[p for pair in pairs for p in pair if CASES[p['index']][1]==slots]
            warm=[p['subsequent_median_ms'] for p in group]
            cells[str(slots)]=dict(warmed_process_ms=warm,
                warmed_process_PP=[read*1000/t for t in warm],
                warmed_PP=read*1000/statistics.median(warm),
                warmed_client_TTFT_s=[statistics.median(s['ttft_s'] for s in p['samples'][1:]) for p in group],
                first_PP=[p['samples'][0]['prompt_tps'] for p in group],
                first_client_TTFT_s=[p['samples'][0]['ttft_s'] for p in group])
        for pair in pairs:
            arms={CASES[p['index']][1]:p for p in pair}
            ratios.append(arms[512]['subsequent_median_ms']/arms[384]['subsequent_median_ms'])
            require(all(a['actual_GU_dispatch']==b['actual_GU_dispatch']
                for a,b in zip(arms[384]['routes'],arms[512]['routes'])),
                'matched actual arithmetic/job counts required')
        points.append(dict(read=read,arms=cells,time_ratios_512_over_384=ratios,
            geometric_mean_ratio=math.exp(statistics.mean(math.log(x) for x in ratios)),
            repeated_prefill_gain=all(x<1 for x in ratios)))
    return dict(points=points,slot_only_repeated_gain=all(p['repeated_prefill_gain'] for p in points),
        inference_adopted=False,sustained_decode_or_task_qualified=False,
        physical262144_RESTORE_qualified=False,ordinary_teardown_qualified=False,
        hardware_saturation_qualified=False,
        next='Keep all outcomes. A useful slot result still needs independent sustained decode preservation,agent/task,physical full context/RESTORE and ordinary teardown. Queue/window/ready changes require new separate plans.')


def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()),'fresh campaign only;no retry')
    native,proof,prior,files=basis()
    revision=core.source.device.common.source.git(P,'rev-parse','HEAD').decode().strip()
    ROOT.mkdir(mode=0o700)
    row=dict(stage='prepared',source_commit=revision,sources=core.pins(revision),
        policies={p:sha(P/p) for p in POLICIES},original_native_proof=proof,
        reference_evidence_sha256=signature(compact(prior)),
        **{k:copy.deepcopy(prior[k]) for k in ('configuration','environment_template','model_assets',
            'template','boot','driver','native_validation_probe','python_contract','python_selected')},
        binary=native['binaries']['strata'],native_source_commit=original.SOURCE,
        runtime_manifest=prior['runtime_manifest'],plans=[],processes=[],request_hashes={},
        host_passed=False,passed=False,complete=False,active=False,ownership_closed=True,
        budgets=dict(wall=1500,cpu=18000,rss_cap=104<<30,text_cap=16<<20,file_cap=32<<20,total_cap=128<<20),
        whole_bytes=1024<<20,
        plan=dict(question='Does384->512 prefill bank capacity improve the actual model with fixed same-queue/JIT math and decode placement?',
            unit='Only STRATA_SYCL_PREFILL_EXPERT_BANK_SLOTS changes. Existing512-ready producer/consumer/layout code is unchanged;queue,window10,ready policy,chunk8192,math,decode/MTP,logical262144/resident20480 and assets stay fixed. Bank rises1022361600->1363148800B(+325MiB).',
            prior='39-case full-math capacity-only component passes all3pairs at128/1024/8192rows;do not attribute its separate-queue gain to this same-queue model. Unique-marker change failed and is excluded.',
            order='One traced/stage-progress65537-input512 process first,then12clean processes:3adjacent alternating384/512pairs per32769/65537 length. Each has3thinking-on fresh exact24-output requests. Commit each closed process before next;first use separate from subsequent2.',
            acceptance='Current-source CPU frontend checks and original14508source/recipe/12ELFs/provider/model identity before launch.512 must actually allocate1.363GBsingle bank,all48packed layersfullfit,allresidualGPU,ordinaryroute,checkeddrains/MTPrestore. Same actual arithmetic/job counts/requestbodies,normal0,reaped/emptyowner,quietvalidation,nonewfault.',
            performance='Assess each length independently:all3matched warmed process-time ratios512/384<1permits further model gates for that length. No minimumgain or retrospectiveexclusion. First-usePP/clientTTFT and warmedTTFT/spreads separately.24outputs do not qualifydecode;no adoption before repeated sustaineddecodepreservation.',
            stopping='First source/build/provider/owner/budget/token/clock/route/validation/fault failure stops,no retry/hashrefresh/criteriachange. Prelaunchfailures stayfailed. No performanceearlystop. Reuseclosedreference15fullhashes with identitychecks;fullrehashall15afterfinalprocess.',
            limits='No modeladoption,physical262144fullfill/RESTORE,task/agent,sustainedTG,ordinaryproviderteardown or physicalbus/VRAM/arithmeticceiling claim. Two lengths establishadjacentslopeonly,notfittedlaw.',
            retention='Keep useful39request samples/13ownership outcomes/actualroutes/decision plus sourceconfigrecipeidentities. Retire successful rawoutput at closure. No newnativegeneration;retain existingprefill/decode references.'))
    for index,(read,slots,traced) in enumerate(CASES):
        root=ROOT/str(index);root.mkdir(mode=0o700)
        for name in ('tmp','cache'):(root/name).mkdir()
        core.write(root/'config.json',dict(row['configuration'],log=str(root/'engine.stderr')))
        core.write(root/'plan.json',dict(binary=row['binary'],native_validation_probe=row['native_validation_probe']))
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        plan=dict(index=index,read=read,slots=slots,heads=1,trace_progress=traced,port=port,root=str(root),
            generated={n:sha(root/n) for n in ('config.json','plan.json')})
        plan['command']=command(plan);plan['environment_sha256']=comparison.fingerprint(environment(row,plan))
        row['plans'].append(plan)
    row['invariant_sha256']=invariant(row,row['plans'][0]);save(row)
    print('Prepared slot-only model capacity comparison;commit before host/native.',flush=True)


def context():
    return SimpleNamespace(ROOT=ROOT,KEY=KEY,ORDER=ORDER,RUNS=RUNS,current=current,
        environment=environment,save=save,request_route=request_route,check_sample=check_sample,
        decision=decision)


def host():
    row=json.loads((ROOT/'record.json').read_text());current(row)
    require(row['stage']=='prepared' and not (ROOT/'unit').exists(),'fresh unit controls only')
    owner=comparison.build.supervision.Owner(ROOT/'unit')
    try:
        run,out,err=owner.run('unit',[PYTHON,'-B','-m','unittest','tools.test_prefill_bank_capacity'],
            environment(row,row['plans'][0]),P,wall=90,cpu=60,rss_cap=1<<30,
            text_cap=128<<10,file_cap=1<<20,total_cap=4<<20)
        require(comparison.build.supervision.closed(run) and run['exit_code']==0
            and 'Ran 4 tests in ' in err.read_text() and err.read_text().rstrip().endswith('OK'),
            'slot isolation/route/refusal controls failed')
        row['unit_controls']=dict(passed=True,exit_code=0,ownership_closed=True,tests=4)
    except BaseException as error:
        row.update(stage='failed',error=repr(error))
    finally:
        row.update(active=bool(owner.active),ownership_closed=owner.active is None
            and all(comparison.build.supervision.closed(v) for v in owner.commands));save(row)
    require(row['stage']=='prepared' and row['ownership_closed'],row.get('error','unit failure'))
    # Host admission reads committed state, including the just-finished controls.
    print('Four controller controls passed;commit before frontend host probes.',flush=True)


if __name__=='__main__':
    require(sys.executable==PYTHON,'use pinned model Python')
    require(len(sys.argv)==2 and (sys.argv[1] in ('prepare','unit','host')
        or sys.argv[1].isdigit() and 0<=int(sys.argv[1])<len(CASES)), 'prepare|unit|host|0..12')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);comparison.build.guards.no_model_owners([])
        if sys.argv[1]=='prepare':prepare()
        elif sys.argv[1]=='unit':host()
        else:
            row=json.loads((ROOT/'record.json').read_text())
            require(row.get('unit_controls')==dict(passed=True,exit_code=0,ownership_closed=True,tests=4),
                'closed committed unit controls required')
            if sys.argv[1]=='host':comparison.host(context())
            else:comparison.run(int(sys.argv[1]),context())
