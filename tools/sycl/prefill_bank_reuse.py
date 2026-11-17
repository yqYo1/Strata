#!/usr/bin/env python3
"""One-variable96K model reuse comparison; Main, repository LICENSE.

Owned coherent-native/bench controls only. No native rebuild, XeStrata input,
queue/math/window/decode change, or inference adoption.
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

sys.dont_write_bytecode=True
import coherent_model_basis as capacity
import compare_official_prefill as comparison
import gu_small_bench as dispatch_reader
import qsa16_stream_bench as core
import ur_validation as validation

P,B,RESULT=capacity.P,capacity.B,capacity.RESULT
ROOT,KEY=B/'prefill-bank-reuse-v4','prefill_bank_reuse_v4'
CAPACITY_KEY='coherent_context_model_v2'
READ,RUNS,STRIDE=98305,3,2662400
SLOTS=(512,384,512,512,384,384,512)
ORDER=[(READ,1)]*len(SLOTS)
PYTHON=capacity.reference.model.PYTHON
require,sha,signature=capacity.require,capacity.sha,capacity.signature
compact=core.compact
POLICIES=capacity.POLICIES


def save(row):
    core.write(ROOT/'record.json',row)
    values=json.loads(RESULT.read_text());values[KEY]=compact(row);core.write(RESULT,values)


def capacity_gate():
    value=validation.committed(RESULT,(CAPACITY_KEY,))
    full=value['outcomes']['full']
    require(value['stage']=='full_closed' and not value['active']
        and all(value[k] for k in ('passed','complete','ownership_closed'))
        and value['source_commit']=='be2f53f7d8ffb6f9580f334e0c264baf1343a679'
        and value['native_source_commit']==capacity.original.SOURCE
        and all(full[k] for k in ('passed','complete','ownership_closed','final_slot_consumed','boot_unchanged'))
        and full['native_exit_code']==0 and not full['fresh_faults']
        and full['actual_batched_rows']==262143 and full['generated_ids']==0
        and full['actual_final_commit_receipt']==capacity.MARKER
        and full['full_model_asset_digests_verified_after_case']==15,
        'closed original standalone physical-fill subgate required')
    fields={k:value[k] for k in ('source_commit','native_source_commit','binary','original_native_proof',
        'source_inputs','boot','driver')}
    fields['full']=full
    return dict(key=CAPACITY_KEY,sha256=signature(fields),original_controller=value['source_commit'],
        original_native=value['native_source_commit'],processed_positions=262144,
        standalone_only=True,serve_allocation_RESTORE_or_speed_inherited=False)


def environment(row,plan):
    env=comparison.environment(row,plan)
    env.update(STRATA_SYCL_PREFILL_EXPERT_BANK_SLOTS=str(plan['slots']),
        STRATA_SYCL_PREFILL_MOE_CENSUS='0',STRATA_PREFILL_LAYER_MAJOR_PROFILE='0',
        STRATA_SYCL_PREFILL_MOE_SPLIT_ACCUMULATOR='0')
    if plan['trace_progress']:env['STRATA_TRACE']='1'
    require(env['STRATA_SYCL_PREFILL_SAME_QUEUE']=='1'
        and env['STRATA_SYCL_PREFILL_EXPERT_UPLOAD_STAGING']=='0'
        and env['STRATA_SYCL_PREFILL_EXPERT_BANK_PACKED']=='1'
        and env['STRATA_SYCL_CACHE_SCRATCH_LEASE']==env['STRATA_SYCL_PREFILL_MIXED_SCRATCH']=='1'
        and env['STRATA_SYCL_PREFILL_MOE_SPLIT_ACCUMULATOR']=='0',
        'preserve production queue,bank and checked server scratch placement')
    return env


def command(plan):
    root=Path(plan['root'])
    return ['/bin/sh',str(P/'bench.sh'),'--config',str(root/'config.json'),'--port',str(plan['port']),
        '--prefill-exact-tokens',str(READ),'--think','--runs',str(RUNS),
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
        and all(sha(P/p)==h for p,h in row['policies'].items()), 'commit/freeze exact model comparison inputs')
    native,proof,prior=capacity.basis(row['original_native_proof'])
    require(row['capacity_gate']==capacity_gate()
        and row['reference_evidence_sha256']==signature(compact(prior))
        and row['binary']==native['binaries']['strata'] and row['native_source_commit']==capacity.original.SOURCE
        and all(row[k]==prior[k] for k in ('configuration','environment_template','model_assets','template',
            'boot','driver','native_validation_probe','python_contract','python_selected')),
        'original source/configuration/model or capacity evidence changed')
    contract,python=core.frontend.python_manifest()
    require(contract==row['python_contract'] and python['selected']==row['python_selected']
        and sha(row['template']['path'])==row['template']['sha256']
        and comparison.build.supervision.artifact_bytes(ROOT)<=row['whole_bytes']
        and len(row['plans'])==len(SLOTS), 'frontend/template/output budget changed')
    for i,plan in enumerate(row['plans']):
        root=Path(plan['root'])
        require(plan['index']==i and plan['read']==READ and plan['heads']==1
            and plan['slots']==SLOTS[i] and plan['trace_progress']==(i==0)
            and plan['command']==command(plan) and invariant(row,plan)==row['invariant_sha256']
            and json.loads((root/'config.json').read_text())==dict(row['configuration'],log=str(root/'engine.stderr'))
            and json.loads((root/'plan.json').read_text())==dict(binary=row['binary'],native_validation_probe=row['native_validation_probe'])
            and all(sha(root/p)==h for p,h in plan['generated'].items())
            and comparison.fingerprint(environment(row,plan))==plan['environment_sha256'], 'unplanned arm or generated input')
    files=json.loads(Path(row['runtime_manifest']['path']).read_text())
    return files,python


def request_route(text,plan):
    result=core.request_route(text,plan)
    bank=comparison.route(text,READ,1,expected_rows=READ-1,requested_slots=plan['slots'],
        draft_bytes=0,multi_window=True,split_requested=False)
    require(result['residual_gpu_rows']==READ-1 and bank['windows']==96
        and bank['maximum_window_rows']==81920 and not bank['split']
        and bank['staged'] and bank['prefill_slots']==plan['slots'] and bank['split_startup_disabled']
        and result['layer_cache_bytes']==plan['slots']*STRIDE,
        'actual98304-row all-GPU two-window capacity route differs')
    require('softmax=parallel-exp-ascending-sum' in text and 'softmax=subgroup' not in text,
        'preserve actual attention math')
    phases=re.findall(r'^strata mtp: prompt expert phase=(\S+) ',text,re.M)
    require(phases==['before-detach','after-detach','after-prompt-before-restore','after-restore']
        and re.findall(r'^strata mtp: experts restored bytes=(\d+),',text,re.M)==['707788800'],
        'actual MTP expert handoff differs')
    if plan['trace_progress']:
        geometry=re.findall(r'^strata packed expert bank: layer=(\d+) slots=(\d+) stride=(\d+) used_bytes=(\d+) backing_bytes=(\d+) full_fit=([01])$',text,re.M)
        require(len(geometry)==48 and [int(g[0]) for g in geometry]==list(range(48))
            and all(int(g[1])==512 and g[5]=='1' and int(g[3])<=int(g[4])==512*STRIDE for g in geometry),
            'actual48fullfit512banks required')
        result['actual_full_fit_layers']=48
    result.update(bank=bank,actual_GU_dispatch=dispatch_reader.dispatch(text,'large'))
    return result


def check_sample(sample,count,read,minimum_ms=10000):
    comparison.check_sample(sample,count,read,minimum_ms)
    require(read==READ and sample['generated_tokens']==24,'exact98305/24thinking sample required')


def decision(processes):
    require(len(processes)==7 and all(p['passed'] and p['complete'] and p['ownership_closed']
        and p['native_exit_code']==0 and not p['fresh_faults'] and p['index']==i
        and p['read']==READ and len(p['samples'])==RUNS for i,p in enumerate(processes)),
        'seven ordered processes/21declared requests required')
    require(all(r['actual_full_fit_layers']==48 for r in processes[0]['routes']), '512admission absent')
    ratios=[];bytes_reduced=[];pairs=[]
    for pair in (processes[1:3],processes[3:5],processes[5:7]):
        arms={SLOTS[p['index']]:p for p in pair};ref,can=arms[384],arms[512]
        ratio=can['subsequent_median_ms']/ref['subsequent_median_ms'];ratios.append(ratio)
        require(all(a['actual_GU_dispatch']==b['actual_GU_dispatch'] for a,b in zip(ref['routes'],can['routes'])),
            'actual arithmetic/job/map counts differ between matched arms')
        reduced=[b['bank']['requested_bank_H2D_bytes']<a['bank']['requested_bank_H2D_bytes']
            and b['bank']['permanent_hits']>a['bank']['permanent_hits']
            for a,b in zip(ref['routes'],can['routes'])];bytes_reduced.extend(reduced)
        pairs.append(dict(indices=[p['index'] for p in pair],time_ratio_512_over_384=ratio,
            ref_requested_H2D=[r['bank']['requested_bank_H2D_bytes'] for r in ref['routes']],
            candidate_requested_H2D=[r['bank']['requested_bank_H2D_bytes'] for r in can['routes']],
            every_request_reused_more_and_requested_fewer_bytes=all(reduced)))
    arms={}
    for slots in (384,512):
        group=[p for p in processes[1:] if SLOTS[p['index']]==slots]
        warm=[p['subsequent_median_ms'] for p in group]
        arms[str(slots)]=dict(warmed_process_ms=warm,warmed_process_PP=[READ*1000/t for t in warm],
            warmed_PP=READ*1000/statistics.median(warm),
            warmed_client_TTFT_s=[statistics.median(s['ttft_s'] for s in p['samples'][1:]) for p in group],
            first_ms=[p['samples'][0]['prompt_ms'] for p in group],first_PP=[p['samples'][0]['prompt_tps'] for p in group],
            first_client_TTFT_s=[p['samples'][0]['ttft_s'] for p in group])
    return dict(read=READ,arms=arms,pairs=pairs,time_ratios_512_over_384=ratios,
        geometric_mean_time_ratio=math.exp(statistics.mean(math.log(x) for x in ratios)),
        requested_copy_reuse_mechanism=all(bytes_reduced),repeated_prefill_gain=all(x<1 for x in ratios),
        eligible_for_further_model_gates=all(bytes_reduced) and all(x<1 for x in ratios),
        prefill1000_at96K_warmed_median=arms['512']['warmed_PP']>=1000,
        inference_adopted=False,sustained_decode_or_task_qualified=False,
        physical262144_RESTORE_qualified=False,ordinary_teardown_qualified=False,hardware_saturation_qualified=False,
        next='Keep original32Kgain/64Kfailure/128Kplacement failure intact. Judge96Kmechanism and time independently;no global adoption from this screen. Repeat sustained decode/agent/task and serve-state gates before adoption. Cross-cohort lengths do not fit a physical time law.')


def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()), 'fresh reuse campaign only')
    previous=validation.committed(RESULT,('prefill_bank_reuse_v3',))
    require(previous['stage']=='failed' and not previous['active'] and not previous['ownership_closed']
        and previous['source_commit']==validation.git('rev-parse','469132998').decode().strip()
        and len(previous['processes'])==1 and not previous['processes'][0]['passed']
        and previous['reader_refusal']['actual_all_GPU_rows']==98304
        and previous['reader_refusal']['actual_host_rows']==0
        and previous['reader_refusal']['actual_requested_split']==0
        and previous['reader_refusal']['actual_admission_marker_count']==0
        and previous['post_stop']['all_recorded_owner_identities_absent']
        and previous['post_stop']['owned_session_empty'] and previous['post_stop']['lock_reacquired']
        and previous['post_stop']['fresh_faults']==[], 'preserve original96Kdisabled-marker refusal')
    native,proof,prior=capacity.basis()
    require(previous['capacity_gate']==capacity_gate() and previous['original_native_proof']==proof,
        'original native and physical-capacity evidence changed')
    revision=validation.git('rev-parse','HEAD').decode().strip();ROOT.mkdir(mode=0o700)
    row=dict(stage='prepared',source_commit=revision,sources=core.pins(revision),
        policies={p:sha(P/p) for p in POLICIES},original_native_proof=proof,
        capacity_gate=capacity_gate(),reference_evidence_sha256=signature(compact(prior)),
        previous_attempt=dict(key='prefill_bank_reuse_v3',source_commit=previous['source_commit'],
            outcome='First96Krequest admits all98304GPUrows/staged512bank but the reader wrongly requires enabled-only split telemetry. Preserve failed reader outcome/supervised termination;no clean arms or normal-exit qualification.',
            corrected_input='Reader/fixture only:explicit split_requested=False requires verified immutable native startup0,no split telemetry and actual ordinary two-window work. Same98305input/ELF/math/settings/384vs512unit;legacy split-enabled marker remains mandatory. Original128Kplacement failure remains unchanged.'),
        **{k:copy.deepcopy(prior[k]) for k in ('configuration','environment_template','model_assets','template',
            'boot','driver','native_validation_probe','python_contract','python_selected')},
        binary=native['binaries']['strata'],native_source_commit=capacity.original.SOURCE,
        runtime_manifest=prior['runtime_manifest'],plans=[],processes=[],request_hashes={},
        host_passed=False,passed=False,complete=False,active=False,ownership_closed=True,
        budgets=dict(wall=2100,cpu=18000,rss_cap=104<<30,text_cap=16<<20,file_cap=32<<20,total_cap=128<<20),whole_bytes=1024<<20,
        plan=dict(question='Does512vs384bank capacity reuse weights across two MoEwindows at98305actual input tokens?',
            unit='Only requested bank slots384/512change. Fixed new setup98305actual input tokens and split0 avoids the independent90113..117488split route. Same14508ELF,queue,window10/chunk8192,all98304residualGPUrows,BF16/ordered-exp/GUlarge/Downlarge,serverlease1/mixedscratch1,decode1256/resident20480,MTPspec4 and assets. Extra325MiBcapacity must actually admit;no fallback to a different placement. This is no adoption or performance claim for the setup change;any later combination needs its own phase-preservation evidence.',
            order='One traced512process then6clean processes384,512,512,384,384,512:three adjacent alternating pairs. Three exact98305-input/24-output thinking-on bench.sh requests per process,plus official short warmup. Commit every closed process;first use separate from subsequent2.',
            acceptance='Current source/helper/workload/native/runtime/owner guards. Native environment must actually have immutable split0 before each request gate;no split admission/complete telemetry may appear. Legacy enabled-split reader still requires admission. Traced512must admit48fullfit banks;every request must use96accumulator windows,max81920,allresidualGPU,requested slots,actualsplit0,checkeddrains/MTPrestore,normal0,reaped/emptyowner,quietvalidation,no fresh fault. Refuse/stop on any placement drift;never silently lower input/changeconfiguration.',
            performance='Same request hashes and actual arithmetic/job/map counts in every pair. Mechanism gate:each of9clean paired requests has fewer requested bankH2Dbytes and more permanent hits with512. Time gate:all3paired warmed process-time ratios512/384<1,no minimumgain. Recordfirst/warmed PP/clientTTFT/spread separately. Onlyboth gates permit further model qualification;1000PPpoint is separate,not adoption.',
            stopping='Stop first source/build/provider/owner/token/clock/route/validation/fault/budget failure;no retry,hashrefresh,exclusion,criteriachange or performanceearlystop.15model identity checks before/after each child;all15full hashes after finalprocess.',
            limits='No sustained TG/task/agent/serveRESTORE/ordinaryteardown or saturation claim. Requested copy bytes are not measured bus traffic. Existing32K/64K/128Koutcomes and historical best remain distinct;new96Kcohort does not alone fit a scaling law.',
            retention='Keep21individual samples,per-process first/warmed metrics and essential route/reuse counters plus source/config/decision. Retire successful rawlogs/JIT atclosure. No newnativegeneration. Closed physical-capacity CLI/test code and original inputs retire;keep only its shared native/model basis.'))
    for i,slots in enumerate(SLOTS):
        root=ROOT/str(i);root.mkdir();(root/'tmp').mkdir();(root/'cache').mkdir()
        core.write(root/'config.json',dict(row['configuration'],log=str(root/'engine.stderr')))
        core.write(root/'plan.json',dict(binary=row['binary'],native_validation_probe=row['native_validation_probe']))
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        plan=dict(index=i,read=READ,slots=slots,heads=1,trace_progress=i==0,root=str(root),port=port,
            generated={name:sha(root/name) for name in ('config.json','plan.json')})
        plan['command']=command(plan);plan['environment_sha256']=comparison.fingerprint(environment(row,plan));row['plans'].append(plan)
    row['invariant_sha256']=invariant(row,row['plans'][0]);save(row)
    print('Prepared96Kcapacity-only reuse screen;commit before host/native.',flush=True)


def context():
    return SimpleNamespace(ROOT=ROOT,KEY=KEY,ORDER=ORDER,RUNS=RUNS,current=current,environment=environment,
        save=save,request_route=request_route,check_sample=check_sample,decision=decision)


def unit():
    row=json.loads((ROOT/'record.json').read_text());current(row)
    require(row['stage']=='prepared' and not (ROOT/'unit').exists(), 'fresh host controls only')
    owner=comparison.build.supervision.Owner(ROOT/'unit')
    try:
        run,out,err=owner.run('unit',[PYTHON,'-B','-m','unittest','tools.test_prefill_bank_reuse'],
            environment(row,row['plans'][0]),P,wall=90,cpu=60,rss_cap=1<<30,text_cap=128<<10,file_cap=1<<20,total_cap=4<<20)
        require(comparison.build.supervision.closed(run) and run['exit_code']==0
            and 'Ran 9 tests in ' in err.read_text() and err.read_text().rstrip().endswith('OK'),
            'reuse isolation/route/refusal controls failed')
        row['unit_controls']=dict(passed=True,exit_code=0,ownership_closed=True,tests=9)
    except BaseException as error:row.update(stage='failed',error=repr(error))
    finally:
        row.update(active=bool(owner.active),ownership_closed=owner.active is None
            and all(comparison.build.supervision.closed(v) for v in owner.commands));save(row)
    require(row['stage']=='prepared' and row['ownership_closed'],row.get('error','unit failure'))
    print('Controller controls passed;commit before frontend host probes.',flush=True)


if __name__=='__main__':
    require(sys.executable==PYTHON,'use pinned model Python')
    require(len(sys.argv)==2 and (sys.argv[1] in ('prepare','unit','host') or
        sys.argv[1].isdigit() and 0<=int(sys.argv[1])<7), 'prepare|unit|host|0..6')
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);comparison.build.guards.no_model_owners([])
        if sys.argv[1]=='prepare':prepare()
        elif sys.argv[1]=='unit':unit()
        else:
            row=json.loads((ROOT/'record.json').read_text());require(row.get('unit_controls',{}).get('passed'), 'closed committed controls required')
            if sys.argv[1]=='host':comparison.host(context())
            else:comparison.run(int(sys.argv[1]),context())
