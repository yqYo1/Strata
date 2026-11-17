#!/usr/bin/env python3
"""Owned one-variable bank/residual placement screen; repository LICENSE.

Native14508 remains unchanged. This prospective controller tests requested
384->256 slots at fixed128K, including the planner's placement consequences.
No XeStrata source, default selection, or inference adoption.
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
import coherent_model_basis as capacity
import compare_official_prefill as comparison
import gu_small_bench as dispatch_reader
import qsa16_stream_bench as core
import ur_validation as validation

P, B, RESULT = capacity.P, capacity.B, capacity.RESULT
ROOT, KEY = B/'prefill-bank-placement-v1', 'prefill_bank_placement_v1'
CAPACITY_KEY = 'coherent_context_model_v2'
READ, RUNS, STRIDE, ROW = 131072, 3, 2662400, 20480
READS = (32769,) + (READ,)*8
SLOTS = (256,384,256,384,256,256,384,384,256)
ORDER = [(read,1) for read in READS]
PYTHON = capacity.reference.model.PYTHON
require, sha, signature = capacity.require, capacity.sha, capacity.signature
compact, POLICIES = core.compact, capacity.POLICIES


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
        STRATA_SYCL_PREFILL_MOE_CENSUS='0',STRATA_PREFILL_LAYER_MAJOR_PROFILE='0')
    if plan['trace_progress']: env['STRATA_TRACE']='1'
    require(env['STRATA_SYCL_PREFILL_SAME_QUEUE']=='1'
        and env['STRATA_SYCL_PREFILL_EXPERT_UPLOAD_STAGING']=='0'
        and env['STRATA_SYCL_PREFILL_EXPERT_BANK_PACKED']=='1'
        and env['STRATA_SYCL_CACHE_SCRATCH_LEASE']==env['STRATA_SYCL_PREFILL_MIXED_SCRATCH']=='1'
        and env['STRATA_SYCL_PREFILL_MOE_SPLIT_ACCUMULATOR']=='1',
        'preserve immutable queue/scratch/default split request;split inactive at these extents')
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
        and all(sha(P/p)==h for p,h in row['policies'].items()),'commit/freeze exact placement comparison')
    native,proof,prior=capacity.basis(row['original_native_proof'])
    require(row['capacity_gate']==capacity_gate()
        and row['reference_evidence_sha256']==signature(compact(prior))
        and row['binary']==native['binaries']['strata'] and row['native_source_commit']==capacity.original.SOURCE
        and all(row[k]==prior[k] for k in ('configuration','environment_template','model_assets','template',
            'boot','driver','native_validation_probe','python_contract','python_selected')),
        'original native,model,decode,safety or physical subgate changed')
    contract,python=core.frontend.python_manifest()
    require(contract==row['python_contract'] and python['selected']==row['python_selected']
        and sha(row['template']['path'])==row['template']['sha256']
        and comparison.build.supervision.artifact_bytes(ROOT)<=row['whole_bytes']
        and len(row['plans'])==len(ORDER),'frontend/template/output budget changed')
    for i,plan in enumerate(row['plans']):
        root=Path(plan['root'])
        require(plan['index']==i and plan['read']==READS[i] and plan['heads']==1
            and plan['slots']==SLOTS[i] and plan['trace_progress']==(i<3)
            and plan['command']==command(plan) and invariant(row,plan)==row['invariant_sha256']
            and json.loads((root/'config.json').read_text())==dict(row['configuration'],log=str(root/'engine.stderr'))
            and json.loads((root/'plan.json').read_text())==dict(binary=row['binary'],native_validation_probe=row['native_validation_probe'])
            and all(sha(root/p)==h for p,h in plan['generated'].items())
            and comparison.fingerprint(environment(row,plan))==plan['environment_sha256'],'unplanned input/arm')
    return json.loads(Path(row['runtime_manifest']['path']).read_text()),python


def records(text,prefix):
    found=[]
    for line in text.splitlines():
        if line.startswith(prefix):
            pairs=re.findall(r'([A-Za-z_][A-Za-z_0-9-]*)=([^\s;]+)',line.split(';',1)[0])
            require(len({k for k,v in pairs})==len(pairs),'duplicate '+prefix)
            found.append(dict(pairs))
    return found


def one(text,prefix):
    values=records(text,prefix)
    require(len(values)==1,'missing/duplicate '+prefix)
    return values[0]


def placement_route(text,plan):
    n=plan['read']-1;requested=plan['slots'];staged=requested==256
    require((plan['read'],requested) in ((32769,256),(READ,256),(READ,384)), 'undeclared placement case')
    main=one(text,'strata prefill: layer-major candidate, ')
    bank=one(text,'strata MoE expert bank: ')
    admission=one(text,'strata MoE accumulator admission: ')
    split=one(text,'strata MoE split admission: ')
    window=one(text,'strata MoE window configuration: ')
    work=one(text,'strata MoE accumulator window: ')
    source=one(text,'strata prefill expert source: ')
    mixed=one(text,'strata MoE mixed banks: ')
    gpu=int(main['residual_gpu_rows']);slots=requested if staged else 512
    require(gpu==(n if staged else 98304) and main['tokens']==str(n)
        and main['chunk']=='8192' and main['residual_dtype']=='bf16' and main['residual_element_bytes']=='2'
        and int(main['residual_total'])==n*ROW and int(main['residual_gpu'])==gpu*ROW
        and int(main['residual_host'])==(n-gpu)*ROW and main['residual_staging']=='0',
        'actual residual placement/byte conservation differs;no automatic recalibration')
    cache=slots*STRIDE;limit=int(bank['allocation_limit_bytes'])
    require(bank['requested_slots']==str(requested) and int(bank['effective_slots'])==slots
        and bank['staged']==str(int(staged)) and int(bank['host_slab_bytes'])==512*STRIDE
        and int(bank['device_bytes'])==int(main['layer_cache'])==cache
        and int(bank['residual_gpu_rows'])==gpu and bank['general_map']=='private_scratch_never_published'
        and max(cache,gpu*ROW)<=limit,'bank/allocation/private-map contract differs')
    require(split['active']==split['primary_rows']==split['tail_payload_bytes']==split['extra_allocation_bytes']=='0'
        and split['rows']==str(n) and split['prefill_slots']==str(slots) and split['decode_changed']=='0'
        and int(split['max_single_allocation_bytes'])==limit and not records(text,'strata MoE split complete: '),
        'unchanged startup split request must actually remain inactive')
    maximum=min(81920,gpu);windows=48*((gpu+81919)//81920)
    require(window['requested_mode']==window['effective_mode']=='10' and window['mixed_fp16_bytes']=='0'
        and int(window['maximum_rows'])==int(work['maximum_rows'])==maximum
        and int(work['windows'])==windows and work['operator_max']=='8192'
        and work['GPU_resident_windows_only']=='1' and int(work['jobs'])>0 and int(work['dequants'])>0,
        'actual window extent/work differs')
    require(mixed['storage']=='scratch' and mixed['extra_allocation_bytes']=='0'
        and int(mixed['primary_bytes'])==min(65536,maximum)*5120
        and int(mixed['tail_bytes'])==max(0,maximum-65536)*5120
        and int(mixed['tail_offset'])>=0 and mixed['phase']=='FFN_only_after_attention_drain',
        'checked borrowed mixed storage differs')
    # Recompute native14508's automatic bank/R admission from its observed
    # free budget. Prefix cells round to16; two BF16 tensors plus INT32 flags.
    prefix=((n+15)//16)*16*2056;other=prefix+256;headroom=536870912
    free=int(admission['free_gpu_bytes']);available=free-other-headroom
    requested_bank=requested*STRIDE
    predicted_staged=available>=requested_bank+n*ROW and max(requested_bank,n*ROW)<=limit
    budget=min(available-cache,limit)
    predicted_gpu=n if n*ROW<=budget else min(n,budget//ROW)//8192*8192
    require(predicted_staged==staged and predicted_gpu==gpu
        and admission['requested_mode']=='10' and int(admission['rows'])==maximum
        and int(admission['gpu_prefix_rows'])==gpu and int(admission['shared_arena_bytes'])>0
        and int(admission['mixed_fp16_bytes'])==maximum*5120
        and int(admission['fixed_including_headroom_bytes'])==cache+other+headroom+gpu*ROW<=free
        and prefix<=limit,'observed native budget does not reproduce planned placement')
    require(source['requested_view']=='1' and source['effective_view']==str(int(staged))
        and int(source['layers'])==(48 if staged else 0)
        and int(source['owned_slab_bytes'])==(0 if staged else 512*STRIDE)
        and (int(source['borrowed_layer_span_bytes'])>0 if staged else source['borrowed_layer_span_bytes']=='0')
        and source['host_copy']==('eliminated' if staged else 'synchronous')
        and source['source_owner']=='caller_through_checked_retirement','immutable expert source placement differs')
    ends=records(text,'strata MoE expert bank complete: ')
    require(len(ends)==int(staged),'staged bank completion differs')
    copied=hits=uploads=None
    if staged:
        end=ends[0];copied=int(end['requested_H2D_bytes']);hits=int(end['permanent_hits']);uploads=int(end['uploads'])
        require(end['final_drain']=='checked' and int(end['total_slots'])>=slots
            and 0<=int(end['permanent_slots'])<=int(end['total_slots'])<=512
            and copied>=0 and hits>=0 and uploads>=0,'staged upload/drain conservation differs')
    require(text.index('strata prefill heads: target detached ')<text.index('strata prefill heads: draft detached ')
        <text.index('strata MoE expert bank: ')<text.index('strata prefill: layer-major candidate, '),
        'head retirement/admission ordering differs')
    if plan['trace_progress']:
        geometry=records(text,'strata packed expert bank: ')
        require(len(geometry)==(48 if staged else 0),'packed view count differs')
        for layer,g in enumerate(geometry):
            stride=int(g['stride']);count=min(512,cache//stride)
            require(int(g['layer'])==layer and stride>0 and stride%256==0
                and int(g['slots'])==count and int(g['used_bytes'])==count*stride<=cache
                and int(g['backing_bytes'])==cache and g['full_fit']==str(int(count==512)),
                'actual per-layer packed geometry differs')
    return dict(rows=n,residual_gpu_rows=gpu,residual_host_rows=n-gpu,prefill_slots=slots,staged=staged,
        layer_cache_bytes=cache,windows=windows,maximum_window_rows=maximum,
        requested_bank_H2D_bytes=copied,permanent_hits=hits,uploads=uploads,
        immutable_all_expert_layer_bytes=int(source['borrowed_layer_span_bytes']) if staged else None,
        requested_source_map_H2D_bytes=int(work['source_H2D_bytes']),free_gpu_bytes=free,
        fixed_including_headroom_bytes=int(admission['fixed_including_headroom_bytes']),
        allocation_limit_bytes=limit,budget_reproduced=True,actual_split=False,
        actual_expert_source_view=staged,actual_packed_layers=48 if staged and plan['trace_progress'] else None,
        physical_bus_or_saturation_qualified=False)


def request_route(text,plan):
    result=core.request_route(text,plan);bank=placement_route(text,plan)
    require('softmax=parallel-exp-ascending-sum' in text and 'softmax=subgroup' not in text,
        'attention arithmetic changed')
    phases=re.findall(r'^strata mtp: prompt expert phase=(\S+) ',text,re.M)
    require(phases==['before-detach','after-detach','after-prompt-before-restore','after-restore']
        and re.findall(r'^strata mtp: experts restored bytes=(\d+),',text,re.M)==['707788800'],
        'actual decode MTP handoff differs')
    result.update(bank=bank,actual_GU_dispatch=dispatch_reader.dispatch(text,'large'))
    return result


def check_sample(sample,count,read,minimum_ms=10000):
    comparison.check_sample(sample,count,read,minimum_ms)
    require(read in READS and sample['generated_tokens']==24,'declared exact thinking-on prefill screen only')


def decision(processes):
    require(len(processes)==9 and all(p['passed'] and p['complete'] and p['ownership_closed']
        and p['native_exit_code']==0 and not p['fresh_faults'] and p['index']==i
        and p['read']==READS[i] and len(p['samples'])==RUNS for i,p in enumerate(processes)),
        'nine ordered processes/27requests required')
    ratios=[];mechanisms=[];pairs=[]
    for pair in (processes[3:5],processes[5:7],processes[7:9]):
        arms={SLOTS[p['index']]:p for p in pair};ref,can=arms[384],arms[256]
        ratio=can['subsequent_median_ms']/ref['subsequent_median_ms'];ratios.append(ratio)
        requested_reference=[];candidate=[]
        for a,b in zip(ref['routes'],can['routes']):
            # Native nonstaged fill_slot copies blob_bytes(layer),NOT padded
            # max stride. Immutable source span sums those exact48*512 blobs.
            r,c=a['bank'],b['bank'];whole=c['immutable_all_expert_layer_bytes']
            requested_reference.append(whole);candidate.append(c['requested_bank_H2D_bytes'])
            mechanisms.append(r['residual_host_rows']==32767 and c['residual_host_rows']==0
                and c['layer_cache_bytes']==r['layer_cache_bytes']//2
                and c['requested_bank_H2D_bytes']<whole)
        pairs.append(dict(indices=[p['index'] for p in pair],time_ratio_256_over_384=ratio,
            source_derived_reference_requested_bank_H2D=requested_reference,
            candidate_logged_requested_bank_H2D=candidate))
    arms={}
    for slots in (384,256):
        group=[p for p in processes[3:] if SLOTS[p['index']]==slots]
        warm=[p['subsequent_median_ms'] for p in group]
        arms[str(slots)]=dict(warmed_process_ms=warm,warmed_process_PP=[READ*1000/t for t in warm],
            warmed_PP=READ*1000/statistics.median(warm),
            warmed_client_TTFT_s=[statistics.median(s['ttft_s'] for s in p['samples'][1:]) for p in group],
            first_ms=[p['samples'][0]['prompt_ms'] for p in group],first_PP=[p['samples'][0]['prompt_tps'] for p in group],
            first_client_TTFT_s=[p['samples'][0]['ttft_s'] for p in group])
    gain=all(x<1 for x in ratios);mechanism=all(mechanisms)
    return dict(read=READ,arms=arms,pairs=pairs,time_ratios_256_over_384=ratios,
        geometric_mean_time_ratio=math.exp(statistics.mean(math.log(x) for x in ratios)),
        actual_residency_and_requested_transfer_mechanism=mechanism,repeated_prefill_gain=gain,
        eligible_for_further_model_gates=gain and mechanism,
        prefill1000_at128K_warmed_median=arms['256']['warmed_PP']>=1000,
        inference_adopted=False,sustained_decode_or_task_qualified=False,
        physical262144_RESTORE_qualified=False,ordinary_teardown_qualified=False,hardware_saturation_qualified=False,
        next='Judge only requested-capacity unit;preserve first/warm and all failures. Scheduling/source-view/workload changes are automatic planner outcomes,not separately adopted kernels. New task/agent/numerical,decode-preservation and serve/RESTORE gates remain before adoption. No cross-cohort timing-law fit or saturation claim.')


def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()),'fresh placement campaign only')
    previous=validation.committed(RESULT,('prefill_bank_reuse_v4',))
    require(previous['stage']=='closed' and previous['passed'] and previous['ownership_closed']
        and not previous['decision']['eligible_for_further_model_gates'], 'preserve failed96K warmed-gain gate')
    native,proof,prior=capacity.basis()
    revision=validation.git('rev-parse','HEAD').decode().strip();ROOT.mkdir(mode=0o700)
    row=dict(stage='prepared',source_commit=revision,sources=core.pins(revision),
        policies={p:sha(P/p) for p in POLICIES},original_native_proof=proof,capacity_gate=capacity_gate(),
        reference_evidence_sha256=signature(compact(prior)),
        **{k:copy.deepcopy(prior[k]) for k in ('configuration','environment_template','model_assets','template',
            'boot','driver','native_validation_probe','python_contract','python_selected')},
        binary=native['binaries']['strata'],native_source_commit=capacity.original.SOURCE,
        runtime_manifest=prior['runtime_manifest'],plans=[],processes=[],request_hashes={},
        host_passed=False,passed=False,complete=False,active=False,ownership_closed=True,
        budgets=dict(wall=2400,cpu=20000,rss_cap=104<<30,text_cap=16<<20,file_cap=32<<20,total_cap=128<<20),whole_bytes=1536<<20,
        plan=dict(question='Does requested256 instead of384slots admit all131071residual rows and improve128Kprefill?',
            unit='Only requested bank slots384->256change at131072actual tokens. Same14508nativeELF/math/chunk8192/window10/samequeue/packed/sourceview/split-request1/lease1/mixedscratch1/decode1256/KVresident20480/MTPspec4/assets/safety. Automatic planner residency,staging,source preparation and scheduling changes are the intended indivisible outcome of that one parameter;no additional kernel or setting change. Native split is inactive both32768and131071rows.',
            order='Three traced3request processes:32769/256small check,131072/384reference,131072/256candidate. Then six clean131072 processes384,256,256,384,384,256,each3requests;three alternating adjacent pairs. Thinking enabled,24outputs diagnostic only. Official bench.sh and identical native_gdn_20261018nonce inputs. First use separate from subsequent2.',
            acceptance='Commit host/refusal controls and frontend probes before native. Fresh source/lifetime audit before128Kcandidate/clean. Actual reference must remain98304GPU/32767host,fallback512/staged0;candidate all131071GPU/staged256. Refuse any other placement rather than recalibrate. Validate native budget/free/max-single bound,source owner,per-layer packed geometry,tracedsmall and both128Karms,split inactive,window10,checked drains/MTPhandoff,normal0/reapedempty,quiet validation/no fresh fault. No simultaneous researcher/reviewer during clean samples.',
            performance='Same fullrequest hashes/native/config except slots. Per-arm GU/Down/job/map conservation required;automatic window/placement differences need not have equal job counts. Mechanism each9clean pair:hostRrows32767->0,bankbyteshalved,candidate logged requested bankH2Dless than reference native source-derived sum48*512*blob_bytes,obtained from same immutable all-expert source span. Never substitute padded max_stride or claim physical bus. Time all3warmed process ratios256/384<1,no minimumgain. First/warm PP,TTFTandindividualsamples/spread retained. Both gates only permit furtherqualification;target separate.',
            stopping='Stop first source/build/provider/owner/token/clock/route/validation/fault/budget failure;no retry/hashrefresh/exclusion/criteriachange/performanceearlystop. No default/adoption change. All15assetidentitiesbefore/after;full15hashesafterfinal.',
            limits='Parameter screen not numerical/task/agent/sustainedTG/decode-preservation/serveRESTORE/ordinaryteardown or saturation qualification. Original physical-full standalone subgate reused narrowly only. Prior32/64/96Kcohorts/best and first-use512observation distinct.',
            retention='Keep27individualsamples,source/config,native/budget/route/requested-byteandpaireddecisions;retire raw/JIT and closed controller atclosure. No new native generation.'))
    for i,(read,slots) in enumerate(zip(READS,SLOTS)):
        root=ROOT/str(i);root.mkdir();(root/'tmp').mkdir();(root/'cache').mkdir()
        core.write(root/'config.json',dict(row['configuration'],log=str(root/'engine.stderr')))
        core.write(root/'plan.json',dict(binary=row['binary'],native_validation_probe=row['native_validation_probe']))
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        plan=dict(index=i,read=read,slots=slots,heads=1,trace_progress=i<3,root=str(root),port=port,
            generated={name:sha(root/name) for name in ('config.json','plan.json')})
        plan['command']=command(plan);plan['environment_sha256']=comparison.fingerprint(environment(row,plan));row['plans'].append(plan)
    row['invariant_sha256']=invariant(row,row['plans'][0]);save(row)
    print('Prepared bank-only128K placement plan;commit before host/native.',flush=True)


def context():
    return SimpleNamespace(ROOT=ROOT,KEY=KEY,ORDER=ORDER,RUNS=RUNS,current=current,environment=environment,
        save=save,request_route=request_route,check_sample=check_sample,decision=decision)


def unit():
    row=json.loads((ROOT/'record.json').read_text());current(row)
    require(row['stage']=='prepared' and not (ROOT/'unit').exists(),'fresh host controls only')
    owner=comparison.build.supervision.Owner(ROOT/'unit')
    try:
        run,out,err=owner.run('unit',[PYTHON,'-B','-m','unittest','tools.test_prefill_bank_placement'],
            environment(row,row['plans'][0]),P,wall=90,cpu=60,rss_cap=1<<30,text_cap=128<<10,file_cap=1<<20,total_cap=4<<20)
        require(comparison.build.supervision.closed(run) and run['exit_code']==0
            and 'Ran 8 tests in ' in err.read_text() and err.read_text().rstrip().endswith('OK'), 'placement host controls failed')
        row['unit_controls']=dict(passed=True,exit_code=0,ownership_closed=True,tests=8)
    except BaseException as error:row.update(stage='failed',error=repr(error))
    finally:
        row.update(active=bool(owner.active),ownership_closed=owner.active is None
            and all(comparison.build.supervision.closed(v) for v in owner.commands));save(row)
    require(row['stage']=='prepared' and row['ownership_closed'],row.get('error','unit failure'))
    print('Placement host controls passed;commit before frontend probes.',flush=True)


if __name__=='__main__':
    require(sys.executable==PYTHON,'use pinned model Python')
    require(len(sys.argv)==2 and (sys.argv[1] in ('prepare','unit','host') or
        sys.argv[1].isdigit() and 0<=int(sys.argv[1])<len(ORDER)), 'prepare|unit|host|0..8')
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);comparison.build.guards.no_model_owners([])
        if sys.argv[1]=='prepare':prepare()
        elif sys.argv[1]=='unit':unit()
        else:
            row=json.loads((ROOT/'record.json').read_text());require(row.get('unit_controls',{}).get('passed'),'closed controls required')
            if sys.argv[1]=='host':comparison.host(context())
            else:comparison.run(int(sys.argv[1]),context())
