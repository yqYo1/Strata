#!/usr/bin/env python3
"""Independent long-prefill coherence/layout/window model comparisons.

Main-authored from owned ce488370d bank and compare_official_prefill controls;
repository LICENSE. No XeStrata source or inferred adoption. Each question gets
its own prospective plan,small/traced admission and three clean matched pairs.
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
import prefill_window_device as device
import qsa16_stream_bench as core

P,B,RESULT=capacity.P,capacity.B,capacity.RESULT
require,sha,signature=capacity.require,capacity.sha,capacity.signature
compact=core.compact
UNITS=('coherence','relocation','window','combined')
RUNS,STRIDE,ROW=3,2662400,20480
PYTHON=capacity.reference.model.PYTHON
ARMS=(1,0,1,0,1,1,0,0,1)
UNIT=sys.argv[1] if len(sys.argv)>1 and sys.argv[1] in UNITS else 'coherence'
ROOT=B/('prefill-window-'+UNIT+'-model-v1')
KEY='prefill_window_'+UNIT+'_model_v1'
LONG_READ=98305 if UNIT=='relocation' else 131072
READS=(32769,)+(LONG_READ,)*8
ORDER=[(read,1) for read in READS]


def arm(unit,index):
    selected=ARMS[index]
    return dict(selected=selected,native='original' if unit=='coherence' and not selected else 'candidate',
        bank=128 if unit=='relocation' else 1 if unit=='window' or unit=='combined' and selected else 256,
        window=16 if unit in ('window','combined') and selected else 10,
        relocate=int(unit=='relocation' and selected))


def save(row):
    core.write(ROOT/'record.json',row)
    values=json.loads(RESULT.read_text());values[KEY]=compact(row);core.write(RESULT,values)


def device_gate():
    value=json.loads((device.ROOT/'record.json').read_text());native=device.verify(value)
    require(value['stage']=='closed' and not value['active'] and value['passed'] and value['complete']
        and value['ownership_closed'] and [v['mode'] for v in value['results']]==device.STAGES
        and all(v['passed'] and v['complete'] and v['ownership_closed'] and not v['fresh_faults']
            and v['boot_unchanged'] for v in value['results'])
        and not value['results'][0]['GPU_executed']
        and all(v['GPU_executed'] and v['native_exit_code']==0 for v in value['results'][1:]),
        'all seven closed same-build component stages required')
    return native,dict(key=device.KEY,source_commit=value['source_commit'],
        record_sha256=sha(device.ROOT/'record.json'),binaries=value['binaries'],
        scope='Component address/math/closure only;actual allocation,model/task/phase/physical-context gates not inherited.')


def predecessor():
    if UNIT=='coherence':return device.build.bank_gate()
    previous=UNITS[UNITS.index(UNIT)-1]
    key='prefill_window_'+previous+'_model_v1'
    value=device.build.ur_validation.committed(RESULT,(key,))
    require(value['stage']=='closed' and value['passed'] and value['complete'] and not value['active']
        and value['ownership_closed'] and len(value['processes'])==9
        and value['decision']['eligible_for_next_independent_question'] is True,
        'close the preceding independently scoped question first;do not bypass its prospective decision')
    return dict(key=key,source_commit=value['source_commit'],source_inputs=value['source_inputs'],
        decision=value['decision'],scope='Original unit outcome remains unchanged;not aggregate adoption.')


def environment(row,plan):
    env=comparison.environment(row,plan)
    env.update(STRATA_SYCL_PREFILL_EXPERT_BANK_SLOTS=str(plan['bank']),
        STRATA_SYCL_PREFILL_MOE_WINDOW=str(plan['window']),
        STRATA_SYCL_PREFILL_MOE_RELOCATE_JOB_BUFFERS=str(plan['relocate']),
        STRATA_SYCL_PREFILL_FULLFIT_UPLOAD='jit',STRATA_SYCL_PREFILL_MOE_CENSUS='0',
        STRATA_PREFILL_LAYER_MAJOR_PROFILE='0')
    if plan['trace_progress']:env['STRATA_TRACE']='1'
    require(env['STRATA_SYCL_PREFILL_SAME_QUEUE']=='1'
        and env['STRATA_SYCL_PREFILL_EXPERT_UPLOAD_STAGING']=='0'
        and env['STRATA_SYCL_PREFILL_EXPERT_BANK_PACKED']=='1'
        and env['STRATA_SYCL_CACHE_SCRATCH_LEASE']==env['STRATA_SYCL_PREFILL_MIXED_SCRATCH']=='1'
        and env['STRATA_SYCL_PREFILL_MOE_SPLIT_ACCUMULATOR']=='1'
        and env['STRATA_SYCL_PREFILL_DOWN_SPLIT_KERNEL']=='original',
        'preserve original queue/scratch/staging/split kernel')
    return env


def native_context(row,plan):
    return dict(row['native_contexts'][plan['native']],native_validation_probe=row['native_validation_probe'],boot=row['boot'])


def configuration(row,plan):
    # Keep the qualified launcher;its private plan selects the actual ELF and
    # installs the same pre-main validation/ownership environment in both arms.
    return dict(row['configuration'],log=str(Path(plan['root'])/'engine.stderr'))


def command(plan):
    root=Path(plan['root'])
    return ['/bin/sh',str(P/'bench.sh'),'--config',str(root/'config.json'),'--port',str(plan['port']),
        '--prefill-exact-tokens',str(plan['read']),'--think','--runs',str(RUNS),
        '--profile-request-gates','--workload-id','native_gdn_20261018','--fail-fast-engine-error',
        '--ready-timeout','600','--out',str(root/'bench')]


def current(row):
    require(device.build.ur_validation.committed(RESULT,(KEY,))==compact(row)
        and core.pins(row['source_commit'])==row['sources']
        and all(sha(P/p)==h for p,h in row['policies'].items()),'commit/freeze the exact independent question')
    original,proof,prior=capacity.basis(row['original_native_proof']);candidate,gate=device_gate()
    require(row['unit']==UNIT and row['component_gate']==gate and row['predecessor']==predecessor()
        and row['reference_evidence_sha256']==signature(compact(prior))
        and row['native_contexts']==dict(
            original=dict(binary=original['binaries']['strata'],native_source_commit=capacity.original.SOURCE),
            candidate=dict(binary=candidate['binaries']['strata'],native_source_commit=candidate['source_commit']))
        and row['candidate_composition_sha256']==signature(candidate['source_files'])
        and all(row[k]==prior[k] for k in ('configuration','environment_template','model_assets','template',
            'boot','driver','native_validation_probe','python_contract','python_selected')),
        'original/candidate generation,model,decode placement or runtime changed')
    contract,python=core.frontend.python_manifest()
    require(contract==row['python_contract'] and python['selected']==row['python_selected']
        and sha(row['template']['path'])==row['template']['sha256']
        and comparison.build.supervision.artifact_bytes(ROOT)<=row['whole_bytes']
        and len(row['plans'])==len(ORDER),'frontend/template/output budget changed')
    for i,plan in enumerate(row['plans']):
        root=Path(plan['root'])
        require(plan['index']==i and plan['read']==READS[i] and plan['heads']==1
            and all(plan[k]==v for k,v in arm(UNIT,i).items()) and plan['trace_progress']==(i<3)
            and plan['command']==command(plan) and json.loads((root/'config.json').read_text())==configuration(row,plan)
            and json.loads((root/'plan.json').read_text())==dict(binary=native_context(row,plan)['binary'],native_validation_probe=row['native_validation_probe'])
            and all(sha(root/p)==h for p,h in plan['generated'].items())
            and comparison.fingerprint(environment(row,plan))==plan['environment_sha256'],'unplanned arm/input')
    return json.loads(Path(row['runtime_manifest']['path']).read_text()),python


def records(text,prefix):
    result=[]
    for line in text.splitlines():
        if line.startswith(prefix):
            pairs=re.findall(r'([A-Za-z_][A-Za-z_0-9-]*)=([^\s;]+)',line.split(';',1)[0])
            require(len({k for k,v in pairs})==len(pairs),'duplicate '+prefix);result.append(dict(pairs))
    return result


def one(text,prefix):
    values=records(text,prefix);require(len(values)==1,'missing/duplicate '+prefix);return values[0]


def workspace(rows,split,relocated):
    # Independent byte census of the pinned owned planner;no runtime allocation.
    primary=65536 if split else rows
    payloads=[primary*10240,*([rows*40]*4),rows*16,8192*5120,8192*2560,
        8192*1280,8192*5120,8192*5120,rows*4,8192*2048,*([8192*1280]*3),4]
    reserved=[(x+255)//256*256+256 for x in payloads]
    tail=(rows-primary)*10240;extra=(tail+255)//256*256+256 if tail else 0
    if relocated:extra+=reserved[6]+reserved[12]
    return dict(region=sum(reserved)-(reserved[6]+reserved[12] if relocated else 0),
        tail_payload=tail,extra=extra,mixed_primary=min(rows,65536)*5120,mixed_tail=max(0,rows-65536)*5120)


def request_route(text,plan):
    result=core.request_route(text,plan);n=plan['read']-1
    split=n>90112 and (plan['window']==16 or n<=117488)
    relocated=split and bool(plan['window']==16 or plan['relocate'])
    maximum=n if split else min(n,81920);layout=workspace(maximum,split,relocated)
    expected_windows=48*((n+maximum-1)//maximum)
    main=one(text,'strata prefill: layer-major candidate, ')
    bank=one(text,'strata MoE expert bank: ');admission=one(text,'strata MoE accumulator admission: ')
    active=one(text,'strata MoE split admission: ');window=one(text,'strata MoE window configuration: ')
    work=one(text,'strata MoE accumulator window: ');mixed=one(text,'strata MoE mixed banks: ')
    source=one(text,'strata prefill expert source: ');end=one(text,'strata MoE expert bank complete: ')
    cache=plan['bank']*STRIDE;limit=int(bank['allocation_limit_bytes']);arena=int(admission['shared_arena_bytes'])
    effective=16 if plan['window']==16 and split else 10
    require(main['tokens']==str(n) and main['chunk']=='8192' and main['residual_dtype']=='bf16'
        and main['residual_element_bytes']=='2' and int(main['residual_gpu_rows'])==n
        and int(main['residual_total'])==int(main['residual_gpu'])==n*ROW
        and main['residual_host']==main['residual_staging']=='0' and int(main['layer_cache'])==cache,
        'actual residual placement differs;no automatic recalibration')
    require(bank['requested_slots']==bank['effective_slots']==str(plan['bank']) and bank['staged']=='1'
        and bank['host_slab_bytes']=='1363148800' and int(bank['device_bytes'])==cache
        and int(bank['residual_gpu_rows'])==n and bank['general_map']=='private_scratch_never_published',
        'requested bank/private-map contract differs')
    require(active['active']==str(int(split)) and int(active['rows'])==n
        and int(active['primary_rows'])==(65536 if split else 0)
        and int(active['tail_payload_bytes'])==layout['tail_payload']
        and int(active['extra_allocation_bytes'])==layout['extra']
        and active['prefill_slots']==str(plan['bank']) and active['decode_changed']=='0'
        and int(active['max_single_allocation_bytes'])==limit,'actual accumulator split/extra owner differs')
    extra_rows=records(text,'strata MoE long workspace: ')
    require(len(extra_rows)==int(plan['window']==16 or plan['relocate']), 'long-workspace admission count differs')
    if extra_rows:
        extra=extra_rows[0]
        require(extra==dict(requested_window=str(plan['window']),relocated_requested=str(plan['relocate']),
            relocated_active=str(int(relocated)),effective_window=str(effective),extra_bytes=str(layout['extra']),
            admission='admitted' if split else 'ordinary_fallback',decode_changed='0'),'long-workspace admission differs')
    require(window['requested_mode']==str(plan['window']) and window['effective_mode']==str(effective)
        and int(window['maximum_rows'])==int(work['maximum_rows'])==maximum and window['mixed_fp16_bytes']=='0'
        and int(work['windows'])==expected_windows and work['operator_max']=='8192'
        and work['GPU_resident_windows_only']=='1' and int(work['source_H2D_bytes'])==48*n*40,
        'actual window/map work differs')
    require(mixed['storage']=='scratch' and mixed['extra_allocation_bytes']=='0'
        and int(mixed['primary_bytes'])==layout['mixed_primary'] and int(mixed['tail_bytes'])==layout['mixed_tail']
        and int(mixed['tail_offset'])==layout['region'] and layout['region']+layout['mixed_tail']<=arena
        and mixed['phase']=='FFN_only_after_attention_drain','actual mixed producer/consumer layout differs')
    prefix=((n+15)//16)*16*2056+256;fixed=cache+prefix+536870912+n*ROW+layout['extra']
    require(admission['requested_mode']==str(plan['window']) and int(admission['rows'])==maximum
        and int(admission['gpu_prefix_rows'])==n and int(admission['mixed_fp16_bytes'])==maximum*5120
        and int(admission['fixed_including_headroom_bytes'])==fixed<=int(admission['free_gpu_bytes'])
        and max(cache,n*ROW,layout['extra'],prefix)<=limit,'observed total/max-allocation admission differs')
    require(source['requested_view']==source['effective_view']=='1' and source['layers']=='48'
        and source['owned_slab_bytes']=='0' and int(source['borrowed_layer_span_bytes'])==50292326400
        and source['host_copy']=='eliminated' and source['source_owner']=='caller_through_checked_retirement',
        'immutable all-layer source view differs')
    require(end['final_drain']=='checked' and 0<=int(end['permanent_slots'])<=int(end['total_slots'])<=512
        and int(end['total_slots'])>=plan['bank']
        and int(end['uploads'])+int(end['permanent_hits'])==int(work['dequants'])
        and min(int(end['requested_H2D_bytes']),int(end['permanent_hits']))>=0,
        'expert bank accepted-upload/retirement conservation differs')
    completes=records(text,'strata MoE split complete: ')
    require(len(completes)==int(split),'split completion count differs')
    if split:
        require(completes[0]==dict(rows=str(n),primary_rows='65536',tail_bytes=str(layout['tail_payload']),
            guard='checked',final_drain='checked',decode_changed='0'),'FP32tail guard/drain differs')
    ends=records(text,'strata MoE long workspace complete: ')
    require(len(ends)==int(relocated),'relocated completion count differs')
    if relocated:
        require(ends[0]==dict(rows=str(n),extra_bytes=str(layout['extra']),guards='3_checked',
            final_drain='checked',decode_changed='0'),'relocated job-plane guards/drain differ')
    require('softmax=parallel-exp-ascending-sum' in text and 'softmax=subgroup' not in text
        and 'strata prefill fullfit upload:' not in text,'independent attention/U1 selection changed')
    require(re.findall(r'^strata mtp: prompt expert phase=(\S+) ',text,re.M)==
        ['before-detach','after-detach','after-prompt-before-restore','after-restore']
        and re.findall(r'^strata mtp: experts restored bytes=(\d+),',text,re.M)==['707788800'],
        'actual decode MTP handoff differs')
    if plan['trace_progress']:
        packed=records(text,'strata packed expert bank: ');require(len(packed)==48,'48actual packed layer views required')
        for layer,g in enumerate(packed):
            stride=int(g['stride']);count=min(512,cache//stride)
            require(int(g['layer'])==layer and stride>0 and stride%256==0 and int(g['slots'])==count
                and int(g['used_bytes'])==count*stride<=cache and int(g['backing_bytes'])==cache
                and g['full_fit']==str(int(count==512)),'actual packed layer geometry differs')
    result.update(layout=dict(**layout,shared_arena_bytes=arena,requested_bank=plan['bank'],window_rows=maximum,
        windows=expected_windows,split=split,relocated=relocated,free_gpu_bytes=int(admission['free_gpu_bytes']),
        fixed_including_headroom_bytes=fixed,allocation_limit_bytes=limit,actual_admission=True),
        actual_GU_dispatch=dispatch_reader.dispatch(text,'large'),
        uploads=int(end['uploads']),permanent_hits=int(end['permanent_hits']),
        requested_bank_H2D_bytes=int(end['requested_H2D_bytes']),requested_source_map_H2D_bytes=int(work['source_H2D_bytes']))
    return result


def check_sample(sample,count,read,minimum_ms=10000):
    comparison.check_sample(sample,count,read,minimum_ms)
    require(read in READS and sample['generated_tokens']==24,'exact thinking-on prefill diagnostic only')


def decision(processes):
    require(len(processes)==9 and all(p['passed'] and p['complete'] and p['ownership_closed']
        and p['native_exit_code']==0 and not p['fresh_faults'] and p['index']==i
        and p['read']==READS[i] and len(p['samples'])==RUNS for i,p in enumerate(processes)),
        'nine ordered normal processes/27requests required')
    require(len({(p['native_identity']['pid'],p['native_identity']['start_ticks']) for p in processes})==9,
        'independent native process identities required')
    pairs=[];ratios=[]
    for pair in (processes[3:5],processes[5:7],processes[7:9]):
        arms={ARMS[p['index']]:p for p in pair};ref,can=arms[0],arms[1]
        ratio=can['subsequent_median_ms']/ref['subsequent_median_ms'];ratios.append(ratio)
        for a,b in zip(ref['routes'],can['routes']):
            require(a['requested_source_map_H2D_bytes']==b['requested_source_map_H2D_bytes'],'map volume changed')
            if UNIT in ('coherence','relocation'):
                require(a['actual_GU_dispatch']==b['actual_GU_dispatch'] and a['uploads']==b['uploads']
                    and a['permanent_hits']==b['permanent_hits'] and a['requested_bank_H2D_bytes']==b['requested_bank_H2D_bytes'],
                    'fixed-work unit changed arithmetic/job/upload work')
            else:
                require(b['layout']['windows']==48 and a['layout']['windows']==96
                    and b['actual_GU_dispatch']['dequants']<=a['actual_GU_dispatch']['dequants']
                    and b['actual_GU_dispatch']['jobs']<=a['actual_GU_dispatch']['jobs'],
                    'larger window did not conserve/merge the declared expert work')
        pairs.append(dict(indices=[p['index'] for p in pair],time_ratio=ratio))
    summaries={}
    for selected in (0,1):
        group=[p for p in processes[3:] if ARMS[p['index']]==selected]
        warm=[p['subsequent_median_ms'] for p in group]
        summaries[str(selected)]=dict(warmed_process_ms=warm,warmed_process_PP=[LONG_READ*1000/t for t in warm],
            warmed_PP=LONG_READ*1000/statistics.median(warm),
            warmed_client_TTFT_s=[statistics.median(s['ttft_s'] for s in p['samples'][1:]) for p in group],
            first_samples=[p['samples'][0] for p in group])
    gain=all(r<1 for r in ratios)
    return dict(unit=UNIT,read=LONG_READ,pairs=pairs,arms=summaries,time_ratios=ratios,
        geometric_mean_time_ratio=math.exp(statistics.mean(math.log(r) for r in ratios)),
        repeated_gain=gain,unit_performance_eligible=gain,
        eligible_for_next_independent_question=gain if UNIT=='window' else UNIT!='combined',
        prospective_rule='Safe fixed-work coherence/relocation outcomes permit independent next questions even if slower;losses stay losses. Only repeated fixed-bank window gain admits a combined question. Neither adopts a route.',
        prefill1000_repeated_long_warmed=all(p>=1000 for p in summaries['1']['warmed_process_PP']),
        inference_adopted=False,decode_preservation_qualified=False,task_or_agent_qualified=False,
        sustained_thinking_MTP_strong_weak_qualified=False,physical262144_RESTORE_qualified=False,
        hardware_resource_saturation_qualified=False,ordinary_runtime_teardown_qualified=False)


def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()),'fresh unit plan only;no retry')
    candidate,gate=device_gate();previous=predecessor();original,proof,prior=capacity.basis()
    comparison.shared.models.model_pins(prior['configuration'],prior['model_assets'],full=True)
    revision=device.build.ur_validation.git('rev-parse','HEAD').decode().strip()
    # Preserve original native identity separately from the explicit composed
    # candidate;the overlay revision alone is not its entire native tree.
    row=dict(stage='prepared',unit=UNIT,source_commit=revision,sources=core.pins(revision),
        policies={p:sha(P/p) for p in capacity.POLICIES},original_native_proof=proof,
        component_gate=gate,predecessor=previous,reference_evidence_sha256=signature(compact(prior)),
        candidate_composition_sha256=signature(candidate['source_files']),
        candidate_native_base=candidate['native_base_commit'],candidate_native_overlays=candidate['overlays'],
        **{k:copy.deepcopy(prior[k]) for k in ('configuration','environment_template','model_assets','template',
            'boot','driver','native_validation_probe','python_contract','python_selected')},
        native_contexts=dict(original=dict(binary=original['binaries']['strata'],native_source_commit=capacity.original.SOURCE),
            candidate=dict(binary=candidate['binaries']['strata'],native_source_commit=candidate['source_commit'])),
        runtime_manifest=prior['runtime_manifest'],plans=[],processes=[],request_hashes={},
        host_passed=False,passed=False,complete=False,active=False,ownership_closed=True,
        budgets=dict(wall=2400,cpu=20000,rss_cap=104<<30,text_cap=16<<20,file_cap=32<<20,total_cap=128<<20),
        whole_bytes=1536<<20,
        plan=dict(question=UNIT,
            units='coherence:original14508 versus fresh candidate,legacy128K/bank256/window10;relocation:98K/split128/same window,only job-buffer owner/layout changes;window:128K/fixed bank1,ordinary two windows versus one long window;combined:128K/bank256/window10 versus bank1/window16. Each gets an independent cohort and outcome. Bank-only cost remains in its original closed cohort.',
            inseparable='The larger-window unit requires its charged extra owner,planner,row limits,caller and producer/consumer views together;old layouts cannot hold131071rows. Relocation is separately variable/tested at98K;bank capacity is separately tested at fixed128K/window10. No independently variable kernel or decode policy enters this unit.',
            constants='Same model/assets,FP32accumulation/BF16tensor contract,chunk8192,samequeue,sourceview,packed bank,heads2,dense1,KVresident20480,decode1256,spec4,original Down,ascending QSA and safety. U1is explicitly jit here to isolate native14508 coherence;keep its separately measured positive eager component and later selection eligible. Candidate source composition excludes unrelated KV/verify proposals.',
            order='Three traced3request processes:32Kcandidate then longreference/candidate. Six clean longprocesses reference,candidate,candidate,reference,reference,candidate,each3requests. Three opposite-order adjacent pairs. Thinking-on24outputs;not sustainedTG. First use separate from warmed median2;no concurrent builds/tests/profilers/researchers/reviewers during clean timing.',
            acceptance='Commit host probes and every closed process before next. Require exact frozen inputs/ELFs/providers,actual GPUresidency,free/total/max-single budget,planned window/layout/guards/retirement,token/request/map/job conservation and actual dispatch,normal0/reapedempty,quiet validation,unchanged boot/driver and zero fresh fault. No silent fallback qualifies the long arm.',
            performance='Keep27individualsamples/3paired warmed ratios and spread,first-use,PP/TTFT and requested bytes/jobs. All3ratios<1 give a repeated unit gain without a blanket minimum. Safe fixed-work coherence/relocation may permit the next separately planned question while retaining losses;only repeated fixed-bank window gain admits the combined cohort. Component gain and1000target are separate;no aggregate result adopts a constituent.',
            stopping='Stop first source,provider,owner,clock,token,route,conservation,validation,fault or budget failure;no retry/hashrefresh/sample replacement/exclusion/criteriachange/speedearlystop. Model assets identities throughout,full15hashes afterfinal. Physical-context/task/decode gates remain open.',
            limits='Prefill-only diagnostic;no decode performance,prefill-to-decode numerical state,task/agent,thinking1Kstrong/weak observedMTP,physical262144/tails/RESTORE,ordinary runtime teardown or hardware saturation qualification. Both prefill strategies and retained9849decode reference remain. Newwindow remains opt-in until all gates pass.',
            retention='Keep useful individual samples/source/config/composition/requested-byte/unit decisions. Commit original inputs then retire successful raw/JIT and one-run controls after their final qualification consumer. No new native generation;third candidate plus independent phase references only for this imminent comparison.'))
    ROOT.mkdir(mode=0o700)
    for i,read in enumerate(READS):
        root=ROOT/str(i);root.mkdir(mode=0o700)
        for name in ('tmp','cache'):(root/name).mkdir(mode=0o700)
        plan=dict(index=i,read=read,heads=1,root=str(root),trace_progress=i<3,**arm(UNIT,i))
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));plan['port']=sock.getsockname()[1]
        core.write(root/'config.json',configuration(row,plan))
        core.write(root/'plan.json',dict(binary=native_context(row,plan)['binary'],native_validation_probe=row['native_validation_probe']))
        plan.update(command=command(plan),generated={p:sha(root/p) for p in ('config.json','plan.json')},
            environment_sha256=comparison.fingerprint(environment(row,plan)));row['plans'].append(plan)
    save(row);print('Prepared independent '+UNIT+' model question;commit before host/native.',flush=True)


def context():
    return SimpleNamespace(ROOT=ROOT,KEY=KEY,ORDER=ORDER,RUNS=RUNS,current=current,environment=environment,
        native_context=native_context,save=save,request_route=request_route,check_sample=check_sample,decision=decision)


if __name__=='__main__':
    require(sys.executable==PYTHON and len(sys.argv)==3 and sys.argv[1] in UNITS
        and (sys.argv[2] in ('prepare','host') or sys.argv[2].isdigit() and 0<=int(sys.argv[2])<9),
        'use pinned Python:coherence|relocation|window|combined prepare|host|0..8')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);comparison.build.guards.no_model_owners([])
        if sys.argv[2]=='prepare':prepare()
        elif sys.argv[2]=='host':comparison.host(context())
        else:comparison.run(int(sys.argv[2]),context())
