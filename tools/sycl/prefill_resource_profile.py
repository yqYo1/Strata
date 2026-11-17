#!/usr/bin/env python3
"""Owned128K attribution and observer-overhead screen; repository LICENSE.

Native14508 remains unchanged. Observe the closed bank256/all-GPU128K path
with the owned Sysman0 indexed collector and host-only CPU-wait partition.
No coarse CUDA-event observer, XeStrata source, default selection or adoption.
"""
import ast
import copy
import fcntl
import json
import math
import os
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
import profile_split as profiler
import qsa16_stream_bench as profile_helpers

P, B, RESULT = capacity.P, capacity.B, capacity.RESULT
ROOT, KEY = B/'prefill-resource-profile-v1', 'prefill_resource_profile_v1'
CAPACITY_KEY = 'coherent_context_model_v2'
READ, RUNS, STRIDE, ROW = 131072, 2, 2662400, 20480
READS = (32769,READ,READ,READ)
SLOTS = (256,)*4
PROFILE = (True,False,True,False)
ORDER = [(read,1) for read in READS]
PYTHON = capacity.reference.model.PYTHON
require, sha, signature = capacity.require, capacity.sha, capacity.signature
compact, POLICIES = core.compact, capacity.POLICIES
THIS = 'tools/sycl/prefill_resource_profile.py'
CONTROL_ROOT, CONTROL_KEY = B/'coherent-prefill-collector-control-v2','coherent_prefill_collector_control_v2'
source, build, shared = core.source, comparison.build, comparison.shared
digest, write = sha, core.write
profile_window = profile_helpers.profile_window
profile_executor_closure = profile_helpers.profile_executor_closure


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
    env.update(STRATA_SYCL_PREFILL_EXPERT_BANK_SLOTS='256',
        STRATA_SYCL_PREFILL_MOE_CENSUS='0',STRATA_PREFILL_LAYER_MAJOR_PROFILE='0')
    if plan['profile']:
        env.update(STRATA_TRACE='1',STRATA_PREFILL_CPU_WAIT_PROFILE='1',
            UNITRACE_ChromeKernelLogging='1',UNITRACE_ChromeEventBufferSize='128')
    require(env['STRATA_SYCL_PREFILL_SAME_QUEUE']=='1'
        and env['STRATA_SYCL_PREFILL_EXPERT_UPLOAD_STAGING']=='0'
        and env['STRATA_SYCL_PREFILL_EXPERT_BANK_PACKED']=='1'
        and env['STRATA_SYCL_CACHE_SCRATCH_LEASE']==env['STRATA_SYCL_PREFILL_MIXED_SCRATCH']=='1'
        and env['STRATA_SYCL_PREFILL_MOE_SPLIT_ACCUMULATOR']=='1'
        and env['STRATA_PREFILL_LAYER_MAJOR_PROFILE']=='0',
        'preserve production placement/safety;no coarse queue-event observer')
    return env


def command(plan):
    root=Path(plan['root'])
    return ['/bin/sh',str(P/'bench.sh'),'--config',str(root/'config.json'),'--port',str(plan['port']),
        '--prefill-exact-tokens',str(plan['read']),'--think','--runs',str(RUNS),
        '--profile-request-gates','--workload-id','native_gdn_20261018',
        '--fail-fast-engine-error','--ready-timeout','600','--out',str(root/'bench')]


def invariant(row,plan):
    env=environment(row,plan)
    for key in ('STRATA_TRACE','STRATA_PREFILL_CPU_WAIT_PROFILE','UNITRACE_ChromeKernelLogging','UNITRACE_ChromeEventBufferSize'):
        env.pop(key,None)
    return signature(json.dumps(dict(config=row['configuration'],env=env),sort_keys=True)
        .replace(plan['root'],'<process>').replace(str(ROOT),'<campaign>'))


def provider_manifest(row,prior):
    manifest=prior['runtime_manifest']
    require(row['runtime_manifest']==manifest,'provider manifest differs from verified original')
    require(sha(manifest['path'])==manifest['sha256'],'provider manifest digest changed')
    files=json.loads(Path(manifest['path']).read_text())
    require(isinstance(files,dict) and len(files)==manifest['files']
        and all(isinstance(p,str) and isinstance(h,str) and sha(p)==h for p,h in files.items()),
        'provider manifest content or executable bytes changed')
    return files


def current(row):
    require(validation.committed(RESULT,(KEY,))==compact(row)
        and core.pins(row['source_commit'])==row['sources']
        and all(sha(P/p)==h for p,h in row['policies'].items()),'commit/freeze exact observer inputs')
    native,proof,prior=capacity.basis(row['original_native_proof'])
    require(row['capacity_gate']==capacity_gate() and row['collector_control']==control_gate()
        and row['reference_evidence_sha256']==signature(compact(prior))
        and row['binary']==native['binaries']['strata'] and row['native_source_commit']==capacity.original.SOURCE
        and all(row[k]==prior[k] for k in ('configuration','environment_template','model_assets','template',
            'boot','driver','native_validation_probe','python_contract','python_selected')),
        'original native/runtime/model or exact collector qualification changed')
    contract,python=core.frontend.python_manifest()
    require(contract==row['python_contract'] and python['selected']==row['python_selected']
        and sha(row['template']['path'])==row['template']['sha256']
        and comparison.build.supervision.artifact_bytes(ROOT)<=row['whole_bytes']
        and len(row['plans'])==len(ORDER),'frontend/template/output budget changed')
    for i,plan in enumerate(row['plans']):
        root=Path(plan['root']);expected=dict(binary=row['binary'],native_validation_probe=row['native_validation_probe'])
        if plan['profile']:expected['native_profile']=plan['native_profile']
        require(plan['index']==i and plan['read']==READS[i] and plan['heads']==1 and plan['slots']==256
            and plan['profile']==plan['trace_progress']==PROFILE[i]
            and plan['command']==command(plan) and invariant(row,plan)==row['invariant_sha256']
            and json.loads((root/'config.json').read_text())==dict(row['configuration'],log=str(root/'engine.stderr'))
            and json.loads((root/'plan.json').read_text())==expected
            and all(sha(root/p)==h for p,h in plan['generated'].items())
            and comparison.fingerprint(environment(row,plan))==plan['environment_sha256'],'unplanned input/observer arm')
        if plan['profile']:
            require(plan['native_profile']['mode']=='aggregate'
                and all(plan['native_profile'][k]==v for k,v in row['collector_control']['assets'].items())
                and plan['native_profile']['environment']['UNITRACE_TraceOutputDir']==str(root/'trace'),
                'exact immutable collector envelope changed')
    files=provider_manifest(row,prior)
    if len(row['processes'])<len(PROFILE) and PROFILE[len(row['processes'])]:
        item=row['collector_control']['assets']['collector'];files[item['path']]=item['sha256']
    return files,python


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



GENERATION_AUDITED_SOURCE='7faf03320cd4bbd188b10fd302cb970f734d86d1'
GENERATION_AUDITED_SHA='eb5bd20c70dbeca6dad253675c76735f9e058507f54668a44074760ae7a5d5d9'


def generation_ast_contract(original,current):
    before,after=ast.parse(original),ast.parse(current)
    added=[n for n in after.body if isinstance(n,ast.FunctionDef) and n.name=='api_segment_evidence']
    require(len(added)==1 and not added[0].decorator_list and added[0].returns is None,
        'only the passive added API-evidence helper is allowed')
    helper=added[0]
    for arg in (*helper.args.posonlyargs,*helper.args.args,*helper.args.kwonlyargs):
        require(arg.annotation is None,'added helper must have no evaluated annotation')
    for value in [*helper.args.defaults,*helper.args.kw_defaults]:
        if value is not None:
            try:ast.literal_eval(value)
            except (ValueError,TypeError):raise RuntimeError('added helper has an evaluated default')
    after.body.remove(helper)
    endpoint=next(n for n in after.body if isinstance(n,ast.FunctionDef) and n.name=='native_end')
    allowed=[n for n in ast.walk(endpoint) if isinstance(n,ast.Compare) and len(n.ops)==1
        and isinstance(n.ops[0],ast.In) and isinstance(n.left,ast.Subscript)
        and isinstance(n.left.value,ast.Name) and n.left.value.id=='profiling'
        and isinstance(n.left.slice,ast.Constant) and n.left.slice.value=='mode'
        and len(n.comparators)==1 and isinstance(n.comparators[0],ast.Tuple)
        and [getattr(v,'value',None) for v in n.comparators[0].elts]==['aggregate','trace','whole_request']]
    require(len(allowed)==1,'exactly one unused whole_request metadata extension required')
    allowed[0].comparators[0].elts.pop()
    require(ast.dump(before,include_attributes=False)==ast.dump(after,include_attributes=False),
        'original host module AST changed beyond the two reviewed passive extensions')
    return dict(added_passive_helper='api_segment_evidence',unchanged_original_AST_except_unused_mode=True,
        permitted_current_mode='aggregate',whole_request_qualified=False)


def generation_metadata_contract():
    path='tools/sycl/generation_engine.py'
    original=validation.git('show',GENERATION_AUDITED_SOURCE+':'+path)
    require(sha_bytes(original)==GENERATION_AUDITED_SHA,'audited original host source is not recoverable')
    current=(P/path).read_bytes();proof=generation_ast_contract(original,current)
    return dict(path=path,original_revision=GENERATION_AUDITED_SOURCE,original_sha256=GENERATION_AUDITED_SHA,
        current_sha256=sha_bytes(current),**proof)


def sha_bytes(raw):
    import hashlib
    return hashlib.sha256(raw).hexdigest()


def reviewed_collector_host_changes(changes,audited):
    # A new prospective host-only rule,not a change to old audit/result hashes.
    # The caller still verifies all original native generated bytes,recipe,
    # 460actual compiler inputs and both exact ELF files after this check.
    proof=generation_metadata_contract();expected=[]
    for item in audited:
        path,current=item['path'],item['current_sha256']
        if path=='tools/sycl/generation_engine.py':
            require(current==proof['original_sha256'],'unexpected original host audit')
            current=proof['current_sha256']
        if path=='tools/sycl/bench_engine.py' and current=='aefdadf94c5580db49cd26002658a62b360557a6bf4ccc0a2de831ac02ed8063':
            current='e263b7c8c8ea723576fffa172d762740dc3769a64e03c958188ec5be43de792a'
        expected.append((path,item['original_sha256'],current))
    require([(i['path'],i['original_sha256'],i['current_sha256']) for i in changes]==expected,
        'collector host-source delta exceeds the prospective AST contract')


def collector_assets():
    module=source.device.load_module('coherent128k_collector_asset_admission',profiler.__file__)
    module.collector_host_changes=reviewed_collector_host_changes
    return module.collector_assets(validation.committed(RESULT))


def collector_context():
    # Reuse the maintained, class-aware known62-kernel compatibility controller.
    # Override its admission with the exact coherent source/fixture, not an old
    # model generation or a nearby similarly named native build.
    ctx=profile_helpers.collector_context()
    ctx.ROOT=ctx.CONTROL_ROOT=CONTROL_ROOT;ctx.KEY=ctx.CONTROL_KEY=CONTROL_KEY
    ctx.SOURCE=P/THIS;ctx.ACTIVE_FROM_EXEC=True
    def admission():
        native,proof,prior=capacity.basis()
        assets=collector_assets()
        require(native['binaries']['sycl_submission_test']['sha256']=='fad9542788d73036841b00dd95d26941b0dd3cd7c480b564f92c5db167fd87e7',
            'exact original compatibility fixture required')
        env=comparison.engine_once.native_environment(prior,prior['environment_template'])
        return dict(native,source_commit=capacity.original.SOURCE),dict(prior,
            environment=env,observer_admission=proof),json.loads(Path(prior['runtime_manifest']['path']).read_text()),assets
    def control_current(row):
        require(validation.committed(RESULT,(CONTROL_KEY,))==ctx.compact(row)
            and ctx.base.controls(row['source_commit'],controller=P/THIS)==row['sources'],
            'commit exact coherent collector-control inputs')
        n,prior,providers,assets=admission()
        require(row['native_source_commit']==capacity.original.SOURCE and row['engine']==n['binaries']['strata']
            and row['binary']==n['binaries']['sycl_submission_test'] and row['observer_admission']==prior['observer_admission']
            and row['native_validation_probe']==prior['native_validation_probe']
            and row['environment_template']==prior['environment'] and row['boot']==prior['boot'] and row['driver']==prior['driver']
            and row['assets']==assets and row['host_metadata_contract']==generation_metadata_contract()
            and row['host_controls']==profile_helpers.collector_controls()
            and row['budgets']==ctx.BUDGET and row['whole_bytes']==128<<20
            and row['profile']['mode']=='aggregate' and all(row['profile'][k]==v for k,v in assets.items())
            and row['profile']['environment']['UNITRACE_TraceOutputDir']==str(CONTROL_ROOT/'trace'),'coherent fixture/collector/runtime/plan changed')
        return providers
    ctx.basis,ctx.current=admission,control_current
    return ctx


def collector_prepare():
    refused=validation.committed(RESULT,('coherent_prefill_collector_control_v1',))
    require(refused['stage']=='failed' and not refused['GPU_executed'] and refused['cases']==[],
        'preserve original collector source-admission refusal')
    ctx=collector_context();ctx.prepare();row=json.loads((CONTROL_ROOT/'record.json').read_text())
    row['host_metadata_contract']=generation_metadata_contract()
    _,prior,_,_=ctx.basis();row['observer_admission']=prior['observer_admission']
    row['host_controls']=profile_helpers.collector_controls()
    row['plan'].update(question='Does the existing owned Sysman0 indexed collector cover the exact coherent14508compatibility fixture,classes,joins,clock and owner retirement?',
        native='All12coherent originalELFs/recipe/dependencies/providers verified fromGit;reuse unchanged fad954compatibility fixture,not its old model qualification. MIT Intel/PTI indexed collector/launcher unchanged;no newnativegeneration.',
        order='Six valid/six rejected CPUinventory controls;then one fresh finite active-from-exec aggregate/kernel-only fixture,ordinary graph path/no arguments. Keep parameter validation/fullsmall trace and all safety settings before exec.',
        acceptance='Exact62instances:1direct/37producer/24consumer,unique native host-flow IDs,finite contained raw clocks,numerical/copy/event/graph/lifetime assertions,normal0,closed/reaped/emptyowner,quietvalidation/no fresh fault,unchangedboot/driver,removednamedcollector session. Stop first mismatch;no retries/hash refresh.',
        limits='Collector/fixture compatibility only;no coherent model trace,observer overhead,physical traffic/saturation,quality/decode/fullcontext/RESTORE/adoption qualification.',
        next='After committed control,run a separate32Kbank256profiled model smoke before the128Kclean/profile/clean bracket.')
    ctx.save(row)
    print('Prepared exact coherent collector control;commit before GPU.',flush=True)


def control_gate():
    value=validation.committed(RESULT,(CONTROL_KEY,));native,_,_=capacity.basis()
    private=json.loads((CONTROL_ROOT/'record.json').read_text());ctx=collector_context()
    require(ctx.compact(private)==value and ctx.base.controls(private['source_commit'],controller=P/THIS)==private['sources'],
        'exact original collector-control source must remain recoverable/current')
    require(value['stage']=='closed' and all(value[k] for k in ('passed','complete','ownership_closed'))
        and not value['active'] and value['native_source_commit']==capacity.original.SOURCE
        and value['engine']==native['binaries']['strata'] and value['binary']==native['binaries']['sycl_submission_test']
        and value['host_metadata_contract']==generation_metadata_contract()
        and len(value['cases'])==1,'fresh coherent collector control required')
    case=value['cases'][0];trace=case['trace']
    require(case['passed'] and case['native_exit_code']==0 and case['ownership_closed'] and not case['fresh_faults']
        and trace['collection_active_from_exec'] and trace['native_host_flow_joins_unique']
        and trace['class_inventory']['observed']==dict(direct=1,producer=37,consumer=24),
        'exact62class/clock/join/lifecycle control required')
    assets=collector_assets();require(assets==value['assets'],'collector bytes changed')
    return dict(key=CONTROL_KEY,source_commit=value['source_commit'],fixture=value['binary'],
        evidence_sha256=signature(value),assets=assets,scope='Current compatibility fixture only;model and overhead separately gated.')


def native_context(row,plan):
    return {k:row[k] for k in ('binary','native_source_commit','native_validation_probe','boot')}


def decision(processes):
    require(len(processes)==4 and all(p['passed'] and p['complete'] and p['ownership_closed']
        and p['native_exit_code']==0 and not p['fresh_faults'] and p['index']==i and p['read']==READS[i]
        and len(p['samples'])==RUNS for i,p in enumerate(processes)),'all4processes/8requests required')
    ratios=[]
    for request in range(RUNS):
        observed=processes[2]['samples'][request];refs=[processes[i]['samples'][request] for i in (1,3)]
        ratios.append(dict(request=request,scope='first-use' if request==0 else 'subsequent',
            prompt=[observed['prompt_ms']/p['prompt_ms'] for p in refs],
            client_TTFT=[observed['ttft_s']/p['ttft_s'] for p in refs]))
    qualified=all(x<=1.2 for r in ratios for key in ('prompt','client_TTFT') for x in r[key])
    return dict(observer_ratios=ratios,maximum_observer_ratio=1.2,observer_cost_qualified=qualified,
        normal_route_attribution_qualified=qualified,clean_speedup_qualified=False,hardware_saturation_known=False,
        inference_adopted=False,limits='Current128Kbank256route attribution only. Requested copies,service and host-wait partitions are not wire/VRAM traffic,arithmetic saturation or complete dependency critical path. No target/quality/decode/RESTORE/adoption qualification.')


def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()),'fresh attribution campaign only')
    winner=validation.committed(RESULT,('prefill_bank_placement_v2',))
    require(winner['stage']=='closed' and winner['passed'] and winner['ownership_closed'] and not winner['active']
        and winner['decision']['eligible_for_further_model_gates'] and winner['native_source_commit']==capacity.original.SOURCE,
        'closed repeated128Kbank256gain required;no inference adoption inherited')
    native,proof,prior=capacity.basis();control=control_gate()
    revision=validation.git('rev-parse','HEAD').decode().strip();ROOT.mkdir(mode=0o700)
    row=dict(stage='prepared',source_commit=revision,sources=core.pins(revision),
        policies={p:sha(P/p) for p in POLICIES},original_native_proof=proof,capacity_gate=capacity_gate(),
        reference_evidence_sha256=signature(compact(prior)),collector_control=control,
        clock_controls=profile_helpers.profile_window_controls(),cleanup_controls=profile_helpers.profile_cleanup_controls(),
        **{k:copy.deepcopy(prior[k]) for k in ('configuration','environment_template','model_assets','template',
            'boot','driver','native_validation_probe','python_contract','python_selected')},
        binary=native['binaries']['strata'],native_source_commit=capacity.original.SOURCE,
        runtime_manifest=prior['runtime_manifest'],plans=[],processes=[],request_hashes={},
        host_passed=False,passed=False,complete=False,active=False,ownership_closed=True,
        budgets=dict(wall=2400,cpu=20000,rss_cap=104<<30,text_cap=32<<20,file_cap=3<<30,total_cap=4<<30),whole_bytes=5<<30,
        plan=dict(question='Which GPUservices,requested copies and disjoint host/wait regions account for actual128Kbank256prefill,with bounded observer overhead?',
            unit='Observation only:unchanged coherent14508ELF,bank256/allR_GPU,BF16/vllm/ordered-exp/GUlarge/Downlarge/window10/chunk8192/samequeue/split-request1inactive/decode1256/resident20480/spec4/assets/safety. No coarse CUDA-event profile;host CPUv4 plus exact Sysman0 indexed collector,active fromexec to avoid known paused-construction loss.',
            order='First separate32769token profiled smoke,2requests. Then128Kclean-before/profile/clean-after,each2distinct-nonce thinking-on24-output requests. Use same pinned bench.sh native_gdn_20261018bodies. Commit each closed process;first-use and subsequent positions separate. No researcher/reviewer during clean speed controls.',
            acceptance='Current source/build/12ELFs/provider/Python/model/owner/token/clock/route guards,all32768or131071Rrows actuallyGPU,staged256/48or96windows,max81920,checkedMTPhandoff,normal0/reapedempty,quietvalidation/no fresh fault. Profile CPUv4/status must be available/complete/nonnegative/disjoint/conserved,zero nesting/open/exception errors. Require flushed timeline,unique native joins and finite contained clocks onPCI05:00.0. Full15modelhashes after finalprocess.',
            overhead='Per position,profile prompt time and TTFTmust each be<=1.2times BOTH bracketing clean values to qualify normal-route attribution. Finish bracket even if this screen fails;preserve original failedscreen. This is observer qualification,not inference speedup or precise variance.',
            clocks='Use existing testedRAW/MONOTONIC bracket mapping beforelaunch,at both requestgates and afterexit;same time namespace,<=1msindividualbracket/<=5msoffsetspan and BOOTTIMEcontinuity. Client windows end at firstnonempty reasoning/content. Retain clipping/boundary counts and uncertainty;no exact native-prefill endpoint claim.',
            stopping='Stop first technical/source/provider/owner/token/clock/route/CPUconservation/validation/fault/budget failure;no retry,pin refresh or criterion changes. Close exact named collector session only after all captured executor identities are confirmed gone;never qualify forced termination as normal.',
            retention='Finite3GiB/file,4GiB/job,5GiBwhole because old32Ktwo-request trace exceeded256MiB;128Kmorejobs requires full active timeline for uniquejoins and request slicing. Reduce immediately at nativeexit;keep useful8samples,clock/conservation/owner evidence,topkernel/families/copybytes and boundedgaps;retire fulltrace/logs/JIT after verifiedreduction/bracket. No newnativegeneration.',
            limitations='Service and API sums may overlap;requested bytes notphysicalPCIe/VRAM traffic;gaps nothardwareidle;saturatedthread maypoll. Perf paranoid4 and prior capability limits retained;no counter/saturation,1000PP,quality/task/agent,sustaineddecode,physical262144serve/RESTORE/ordinaryteardown or adoption claim.'))
    for i,read in enumerate(READS):
        root=ROOT/str(i);root.mkdir();(root/'tmp').mkdir();(root/'cache').mkdir()
        plan=dict(index=i,read=read,slots=256,heads=1,profile=PROFILE[i],trace_progress=PROFILE[i],root=str(root))
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));plan['port']=sock.getsockname()[1]
        generated=dict(binary=row['binary'],native_validation_probe=row['native_validation_probe'])
        if plan['profile']:
            (root/'trace').mkdir();plan['native_profile']=dict(profiler.profiling(root,control['assets']),mode='aggregate');generated['native_profile']=plan['native_profile']
        write(root/'config.json',dict(row['configuration'],log=str(root/'engine.stderr')));write(root/'plan.json',generated)
        plan['generated']={name:sha(root/name) for name in ('config.json','plan.json')};plan['command']=command(plan)
        plan['environment_sha256']=comparison.fingerprint(environment(row,plan));row['plans'].append(plan)
    row['invariant_sha256']=invariant(row,row['plans'][0]);save(row)
    print('Prepared current128Kobserver bracket;commit before host/native.',flush=True)


def close_profile_session(row, plan):
    root, prof = Path(plan['root']), plan['native_profile']
    execution = json.loads((root / 'execution.json').read_text())
    physical = profile_executor_closure(execution)
    start = json.loads((root / 'engine-start.json').read_text())
    profile = source.compatibility.profile
    require(start['native_profile'] == prof and start['boot_id'] == row['boot'] ==
        Path('/proc/sys/kernel/random/boot_id').read_text().strip(), 'original session/boot differs')
    for item in (prof['launcher'], prof['collector']):
        require(digest(item['path']) == item['sha256'] and Path(item['path']).stat().st_size == item['bytes'],
            'original collector executable changed')
    control_root = root / 'collector-controls'; control_root.mkdir(exist_ok=True)
    receipt = control_root / 'execution.json'
    require(not receipt.exists(), 'existing collector-control receipt must be reconciled,not retried')
    outcome, owner = dict(session=prof['session'], physical_closure=physical,
        source_commit=source.device.common.source.git(P, 'rev-parse', 'HEAD').decode().strip()), None

    def persist():
        write(receipt, dict(outcome, active=bool(owner and owner.active),
            commands=owner.commands if owner else []))

    owner = build.supervision.Owner(control_root / 'owned', persist)
    try:
        if profile.session_path(prof['session']).exists():
            command, _, _ = owner.run('stop', [prof['launcher']['path'], '--stop', prof['session']],
                environment(row, plan), P, wall=15, cpu=15, rss_cap=2 << 30,
                text_cap=1 << 20, file_cap=2 << 20, total_cap=8 << 20)
            require(build.supervision.closed(command) and command['exit_code'] == 0, 'collector stop failed')
        require(not profile.session_path(prof['session']).exists(), 'collector session remains')
        outcome['session_removed'] = True
    except BaseException as error:
        outcome['error'] = repr(error)
    finally:
        outcome.update(control_closed=owner.active is None and all(build.supervision.closed(c) for c in owner.commands),
            active=bool(owner.active), session_removed=not profile.session_path(prof['session']).exists())
        persist()
    return outcome


def profile_run(index):
    row = json.loads((ROOT / 'record.json').read_text()); plan = row['plans'][index]
    root, prof = Path(plan['root']), plan.get('native_profile')
    module = source.device.load_module('qsa16_model_profile_comparison', comparison.__file__)
    module.engine_once = source.device.load_module('qsa16_model_profile_native_environment', comparison.engine_once.__file__)
    original_environment = module.engine_once.native_environment
    profile = source.compatibility.profile
    clocks, capture = [profile.clock_pair()], {}
    captured_routes = []

    def native_context_for_profile(value, selected):
        n = native_context(value, selected)
        if prof: n['native_profile'] = prof
        return n

    def expected(n, env):
        claim = json.loads((root / 'engine-start.json').read_text())['native']
        require(os.readlink('/proc/' + str(claim['pid']) + '/ns/time') == os.readlink('/proc/self/ns/time'),
            'native/controller time namespace differs')
        clocks.append(profile.clock_pair())
        if not prof: return original_environment(n, env)
        profile.session_state(prof['session'], 0)
        require(prof['collector']['path'] in shared.runtime.executable_mappings(
            Path('/proc', str(claim['pid']), 'maps').read_text()), 'collector not mapped at native gate')
        value = profile.expected_target_environment(n, env, prof)
        for key in ('UNITRACE_StartPaused', 'UNITRACE_HostTiming', 'UNITRACE_ChromeCallLogging'): value.pop(key)
        value['UNITRACE_Verbose'] = '1'
        return value

    def route(text, selected):
        value = request_route(text, selected)
        if prof:
            value['cpu_partition'] = profile.d.gen.common.extract_cpu_observers(text, 4)
            value['status_partition'] = profile.status_summary(text)
            require(value['status_partition']['scopes'] == (48 if selected['read']==32769 else 96), 'actual per-request48/96window CPU/status coverage required')
            if len(captured_routes) < RUNS: captured_routes.append(value)
        return value

    def analyze(text, selected):
        if not prof:
            require('strata prefill cpu wait:' not in text and 'strata prefill status:' not in text,
                'clean control contains CPU observer')
            return
        require(not any(x in text for x in ('[WARNING]', '[ERROR]')), 'collector warning/error')
        clocks.append(profile.clock_pair())
        native = json.loads((root / 'engine-exit.json').read_text())['native']
        paths = list((root / 'trace').rglob('*.json'))
        require(len(paths) == 1 and paths[0].stat().st_size <= row['budgets']['file_cap'],
            'one bounded fully flushed timeline required')
        events = json.loads(paths[0].read_text())['traceEvents']
        full = profile.trace_summary(events, native['pid'], clocks[0]['raw_before_ns'], clocks[-1]['raw_after_ns'])
        files = list((root / 'bench').glob('*/results.json')); require(len(files) == 1, 'one client result required')
        samples = json.loads(files[0].read_text())['results']['prefill_only']['requests']
        require(len(samples) == RUNS and len(clocks) == 2 + 2 * RUNS, 'complete request/clock samples required')
        windows = []
        require(len(captured_routes)==RUNS,'all live request routes needed for source/collector reconciliation')
        for position,sample in enumerate(samples):
            window = profile_window(sample['client_interval'], clocks)
            begin, end = window['begin_raw_ns'] / 1000, window['end_raw_ns'] / 1000
            clipped, boundary = [], 0
            for e in events:
                if e.get('ph') != 'X' or e.get('cat') not in ('gpu_op', 'cpu_op'): continue
                if not (e['ts'] < end and e['ts'] + e['dur'] > begin): continue
                start, finish = max(begin, e['ts']), min(end, e['ts'] + e['dur'])
                boundary += int(e.get('cat') == 'gpu_op' and (start != e['ts'] or finish != e['ts'] + e['dur']))
                clipped.append(dict(e, ts=start, dur=finish - start))
            reduced = profile.family_summary(clipped, window['begin_raw_ns'], window['end_raw_ns'])
            reconciliation=fused_reconciliation(reduced,captured_routes[position])
            windows.append(dict(clock=window, GPU_boundary_commands=boundary, services=reduced,source_dispatch=reconciliation))
        capture.update(complete=True, native_PID=native['pid'], clocks=clocks, full_capture=full,
            windows=windows, timeline=dict(path=str(paths[0]),sha256=digest(paths[0]),bytes=paths[0].stat().st_size),
            limits='Window service is clipped;payload counts are complete API-request bytes for touching commands,not proportional boundary or physical wire traffic. Native startup/warmups excluded by client windows. No resource saturation or complete dependency-path proof.')

    def finish(value):
        p = value['processes'][-1]
        if prof:
            cleanup = None
            try:
                cleanup = close_profile_session(value, plan)
                p['collector_cleanup'] = cleanup
                require(cleanup['session_removed'] and cleanup['control_closed'] and not cleanup.get('error'),
                    'collector cleanup failed')
                p['resource_observation'] = capture
                require(capture.get('complete'), 'model trace reduction incomplete')
                p['collector_session_removed'] = True
            except BaseException as error:
                p.update(passed=False, error=p.get('error', repr(error)), resource_observation=capture)
                value.update(stage='failed', passed=False, complete=False)
                value.pop('decision', None)
            finally:
                closed = bool(cleanup and cleanup['control_closed'] and cleanup['session_removed'])
                p['ownership_closed'] = bool(p.get('ownership_closed') and closed)
                value['ownership_closed'] = bool(value['ownership_closed'] and closed)
                value['active'] = bool(value['active'] or (cleanup and cleanup['active']))
        save(value)

    module.engine_once.native_environment = expected
    policy = SimpleNamespace(ROOT=ROOT, ORDER=ORDER, RUNS=RUNS, current=current,
        environment=environment, native_context=native_context_for_profile, request_route=route,
        validate_model_text=analyze, decision=decision, save=finish,
        check_sample=globals().get('check_sample', comparison.check_sample))
    if prof: require(not profile.session_path(prof['session']).exists(), 'fresh owned collector session required')
    module.run(index, policy)




def fused_reconciliation(services,route):
    families={v['name']:v for v in services['families']}
    actual=route['actual_GU_dispatch']
    expected={name:actual[part]['fused_jobs'] for name,part in
        (('native16_gu_fused','gu'),('native16_down_fused','down'))}
    observed={name:families.get(name,{}).get('count',0) for name in expected}
    require(expected==observed,'source GU/Down fused dispatch and UnitTrace kernel counts differ')
    return dict(expected=expected,observed=observed,matched=True,
        scope='These two source-counted fused classes only;unobserved operator coverage and physical saturation remain unqualified.')


HOST_ROOT,HOST_KEY=B/'prefill-resource-host-v4','prefill_resource_host_v4'


def host_compact(row):
    value={k:v for k,v in row.items() if k not in ('sources','environment_template')}
    value['source_inputs']=dict(files=len(row['sources']),sha256=signature(row['sources']))
    value['environment_template_sha256']=signature(row['environment_template'])
    return value


def host_save(row):
    write(HOST_ROOT/'record.json',row)
    x=json.loads(RESULT.read_text());x[HOST_KEY]=host_compact(row);write(RESULT,x)


def host_prepare():
    require(not HOST_ROOT.exists() and HOST_KEY not in json.loads(RESULT.read_text()),'fresh host controls only')
    previous=validation.committed(RESULT,('prefill_resource_host_v1',))
    require(previous['stage']=='failed' and previous['no_host_test_or_native_executed']
        and previous['ownership_closed'],'preserve original host-only serialization refusal')
    previous_unit=validation.committed(RESULT,('prefill_resource_host_v2',))
    require(previous_unit['stage']=='closed' and previous_unit['passed'] and previous_unit['tests']==8,
        'preserve exact earlier host controls')
    previous_plan=validation.committed(RESULT,('prefill_resource_host_v3',))
    require(previous_plan['stage']=='superseded_before_execution'
        and previous_plan['no_host_test_or_native_executed'] and previous_plan['ownership_closed'],
        'preserve unusedHOSTv3 source before fresh provider-manifest controls')
    native,proof,prior=capacity.basis();revision=validation.git('rev-parse','HEAD').decode().strip()
    HOST_ROOT.mkdir(mode=0o700)
    for name in ('tmp','cache'):(HOST_ROOT/name).mkdir()
    old=prior['environment_template']['STRATA_MEASUREMENT_ROOT']
    env={k:v.replace(old,str(HOST_ROOT)) for k,v in prior['environment_template'].items()}
    row=dict(stage='prepared',source_commit=revision,sources=core.pins(revision),environment_template=env,
        command=[PYTHON,'-B','-m','unittest','tools.test_prefill_resource_profile'],
        native_source_commit=capacity.original.SOURCE,original_native_proof=proof,
        plan='Nine host route/budget/source/drain/math/observer/provider-manifest refusal controls plus existing exact62class,RAWclock-window and physical-owner-cleanup controls. No GPU/device initialization or model. Commit exact source/recipe/environment before run;stop first failure,no retry/hash refresh.',
        passed=False,complete=False,active=False,ownership_closed=True)
    host_save(row);print('Prepared host controls;commit before run.',flush=True)


def host_gate():
    row=validation.committed(RESULT,(HOST_KEY,))
    private=json.loads((HOST_ROOT/'record.json').read_text())
    require(host_compact(private)==row and core.pins(private['source_commit'])==private['sources'],
        'exact source of closed host controls changed')
    require(row['stage']=='closed' and all(row[k] for k in ('passed','complete','ownership_closed'))
        and not row['active'],'closed committed host controls required')
    require(row['tests']==9,'all nine current host controls required')
    return dict(key=HOST_KEY,source_commit=row['source_commit'],evidence_sha256=signature(row),tests=9)


def host_check():
    row=json.loads((HOST_ROOT/'record.json').read_text())
    require(validation.committed(RESULT,(HOST_KEY,))==host_compact(row)
        and core.pins(row['source_commit'])==row['sources'] and row['stage']=='prepared'
        and row['command']==[PYTHON,'-B','-m','unittest','tools.test_prefill_resource_profile'],
        'commit exact fresh host-control inputs')
    capacity.basis(row['original_native_proof']);owner=comparison.build.supervision.Owner(HOST_ROOT/'owned')
    try:
        run,out,err=owner.run('host',row['command'],row['environment_template'],P,
            wall=90,cpu=60,rss_cap=1<<30,text_cap=256<<10,file_cap=2<<20,total_cap=8<<20)
        require(build.supervision.closed(run) and run['exit_code']==0 and 'Ran 9 tests in ' in err.read_text()
            and err.read_text().rstrip().endswith('OK'),'resource host controls failed')
        row.update(stage='closed',passed=True,complete=True,tests=9)
    except BaseException as error:row.update(stage='failed',error=repr(error))
    finally:
        row.update(active=bool(owner.active),ownership_closed=owner.active is None
            and all(build.supervision.closed(c) for c in owner.commands));host_save(row)
    require(row['passed'] and row['ownership_closed'],row.get('error','host controls failed'))
    print('Nine resource host controls passed;commit before collector/native.',flush=True)


if __name__=='__main__':
    require(sys.executable==PYTHON,'use pinned model Python')
    actions=('host-prepare','host-check','collector-prepare','collector-check','prepare','host')
    require(len(sys.argv)==2 and (sys.argv[1] in actions or sys.argv[1].isdigit()
        and 0<=int(sys.argv[1])<len(ORDER)),'host-prepare|host-check|collector-prepare|collector-check|prepare|host|0..3')
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);comparison.build.guards.no_model_owners([])
        action=sys.argv[1]
        if action=='host-prepare':host_prepare()
        elif action=='host-check':host_check()
        else:
            host_gate()
            if action=='collector-prepare':collector_prepare()
            elif action=='collector-check':collector_context().check()
            elif action=='prepare':prepare()
            elif action=='host':comparison.host(SimpleNamespace(ROOT=ROOT,KEY=KEY,current=current,environment=environment,save=save))
            else:profile_run(int(action))
