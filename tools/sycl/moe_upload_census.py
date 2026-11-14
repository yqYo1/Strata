#!/usr/bin/env python3
"""Observe planned MoE job geometry on the qualified native engine.

Main-authored; repository LICENSE applies. This observer changes no inference
math and does not convert planned-job counts into physical traffic or speed.
"""
import copy
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import qsa_cached_bench as cached

P, B, RESULT = cached.P, cached.B, cached.RESULT
THIS = 'tools/sycl/moe_upload_census.py'
ROOT, KEY = B/'moe-job-geometry-v1', 'moe_job_geometry_v1'
READS, RUNS = (32769,65537), 2
BOUNDS = (1,16,32,64,128,256,384,512,640,768,1024,1280,1536,2048,2334,4096,8193)
require = cached.require


def unique_object(pairs):
    result = {}
    for k,v in pairs:
        require(k not in result, 'duplicate census JSON key')
        result[k] = v
    return result


def histogram(values, low, high):
    require(isinstance(values,list) and values, 'nonempty histogram required')
    out = {}
    for pair in values:
        require(isinstance(pair,list) and len(pair)==2, 'histogram pair required')
        index,count = pair
        require(type(index) is int and type(count) is int
            and low<=index<=high and count>0 and index not in out,
            'invalid/duplicate histogram coordinate')
        out[index] = count
    return out


def census(text, read):
    docs = [json.loads(v,object_pairs_hook=unique_object) for v in re.findall(
        r'^strata MoE planned job census: (.+)$',text,re.M)]
    require(len(docs)==48 and [d.get('layer') for d in docs]==list(range(48)),
        'one ordered complete MoE window per layer required')
    jobs, assignments, nonempty, split_experts, maximum = 0,0,0,0,0
    counts = [0]*(len(BOUNDS)-1)
    row_work = [0]*len(counts)
    router_bytes, map_bytes = 0,0
    for d in docs:
        require(all(type(d.get(k)) is int for k in
            ('version','layer','first_row','rows','jobs','allocated_weight_pairs',
             'used_weight_pairs','weight_pair_bytes','requested_router_D2H_bytes',
             'requested_map_H2D_bytes')), 'integer census metadata required')
        require(d['version']==1 and d['first_row']==0 and d['rows']==read-1
            and d['gu_nk']==[1280,2560] and d['down_nk']==[2560,640]
            and d['gu_down_storage']=='bf16' and d['hidden_storage']=='fp16'
            and d['allocated_weight_pairs']==2 and d['used_weight_pairs']==1
            and d['weight_pair_bytes']==9830400,
            'unexpected native16 MoE window/geometry/storage')
        rows = histogram(d['job_rows_histogram'],1,8192)
        experts = histogram(d['jobs_per_expert_histogram'],0,32)
        require(sum(rows.values())==d['jobs']==sum(k*v for k,v in experts.items())
            and sum(k*v for k,v in rows.items())==(read-1)*10
            and sum(experts.values())==512
            and d['requested_router_D2H_bytes']==(read-1)*10*4
            and d['requested_map_H2D_bytes']==(read-1)*10*8,
            'planned jobs/expert/row/metadata-byte conservation failed')
        jobs += d['jobs']; assignments += (read-1)*10
        nonempty += 512-experts.get(0,0)
        split_experts += sum(v for k,v in experts.items() if k>1)
        maximum = max(maximum,max(rows))
        router_bytes += d['requested_router_D2H_bytes']
        map_bytes += d['requested_map_H2D_bytes']
        for n,count in rows.items():
            index = next(i for i in range(len(counts)) if BOUNDS[i]<=n<BOUNDS[i+1])
            counts[index] += count
            row_work[index] += n*count
    require(sum(counts)==jobs and sum(row_work)==assignments,
        'folded job geometry does not conserve original work')
    return dict(version=1,read=read,completed_window_coverage=48,
        original_census_sha256=hashlib.sha256(json.dumps(docs,sort_keys=True,
            separators=(',',':')).encode()).hexdigest(),
        planned_jobs=jobs,planned_expert_row_assignments=assignments,
        nonempty_experts_across_layers=nonempty,split_experts_across_layers=split_experts,
        maximum_job_rows=maximum,
        row_bins=[dict(lower_inclusive=a,upper_exclusive=b,jobs=n,expert_rows=w)
            for a,b,n,w in zip(BOUNDS,BOUNDS[1:],counts,row_work)],
        requested_router_D2H_bytes=router_bytes,requested_map_H2D_bytes=map_bytes,
        first_job_histogram_equals_job_histogram=split_experts==0,
        actual_upload_misses_by_row_qualified=False,per_layer_weight_format_qualified=False,
        limits='Planned jobs after the existing router drain,folded after successful request completion. No extra GPU readback. Job sizes are not upload counts:cache hits and repeated split-expert jobs require separate reconciliation. Requested metadata bytes are not measured physical PCIe traffic.')


def parser_controls():
    # Literal balanced and split examples exercise the scientific conservation
    # contract independently of the production routing implementation.
    def fixture(rows, job_hist, expert_hist):
        return [dict(version=1,layer=layer,first_row=0,rows=rows,
            jobs=sum(v for _,v in job_hist),gu_nk=[1280,2560],down_nk=[2560,640],
            gu_down_storage='bf16',hidden_storage='fp16',allocated_weight_pairs=2,
            used_weight_pairs=1,weight_pair_bytes=9830400,
            requested_router_D2H_bytes=rows*40,requested_map_H2D_bytes=rows*80,
            job_rows_histogram=copy.deepcopy(job_hist),
            jobs_per_expert_histogram=copy.deepcopy(expert_hist)) for layer in range(48)]
    def render(docs):
        return ''.join('strata MoE planned job census: '+json.dumps(d)+'\n' for d in docs)
    balanced = fixture(32768,[[640,512]],[[1,512]])
    result = census(render(balanced),32769)
    require(result['planned_jobs']==48*512 and result['split_experts_across_layers']==0
        and result['first_job_histogram_equals_job_histogram']
        and next(v for v in result['row_bins'] if v['lower_inclusive']==640)['jobs']==48*512,
        'balanced census control')
    split = fixture(32768,[[8192,1],[808,1],[623,184],[624,327]],[[1,511],[2,1]])
    result = census(render(split),32769)
    require(result['planned_jobs']==48*513 and result['split_experts_across_layers']==48
        and not result['first_job_histogram_equals_job_histogram'],
        'split-expert census control')
    cases=[]
    v=copy.deepcopy(balanced);v.pop();cases.append(('missing-layer',render(v)))
    v=copy.deepcopy(balanced);v[-1]['layer']=0;cases.append(('duplicate-layer',render(v)))
    for name,key,value in (
        ('invalid-job-total','jobs',511),
        ('wrong-row-mass','job_rows_histogram',[[639,512]]),
        ('duplicate-row-bin','job_rows_histogram',[[640,256],[640,256]]),
        ('out-of-range-job','job_rows_histogram',[[8193,512]]),
        ('bool-count','job_rows_histogram',[[640,True]]),
        ('expert-total','jobs_per_expert_histogram',[[1,511]]),
        ('metadata-bytes','requested_map_H2D_bytes',1),
        ('wrong-storage','hidden_storage','fp32'),
        ('wrong-window','first_row',8192)):
        v=copy.deepcopy(balanced);v[0][key]=value;cases.append((name,render(v)))
    for name,text in cases:
        failed=False
        try:census(text,32769)
        except RuntimeError:failed=True
        require(failed,'census parser failed to reject '+name)
    failed=False
    try:json.loads('{"rows":1,"rows":2}',object_pairs_hook=unique_object)
    except RuntimeError:failed=True
    require(failed,'duplicate JSON field was accepted')
    return dict(passed=True,positive_cases=2,negative_cases=len(cases)+1,
        scope='CPU-only literal balanced/split,coverage,conservation,bounds,dtype and duplicate-key controls;not GPU or model correctness.')


def context():
    spec=importlib.util.spec_from_file_location('moe_job_geometry_policy',cached.original.__file__)
    ctx=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctx)
    ctx.ROOT,ctx.KEY,ctx.THIS=ROOT,KEY,THIS
    ctx.READS,ctx.ORDER,ctx.RUNS=READS,[(n,1) for n in READS],RUNS
    ctx.CLEAN_FAST=True
    ctx.basis=lambda:cached.basis(observer_files={THIS})
    ctx.decision=lambda processes:dict(census_complete=True,
        observed_lengths=[p['read'] for p in processes],
        captured_requests=sum(len(p['samples']) for p in processes),
        speed_qualified=False,hardware_saturation_qualified=False,inference_adopted=False,
        next='Use folded job geometry to choose a prospective selective-upload or grouped-GEMM comparison. Reconcile split experts and actual cache misses before calling job bins an upload histogram;weight format and per-branch performance remain separate.')
    base_env,base_current,base_route=ctx.environment,ctx.current,ctx.request_route
    def environment(row,plan):
        env=base_env(row,plan)
        env.update(STRATA_SYCL_PREFILL_QSA_MATH='bf16-prefix-ordered-exp',
            STRATA_SYCL_PREFILL_QSA_BULK_HOST_MIRROR='1',
            STRATA_SYCL_PREFILL_EXPERT_UPLOAD_STAGING='0',
            STRATA_TRACE='1',STRATA_PREFILL_LAYER_MAJOR_PROFILE='1')
        return env
    def current(row):
        require(row['parser_controls']['passed'] and len(row['plans'])==2
            and all(p['index']==i and p['read']==READS[i] and p['heads']==1
                and p['trace_progress'] is True for i,p in enumerate(row['plans'])),
            'frozen observer plan or parser controls changed')
        return base_current(row)
    def request_route(text,plan):
        route=base_route(text,plan)
        attention=re.findall(r'^strata prefill: QSA BF16 operands, FP32 online state/accumulation; '
            r'KV=(\S+) transient_bytes=(\d+); source=resident-Mode0 score_pitch=(\d+) '
            r'operand_layout=(\S+) softmax=(\S+)$',text,re.M)
        require(len(attention)==1 and attention[0][0]=='active-logical-prefix'
            and int(attention[0][1])>0
            and attention[0][2:]==('33','row-major','parallel-exp-ascending-sum'),
            'actual ordered-exp attention entry differs')
        route.update(actual_attention_math='bf16-prefix-ordered-exp',bulk_mirror=1,
            expert_upload_staging=0)
        route['planned_job_geometry']=census(text,plan['read'])
        return route
    ctx.environment,ctx.current,ctx.request_route=environment,current,request_route
    return ctx


def prepare():
    controls=parser_controls()
    ctx=context();ctx.prepare()
    row=json.loads((ROOT/'record.json').read_text())
    row['parser_controls']=controls
    for plan in row['plans']:
        plan['trace_progress']=True
        require(plan['environment_sha256']==ctx.comparison.fingerprint(ctx.environment(row,plan)),
            'prepared observer environment changed')
    row['plan']=dict(
        question='What MoE job rows and split-expert coverage actually occur at32K/64K on the current all-GPU ordinary packed384 route?',
        order='Two fresh processes,32K then64K. Two distinct-nonce thinking-on bench.sh requests each,24outputs. Commit each closed process before advancing;all observation costs charged.',
        settings='Qualified cf7/9af eight-ELF build unchanged. Logical262144,resident20480INT8,8192chunk,packed384/source-view1,owned stage,independent decode settings. Ordered-exp QSA,bulk1,source-staging0;qualified runtime safety and private pre-main validation preserved. Trace/stage and existing layer-major CPU census enabled before exec;no unitrace or extra GPU tensor readback.',
        acceptance='CPU parser controls;fresh benchmark/server provider probes;exact source/native/recipe/ELFs/models/runtime and immutable live env;complete thinking/read/reuse/output/clocks/journal;all-GPU ordinary route and source-owned drains;48ordered census windows/request with complete job/expert/assignment/metadata-byte conservation;quiet validation,normal0,reaped owners,unchanged boot/no fresh fault. Full15asset hashes after last process.',
        stopping='Stop first source/provider/owner/count/route/census/conservation/validation/fault/budget failure. No retry,hash refresh,dropped samples or criterion change.',
        limits='Observed planned host routing metadata,not actual upload misses or weight-format coverage,physical PCIe bytes,resource saturation,speed/adoption,sustainedTG,agent/task or full262144/RESTORE qualification.',
        retention='Retain fixed row-bin summaries,split/nonempty counts,source/workload identity and conservation/hash evidence. Full per-layer histograms are temporary inputs to this actual-geometry decision;retire successful logs/journals after reconciliation. Reuse the current engine;no new native generation.')
    ctx.save(row)
    print('Prepared current-engine MoE census;commit before frontend host checks.',flush=True)


if __name__=='__main__':
    require(sys.executable==cached.model.PYTHON,'Use pinned benchmark interpreter:'+cached.model.PYTHON)
    require(len(sys.argv)==2 and sys.argv[1] in ('prepare','host','0','1'),'prepare|host|0|1')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cached.model.original.guards.no_model_owners([])
        if sys.argv[1]=='prepare':prepare()
        elif sys.argv[1]=='host':cached.original.comparison.host(context())
        else:cached.original.comparison.run(int(sys.argv[1]),context())
