#!/usr/bin/env python3
"""Matched Q6 native-half decode comparison; Main, repository LICENSE.

Reuse this fork's finite bench.sh v5 supervisor. Original short-entry inputs
stay unchanged; this new controller is their sole prospective extension.
"""
import copy
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import check_q6_native_half_model as short
import pool_spin as protocol

P,B,RESULT,base=short.P,short.B,short.RESULT,short.base
ROOT=short.build.ROOT/'generation-half-v1'
KEY='q6_native_half_generation_speed_v1'
SELF=Path(__file__).resolve()
ARMS=(('reference',0),('candidate',0),('candidate',0),
      ('reference',0),('reference',0),('candidate',0))
require,digest=short.require,short.digest


def pinned_sources(revision):
    return dict(short.model_context(0).pinned_sources(revision),**base.controls(revision,controller=SELF))


def normalized(value,root):
    if isinstance(value,str):return value.replace(str(root),'<entry>')
    if isinstance(value,dict):return {k:normalized(v,root) for k,v in value.items()}
    if isinstance(value,list):return [normalized(v,root) for v in value]
    return value


def basis(revision):
    first=short.model_context(0);prior=json.loads((first.ROOT/'record.json').read_text())
    providers,manifest=first.current(prior)
    current=pinned_sources(revision)
    require(set(current)-set(prior['sources'])=={str(SELF.relative_to(P))} and
        all(current.get(n)==h for n,h in prior['sources'].items()),
        'only the new comparison controller may extend the original model inputs')
    data=base.committed(RESULT);environments=[];configurations=[]
    for arm in (0,1):
        ctx=short.model_context(arm);row=json.loads((ctx.ROOT/'record.json').read_text());out=row.get('outcome',{})
        require(data[ctx.KEY]==ctx.compact(row) and row['stage']=='closed' and row['passed'] and
            row['complete'] and row['host_passed'] and row['ownership_closed'] and not row['active'] and
            row['model_executed'] and out['native_exit_code']==0 and out['fresh_faults']==[] and out['boot_unchanged'] and
            all(row[k]==prior[k] for k in ['binary','native_source_commit','sources','model_assets','native_validation_probe']) and
            ctx.pinned_sources(row['source_commit'])==row['sources'],
            'closed same-build half0/1 model entries with unchanged original inputs required')
        require(len(out['samples'])==len(out['routes'])==2 and
            all(s['actual_read_tokens']==32769 and s['actual_reused_tokens']==0 and
                1<=s['generated_tokens']<=24 and s['reasoning_observed'] for s in out['samples']) and
            [s['request_sha256'] for s in out['samples']]==[s['request_sha256'] for s in prior['outcome']['samples']],
            'short-entry request bodies or actual extent differ')
        for request,route in enumerate(out['routes'],2):
            snap=route['Q6_live_snapshot']
            require(snap['request']==request and snap['scope']=='observed_live_capture_cumulative' and
                snap['completed_head_capture_ids'] and all(w['replay_started']==w['replay_completed']
                for w in snap['windows']),'closed original Q6 head/window qualification differs')
        env=normalized(row['environment'],ctx.ROOT)
        require(env.pop('STRATA_SYCL_DECODE_Q6_NATIVE_HALF')==str(arm),'admitted half selector differs')
        environments.append(env);configurations.append(normalized(row['configuration'],ctx.ROOT))
        inv=row['runtime_manifest']
        require(digest(inv['path'])==inv['sha256'] and json.loads(Path(inv['path']).read_text())==providers and
            all(digest(ctx.ROOT/n)==h for n,h in row['generated'].items()),'admitted generated/provider inputs changed')
    require(environments[0]==environments[1] and configurations[0]==configurations[1],
        'only the admitted half selector/private paths may differ')
    native=json.loads((short.build.ROOT/'record.json').read_text())
    names=('src/kernels/cpu/pool.cpp','include/strata/kernels/cpu/pool.hpp')
    pool={n:native['native_sources'][n] for n in names}
    require(native['binaries']['strata']==prior['binary'] and native['source_commit']==prior['native_source_commit'] and
        all(digest(short.build.ROOT/'source'/n)==h for n,h in pool.items()),'qualified native pool changed')
    text=(short.build.ROOT/'source'/names[0]).read_text()
    require('std::getenv("STRATA_POOL_SPIN_US")' in text and
        'spin_before_sleep_ = std::chrono::microseconds' in text,'qualified immutable pool-spin control required')
    return prior,providers,manifest,pool


def context(index=None):
    spec=importlib.util.spec_from_file_location('strata_q6_half_speed_context',protocol.__file__)
    ctx=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctx);ctx.select_mode('cohort')
    ctx.ROOT,ctx.KEY,ctx.ARMS=ROOT,KEY,ARMS
    ctx.model=short.model_context(0);ctx.model.pinned_sources=pinned_sources;ctx.basis=basis
    old_environment,old_route,old_decision,old_checks=ctx.environment,ctx.route,ctx.decision,ctx.selfcheck
    ctx.boundary_checks=[]

    def selected_arm():
        require(type(index) is int and 0<=index<len(ARMS),'actual process index required')
        return int(ARMS[index][0]=='candidate')

    def environment(row,plan):
        baseline=dict(row,environment_template=dict(row['environment_template'],
            STRATA_IQ3_PAIR_AUDIT='0',STRATA_SYCL_Q6_ROUTE_AUDIT='0'))
        env=old_environment(baseline,plan)
        env.update(STRATA_SYCL_DECODE_Q6_NATIVE_HALF=str(int(plan['kernel']=='candidate')),
            STRATA_SYCL_DECODE_Q6_COLUMNS='1',STRATA_SYCL_DECODE_Q6_CENTERING='0',STRATA_SYCL_Q6_ROUTE_AUDIT='1',
            STRATA_SYCL_ADAPT_SYNC='1',STRATA_IQ3_PAIR='1',STRATA_IQ3_PAIR_AUDIT='1',STRATA_IQ_SINGLE_EXACT='0',
            STRATA_CPU_SERVICE='0',STRATA_CPU_WORK_CENSUS='0')
        args=row['configuration']['args']
        require(args[args.index('--spec')+1]=='4' and env['STRATA_SYCL_CACHE_SCRATCH_LEASE']=='1' and
            env['EnableDirectSubmission']=='0' and env['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD']=='1' and
            env['STRATA_SYCL_VERIFY_NATIVE_CAPTURE']==env['ZES_ENABLE_SYSMAN']=='0' and
            env['STRATA_POOL_SPIN_US']=='20000' and plan['adapt_swaps']==0 and
            'STRATA_DECODE_TIMING' not in env and 'STRATA_TRACE' not in env,
            'freeze spec4/lease1,static placement,runtime safety and timing envelope')
        return env

    def invariant(row,plan):
        config=json.loads((Path(plan['root'])/'config.json').read_text())
        require(config['args']==row['configuration']['args'],'native arguments must match')
        env=environment(row,plan);env.pop('STRATA_SYCL_DECODE_Q6_NATIVE_HALF')
        return hashlib.sha256(json.dumps(dict(config=config,environment=env),sort_keys=True)
            .replace(plan['root'],'<process>').encode()).hexdigest()

    def route(text,read):
        result=old_route(text,read);rows=short.dispatches(text,1,positive=True)
        require(len(rows)==1 and rows[0]['request'] in (5,6),'one actual common IQ3 footer per long request')
        result['IQ3_dispatch']=rows[0]
        result['Q6_live_snapshot']=short.snapshots(text,selected_arm(),rows[0]['request'])
        return result

    def boundary(root,expected):
        # Native source emits Q6,then IQ3,then the common cache footer. The
        # existing bounded footer wait therefore includes the complete audit.
        raw,history=protocol.cache_boundary(root,expected)
        rows=short.dispatches(raw.decode(),1,positive=False)
        require([v['request'] for v in rows]==list(range(1,expected+1)) and
            all(v['calls']>0 and v['nt1_rows']>0 for v in rows[4:]),'bind all warmup/long IQ3 request IDs')
        ctx.boundary_checks.append(expected)
        return raw,history

    def decision(processes):
        result=old_decision(processes)
        for name,value in result['workloads'].items():
            wanted='strong' if name=='regular-code' else 'weak'
            value['expected_MTP_class_observed']=all(a['observed_MTP_class']==wanted for a in value['arms'].values())
            value['model_TG_gain_qualified']=value['reproducible_TG_gain'] and value['expected_MTP_class_observed']
        result.update(native_half_model_TG_gain_qualified=all(v['model_TG_gain_qualified'] for v in result['workloads'].values()),
            next='Judge actual MTP workloads separately;retain small gains and unresolved variance. Dedicated64K/117K PP/TTFT,agent/task,physical262144/tails/RESTORE,ordinary provider teardown and adoption remain separate. No saturation claim.')
        return result

    def checks():
        result=old_checks()
        rows=[dict(kernel=arm,adapt_swaps=swaps,passed=True,complete=True,ownership_closed=True,
            native_identity=dict(pid=i+1,start_ticks=i+10),samples=[dict(generated_tokens=1024,
                decode_ms=999.9 if arm=='candidate' else 1000.,prompt_ms=1000.,ttft_s=1.1,
                acceptance_rate=.8 if name=='regular-code' else .3) for name in ctx.WORKLOADS])
            for i,(arm,swaps) in enumerate(ARMS)]
        require(decision(rows)['native_half_model_TG_gain_qualified'],'small paired gains remain eligible')
        slower=copy.deepcopy(rows);slower[1]['samples'][0]['decode_ms']=1000.1
        require(not decision(slower)['native_half_model_TG_gain_qualified'],'retain a slower pair')
        wrong=copy.deepcopy(rows);wrong[1]['samples'][1]['acceptance_rate']=.5
        value=decision(wrong)['workloads']['varied-prose']
        require(value['reproducible_TG_gain'] and not value['model_TG_gain_qualified'],'speed and MTP class are separate')
        result.update(Q6_parser=short.parser_controls(),small_gain_retained=True,slower_pair_retained=True,MTP_class_separate=True)
        return result

    ctx.environment,ctx.invariant,ctx.route=environment,invariant,route
    ctx.cache_boundary,ctx.decision,ctx.selfcheck=boundary,decision,checks
    return ctx


def prepare(ctx):
    ctx.prepare();row=json.loads((ROOT/'record.json').read_text())
    row['qualification_reuse']=dict(build_result_key=short.build.KEY,device_result_key=short.KEY,
        model_result_keys=[short.model_context(a).KEY for a in (0,1)],native_source_commit=row['native_source_commit'],
        engine_sha256=row['binary']['sha256'],original_model_inputs=317,
        scope='Original native/fixture/recipe/runtime checks and both closed spec4/lease1 model entries;only this new comparison controller extends their unchanged317inputs. No native rebuild or inherited sustained-speed result.')
    row['plan']=dict(mode='cohort',workload_version=5,
        question='Does the26-cell decode-only native-half scale-load selector improve sustained generation?',
        order='Six fresh processes0,1,1,0,0,1;three counterbalanced adjacent pairs. Four official warmups,then fixed v5 code/prose once per process. Commit each closed process before next;no retries.',
        constants='Same cb7dd5aad4cb engine/ec4ab9a0e native source,six ELFs,84providers/safety/private preload,throughput BF16 HC/GDN16/MTP16,heads1,131072INT8resident,1256static slots,5workers,spec4/lease1,8192chunk,short-read0,caches0,20ms pool spin,Q6columns1/centering0,IQ3_PAIR1. Only immutable STRATA_SYCL_DECODE_Q6_NATIVE_HALF0/1 differs.',
        observers='Keep parameter validation,retained-runtime preload,common IQ3 audit and Q6 live capture/replay audit. Disable TRACE,CPU attribution,DECODE_TIMING,full API trace and profiler. Audits are common;this does not quantify their overhead or ordinary provider teardown.',
        acceptance='Frozen source/build/fixture/asset/provider/body/route/journal/clock/fault/owner gates;each long request>=32768fresh input,>=1024generated including thinking,>=512answer characters and>=256drafts. Require exact per-request phase/T/shape policy and completed selected Q6 head replay. All four paused cache gates,ordered6IQ3 request IDs,normal0,quiet validation,and full15asset hashes before/after cohort.',
        interpretation='Q6 shape calls describe currently live capture epochs cumulatively,not per-request deltas or retired history. Replay started/completed counts and per-long-request positive head use are separate. No exact-output,all26-shapes,task-quality or saturation claim.',
        decision='Retain all3exact TG ratios/geomean,arm median/min/max,PP/TTFT,lengths and MTP samples per workload. Reproducible gain requires all3candidate/referenceTG>1 without a minimum percentage. Code strong only if every sample>=.60,prose weak only if every sample<=.40,per arm. Classification/speed failure preserves samples and does not stop collection;inconclusive variance remains open.',
        stopping='Stop first source,owner,provider,budget,route,body/token,clock,journal,counter,validation or fresh driver-fault failure. No retry,replacement,hash refresh,speed-based stop or changed criteria.',
        retention='Keep compact individual samples,actual route/cache conservation,identities and decisions. Retire normal logs,journals/JIT after each verified result commit. Existing same-binary comparison plus552b fallback only;no extra engine generation.')
    row['budgets']=dict(wall=3600,cpu=43200,rss_cap=104<<30,text_cap=16<<20,file_cap=32<<20,total_cap=64<<20)
    ctx.save(row);print('Prepared six matched native-half processes/twelve long requests;commit before host probes.',flush=True)


def run(ctx,index):
    ctx.run(index);row=json.loads((ROOT/'record.json').read_text())
    try:
        require(ctx.boundary_checks==[4,5,5,6],'all four paused request/cache gates required')
        proc=row['processes'][index]
        require([r['IQ3_dispatch']['request'] for r in proc['routes']]==[5,6] and
            all(r['Q6_live_snapshot']['completed_head_capture_ids'] for r in proc['routes']),
            'final selected Q6 head/request evidence differs')
        proc['native_half_route']=dict(passed=True,arm=int(ARMS[index][0]=='candidate'),
            boundary_checks=ctx.boundary_checks,completed_requests=6,scope='observed_live_capture_cumulative')
    except BaseException as error:row.update(stage='failed',passed=False,complete=False,native_half_route_error=repr(error))
    ctx.save(row);require(row['stage']!='failed',row.get('native_half_route_error','native-half comparison failed'))


if __name__=='__main__':
    args=sys.argv[1:]
    require(len(args)==1 and (args[0] in ('prepare','host','controls') or
        args[0].isdigit() and int(args[0])<len(ARMS)),'use prepare|host|controls|INDEX0..5')
    index=int(args[0]) if args[0].isdigit() else None;ctx=context(index)
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);base.guards.no_model_owners([])
        if args[0]=='prepare':prepare(ctx)
        elif args[0]=='host':ctx.comparison.host(ctx)
        elif args[0]=='controls':print(json.dumps(ctx.selfcheck(),sort_keys=True))
        else:run(ctx,index)
