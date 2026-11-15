#!/usr/bin/env python3
"""Official, independently repeated fixed/adaptive-cache generation comparison.

Main-authored; repository LICENSE. Reuse the maintained owned bench runner;
qualify this engine, environment and focused workload prospectively.
"""
import copy
import fcntl
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import socket
import statistics
import sys

sys.dont_write_bytecode = True
import check_adaptive_record_failure as refusal
import generation_engine as generation
import pool_spin

positive = refusal.positive
P, B, RESULT, base = refusal.P, refusal.B, refusal.RESULT, refusal.base
SELF = Path(__file__).resolve()
PROFILE = 'data/expert-profile.bin'
require, digest, write, signature = refusal.require, refusal.digest, refusal.write, positive.width.signature
ARMS = (('reference',0),('candidate',8),('candidate',8),('reference',0),('reference',0),('candidate',8))
CAPS = {'regular-code':4, 'varied-prose':2}
OBSERVERS = ('STRATA_TRACE','STRATA_TRACE_ADAPT','STRATA_SYCL_ADAPT_AUDIT',
    'STRATA_SYCL_TEST_ADAPT_RECORD_FAILURE','STRATA_DECODE_TIMING','STRATA_VERIFY_PROFILE',
    'STRATA_DRAFT_TIMING','STRATA_PREFILL_CPU_WAIT_PROFILE')


def source_pins(revision):
    pins=dict(generation.benchmark_pins(revision),**base.controls(revision,controller=SELF))
    raw=base.basis.build.git(P,'show',revision+':'+PROFILE)
    require((P/PROFILE).read_bytes()==raw,'initial expert placement must match its exact Git input')
    pins[PROFILE]=hashlib.sha256(raw).hexdigest()
    return pins


def compact(row):
    out = {k:v for k,v in row.items() if k not in ('sources','environment_template','model_assets')}
    out['input_signatures'] = {k:signature(row[k]) for k in ('sources','environment_template','model_assets')}
    return out


def basis(full=False):
    normal = refusal.normal_entry(); providers = positive.current(normal,full=full)
    failed = json.loads((refusal.ROOT/'record.json').read_text())
    require(failed['stage']=='closed' and failed['passed'] and failed['complete'] and
            failed['ownership_closed'] and not failed['active'] and failed['binary']==normal['binary'] and
            failed['outcome']['native_exit_code']==1 and
            failed['outcome']['adaptive']['stale_event_not_published'], 'native second-record refusal must pass first')
    refusal.current(failed)
    prior = json.loads((positive.width.SPEED_ROOT/'record.json').read_text())
    contract, manifest = positive.width.context(4).python_manifest()
    require(contract==prior['python_contract'] and manifest['selected']==prior['python_selected'],
            'qualified frontend runtime changed')
    refs = {k:dict(path=str(root/'record.json'),sha256=digest(root/'record.json'))
        for k,root in (('positive',positive.ROOT),('record_failure',refusal.ROOT))}
    providers = dict(providers)
    probe = normal['native_validation_probe']['library']; providers[probe['path']] = probe['sha256']
    return normal, prior, contract, manifest, providers, refs


def environment(row,plan):
    original = row['environment_template']['STRATA_MEASUREMENT_ROOT']
    env = {k:v.replace(original,plan['root']) for k,v in row['environment_template'].items()}
    for key in OBSERVERS: env.pop(key,None)
    env['STRATA_ADAPT_NOWAIT'] = '0'
    require(not any(k in env for k in OBSERVERS) and not env.get('LD_PRELOAD') and
            env['UR_ENABLE_LAYERS']=='UR_LAYER_PARAMETER_VALIDATION' and
            env['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD']=='1' and env['EnableDirectSubmission']=='0' and
            env['STRATA_SYCL_PREFILL_SAME_QUEUE']=='1' and env['ZES_ENABLE_SYSMAN']=='0' and
            env['STRATA_SYCL_VERIFY_NATIVE_CAPTURE']=='0' and env['STRATA_SYCL_ADAPT_SYNC']=='1' and
            env['STRATA_SYCL_CACHE_SCRATCH_LEASE']=='1' and env['STRATA_SYCL_CACHE_LOAN_AUDIT']=='0' and
            env['STRATA_SYCL_DECODE_Q6_NATIVE_HALF']=='1', 'immutable clean/safety configuration differs')
    return env


def configuration(row,plan):
    config = copy.deepcopy(row['configuration']); config['log']=str(Path(plan['root'])/'engine.stderr')
    args = config['args']
    require(args.count('--expert-profile')==1 and args[args.index('--expert-profile')+1]==str(P/PROFILE),
            'initial placement must use the pinned profile')
    for key,value in (('--adapt-swaps',plan['adapt_swaps']),('--mtp-max-t',CAPS[row['workload']])):
        require(args.count(key)==1, 'ambiguous selection: '+key); args[args.index(key)+1]=str(value)
    require('--adapt-every' not in args and '--eos-ids' not in args, 'diagnostic EOS or existing adaptive interval')
    args += ['--adapt-every','8']
    return config


def invariant(row,plan):
    value = dict(config=configuration(row,plan),environment=environment(row,plan))
    args = value['config']['args']; args[args.index('--adapt-swaps')+1]='<arm>'
    return signature(json.loads(json.dumps(value).replace(plan['root'],'<process>')))


def decision(processes):
    require(len(processes)==6 and all(p['passed'] and p['complete'] and p['ownership_closed'] and
        len(p['samples'])==1 and (p['kernel'],p['adapt_swaps'])==ARMS[i] for i,p in enumerate(processes)),
        'all six ordered independent processes required')
    ratios=[]; prompt=[]; ttft=[]
    for i in (0,2,4):
        pair = {p['kernel']:p['samples'][0] for p in processes[i:i+2]}
        a,b = pair['candidate'],pair['reference']
        ratios.append(a['decode_tps']/b['decode_tps'])
        prompt.append(a['prompt_ms']/b['prompt_ms']); ttft.append(a['ttft_s']/b['ttft_s'])
    values={}
    for arm in ('reference','candidate'):
        samples=[p['samples'][0] for p in processes if p['kernel']==arm]
        speeds=[s['decode_tps'] for s in samples]
        values[arm]=dict(median_TG=statistics.median(speeds),minimum_TG=min(speeds),maximum_TG=max(speeds),
            acceptance=[s['acceptance_rate'] for s in samples],generated_tokens=[s['generated_tokens'] for s in samples])
    return dict(candidate_over_reference_TG=ratios,geometric_mean_TG_ratio=math.exp(statistics.mean(map(math.log,ratios))),
        repeated_TG_gain=all(r>1 for r in ratios),minimum_gain_floor=False,arms=values,
        candidate_over_reference_prompt_time=prompt,candidate_over_reference_TTFT=ttft,
        prompt_nonregression=all(r<=1 for r in prompt),TTFT_nonregression=all(r<=1 for r in ttft),
        limits='Matched focused workload at one context/width only. Cap2 acceptance uses its own denominator;no strong/weak relabeling,quality,agent,64K/117K preservation,physical262144/RESTORE,adoption or70TG claim.')


def controls():
    rows=[]
    for i,(arm,swaps) in enumerate(ARMS):
        rows.append(dict(kernel=arm,adapt_swaps=swaps,passed=True,complete=True,ownership_closed=True,
            samples=[dict(decode_tps=10.0001 if swaps else 10.,prompt_ms=1000.,ttft_s=1.1,
                acceptance_rate=.7,generated_tokens=2048)]))
    require(decision(rows)['repeated_TG_gain'],'small repeated gain must remain eligible')
    changed=copy.deepcopy(rows);changed[1]['samples'][0]['decode_tps']=9.9999
    require(not decision(changed)['repeated_TG_gain'],'one slower pair cannot be dismissed as noise')
    require(decision(rows)['prompt_nonregression'] and decision(rows)['TTFT_nonregression'], 'independent phase decisions')
    changed=copy.deepcopy(rows)
    for p in changed:
        if p['kernel']=='candidate':p['samples'][0].update(prompt_ms=1010.,ttft_s=1.2)
    verdict=decision(changed)
    require(verdict['repeated_TG_gain'] and not verdict['prompt_nonregression'] and not verdict['TTFT_nonregression'],
            'TG gain cannot conceal prompt/TTFT regression')
    prior=json.loads((positive.width.SPEED_ROOT/'record.json').read_text())
    template=positive.width.generation_context(0).environment(prior,prior['plans'][0])
    original=signature(dict(config=prior['configuration'],environment=template))
    for workload in CAPS:
        row=dict(configuration=copy.deepcopy(prior['configuration']),environment_template=dict(template),workload=workload)
        plans=[dict(root=str(B/('control-'+arm)),adapt_swaps=swaps) for arm,swaps in ARMS[:2]]
        require(invariant(row,plans[0])==invariant(row,plans[1]),'only cache selection/private paths may differ')
        for plan in plans:
            args=configuration(row,plan)['args']
            require(args[args.index('--mtp-max-t')+1]==str(CAPS[workload]) and
                args[args.index('--adapt-swaps')+1]==str(plan['adapt_swaps']) and
                args[args.index('--adapt-every')+1]=='8','actual focused selectors differ')
        row['environment_template'].update({k:'1' for k in OBSERVERS})
        require(not any(k in environment(row,plans[0]) for k in OBSERVERS),'observers must be removed before exec')
        row['environment_template']['EnableDirectSubmission']='1'
        try:environment(row,plans[0])
        except RuntimeError:pass
        else:raise RuntimeError('unsafe runtime setting accepted')
    require(signature(dict(config=prior['configuration'],environment=template))==original,'controls mutated original inputs')
    return dict(passed=True,small_gain_retained=True,slower_pair_retained=True,phase_decisions_separate=True,
        actual_configuration_controls=4,observer_removal=True,unsafe_runtime_rejected=True,GPU_executed=False)


def context(workload):
    require(workload in CAPS, 'unknown focused workload')
    spec=importlib.util.spec_from_file_location('strata_current_adaptive_generation',pool_spin.__file__)
    ctx=importlib.util.module_from_spec(spec); spec.loader.exec_module(ctx)
    ctx.ROOT=B/('adaptive-generation-'+workload+'-v2'); ctx.KEY='adaptive_generation_'+workload.replace('-','_')+'_v2'
    ctx.WORKLOADS=(workload,); ctx.ARMS=ARMS; ctx.MODE='cohort'
    ctx.WORKLOAD_ID='adaptive_generation_20261107_'+workload
    ordinary_route, clean = ctx.route, ctx.clean_observers

    def save(row):
        write(ctx.ROOT/'record.json',row)
        data=json.loads(RESULT.read_text());data[ctx.KEY]=compact(row);write(RESULT,data)

    def current(row):
        require(base.committed(RESULT)[ctx.KEY]==compact(row) and
            source_pins(row['source_commit'])==row['sources'],
            'commit exact comparison source/plan')
        normal,prior,contract,manifest,providers,refs=basis()
        require(refs==row['qualification_refs'] and row['configuration']==prior['configuration'] and
            row['python_contract']==contract and row['python_selected']==manifest['selected'] and
            row['model_assets']==normal['model_assets'] and
            all(row[k]==normal[k] for k in ('binary','native_source_commit','native_validation_probe','runtime_manifest','boot','driver')),
            'native/runtime/model qualification changed')
        require(digest(row['template']['path'])==row['template']['sha256'] and
            base.supervision.artifact_bytes(ctx.ROOT)<=row['whole_bytes'],'template or whole output budget changed')
        require(row['environment_template']==positive.width.generation_context(0).environment(prior,prior['plans'][0]) and
            row['workload']==workload and row['actual_width_cap']==CAPS[workload], 'comparison envelope changed')
        for plan in row['plans']:
            root=Path(plan['root'])
            require((plan['kernel'],plan['adapt_swaps'])==ARMS[plan['index']] and plan['heads']==1 and
                json.loads((root/'config.json').read_text())==configuration(row,plan) and
                all(digest(root/n)==h for n,h in plan['generated'].items()) and
                signature(environment(row,plan))==plan['environment_sha256'] and
                invariant(row,plan)==row['invariant_signature'], 'generated configuration or environment changed')
        return providers,manifest

    def route(text,read):
        result=ordinary_route(text,read)
        dispatch=positive.width.short.dispatches(text,1,positive=True)
        require(len(dispatch)==1 and dispatch[0]['request']==5,'join focused long request to actual dispatch')
        snap=positive.width.short.snapshots(text,1,5)
        require(len(snap['windows'])==2 and {v['tokens'] for v in snap['windows']}=={1,CAPS[workload]} and
            all(v['phase']=='decode' and all(s['shape'][2]==v['tokens'] for s in v['shape_calls'])
                for v in snap['windows']) and snap['completed_head_capture_ids'], 'actual fixed width/heads differ')
        result.update(actual_width_cap=CAPS[workload],request=5,
            IQ3_calls=dispatch[0]['calls'],completed_head_captures=len(snap['completed_head_capture_ids']))
        return result

    def contract(suite,rows):
        require(suite.get('workload_version')==5 and suite.get('thinking') is True and
            suite.get('reasoning_budget_tokens')==0 and suite.get('max_tokens')==16384 and suite.get('speed_only') is True and
            suite.get('diagnostic_subset') is True and suite.get('selected_workloads')==[workload] and
            len(rows)==1 and (rows[0].get('workload'),rows[0].get('run'),rows[0].get('order'))==(workload,0,0) and
            rows[0].get('controller_eligible') is True,'focused official v5 workload/eligibility differs')

    def clean_observers(text):
        clean(text); require('ADAPT_RECORD_DIAGNOSTIC' not in text and 'ADAPT_AUDIT_FAILURE' not in text,
            'diagnostic appeared in clean timing')

    ctx.save,ctx.current,ctx.environment,ctx.decision=save,current,environment,decision
    ctx.route,ctx.generation_contract,ctx.clean_observers=route,contract,clean_observers
    return ctx


def prepare(ctx):
    require(not ctx.ROOT.exists() and ctx.KEY not in base.committed(RESULT),'one fresh focused cohort only')
    old=base.committed(RESULT).get(ctx.KEY.removesuffix('_v2')+'_v1')
    require(old is None or old.get('stage')=='cancelled' and old.get('cancelled_before_native') is True and
        old.get('GPU_executed') is False and not old.get('processes'),'close original unlaunched V1 first')
    host=controls();normal,prior,contract,manifest,providers,refs=basis(full=True)
    revision=base.basis.build.git(P,'rev-parse','HEAD').decode().strip()
    row={k:copy.deepcopy(normal[k]) for k in ('binary','native_source_commit','native_validation_probe','runtime_manifest','boot','driver','model_assets')}
    row.update(stage='prepared',source_commit=revision,
        sources=source_pins(revision),
        configuration=copy.deepcopy(prior['configuration']),template=prior['template'],
        python_contract=contract,python_selected=manifest['selected'],qualification_refs=refs,
        environment_template=positive.width.generation_context(0).environment(prior,prior['plans'][0]),
        workload=ctx.WORKLOADS[0],actual_width_cap=CAPS[ctx.WORKLOADS[0]],plans=[],processes=[],
        host_controls=host,host_passed=False,passed=False,complete=False,active=False,ownership_closed=True,
        whole_bytes=448<<20,budgets=dict(wall=2700,cpu=16200,rss_cap=104<<30,text_cap=16<<20,file_cap=32<<20,total_cap=64<<20),
        plan='Six fresh owned processes R/C,C/R,R/C. Same429b native engine and qualified safety;only adapt-swaps0/8 differs,every8rounds in both. Four official warmups then one focused v5 thinking-on >=32K fresh input/1K generated/512answer-character request;16K output allowance. Code fixed4,prose fixed2 in separate cohorts. Fixed bodies within cohort;no cross-workload history. Disable heavy payload/map/loan/trace/profiler/poll observers;retain existing lightweight IQ3/Q6 route footers and strict private validation. Check actual width/head retirement,normal native/benchmark0,quiet validation,body/journal/native-clock/TTFT/provider/owner gates and no fresh faults. Commit each process before the next;full15asset hashes before and after each six-process cohort. All3paired TG ratios>1 qualifies repeatability,no minimum floor. Judge prompt/TTFT separately. Classification never relabels cap2 acceptance. Stop first safety/ownership/eligibility failure,no retries/hash refresh/exclusions.',
        retention='Keep compact individual samples/routes/cache counts,source/configuration and independent phase decisions. Recover sources from Git;retire successful journals/logs/JIT after verification and pushed results. Same two retained builds,no new engine generation.')
    ctx.ROOT.mkdir(mode=0o700)
    for index,(arm,swaps) in enumerate(ARMS):
        root=ctx.ROOT/str(index);root.mkdir(mode=0o700)
        for name in ('tmp','cache'):(root/name).mkdir(mode=0o700)
        plan=dict(index=index,kernel=arm,adapt_swaps=swaps,heads=1,root=str(root))
        write(root/'config.json',configuration(row,plan))
        write(root/'plan.json',dict(binary=row['binary'],native_validation_probe=row['native_validation_probe']))
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        plan.update(generated={n:digest(root/n) for n in ('config.json','plan.json')},
            command=['/bin/sh',str(P/'bench.sh'),'--config',str(root/'config.json'),'--port',str(port),
                '--long-generation-only','--long-generation-workload',row['workload'],'--long-generation-speed-only',
                '--think','--runs','1','--profile-request-gates','--workload-id',ctx.WORKLOAD_ID,
                '--fail-fast-engine-error','--ready-timeout','600','--out',str(root/'bench')])
        plan['environment_sha256']=signature(environment(row,plan));row['plans'].append(plan)
    row['invariant_signature']=invariant(row,row['plans'][0])
    require(all(invariant(row,p)==row['invariant_signature'] for p in row['plans']),'undeclared arm difference')
    ctx.save(row);print('Prepared six-process focused adaptive cohort;commit before frontend probes.',flush=True)


if __name__=='__main__':
    args=sys.argv[1:]
    require(len(args)==2 and args[0] in CAPS and (args[1] in ('prepare','host') or args[1].isdigit() and int(args[1])<6),
        'use regular-code|varied-prose prepare|host|INDEX0..5')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);base.guards.no_model_owners([])
        ctx=context(args[0])
        if args[1]=='prepare':prepare(ctx)
        elif args[1]=='host':ctx.comparison.host(ctx)
        else:ctx.run(int(args[1]))
