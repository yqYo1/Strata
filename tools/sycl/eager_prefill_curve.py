#!/usr/bin/env python3
"""Owned same-image eager-prefill length/time/client-TTFT characterization.

Main-authored,repository LICENSE. Uses the owned bench.sh supervisor and
workspace admission readers. No native rebuild,XeStrata input,or source-unit
adoption. The64Ksource-unit failure remains part of qualification evidence.
"""
import copy
import fcntl
import hashlib
import json
import math
from pathlib import Path
import socket
import statistics
import sys
from types import SimpleNamespace

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import prefill_curve_fit as fits
import prefill_workspace_admission as admission

capacity,core=admission.capacity,admission.core
comparison=core.comparison
P,B,RESULT=capacity.P,capacity.B,capacity.RESULT
require,sha,signature=capacity.require,capacity.sha,capacity.signature
THIS='tools/sycl/eager_prefill_curve.py'
ROOT,KEY=B/'eager-prefill-length-curve-v2','eager_prefill_length_curve_v2'
MODEL_KEY='prefill_64k_revalidation_v2'
PYTHON=capacity.reference.model.PYTHON
READS=(16385,81921,65537,16385,81921,32769,49153,57345)
ORDER=[(n,1) for n in READS]
RUNS=3
compact=core.compact


def save(row):
    core.write(ROOT/'record.json',row)
    values=json.loads(RESULT.read_text());values[KEY]=compact(row);core.write(RESULT,values)


def previous_failure():
    key='eager_prefill_length_curve_v1'
    p=admission.device.build.ur_validation.committed(RESULT,(key,))
    require(p['stage']=='failed' and not p['passed'] and not p['complete'] and not p['active']
        and not p['ownership_closed'] and p['closure_reconciliation']['ownership_reconciled_closed']
        and not p['failure_diagnosis']['native_engine_started'] and not p['failure_diagnosis']['GPU_executed']
        and p['failure_diagnosis']['missing_argument']=='--prefill-short-diagnostic'
        and not p['processes'][0]['model_executed'] and p['raw_capture_retirement']['complete'],
        'preserve and close the rejected V1client admission before the corrected plan')
    return dict(key=key,source_commit=p['source_commit'],evidence_sha256=signature(p),
        original_archive=p['original_evidence_archive']['commit'],original_failed_outcome_preserved=True,
        ownership_reconciled_closed=True,native_engine_started=False,GPU_executed=False)


def prior_gate(native,component,prior):
    p=admission.device.build.ur_validation.committed(RESULT,(MODEL_KEY,))
    archive=p['original_evidence_archive']
    original=json.loads(admission.device.build.source.git(P,'show',
        archive['commit']+':'+archive['path']))[MODEL_KEY]
    require(p['stage']=='closed' and p['passed'] and p['complete'] and p['ownership_closed']
        and not p['active'] and len(p['processes'])==12 and p['decision']==original['decision']
        and signature(original)==archive['original_row_sha256']
        and p['source_commit']==original['source_commit'] and p['component_gate']==component
        and p['native_contexts']['candidate']==dict(binary=native['binaries']['strata'],
            native_source_commit=native['source_commit']), 'closed exact same-image model evidence required')
    for current,old in zip(p['processes'],original['processes']):
        require(all(current[k]==old[k] for k in ('index','read','kernel','heads','samples',
            'subsequent_median_ms','native_identity','passed','complete','ownership_closed',
            'native_exit_code','fresh_faults','boot_unchanged','clean_speed'))
            and current['passed'] and current['complete'] and current['ownership_closed']
            and current['native_exit_code']==0 and not current['fresh_faults'] and current['boot_unchanged'],
            'original model outcomes/samples changed')
        for i,name in enumerate(current['route_refs']):
            restored=copy.deepcopy(p['shared_workcases'][name])
            restored['layout']['free_gpu_bytes']=current['observed_free_gpu_bytes'][i]
            require(restored==old['routes'][i],'original actual route reconstruction differs')
    units=p['decision']['units']
    pairs=((4,5),(7,6),(9,11))
    times=[p['processes'][b]['subsequent_median_ms']/p['processes'][a]['subsequent_median_ms']
        for a,b in pairs]
    ttfts=[statistics.median(s['ttft_s'] for s in p['processes'][b]['samples'][1:])/
        statistics.median(s['ttft_s'] for s in p['processes'][a]['samples'][1:]) for a,b in pairs]
    require(times==units['eager']['warmed_time_ratios'] and ttfts==units['eager']['warmed_TTFT_ratios']
        and all(t<1 for t in times) and statistics.median(ttfts)<=1
        and units['eager']['performance_eligible'] and units['combined']['performance_eligible']
        and not units['coherence']['performance_eligible'] and not p['decision']['inference_adopted']
        and not p['decision']['current9849decode_preserved'],'preserve separate unit gates;only U1is eligible')
    require(all(p[k]==prior[k] for k in ('configuration','template','boot','driver',
        'native_validation_probe','python_contract','python_selected','runtime_manifest'))
        and all(p[k]==core.compact(prior)[k] for k in ('environment_template_sha256','model_assets_sha256')),
        'original model/decode placement,providers or template changed')
    eager=p['processes'][2]
    return dict(key=MODEL_KEY,evidence_sha256=signature(p),archive=archive['commit'],
        source_commit=p['source_commit'],eager_time_ratios=times,eager_TTFT_ratios=ttfts,
        source_unit_eligible=False,source_unit_adopted=False,
        request_hashes=p['request_hashes']['65537'],
        actual64Kroute_sha256=[signature(p['shared_workcases'][n]) for n in eager['route_refs']],
        limits='Independent U1eligibility only;source selection,current decode,physical context and model adoption remain pending.')


def environment(row,plan):
    env=comparison.environment(row,plan)
    env.update(STRATA_SYCL_PREFILL_EXPERT_BANK_SLOTS='384',STRATA_SYCL_PREFILL_MOE_WINDOW='10',
        STRATA_SYCL_PREFILL_MOE_RELOCATE_JOB_BUFFERS='0',STRATA_SYCL_PREFILL_FULLFIT_UPLOAD='eager',
        STRATA_SYCL_PREFILL_MOE_CENSUS='0',STRATA_PREFILL_LAYER_MAJOR_PROFILE='0')
    if plan['trace_progress']:env['STRATA_TRACE']='1'
    require(env['STRATA_SYCL_PREFILL_SAME_QUEUE']=='1'
        and env['STRATA_SYCL_PREFILL_EXPERT_UPLOAD_STAGING']=='0'
        and env['STRATA_SYCL_PREFILL_EXPERT_BANK_PACKED']=='1'
        and env['STRATA_SYCL_CACHE_SCRATCH_LEASE']==env['STRATA_SYCL_PREFILL_MIXED_SCRATCH']=='1'
        and env['STRATA_SYCL_PREFILL_DOWN_SPLIT_KERNEL']=='original'
        and env['STRATA_SYCL_PREFILL_MOE_SPLIT_ACCUMULATOR']=='1','fixed qualified queue/layout required')
    return env


def native_context(row,plan):
    return {k:row[k] for k in ('binary','native_source_commit','native_validation_probe','boot')}


def configuration(row,plan):
    return dict(row['configuration'],log=str(Path(plan['root'])/'engine.stderr'))


def command(plan):
    root=Path(plan['root'])
    return ['/bin/sh',str(P/'bench.sh'),'--config',str(root/'config.json'),'--port',str(plan['port']),
        '--prefill-exact-tokens',str(plan['read']),'--think','--runs',str(RUNS),'--profile-request-gates',
        '--workload-id','native_gdn_20261018','--fail-fast-engine-error','--ready-timeout','600',
        '--out',str(root/'bench')]+(['--prefill-short-diagnostic'] if plan['read']<32768 else [])


def committed_sources(revision):
    sources=core.pins(revision)
    require({THIS,'tools/prefill_curve_fit.py','tools/test_prefill_curve_fit.py'}<=sources.keys(),
        'commit the exact controller/fit/test sources before preparation')
    blobs=dict(core.source.device.common.source.blobs(P,revision,sources))
    require(set(blobs)==set(sources) and all(hashlib.sha256(blobs[p]).hexdigest()==h
        and sha(P/p)==h for p,h in sources.items()),'source snapshot is not committed/recoverable')
    return sources


def normalized_route(route):
    value=copy.deepcopy(route);value['layout'].pop('free_gpu_bytes');return value


def validate_receipts(processes,reference):
    require(len(processes)<=len(READS),'too many curve receipts')
    require(len({(p['native_identity']['pid'],p['native_identity']['start_ticks']) for p in processes})==len(processes),
        'independent native identities required')
    previous={}
    for i,p in enumerate(processes):
        require(p['index']==i and p['read']==READS[i] and p['heads']==1 and p['kernel']=='eager'
            and p['clean_speed']==(i>=2) and p['passed'] and p['complete'] and p['ownership_closed']
            and p['diagnostic_short_input']==(p['read']<32768)
            and p['native_exit_code']==0 and not p['fresh_faults'] and p['boot_unchanged']
            and len(p['samples'])==len(p['routes'])==RUNS,'ordered exact closed curve receipts required')
        require(all(s['actual_read_tokens']==p['read'] and s['actual_reused_tokens']==0
            and s['generated_tokens']==24 and math.isfinite(s['prompt_ms']) and s['prompt_ms']>10000
            for s in p['samples']) and p['subsequent_median_ms']==statistics.median(
                s['prompt_ms'] for s in p['samples'][1:]),'sample extents/clocks/reduction changed')
        fingerprint=dict(requests=[s['request_sha256'] for s in p['samples']],
            routes=[signature(normalized_route(r)) for r in p['routes']])
        if p['read'] in previous:require(fingerprint==previous[p['read']],'traced/clean actual work changed')
        previous[p['read']]=fingerprint
        if p['read']==65537:
            require(fingerprint==dict(requests=reference['request_hashes'],routes=reference['actual64Kroute_sha256']),
                'same-image64Kbaseline request/work changed')


def after_execution(outcome,row,plan):
    require(outcome['binary']==row['binary'] and outcome['native_source_commit']==row['native_source_commit'],
        'native source/binary changed')
    outcome['diagnostic_short_input']=plan['read']<32768
    validate_receipts([*row['processes'],outcome],row['prior_model_gate'])


def current(row):
    require(admission.device.build.ur_validation.committed(RESULT,(KEY,))==compact(row)
        and committed_sources(row['source_commit'])==row['sources']
        and row['previous_attempt']==previous_failure()
        and all(sha(P/p)==h for p,h in row['policies'].items()),'frozen committed curve inputs required')
    _,proof,prior=capacity.basis();native,component=admission.device_gate()
    require(row['original_reference_proof_sha256']==signature(proof) and row['component_gate']==component
        and row['binary']==native['binaries']['strata'] and row['native_source_commit']==native['source_commit']
        and row['native_composition_sha256']==signature(native['source_files'])
        and row['prior_model_gate']==prior_gate(native,component,prior)
        and all(row[k]==prior[k] for k in ('configuration','environment_template','model_assets','template',
            'boot','driver','native_validation_probe','python_contract','python_selected','runtime_manifest')),
        'native generation,prior gate,model,decode placement or runtime changed')
    contract,python=core.frontend.python_manifest()
    require(contract==row['python_contract'] and python['selected']==row['python_selected']
        and sha(row['template']['path'])==row['template']['sha256']
        and comparison.build.supervision.artifact_bytes(ROOT)<=row['whole_bytes']
        and len(row['plans'])==len(READS),'frontend,template or output envelope changed')
    for i,plan in enumerate(row['plans']):
        root=Path(plan['root'])
        require(plan['index']==i and plan['read']==READS[i] and plan['heads']==1 and plan['kernel']=='eager'
            and plan['upload']=='eager' and plan['bank']==384 and plan['window']==10 and plan['relocate']==0
            and plan['trace_progress']==(i<2) and plan['command']==command(plan)
            and json.loads((root/'config.json').read_text())==configuration(row,plan)
            and json.loads((root/'plan.json').read_text())==dict(binary=row['binary'],native_validation_probe=row['native_validation_probe'])
            and all(sha(root/p)==h for p,h in plan['generated'].items())
            and comparison.fingerprint(environment(row,plan))==plan['environment_sha256'],'unplanned curve case')
    validate_receipts(row['processes'],row['prior_model_gate'])
    for case in row['processes']:
        e=json.loads((ROOT/str(case['index'])/'execution.json').read_text())
        require(not e['active'] and all(e.get(k)==v for k,v in case.items())
            and all(comparison.build.supervision.closed(c) for c in e['commands']),
            'original closed execution receipt changed')
    return json.loads(Path(row['runtime_manifest']['path']).read_text()),python


def check_sample(sample,count,read,minimum_ms=10000):
    comparison.check_sample(sample,count,read,minimum_ms)
    require(read in READS and sample['generated_tokens']==24,'exact planned length/24outputs required')
    require(sample.get('long_input_extent') is (read>=32768),'short diagnostic/long-input label differs')


def decision(processes):
    require(len(processes)==len(READS),'complete eight-process/24request cohort required')
    points=[]
    for p in processes[2:]:
        points.append(dict(read=p['read'],held_out=p['read']==fits.HELD_OUT_READ,
            diagnostic_short_input=p['diagnostic_short_input'],first_use=p['samples'][0],
            subsequent_prompt_ms=[s['prompt_ms'] for s in p['samples'][1:]],
            subsequent_TTFT_s=[s['ttft_s'] for s in p['samples'][1:]],
            median_prompt_ms=p['subsequent_median_ms'],median_PP=p['read']*1000/p['subsequent_median_ms'],
            median_TTFT_s=statistics.median(s['ttft_s'] for s in p['samples'][1:])))
    return dict(points=sorted(points,key=lambda p:p['read']),approximation=fits.characterize(points),
        single_independent_process_per_length=True,source_unit_adopted=False,matched_speedup=False,
        current9849decode_preserved=False,physical262144_RESTORE_task_agent_longthinking_qualified=False,
        ordinary_teardown_qualified=False,throughput_target_qualified=False,inference_adopted=False,
        hardware_limiting_resource='unknown',
        next='Use held-out residuals and actual job/copy/placement work to select the next bounded performance question;'
            'repeat uncertain curve evidence prospectively. No mixed-route physical intercept or extrapolation.')


def context():
    return SimpleNamespace(ROOT=ROOT,KEY=KEY,ORDER=ORDER,RUNS=RUNS,current=current,environment=environment,
        native_context=native_context,save=save,request_route=admission.request_route,
        check_sample=check_sample,after_execution=after_execution,decision=decision)


def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()),'fresh curve only;no retry')
    failed=previous_failure()
    _,proof,prior=capacity.basis();native,component=admission.device_gate();previous=prior_gate(native,component,prior)
    revision=admission.device.build.source.git(P,'rev-parse','HEAD').decode().strip();sources=committed_sources(revision)
    comparison.shared.models.model_pins(prior['configuration'],prior['model_assets'],full=True)
    ROOT.mkdir(mode=0o700)
    row=dict(stage='prepared',source_commit=revision,sources=sources,previous_attempt=failed,
        policies={p:sha(P/p) for p in capacity.POLICIES},original_reference_proof_sha256=signature(proof),
        component_gate=component,prior_model_gate=previous,binary=native['binaries']['strata'],
        native_source_commit=native['source_commit'],native_base=native['native_base_commit'],
        native_overlays=native['overlays'],native_composition_sha256=signature(native['source_files']),
        **{k:copy.deepcopy(prior[k]) for k in ('configuration','environment_template','model_assets','template',
            'boot','driver','native_validation_probe','python_contract','python_selected','runtime_manifest')},
        plans=[],processes=[],request_hashes={},host_passed=False,passed=False,complete=False,active=False,
        ownership_closed=True,budgets=dict(wall=2400,cpu=20000,rss_cap=104<<30,text_cap=16<<20,
            file_cap=32<<20,total_cap=128<<20),whole_bytes=2048<<20,
        plan=dict(question='Describe actual input-length/prefill-time/clientTTFT scaling on the same eager native image.',
            provenance='Main-owned existing3a195fexplicit15overlay composition over14508;U1individual copies. '
                'No new native generation,math,queue,precision,decode selector or XeStrata inputs. Preserve failed C0source gate.',
            order=list(READS),samples='Traced16Kand80Kadmissions,then clean64K/16K/80K/32K/48K/held-out56K. '
                'Each is one fresh owned process with3thinking-on24-output bench.sh requests. First separate;'
                'median subsequent2is the length point. All24samples retained. Commit every closed case before next. '
                'No concurrent builds,tests,GPUjobs,profilers,researchers or reviewers during clean timing.',
            settings='Fixed384bank/window10/chunk8192,eager,all BF16residual rowsGPU/48windows,extra0;'
                '16Kuses the existing explicit --prefill-short-diagnostic CLI scope. '
                'native ordered-exp QSA,largeGU/Down,split requested1/active0,private source map. '
                'Keep current pinned decode1256/KVresident20480/spec4/MTP,heads,dense,Sysman0and copy/direct-submission safety. '
                'Provider validation remains charged;STRATA_TRACEonly for admissions.',
            acceptance='Exact committed source/build/8fixtures/helpers/generated inputs/runtime/Python/model identities. '
                'Exact actual token counts,zero reuse,thinking and24outputs,valid client/engine clocks and journals. '
                'All-GPU residuals,fixed ordinary admission and actual dispatch,24eager layers,copy conservation,checked drains. '
                'Traced/clean repeated lengths and prior64Kmust conserve request/job/map/upload work. Normal0/reapedempty,'
                'quiet validation,unchangedboot/driver,no fresh fault. Stop first mismatch;no silent fallback.',
            analysis=fits.SELECTION+' Fit first/warmed prompt/clientTTFT separately using actual reads and milliseconds. '
                'Retain every affine/quadratic/NlogNcoefficient,residual and LOOerror;reserved56Kerror is reported even if large. '
                'The empirical two-sample half-range is not a confidence bound. Report adjacent slopes;'
                'no extrapolation,mixed-route intercept or physical fixed-cost/saturation claim.',
            stopping='Stop first source,owner,provider,route,clock,count,validation,fault or finite budget failure. '
                'No retry,hash refresh,replacement,speed-based exclusion or earlystop. Full15model hashes before preparation '
                'and after process7. A poor held-out fit stays a result;further evidence requires a new committed plan.',
            limits='Characterization,not a paired source gain,independent repetitions at each length,1000PPqualification,'
                'sustainedTG,task/agent,longthinking MTPstrong/weak,physical262144/tails/RESTORE,current9849decode preservation '
                'or ordinary teardown.16Kis diagnostic. Original ordinary path and both prefill strategies remain.',
            retention='Reuse the three existing generations and required shared records. Keep useful24samples,'
                'work/placement/source identities and fit errors. Retire successful raw captures and closed campaign modes '
                'after verification. Select/retire a superseded generation before a fourth native build.'))
    for i,read in enumerate(READS):
        root=ROOT/str(i);root.mkdir(mode=0o700)
        for part in ('tmp','cache'):(root/part).mkdir(mode=0o700)
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        plan=dict(index=i,read=read,heads=1,kernel='eager',upload='eager',bank=384,window=10,relocate=0,
            trace_progress=i<2,root=str(root),port=port)
        core.write(root/'config.json',configuration(row,plan));core.write(root/'plan.json',
            dict(binary=row['binary'],native_validation_probe=row['native_validation_probe']))
        plan.update(command=command(plan),generated={p:sha(root/p) for p in ('config.json','plan.json')},
            environment_sha256=comparison.fingerprint(environment(row,plan)));row['plans'].append(plan)
    save(row);print('Prepared same-image five-length curve plus independent56K;commit before host/native.',flush=True)


if __name__=='__main__':
    require(sys.executable==PYTHON and len(sys.argv)==2 and (sys.argv[1] in ('prepare','host')
        or sys.argv[1].isdigit() and 0<=int(sys.argv[1])<len(READS)),'pinned Python:prepare|host|0..7')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);comparison.build.guards.no_model_owners([])
        if sys.argv[1]=='prepare':prepare()
        elif sys.argv[1]=='host':comparison.host(context())
        else:comparison.run(int(sys.argv[1]),context())
