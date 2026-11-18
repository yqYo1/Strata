#!/usr/bin/env python3
"""Owned64Ksource/full-fit revalidation; Main controls all native execution.

Repository LICENSE. Uses owned admission/run readers from570f6903e and the
U1contract from5f1a95453. No XeStrata source. No new native generation. The
closed window campaign is historical evidence,not permission to adopt a route.
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
import prefill_workspace_admission as admission

capacity,core=admission.capacity,admission.core
comparison=core.comparison
P,B,RESULT=capacity.P,capacity.B,capacity.RESULT
require,sha,signature=capacity.require,capacity.sha,capacity.signature
ROOT,KEY=B/'prefill-64k-revalidation-v1','prefill_64k_revalidation_v1'
PYTHON=capacity.reference.model.PYTHON
READ,RUNS=65537,3
# Three traced admissions,then three balanced clean triples. Every question
# has both process orders;the shared JIT process is an explicit common control.
ARMS=('original','jit','eager','original','jit','eager',
      'eager','jit','original','jit','original','eager')
ORDER=[(READ,1) for _ in ARMS]
PAIRS={'coherence':((3,4),(8,7),(10,9)),
       'eager':((4,5),(7,6),(9,11)),
       'combined':((3,5),(8,6),(10,11))}
compact=core.compact


def selection(index):
    arm=ARMS[index]
    return dict(kernel=arm,native='original' if arm=='original' else 'candidate',
                upload='eager' if arm=='eager' else 'jit',bank=384,window=10,relocate=0)


def save(row):
    core.write(ROOT/'record.json',row)
    values=json.loads(RESULT.read_text());values[KEY]=compact(row);core.write(RESULT,values)


def preceding_evidence():
    key='prefill_window_combined_model_v1'
    row=admission.device.build.ur_validation.committed(RESULT,(key,))
    require(row['stage']=='closed' and row['passed'] and row['complete']
        and row['ownership_closed'] and not row['active'] and len(row['processes'])==9
        and row['decision']['unit_performance_eligible'] and not row['decision']['inference_adopted'],
        'close the preceding combined question;do not inherit model adoption')
    return dict(key=key,source_commit=row['source_commit'],source_inputs=row['source_inputs'],
                decision=row['decision'])


def environment(row,plan):
    env=comparison.environment(row,plan)
    env.update(STRATA_SYCL_PREFILL_EXPERT_BANK_SLOTS='384',STRATA_SYCL_PREFILL_MOE_WINDOW='10',
        STRATA_SYCL_PREFILL_MOE_RELOCATE_JOB_BUFFERS='0',STRATA_SYCL_PREFILL_FULLFIT_UPLOAD=plan['upload'],
        STRATA_SYCL_PREFILL_MOE_CENSUS='0',STRATA_PREFILL_LAYER_MAJOR_PROFILE='0')
    if plan['trace_progress']:env['STRATA_TRACE']='1'
    require(env['STRATA_SYCL_PREFILL_SAME_QUEUE']=='1'
        and env['STRATA_SYCL_PREFILL_EXPERT_UPLOAD_STAGING']=='0'
        and env['STRATA_SYCL_PREFILL_EXPERT_BANK_PACKED']=='1'
        and env['STRATA_SYCL_CACHE_SCRATCH_LEASE']==env['STRATA_SYCL_PREFILL_MIXED_SCRATCH']=='1'
        and env['STRATA_SYCL_PREFILL_DOWN_SPLIT_KERNEL']=='original'
        and env['STRATA_SYCL_PREFILL_MOE_SPLIT_ACCUMULATOR']=='1','fixed qualified placement/queue required')
    return env


def native_context(row,plan):
    return dict(row['native_contexts'][plan['native']],
                native_validation_probe=row['native_validation_probe'],boot=row['boot'])


def configuration(row,plan):
    return dict(row['configuration'],log=str(Path(plan['root'])/'engine.stderr'))


def command(plan):
    root=Path(plan['root'])
    return ['/bin/sh',str(P/'bench.sh'),'--config',str(root/'config.json'),'--port',str(plan['port']),
        '--prefill-exact-tokens',str(READ),'--think','--runs',str(RUNS),'--profile-request-gates',
        '--workload-id','native_gdn_20261018','--fail-fast-engine-error','--ready-timeout','600',
        '--out',str(root/'bench')]


def committed_sources(revision):
    sources=core.pins(revision)
    blobs=dict(core.source.device.common.source.blobs(P,revision,list(sources)))
    require(set(blobs)==set(sources) and all(hashlib.sha256(blobs[p]).hexdigest()==h
        and sha(P/p)==h for p,h in sources.items()),'commit exact source/controller inputs before preparation')
    return sources


def validate_case(case,index):
    require(type(index) is int and 0<=index<len(ARMS) and case['index']==index
        and case['kernel']==ARMS[index] and case['read']==READ and case['heads']==1
        and case['clean_speed']==(index>=3) and case['passed'] and case['complete']
        and case['ownership_closed'] and case['native_exit_code']==0 and not case['fresh_faults']
        and case['boot_unchanged'] and len(case['routes'])==len(case['samples'])==RUNS,
        'ordered closed exact-arm receipts required')
    require(all(r['fullfit_upload']['policy']==selection(index)['upload'] for r in case['routes']),
        'actual upload selection differs from the source arm')
    require(all(s['actual_read_tokens']==READ and s['actual_reused_tokens']==0
        and s['generated_tokens']==24 and s['request_sha256'] for s in case['samples'])
        and math.isfinite(case['subsequent_median_ms']) and case['subsequent_median_ms']>10000
        and case['subsequent_median_ms']==statistics.median(s['prompt_ms'] for s in case['samples'][1:]),
        'actual sample lengths/warmed reduction differ')


def work(route):
    value=copy.deepcopy(route);value.pop('fullfit_upload')
    value['layout'].pop('free_gpu_bytes')
    return value


def validate_receipts(processes):
    require(len(processes)<=len(ARMS),'too many process receipts')
    for i,case in enumerate(processes):validate_case(case,i)
    require(len({(c['native_identity']['pid'],c['native_identity']['start_ticks']) for c in processes})==len(processes),
        'fresh independent native identities required')
    if processes:
        baseline=processes[0]
        for case in processes[1:]:
            require([s['request_sha256'] for s in case['samples']]==[s['request_sha256'] for s in baseline['samples']]
                and [work(r) for r in case['routes']]==[work(r) for r in baseline['routes']],
                'fixed source/full-fit question changed request,math/job/map/upload or placement work')


def after_execution(outcome,row,plan):
    # Fail through the existing supervised per-case path before final reduction.
    native=native_context(row,plan)
    require(outcome['binary']==native['binary']
        and outcome['native_source_commit']==native['native_source_commit'],'closed case source/binary differs')
    validate_receipts([*row['processes'],outcome])


def current(row):
    require(admission.device.build.ur_validation.committed(RESULT,(KEY,))==compact(row)
        and committed_sources(row['source_commit'])==row['sources']
        and all(sha(P/p)==h for p,h in row['policies'].items()),'frozen committed64Kquestion required')
    original,proof,prior=capacity.basis(row['original_native_proof']);candidate,gate=admission.device_gate()
    contexts=dict(original=dict(binary=original['binaries']['strata'],native_source_commit=capacity.original.SOURCE),
        candidate=dict(binary=candidate['binaries']['strata'],native_source_commit=candidate['source_commit']))
    require(row['component_gate']==gate and row['preceding_evidence']==preceding_evidence()
        and row['reference_evidence_sha256']==signature(compact(prior)) and row['native_contexts']==contexts
        and row['candidate_composition_sha256']==signature(candidate['source_files'])
        and all(row[k]==prior[k] for k in ('configuration','environment_template','model_assets','template',
            'boot','driver','native_validation_probe','python_contract','python_selected')),
        'native generation,composition,model,decode policy or runtime changed')
    contract,python=core.frontend.python_manifest()
    require(contract==row['python_contract'] and python['selected']==row['python_selected']
        and sha(row['template']['path'])==row['template']['sha256']
        and comparison.build.supervision.artifact_bytes(ROOT)<=row['whole_bytes']
        and len(row['plans'])==len(ARMS),'frontend/template/output cap changed')
    for i,plan in enumerate(row['plans']):
        root=Path(plan['root'])
        require(plan['index']==i and plan['read']==READ and plan['heads']==1
            and all(plan[k]==v for k,v in selection(i).items()) and plan['trace_progress']==(i<3)
            and plan['command']==command(plan) and json.loads((root/'config.json').read_text())==configuration(row,plan)
            and json.loads((root/'plan.json').read_text())==dict(binary=native_context(row,plan)['binary'],
                native_validation_probe=row['native_validation_probe'])
            and all(sha(root/p)==h for p,h in plan['generated'].items())
            and comparison.fingerprint(environment(row,plan))==plan['environment_sha256'],'unplanned64Karm/input')
    validate_receipts(row['processes'])
    for case in row['processes']:
        receipt=json.loads((ROOT/str(case['index'])/'execution.json').read_text())
        require(not receipt['active'] and all(receipt.get(k)==v for k,v in case.items())
            and all(comparison.build.supervision.closed(v) for v in receipt['commands']),
            'original closed execution receipt changed')
        require(case['binary']==contexts[selection(case['index'])['native']]['binary']
            and case['native_source_commit']==contexts[selection(case['index'])['native']]['native_source_commit'],
            'case source/binary changed')
    return json.loads(Path(row['runtime_manifest']['path']).read_text()),python


def check_sample(sample,count,read,minimum_ms=10000):
    comparison.check_sample(sample,count,read,minimum_ms)
    require(read==READ and sample['generated_tokens']==24,'exact64Kprefill diagnostic only')


def decision(processes):
    require(len(processes)==len(ARMS),'complete12process/36request cohort required')
    validate_receipts(processes)
    units={}
    for name,pairs in PAIRS.items():
        times=[];ttfts=[];first=[]
        for ref,can in pairs:
            a,b=processes[ref],processes[can]
            times.append(b['subsequent_median_ms']/a['subsequent_median_ms'])
            ttfts.append(statistics.median(s['ttft_s'] for s in b['samples'][1:])/
                         statistics.median(s['ttft_s'] for s in a['samples'][1:]))
            first.append(b['samples'][0]['prompt_ms']/a['samples'][0]['prompt_ms'])
        gain=all(r<1 for r in times);ttft=statistics.median(ttfts)<=1
        units[name]=dict(indices=pairs,warmed_time_ratios=times,warmed_TTFT_ratios=ttfts,
            first_time_ratios=first,geometric_mean_time_ratio=math.exp(statistics.mean(map(math.log,times))),
            repeated_warmed_gain=gain,warmed_TTFT_median_preserved=ttft,performance_eligible=gain and ttft)
    arms={}
    for arm in ('original','jit','eager'):
        selected=[c for c in processes[3:] if c['kernel']==arm]
        times=[c['subsequent_median_ms'] for c in selected]
        arms[arm]=dict(warmed_process_ms=times,warmed_process_PP=[READ*1000/t for t in times],
            warmed_PP=READ*1000/statistics.median(times),
            warmed_client_TTFT_s=[statistics.median(s['ttft_s'] for s in c['samples'][1:]) for c in selected],
            first_samples=[c['samples'][0] for c in selected],
            all_warmed_processes_at_least1000=all(READ*1000/t>=1000 for t in times))
    return dict(units=units,arms=arms,inference_adopted=False,current9849decode_preserved=False,
        sustained_TG_task_agent_physical262144_RESTORE_qualified=False,ordinary_teardown_qualified=False,
        hardware_saturation_qualified=False,physical_scaling_law_qualified=False,
        rule='Keep every unit and first-use loss. Only matched unit gains permit further phase-specific qualification;'
             'neither a source observation nor1000PPalone adopts the model route. New work needs a new plan.')


def context():
    return SimpleNamespace(ROOT=ROOT,KEY=KEY,ORDER=ORDER,RUNS=RUNS,current=current,environment=environment,
        native_context=native_context,save=save,request_route=admission.request_route,
        check_sample=check_sample,after_execution=after_execution,decision=decision)


def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()),'fresh question only;no retry')
    preceding=preceding_evidence();original,proof,prior=capacity.basis();candidate,gate=admission.device_gate()
    revision=admission.device.build.source.git(P,'rev-parse','HEAD').decode().strip()
    sources=committed_sources(revision)
    comparison.shared.models.model_pins(prior['configuration'],prior['model_assets'],full=True)
    ROOT.mkdir(mode=0o700)
    row=dict(stage='prepared',source_commit=revision,sources=sources,
        policies={p:sha(P/p) for p in capacity.POLICIES},original_native_proof=proof,component_gate=gate,
        preceding_evidence=preceding,reference_evidence_sha256=signature(compact(prior)),
        candidate_native_base=candidate['native_base_commit'],candidate_native_overlays=candidate['overlays'],
        candidate_composition_sha256=signature(candidate['source_files']),
        native_contexts=dict(original=dict(binary=original['binaries']['strata'],native_source_commit=capacity.original.SOURCE),
            candidate=dict(binary=candidate['binaries']['strata'],native_source_commit=candidate['source_commit'])),
        **{k:copy.deepcopy(prior[k]) for k in ('configuration','environment_template','model_assets','template',
            'boot','driver','native_validation_probe','python_contract','python_selected','runtime_manifest')},
        plans=[],processes=[],request_hashes={},host_passed=False,passed=False,complete=False,active=False,
        ownership_closed=True,budgets=dict(wall=2400,cpu=20000,rss_cap=104<<30,text_cap=16<<20,
            file_cap=32<<20,total_cap=128<<20),whole_bytes=2048<<20,
        plan=dict(question='At64K/fixed384bank,is the coherent candidate as fast as the retained14508image,and '
            'does its independently selected U1eager upload improve its JITpath and the net source choice?',
            provenance='Main-owned14508reference plus existing3a195fexplicit15overlays;U1preserved from51ea99/f13b3. '
                'No new native build,XeStrata code,math,placement,chunk,decode policy or copy coalescing.',
            units='C0:original14508versus candidateJIT. U1:same candidateJITversus eager. Net:original versus '
                'eager. Same384bank/window10/64K;each unit has three paired independent processes. '
                'The declared common JITcontrols are shared across the paired questions,not pooled across timing periods.',
            order=ARMS,samples='Three traced64Kadmissions,one per arm,then nine clean processes in three balanced '
                'triples:original/JIT/eager,eager/JIT/original,JIT/original/eager. Each has3thinking-on24output '
                'requests. First separate;median subsequent2is the process unit. No concurrent builds,tests,GPUjobs,'
                'profilers,researchers or reviewers during clean timing. Commit every closed process before next.',
            acceptance='Exact committed source/recipe/ELFs/helpers/configs/providers/model hashes and owned identities. '
                'Actual65536BF16rowsGPU/48windows/384bank/nativeQSAand GU/Down. JITzero U1headers,eager24actual '
                'unique fullfit layers/individual-copy bytes and checked terminal bank drain. Same request/job/map/'
                'upload/payload work across all arms. Normal0/reapedempty,quiet validation,unchangedboot/driver,no freshfault.',
            performance='Keep all36samples and all three unit outcomes. Per unit,all3warmed time ratios<1and '
                'median paired warmedTTFT ratio<=1permit further phase-specific qualification;no minimum percentage. '
                'Keep first use/spread separate. A C0loss cannot be offset to adopt the ordinary path. U1mechanism '
                'and net source eligibility are distinct.1000PPrequires all3independent warmed process PP>=1000.',
            stopping='Stop first source,owner,clock,route,conservation,warning,fault or budget failure. No retry,'
                'hashrefresh,replacement,exclusion,criterion change or speed-based earlystop. Full15model hashes '
                'before preparation and after process11;identity checks throughout.',
            limits='64Kquestion only;not a five-length fit or timing-epoch comparison. No decode preservation,'
                'task/agent,longthinking1Kstrong/weak MTP,physical262144/tails/RESTORE,ordinaryteardown or saturation '
                'qualification. Existing fast ordinary/current9849decode and both prefill strategies remain.',
            retention='Reuse the three retained generations,all required same-build fixtures and source/recipe '
                'records only for this imminent question and phase qualification. Select/retire a superseded '
                'generation before any new native build. Keep useful samples/units/source/actualwork;retire raw '
                'captures and this campaign after its final consumer.'))
    for i in range(len(ARMS)):
        root=ROOT/str(i);root.mkdir(mode=0o700)
        for part in ('tmp','cache'):(root/part).mkdir(mode=0o700)
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        plan=dict(index=i,read=READ,heads=1,trace_progress=i<3,root=str(root),port=port,**selection(i))
        core.write(root/'config.json',configuration(row,plan));core.write(root/'plan.json',
            dict(binary=native_context(row,plan)['binary'],native_validation_probe=row['native_validation_probe']))
        plan.update(command=command(plan),generated={p:sha(root/p) for p in ('config.json','plan.json')},
            environment_sha256=comparison.fingerprint(environment(row,plan)));row['plans'].append(plan)
    save(row);print('Prepared64Kthree-arm source/U1question;commit before host/native.',flush=True)


if __name__=='__main__':
    require(sys.executable==PYTHON and len(sys.argv)==2 and (sys.argv[1] in ('prepare','host')
        or sys.argv[1].isdigit() and 0<=int(sys.argv[1])<len(ARMS)),'pinned Python:prepare|host|0..11')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);comparison.build.guards.no_model_owners([])
        if sys.argv[1]=='prepare':prepare()
        elif sys.argv[1]=='host':comparison.host(context())
        else:comparison.run(int(sys.argv[1]),context())
