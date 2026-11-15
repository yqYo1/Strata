#!/usr/bin/env python3
"""Prospective isolated U2 and separately gated combination model comparisons; Main only.

Owned benchmark/qualification readers, repository LICENSE. This isolates upload
run-copy policy with eager scheduling fixed;no new queue,math or decode tuning.
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
import fullfit_upload_device as device
import coherent_prefill_bench as speed

build=device.build
core=speed.core
comparison=core.comparison
P,B,RESULT=build.P,build.B,build.RESULT
MODE=sys.argv[1] if len(sys.argv)>1 else ''
require_mode=MODE in ['unit','combined']
if not require_mode:raise RuntimeError('unit|combined prepare|host|0..13')
ROOT=B/('prefill-coalesced-'+MODE+'-v1')
KEY='prefill_coalesced_'+MODE+'_v1'
REFERENCE='eager' if MODE=='unit' else 'jit'
THIS='tools/sycl/fullfit_upload_bench.py'
RUNS=3
CASES=[(32769,REFERENCE,True),(32769,'coalesced',True)]
for pair in range(3):
    for length in (32769,65537):
        arms=(REFERENCE,'coalesced') if (pair+(length==65537))%2==0 else ('coalesced',REFERENCE)
        CASES.extend((length,arm,False) for arm in arms)
ORDER=[(length,1) for length,_,_ in CASES]
require,sha,write=build.require,build.sha,build.write


def compact(row):
    return core.compact(row)


def save(row):
    write(ROOT/'record.json',row);data=json.loads(RESULT.read_text());data[KEY]=compact(row);write(RESULT,data)


def admitted():
    row=json.loads((device.ROOT/'record.json').read_text());device.verify(row)
    require(row['stage']=='closed' and row['passed'] and row['complete'] and row['ownership_closed']
        and not row['active'] and len(row['results'])==4 and all(v['passed'] and v['complete']
            and v['native_exit_code']==0 and v['fresh_faults']==[] for v in row['results']),
        'four fresh same-build upload safety stages required')
    return row


def environment(row,plan):
    env=speed.context().environment(row,plan)
    env['STRATA_SYCL_PREFILL_FULLFIT_UPLOAD']=plan['upload']
    if plan['trace_progress']:env['STRATA_TRACE']='1'
    return env


def current(row):
    require(build.ur_validation.committed(RESULT,(KEY,))==compact(row),'commit preceding model state')
    require(build.source.pin(P,list(row['sources']))['source_sha256']==row['sources'],'frozen frontend/controls changed')
    safety=admitted()
    require(row['binary']==safety['binaries']['strata'] and row['device_evidence_sha256']==build.original.fingerprint(device.compact(safety))
        and row['boot']==safety['boot'] and row['driver']==safety['driver'],'candidate/safety/platform changed')
    manifest=row['runtime_manifest'];require(sha(manifest['path'])==manifest['sha256'],'runtime manifest changed')
    files=json.loads(Path(manifest['path']).read_text())
    require(all(sha(p)==h for p,h in files.items()),'runtime provider changed')
    require(sha(row['template']['path'])==row['template']['sha256'],'pinned bench template changed')
    contract,python=core.frontend.python_manifest()
    require(contract==row['python_contract'] and python['selected']==row['python_selected'],'host Python providers changed')
    comparison.shared.models.model_pins(row['configuration'],row['model_assets'],full=False)
    require(comparison.build.supervision.artifact_bytes(ROOT)<=row['whole_bytes'],'whole comparison output cap')
    return files,python


def eager_route(text,plan):
    matches=re.findall(r'^strata prefill fullfit upload: policy=(eager|coalesced) layer=(\d+) first_row=(\d+) '
        r'logical_experts=(\d+) accepted_copy_commands=(\d+) requested_H2D_bytes=(\d+) '
        r'source_stride=(\d+) bank_stride=(\d+) '
        r'publication=accepted_same_queue native_completion=unchecked_until_outer_drain$',text,re.M)
    require(text.count('strata prefill fullfit upload:')==len(matches),'malformed/unknown upload route')
    require(all(v[0]==plan['upload'] for v in matches),'actual upload policy differs')
    records=[tuple(map(int,v[1:])) for v in matches]
    if plan['upload']=='jit':require(not records,'JIT arm unexpectedly used eager upload')
    else:
        require(len(records)==24 and len({v[0] for v in records})==24 and all(0<=l<48 and first==0
            and 0<n<=512 and 0<commands<=n and size==n*bank and source==bank
            and bank in [1510400,1971200,1715200,1868800]
            and (commands==n if plan['upload']=='eager' else commands<=min(n,513-n))
            for l,first,n,commands,size,source,bank in records),'actual24full-fit upload layers/stride/copy conservation differs')
        if plan['upload']=='coalesced':require(sum(v[3] for v in records)<sum(v[2] for v in records),'run copies did not reduce actual API commands')
    return dict(policy=plan['upload'],fullfit_layers=len(records),logical_experts=sum(v[2] for v in records),
        accepted_copy_commands=sum(v[3] for v in records),requested_fullfit_H2D_bytes=sum(v[4] for v in records),
        actual_equal_source_bank_strides_verified=len(records),native_completion='outer checked bank drain',
        physical_bus_traffic_or_overlap_measured=False)


def route_controls():
    def body(policy):
        return '\n'.join('strata prefill fullfit upload: policy=%s layer=%d first_row=0 logical_experts=400 accepted_copy_commands=%d requested_H2D_bytes=604160000 source_stride=1510400 bank_stride=1510400 publication=accepted_same_queue native_completion=unchecked_until_outer_drain'%(policy,l,400 if policy=='eager' else 64) for l in range(24))
    for arm in ['eager','coalesced','jit']:eager_route('' if arm=='jit' else body(arm),{'upload':arm})
    text=body('coalesced')
    bad=[text.replace('source_stride=1510400','source_stride=1510656'),text.replace('accepted_copy_commands=64','accepted_copy_commands=400'),text.replace('first_row=0','first_row=1'),text.replace('604160000','604159999'),text.replace('layer=23','layer=22'),text.replace('policy=coalesced','policy=eager')]
    for value in bad:
        try:eager_route(value,{'upload':'coalesced'})
        except RuntimeError:continue
        raise RuntimeError('invalid actual upload route accepted')
    return dict(valid=3,refused=len(bad))


def request_route(text,plan):
    out=speed.context().request_route(text,plan);out['fullfit_upload']=eager_route(text,plan)
    return out


def native_context(row,plan):
    require((plan['read'],plan['upload'],plan['trace_progress'])==CASES[plan['index']],'unplanned arm')
    return {k:row[k] for k in ['binary','native_source_commit','native_validation_probe','boot']}


def decision(processes):
    require(len(processes)==len(CASES) and all(v['passed'] and v['complete'] and v['ownership_closed']
        and v['native_exit_code']==0 and not v['fresh_faults'] and len(v['samples'])==RUNS
        for v in processes),'complete ordered model cohort required')
    points=[]
    for length in (32769,65537):
        selected=[v for v in processes[2:] if v['read']==length];require(len(selected)==6,'three independent pairs required')
        ratios=[];arms={k:[] for k in [REFERENCE,'coalesced']}
        for i in range(0,6,2):
            pair={v['kernel']:v for v in selected[i:i+2]};require(set(pair)==set(arms),'paired arm order differs')
            r,c=pair[REFERENCE],pair['coalesced'];ratios.append(dict(
                warmed_prompt_ms=c['subsequent_median_ms']/r['subsequent_median_ms'],
                warmed_client_TTFT=statistics.median(s['ttft_s'] for s in c['samples'][1:])/
                    statistics.median(s['ttft_s'] for s in r['samples'][1:]),
                first_prompt_ms=c['samples'][0]['prompt_ms']/r['samples'][0]['prompt_ms']))
            for arm,v in pair.items():arms[arm].append(dict(first_PP=v['samples'][0]['prompt_tps'],
                first_TTFT_s=v['samples'][0]['ttft_s'],warmed_PP=length*1000/v['subsequent_median_ms'],
                warmed_prompt_ms=v['subsequent_median_ms'],warmed_TTFT_s=statistics.median(s['ttft_s'] for s in v['samples'][1:])))
        points.append(dict(read=length,individual_processes=arms,paired_time_ratios=ratios,
            warmed_geomean_time_ratio=math.exp(statistics.mean(math.log(v['warmed_prompt_ms']) for v in ratios)),
            reproducible_warmed_gain=all(v['warmed_prompt_ms']<1 for v in ratios),
            warmed_TTFT_median_preserved=statistics.median(v['warmed_client_TTFT'] for v in ratios)<=1,
            coalesced_all_warmed_processes_at_least1000=all(v['warmed_PP']>=1000 for v in arms['coalesced'])))
    return dict(unit=MODE,reference=REFERENCE,candidate='coalesced',same_candidate_ELF=True,points=points,
        mechanism_screen_passed=all(v['reproducible_warmed_gain'] and v['warmed_TTFT_median_preserved'] for v in points),
        inference_adopted=False,sustained_decode_qualified=False,physical262144_RESTORE_qualified=False,
        whole_goal_complete=False,U2coalescing_measured=True)


def context():
    return SimpleNamespace(ROOT=ROOT,KEY=KEY,ORDER=ORDER,RUNS=RUNS,current=current,environment=environment,
        native_context=native_context,request_route=request_route,check_sample=speed.context().check_sample,
        decision=decision,compact=compact,save=save)


def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()),'fresh comparison only;no retry')
    controls=route_controls()
    if MODE=='combined':
        previous=build.ur_validation.committed(RESULT,('prefill_coalesced_unit_v1',))
        require(previous['stage']=='closed' and previous['complete'] and previous['passed'] and previous['decision']['mechanism_screen_passed'],'U2 individual/coalesced performance gate must pass before combined/JIT trial')
    safety=admitted();n=json.loads((build.ROOT/'record.json').read_text())
    old=json.loads((B/'coherent-prefill-speed-v1/record.json').read_text())
    config=copy.deepcopy(old['configuration']);config.update(exe=str(P/'tools/sycl/engine_once.py'),cwd=str(P))
    require('--serve' not in config['args'],'wrapper config requires serve flag owned by frontend')
    contract,python=core.frontend.python_manifest();env=copy.deepcopy(old['environment_template'])
    env={k:v.replace(str(B/'coherent-prefill-speed-v1'),str(ROOT)) for k,v in env.items()}
    env['PATH']=str(Path(python['selected']['path']).parent)+':/usr/bin:/bin'
    sources=core.pins(build.source.git(P,'rev-parse','HEAD').decode().strip())
    comparison.shared.models.model_pins(config,old['model_assets'],full=True)
    ROOT.mkdir(mode=0o700)
    row=dict(stage='prepared',route_controls=controls,comparison_mode=MODE,reference_arm=REFERENCE,source_commit=build.source.git(P,'rev-parse','HEAD').decode().strip(),sources=sources,
        binary=n['binaries']['strata'],native_source_commit=n['source_commit'],native_base_commit=n['native_base_commit'],
        native_source_identity=n['native_source_identity'],native_overlays=n['overlays'],
        device_evidence_sha256=build.original.fingerprint(device.compact(safety)),
        native_validation_probe=safety['native_validation_probe'],boot=safety['boot'],driver=safety['driver'],
        configuration=config,environment_template=env,model_assets=old['model_assets'],template=old['template'],
        python_contract=contract,python_selected=python['selected'],runtime_manifest=safety['runtime_manifest'],
        plans=[],processes=[],request_hashes={},host_passed=False,passed=False,complete=False,active=False,ownership_closed=True,
        whole_bytes=1024<<20,budgets=dict(wall=1800,cpu=28800,rss_cap=104<<30,text_cap=16<<20,file_cap=32<<20,total_cap=64<<20),
        plan=dict(unit='U2only:eager individual versus eager coalesced,then a separately admitted combined/JIT cohort. Same bytes,queue,math,placement and decode. Descriptor producer/header,caller/publication and oracle form the smallest coherent selection unit.',
            order=CASES,repetitions='First2processes are stage-progress parameter-validated32K reference/coalesced diagnostics. Then12clean processes,three independent matched pairs at32Kand64K,three fresh thinking-on24output bench.sh requests each. First-use and warmed2samples separate. Commit every closed process.',
            acceptance='Current frontend host probes;exact native/source/recipe/ELF/SDK/provider/Python/model/workload identities. Actual24full-fit layers with observed equal source/bank strides and logical-expert/payload conservation;individual copies==experts,coalesced copies<=min(n,513-n);JIT has none;unchanged actual fast GU/QSA/residual/bank and MTP restoration route. Normal SSE,exact fresh input andzero reuse,clientTTFT,quiet validation,no fresh fault,normal0/reaped empty owners. No speed-based exclusions.',
            mechanism='At each length,all3paired warmed prompt-time ratios<1and median paired warmedTTFT ratio<=1. No arbitrary minimum gain. Preserve all original samples/outcomes if a screen fails;report first-use separately. This is a prospective U2/combination screen,not route adoption.',
            limits='24outputs do not qualify decode. Sustained>=1Kstrong/weak MTP,agent/task/numerical and fullphysical262144/tails/RESTORE remain open. No hardware saturation or independent-engine overlap inferred. Combined/JIT can start only after the separately preserved U2eager/coalesced screen passes.',
            retention='Immediate U2decision;keep individual samples and compact actualroute/conservation. Retire successful logs/caches after closure. Existing two phase refs plus bounded coherent U2candidate;no extra generation.'))
    for i,(read,upload,trace) in enumerate(CASES):
        root=ROOT/str(i);root.mkdir(mode=0o700)
        for part in ['tmp','cache']:(root/part).mkdir(mode=0o700)
        cfg=copy.deepcopy(config);cfg['log']=str(root/'engine.stderr');write(root/'config.json',cfg)
        write(root/'plan.json',dict(binary=row['binary'],native_validation_probe=row['native_validation_probe']))
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        plan=dict(index=i,read=read,heads=1,kernel=upload,upload=upload,trace_progress=trace,root=str(root),
            generated={name:sha(root/name) for name in ['config.json','plan.json']},
            command=['/bin/sh',str(P/'bench.sh'),'--config',str(root/'config.json'),'--port',str(port),
                '--prefill-exact-tokens',str(read),'--think','--runs',str(RUNS),'--profile-request-gates',
                '--workload-id','native_gdn_20261018','--fail-fast-engine-error','--ready-timeout','600','--out',str(root/'bench')])
        plan['environment_sha256']=comparison.fingerprint(environment(row,plan));row['plans'].append(plan)
    save(row);print('Prepared isolated ' + MODE + ' model diagnostics and matched32K/64K pairs;commit before host/native.',flush=True)


if __name__=='__main__':
    require(sys.executable==speed.model.PYTHON,'use qualified benchmark Python')
    require(len(sys.argv)==3 and (sys.argv[2] in ['prepare','host'] or sys.argv[2].isdigit() and int(sys.argv[2])<len(CASES)),
        'prepare|host|0..13')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);build.guards.no_model_owners([])
        if sys.argv[2]=='prepare':prepare()
        elif sys.argv[2]=='host':comparison.host(context())
        else:comparison.run(int(sys.argv[2]),context())
