#!/usr/bin/env python3
"""Prospective U1 same-image JIT/eager model comparison; Main only.

Owned benchmark/qualification readers, repository LICENSE. This isolates upload
position, not copy coalescing, another queue, changed math or decode tuning.
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
import eager_fullfit_device as device
import coherent_prefill_bench as speed

build=device.build
core=speed.core
comparison=core.comparison
P,B,RESULT=build.P,build.B,build.RESULT
ROOT,KEY=B/'prefill-eager-model-v1','prefill_eager_model_v1'
THIS='tools/sycl/eager_fullfit_bench.py'
RUNS=3
CASES=[(32769,'jit',True),(32769,'eager',True)]
for pair in range(3):
    for length in (32769,65537):
        arms=('jit','eager') if (pair+(length==65537))%2==0 else ('eager','jit')
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
        and not row['active'] and len(row['results'])==2 and all(v['passed'] and v['complete']
            and v['native_exit_code']==0 and v['fresh_faults']==[] for v in row['results']),
        'two fresh same-build upload safety stages required')
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
    matches=re.findall(r'^strata prefill fullfit upload: policy=eager layer=(\d+) first_row=(\d+) '
        r'logical_experts=(\d+) accepted_copy_commands=(\d+) requested_H2D_bytes=(\d+) '
        r'publication=accepted_same_queue native_completion=unchecked_until_outer_drain$',text,re.M)
    records=[tuple(map(int,v)) for v in matches]
    require(text.count('strata prefill fullfit upload:')==len(records),'malformed/unknown upload route')
    if plan['upload']=='jit':require(not records,'JIT arm unexpectedly used eager upload')
    else:
        require(len(records)==24 and len({v[0] for v in records})==24 and all(0<=l<48 and first==0
            and 0<n<=512 and commands==n and size>0 and size%n==0 and
            size//n in [1510400,1971200,1715200,1868800] for l,first,n,commands,size in records),
            'actual24full-fit eager layers/individual copies differ')
    return dict(policy=plan['upload'],fullfit_layers=len(records),accepted_copy_commands=sum(v[3] for v in records),
        requested_fullfit_H2D_bytes=sum(v[4] for v in records),native_completion='outer checked bank drain',
        physical_bus_traffic_or_overlap_measured=False)


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
        ratios=[];arms={k:[] for k in ['jit','eager']}
        for i in range(0,6,2):
            pair={v['kernel']:v for v in selected[i:i+2]};require(set(pair)==set(arms),'paired arm order differs')
            r,c=pair['jit'],pair['eager'];ratios.append(dict(
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
            eager_all_warmed_processes_at_least1000=all(v['warmed_PP']>=1000 for v in arms['eager'])))
    return dict(unit='U1eager individual upload vs JIT individual upload;same candidate ELF',points=points,
        mechanism_screen_passed=all(v['reproducible_warmed_gain'] and v['warmed_TTFT_median_preserved'] for v in points),
        inference_adopted=False,sustained_decode_qualified=False,physical262144_RESTORE_qualified=False,
        whole_goal_complete=False,U2coalescing_measured=False)


def context():
    return SimpleNamespace(ROOT=ROOT,KEY=KEY,ORDER=ORDER,RUNS=RUNS,current=current,environment=environment,
        native_context=native_context,request_route=request_route,check_sample=speed.context().check_sample,
        decision=decision,compact=compact,save=save)


def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()),'fresh comparison only;no retry')
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
    row=dict(stage='prepared',source_commit=build.source.git(P,'rev-parse','HEAD').decode().strip(),sources=sources,
        binary=n['binaries']['strata'],native_source_commit=n['source_commit'],native_base_commit=n['native_base_commit'],
        native_source_identity=n['native_source_identity'],native_overlays=n['overlays'],
        device_evidence_sha256=build.original.fingerprint(device.compact(safety)),
        native_validation_probe=safety['native_validation_probe'],boot=safety['boot'],driver=safety['driver'],
        configuration=config,environment_template=env,model_assets=old['model_assets'],template=old['template'],
        python_contract=contract,python_selected=python['selected'],runtime_manifest=safety['runtime_manifest'],
        plans=[],processes=[],request_hashes={},host_passed=False,passed=False,complete=False,active=False,ownership_closed=True,
        whole_bytes=1024<<20,budgets=dict(wall=1800,cpu=28800,rss_cap=104<<30,text_cap=16<<20,file_cap=32<<20,total_cap=64<<20),
        plan=dict(unit='U1only:individual uploads earlier in the same in-order compute queue on empty full-fit512private banks. No coalescing,newqueue,math,placement or decode change.',
            order=CASES,repetitions='First2processes are traced parameter-validated32K JIT/eager diagnostics. Then12clean processes,three independent matched pairs at32Kand64K,three fresh thinking-on24output bench.sh requests each. First-use and warmed2samples separate. Commit every closed process.',
            acceptance='Current frontend host probes;exact native/source/recipe/ELF/SDK/provider/Python/model/workload identities. Actual24eager layers and individual-copy/byte conservation;JIT has none;unchanged actual fast GU/QSA/residual/bank and MTP restoration route. Normal SSE,exact fresh input andzero reuse,clientTTFT,quiet validation,no fresh fault,normal0/reaped empty owners. No speed-based exclusions.',
            mechanism='At each length,all3paired warmed prompt-time ratios<1and median paired warmedTTFT ratio<=1. No arbitrary minimum gain. Preserve all original samples/outcomes if a screen fails;report first-use separately. This is a U1screen,not route adoption.',
            limits='24outputs do not qualify decode. Sustained>=1Kstrong/weak MTP,agent/task/numerical and fullphysical262144/tails/RESTORE remain open. No hardware saturation or independent-engine overlap inferred. U2requires its own new committed unit comparison.',
            retention='Immediate U1decision;keep individual samples and compact actualroute/conservation. Retire successful logs/caches after closure. Existing two phase refs plus bounded coherent U1candidate;no extra generation.'))
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
    save(row);print('Prepared isolated U1model diagnostics and matched32K/64K pairs;commit before host/native.',flush=True)


if __name__=='__main__':
    require(sys.executable==speed.model.PYTHON,'use qualified benchmark Python')
    require(len(sys.argv)==2 and (sys.argv[1] in ['prepare','host'] or sys.argv[1].isdigit() and int(sys.argv[1])<len(CASES)),
        'prepare|host|0..13')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);build.guards.no_model_owners([])
        if sys.argv[1]=='prepare':prepare()
        elif sys.argv[1]=='host':comparison.host(context())
        else:comparison.run(int(sys.argv[1]),context())
