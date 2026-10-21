#!/usr/bin/env python3
"""Main-owned IQ2 pair model pilot: fresh bank gate, official warmups and two long cases."""
import argparse
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import re
import select
import shutil
import socket
import sys
import time

sys.dont_write_bytecode=True
import cpu_capacity as assets
import cpu_iq2 as component
import cpu_census_engine as census_run
import cpu_work_census as census
import generation_compare as comparison
import generation_engine as gen
import measurement_guards as guards
import measurement_owner as supervision
import query_profile as q
from check_expert_bank import check_packed_cases

P,B,A=gen.P,gen.B,gen.A
ROOT=B/'iq2-pair-model-v1'
MODEL=ROOT/'model'
RESULT=A/'iq2-pair-model-current.json'
REFERENCE=A/'cpu-work-census-current.json'
WORKLOADS=('regular-code','varied-prose')


def write(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def qualification():
    full,count=q.current_build()
    pool=gen.committed(component.RESULT)
    guards.require(pool['stage']=='closed' and pool['passed'] and pool['complete'] and pool['ownership_closed'] and
        not pool['active'] and pool['eligible_down_formats']==[20,42] and
        all(d['performance_gate_passed']for d in pool['decisions']) and
        pool['plan']['environment']['STRATA_IQ_SINGLE_EXACT']=='0' and
        component.base.source_pin(P,pool['plan']['source']['revision'],component.PREFIXES)==pool['plan']['source'],
        'same native CPU source and actual-baseline independent format gates required')
    reference=gen.committed(REFERENCE)
    guards.require(reference['stage']=='closed' and reference['passed'] and reference['complete'] and
        reference['ownership_closed'] and not reference['active'] and
        reference['environment']['STRATA_IQ_SINGLE_EXACT']=='0','closed actual model baseline required')
    return full,dict(original_inputs_recovered=count,build_receipt_sha256=gen.common.digest(q.BUILD_RECORD),
        shared_component_sha256=gen.common.digest(q.COMPONENT),pool_result_sha256=gen.common.digest(component.RESULT),
        original_model_sha256=gen.common.digest(REFERENCE)),reference


def plan():
    return dict(question='Does the checked IQ2_S NT1 producer actually execute and conserve target CPU work in the real long-generation engine?',
        order='Fresh same-build packed-smoke then packed-validate. One official bench.sh process:four warmups,regular-code,varied-prose.',
        constants='Same closed model/config/profile:131072 resident INT8 context,1256 expert slots,8192 prefill,5 workers+host,pcie0,adapt0,spec4,suffix0,caches0,greedy,no token graph/no coupled draft,Q8 MTP. SINGLE_EXACT=0,PAIR=1,AUDIT=1 fixed before exec.',
        gates='Match committed source,all native/fixture bytes,compiler,model/runtime/config and generated inputs. Before each long case verify the owned native executable/environment,the four warmups and prior requests,batched draft cells,IQ2 audit framing/deltas and exact target-census row conservation,no errors or fresh faults. Each long case >=32768 fresh input,1024..16384 generated tokens including thinking,>=512 answer characters,six physically pinned CPU workers/host with actual IQ2 NT1 pair work and complete stream/journal/native exit. Stop on first failure; no retries.',
        interpretation='Parameter validation,stage/census/timing and existing CPU callback counters are charged observers. No clean speed,observer overhead,model speedup,task/agent quality,physical262144/session/RESTORE or adoption qualification. Keep strong/weak names separate from observed MTP acceptance.',
        retention='Keep two samples,compact per-format/NT work,actual pair rows,source/fault/owner/route results and decision. Retire successful logs,JIT and duplicated state after verification; no full API trace or tensor capture.')


def current(record,full_models=False):
    full,proof,reference=qualification()
    guards.require(record['binary']==full['binary'] and record['build_source_commit']==full['source_commit'] and
        record['qualification']==proof and record['plan']==plan(),'IQ2 model qualification/plan drift')
    pins=gen.benchmark_pins(record['source_commit'])
    guards.require(len(pins)==record['source_count'] and assets.fingerprint(pins)==record['benchmark_fingerprint'] and
        gen.common.digest(gen.SETTINGS)==record['settings_sha256'],'IQ2 model controller/settings drift')
    for path,sha in record['generated_inputs'].items():
        guards.require(gen.common.digest(path)==sha,'IQ2 generated model input drift')
    guards.require(assets.machine_identity()==record['machine_identity'] and
        assets.model_pins(record['config'],record['model_assets'],full=full_models)==record['model_assets'],
        'IQ2 model/platform drift')
    gen.runtime_assets_unchanged(record)
    return full


def prepare():
    guards.require(not ROOT.exists() and not RESULT.exists(),'single-use IQ2 model root/result; never retry')
    full,proof,reference=qualification()
    revision=gen.git('rev-parse','HEAD').decode().strip();pins=gen.benchmark_pins(revision)
    models=assets.model_pins(reference['config'],reference['model_assets'],full=True)
    ROOT.mkdir(mode=0o700);MODEL.mkdir()
    for name in ('cache','tmp'):(MODEL/name).mkdir()
    config=copy.deepcopy(reference['config']);old_root=str(Path(config['log']).parent)
    config['args'][config['args'].index('--expert-profile')+1]=str(MODEL/'expert-profile.bin')
    config.update(log=str(MODEL/'engine.stderr'),env={})
    env={k:v.replace(old_root,str(MODEL))for k,v in reference['environment'].items()}
    env.update(STRATA_IQ_SINGLE_EXACT='0',STRATA_IQ2_PAIR='1',STRATA_IQ2_PAIR_AUDIT='1',
        STRATA_CPU_WORK_CENSUS='1',STRATA_DECODE_TIMING='1')
    shutil.copyfile(P/'data/expert-profile.bin',MODEL/'expert-profile.bin')
    write(MODEL/'config.json',config);write(MODEL/'plan.json',dict(binary=full['binary'],native_validation_probe=None))
    with socket.socket()as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    record=dict(mode='engine-compare',stage='prepared',source_commit=revision,build_source_commit=full['source_commit'],
        binary=full['binary'],qualification=proof,plan=plan(),config=config,environment=env,native_validation_probe=None,
        runtime_asset_delta=reference['runtime_asset_delta'],model_assets=models,
        generated_inputs={str(MODEL/n):gen.common.digest(MODEL/n)for n in ('config.json','plan.json','expert-profile.bin')},
        command=['/bin/sh',str(P/'bench.sh'),'--config',str(MODEL/'config.json'),'--port',str(port),
            '--long-generation-only','--think','--runs','1','--profile-request-gates','--workload-id',comparison.WORKLOAD_ID,
            '--fail-fast-engine-error','--ready-timeout','600','--out',str(MODEL/'bench')],
        source_count=len(pins),benchmark_fingerprint=assets.fingerprint(pins),machine_identity=assets.machine_identity(),
        settings_sha256=gen.common.digest(gen.SETTINGS),complete=False,passed=False,active=False,ownership_closed=True,
        clean_speed_qualified=False,hardware_saturation_qualified=False,model_adoption_qualified=False,
        budgets=dict(wall=3600,cpu=108000,rss_cap=96<<30,text_cap=64<<20,file_cap=64<<20,total_cap=64<<20))
    current(record);write(ROOT/'frozen-plan.json',record);write(RESULT,record)
    print('Prepared fresh IQ2 bank gate and one observed model pilot; commit before qualification.')


def work_evidence(text,samples,record):
    timing=gen.decode_timing_evidence(text,samples)
    bindings=[dict(prompt_tokens=s['prompt_tokens'],windows=t['windows'],rows=t['work']['target_rows'],
        cpu_jobs=t['work']['cpu_unique_jobs'],cpu_entries=t['work']['cpu_entries'],
        pool_ms=t['ms_per_window']['CPU_pool_run']*t['windows'])for s,t in zip(samples,timing)]
    args=record['config']['args'];formats=census.manifest_formats(Path(args[args.index('--pack')+1])/'native_experts.txt')
    work=census.extract(text,bindings,formats,census.expected_membership(MODEL/'expert-profile.bin'))
    paired=gen.iq2_pair_evidence(text,samples,1,component.base.machine()['physical_cpus'])
    for audit,observed in zip(paired,work):
        guards.require(audit['row_ops']==640*observed['cpu_entries'] and
            audit['nt1_rows']==640*sum(b['jobs_NT'][0]for b in observed['format_bins']if b['gu']==22),
            'IQ2 callback rows do not conserve actual target census')
    return paired,work,timing


def execute(action):
    record=gen.committed(RESULT);full=current(record)
    frozen=json.loads((ROOT/'frozen-plan.json').read_text())
    mutable={'stage','bank','requests','warmups','routes','iq2_dispatch','census','decode_timing','native_identity',
        'effective_environment_verified','native_exit_code','complete','passed','active','ownership_closed',
        'fresh_faults','fault_probe_successful','boot_unchanged','error','error_excerpt','fault_probe_error'}
    guards.require({k:v for k,v in record.items()if k not in mutable}==
        {k:v for k,v in frozen.items()if k not in mutable},'frozen IQ2 model plan changed')
    bank_run=action=='qualify'
    guards.require(record['stage']==('prepared'if bank_run else'model_prepared') and not record['active'] and
        (bank_run or record['bank']['passed']),'wrong IQ2 model stage; no retries')
    root=ROOT/'bank'if bank_run else MODEL
    guards.require(not(root/'record.json').exists(),'existing IQ2 model attempt')
    if bank_run:
        root.mkdir();(root/'tmp').mkdir();(root/'cache').mkdir()
    boot,cursor=census_run.start_fault_scope();own=None;row=dict(passed=False,complete=False)
    def persist():
        row['active']=bool(own and own.active is not None)
        write(root/'record.json',dict(row,commands=own.commands if own else []))
    def interact(io,leader):
        for i,name in enumerate(WORKLOADS):
            for stage in ('begin','end'):
                while True:
                    line=io.line(3300)
                    if line.startswith('STRATA_PROFILE_GATE '):break
                guards.require(json.loads(line.removeprefix('STRATA_PROFILE_GATE '))==
                    dict(stage=stage,phase='long-generation/'+name,run=0),'IQ2 official request gate differs')
                if stage=='begin':
                    native=json.loads((MODEL/'engine-start.json').read_text())['native'];identity=supervision.identity(native['pid'])
                    guards.require(identity and identity['sid']==leader['pid'] and identity['start_ticks']==native['start_ticks'] and
                        Path('/proc',str(native['pid']),'exe').resolve()==Path(record['binary']['path']).resolve(),'IQ2 native owner differs')
                    actual=dict(x.split(b'=',1)for x in Path('/proc',str(native['pid']),'environ').read_bytes().split(b'\0')if b'='in x)
                    guards.require(all(actual.get(k.encode())==v.encode()for k,v in record['environment'].items()) and
                        b'LD_PRELOAD'not in actual,'IQ2 native immutable environment differs')
                    path,=(MODEL/'bench').glob('*/requests.jsonl');events=[json.loads(s)for s in path.read_text().splitlines()]
                    guards.require(len(events)==10+2*i and events[-1]['kind']=='start' and
                        events[-1]['phase']=='long-generation/'+name,'IQ2 warmup/prior journal extent differs')
                    gen.warmup_reconciliation(events[:-1],WORKLOADS[:i])
                    samples=[e['sample']for e in events[:-1]if e['kind']=='result'];text=(MODEL/'engine.stderr').read_text(errors='replace')
                    guards.require(all(gen.sample_eligible(s)for s in samples[4:]),'prior IQ2 long request ineligible')
                    gen.request_evidence(text,samples);work_evidence(text,samples,record)
                    guards.require(not re.search(comparison.FAILURE,text),'IQ2 small/prior native check failed')
                    fault=census_run.fault_evidence(cursor,boot)
                    guards.require(fault['boot_unchanged'] and not fault['fresh_faults'],'fresh IQ2 fault before long request')
                    row.update(native_identity=native,effective_environment_verified=True);persist()
                io.send(b'continue\n')
        io.proc.stdin.close();deadline=time.monotonic()+180
        while not io.exit_watch.poll(0)or not io.eof:
            io.poll();guards.require(time.monotonic()<deadline,'IQ2 benchmark teardown budget')
            if select.select([io.proc.stdout],[],[],.05)[0]:io.read();io.pending.clear()
        io.exit(1)
    own=supervision.Owner(root/'owned',persist)
    try:
        if bank_run:
            env=dict(gen.committed(gen.SETTINGS)['environment'],TMPDIR=str(root/'tmp'),XDG_CACHE_HOME=str(root/'cache'));row['checks']=[]
            for mode in ('packed-smoke','packed-validate'):
                if mode.endswith('validate'):
                    env=dict(env,UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION');env.pop('UR_LOG_TRACING',None)
                entry,out,err=own.run(mode,[full['fixtures']['sycl_expert_bank_test']['path'],'--'+mode],
                    env,P,wall=180,cpu=600,rss_cap=8<<30,text_cap=64<<20,file_cap=64<<20,total_cap=64<<20)
                guards.require(supervision.closed(entry)and entry['exit_code']==0 and
                    not re.search(comparison.FAILURE,err.read_text()),'IQ2 packed bank failed')
                row['checks'].append(dict(mode=mode,cases=check_packed_cases(out.read_text(),mode),passed=True,
                    fixture=full['fixtures']['sycl_expert_bank_test'],exit_code=0))
        else:
            entry,_,_=own.run('bench',record['command'],record['environment'],P,interaction=interact,**record['budgets'])
            paths=list((MODEL/'bench').glob('*/results.json'))
            if paths:row['requests']=gen.collect_samples(json.loads(paths[0].read_text())['results'].get('long_generation',{}))
            guards.require(supervision.closed(entry)and entry['exit_code']==0 and len(paths)==1,'IQ2 official benchmark failed')
            result=json.loads(paths[0].read_text());samples,journal,events=comparison.official(MODEL,result)
            native=gen.native_end(MODEL,record,record,boot,entry['pid']);text=(MODEL/'engine.stderr').read_text(errors='replace')
            guards.require(not re.search(comparison.FAILURE,text),'IQ2 native error after long generation')
            paired,work,timing=work_evidence(text,[e['sample']for e in events if e['kind']=='result'],record)
            row.update(requests=samples,warmups=journal,routes=comparison.native_routes(text,events,1),
                iq2_dispatch=paired[4:],census=work[4:],decode_timing=timing[4:],native_exit_code=native['exit_code'])
        current(record,full_models=not bank_run);row.update(passed=True,complete=True)
    except BaseException as error:
        row['error']=str(error)
        if(MODEL/'engine.stderr').exists():row['error_excerpt']=(MODEL/'engine.stderr').read_text(errors='replace')[-4096:]
    finally:
        row['ownership_closed']=own.active is None and all(supervision.closed(c)for c in own.commands)
        try:row.update(census_run.fault_evidence(cursor,boot))
        except Exception as error:row.update(fault_probe_error=str(error),passed=False)
        row['passed']=bool(row['passed']and row['ownership_closed']and row.get('boot_unchanged') and
            row.get('fault_probe_successful')and row.get('fresh_faults')==[])
        persist()
        if bank_run:
            record.update(bank=row,stage='model_prepared'if row['passed']else'closed',
                active=row['active'],ownership_closed=row['ownership_closed'])
            if not row['passed']:record['error']=row.get('error',row.get('fault_probe_error'))
        else:record.update(row,stage='closed')
        write(RESULT,record)
    print(json.dumps({k:row.get(k)for k in ('passed','complete','ownership_closed','error')}))
    return 0 if row['passed']else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','qualify','run'));args=parser.parse_args()
    with (B/'owned-v0141-measurement.lock').open('a+')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([]);guards.driver_guard()
        return prepare()if args.action=='prepare'else execute(args.action)


if __name__=='__main__':sys.exit(main())
