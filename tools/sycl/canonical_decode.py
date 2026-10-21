#!/usr/bin/env python3
"""Main-owned repeated long decode after the fixed-placement prefill screen.

No native rebuild, output-equality requirement or permitted decode regression.
The clean screen does not qualify task quality or physical262144/session/RESTORE.
"""
import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import socket
import statistics
import sys
sys.dont_write_bytecode=True
import canonical_speed as speed

base,gen,assets,guards,owner_lib=speed.base,speed.gen,speed.assets,speed.guards,speed.owner_lib
gate=speed.check.gate
P,B,A=speed.P,speed.B,speed.A
ROOT=B/'canonical-restore-decode-v1'
HOST=B/'canonical-restore-decode-host-v1'
RESULT=A/'canonical-restore-decode-current.json'
ORDER=speed.ORDER
PAIRS=speed.PAIRS
WORKLOADS=base.generation.WORKLOADS
TEST='tests/core/test_canonical_decode.py'

def write(path,value):speed.write(path,value)

def plan():
    return dict(order=list(ORDER),pairs=[list(p)for p in PAIRS],workloads=list(WORKLOADS),
        question='Does the prefill-only package preserve current sustained decode in both observed MTP regimes?',
        execution='Six independent native processes,retained6e40/current362/current362/retained6e40/retained6e40/current362. Official bench.sh --long-generation-only --think --runs1,each with four official warmups then the two version4 long-generation workloads. Same fresh request bodies,>=32768 actual input,1024..16384 generated tokens including thinking,>=512 answer characters and>=256 offered drafts. No equal-output requirement.',
        constants='Same1256-slot decode placement,131072 resident INT8 context,8192 prefill,5 workers+host,packed1,IQ2pair1,SINGLE_EXACT0,Q8 MTP,pcie0,adapt0,spec4,suffix0,caches0,no token graph/no coupled draft,no F16 MTP batch. Only native package and pre-exec prefill canonical chunk selector differ.',
        decision='Repetition unit is one independent process per workload. Calculate exact achieved TG from actual completion tokens/engine decode_ms;retain printed TG separately. Conservative no-decline screen requires all three candidate/reference paired ratios>=1 for EACH workload. Any lower ratio leaves preservation unqualified;do not dismiss it as noise or permit a prefill/decode tradeoff. Confirm strong>=0.60 and weak<=0.40 acceptance in every sample of BOTH arms. This screen is not statistical proof of zero effect. TG70 requires every candidate workload sample>=70;passing preservation alone does not meet the goal.',
        admission='Require the closed successful six-process clean prefill screen and its prospective performance gate,actual-model checked off/on state restoration,and recoverable same-build engine/fixtures for both packages. Pin all benchmark helpers,generated configuration,model/runtime,machine and immutable environment before launch. Verify original inputs and exact owned native exit.',
        route='Bind all six native footers to the complete official journal. Require each long request to report full GPU BF16 residuals and mode10 with the correct row extent;verify1256-slot borrowed placement. Clean copy/MTP selections are source-verified through the pinned launch chain. Exact payload/map and dense restoration belong to the preceding checked model admission,not unlogged per-process measurements here.',
        models='Full model hashes before the cohort and after process5;device/inode/size/mtime/ctime guards before and after every process. No model digest I/O during native execution.',
        stopping='Stop on first source/model/runtime/configuration/journal/route/owner/fault/budget failure;never retry a root or replace a sample. Commit every closed outcome before the next process. Never stop early on speed or MTP acceptance.',
        limits='No inference adoption,task-quality,agent-session,physical262144 or session/RESTORE qualification. No hardware arithmetic/RAM/PCIe saturation claim from TG or stage clocks. A demonstrated decode impact is a separation defect to fix.',
        retention='Keep individual samples,source/configuration,actual route/journal/owner/fault outcomes and paired decisions. Retire successful raw output/JIT after committed closure;preserve attempt barriers while the cohort is active. Both native generations have this named immediate consumer;select and retire the extra only after comparison and provenance review.')

def environment(setup,root,arm):
    return speed.environment(setup,root,arm)

def summarize(rows):
    guards.require(len(rows)==6 and [r['arm']for r in rows]==list(ORDER)and
        all(r['passed']and r['complete']and r['ownership_closed']and r['fault_probe_successful']and
            r['boot_unchanged']and r['fresh_faults']==[]for r in rows),'complete successful ordered decode cohort required')
    guards.require(len({(r['native_identity']['pid'],r['native_identity']['start_ticks'])for r in rows})==6,
        'independent native decode processes required')
    for row in rows:
        samples=row['requests']
        guards.require(len(samples)==2 and [(s.get('workload'),s.get('run'),s.get('order'))for s in samples]==
            [(name,0,i)for i,name in enumerate(WORKLOADS)]and all(gen.sample_eligible(s)for s in samples),
            'ineligible or incomplete thinking long generation')
    for a,b in PAIRS:
        guards.require([s['request_sha256']for s in rows[a]['requests']]==
            [s['request_sha256']for s in rows[b]['requests']],'paired long request changed')
    workloads={}
    for j,name in enumerate(WORKLOADS):
        tps=[r['requests'][j]['generated_tokens']*1000/r['requests'][j]['decode_ms']for r in rows]
        ratios=[tps[b]/tps[a]for a,b in PAIRS]
        acceptance=[r['requests'][j]['acceptance_rate']for r in rows]
        classified=all(x>=.60 for x in acceptance)if j==0 else all(x<=.40 for x in acceptance)
        workloads[name]=dict(exact_tok_s=tps,printed_tok_s=[r['requests'][j]['decode_tps']for r in rows],
            pair_ratios=ratios,ratio_geomean=math.exp(statistics.mean(map(math.log,ratios))),
            no_decline_screen=all(x>=1 for x in ratios),acceptance=acceptance,
            observed_MTP_class_confirmed=classified,candidate70_each_process=all(tps[i]>=70 for i in(1,2,5)))
    return dict(workloads=workloads,MTP_pair_confirmed=all(w['observed_MTP_class_confirmed']for w in workloads.values()),
        decode_preservation_screen=all(w['no_decline_screen']and w['observed_MTP_class_confirmed']for w in workloads.values()),
        candidate70_each_workload_process=all(w['candidate70_each_process']for w in workloads.values()),
        zero_effect_statistically_proven=False,adoption_qualified=False)

def qualification(full_models=False):
    clean=gen.committed(speed.RESULT)
    guards.require(clean['stage']=='closed'and clean.get('analysis')==speed.summarize(clean['processes'])and
        clean['analysis']['candidate_faster_screen']and all(r['complete']and r['fault_probe_successful']and
        r['boot_unchanged']for r in clean['processes']),'closed eligible prefill screen required')
    admitted=gen.committed(speed.check.RESULT)
    guards.require(gen.common.digest(speed.check.RESULT)==clean['qualification']['admission_sha256']and
        admitted['stage']=='closed'and [r['arm']for r in admitted['processes']]==[0,1]and
        all(r['passed']and r['complete']and r['ownership_closed']and r['fresh_faults']==[]for r in admitted['processes']),
        'checked actual-model restoration proof changed')
    # This reader deliberately does not consume the closed cohorts' generated
    # configs/profiles. Their immutable identities remain in committed receipts.
    full=gate.current(admitted,full_models)
    retained=gen.committed(gate.q.BUILD)['retained_validation_reference']
    proof=base.generation.recover_retained(retained)
    guards.require(retained['source_commit']=='6e40afc1beddd059ee518617f6c1efce1da14aca'and
        full['source_commit']=='362c16aad19cb00dc158c03a279e6c28cc8cc7d3'and
        clean['binaries']==[retained['binary'],full['binary']]and proof==clean['qualification']['retained'],
        'wrong retained/current decode packages')
    return full,retained,dict(prefill_sha256=gen.common.digest(speed.RESULT),
        admission_sha256=gen.common.digest(speed.check.RESULT),retained=proof)

def current(record,full_models=False):
    full,retained,proof=qualification(full_models)
    guards.require(record['qualification']==proof and record['plan']==plan()and
        record['binaries']==[retained['binary'],full['binary']]and
        assets.fingerprint(gen.benchmark_pins(record['source_commit']))==record['benchmark_fingerprint']and
        gen.common.digest(P/TEST)==record['host']['test_sha256'],'decode source/qualification drift')
    setup=base.setup_reader.pilot()
    for i,row in enumerate(record['plans']):
        root=Path(row['root']);arm=ORDER[i]
        guards.require(row['cohort']==i and row['arm']==arm and row['environment']==environment(setup,root,arm)and
            json.loads((root/'config.json').read_text())==base.setup_reader.config(setup,root)and
            json.loads((root/'plan.json').read_text())==dict(binary=record['binaries'][arm],native_validation_probe=None),
            'decode configuration/immutable environment drift')
        for name,sha in row['inputs'].items():guards.require(gen.common.digest(name)==sha,'decode generated input drift')

def native_routes(text,events):
    samples=[e['sample']for e in events if e.get('kind')=='result']
    headers=list(re.finditer(r'^strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read[^\n]*, (\d+) generated[^\n]*$',text,re.M))
    guards.require(len(headers)==len(samples)==6,'native long-generation request extent differs')
    for h,s in zip(headers,samples):
        guards.require(tuple(map(int,h.groups()))==(s['prompt_tokens'],s['reused_tokens'],s['read_tokens'],s['generated_tokens'])and
            s['reused_tokens']==0,'native long-generation request binding differs')
    placement=base.history.cache_placement(text,1,base.history.admitted_loan());routes=[]
    for i in(4,5):
        sample=samples[i];cells=sample['read_tokens']-1;segment=text[headers[i-1].end():headers[i].start()]
        footer=re.fullmatch(r'strata serve: prompt \d+ tokens = \d+ reused \+ \d+ read in (\d+) ms '
            r'\([^)]+\), \d+ generated in (\d+) ms \([^)]+\), drafts accepted (\d+) of (\d+), \d+ checkpoints',headers[i].group())
        guards.require(footer is not None and all(type(sample.get(k))in(int,float)and math.isfinite(sample[k])and sample[k]>0
            for k in('prompt_ms','decode_ms')),'missing native long clocks')
        prompt_ms,decode_ms,accepted,offered=map(int,footer.groups())
        # DONE uses0.1ms;the stderr footer uses1ms. Both round the same clocks.
        guards.require(abs(prompt_ms-sample['prompt_ms'])<=.5500001 and
            abs(decode_ms-sample['decode_ms'])<=.5500001 and
            (accepted,offered)==(sample['drafts_accepted'],sample['drafts_offered']),
            'native/API long clocks or MTP counters differ')
        actual=re.findall(r'^strata prefill: layer-major candidate, tokens=(\d+) chunk=8192 residual_dtype=bf16 '
            r'residual_element_bytes=2 residual_total=(\d+) residual_gpu_rows=(\d+) residual_gpu=(\d+) '
            r'residual_host=0 residual_staging=0 ',segment,re.M)
        cap=min(cells,81920)
        guards.require(cells>=32767 and len(actual)==1 and tuple(map(int,actual[0]))==(cells,cells*20480,cells,cells*20480)and
            len(re.findall(r'^strata MoE window configuration: requested_mode=10 effective_mode=10 maximum_rows='
                +str(cap)+r' mixed_fp16_bytes=0$',segment,re.M))==1 and 'effective_mode=2 'not in segment,
            'actual long GPU/BF16/mode10 route differs')
        routes.append(dict(actual_GPU_BF16_rows=cells,reported_MoE_mode=10,reported_maximum_rows=cap,
            prompt_clock_difference_ms=abs(prompt_ms-sample['prompt_ms']),
            decode_clock_difference_ms=abs(decode_ms-sample['decode_ms']),rounding_bound_ms=.55))
    guards.require(not any(s in text for s in('strata trace:','strata prefill dense:','strata loan audit:',
        'strata prefill transition:','strata canonical restore:','strata decode timing:','strata decode cpu:',
        'strata MoE mixed owners:','strata serve: DRAFT_PREFILL','SYCL_EVENT_QUERY')),'clean decode observer emitted')
    return dict(placement=placement,requests=routes,restoration_proof='preceding checked actual-model admission;not per-process payload logging')

def host():
    guards.require(not HOST.exists()and not RESULT.exists(),'fresh single-use decode host check only')
    source=gen.git('rev-parse','HEAD').decode().strip();pins=gen.benchmark_pins(source)
    sha=hashlib.sha256(gen.git('show',source+':'+TEST)).hexdigest()
    guards.require(gen.common.digest(P/TEST)==sha,'uncommitted decode tests')
    HOST.mkdir(mode=0o700);row=dict(source_commit=source,passed=False,tests=7,GPU_executed=False,
        controller_sha256=pins['tools/sycl/canonical_decode.py'],test_sha256=sha);owner=None
    def persist():write(HOST/'record.json',dict(row,commands=owner.commands if owner else []))
    owner=owner_lib.Owner(HOST/'owned',persist)
    try:
        entry,_,err=owner.run('host',['/usr/bin/python3','-B',TEST],
            dict(PATH='/home/yayoi/.nix-profile/bin:/usr/bin:/bin',LANG='C.UTF-8',PYTHONDONTWRITEBYTECODE='1'),P,
            wall=30,cpu=30,rss_cap=1<<30,text_cap=1<<20,file_cap=1<<20,total_cap=4<<20)
        text=err.read_text();guards.require(owner_lib.closed(entry)and entry['exit_code']==0 and
            'Ran 7 tests'in text and text.rstrip().endswith('OK'),'decode host tests failed');row['passed']=True
    except BaseException as error:row['error']=str(error)
    finally:
        row['ownership_closed']=owner.active is None and all(owner_lib.closed(e)for e in owner.commands)
        row['passed']=row['passed']and row['ownership_closed'];persist();write(RESULT,dict(stage='host-checked',host=row,plan=plan()))
    print(json.dumps(row));return 0 if row['passed']else 1

def prepare():
    previous=gen.committed(RESULT);host=previous['host']
    guards.require(previous['stage']=='host-checked'and previous['plan']==plan()and host['passed']and
        host['ownership_closed']and host['controller_sha256']==gen.common.digest(Path(__file__))and
        host['test_sha256']==gen.common.digest(P/TEST)and not ROOT.exists(),'fresh qualified decode plan required')
    full,retained,proof=qualification(True);setup=base.setup_reader.pilot()
    source=gen.git('rev-parse','HEAD').decode().strip();ROOT.mkdir(mode=0o700);plans=[];ports=set()
    binaries=[retained['binary'],full['binary']]
    for i,arm in enumerate(ORDER):
        root=ROOT/str(i);root.mkdir()
        for n in('cache','tmp'):(root/n).mkdir()
        shutil.copyfile(P/'data/expert-profile.bin',root/'expert-profile.bin')
        flags=setup['config']['args'];original=flags[flags.index('--expert-profile')+1]
        guards.require(gen.common.digest(root/'expert-profile.bin')==setup['generated_inputs'][original],'decode profile changed')
        write(root/'config.json',base.setup_reader.config(setup,root))
        write(root/'plan.json',dict(binary=binaries[arm],native_validation_probe=None))
        with socket.socket()as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
        guards.require(port not in ports,'duplicate decode port');ports.add(port)
        plans.append(dict(cohort=i,arm=arm,root=str(root),environment=environment(setup,root,arm),
            inputs={str(root/n):gen.common.digest(root/n)for n in('config.json','plan.json','expert-profile.bin')},
            command=['/bin/sh',str(P/'bench.sh'),'--config',str(root/'config.json'),'--port',str(port),
                '--long-generation-only','--think','--runs','1','--workload-id',base.generation.WORKLOAD_ID,
                '--fail-fast-engine-error','--ready-timeout','600','--out',str(root/'bench')]))
    record=dict(stage='prepared',source_commit=source,host=host,plan=plan(),qualification=proof,
        benchmark_fingerprint=assets.fingerprint(gen.benchmark_pins(source)),binaries=binaries,plans=plans,processes=[],
        budgets=dict(wall_s=3600,cpu_s=72000,rss_bytes=96<<30,text_bytes=64<<20,total_bytes=64<<20))
    current(record);write(ROOT/'frozen.json',record);write(RESULT,record)

def run():
    record=gen.committed(RESULT);current(record)
    frozen=json.loads((ROOT/'frozen.json').read_text());mutable={'stage','processes','analysis'}
    guards.require({k:v for k,v in record.items()if k not in mutable}=={k:v for k,v in frozen.items()if k not in mutable}and
        record['stage']in('prepared','collecting')and len(record['processes'])<6 and
        all(r['passed']for r in record['processes']),'closed/changed decode cohort;no retry')
    i=len(record['processes']);selected=record['plans'][i];root=Path(selected['root'])
    guards.require(not(root/'record.json').exists(),'existing decode attempt;inspect owner')
    boot,cursor=gate.faults.start_fault_scope();row=dict(cohort=i,arm=selected['arm'],passed=False,complete=False);owner=None
    def persist():write(root/'record.json',dict(row,active=bool(owner and owner.active is not None),commands=owner.commands if owner else []))
    owner=owner_lib.Owner(root/'owned',persist)
    try:
        entry,_,_=owner.run('bench',selected['command'],selected['environment'],P,wall=3600,cpu=72000,
            rss_cap=96<<30,text_cap=64<<20,file_cap=64<<20,total_cap=64<<20)
        paths=list((root/'bench').glob('*/results.json'));data=None
        if len(paths)==1:
            try:
                data=json.loads(paths[0].read_text());row['requests']=gen.collect_samples(data.get('results',{}).get('long_generation',{}))
            except json.JSONDecodeError:row['result_json_incomplete']=True
        if not row.get('requests'):
            row['requests']=[];journals=list((root/'bench').glob('*/requests.jsonl'))
            if len(journals)==1:
                for line in journals[0].read_text().splitlines():
                    try:event=json.loads(line)
                    except json.JSONDecodeError:row['partial_journal_json']=True;break
                    if event.get('kind')=='result'and event.get('phase','').startswith('long-generation/'):
                        row['requests'].append(dict(event['sample'],workload=event['phase'].split('/',1)[1],run=event['run']))
                row['requests_from_unvalidated_partial_journal']=True
        guards.require(owner_lib.closed(entry)and entry['exit_code']==0 and data is not None,'official decode benchmark/owner failed')
        samples,journal,events=base.generation.official(root,data);row.update(requests=samples,journal=journal)
        native=gen.native_end(root,dict(binary=record['binaries'][selected['arm']],native_validation_probe=None),
            dict(config=json.loads((root/'config.json').read_text())),boot,entry['pid'])
        text=(root/'engine.stderr').read_text();assets.verify_pool(text,5)
        guards.require(not re.search(base.generation.FAILURE,text),'native decode error')
        row['route']=native_routes(text,events)
        if record['processes']:
            guards.require([s['request_sha256']for s in samples]==
                [s['request_sha256']for s in record['processes'][0]['requests']],'long decode request changed')
        current(record,full_models=i==5)
        row.update(native_identity=native['native'],passed=True,complete=True)
    except BaseException as error:
        row['error']=str(error)
        if(root/'engine.stderr').exists():row['error_excerpt']=(root/'engine.stderr').read_text(errors='replace')[-4096:]
    finally:
        row['ownership_closed']=owner.active is None and all(owner_lib.closed(e)for e in owner.commands)
        row.update(fault_probe_successful=False,fresh_faults=None,boot_unchanged=False)
        try:row.update(gate.faults.fault_evidence(cursor,boot))
        except Exception as error:row['fault_probe_error']=str(error)
        row['passed']=bool(row['passed']and row['ownership_closed']and row['fault_probe_successful']and row['fresh_faults']==[]and row['boot_unchanged']);persist()
        record['processes'].append(row);record['stage']='collecting'if row['passed']and i<5 else'closed'
        if row['passed']and i==5:
            try:record['analysis']=summarize(record['processes'])
            except Exception as error:row.update(passed=False,error='decode cohort summary: '+str(error));persist()
        write(RESULT,record)
    print(json.dumps({k:row.get(k)for k in('cohort','arm','passed','complete','ownership_closed','error')}));return 0 if row['passed']else 1

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('host','prepare','run'));args=parser.parse_args()
    with(B/'owned-v0141-measurement.lock').open('a+')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([]);guards.driver_guard()
        return {'host':host,'prepare':prepare,'run':run}[args.action]()
if __name__=='__main__':sys.exit(main())
