#!/usr/bin/env python3
"""Main-owned clean117K comparison against the retained decode-preserving build.

No native rebuild. This screen cannot qualify sustained decode or adoption.
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
import canonical_model as check

base,gen,assets,guards,owner_lib=check.base,check.gen,check.assets,check.guards,check.owner_lib
P,B,A=check.P,check.B,check.A
ROOT=B/'canonical-restore-clean-v1'
HOST=B/'canonical-restore-clean-host-v1'
RESULT=A/'canonical-restore-speed-current.json'
ORDER=(0,1,1,0,0,1)
PAIRS=((0,1),(3,2),(4,5))
TEST='tests/core/test_canonical_speed.py'

def write(path,value):check.write(path,value)

def plan():
    return dict(order=list(ORDER),pairs=[list(p)for p in PAIRS],requests_per_process=4,
        arms=['retained6e40-whole','current362-chunk64'],
        question='Does the complete prefill-only change improve117K with the original1256-slot decode placement preserved?',
        scope='Package comparison,not an isolated copy experiment. Retained6e40 is the current-decode reference;current362 also contains default-off transition instrumentation. The preceding same-native checked off/on and standalone copy results isolate the proposed mechanism separately.',
        constants='Official bench.sh --prefill-only --prefill-tokens98304 --think --runs4;one short warmup then four117489-fresh-input requests/process,<=24 thinking output. Same model/runtime/profile,1256 expert slots,131072 resident INT8 context,8192 prefill,5 workers+host,packed1,IQ2pair1,Q8 MTP,pcie0,adapt0,spec4,suffix0,caches0,no token graph/no coupled draft. Only native build and immutable prefill canonical selector differ.',
        decision='Keep first-use separate. Repetition unit is one independent native process,using its three subsequent-request median. Candidate-faster screen requires every paired ratio>1 and geometric mean>=1.0005;otherwise no improvement established. This0.05% screen is prospective,below the approximately0.14% copy-only expectation. The1000 target requires all three candidate process medians>=1000. No allowed prefill/decode regression tradeoff and no adoption from this screen.',
        stopping='Stop on the first source/model/runtime/config/route/journal/owner/fault/budget failure;never retry a root,replace a sample or stop on speed. Commit each closed process before the next.',
        models='Full hashes before this cohort and after process5;device/inode/size/mtime/ctime guards before and after every process. No digest I/O during native execution.',
        observers='No parameter-validation/API trace/snapshot/cache-payload/transition/census/decode clocks. Both arms receive canonical audit0;actual optimized copy execution was qualified in the preceding checked same-build admission. Default route/placement records remain and are checked per request.',
        limits='Short output does not qualify sustained decode,agent/task quality or physical262144/session/RESTORE. Next compare repeated thinking-on long generation against the same retained6e40 reference for both observed MTP classes;any decode impact is a separation defect. No assumption of identical generated text.',
        retention='Keep all individual samples,route/owner/fault/source/config and paired decisions. Retire successful normal logs,JIT and duplicate receipts after committed closure. Retain two native generations only through this named prefill/long-generation comparison.')

def environment(setup,root,arm):
    guards.require(type(arm)is int and arm in(0,1),'unknown clean arm')
    return dict(base.environment(setup,root,False),STRATA_SYCL_PREFILL_CANONICAL_CHUNK64=str(arm),
        STRATA_SYCL_PREFILL_CANONICAL_RESTORE_AUDIT='0',STRATA_PREFILL_TRANSITION_PROFILE='0',
        STRATA_PREFILL_LAYER_MAJOR_PROFILE='0')

def summarize(rows):
    guards.require(len(rows)==6 and [r['arm']for r in rows]==list(ORDER)and
        all(r['passed']and r['ownership_closed']and r['fresh_faults']==[]for r in rows),
        'complete successful ordered cohort required')
    guards.require(len({(r['native_identity']['pid'],r['native_identity']['start_ticks'])for r in rows})==6,
        'independent native processes required')
    medians=[];first=[]
    for row in rows:
        samples=row['requests'];guards.require(len(samples)==4,'four fresh requests required')
        values=[]
        for i,s in enumerate(samples):
            guards.require(s['run']==i and all(s.get(k)is True for k in
                ('ok','eligible','stream_complete','thinking_requested','reasoning_observed'))and
                s['actual_prompt_tokens']==s['actual_read_tokens']==117489 and s['actual_reused_tokens']==0 and
                type(s['generated_tokens'])is int and 1<=s['generated_tokens']<=24 and
                s['prompt_tps_source']=='engine'and type(s['prompt_ms'])in(int,float)and
                math.isfinite(s['prompt_ms'])and s['prompt_ms']>0,'ineligible clean sample')
            values.append(117489000/s['prompt_ms'])
        first.append(values[0]);medians.append(statistics.median(values[1:]))
    for a,b in PAIRS:
        guards.require([r['request_sha256']for r in rows[a]['requests']]==
            [r['request_sha256']for r in rows[b]['requests']],'paired request changed')
    ratios=[medians[b]/medians[a]for a,b in PAIRS];geo=math.exp(statistics.mean(map(math.log,ratios)))
    return dict(first_use_tok_s=first,subsequent_median_tok_s=medians,pair_ratios=ratios,
        ratio_geomean=geo,candidate_faster_screen=all(x>1 for x in ratios)and geo>=1.0005,
        candidate1000_each_process=all(medians[i]>=1000 for i,arm in enumerate(ORDER)if arm),
        sustained_decode_qualified=False,adoption_qualified=False)

def qualification(full_models=False):
    admitted=gen.committed(check.RESULT)
    guards.require(admitted['stage']=='closed'and len(admitted['processes'])==2 and
        [r['arm']for r in admitted['processes']]==[0,1]and all(r['passed']and r['complete']and
        r['ownership_closed']and r['fresh_faults']==[]for r in admitted['processes']),
        'closed same-native actual-model admission required')
    full=check.current(admitted,full_models)
    retained=gen.committed(check.gate.q.BUILD)['retained_validation_reference']
    recovered=base.generation.recover_retained(retained)
    guards.require(retained['source_commit']=='6e40afc1beddd059ee518617f6c1efce1da14aca'and
        full['source_commit']==admitted['native_source'],'wrong reference/candidate')
    return admitted,full,retained,dict(admission_sha256=gen.common.digest(check.RESULT),retained=recovered)

def current(record,full_models=False):
    admitted,full,retained,proof=qualification(full_models)
    guards.require(record['qualification']==proof and record['plan']==plan()and
        record['binaries']==[retained['binary'],full['binary']]and
        assets.fingerprint(gen.benchmark_pins(record['source_commit']))==record['benchmark_fingerprint']and
        gen.common.digest(P/TEST)==record['host']['test_sha256'],'clean source/qualification drift')
    setup=base.setup_reader.pilot()
    for i,row in enumerate(record['plans']):
        root=Path(row['root']);arm=ORDER[i]
        guards.require(row['cohort']==i and row['arm']==arm and row['environment']==environment(setup,root,arm)and
            json.loads((root/'config.json').read_text())==base.setup_reader.config(setup,root)and
            json.loads((root/'plan.json').read_text())==dict(binary=record['binaries'][arm],native_validation_probe=None),
            'decode/config/immutable environment drift')
        for name,sha in row['inputs'].items():guards.require(gen.common.digest(name)==sha,'clean generated input drift')
    return admitted

def host():
    guards.require(not HOST.exists()and not RESULT.exists(),'fresh single-use clean host check only')
    source=gen.git('rev-parse','HEAD').decode().strip();pins=gen.benchmark_pins(source)
    test_hash=hashlib.sha256(gen.git('show',source+':'+TEST)).hexdigest()
    guards.require(gen.common.digest(P/TEST)==test_hash,'uncommitted clean tests')
    HOST.mkdir(mode=0o700);row=dict(source_commit=source,passed=False,tests=6,GPU_executed=False,
        controller_sha256=pins['tools/sycl/canonical_speed.py'],test_sha256=test_hash);owner=None
    def persist():write(HOST/'record.json',dict(row,commands=owner.commands if owner else []))
    owner=owner_lib.Owner(HOST/'owned',persist)
    try:
        entry,_,stderr=owner.run('host',['/usr/bin/python3','-B',TEST],
            dict(PATH='/home/yayoi/.nix-profile/bin:/usr/bin:/bin',LANG='C.UTF-8',PYTHONDONTWRITEBYTECODE='1'),P,
            wall=30,cpu=30,rss_cap=1<<30,text_cap=1<<20,file_cap=1<<20,total_cap=4<<20)
        text=stderr.read_text();guards.require(owner_lib.closed(entry)and entry['exit_code']==0 and
            'Ran 6 tests'in text and text.rstrip().endswith('OK'),'clean host tests failed');row['passed']=True
    except BaseException as error:row['error']=str(error)
    finally:
        row['ownership_closed']=owner.active is None and all(owner_lib.closed(e)for e in owner.commands)
        row['passed']=row['passed']and row['ownership_closed'];persist()
        write(RESULT,dict(stage='host-checked',host=row,plan=plan()))
    print(json.dumps(row));return 0 if row['passed']else 1

def prepare():
    previous=gen.committed(RESULT);host=previous['host']
    guards.require(previous['stage']=='host-checked'and previous['plan']==plan()and host['passed']and
        host['ownership_closed']and host['controller_sha256']==gen.common.digest(Path(__file__))and
        host['test_sha256']==gen.common.digest(P/TEST)and not ROOT.exists(),'fresh qualified clean plan required')
    admitted,full,retained,proof=qualification(True);setup=base.setup_reader.pilot()
    source=gen.git('rev-parse','HEAD').decode().strip();ROOT.mkdir(mode=0o700);plans=[]
    binaries=[retained['binary'],full['binary']]
    for i,arm in enumerate(ORDER):
        root=ROOT/str(i);root.mkdir()
        for n in('cache','tmp'):(root/n).mkdir()
        shutil.copyfile(P/'data/expert-profile.bin',root/'expert-profile.bin')
        original=setup['config']['args'];original=original[original.index('--expert-profile')+1]
        guards.require(gen.common.digest(root/'expert-profile.bin')==setup['generated_inputs'][original],'profile changed')
        write(root/'config.json',base.setup_reader.config(setup,root))
        write(root/'plan.json',dict(binary=binaries[arm],native_validation_probe=None))
        with socket.socket()as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
        plans.append(dict(cohort=i,arm=arm,root=str(root),environment=environment(setup,root,arm),
            inputs={str(root/n):gen.common.digest(root/n)for n in('config.json','plan.json','expert-profile.bin')},
            command=['/bin/sh',str(P/'bench.sh'),'--config',str(root/'config.json'),'--port',str(port),
                '--prefill-only','--prefill-tokens','98304','--think','--runs','4','--workload-id','native_gdn_20261018',
                '--fail-fast-engine-error','--ready-timeout','600','--out',str(root/'bench')]))
    record=dict(stage='prepared',source_commit=source,host=host,plan=plan(),qualification=proof,
        benchmark_fingerprint=assets.fingerprint(gen.benchmark_pins(source)),binaries=binaries,
        plans=plans,processes=[],expected_request_sha256=base.gen.committed(base.HISTORY)['processes'][0]['requests'],
        budgets=dict(wall_s=2400,cpu_s=72000,rss_bytes=96<<30,text_bytes=64<<20,total_bytes=64<<20))
    record['expected_request_sha256']=[s['request_sha256']for s in record['expected_request_sha256'][:4]]
    current(record);write(ROOT/'frozen.json',record);write(RESULT,record)

def run():
    record=gen.committed(RESULT);current(record)
    frozen=json.loads((ROOT/'frozen.json').read_text());mutable={'stage','processes','analysis'}
    guards.require({k:v for k,v in record.items()if k not in mutable}=={k:v for k,v in frozen.items()if k not in mutable}and
        record['stage']in('prepared','collecting')and len(record['processes'])<6 and
        all(r['passed']for r in record['processes']),'closed/changed clean cohort;no retry')
    i=len(record['processes']);selected=record['plans'][i];root=Path(selected['root'])
    guards.require(not(root/'record.json').exists(),'existing attempt;inspect owner')
    boot,cursor=check.gate.faults.start_fault_scope();row=dict(cohort=i,arm=selected['arm'],passed=False,complete=False);owner=None
    def persist():write(root/'record.json',dict(row,active=bool(owner and owner.active is not None),commands=owner.commands if owner else []))
    owner=owner_lib.Owner(root/'owned',persist)
    try:
        entry,_,_=owner.run('bench',selected['command'],selected['environment'],P,wall=2400,cpu=72000,
            rss_cap=96<<30,text_cap=64<<20,file_cap=64<<20,total_cap=64<<20)
        paths=list((root/'bench').glob('*/results.json'));data=None
        if len(paths)==1:
            try:data=json.loads(paths[0].read_text());row['requests']=data.get('results',{}).get('prefill_only',{}).get('requests',[])
            except json.JSONDecodeError:row['result_json_incomplete']=True
        if not row.get('requests'):
            row['requests']=[];journals=list((root/'bench').glob('*/requests.jsonl'))
            if len(journals)==1:
                for line in journals[0].read_text().splitlines():
                    try:event=json.loads(line)
                    except json.JSONDecodeError:row['partial_journal_json']=True;break
                    if event.get('kind')=='result'and event.get('phase')=='prefill-only':row['requests'].append(event['sample'])
                row['requests_from_unvalidated_partial_journal']=True
        guards.require(owner_lib.closed(entry)and entry['exit_code']==0 and data is not None,'official benchmark/owner failed')
        samples,journal=base.history.official(root,data,4);row.update(requests=samples,journal=journal)
        native=gen.native_end(root,dict(binary=record['binaries'][selected['arm']],native_validation_probe=None),
            dict(config=json.loads((root/'config.json').read_text())),boot,entry['pid'])
        text=(root/'engine.stderr').read_text();assets.verify_pool(text,5)
        row['route']=base.history.native_evidence(text,samples,1,False,base.history.admitted_loan(),native['native']['pid'],None)
        guards.require(not any(s in text for s in('strata prefill transition:','strata canonical restore:')),'clean observer emitted')
        guards.require([s['request_sha256']for s in samples]==record['expected_request_sha256'],'request bytes changed')
        current(record,full_models=i==5)
        row.update(native_identity=native['native'],passed=True,complete=True,
            prompt_tok_s=[117489000/s['prompt_ms']for s in samples])
    except BaseException as error:
        row['error']=str(error)
        if(root/'engine.stderr').exists():row['error_excerpt']=(root/'engine.stderr').read_text(errors='replace')[-4096:]
    finally:
        row['ownership_closed']=owner.active is None and all(owner_lib.closed(e)for e in owner.commands)
        row.update(fault_probe_successful=False,fresh_faults=None,boot_unchanged=False)
        try:row.update(check.gate.faults.fault_evidence(cursor,boot))
        except Exception as error:row['fault_probe_error']=str(error)
        row['passed']=bool(row['passed']and row['ownership_closed']and row['fault_probe_successful']and row['fresh_faults']==[]and row['boot_unchanged']);persist()
        record['processes'].append(row);record['stage']='collecting'if row['passed']and i<5 else'closed'
        if row['passed']and i==5:
            try:record['analysis']=summarize(record['processes'])
            except Exception as error:
                row.update(passed=False,error='cohort summary: '+str(error));persist()
        write(RESULT,record)
    print(json.dumps({k:row.get(k)for k in('cohort','arm','passed','complete','ownership_closed','error')}));return 0 if row['passed']else 1

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('host','prepare','run'));args=parser.parse_args()
    with(B/'owned-v0141-measurement.lock').open('a+')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([]);guards.driver_guard()
        return {'host':host,'prepare':prepare,'run':run}[args.action]()
if __name__=='__main__':sys.exit(main())
