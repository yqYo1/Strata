#!/usr/bin/env python3
"""Main-owned checked117K admission before clean prefill/decode comparison."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shutil
import socket
import sys
sys.dont_write_bytecode=True
import canonical_restore as gate
import prefill_transition as observer

base=gate.base
gen,assets,guards,owner_lib=gate.gen,gate.assets,gate.guards,gate.owner_lib
P,B,A=gate.P,gate.B,gate.A
ROOT=B/'canonical-restore-model-check-v2'
RESULT=A/'canonical-restore-model-current.json'
BYTES=1538035200
PREVIOUS='9555a7bd2'

def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')

def environment(setup,root,arm):
    guards.require(type(arm)is int and arm in (0,1),'unknown canonical arm')
    return dict(base.environment(setup,root,True),STRATA_PREFILL_TRANSITION_PROFILE='1',
        STRATA_PREFILL_LAYER_MAJOR_PROFILE='1',STRATA_SYCL_PREFILL_CANONICAL_CHUNK64=str(arm),
        STRATA_SYCL_PREFILL_CANONICAL_RESTORE_AUDIT='1')

def coverage(text,arm):
    headers=list(re.finditer(r'^strata serve: prompt .*$',text,re.M))
    rows=list(re.finditer(r'^strata canonical restore: phase=bank_full bytes=(\d+) copies=(\d+) chunk=(\d+)$',text,re.M))
    guards.require(len(headers)==2 and len(rows)==1 and text.count('strata canonical restore:')==1 and
        headers[0].end()<rows[0].start()<headers[1].start(),'missing/extra/misplaced canonical restore')
    expected=(BYTES,23,67108864)if arm else(BYTES,1,0)
    guards.require(tuple(map(int,rows[0].groups()))==expected,'actual canonical bytes/copies/chunk differ')
    return dict(bytes=expected[0],copies=expected[1],chunk=expected[2],between_warmup_and_long_prompt_footer=True)

def compact_profiles(rows):
    clocks=('wall_s','thread_cpu_s','wait_wall_s','wait_cpu_s')
    result=[]
    for r in rows:
        s={k:r[k]for k in ('scope',*clocks,'other_wall_s','other_cpu_s')}
        s['parts']={name:{k:v for k,v in part.items()if k in clocks or k=='calls' or
            (k in ('items','logical_bytes')and v)}for name,part in r['parts'].items()}
        result.append(s)
    return result

def dense_memory(text):
    # Each completed request owns three observations. Warmup also retires and
    # restores readers, but keeps Full residency instead of entering the bank.
    headers=list(re.finditer(r'^strata serve: prompt .*$',text,re.M))
    guards.require(len(headers)==2,'dense observations need two completed requests')
    a=base.history.admission
    a.records(text,'strata prefill dense',6)
    warm=a.records(text[:headers[0].start()],'strata prefill dense',3)
    long=a.records(text[headers[0].end():headers[1].start()],'strata prefill dense',3)
    guards.require([r['phase']for _,r in warm]==['retired','cache-loan','restored'] and
        [r['phase']for _,r in long]==['retired','bank','restored'],
        'dense observations escaped request phases')
    return [{k:r[k]for k in ('phase','free_bytes','total_bytes','full_bytes','bank_bytes')}
        for _,r in long]

def current(record,full_models=False):
    full=gate.current(record,full_models)
    test='tests/core/test_canonical_model.py'
    guards.require(gen.common.digest(P/test)==record['host_contract']['test_sha256'] and
        hashlib.sha256(gen.git('show',record['source_commit']+':'+test)).hexdigest()==record['host_contract']['test_sha256'],
        'canonical host-contract test drift')
    proof=gen.committed(gate.RESULT)
    guards.require(gen.common.digest(gate.RESULT)==record['device_gate_sha256'] and proof['stage']=='closed' and
        proof['native_source']==record['native_source'] and proof['binary']==full['binary'] and
        proof['result']['passed'] and proof['result']['complete'] and proof['result']['ownership_closed'] and
        not proof['result']['fresh_faults'],'canonical device qualification drift')
    setup=base.setup_reader.pilot()
    for arm,row in enumerate(record['plans']):
        root=Path(row['root'])
        guards.require(row['arm']==arm and row['environment']==environment(setup,root,arm) and
            json.loads((root/'config.json').read_text())==base.setup_reader.config(setup,root),
            'current decode/configuration drift')
        for name,sha in row['inputs'].items():guards.require(gen.common.digest(name)==sha,'generated model input changed')
    return full

def host():
    record=gen.committed(base.RESULT);plan=record['canonical_model_host_plan']
    root=Path(plan['root']);guards.require(not root.exists(),'single-use host check only')
    source=gen.git('rev-parse','HEAD').decode().strip()
    for name,sha in plan['inputs'].items():
        guards.require(gen.common.digest(P/name)==sha and
            hashlib.sha256(gen.git('show',source+':'+name)).hexdigest()==sha,'host source drift')
    root.mkdir(mode=0o700);row=dict(source_commit=source,tests=7,passed=False,GPU_executed=False)
    owner=None
    def persist():write(root/'record.json',dict(row,commands=owner.commands if owner else []))
    owner=owner_lib.Owner(root/'owned',persist)
    try:
        entry,_,stderr=owner.run('host',plan['command'],plan['environment'],P,wall=30,cpu=30,
            rss_cap=1<<30,text_cap=1<<20,file_cap=1<<20,total_cap=4<<20)
        output=stderr.read_text()
        guards.require(owner_lib.closed(entry)and entry['exit_code']==0 and
            'Ran 7 tests' in output and output.rstrip().endswith('OK'),'host tests/owner failed')
        row.update(passed=True,controller_sha256=plan['inputs']['tools/sycl/canonical_model.py'],
            test_sha256=plan['inputs']['tests/core/test_canonical_model.py'])
    except BaseException as error:row['error']=str(error)
    finally:
        row['ownership_closed']=owner.active is None and all(owner_lib.closed(e)for e in owner.commands)
        row['passed']=row['passed']and row['ownership_closed'];persist()
        record['canonical_model_host']=row;write(base.RESULT,record)
    print(json.dumps(row));return 0 if row['passed']else 1

def prepare():
    guards.require(not ROOT.exists(),'fresh single-use canonical model check only')
    previous=json.loads(gen.git('show',PREVIOUS+':'+str(RESULT.relative_to(P))))
    guards.require(gen.committed(RESULT)==previous and previous['stage']=='closed' and
        len(previous['processes'])==1 and not previous['processes'][0]['passed'] and
        previous['processes'][0]['error']=='missing/extra strata prefill dense' and
        previous['processes'][0]['ownership_closed'],'original failed attempt must stay archived')
    full,count=gate.q.current_build();setup=base.setup_reader.pilot();parent=gen.committed(base.RESULT)
    host=parent['canonical_model_host']
    guards.require(host['passed']and host['ownership_closed']and host['tests']==7 and
        host['controller_sha256']==gen.common.digest(Path(__file__))and
        host['test_sha256']==gen.common.digest(P/'tests/core/test_canonical_model.py'),'matching model-launch/route host tests required')
    source=gen.git('rev-parse','HEAD').decode().strip()
    record=dict(stage='prepared',source_commit=source,native_source=full['source_commit'],binary=full['binary'],
        previous_attempt=dict(commit=PREVIOUS,path=str(RESULT.relative_to(P)),
            reason='Closed failed parser scope:three warmup plus three long-request dense records were mistaken for three whole-process records. The original sample and failure remain unchanged in Git. Fresh two-arm evidence uses the committed corrected parser;no replacement in the original cohort.'),
        native_inputs_recovered=count,build_sha256=gen.common.digest(gate.q.BUILD_RECORD),
        component_sha256=gen.common.digest(gate.q.COMPONENT),settings_sha256=gen.common.digest(gen.SETTINGS),
        device_gate_sha256=gen.common.digest(gate.RESULT),controller_fingerprint=assets.fingerprint(gen.benchmark_pins(source)),
        machine=assets.machine_identity(),models=assets.model_pins(setup['config'],parent['models'],full=True),host_contract=host,
        expected_request_sha256=parent['processes'][0]['requests'][0]['request_sha256'],plans=[],processes=[],
        plan=dict(order=[0,1],question='Does canonical chunk64 restore exact current decode state under the actual117K memory admission?',
            constants='Same native binary,1256 expert slots,131072 resident INT8 context,8192 prefill,5 workers+host,packed1,IQ2pair1,Q8 MTP,greedy,pcie0,adapt0,spec4,suffix0,caches0,no coupled draft/token graph. Only pre-exec canonical chunk flag differs.',
            execution='Two independent checked native processes,flag0 then1. Each uses pinned upstream bench.sh --prefill-only --prefill-tokens98304 --think --runs1,one short warmup then117489 fresh input and<=24 thinking output. Parameter validation,warnings,stage trace,snapshot/cache audit and transition observers identical in both arms;no full API trace.',
            correctness='Actual1/23 canonical copies between warmup and long prompt footer. Require full GPU BF16 residual admission,all48 dense layers restored,two exact cache payload/map cycles,complete disjoint wall/thread CPU/wait accounting,same model hashes and normal native/process/driver closure. Preserve each sample before any rejecting validator.',
            next_comparison='After both checked arms pass,commit a separate clean plan:three fixed process pairs off/on/on/off/off/on,four117K requests/process,first separate from three subsequent requests. Expected copy-only gain about0.14% of prompt;keep individual samples and require prospective mechanism/target gates. Repeated>=1K thinking generation for both observed MTP classes and agent/full-context/session checks remain required;no decode impact is acceptable.',
            stopping='Stop on first source/model/runtime/route/owner/fault/budget/accounting failure. No retry/replaced sample,no inference adoption or causal speed conclusion from these two observed first-use samples.',
            budgets=dict(wall_s=2400,cpu_s=72000,rss_bytes=96<<30,text_bytes=64<<20,total_bytes=64<<20),
            retention='Keep samples,compact individual scope/phase clocks,actual route/memory/copy/owner/fault/source proof. Retire successful full logs/JIT after verified committed closure;frozen inputs last through this two-arm admission only.'))
    ROOT.mkdir(mode=0o700);ports=set()
    for arm in (0,1):
        root=ROOT/str(arm);root.mkdir()
        for n in ('tmp','cache'):(root/n).mkdir()
        shutil.copyfile(P/'data/expert-profile.bin',root/'expert-profile.bin')
        old=setup['config']['args'];old=old[old.index('--expert-profile')+1]
        guards.require(gen.common.digest(root/'expert-profile.bin')==setup['generated_inputs'][old],'profile identity changed')
        write(root/'config.json',base.setup_reader.config(setup,root));write(root/'plan.json',dict(binary=full['binary'],native_validation_probe=None))
        with socket.socket()as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
        guards.require(port not in ports,'duplicate port;no retry');ports.add(port)
        record['plans'].append(dict(arm=arm,root=str(root),environment=environment(setup,root,arm),
            inputs={str(root/n):gen.common.digest(root/n)for n in ('config.json','plan.json','expert-profile.bin')},
            command=['/bin/sh',str(P/'bench.sh'),'--config',str(root/'config.json'),'--port',str(port),
                '--prefill-only','--prefill-tokens','98304','--think','--runs','1','--workload-id','native_gdn_20261018',
                '--fail-fast-engine-error','--ready-timeout','600','--out',str(root/'bench')]))
    current(record);write(ROOT/'frozen.json',record);write(RESULT,record)

def run():
    record=gen.committed(RESULT);current(record)
    frozen=json.loads((ROOT/'frozen.json').read_text());mutable={'stage','processes'}
    guards.require({k:v for k,v in record.items()if k not in mutable}=={k:v for k,v in frozen.items()if k not in mutable},'frozen admission changed')
    guards.require(record['stage']in('prepared','collecting')and len(record['processes'])<2 and
        all(r['passed']for r in record['processes']),'closed/failed canonical model attempt;no retry')
    selected=record['plans'][len(record['processes'])];root=Path(selected['root']);guards.require(not(root/'record.json').exists(),'existing attempt;inspect owner')
    boot,cursor=gate.faults.start_fault_scope();row=dict(arm=selected['arm'],passed=False,complete=False);owner=None
    def persist():
        row['active']=bool(owner and owner.active is not None);write(root/'record.json',dict(row,commands=owner.commands if owner else []))
    owner=owner_lib.Owner(root/'owned',persist)
    try:
        e,_,_=owner.run('bench',selected['command'],selected['environment'],P,wall=2400,cpu=72000,
            rss_cap=96<<30,text_cap=64<<20,file_cap=64<<20,total_cap=64<<20)
        paths=list((root/'bench').glob('*/results.json'));data=None
        if len(paths)==1:
            try:
                data=json.loads(paths[0].read_text());row['requests']=data.get('results',{}).get('prefill_only',{}).get('requests',[])
            except json.JSONDecodeError:row['result_json_incomplete']=True
        if not row.get('requests'):
            # A budget-stopped client may have durable completed journal rows
            # before results.json exists. Preserve those before rejecting it.
            row['requests']=[]
            journals=list((root/'bench').glob('*/requests.jsonl'))
            if len(journals)==1:
                for line in journals[0].read_text().splitlines():
                    try:event=json.loads(line)
                    except json.JSONDecodeError:
                        row['partial_journal_json']=True;break
                    if event.get('kind')=='result'and event.get('phase')=='prefill-only':
                        row['requests'].append(event['sample'])
                row['requests_from_unvalidated_partial_journal']=True
        guards.require(owner_lib.closed(e)and e['exit_code']==0 and len(paths)==1 and data is not None,'official benchmark/owner failed')
        samples,journal=base.history.official(root,data,1);row.update(requests=samples,journal=journal)
        native=gen.native_end(root,dict(binary=record['binary'],native_validation_probe=None),
            dict(config=json.loads((root/'config.json').read_text())),boot,e['pid'])
        text=(root/'engine.stderr').read_text();guards.require(not re.search(base.generation.FAILURE,text),'native failure')
        assets.verify_pool(text,5);row['route']=base.history.admission.mixed_evidence(text,samples[0])
        row['canonical_restore']=coverage(text,selected['arm'])
        profiles=observer.profiles(text)
        guards.require([r['scope']for r in profiles]==['dense_restore','cache_refill','native_owner_entry','dense_restore','cache_refill'],
            'missing/extra transition accounting')
        row['profiles']=compact_profiles(profiles)
        row['dense_memory']=dense_memory(text)
        guards.require(samples[0]['request_sha256']==record['expected_request_sha256'],'matched request bytes changed')
        current(record,full_models=True)
        row.update(passed=True,complete=True,native_identity=native['native'],native_exit_code=0,
            prompt_tok_s=samples[0]['actual_read_tokens']*1000/samples[0]['prompt_ms'])
    except BaseException as error:
        row['error']=str(error)
        if(root/'engine.stderr').exists():row['error_excerpt']=(root/'engine.stderr').read_text(errors='replace')[-4096:]
    finally:
        row['ownership_closed']=owner.active is None and all(owner_lib.closed(c)for c in owner.commands)
        row.update(fault_probe_successful=False,fresh_faults=None,boot_unchanged=False)
        try:row.update(gate.faults.fault_evidence(cursor,boot))
        except Exception as error:row['fault_probe_error']=str(error)
        row['passed']=bool(row['passed']and row['ownership_closed']and row['fault_probe_successful']and row['fresh_faults']==[]and row['boot_unchanged']);persist()
        record['processes'].append(row);record['stage']='collecting'if row['passed']and len(record['processes'])<2 else'closed';write(RESULT,record)
    print(json.dumps({k:row.get(k)for k in ('arm','passed','complete','ownership_closed','error')}));return 0 if row['passed']else 1

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('host','prepare','run'));args=parser.parse_args()
    with(B/'owned-v0141-measurement.lock').open('a+')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([]);guards.driver_guard()
        return host()if args.action=='host'else prepare()if args.action=='prepare'else run()
if __name__=='__main__':sys.exit(main())
