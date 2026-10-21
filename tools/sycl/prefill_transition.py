#!/usr/bin/env python3
"""Main-only owner off/on checks and one observed117K transition attribution.

No overlap, allocation optimization, speed adoption or sustained decode claim.
"""
import argparse
import fcntl
import json
import math
from pathlib import Path
import re
import shutil
import socket
import sys
sys.dont_write_bytecode = True
import prefill_recovery as base
from check_expert_bank import check_packed_cases

gen, guards, assets, faults, supervision = base.gen, base.guards, base.assets, base.faults, base.supervision
P, B, RESULT = base.P, base.B, base.RESULT
ROOT = B/'prefill-transition-profile-v1'
PREFIX = 'strata prefill transition'


def profiles(text):
    """Check disjoint closure and nested waits; retain only useful phase samples."""
    result = []
    for line in text.splitlines():
        if not line.startswith(PREFIX): continue
        part = line.startswith(PREFIX+' part: ')
        guards.require(line.startswith(PREFIX+(' part: ' if part else ': ')), 'malformed transition prefix')
        pairs = [word.split('=',1) for word in line.split(': ',1)[1].split()]
        guards.require(all(len(p)==2 for p in pairs) and len({p[0]for p in pairs})==len(pairs), 'malformed/duplicate transition fields')
        row = dict(pairs)
        guards.require(row['v']=='1', 'unknown transition schema')
        for key in list(row):
            if key.endswith('_s'):
                row[key]=float(row[key]);guards.require(math.isfinite(row[key]) and row[key]>=0, 'invalid transition duration')
        if part:
            guards.require(result and row['scope']==result[-1]['scope'] and row['kind'] not in result[-1]['parts'], 'orphan/duplicate transition part')
            for key in ('calls','items','logical_bytes','throws'):
                guards.require(re.fullmatch(r'\d+',row[key]) is not None,'invalid transition counter');row[key]=int(row[key])
            guards.require(row['calls']>0 and row['throws']==0,'transition part error')
            for clock in ('wall','cpu'):
                total=row['wall_s' if clock=='wall' else 'thread_cpu_s']
                wait=row['wait_'+clock+'_s'];outside=row['outside_wait_'+clock+'_s']
                guards.require(abs(total-wait-outside)<=5e-9,'part/wait conservation failed')
            result[-1]['parts'][row.pop('kind')]={k:v for k,v in row.items() if k not in ('v','scope')}
        else:
            guards.require(row['available']==row['complete']=='1' and row['open_parts']==row['wait_errors']=='0' and
                row['parts_disjoint']==row['waits_nested']==row['logical_bytes_not_bus_traffic']=='1' and
                row['units']=='charged_wrapper_intervals','incomplete transition accounting')
            row['parts']={};result.append(row)
    for row in result:
        for clock in ('wall','cpu'):
            key='wall_s' if clock=='wall' else 'thread_cpu_s'
            total=row[key];cells=list(row['parts'].values())
            guards.require(abs(total-row['other_'+clock+'_s']-sum(c[key]for c in cells))<=64e-9,
                'scope/part conservation failed')
            guards.require(sum(c['wait_'+clock+'_s']for c in cells)<=row['wait_'+clock+'_s']+32e-9<=total+64e-9,
                'scope/wait conservation failed')
    return result


def current(record, full_models=False):
    full,count=base.history.q.current_build()
    guards.require(full['source_commit']==record['native_source'] and full['binary']==record['binary'] and
        count==record['original_inputs_recovered'] and
        gen.common.digest(base.history.q.BUILD_RECORD)==record['build_sha256'] and
        gen.common.digest(base.history.q.COMPONENT)==record['component_sha256'] and
        assets.fingerprint(base.source_pins(record['source_commit']))==record['controller_fingerprint'] and
        gen.common.digest(gen.SETTINGS)==record['settings_sha256'] and
        assets.machine_identity()==record['machine'],'transition source/build/configuration drift')
    setup=base.setup_reader.pilot()
    guards.require(assets.model_pins(setup['config'],record['models'],full=full_models)==record['models'],'model assets changed')
    for name,sha in record['inputs'].items():guards.require(gen.common.digest(name)==sha,'transition input changed')
    guards.require(record['config']==base.setup_reader.config(setup,ROOT/'model') and
        record['environment']==dict(base.environment(setup,ROOT/'model',True),
            STRATA_PREFILL_TRANSITION_PROFILE='1',STRATA_PREFILL_LAYER_MAJOR_PROFILE='1'), 'current decode/environment drift')
    gen.runtime_assets_unchanged(dict(mode='engine-compare',runtime_asset_delta=setup['runtime_asset_delta']))
    return full


def prepare():
    parent=gen.committed(RESULT)
    guards.require(not ROOT.exists() and 'transition_profile' not in parent and parent['stage']=='closed' and
        len(parent['processes'])==4 and all(x['passed']for x in parent['processes']), 'fresh closed baseline required')
    full,count=base.history.q.current_build(); host=full['prefill_transition_host_contract']
    guards.require(host['passed'] and host['ownership_closed'] and host['cases']==11 and
        host['fixture_sha256']==full['fixtures']['prefill_transition_host_test']['sha256'],'same-build observer host cases required')
    source=gen.git('rev-parse','HEAD').decode().strip(); setup=base.setup_reader.pilot()
    ROOT.mkdir(mode=0o700);root=ROOT/'model';root.mkdir()
    for name in ('tmp','cache'): (root/name).mkdir()
    shutil.copyfile(P/'data/expert-profile.bin',root/'expert-profile.bin')
    config=base.setup_reader.config(setup,root);base.write(root/'config.json',config)
    base.write(root/'plan.json',dict(binary=full['binary'],native_validation_probe=None))
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    record=dict(stage='prepared',source_commit=source,native_source=full['source_commit'],binary=full['binary'],
        original_inputs_recovered=count,build_sha256=gen.common.digest(base.history.q.BUILD_RECORD),
        component_sha256=gen.common.digest(base.history.q.COMPONENT),controller_fingerprint=assets.fingerprint(base.source_pins(source)),
        settings_sha256=gen.common.digest(gen.SETTINGS),machine=assets.machine_identity(),
        expected_request_sha256=parent['processes'][0]['requests'][0]['request_sha256'],
        models=assets.model_pins(setup['config'],parent['models'],full=True),config=config,
        environment=dict(base.environment(setup,root,True),STRATA_PREFILL_TRANSITION_PROFILE='1',STRATA_PREFILL_LAYER_MAJOR_PROFILE='1'),
        inputs={str(root/n):gen.common.digest(root/n)for n in ('config.json','plan.json','expert-profile.bin')},
        command=['/bin/sh',str(P/'bench.sh'),'--config',str(root/'config.json'),'--port',str(port),'--prefill-only',
            '--prefill-tokens','98304','--think','--runs','1','--workload-id','native_gdn_20261018',
            '--fail-fast-engine-error','--ready-timeout','600','--out',str(root/'bench')],
        plan='First four independent device children: packed-smoke,packed-validate,native owner profile0,profile1. Native owner children use full UR tracing/parameter validation, exact real model payload/metadata, two enter/restore cycles. Then one checked117K thinking request through official bench.sh, with current1256-slot decode settings fixed. Attribute charged wrapper wall/thread CPU and nested waits. All model samples retained before validators. Stop at first source/owner/fault/budget/route/accounting failure; no retries.',
        limits='No performance adoption. One observed first-use request with24 output tokens does not establish steady speed,hardware saturation,sustained decode,agent/task quality orphysical262144/session/RESTORE. Native entry profile excludes prior graph/borrower retirement/canonical preparation; use outer v2 for context. No overlap/allocator change. Existing waits/operation/publication/prompt timing preserved.',
        retention='Keep useful individual phase samples/counters,source/configuration,model and owner/fault decisions. Retire successful API logs/JIT after verified committed closure. Retain6e40 baseline only through this named qualification/comparison.',device=None,model=None)
    current(record);base.write(ROOT/'frozen.json',record);parent['transition_profile']=record;base.write(RESULT,parent)


def execute(action):
    parent=gen.committed(RESULT);record=parent['transition_profile'];full=current(record)
    frozen=json.loads((ROOT/'frozen.json').read_text());mutable={'stage','device','model'}
    guards.require({k:v for k,v in record.items()if k not in mutable}=={k:v for k,v in frozen.items()if k not in mutable},'transition plan changed')
    device=action=='device-check'
    guards.require(record['stage']==('prepared'if device else 'device-qualified') and
        (device or record['device']['passed']), 'closed/failed transition stage; never retry')
    root=ROOT/'device' if device else ROOT/'model'
    guards.require(not (root/'record.json').exists(),'existing attempt; inspect its owner')
    if device:
        root.mkdir()
        for name in ('tmp','cache'):(root/name).mkdir()
    boot,cursor=faults.start_fault_scope();row=dict(passed=False,complete=False);owner=None
    def persist():
        row['active']=bool(owner and owner.active is not None);base.write(root/'record.json',dict(row,commands=owner.commands if owner else []))
    owner=supervision.Owner(root/'owned',persist)
    try:
        if device:
            row['checks']=[];settings=gen.committed(gen.SETTINGS)['environment']
            pack=record['config']['args'];pack=pack[pack.index('--pack')+1]
            shards=[name for name in record['models']if name.endswith('.gguf')]
            for mode in ('packed-smoke','packed-validate','native-off','native-on'):
                native=mode.startswith('native-');fixture=full['fixtures']['sycl_native_dense_bank_test'if native else 'sycl_expert_bank_test']
                env=dict(settings,TMPDIR=str(root/'tmp'),XDG_CACHE_HOME=str(root/'cache'),
                    STRATA_PREFILL_TRANSITION_PROFILE='1'if mode=='native-on'else'0')
                if not native:env['UR_ENABLE_LAYERS']='UR_LAYER_PARAMETER_VALIDATION';env.pop('UR_LOG_TRACING',None)
                command=[fixture['path'],pack,*shards]if native else[fixture['path'],'--'+mode]
                entry,out,err=owner.run(mode,command,env,P,wall=360,cpu=900,rss_cap=12<<30,text_cap=64<<20,file_cap=64<<20,total_cap=64<<20)
                guards.require(supervision.closed(entry) and entry['exit_code']==0 and not re.search(base.generation.FAILURE,err.read_text()),'device prerequisite failed')
                metrics=profiles(err.read_text())
                if native:
                    guards.require(out.read_text()=='PASS matrices=300 layers=48 snapshot=2116951040 bank=50918400\n','native owner/payload fixture incomplete')
                    guards.require([x['scope']for x in metrics]==(['native_owner_entry','native_owner_restore']*2 if mode=='native-on'else[]),'native observer activation differs')
                    for x in metrics:
                        part=x['parts']['release'if x['scope']=='native_owner_entry'else'submit']
                        guards.require(part['items']==300 and part['logical_bytes']==2116951040,'native owner/byte conservation differs')
                else:check_packed_cases(out.read_text(),mode)
                row['checks'].append(dict(mode=mode,fixture_sha256=fixture['sha256'],passed=True,profiles=metrics))
        else:
            entry,_,_=owner.run('bench',record['command'],record['environment'],P,wall=2400,cpu=72000,rss_cap=96<<30,text_cap=64<<20,file_cap=64<<20,total_cap=64<<20)
            paths=list((root/'bench').glob('*/results.json'))
            if len(paths)==1:
                data=json.loads(paths[0].read_text());row['requests']=data.get('results',{}).get('prefill_only',{}).get('requests',[])
            guards.require(supervision.closed(entry) and entry['exit_code']==0 and len(paths)==1,'official benchmark failed')
            samples,journal=base.history.official(root,data,1);row.update(requests=samples,journal=journal)
            native=gen.native_end(root,dict(binary=record['binary'],native_validation_probe=None),dict(config=record['config']),boot,entry['pid'])
            log=(root/'engine.stderr').read_text();assets.verify_pool(log,5)
            guards.require(not re.search(base.generation.FAILURE,log),'native runtime failure')
            row['route']=base.history.admission.mixed_evidence(log,samples[0]);row['profiles']=profiles(log)
            guards.require([x['scope']for x in row['profiles']]==['dense_restore','cache_refill','native_owner_entry','dense_restore','cache_refill'],'missing/extra request transition profiles')
            guards.require(samples[0]['request_sha256']==record['expected_request_sha256'],'baseline request bytes changed')
            entry_profile,restore_profile,refill_profile=row['profiles'][2:]
            for scope,kind in ((entry_profile,'release'),(restore_profile,'allocate'),(restore_profile,'submit')):
                part=scope['parts'][kind]
                guards.require(part['items']==300 and part['logical_bytes']==2116951040,'model native owner/byte conservation differs')
            part=refill_profile['parts']['submit'];loan=row['route']['long_loan']
            guards.require(part['items']==loan['lent'] and 0<part['logical_bytes']<=loan['capacity'],
                'model cache refill count/byte conservation differs')
            row['native_identity']=native['native'];row['native_exit_code']=0
        current(record,full_models=not device);row.update(passed=True,complete=True)
    except BaseException as error:row['error']=str(error)
    finally:
        row['ownership_closed']=owner.active is None and all(supervision.closed(x)for x in owner.commands)
        row.update(fault_probe_successful=False,fresh_faults=None,boot_unchanged=False)
        try:row.update(faults.fault_evidence(cursor,boot))
        except Exception as error:row['fault_probe_error']=str(error)
        row['passed']=bool(row['passed'] and row['ownership_closed'] and row['fault_probe_successful'] and row['fresh_faults']==[] and row['boot_unchanged']);persist()
        record['device'if device else'model']=row;record['stage']='device-qualified'if device and row['passed']else'closed';base.write(RESULT,parent)
    print(json.dumps({k:row.get(k)for k in ('passed','complete','ownership_closed','error')}));return 0 if row['passed']else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','device-check','model-profile'));args=parser.parse_args()
    with (B/'owned-v0141-measurement.lock').open('a+')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([]);guards.driver_guard()
        return prepare()if args.action=='prepare'else execute(args.action)
if __name__=='__main__':sys.exit(main())
