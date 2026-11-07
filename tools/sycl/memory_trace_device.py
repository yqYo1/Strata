#!/usr/bin/env python3
"""Bounded actual GPU allocation/submission qualification; Main only, no model.

Main-authored Strata diagnostic; repository LICENSE applies.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode=True
import memory_trace as host

P,B,RESULT=host.P,host.B,host.RESULT
ROOT=B/'ur-memory-device-v1'
KEY='ur_memory_device_v1'
INPUTS=sorted(set([*host.INPUTS,'tools/sycl/memory_trace_device.py','tests/sycl/compat.cpp']))
require,digest,write=host.require,host.digest,host.write


def pins(revision):
    result={}
    for name,raw in host.model.gate.native.build.blobs(P,revision,INPUTS):
        require(raw==(P/name).read_bytes(),'commit exact device-trace input: '+name)
        result[name]=hashlib.sha256(raw).hexdigest()
    return result


def compact(r):
    return {k:v for k,v in r.items() if k not in ('environment','dependencies','commands')}


def prepare(container):
    require(KEY not in container and not ROOT.exists(),'fresh device memory gate only')
    old=json.loads((host.ROOT/'record.json').read_text());host.admission(old,container)
    require(old['passed'] and old['complete'] and not old['active'] and old['ownership_closed'] and
        old['outcomes']['check']['packet_controls']==12 and old['outcomes']['check']['host_xpti_activation'],
        'completed host packet and loader activation required')
    _,baseline=host.model.gate.native.source.baseline()
    selected=json.loads((B/'main-native-gdn-engine/record.json').read_text())
    fixture=selected['fixtures']['sycl_submission_test']
    require(digest(fixture['path'])==fixture['sha256'],'selected submission fixture changed')
    revision=host.model.gate.native.build.git(P,'rev-parse','HEAD').decode().strip()
    source=pins(revision)
    ROOT.mkdir(mode=0o700)
    for name in ('tmp','cache'):(ROOT/name).mkdir(mode=0o700)
    env=dict(old['environment']['trace'],TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'cache'))
    r=dict(stage='prepared',source_commit=revision,input_sha256=source,selected_original=baseline,
        host_qualification=dict(result_key=host.KEY,source_commit=old['source_commit'],
            evidence_sha256=host.model.model.fingerprint(container[host.KEY])),
        collector=old['assets']['libstrata_ur_memory.so'],dependencies=old['dependencies'],
        command=[fixture['path'],'--submission'],fixture=fixture,fixture_source_commit=selected['source_commit'],
        environment=env,environment_sha256=host.model.model.fingerprint(env),
        driver_before=host.model.gate.native.guards.driver_guard(),
        boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        active=False,passed=False,complete=False,ownership_closed=True,GPU_executed=False,
        budgets=dict(wall_s=120,cpu_s=120,rss_bytes=4<<30,text_bytes=16<<20,file_bytes=16<<20,total_bytes=64<<20),
        criteria='Stop first failure,no retry. Exact selected6e40 sycl_submission_test --submission,with unchanged qualified safety and validation plus bounded memory-only XPTI subscriber. Require all10 original numerical/copy/event/profiling/lifetime checks,normal owned exit0,actual successful host/device USM allocations and frees,one collector activation,zero invalid/unsupported/lost records,no free-before-known-allocation,unchanged identities/boot/driver and no fresh fault. Residual requested allocations are reported,not called leaks;UR pools/driver storage are not physical-byte accounting. No model,speed,task,phase preservation or ordinary runtime teardown qualification.',
        retention='Retain compact individual gate/byte conservation and distinct unresolved errors. Retire successful raw trace/JIT after the immediate model-trace plan consumes this gate. Collector is a sidecar to the existing candidate;no engine generation added.')
    write(ROOT/'record.json',r);container[KEY]=compact(r);write(RESULT,container)
    print('Prepared actual GPU memory gate;commit before execution.',flush=True)


def admission(r,container):
    require(RESULT.read_bytes()==host.model.gate.native.build.git(P,'show','HEAD:'+str(RESULT.relative_to(P))) and
        compact(r)==container[KEY] and pins(r['source_commit'])==r['input_sha256'],'commit device-trace plan')
    require(host.model.model.fingerprint(r['environment'])==r['environment_sha256'] and
        all(digest(p)==h for p,h in r['dependencies'].items()),'effective environment/header/runtime changed')
    q=r['host_qualification'];require(host.model.model.fingerprint(container[q['result_key']])==q['evidence_sha256'],
        'consumed host qualification changed')
    require(host.model.gate.native.source.baseline()[1]==r['selected_original'],'selected native reference changed')
    for asset in (r['collector'],r['fixture']):require(digest(asset['path'])==asset['sha256'],'device-trace artifact changed')
    require(host.model.gate.native.guards.driver_guard()==r['driver_before'] and
        Path('/proc/sys/kernel/random/boot_id').read_text().strip()==r['boot'],'driver/boot changed')


def evidence(text,stdout,pid):
    activation=host.model.gate.validation.preload_evidence(text,pid)
    cases=re.findall(r'^case (.+)=PASS$',stdout,re.M)
    require(len(cases)==10 and 'fixture_status=PASS mode=submission' in stdout,'submission checks incomplete')
    require(text.count('activated stream=ur.call format=1 limit=25000')==1,'collector not activated once')
    summaries=re.findall(r'^strata UR memory summary: active=1 records=(\d+) invalid=0 unsupported=0 lost=0 ',text,re.M)
    require(len(summaries)==1,'collector incomplete or unsupported memory API')
    lines=re.findall(r'^strata UR memory: seq=.*$',text,re.M)
    require(len(lines)==int(summaries[0]) and lines,'memory record conservation failed')
    rows=[dict(re.findall(r'(\w+)=([^ ]*)',line)) for line in lines]
    require(sorted(int(x['seq']) for x in rows)==list(range(1,len(rows)+1)),'missing/repeated memory sequence')
    live={};allocations={};frees=0
    for row in sorted(rows,key=lambda x:int(x['seq'])):
        require(row['result']=='0','memory API failed during successful submission gate')
        kind,domain,pointer=row['kind'],row['domain'],row['ptr']
        require(kind in ('alloc','free'),'unplanned buffer/alias API needs separate ledger qualification')
        if kind=='alloc':
            size=int(row['requested_bytes']);require(size>0 and pointer!='(nil)' and pointer not in live,
                'invalid/reused live allocation')
            live[pointer]=(domain,size)
            a=allocations.setdefault(domain,dict(count=0,requested_bytes=0));a['count']+=1;a['requested_bytes']+=size
        else:
            require(pointer in live,'free has no known allocation');del live[pointer];frees+=1
    require(all(allocations.get(domain,{}).get('count',0)>0 for domain in ('host','device')) and frees>0,
        'real host/device allocation and free not exercised')
    return dict(activation=activation,original_submission_checks=10,memory_records=len(rows),
        allocations=allocations,successful_frees=frees,residual_requested_allocations=len(live),
        residual_requested_bytes=sum(size for _,size in live.values()),
        scope='Requested UR allocation lifetime only;residuals are not a physical-residency/leak claim.')


def run(container):
    r=json.loads((ROOT/'record.json').read_text());admission(r,container)
    require(r['stage']=='prepared','closed device gate cannot be retried')
    original=json.loads(json.dumps(r));owner=None
    cursor_text=subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15)
    cursor=next(s.split(': ',1)[1] for s in cursor_text.splitlines() if s.startswith('-- cursor: '))
    def persist():
        r['active']=bool(owner and owner.active is not None)
        write(ROOT/'record.json',dict(r,commands=owner.commands if owner else []))
    owner=host.model.gate.native.supervision.Owner(ROOT/'owned',persist)
    r.update(stage='running',GPU_executed=True);persist()
    try:
        b=r['budgets'];entry,out,err=owner.run('device',r['command'],r['environment'],P,
            wall=b['wall_s'],cpu=b['cpu_s'],rss_cap=b['rss_bytes'],text_cap=b['text_bytes'],
            file_cap=b['file_bytes'],total_cap=b['total_bytes'])
        r['exit_code']=entry['exit_code']
        require(host.model.gate.native.supervision.closed(entry) and entry['exit_code']==0,
            'normal device exit required: '+entry.get('error',str(entry['exit_code'])))
        r['evidence']=evidence(err.read_text(errors='replace'),out.read_text(errors='replace'),entry['pid'])
        admission(original,container);r['host_probe_accepted']=True
    except BaseException as error:r['error']=repr(error)
    finally:
        host.model.gate.validation.finish(r,owner,cursor,r['boot'])
        r['passed']=bool(r['passed'] and r.get('driver_after')==r['driver_before'])
        r['complete']=r['passed'];r['stage']='closed' if r['passed'] else 'failed';persist()
        container[KEY]=compact(r);write(RESULT,container)
    print(json.dumps({k:r.get(k) for k in ('passed','complete','ownership_closed','exit_code','error','fresh_faults')}),flush=True)
    return 0 if r['passed'] else 1


if __name__=='__main__':
    require(sys.argv[1:] in (['prepare'],['run']),'prepare|run')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);host.model.gate.native.guards.no_model_owners([])
        container=json.loads(RESULT.read_text())
        if sys.argv[1]=='prepare':prepare(container)
        else:sys.exit(run(container))
