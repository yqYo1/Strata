#!/usr/bin/env python3
"""Collect the known short-prefill memory interval; Main only, never speed.

Main-authored diagnostic, repository LICENSE. The original engine is unchanged.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode=True
import memory_trace_device as device

host=device.host
model=host.model
P,B,RESULT=host.P,host.B,host.RESULT
ROOT=B/'ur-memory-model-v1'
KEY='ur_memory_model_v1'
INPUTS=sorted(set([*device.INPUTS,'tools/sycl/memory_trace_model.py']))
require,digest,write=host.require,host.digest,host.write


def pins(revision):
    result={}
    for name,raw in model.gate.native.build.blobs(P,revision,INPUTS):
        require(raw==(P/name).read_bytes(),'commit model-trace input: '+name)
        result[name]=hashlib.sha256(raw).hexdigest()
    return result


def compact(r):
    return {k:v for k,v in r.items() if k not in ('environment','dependencies','commands')}


def prepare(container):
    require(KEY not in container and not ROOT.exists(),'fresh model memory trace only')
    parent=json.loads((device.ROOT/'record.json').read_text());device.admission(parent,container)
    require(parent['complete'] and parent['passed'] and not parent['active'] and parent['ownership_closed'] and
        parent['fresh_faults']==[],'completed actual GPU observer gate required')
    state=container[model.KEY]
    require(state['complete'] and state['passed'] and not state['active'] and state['ownership_closed'] and
        state['binary']==model.gate.engine()['binary'],'completed same-native model-state reference required')
    base=model.model.setup();require(base['models']==state['model_assets'],'model reference differs')
    model.model.model_pins(base['config'],state['model_assets'])
    revision=model.gate.native.build.git(P,'rev-parse','HEAD').decode().strip();source=pins(revision)
    ROOT.mkdir(mode=0o700)
    for name in ('tmp','cache','maps'):(ROOT/name).mkdir(mode=0o700)
    env=dict(parent['environment'],TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'cache'))
    r=dict(stage='prepared',source_commit=revision,input_sha256=source,
        native_source_commit=state['native_source_commit'],binary=state['binary'],
        command=state['cases']['small']['command'],
        cases={'small':state['cases']['small']},vocab_size=state['vocab_size'],
        model_assets=state['model_assets'],model_digest_reuse=state['model_digest_reuse'],
        reference_model_state=dict(result_key=model.KEY,sha256=model.model.fingerprint(state)),
        observer_gate=dict(result_key=device.KEY,sha256=model.model.fingerprint(container[device.KEY])),
        collector=parent['collector'],dependencies=parent['dependencies'],
        environment=env,environment_sha256=model.model.fingerprint(env),
        driver_before=parent['driver_before'],boot=parent['boot'],
        row={},maps=[],active=False,passed=False,complete=False,ownership_closed=True,
        GPU_executed=False,speed_eligible=False,model_adoption_qualified=False,
        decode_performance_qualified=False,physical262144_qualified=False,
        ordinary_runtime_teardown_qualified=False,physical_owner_attribution_qualified=False,
        budgets=dict(wall_s=1200,cpu_s=7200,rss_bytes=104<<30,text_bytes=32<<20,
            file_bytes=32<<20,total_bytes=256<<20,map_snapshot_bytes=1<<20),
        criteria='Same current native590f small A16385/B33/A+16 protocol and model-state evidence,including all1256 cache payloads/tables and main/draft/running restore. Only new observer is the already host/device-qualified memory-only XPTI subscriber;qualified safety remains. Require one activation,exact sequence/count conservation,zero invalid/unsupported/lost,normal owned exit0,unchanged source/observer/runtime/model identities,full15 model digests after collection,unchanged boot/driver and no fresh fault. Capture maps at READY and each DONE while owned PID is live for later module symbolization. Requested bytes and stderr-phase ordering are collection evidence,not physical ownership,pool residency,leak,CPU/GPU saturation or clean speed. Stop first failure,no retry;interpret captured trace in a separate bounded offline analysis.',
        retention=dict(owner='Main',consumer='Immediate offline attribution of the B33 free-memory drop and allocation stacks to loaded modules;retire trace/maps/JIT after useful owner/cost facts are recorded or the question is superseded.',
            raw_text_budget_bytes=32<<20,map_budget_bytes=4<<20,
            permanent='Source,recipe,individual request/state and trace-conservation/free-memory results,qualification limits. No full trace or maps in Git.'))
    write(ROOT/'record.json',r);container[KEY]=compact(r);write(RESULT,container)
    print('Prepared bounded same-native model allocation trace;commit before run.',flush=True)


def admission(r,container):
    require(RESULT.read_bytes()==model.gate.native.build.git(P,'show','HEAD:'+str(RESULT.relative_to(P))) and
        compact(r)==container[KEY] and pins(r['source_commit'])==r['input_sha256'],'commit exact model trace plan')
    require(model.model.fingerprint(r['environment'])==r['environment_sha256'] and
        all(digest(p)==h for p,h in r['dependencies'].items()),'observer environment/header/runtime changed')
    for key in ('reference_model_state','observer_gate'):
        ref=r[key];require(model.model.fingerprint(container[ref['result_key']])==ref['sha256'],
            'consumed qualification changed: '+ref['result_key'])
    native=model.gate.engine();require(native['binary']==r['binary'] and native['source_commit']==r['native_source_commit'],
        'native engine changed')
    require(digest(r['collector']['path'])==r['collector']['sha256'],'collector changed')
    base=model.model.setup();require(base['models']==r['model_assets'],'model reference changed')
    model.model.model_pins(base['config'],r['model_assets'])
    require(model.gate.native.guards.driver_guard()==r['driver_before'] and
        Path('/proc/sys/kernel/random/boot_id').read_text().strip()==r['boot'],'boot/driver changed')


class MapsAtBoundary:
    def __init__(self,io,pid,r):self.io,self.pid,self.r=io,pid,r
    def __getattr__(self,name):return getattr(self.io,name)
    def line(self,seconds=700):
        line=self.io.line(seconds)
        if line.startswith(('READY ','DONE ')):
            cap=self.r['budgets']['map_snapshot_bytes']
            with open('/proc/'+str(self.pid)+'/maps','rb') as f:raw=f.read(cap+1)
            require(0<len(raw)<=cap,'owned model map snapshot exceeds budget')
            path=ROOT/'maps'/('boundary-'+str(len(self.r['maps']))+'.txt');path.write_bytes(raw)
            self.r['maps'].append(dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),
                boundary='READY' if line.startswith('READY ') else 'DONE'))
        return line


def trace_evidence(text):
    require(text.count('activated stream=ur.call format=1 limit=25000')==1,'collector not activated once')
    summary=re.findall(r'^strata UR memory summary: active=1 records=(\d+) invalid=0 unsupported=0 lost=0 ',text,re.M)
    require(len(summary)==1,'incomplete collector or unsupported memory API')
    rows=[dict(re.findall(r'(\w+)=([^ ]*)',s)) for s in re.findall(r'^strata UR memory: seq=.*$',text,re.M)]
    require(len(rows)==int(summary[0]) and rows and
        sorted(int(r['seq']) for r in rows)==list(range(1,len(rows)+1)),'memory sequence/count conservation failed')
    require(all(r['kind'] in ('alloc','free','retain','release','alias') for r in rows),'unknown packet interpretation')
    starts=list(re.finditer(r'^strata trace: request (\d+) \d+$',text,re.M))
    require([int(m[1]) for m in starts]==[16385,33,16401],'actual observed request framing differs')
    frame=text[starts[1].end():starts[2].start()]
    begin=re.search(r'^strata prefill dense: phase=cache-loan residency=Full activations=0 .*free_bytes=(\d+) total_bytes=(\d+)$',frame,re.M)
    end=re.search(r'^strata prefill dense: phase=restored residency=Full activations=0 .*free_bytes=(\d+) total_bytes=(\d+)$',frame,re.M)
    require(begin and end and begin.end()<end.start() and begin[2]==end[2],'short32 phase boundaries missing')
    return dict(records=len(rows),api_failures=sum(r['result']!='0' for r in rows),
        short32_interval=dict(before_reported_free_bytes=int(begin[1]),after_reported_free_bytes=int(end[1]),
            decrease_bytes=int(begin[1])-int(end[1]),total_reported_bytes=int(begin[2]),
            memory_records_in_stderr_interval=len(re.findall(r'^strata UR memory: seq=',frame[begin.end():end.start()],re.M))),
        interpretation='Collection integrity only. Buffer alias/refcount ledger,driver storage and module/caller attribution remain offline;stderr line bounds can differ from asynchronous callback start time.')


def run(container):
    r=json.loads((ROOT/'record.json').read_text());admission(r,container)
    require(r['stage']=='prepared','closed model trace cannot be retried')
    original=json.loads(json.dumps(r));owner=None
    journal=subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15)
    cursor=next(s.split(': ',1)[1] for s in journal.splitlines() if s.startswith('-- cursor: '))
    def persist():
        r['active']=bool(owner and owner.active is not None)
        write(ROOT/'record.json',dict(r,commands=owner.commands if owner else []))
    owner=model.gate.native.supervision.Owner(ROOT/'owned',persist);r.update(stage='running',GPU_executed=True);persist()
    try:
        b=r['budgets'];entry,out,err=owner.run('model',r['command'],r['environment'],P,
            interaction=lambda io,e:model.interact(MapsAtBoundary(io,e['pid'],r),'small',r['row'],r),
            wall=b['wall_s'],cpu=b['cpu_s'],rss_cap=b['rss_bytes'],text_cap=b['text_bytes'],
            file_cap=b['file_bytes'],total_cap=b['total_bytes'])
        r['exit_code']=entry['exit_code']
        require(model.gate.native.supervision.closed(entry) and entry['exit_code']==0,
            'normal model exit required: '+entry.get('error',str(entry['exit_code'])))
        text=err.read_text(errors='replace');model.evidence(text,'small',r['row'],r,entry['pid'])
        r['native_state_diagnostic_passed']=True
        require(len(r['maps'])==4,'map snapshots incomplete')
        r['trace']=trace_evidence(text);admission(original,container)
        model.model.model_pins(model.model.setup()['config'],r['model_assets'],full=True)
        r['full_model_hashes_matched']=15;r['host_probe_accepted']=True
    except BaseException as error:r['error']=repr(error)
    finally:
        model.gate.validation.finish(r,owner,cursor,r['boot'])
        r['passed']=bool(r['passed'] and r.get('driver_after')==r['driver_before'])
        r['complete']=r['passed'];r['stage']='closed' if r['passed'] else 'failed';persist()
        container[KEY]=compact(r);write(RESULT,container)
    print(json.dumps({k:r.get(k) for k in ('passed','complete','ownership_closed','exit_code','error','fresh_faults')}),flush=True)
    return 0 if r['passed'] else 1


if __name__=='__main__':
    require(sys.argv[1:] in (['prepare'],['run']),'prepare|run')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);model.gate.native.guards.no_model_owners([])
        container=json.loads(RESULT.read_text())
        if sys.argv[1]=='prepare':prepare(container)
        else:sys.exit(run(container))
