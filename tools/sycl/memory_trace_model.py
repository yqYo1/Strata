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
import struct

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


def analyze(container):
    """Offline requested-byte conservation; no GPU/runtime initialization."""
    r=json.loads((ROOT/'record.json').read_text())
    require(RESULT.read_bytes()==model.gate.native.build.git(P,'show','HEAD:'+str(RESULT.relative_to(P))) and
        compact(r)==container[KEY],'commit the original closed collection first')
    require(r['complete'] and r['passed'] and not r['active'] and r['ownership_closed'] and
        r['exit_code']==0 and r['full_model_hashes_matched']==15,'completed collection required')
    require('ur_allocation_analysis' not in container['owner_trace_obligations'],'no analysis overwrite')
    revision=model.gate.native.build.git(P,'rev-parse','HEAD').decode().strip()
    current_inputs=pins(revision)
    for name,h in r['input_sha256'].items():
        original=model.gate.native.build.git(P,'show',r['source_commit']+':'+name)
        require(hashlib.sha256(original).hexdigest()==h,'original collector source unrecoverable: '+name)
    path=ROOT/'owned/model.stderr';require(path.stat().st_size<=r['budgets']['text_bytes'],'trace size cap')
    raw=path.read_bytes();text=raw.decode();require(trace_evidence(text)==r['trace'],'collection evidence changed')
    maps=[]
    for snapshot in r['maps']:
        rawmap=Path(snapshot['path']).read_bytes()
        require(len(rawmap)==snapshot['bytes'] and hashlib.sha256(rawmap).hexdigest()==snapshot['sha256'],
            'captured map identity changed')
        maps.append([s.split(maxsplit=5) for s in rawmap.decode().splitlines()])
    binary=Path(r['binary']['path']);require(digest(binary)==r['binary']['sha256'],'native symbol module changed')
    # This qualified engine is ET_EXEC. Do not guess a PIE/shared-library load
    # bias or symbolize unpinned library frames from their pathname alone.
    with binary.open('rb') as stream:header=stream.read(64)
    require(header[:6]==b'\x7fELF\x02\x01' and struct.unpack_from('<H',header,16)[0]==2,
        'native symbolization supports the pinned little-endian ELF64 ET_EXEC only')
    ranges=[]
    for snapshot in maps:
        found=[tuple(int(v,16) for v in s[0].split('-')) for s in snapshot
            if len(s)==6 and 'x' in s[1] and s[5]==str(binary)]
        require(found,'native executable mapping absent');ranges.append(found)
    require(all(v==ranges[0] for v in ranges),'native executable mappings changed')
    rows=[]
    for match in re.finditer(r'^strata UR memory: seq=.*$',text,re.M):
        row=dict(re.findall(r'(\w+)=([^ ]*)',match[0]));row['offset']=match.start();rows.append(row)
    require(all(row['result']=='0' and row['kind'] in ('alloc','free','release') for row in rows),
        'this trace requires another failed-call/refcount/alias decoder')
    live={};totals={};peak={};current={};frees=0
    for row in sorted(rows,key=lambda v:int(v['seq'])):
        pointer=row['ptr']
        if row['kind']=='alloc':
            domain=row['domain'];size=int(row['requested_bytes'])
            require(pointer!='(nil)' and pointer not in live and size>0,'invalid or overlapping live allocation')
            live[pointer]=(domain,size)
            item=totals.setdefault(domain,dict(count=0,requested_bytes=0));item['count']+=1;item['requested_bytes']+=size
            current[domain]=current.get(domain,0)+size;peak[domain]=max(peak.get(domain,0),current[domain])
        else:
            require(pointer in live,'free/release without a recorded live allocation')
            domain,size=live.pop(pointer)
            require((row['kind']=='release')==(domain=='buffer'),'USM/buffer release kind differs')
            current[domain]-=size;frees+=1
    require(not live and all(v==0 for v in current.values()),'requested-byte ledger does not close')
    start=text.index('strata trace: request 33 0\n');stop=text.index('strata trace: request 16401 ',start)
    begin=text.index('strata prefill dense: phase=cache-loan ',start)
    end=text.index('strata prefill dense: phase=restored ',begin)
    require(start<begin<end<stop,'short request marker order differs')
    interval=[row for row in rows if begin<row['offset']<end]
    addresses=sorted({int(pc,16) for row in interval if row['kind']=='alloc' for pc in row['stack'].split(',')
        if pc and any(lo<=int(pc,16)<hi for lo,hi in ranges[0])})
    require(0<len(addresses)<=4096,'bounded native symbol address set required')
    tool=Path('/usr/bin/addr2line').resolve();tool_hash=digest(tool)
    command=[str(tool),'--functions','--demangle','--exe',str(binary),*[hex(pc-1) for pc in addresses]]
    symbols=subprocess.check_output(command,timeout=30)
    require(len(symbols)<=1<<20 and digest(tool)==tool_hash and digest(binary)==r['binary']['sha256'],
        'symbolizer output or input identity changed')
    lines=symbols.decode().splitlines();require(len(lines)==2*len(addresses),'unexpected symbolizer fields')
    functions={pc:lines[2*i] for i,pc in enumerate(addresses)}
    events=[]
    for row in interval:
        native=[dict(return_pc=pc,function=functions[int(pc,16)]) for pc in row['stack'].split(',')
            if pc and int(pc,16) in functions]
        events.append(dict(seq=int(row['seq']),kind=row['kind'],domain=row['domain'],
            requested_bytes=int(row['requested_bytes']),tid=int(row['tid']),native_frames=native))
    result=dict(source_commit=revision,input_sha256=current_inputs,collection_source_commit=r['source_commit'],
        trace=dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()),
        maps_sha256=[v['sha256'] for v in r['maps']],native_symbol_module=r['binary'],
        symbolizer=dict(path=str(tool),sha256=tool_hash,return_pc_adjustment=-1),
        successful_allocations=totals,successful_releases=frees,requested_peak_bytes_by_domain=peak,
        final_requested_allocations=0,final_requested_bytes=0,stderr_interval_events=events,
        reported_free_memory_decrease_bytes=r['trace']['short32_interval']['decrease_bytes'],
        causal_phase_attribution_qualified=False,physical_owner_attribution_qualified=False,
        interpretation='All observed successful USM allocations/frees and the two simple buffer create/release pairs conserve requested bytes. Peak is callback-sequence requested-byte accounting,not physical residency. The stderr-bounded event list is not causal B33 assignment;library frames remain unresolved and driver/pool/program/command storage is unobserved. Zero final requested bytes is not physical allocator cleanup or ordinary-runtime-teardown qualification.',
        next_action='Measure the owning DRM client and lower-level allocator/program/command backing at matched phase boundaries;do not name a leak or library owner from global free memory alone.')
    container['owner_trace_obligations']['ur_allocation_analysis']=result;write(RESULT,container)
    print(json.dumps({k:result[k] for k in ('successful_allocations','successful_releases',
        'final_requested_bytes','stderr_interval_events','reported_free_memory_decrease_bytes')}),flush=True)


if __name__=='__main__':
    require(sys.argv[1:] in (['prepare'],['run'],['analyze']),'prepare|run|analyze')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);model.gate.native.guards.no_model_owners([])
        container=json.loads(RESULT.read_text())
        if sys.argv[1]=='prepare':prepare(container)
        elif sys.argv[1]=='run':sys.exit(run(container))
        else:analyze(container)
