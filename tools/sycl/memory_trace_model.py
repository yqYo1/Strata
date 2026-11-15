#!/usr/bin/env python3
"""Main-only matched control of readback-audit residency; not speed/adoption.

Main-authored Strata diagnostic, repository LICENSE. No engine/kernel rebuild.
"""
from collections import ChainMap
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

sys.dont_write_bytecode=True
import memory_trace_device as device
host=device.host
model=host.model
P,B,RESULT=host.P,host.B,host.RESULT
ROOT=B/'neo-readback-control-v1'
KEY='neo_readback_control_v1'
PARENT_ROOT=B/'neo-memory-model-v3'
PARENT_KEY='neo_memory_model_v3'
FIXTURE_KEY='neo_memory_fixture_v1'
REVIEW_KEY='neo_logging_source_review'
PROTOCOL_KEY='neo_trailing_protocol_host_control_v1'
AUDITS=('STRATA_STATE_HASH','STRATA_SNAPSHOT_VERIFY','STRATA_SYCL_CACHE_LOAN_AUDIT')
ORDER=('on','off','off','on')
INPUTS=sorted(set([*device.INPUTS,'tools/sycl/memory_trace_model.py',
    'bench/sycl/client_resources.py','tests/core/test_client_resources.py',
    'tools/sycl/memory_trace_protocol_test.py']))
sys.path.insert(0,str(P/'bench/sycl'))
import client_resources as resources
require,digest,write=host.require,host.digest,host.write


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def protocol_vectors(container):
    return [v['representative'] for v in container[FIXTURE_KEY]['logging_evidence']['categories'].values()]


def protocol_input_fingerprint(container):
    failed=container['neo_memory_model_v1']
    return fingerprint(dict(contract=container['owner_trace_obligations'][REVIEW_KEY]['output_contract'],
        vectors=protocol_vectors(container),failed={k:failed[k] for k in ('passed','complete')}))


def audit_environment(base,private,audit):
    require(audit in ('on','off') and all(base.get(k)=='1' for k in AUDITS),
        'matched baseline must enable exactly the declared readback audits')
    environment=dict(base,TMPDIR=str(private/'tmp'),XDG_CACHE_HOME=str(private/'cache'))
    if audit=='off':
        for name in AUDITS:environment.pop(name)
    return environment


def pins(revision):
    result={}
    for name,raw in model.gate.native.build.blobs(P,revision,INPUTS):
        require(raw==(P/name).read_bytes(),'commit exact control input: '+name)
        result[name]=hashlib.sha256(raw).hexdigest()
    return result


def capture_view(r):
    return {k:v for k,v in r.items() if k not in ('environment','dependencies','commands')}


def sample_view(s):
    result={k:v for k,v in s.items() if k not in ('row','client_boundaries','commands')}
    if 'row' in s:
        result['requests']=[{k:v for k,v in q.items() if k in ('prompt','generated','accepted','offered',
            'reused','read','state_L','state_sha256','decode_cache_restore')} for q in s['row'].get('requests',[])]
    return result


def compact(r):
    result={k:v for k,v in r.items() if k not in ('environment','environments','dependencies','commands',
        'source_review','model_assets','outcomes')}
    result['outcomes']={k:sample_view(v) for k,v in r['outcomes'].items()}
    return result


def parent(container):
    r=json.loads((PARENT_ROOT/'record.json').read_text())
    require(capture_view(r)==container[PARENT_KEY] and r['complete'] and r['passed'] and
        not r['active'] and r['ownership_closed'] and r['fresh_faults']==[] and r['exit_code']==0 and
        r['full_model_hashes_matched']==15,'completed same-native NEO/state model qualification required')
    # Only Main's controller and its CPU protocol test change. Recover their
    # original inputs; all native/kernel/dependency/workload inputs still match.
    for name,h in r['input_sha256'].items():
        raw=model.gate.native.build.git(P,'show',r['source_commit']+':'+name)
        require(hashlib.sha256(raw).hexdigest()==h,'original capture source unrecoverable: '+name)
        if name not in ('tools/sycl/memory_trace_model.py','tools/sycl/memory_trace_protocol_test.py'):
            require(digest(P/name)==h,'native/observer/workload source changed: '+name)
    require(all(digest(p)==h for p,h in r['dependencies'].items()) and
        digest(r['collector']['path'])==r['collector']['sha256'] and
        model.model.fingerprint(r['environment'])==r['environment_sha256'],
        'qualified collector/dependencies/environment changed')
    return r


def runtime(p):
    native=model.gate.engine();preload=model.gate.validation.preload_qualification()
    require(native['binary']==p['binary'] and native['source_commit']==p['native_source_commit'] and
        p['environment']['LD_PRELOAD']==preload['library']['path'] and
        p['environment']['STRATA_UR_PROBE_EXPECTED_LOADER']==preload['loader']['path'],
        'complete native SDK/runtime/preload qualification differs')
    return dict(binary=native['binary'],source_commit=native['source_commit']),preload


def prepare(container):
    require(KEY not in container and not ROOT.exists(),'fresh matched cohort only')
    p=parent(container);native,preload=runtime(p)
    review=container['owner_trace_obligations'][REVIEW_KEY]
    require(p['source_review']==review,'passed logging contract changed')
    controls=container['owner_trace_obligations'][PROTOCOL_KEY]
    require(controls['passed'] and controls['tests']==11 and controls['exit_code']==0 and
        controls['input_fingerprint']==protocol_input_fingerprint(container) and
        all(digest(P/n)==h for n,h in controls['input_sha256'].items()),'current eleven CPU controls required')
    for pattern in review['output_contract']['required_patterns'].values():re.compile(pattern)
    driver=review['installed_driver'];require(digest(driver['path'])==driver['sha256'],'driver DSO changed')
    selected,proof=model.gate.native.source.baseline()
    base=model.model.setup();require(base['models']==p['model_assets'],'model reference changed')
    model.model.model_pins(base['config'],p['model_assets'])
    revision=model.gate.native.build.git(P,'rev-parse','HEAD').decode().strip();source=pins(revision)
    ROOT.mkdir(mode=0o700);environments={}
    for i,audit in enumerate(ORDER):
        private=ROOT/str(i);private.mkdir();(private/'tmp').mkdir();(private/'cache').mkdir()
        environments[str(i)]=audit_environment(p['environment'],private,audit)
    r=dict(stage='prepared',source_commit=revision,input_sha256=source,order=list(ORDER),outcomes={},
        parent_qualification_sha256=model.model.fingerprint(container[PARENT_KEY]),
        source_review_sha256=model.model.fingerprint(review),source_review=review,
        protocol_qualification_sha256=model.model.fingerprint(controls),selected_original=proof,
        runtime_native=native,preload=preload,driver_library=driver,collector=p['collector'],
        dependencies=p['dependencies'],environments=environments,
        environment_sha256={k:model.model.fingerprint(v) for k,v in environments.items()},
        command=p['command'],cases={'small':{'requests':p['cases']['small']['requests']}},vocab_size=p['vocab_size'],
        model_assets=p['model_assets'],model_assets_reference=dict(result_key=PARENT_KEY,sha256=fingerprint(p['model_assets'])),
        driver_before=model.gate.native.guards.driver_guard(),boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        active=False,passed=False,complete=False,ownership_closed=True,GPU_executed=False,
        speed_eligible=False,decode_performance_qualified=False,model_adoption_qualified=False,
        ordinary_runtime_teardown_qualified=False,physical_owner_attribution_qualified=False,
        budgets=dict(wall_s=1200,cpu_s=7200,rss_bytes=104<<30,text_bytes=64<<20,
            file_bytes=64<<20,total_bytes=256<<20,map_read_bytes=1<<20),
        criteria='Four independent native590f processes in ON/OFF/OFF/ON order,same A16385/B33/A+16 requests,original decode configuration and1256 slots. OFF removes only STATE_HASH,SNAPSHOT_VERIFY,CACHE_LOAN_AUDIT before exec;no mid-runtime environment mutation. Preserve safety/preload,UR/NEO logging,queues,model,weights,cache and all other flags. Every process requires exact request extents/valid IDs/actual reuse,QSA12-layer route,complete zero-balance UR ledger,all logging categories/zero binding errors,four stable owning-client and mapped-driver reads,normal exit0,closed ownership and no fresh fault. ON additionally repeats all1256 payload/table/main/draft/running checks;OFF must emit none of those readback-audit markers and is not a new payload/state qualification. Full15 model hashes after the final process;identity/stat pins before/after each. Preserve strict known trailing records,unknown protocol remains rejected. Stop first failure;no retry or skipped arm. No clean speed,decode preservation,physical migration or adoption claim.',
        decision_rule='Keep every individual short-request creation sum,max and DRM/global-free delta. Observer-contribution screen requires max(abs(OFF VRAM delta))+256MiB < min(abs(ON VRAM delta));independently,if both OFF abs deltas>=1GiB,readback audits are not necessary for a large shift. Otherwise inconclusive. These screens do not identify KMD movement,physical PCIe traffic or production throughput. Never change thresholds after observing samples.',
        synchronization_source_review='Refill always sync_queued before optional payload audit and checked residency publication. Native waits ver.wait_commit before DONE and next request. Snapshot restore has its own initial checked sync and checked copies/checkpoint restore;optional verify adds readback only. Keep original source and safety;no queue/context/lifetime change.',
        retention='Main one audit-control comparison. Preserve source/recipe,individual counter and creation samples,ON state checks/OFF limits and faults. Cohort256MiB supervised raw cap,64MiB text per process. Retire successful raw/JIT and previous v3 captures after compact comparison;no extra engine generation.')
    write(ROOT/'record.json',r);container[KEY]=compact(r);write(RESULT,container)
    print('Prepared four-process ON/OFF/OFF/ON residency control;commit before execution.',flush=True)


def admission(r,container):
    require(RESULT.read_bytes()==model.gate.native.build.git(P,'show','HEAD:'+str(RESULT.relative_to(P))) and
        compact(r)==container[KEY] and pins(r['source_commit'])==r['input_sha256'],'commit frozen control plan')
    p=parent(container);native,preload=runtime(p)
    require(native==r['runtime_native'] and preload==r['preload'] and
        model.model.fingerprint(container[PARENT_KEY])==r['parent_qualification_sha256'] and
        model.model.fingerprint(container['owner_trace_obligations'][REVIEW_KEY])==r['source_review_sha256'] and
        model.model.fingerprint(container['owner_trace_obligations'][PROTOCOL_KEY])==r['protocol_qualification_sha256'],
        'native/parent/logging/protocol qualification changed')
    require(all(model.model.fingerprint(v)==r['environment_sha256'][k] for k,v in r['environments'].items()) and
        all(digest(p)==h for p,h in r['dependencies'].items()),'control environment/dependencies changed')
    for asset in (r['collector'],r['driver_library']):require(digest(asset['path'])==asset['sha256'],'observer/driver changed')
    require(model.gate.native.source.baseline()[1]==r['selected_original'] and
        model.gate.native.guards.driver_guard()==r['driver_before'] and
        Path('/proc/sys/kernel/random/boot_id').read_text().strip()==r['boot'],'fallback/boot/driver changed')
    base=model.model.setup();require(base['models']==r['model_assets'],'model reference changed')
    model.model.model_pins(base['config'],r['model_assets'])


def logging_evidence(stdout,stderr,contract):
    require(contract['required_patterns'],'nonempty predeclared NEO output contract required')
    result={}
    required=contract['required_patterns'];optional=contract.get('optional_patterns',{})
    require(not set(required)&set(optional),'overlapping required/optional log categories')
    for category,pattern in {**required,**optional}.items():
        require(0<len(pattern)<=512 and '\n' not in pattern,'bounded log decoder pattern required')
        rows=[]
        for sink,text in (('stdout',stdout),('stderr',stderr)):
            for line in text.splitlines():
                match=re.search(pattern,line)
                if match:
                    require(len(line)<=4096,'NEO record exceeds bounded line size')
                    fields=match.groupdict()
                    require('result' not in fields or int(fields['result'])==0,
                        'NEO binding reports nonzero result: '+line)
                    rows.append((sink,line,fields))
        require(rows or category not in required,'required NEO output missing: '+category)
        result[category]=dict(records=len(rows),required=category in required,
            sinks=sorted({row[0] for row in rows}),representative=rows[0][1] if rows else None,
            first_fields=rows[0][2] if rows else None)
    return dict(categories=result,scope='Observed logging capability only;not complete BO identity/lifetime,UMD-to-BO pointer join,KMD migration or ordinary teardown.')

class AtBoundary:
    def __init__(self,io,entry,r):
        self.io,self.r=io,r
        self.target={k:entry['direct_identity'][k] for k in ('pid','start_ticks','sid')}
    def __getattr__(self,name):return getattr(self.io,name)
    def exit(self,seconds):
        try:return self.io.exit(seconds)
        except RuntimeError as error:
            if str(error)!='unexpected trailing protocol':raise
            require(self.io.eof and self.io.exit_watch.poll(0),
                'trailing logger validation requires completed native exit and pipe EOF')
            raw=bytes(self.io.pending)
            require(0<len(raw)<=512<<10 and raw.endswith(b'\n'),'bounded complete trailing records required')
            contract=self.r['source_review']['output_contract']
            patterns=[re.compile(p) for p in {**contract['required_patterns'],
                **contract.get('optional_patterns',{})}.values()]
            count=0
            for line in raw.decode('utf-8').splitlines():
                line=line.strip()
                if not line:continue
                matches=[m for p in patterns if (m:=p.search(line)) and m.start()==0 and m.end()==len(line)]
                require(matches,'unknown trailing native protocol: '+line[:256])
                require(all('result' not in m.groupdict() or int(m['result'])==0 for m in matches),
                    'trailing NEO binding failed')
                count+=1
            require(count>0,'no recognized trailing NEO records')
            self.r['trailing_neo_records']=dict(records=count,bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(),scope='Known complete logging records after QUIT/EOF;not ordinary runtime teardown qualification.')
            self.io.pending.clear()
            return self.io.exit(seconds)
    def line(self,seconds=700):
        line=self.io.line(seconds)
        if line.startswith(('READY ','DONE ')):
            cpu=time.thread_time_ns();sample=resources.snapshot(self.target,'0000:05:00.0')
            clients=[dict(client_id=c['client_id'],memory_bytes=c['memory_bytes']) for c in sample['clients']]
            require(0<len(clients)<=8 and all('drm-resident-vram0' in c['memory_bytes'] and
                'drm-total-vram0' in c['memory_bytes'] for c in clients),'required client VRAM counters absent')
            require(all(v==0 for c in clients for k,v in c['memory_bytes'].items() if k.startswith('drm-shared-')),
                'shared client counters need another comparison contract')
            ids=sorted(c['client_id'] for c in clients)
            require(not self.r['client_boundaries'] or
                ids==sorted(c['client_id'] for c in self.r['client_boundaries'][0]['clients']),
                'owning DRM client identities changed')
            with open('/proc/'+str(self.target['pid'])+'/maps','rb') as f:
                raw=f.read(self.r['budgets']['map_read_bytes']+1)
            require(0<len(raw)<=self.r['budgets']['map_read_bytes'],'owned module map exceeds budget')
            driver=self.r['driver_library'];inode=Path(driver['path']).stat().st_ino
            matches=[s.split(maxsplit=5) for s in raw.decode().splitlines() if s.endswith(' '+driver['path'])]
            require(matches and all(int(v[4])==inode for v in matches),'pinned driver DSO is not the mapped module')
            self.r['client_boundaries'].append(dict(boundary='READY' if line.startswith('READY ') else 'DONE',
                begin_monotonic_ns=sample['begin_monotonic_ns'],end_monotonic_ns=sample['end_monotonic_ns'],
                reader_thread_cpu_ns=time.thread_time_ns()-cpu,clients=clients,mapped_driver_inode=inode))
        return line

def model_memory_evidence(text):
    require(text.count('activated stream=ur.call format=1 limit=25000')==1,'collector not activated once')
    summary=re.findall(r'^strata UR memory summary: active=1 records=(\d+) invalid=0 unsupported=0 lost=0 ',text,re.M)
    rows=[dict(re.findall(r'(\w+)=([^ ]*)',s)) for s in re.findall(r'^strata UR memory: seq=.*$',text,re.M)]
    require(len(summary)==1 and rows and len(rows)==int(summary[0]) and
        sorted(int(r['seq']) for r in rows)==list(range(1,len(rows)+1)),'memory record conservation failed')
    live={};totals={};frees=0
    for row in sorted(rows,key=lambda v:int(v['seq'])):
        require(row['result']=='0' and row['kind'] in ('alloc','free','release'),
            'failed memory call or unqualified alias/refcount packet')
        pointer=row['ptr']
        if row['kind']=='alloc':
            domain=row['domain'];size=int(row['requested_bytes'])
            require(pointer!='(nil)' and pointer not in live and size>0,'invalid live allocation')
            live[pointer]=(domain,size);item=totals.setdefault(domain,dict(count=0,requested_bytes=0))
            item['count']+=1;item['requested_bytes']+=size
        else:
            require(pointer in live,'free/release without recorded allocation');domain,size=live.pop(pointer)
            require((row['kind']=='release')==(domain=='buffer'),'USM/buffer release kind differs');frees+=1
    require(not live,'requested-byte ledger does not close')
    starts=list(re.finditer(r'^strata trace: request (\d+) \d+$',text,re.M))
    require([int(m[1]) for m in starts]==[16385,33,16401],'actual request framing differs')
    frame=text[starts[1].end():starts[2].start()]
    begin=re.search(r'^strata prefill dense: phase=cache-loan residency=Full activations=0 .*free_bytes=(\d+) total_bytes=(\d+)$',frame,re.M)
    end=re.search(r'^strata prefill dense: phase=restored residency=Full activations=0 .*free_bytes=(\d+) total_bytes=(\d+)$',frame,re.M)
    require(begin and end and begin.end()<end.start() and begin[2]==end[2],'short dense interval missing')
    return dict(records=len(rows),successful_allocations=totals,successful_releases=frees,final_requested_bytes=0,
        short32_interval=dict(before_reported_free_bytes=int(begin[1]),after_reported_free_bytes=int(end[1]),
            decrease_bytes=int(begin[1])-int(end[1]),total_reported_bytes=int(begin[2])),
        scope='Successful logical UR request ledger;not physical residency,causal phase assignment or ordinary teardown.')


def request_delta(sample):
    before,after=sample['client_boundaries'][1:3]
    require(len(before['clients'])==len(after['clients'])==1 and
        before['clients'][0]['client_id']==after['clients'][0]['client_id'],'one stable client delta required')
    a,b=before['clients'][0]['memory_bytes'],after['clients'][0]['memory_bytes']
    require(a['drm-total-vram0']==a['drm-resident-vram0'] and b['drm-total-vram0']==b['drm-resident-vram0'],
        'resident/total VRAM differ;another accounting contract required')
    return dict(vram_bytes=b['drm-total-vram0']-a['drm-total-vram0'],
        gtt_bytes=b['drm-total-gtt']-a['drm-total-gtt'],
        all_total_region_bytes=sum(b[k]-a[k] for k in a if k.startswith('drm-total-')))


def allocation_delta(stdout,contract):
    lines=stdout.splitlines();marks=[i for i,s in enumerate(lines) if s.startswith(('READY ','DONE '))]
    require(len(marks)==4,'four stdout request boundaries required')
    pattern=re.compile(contract['required_patterns']['allocation_pool_size']);groups={}
    for line in lines[marks[1]+1:marks[2]]:
        m=pattern.search(line)
        if not m:continue
        a=m.groupdict();key=(a['type'],a['pool']);g=groups.setdefault(key,dict(type=a['type'],pool=a['pool'],records=0,requested_bytes=0,largest_bytes=0))
        n=int(a['size']);g['records']+=1;g['requested_bytes']+=n;g['largest_bytes']=max(g['largest_bytes'],n)
    return dict(records=sum(g['records'] for g in groups.values()),
        sum_requested_bytes=sum(g['requested_bytes'] for g in groups.values()),
        largest_requested_bytes=max((g['largest_bytes'] for g in groups.values()),default=0),
        types=sorted(groups.values(),key=lambda g:(-g['requested_bytes'],g['type'])),
        scope='Logged creation requests between DONE A and DONE B,not live/physical bytes or causal phase assignment.')


def off_evidence(stderr,row,pid):
    activation=model.gate.validation.preload_evidence(stderr,pid)
    boundary=re.search(r'^strata UR validation: .*phase=before-main[^\n]*\n',stderr,re.M)
    require(boundary,'missing validation boundary');body=stderr[boundary.end():]
    require(not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|teardown uncertain|cleanup failed',body),
        'runtime error after pre-main canary')
    require(not re.search(r'^strata (?:serve: (?:STATE_HASH|SNAPSHOT_VERIFY|DRAFT_PREFILL)|loan audit:)',body,re.M),
        'OFF process still emits readback-audit markers')
    route=[tuple(map(int,m)) for m in re.findall(r'^strata prefill QSA16 streamed: layer=(\d+) chunks=(\d+) rows=(\d+) pos0=(\d+) prefix_H2D_requested_bytes=(\d+) host_publication=GPU_Q8 stage=full_logical_layer checked_drain=PASS$',body,re.M)]
    require(route==[(l,2,16384,0,0) for l in range(3,48,4)],'OFF process actual long route differs')
    require([q['read'] for q in row['requests']]==[16385,33,16] and
        [int(n) for n in re.findall(r'^strata serve: conversation cache: restored (\d+) tokens',body,re.M)]==[16385],
        'OFF process actual native wire/RESTORE behavior differs')
    return dict(activation=activation,actual_qsa_layers=12,readback_audit_markers_absent=True,
        cache_payload_and_snapshot_byte_verification=False,
        limits='Wire/reuse/route and ownership/fault checks only;no new payload/state numerical qualification.')


def run(container,index):
    r=json.loads((ROOT/'record.json').read_text());admission(r,container)
    require(0<=index<len(ORDER) and str(index) not in r['outcomes'] and
        all(r['outcomes'].get(str(i),{}).get('passed') for i in range(index)), 'no retry/skipped control sample')
    original=json.loads(json.dumps(r));owner=None
    journal=subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15)
    cursor=next(s.split(': ',1)[1] for s in journal.splitlines() if s.startswith('-- cursor: '))
    sample=dict(audit=ORDER[index],stage='running',row={},client_boundaries=[],passed=False,complete=False,ownership_closed=False)
    r['outcomes'][str(index)]=sample
    def persist():
        r['active']=bool(owner and owner.active is not None);r['commands']=owner.commands if owner else []
        write(ROOT/'record.json',r)
    owner=model.gate.native.supervision.Owner(ROOT/'owned',persist);r.update(stage='running',GPU_executed=True);persist()
    try:
        b=r['budgets'];context=ChainMap(sample,r)
        entry,out,err=owner.run('sample-'+str(index),r['command'],r['environments'][str(index)],ROOT/str(index),
            interaction=lambda io,e:model.interact(AtBoundary(io,e,context),'small',sample['row'],r),
            wall=b['wall_s'],cpu=b['cpu_s'],rss_cap=b['rss_bytes'],text_cap=b['text_bytes'],file_cap=b['file_bytes'],total_cap=b['total_bytes'])
        sample['exit_code']=entry['exit_code']
        require(model.gate.native.supervision.closed(entry) and entry['exit_code']==0,'normal owned native exit required')
        stdout,stderr=out.read_text(errors='replace'),err.read_text(errors='replace')
        if sample['audit']=='on':
            model.evidence(stderr,'small',sample['row'],r,entry['pid']);sample['readback_state_verified']=True
        else:sample['off_evidence']=off_evidence(stderr,sample['row'],entry['pid']);sample['readback_state_verified']=False
        require(len(sample['client_boundaries'])==4,'four owning-client boundary reads required')
        sample['trace']=model_memory_evidence(stderr)
        observed=logging_evidence(stdout,stderr,r['source_review']['output_contract'])
        sample['logging_counts']={k:v['records'] for k,v in observed['categories'].items()}
        sample['short_request']=dict(client_region_delta=request_delta(sample),
            allocation_details=allocation_delta(stdout,r['source_review']['output_contract']),
            dense_reported_free_decrease_bytes=sample['trace']['short32_interval']['decrease_bytes'])
        sample['logs']={name:dict(path=str(path),bytes=path.stat().st_size,sha256=digest(path)) for name,path in [('stdout',out),('stderr',err)]}
        if index==len(ORDER)-1:
            model.model.model_pins(model.model.setup()['config'],r['model_assets'],full=True)
            r['full_model_hashes_matched']=15
        admission(original,container);sample['host_probe_accepted']=True
    except BaseException as error:sample['error']=repr(error)
    finally:
        model.gate.validation.finish(sample,owner,cursor,r['boot'])
        sample['complete']=sample['passed'];sample['stage']='closed' if sample['passed'] else 'failed'
        r.update(stage='sample_'+str(index)+'_closed' if sample['passed'] else 'failed',
            passed=all(r['outcomes'].get(str(i),{}).get('passed') for i in range(len(ORDER))),
            ownership_closed=owner.active is None and all(model.gate.native.supervision.closed(c) for c in owner.commands))
        r['complete']=bool(r['passed'] and r.get('full_model_hashes_matched')==15)
        if r['complete']:
            on=[abs(r['outcomes'][str(i)]['short_request']['client_region_delta']['vram_bytes']) for i in range(4) if ORDER[i]=='on']
            off=[abs(r['outcomes'][str(i)]['short_request']['client_region_delta']['vram_bytes']) for i in range(4) if ORDER[i]=='off']
            r['comparison']=dict(audit_contribution_screen=max(off)+(256<<20)<min(on),
                large_shift_without_readback_in_both_samples=all(v>=1<<30 for v in off),
                individual_on_vram_delta_magnitudes=on,individual_off_vram_delta_magnitudes=off,
                speed_or_physical_owner_qualified=False)
        persist();container[KEY]=compact(r);write(RESULT,container)
    print(json.dumps(dict(index=index,audit=ORDER[index],passed=sample['passed'],ownership_closed=sample['ownership_closed'],
        short_request=sample.get('short_request'),error=sample.get('error'),fresh_faults=sample.get('fresh_faults'))),flush=True)
    return 0 if sample['passed'] else 1


if __name__=='__main__':
    require(sys.argv[1:] in (['prepare'],['run','0'],['run','1'],['run','2'],['run','3']),'prepare|run 0..3')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);model.gate.native.guards.no_model_owners([])
        container=json.loads(RESULT.read_text())
        if sys.argv[1]=='prepare':prepare(container)
        else:sys.exit(run(container,int(sys.argv[2])))
