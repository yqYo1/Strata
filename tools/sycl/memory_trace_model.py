#!/usr/bin/env python3
"""Main-only bounded NEO allocation logging qualification; never clean speed.

Main-authored Strata diagnostic, repository LICENSE. Reuses the selected
submission fixture and the independently qualified UR memory observer.
"""
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
PARENT_ROOT=B/'ur-memory-client-v1'
PARENT_KEY='ur_memory_client_v1'
FIXTURE_KEY='neo_memory_fixture_v1'
ROOT=B/'neo-memory-fixture-v1'
KEY=FIXTURE_KEY
REVIEW_KEY='neo_logging_source_review'
INPUTS=sorted(set([*device.INPUTS,'tools/sycl/memory_trace_model.py',
    'bench/sycl/client_resources.py','tests/core/test_client_resources.py',
    'tools/sycl/memory_trace_protocol_test.py']))
sys.path.insert(0,str(P/'bench/sycl'))
import client_resources as resources
require,digest,write=host.require,host.digest,host.write
LOGGING_KEYS={'LogAllocationType','LogAllocationMemoryPool','LogAllocationStdout',
    'PrintBOCreateDestroyResult','PrintBOBindingResult','PrintBOPrefetchingResult'}


def pins(revision):
    result={}
    for name,raw in model.gate.native.build.blobs(P,revision,INPUTS):
        require(raw==(P/name).read_bytes(),'commit exact logging input: '+name)
        result[name]=hashlib.sha256(raw).hexdigest()
    return result


def compact(r):
    return {k:v for k,v in r.items() if k not in ('environment','dependencies','commands')}


def parent(container):
    r=json.loads((PARENT_ROOT/'record.json').read_text())
    require(compact(r)==container[PARENT_KEY] and r['complete'] and r['passed'] and
        not r['active'] and r['ownership_closed'] and r['fresh_faults']==[] and
        r['full_model_hashes_matched']==15,'completed model/client observer required')
    for name,h in r['input_sha256'].items():
        raw=model.gate.native.build.git(P,'show',r['source_commit']+':'+name)
        require(hashlib.sha256(raw).hexdigest()==h,'original client source unrecoverable: '+name)
        if name!='tools/sycl/memory_trace_model.py':require(digest(P/name)==h,'native/observer source changed: '+name)
    require(all(digest(p)==h for p,h in r['dependencies'].items()) and
        digest(r['collector']['path'])==r['collector']['sha256'] and
        model.model.fingerprint(r['environment'])==r['environment_sha256'],
        'qualified collector/dependencies/environment changed')
    return r


def runtime(r):
    native=model.gate.engine();preload=model.gate.validation.preload_qualification()
    require(native['binary']==r['binary'] and native['source_commit']==r['native_source_commit'] and
        r['environment']['LD_PRELOAD']==preload['library']['path'] and
        r['environment']['STRATA_UR_PROBE_EXPECTED_LOADER']==preload['loader']['path'],
        'full native SDK/runtime/validation preload differs')
    return dict(binary=native['binary'],source_commit=native['source_commit']),preload


def prepare(container,kind):
    require(KEY not in container and not ROOT.exists(),'fresh logging fixture only')
    review=container['owner_trace_obligations'][REVIEW_KEY]
    require(review['native_review_passed'] and review['logging_only'] and
        review['flag_map'] and set(review['flag_map'])<=LOGGING_KEYS and
        all(v=='1' for v in review['flag_map'].values()),'reviewed logging-only flags required')
    contract=review['output_contract'];required=contract['required_patterns']
    require(required and not set(required)&set(contract.get('optional_patterns',{})),
        'nonempty disjoint logging categories required')
    for pattern in {**required,**contract.get('optional_patterns',{})}.values():
        require(0<len(pattern)<=512 and '\n' not in pattern,'bounded logging pattern required')
        re.compile(pattern)  # Reject invalid decoders before creating/running a plan.
    p=parent(container);native,preload=runtime(p)
    if kind=='model':
        controls=container['owner_trace_obligations']['neo_trailing_protocol_host_v2']
        require(controls['passed'] and controls['tests']==9 and controls['exit_code']==0 and
            controls['contract_encoding']=='json_sorted_compact' and
            controls['contract_sha256']==hashlib.sha256(json.dumps(review['output_contract'],
                sort_keys=True,separators=(',',':')).encode()).hexdigest() and
            all(digest(P/n)==h for n,h in controls['input_sha256'].items()),
            'passed current strict trailing-protocol host controls required')
        q=container[FIXTURE_KEY]
        require(q['passed'] and q['complete'] and not q['active'] and q['ownership_closed'] and
            q['fresh_faults']==[] and q['exit_code']==0 and q['source_review']==review and
            q['runtime_native']==native and q['preload']==preload and q['collector']==p['collector'],
            'passed unchanged actual fixture logging capability required')
        for name,h in q['input_sha256'].items():
            raw=model.gate.native.build.git(P,'show',q['source_commit']+':'+name)
            require(hashlib.sha256(raw).hexdigest()==h,'fixture original source unrecoverable: '+name)
            if name!='tools/sycl/memory_trace_model.py':require(digest(P/name)==h,'qualified fixture input changed: '+name)
        base=model.model.setup();require(base['models']==p['model_assets'],'model reference changed')
        model.model.model_pins(base['config'],p['model_assets'])
    selected,proof=model.gate.native.source.baseline();fixture=selected['fixtures']['sycl_submission_test']
    require(digest(fixture['path'])==fixture['sha256'],'selected fixture changed')
    info=container['owner_trace_obligations']['post_collection_driver_identification']
    # Exact installed DSO identity is a new pre-exec pin, not retrospective
    # qualification of its source mapping or prior runtime teardown.
    driver=review['installed_driver']
    require(driver=={k:info[k] for k in ('path','sha256')},'source review driver identity differs')
    require(digest(driver['path'])==driver['sha256'],'observed installed driver changed')
    binary=Path(driver['path']).read_bytes()
    require(all(k.encode()+b'\0' in binary for k in review['flag_map']),'logging key absent in installed driver')
    revision=model.gate.native.build.git(P,'rev-parse','HEAD').decode().strip();source=pins(revision)
    ROOT.mkdir(mode=0o700)
    for name in ('tmp','cache'):(ROOT/name).mkdir(mode=0o700)
    environment=dict(p['environment'],TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'cache'),**review['flag_map'])
    changed={k for k in set(environment)|set(p['environment']) if environment.get(k)!=p['environment'].get(k)}
    require(changed<=set(review['flag_map'])|{'TMPDIR','XDG_CACHE_HOME'},'unplanned runtime environment change')
    r=dict(stage='prepared',source_commit=revision,input_sha256=source,
        parent_qualification_sha256=model.model.fingerprint(container[PARENT_KEY]),
        source_review_sha256=model.model.fingerprint(review),source_review=review,
        selected_original=proof,runtime_native=native,preload=preload,
        fixture=fixture,fixture_source_commit=selected['source_commit'],
        collector=p['collector'],dependencies=p['dependencies'],
        driver_library=driver,environment=environment,environment_sha256=model.model.fingerprint(environment),
        command=[fixture['path'],'--submission'],cwd=str(ROOT),
        driver_before=model.gate.native.guards.driver_guard(),boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        active=False,passed=False,complete=False,ownership_closed=True,GPU_executed=False,
        speed_eligible=False,physical_owner_attribution_qualified=False,ordinary_runtime_teardown_qualified=False,
        model_adoption_qualified=False,decode_performance_qualified=False,
        budgets=dict(wall_s=120,cpu_s=120,rss_bytes=4<<30,text_bytes=16<<20,file_bytes=16<<20,total_bytes=64<<20),
        criteria='One selected original10-check submission fixture,no model or engine rebuild. Preserve the complete immutable qualified safety/preload/UR observer environment;add only reviewed logging keys and private temporary/cache paths before exec. Pin original source/recipe,full SDK,runtime,collector239 dependencies,driver DSO and fixture. Require the original10 numerical/copy/event/profiling/lifetime checks,complete UR memory ledger,each predeclared minimum NEO log category,normal direct native exit0,closed ownership,unchanged inputs/boot/driver and no fresh fault. Capture stdout/stderr in private cwd under finite budgets. Stop first failure;no retry. NEO memory pool names describe creation decisions,not necessarily current KMD placement;UMD logs alone do not establish migration,physical residency,PCIe traffic or speed. Exit summary and ordinary teardown are not required or qualified.',
        retention='Main consumes small output to select one finite same-native model capture. Keep compact sample counts,representative output,identities and original outcome;retire raw logs/JIT after that plan or refusal. No extra engine generation.')
    r['kind']=kind
    if kind=='model':
        r.update(command=p['command'],binary=p['binary'],native_source_commit=p['native_source_commit'],
            cases=p['cases'],vocab_size=p['vocab_size'],model_assets=p['model_assets'],row={},client_boundaries=[],
            fixture_qualification_sha256=model.model.fingerprint(container[FIXTURE_KEY]),
            trailing_protocol_qualification_sha256=model.model.fingerprint(controls),
            budgets=dict(wall_s=1200,cpu_s=7200,rss_bytes=104<<30,text_bytes=64<<20,
                file_bytes=64<<20,total_bytes=256<<20,map_read_bytes=1<<20),
            criteria='One same-native590f A16385/B33/A+16 wire/model-state capture,no benchmark or engine rebuild. Require the committed passed selected-fixture logging gate and unchanged reviewed five logging keys,UR observer,full SDK/preload safety and model inputs. Preserve all1256 expert payloads/publications and main/draft/running snapshot checks;full15 model digests after collection. Require bounded4 READY/DONE owning-PID/start/SID resource reads,stable nonempty clients,total/resident VRAM exports,zero shared counters and actual mapped pinned driver DSO. Require complete successful UR allocation/free/simple-buffer ledger with zero final requested balance and all predeclared NEO categories with binding result0. After QUIT accept only bounded complete known NEO records following confirmed exit/pipe EOF;reject other protocol,unknown/truncated data or nonzero binding results. Native normal exit0,closed ownership,unchanged identities/boot/driver and no fresh fault. Stop first failure,no retry. Global-free internal dense interval and whole-request client/log boundaries differ;no causal phase,complete BO ledger,migration,physical residency,PCIe traffic,speed,decode-performance,ordinary teardown or adoption claim.',
            retention='Main immediate allocation-type/BO lifecycle comparison against committed earlier UR/client changes. Keep compact request/state,conservation,counter deltas,type/pool summaries and limitations. Raw stdout/stderr only for this bounded offline analysis,64MiB combined;retire logs/JIT and prior fulltrace after useful evidence is committed. No extra engine generation.')
    write(ROOT/'record.json',r);container[KEY]=compact(r);write(RESULT,container)
    print('Prepared bounded '+kind+' NEO logging gate;commit before run.',flush=True)


def admission(r,container):
    require(RESULT.read_bytes()==model.gate.native.build.git(P,'show','HEAD:'+str(RESULT.relative_to(P))) and
        compact(r)==container[KEY] and pins(r['source_commit'])==r['input_sha256'],
        'commit exact logging capability plan')
    p=parent(container);native,preload=runtime(p)
    require(native==r['runtime_native'] and preload==r['preload'] and
        model.model.fingerprint(container[PARENT_KEY])==r['parent_qualification_sha256'] and
        model.model.fingerprint(container['owner_trace_obligations'][REVIEW_KEY])==r['source_review_sha256'],
        'native/preload/parent/source review qualification changed')
    require(model.model.fingerprint(r['environment'])==r['environment_sha256'] and
        all(digest(p)==h for p,h in r['dependencies'].items()),'logging environment/dependencies changed')
    for asset in (r['fixture'],r['collector'],r['driver_library']):
        require(digest(asset['path'])==asset['sha256'],'logging artifact changed')
    if r['kind']=='model':
        require(model.model.fingerprint(container['owner_trace_obligations']['neo_trailing_protocol_host_v2'])==
            r['trailing_protocol_qualification_sha256'],'strict protocol host controls changed')
        require(model.model.fingerprint(container[FIXTURE_KEY])==r['fixture_qualification_sha256'],
            'actual fixture qualification changed')
        base=model.model.setup();require(base['models']==r['model_assets'],'model reference changed')
        model.model.model_pins(base['config'],r['model_assets'])
    require(model.gate.native.source.baseline()[1]==r['selected_original'] and
        model.gate.native.guards.driver_guard()==r['driver_before'] and
        Path('/proc/sys/kernel/random/boot_id').read_text().strip()==r['boot'],
        'selected baseline/driver/boot changed')


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


def run(container):
    r=json.loads((ROOT/'record.json').read_text());admission(r,container)
    require(r['stage']=='prepared','closed logging gate cannot be retried')
    original=json.loads(json.dumps(r));owner=None
    journal=subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15)
    cursor=next(s.split(': ',1)[1] for s in journal.splitlines() if s.startswith('-- cursor: '))
    def persist():
        r['active']=bool(owner and owner.active is not None)
        write(ROOT/'record.json',dict(r,commands=owner.commands if owner else []))
    owner=model.gate.native.supervision.Owner(ROOT/'owned',persist)
    r.update(stage='running',GPU_executed=True);persist()
    try:
        interaction=(lambda io,e:model.interact(AtBoundary(io,e,r),'small',r['row'],r)) if r['kind']=='model' else None
        b=r['budgets'];entry,out,err=owner.run(r['kind'],r['command'],r['environment'],Path(r['cwd']),
            interaction=interaction,
            wall=b['wall_s'],cpu=b['cpu_s'],rss_cap=b['rss_bytes'],text_cap=b['text_bytes'],
            file_cap=b['file_bytes'],total_cap=b['total_bytes'])
        r['exit_code']=entry['exit_code']
        require(model.gate.native.supervision.closed(entry) and entry['exit_code']==0,'normal owned direct native exit required')
        stdout,stderr=out.read_text(errors='replace'),err.read_text(errors='replace')
        if r['kind']=='fixture':r['device_evidence']=device.evidence(stderr,stdout,entry['pid'])
        else:
            model.evidence(stderr,'small',r['row'],r,entry['pid'])
            require(len(r['client_boundaries'])==4,'four client boundary reads required')
            r['trace']=model_memory_evidence(stderr)
            model.model.model_pins(model.model.setup()['config'],r['model_assets'],full=True)
            r['full_model_hashes_matched']=15
        r['logging_evidence']=logging_evidence(stdout,stderr,r['source_review']['output_contract'])
        admission(original,container);r['host_probe_accepted']=True
    except BaseException as error:r['error']=repr(error)
    finally:
        model.gate.validation.finish(r,owner,cursor,r['boot'])
        r['passed']=bool(r['passed'] and r.get('driver_after')==r['driver_before'])
        r['complete']=r['passed'];r['stage']='closed' if r['passed'] else 'failed';persist()
        container[KEY]=compact(r);write(RESULT,container)
    print(json.dumps({k:r.get(k) for k in ('passed','complete','ownership_closed','exit_code','logging_evidence','error','fresh_faults')}),flush=True)
    return 0 if r['passed'] else 1


def analyze(container):
    """Bounded offline allocation taxonomy; no GPU/runtime initialization."""
    r=json.loads((ROOT/'record.json').read_text())
    require(RESULT.read_bytes()==model.gate.native.build.git(P,'show','HEAD:'+str(RESULT.relative_to(P))) and
        compact(r)==container[KEY],'commit the original closed capture first')
    require(r['passed'] and r['complete'] and r['ownership_closed'] and not r['active'] and
        r['exit_code']==0 and r['fresh_faults']==[] and r['full_model_hashes_matched']==15,
        'completed same-native model capture required')
    target='neo_client_memory_analysis_v3'
    require(target not in container['owner_trace_obligations'],'no analysis overwrite')
    revision=model.gate.native.build.git(P,'rev-parse','HEAD').decode().strip();source=pins(revision)
    for name,h in r['input_sha256'].items():
        raw=model.gate.native.build.git(P,'show',r['source_commit']+':'+name)
        require(hashlib.sha256(raw).hexdigest()==h,'original capture source unrecoverable: '+name)
        if name!='tools/sycl/memory_trace_model.py':require(digest(P/name)==h,'capture dependency source changed: '+name)
    logs={}
    for name in ('stdout','stderr'):
        path=ROOT/'owned'/('model.'+name);require(path.stat().st_size<=r['budgets']['text_bytes'],'bounded captured output')
        raw=path.read_bytes();logs[name]=dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),text=raw.decode())
    stdout,stderr=logs['stdout']['text'],logs['stderr']['text']
    require(logging_evidence(stdout,stderr,r['source_review']['output_contract'])==r['logging_evidence'] and
        model_memory_evidence(stderr)==r['trace'],'captured compact output evidence changed')
    row=json.loads(json.dumps(r['row']));model.evidence(stderr,'small',row,r,r['commands'][0]['pid'])
    require(row==r['row'],'original request/state evidence changed')
    lines=stdout.splitlines();boundaries=[i for i,line in enumerate(lines) if line.startswith(('READY ','DONE '))]
    require(len(boundaries)==4 and lines[boundaries[0]]=='READY 262144 stop' and
        all(lines[i].startswith('DONE ') for i in boundaries[1:]),'captured protocol boundaries differ')
    pattern=re.compile(r['source_review']['output_contract']['required_patterns']['allocation_pool_size'])
    all_rows=[]
    for line_no,line in enumerate(lines):
        match=pattern.search(line)
        if match:
            fields=match.groupdict();all_rows.append(dict(line=line_no,type=fields['type'],
                pool=fields['pool'],requested_bytes=int(fields['size'])))
    require(len(all_rows)==r['logging_evidence']['categories']['allocation_pool_size']['records'],
        'allocation detail row conservation failed')
    short=[a for a in all_rows if boundaries[1]<a['line']<boundaries[2]]
    groups={}
    for a in short:
        g=groups.setdefault((a['type'],a['pool']),dict(type=a['type'],pool=a['pool'],records=0,
            sum_requested_bytes=0,largest_requested_bytes=0))
        g['records']+=1;g['sum_requested_bytes']+=a['requested_bytes']
        g['largest_requested_bytes']=max(g['largest_requested_bytes'],a['requested_bytes'])
    before,after=r['client_boundaries'][1:3]
    require(len(before['clients'])==len(after['clients'])==1 and
        before['clients'][0]['client_id']==after['clients'][0]['client_id'],'one same client comparison required')
    old,new=before['clients'][0]['memory_bytes'],after['clients'][0]['memory_bytes']
    delta={k:new[k]-old[k] for k in old if k.startswith(('drm-total-','drm-resident-'))}
    result=dict(source_commit=revision,input_sha256=source,collection_source_commit=r['source_commit'],
        capture_logs={k:{n:v[n] for n in ('path','bytes','sha256')} for k,v in logs.items()},
        short_request=dict(stdout_interval='after DONE A16385 through before DONE B33',
            allocation_detail_records=len(short),sum_requested_allocation_bytes=sum(a['requested_bytes'] for a in short),
            largest_requested_allocation_bytes=max((a['requested_bytes'] for a in short),default=0),
            allocation_types=sorted(groups.values(),key=lambda a:(-a['sum_requested_bytes'],a['type'],a['pool'])),
            owning_client_region_delta_bytes=delta,
            all_total_region_delta_bytes=sum(v for k,v in delta.items() if k.startswith('drm-total-')),
            internal_dense_reported_free_decrease_bytes=r['trace']['short32_interval']['decrease_bytes']),
        UR_requested_byte_ledger=r['trace'],decode_state_restore_verified=True,
        physical_owner_attribution_qualified=False,causal_phase_attribution_qualified=False,
        decode_performance_qualified=False,speed_eligible=False,
        limits='Allocation-detail sums are logged creation requests,not live or physical bytes;pool labels are creation decisions. Stdout boundaries differ from inner stderr dense boundaries and callback timing. BO create coverage is partial,close text is an attempt,no complete NEO-allocation/BO join or KMD migration proof. State/cache readback audits and observers are enabled;these diagnostic observations do not establish the production speed or memory-pressure effect. Preserve original failed v1 observer and UNRUN v2 preparation outcomes.')
    container['owner_trace_obligations'][target]=result;write(RESULT,container)
    print(json.dumps(result['short_request']),flush=True)


if __name__=='__main__':
    require(sys.argv[1:] in (['prepare'],['run'],['prepare-model'],['run-model'],['analyze-model']),
        'prepare|run|prepare-model|run-model|analyze-model')
    kind='model' if sys.argv[1].endswith('-model') else 'fixture'
    if kind=='model':ROOT=B/'neo-memory-model-v3';KEY='neo_memory_model_v3'
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);model.gate.native.guards.no_model_owners([])
        container=json.loads(RESULT.read_text())
        if sys.argv[1].startswith('prepare'):prepare(container,kind)
        elif sys.argv[1]=='analyze-model':analyze(container)
        else:sys.exit(run(container))
