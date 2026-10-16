#!/usr/bin/env python3
"""Main-owned actual per-format CPU service ledger on qualified FP32/4096 placement."""
import argparse, fcntl, hashlib, importlib.util, json, math, os, re, time
from pathlib import Path
P=Path('/home/yayoi/orca/workspaces/Strata/sycl')
A=P/'bench/results/2026-10-15-pr6-recovery-audit'
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
DEVICE=Path('/sys/bus/pci/devices/0000:05:00.0')
OWNER=P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'
OWNER_SHA='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
BINARY_SHA='b6838d286a6ec00b0d92c7ce8f8480c81511c4ab823e400273c52b2585c55aa6'
FAULT=re.compile(r'(?i)(\bxe\b.*(fault|hang|reset|error|wedg|timeout|stuck)|GPU.*(fault|hang|wedg|reset)|GuC.*(fault|hang|reset|error))')
def require(v,s):
    if not v: raise RuntimeError(s)
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def load(p): return json.loads(Path(p).read_text())
def good(d): return d.get('passed') and d.get('complete') and not d.get('active')
def kv(s): return dict(re.findall(r'(\w+)=([^ ]+)',s))
def integer(d,k): return int(d[k])
def ledger(text,done,coupled):
    lines=text.splitlines(); require(len(text.encode())<=98304,'ledger output budget')
    h=kv(lines[0]); f=kv(lines[-1]); require(h['schema']=='mtp4-route-service-v2','schema')
    require(integer(h,'coupled')==coupled and h['request']==f['request'],'request join')
    for k in ['complete','coverage_complete','service_complete','output_bounded','snapshot_before','snapshot_after','done_counts_available','commit_gpu_qualified','terminal_seen']:
        require(integer(f,k)==1,'footer '+k)
    for k in ['invalid','overflow','omitted_rows']: require(integer(f,k)==0,'footer '+k)
    c=kv(next(s for s in lines if s.startswith('coverage ')))
    W,R,D,acc,C,E=[integer(c,k) for k in ['W','R','D','A','C','E']]
    require(R-W==D and W+acc==C and E<=C and E==done['generated'],'coverage algebra')
    require(integer(c,'attempted')==integer(c,'verified')==W and integer(c,'run_rows')==R,'verified windows')
    require(integer(c,'route_entries')==R*48*10,'expert token rows denominator')
    require(integer(h,'layers')==48 and integer(h,'k')==10 and integer(h,'workers')==5 and integer(h,'host')==1,'shape')
    require(integer(h,'prompt')==done['prompt'] and h['status']==done['finish'],'DONE identity')
    require(acc==done['accepted'] and D==done['offered'],'draft counters')
    dispatch=[kv(s) for s in lines if s.startswith('dispatch ')]
    require(sum(integer(x,'rows') for x in dispatch)==R*48,'dispatch rows')
    require(sum(integer(x,'entries') for x in dispatch)==R*48*10,'dispatch entries')
    for x in dispatch: require(x['attempted']==x['completed'],'dispatch complete')
    owners=[kv(s) for s in lines if s.startswith('owner ')]
    require(sum(integer(x,'entries') for x in owners)==R*48*10,'owner entries')
    require(integer(owners[0],'entries')==integer(c,'cpu_rows') and integer(owners[0],'groups')==integer(c,'cpu_jobs'),'CPU owners')
    hist=[kv(s) for s in lines if s.startswith('window_hist ')]
    for key in ['W','R','D','A','C','E']:
        require(sum(integer(x,key) for x in hist)==integer(c,key),'hist '+key)
    jobs=[kv(s) for s in lines if s.startswith('cpu_job_nt ')]
    require(sum(integer(x,'jobs') for x in jobs)==integer(c,'cpu_jobs'),'job join')
    require(sum(integer(x,'rows') for x in jobs)==integer(c,'cpu_rows'),'row join')
    phases=[kv(s) for s in lines if s.startswith('native_phase ')]
    require(len(phases)==2,'phases')
    for x in phases:
        require(x['planned']==x['complete']==x['done_boundaries']==x['park_boundaries'],'phase boundaries')
        require(x['tasks_planned']==x['claimed']==x['completed'],'phase tasks')
        require(x['rows_planned']==x['rows_claimed']==x['rows_completed'],'phase rows')
    services=[kv(s) for s in lines if s.startswith('service ')]
    for x in services:
        require(integer(x,'invalid')==integer(x,'overflow')==0,'service clocks')
        require(integer(x,'cpu_ns')>=0 and integer(x,'wall_ns')>=0,'service nonnegative')
    require(sum(integer(x,'calls') for x in services if x['group']=='issuer' and integer(x,'role') in [1,4])==integer(c,'cpu_rows'),'quant operation count')
    for x in phases:
        require(integer(x,'rows_planned')==integer(c,'cpu_jobs')*(640 if x['ph']=='0' else 2560),'native row operations')
    # Concurrent thread CPU consumption is a capacity lower bound, never exclusive wall.
    cpu=sum(integer(x,'cpu_ns') for x in services)
    return dict(header=h,footer=f,coverage=c,lines=lines,
        cpu_expert_row_fraction=integer(c,'cpu_rows')/integer(c,'route_entries') if R else None,
        cpu_service_sum_ns=cpu,expected_service_capacity_lower_bound_ms=cpu/6/1e6,
        service_bound_is_not_critical_path=True,existing_ms=[kv(s) for s in lines if s.startswith('existing_ms ')])
def format_service(joined):
    lines=joined['lines']; header=joined['header']; coverage=joined['coverage']
    require(integer(header,'service_format')==1,'format marker')
    formats=[kv(s) for s in lines if s.startswith('service_format ')]
    counters=[kv(s) for s in lines if s.startswith('service_format_counter ')]
    closes=[kv(s) for s in lines if s.startswith('service_format_close ')]
    require(len(closes)==1,'one format close'); close=closes[0]
    require(integer(close,'complete')==1,'format complete')
    for key in ['invalid','overflow','pending']: require(integer(close,key)==0,'format '+key)
    require(integer(close,'dispatches')==integer(coverage,'dispatch_attempts')==integer(coverage,'W')*48,'format dispatch conservation')
    require(len(formats)==integer(close,'pairs') and 1<=len(formats)<=16,'format bound')
    pairs=[integer(f,'pair') for f in formats]
    require(pairs==list(range(len(formats))),'format pair uniqueness')
    for f in formats:
        require(0<=integer(f,'representative_layer')<48,'format representative')
        for key in ['gu_row','down_row','act_bytes','h_bytes']: require(integer(f,key)>0,'format geometry '+key)
    seen=set(); totals={}
    per_pair={pair:dict(cpu_ns=0,wall_ns=0,GU_cpu_ns=0,Down_cpu_ns=0,quant_cpu_ns=0) for pair in pairs}
    for f in counters:
        pair=integer(f,'pair'); group=f['group']; role=integer(f,'role')
        require(pair in pairs and group in ['issuer','workers'] and 0<=role<6,'format counter identity')
        key=(pair,group,role);require(key not in seen,'format counter unique');seen.add(key)
        require(integer(f,'calls')>0 and integer(f,'cpu_ns')>=0 and integer(f,'wall_ns')>=0,'format clock counts')
        total=totals.setdefault((group,role),dict(calls=0,cpu_ns=0,wall_ns=0))
        for name in total: total[name]+=integer(f,name)
        p=per_pair[pair];p['cpu_ns']+=integer(f,'cpu_ns');p['wall_ns']+=integer(f,'wall_ns')
        p['GU_cpu_ns' if role in [0,3] else 'Down_cpu_ns' if role in [2,5] else 'quant_cpu_ns']+=integer(f,'cpu_ns')
    old={}
    for line in lines:
        if not line.startswith('service '):continue
        f=kv(line);key=(f['group'],integer(f,'role'));require(key not in old,'service unique')
        old[key]={name:integer(f,name) for name in ['calls','cpu_ns','wall_ns']}
    require(totals==old,'all format/group/role totals equal existing service')
    require(sum(p['cpu_ns'] for p in per_pair.values())==joined['cpu_service_sum_ns'],'format total CPU')
    return dict(complete=True,formats=formats,counters=counters,close=close,per_pair=per_pair,
        scope='Format marginals only, no format-by-nt time claim. Concurrent CPU thread service, not critical wall.')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('mode',choices=['normal','coupled']); a=ap.parse_args()
    coupled=int(a.mode=='coupled'); O=B/('main-primary-decode-format-service-'+a.mode+'-root-v1')
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        build=load(A/'boundary-service-build-v1.json'); require(good(build),'build gate')
        for size in ['small','long-fixed']:
            g=load(A/('boundary-audit-'+size+'-v1.json'))
            require(good(g) and not g['fresh_faults'] and g['boot_unchanged'] and not g['driver_dump_exists'],'prerequisite '+size)
            require(g['binary_sha256']==BINARY_SHA and g['source_pins']==build['source_pins'],'gate identity')
        exe=Path(build['build_path'])/'strata'
        require(sha(exe)==build['binaries']['strata']==BINARY_SHA and sha(OWNER)==OWNER_SHA,'binary/owner pins')
        for n,h in build['source_pins'].items(): require(sha(P/n)==h,'source '+n)
        require((DEVICE/'driver').resolve().name=='xe' and not (DEVICE/'devcoredump').exists(),'driver/dump')
        boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        require(boot==g['boot_id'],'prerequisite same boot')
        natural=load(A/'natural-layout-long-pair-v2.json'); storage=load(A/'layer-major-storage-small-v1.json')
        require(good(natural) and good(storage),'input/environment gate')
        args=list(natural['engine_args']); tp=Path(args[args.index('--tokens-file')+1]); raw=tp.read_bytes()
        require(sha(tp)==natural['input']['tokens_sha256'],'input pin')
        ids=[int(x) for x in raw.decode().strip().split(',')]; require(len(ids)==34603,'actual context')
        shortp=Path(storage['engine_args'][storage['engine_args'].index('--tokens-file')+1]); require(sha(shortp)==storage['tokens_sha256'],'short pin')
        short=[int(x) for x in shortp.read_text().strip().split(',')]; require(len(short)==513,'short tokens')
        for key in ['--tokens-file','--max-new']:
            i=args.index(key); del args[i:i+2]
        args+=['--serve','--coupled-draft' if coupled else '--no-coupled-draft']
        for key,value in {'--prompt-cache':'0','--turn-token':'-1','--short-read':'0','--prefill':'4096','--spec':'4','--suffix-draft':'0','--expert-cache':'128','--pool-workers':'5'}.items():
            require(args[args.index(key)+1]==value,'argument '+key)
        require('--no-spec-split' in args and '--expert-cache-per-layer' in args,'routing')
        env=dict(storage['environment'])
        for k,v in {'UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD':'1','EnableDirectSubmission':'0','STRATA_SYCL_PREFILL_SAME_QUEUE':'1','STRATA_SYCL_VERIFY_NATIVE_CAPTURE':'0'}.items(): require(env[k]==v,'safe env '+k)
        env.update(STRATA_DECODE_ROUTE_LEDGER='1',STRATA_DECODE_SERVICE_CPU='1',STRATA_PREFILL_LAYER_MAJOR='1',STRATA_PREFILL_LAYER_MAJOR_R_DTYPE='fp32',STRATA_PREFILL_LAYER_MAJOR_R_GPU='4096',STRATA_PREFILL_RELEASE_MTP_EXPERTS='1',STRATA_PREFILL_BOUNDARY_AUDIT='0')
        for k in ['STRATA_PREFILL_LOGITS','STRATA_PREFILL_LOGITS_POSITIONS','STRATA_DUMP_FIRST_LOGITS','STRATA_PREFILL_BOUNDARY_PATH','STRATA_PREFILL_BOUNDARY_ROWS']: env.pop(k,None)
        require(not O.exists(),'single use root'); O.mkdir(mode=0o700); env['XDG_CACHE_HOME']=str(O/'cache')
        identity_receipts={n:sha(A/n) for n in ['boundary-service-build-v1.json','boundary-audit-small-v1.json','boundary-audit-long-fixed-v1.json','natural-layout-long-pair-v2.json','layer-major-storage-small-v1.json']}
        driver_identity={str(p):p.read_text().strip() for p in [Path('/proc/sys/kernel/osrelease'),DEVICE/'uevent'] if p.is_file()}
        r=dict(identity_receipts=identity_receipts,driver_identity=driver_identity,active=True,passed=False,complete=False,executed=True,mode=a.mode,requests=[],commands=[],source_pins=build['source_pins'],binary_sha256=sha(exe),recipe_sha256=sha(__file__),owner_sha256=sha(OWNER),input_sha256=sha(tp),short_input_sha256=sha(shortp),boot_id=boot,argv=[str(exe),*args],environment=env,limits=dict(model_wall_seconds=3000,request_wall_seconds=1800,RSS_bytes=88<<30,text_bytes=64<<20,job_bytes=512<<20,ledger_bytes=98304,evidence_bytes=256<<10),performance_qualified=False,quality_qualified=False,physical_262144_qualified=False,adopted=False,configuration='FP32 residuals, fixed4096 GPU rows, validated long-fixed gate; automatic larger-prefix gate remains FAIL',proposal_recipe_sha256='a41b1957faf1a8cf0698d6ebd9d0d7af530615e744f1f173119e67e186280894')
        def persist(): (O/'record.json').write_text(json.dumps(r,indent=2)+'\n')
        spec=importlib.util.spec_from_file_location('owner',OWNER); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        owner=m.Owner(O/'owned',persist); r['commands']=owner.commands; persist()
        def utility(label,argv):
            e,o,s=owner.run(label,argv,build['environment'],P,wall=30,total_cap=512<<20)
            require(m.closed(e) and e['exit_code']==0,'utility '+label); return o.read_text()
        cursor=None
        try:
            cursor=re.search(r'-- cursor: (\S+)',utility('journal-before',['journalctl','-k','-b','-n','0','--show-cursor','--no-pager']))[1]
            r['journal_before_cursor']=cursor
            def interaction(io,entry):
                while True:
                    line=io.line(1800)
                    if line.startswith('READY '):
                        require(line=='READY 262144 stop','READY exact')
                        print('decode ledger '+a.mode+': model READY',flush=True)
                        break
                for index,tokens in enumerate([ids,ids,short]):
                    n=128 if index<2 else 8
                    print('decode ledger '+a.mode+': request '+str(index)+' input='+str(len(tokens))+' max_new='+str(n),flush=True)
                    wire=('GEN '+str(n)+' temperature='+('0.7' if coupled else '0')+' seed=42 '+','.join(map(str,tokens))+'\n').encode()
                    require(len(wire)<1<<20,'stdin cap'); io.send(wire)
                    got=[]; progress=[]; deadline=time.monotonic()+1800
                    while True:
                        remaining=deadline-time.monotonic(); require(remaining>0,'whole request wall deadline')
                        line=io.line(remaining)
                        if re.fullmatch(r'T \d+',line): got.append(int(line[2:])); require(len(got)<=n,'output cap')
                        elif line.startswith('DONE '):
                            f=line.split(); require(len(f)==15,'DONE exact field count')
                            d=dict(generated=int(f[1]),prompt=int(f[2]),prompt_ms=float(f[3]),decode_ms=float(f[4]),finish=f[5],accepted=int(f[6]),offered=int(f[7]),reused=int(f[8]),read=int(f[14]))
                            require(d['generated']==len(got) and d['prompt']==len(tokens) and d['reused']==0 and d['read']==len(tokens),'DONE tokens')
                            require(d['finish'] in ['length','stop'] and all(math.isfinite(d[k]) and d[k]>=0 for k in ['prompt_ms','decode_ms']),'DONE normal')
                            require(d['finish']!='length' or len(got)==n,'length output')
                            r['requests'].append(dict(index=index,done=d,generated_ids=got,wire_sha256=hashlib.sha256(wire).hexdigest(),wire_bytes=len(wire),coverage_sufficient=len(got)>=n,progress=progress)); persist()
                            print('decode ledger '+a.mode+': request '+str(index)+' DONE '+json.dumps(d),flush=True)
                            break
                        elif re.fullmatch(r'(?:RESUME \d+|REUSED \d+|PP \d+ \d+ [\d.]+ [\d.]+)',line):
                            if len(progress)<16: progress.append(line)
                        else: raise RuntimeError('unexpected protocol '+line[:200])
                io.send(b'QUIT\n'); io.exit(60)
            e,out,err=owner.run('serve',[str(exe),*args],env,P,interaction=interaction,wall=3000,cpu=3000*len(os.sched_getaffinity(0)),rss_cap=88<<30,text_cap=64<<20,total_cap=512<<20,file_cap=64<<20)
            require(m.closed(e) and e['exit_code']==0,'owned normal closure')
            text=err.read_text(); require(not re.search(r'(?:UR|ZE)_RESULT_ERROR_|terminate called|what\(\):|Strata SYCL cleanup:|retirement failed|stopping with owners live',text),'runtime fault')
            blocks=re.findall(r'^strata decode joined begin .*?^strata decode joined close [^\n]*\n?',text,re.M|re.S)
            require(len(blocks)==3,'three joined requests')
            for req,block in zip(r['requests'],blocks):
                req['ledger']=ledger(block,req['done'],coupled)
                req['format_service']=format_service(req['ledger'])
            # Before-detach starts each long request segment; short must add none.
            starts=[m.start() for m in re.finditer(r'^strata mtp: prompt expert phase=before-detach ',text,re.M)]
            require(len(starts)==2,'exactly two long detaches, zero short detaches')
            closes=[text.index(b)+len(b) for b in blocks]
            require(starts[0]<closes[0]<starts[1]<closes[1],'lifetime ordering')
            for i in range(3):
                segment=text[(0 if i==0 else closes[i-1]):closes[i]]
                marks=re.findall(r'prompt expert phase=([\w-]+) free_bytes=(\d+) total_bytes=(\d+)',segment)
                require([x[0] for x in marks]==(['before-detach','after-detach','after-prompt-before-restore','after-restore'] if i<2 else []),'memory mark order')
                captures=re.findall(r'expert decode recapture kind=(round|step) coupled=([01]) ok=([01]) ms=([\d.]+) cumulative_ms=([\d.]+)',segment)
                require(all(int(x[1])==coupled and x[2]=='1' for x in captures),'recapture success')
                if i<2:
                    require(captures,'long graph recapture coverage')
                    require(re.search(r'experts restored bytes=707788800,',segment),'checked restore receipt')
                    require(int(marks[1][1])-int(marks[0][1])>=707788800-(8<<20),'detach bytes')
                    require(abs(int(marks[2][1])-int(marks[3][1])-707788800)<=32<<20,'restore bytes')
                r['requests'][i].update(memory_marks=marks,recaptures=captures)
            r['diagnostic_excerpt']=[s for s in text.splitlines() if s.startswith(('strata mtp:','strata trace: layer-major'))]
            require(len(json.dumps(r['requests']).encode())+len(json.dumps(r['diagnostic_excerpt']).encode())<=256<<10,'compact evidence budget')
            for n,h in build['source_pins'].items(): require(sha(P/n)==h,'post source pin')
            require(sha(exe)==BINARY_SHA,'post binary pin')
            r.update(passed=True,complete=True,coverage_sufficient=all(x['coverage_sufficient'] for x in r['requests']),repeated_lifecycle_observed=True)
        except BaseException as ex: r['error']=repr(ex); raise
        finally:
            try:
                if cursor:
                    rows=[json.loads(v) for v in utility('journal-final',['journalctl','-k','-b','--after-cursor',cursor,'--no-pager','-o','json']).splitlines() if v.strip()]
                    r['fresh_faults']=[v for v in rows if FAULT.search(str(v.get('MESSAGE','')))]; r['driver_dump_exists']=(DEVICE/'devcoredump').exists(); r['boot_unchanged']=Path('/proc/sys/kernel/random/boot_id').read_text().strip()==boot
                    if r['fresh_faults'] or r['driver_dump_exists'] or not r['boot_unchanged']: r.update(passed=False,complete=False)
            except BaseException as ex: r.update(passed=False,complete=False,journal_error=repr(ex))
            finally:
                r['active']=False
                r['retention']=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),reason='Main review before retirement; no automatic cleanup',consumer='joined ledger and lifetime gate',byte_budget=64<<20,next_review='after compact evidence committed') for p in (O/'owned').glob('*.stdout')]+[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),reason='Main review before retirement; no automatic cleanup',consumer='joined ledger and lifetime gate',byte_budget=64<<20,next_review='after compact evidence committed') for p in (O/'owned').glob('*.stderr')]
                persist()
                (A/('decode-format-service-'+a.mode+'-v1.json')).write_text(json.dumps(r,indent=2)+'\n')
        print(json.dumps(dict(passed=r['passed'],complete=r['complete'],coverage_sufficient=r.get('coverage_sufficient'),requests=len(r['requests']),fresh_faults=len(r.get('fresh_faults',[])))),flush=True)
if __name__=='__main__': main()
