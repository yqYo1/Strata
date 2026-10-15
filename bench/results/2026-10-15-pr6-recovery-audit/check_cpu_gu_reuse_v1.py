#!/usr/bin/env python3
"""Main-owned OFF/ON observer qualification and bounded actual recurrence measurement."""
import argparse,base64,fcntl,hashlib,importlib.util,json,math,os,re,time
from pathlib import Path
from check_gemm_orientation import P,B,D,OWNER,DEVICE,FAULT,sha
from check_decode_format_service_v1 import ledger,format_service


def good(d):return d['passed'] and d['complete'] and not d['active']
def fields(s):return dict(re.findall(r'(\w+)=([^ ]+)',s))
def ints(s):return {k:int(v) if re.fullmatch(r'-?\d+',v) else v for k,v in fields(s).items()}

def recurrence(block,joined):
    lines=block.splitlines();h=ints(lines[0]);assert h['complete']==1 and h['overflow']==0
    assert h['state_bytes']<=2621440 and h['lifetime']=='request_cold' and h['budgets_ignore_live_ram']==1
    c=joined['coverage'];assert h['jobs']==h['completed_jobs']==int(c['cpu_jobs'])
    assert h['tokens']==h['completed_tokens']==int(c['cpu_rows'])
    assert h['request']==int(joined['header']['request'])
    types=list(map(int,next(s for s in lines if s.startswith('formats ')).split()[1:]));assert len(types)==48
    payload=lambda t:3404800 if t in [18,22] else 3353600 if t==21 else 0
    group=lambda t:0 if t==18 else 1 if t==22 else 2 if t==21 else 3
    encoded=next(s.split(' ',1)[1] for s in lines if s.startswith('expert_jobs_uleb128_base64 '))
    raw=base64.b64decode(encoded,validate=True);assert base64.b64encode(raw).decode()==encoded
    counts=[];value=shift=0
    for byte in raw:
        assert shift<64 and not(shift==63 and byte>1)
        value|=(byte&127)<<shift
        if byte&128:shift+=7
        else:counts.append(value);value=shift=0
    assert not shift and len(counts)==48*512*4
    fmt=[[0]*4 for _ in range(4)];unique=0;nt=[0]*4
    for expert in range(48*512):
        row=counts[expert*4:expert*4+4];unique+=any(row)
        for n,v in enumerate(row):fmt[group(types[expert//512])][n]+=v;nt[n]+=v
    assert sum(counts)==h['jobs'] and sum(nt[n]*(n+1) for n in range(4))==h['tokens']
    assert unique==h['first_touches_unique'] and h['retouches']==h['jobs']-unique
    declared=[ints(s) for s in lines if s.startswith('format_nt ')];assert len(declared)==16
    assert {(x['group'],x['nt']) for x in declared}=={(f,n+1) for f in range(4) for n in range(4)}
    for x in declared:assert x['jobs']==fmt[x['group']][x['nt']-1] and x['tokens']==x['jobs']*x['nt']
    jobs=[ints(s) for s in joined['lines'] if s.startswith('cpu_job_nt ')]
    assert {x['nt']:x['jobs'] for x in jobs}=={n+1:nt[n] for n in range(4)}
    assert h['unsupported']==sum(fmt[3])
    skipped=[ints(s) for s in lines if s.startswith('policy_skipped ')];assert len(skipped)==8
    assert {(x['policy'],x['nt']) for x in skipped}=={(p,n) for p in range(2) for n in range(1,5)}
    for x in skipped:
        n=x['nt']-1;assert x['unsupported']==fmt[3][n]
        assert x['bypassed_supported_nt']==(sum(fmt[f][n] for f in range(3)) if x['policy']==1 and n else 0)
    lrus=[ints(s) for s in lines if s.startswith('lru ')];detail=[ints(s) for s in lines if s.startswith('lru_format_nt ')]
    budgets=[268435456,1073741824,4294967296,8589934592]
    assert len(lrus)==8 and {(x['policy'],x['budget']) for x in lrus}=={(p,b) for p in range(2) for b in budgets}
    assert len(detail)==96 and len({(x['policy'],x['budget'],x['group'],x['nt']) for x in detail})==96
    for x in lrus:
        p,b=x['policy'],x['budget'];sub=[y for y in detail if y['policy']==p and y['budget']==b]
        assert len(sub)==12
        for y in sub:
            n,f=y['nt']-1,y['group'];assert 0<=f<3 and 0<=n<4
            assert y['hits']+y['misses']==(0 if p==1 and n else fmt[f][n])
            assert y['hits']>=0 and y['misses']>=0
        assert sum(y['hits'] for y in sub)==x['hits'] and sum(y['misses'] for y in sub)==x['misses']
        assert sum(y['hits']*y['nt'] for y in sub)==x['hit_tokens']
        assert sum(y['misses']*payload([18,22,21][y['group']]) for y in sub)==x['fill_payload_bytes']
        assert x['fill_payload_bytes']-x['evicted_payload_bytes']==x['resident_payload_bytes']<=b
        assert 0<=x['evictions']<=x['misses']
        x['hit_rate']=x['hits']/(x['hits']+x['misses']) if x['hits']+x['misses'] else None
    return dict(header=h,formats=types,format_nt=declared,policies=skipped,lrus=lrus,lru_format_nt=detail,
        encoded_counts_sha256=hashlib.sha256(raw).hexdigest(),encoded_bytes=len(raw),nonzero_expert_nt=sum(v!=0 for v in counts),
        expert_counts_base64=encoded,scope='Exact request-cold ordered simulation, no payload cache or live-RAM admission, no performance qualification.')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['small','long']);a=ap.parse_args()
    bp=D/'cpu-gu-reuse-build-v3.json';build=json.loads(bp.read_text());assert good(build)
    prior=json.loads((D/'model-device-profile-v2.json').read_text());assert good(prior) and not prior['fresh_faults']
    sm=json.loads((D/'layer-major-storage-small-v1.json').read_text());assert good(sm)
    natural=json.loads((D/'natural-layout-long-pair-v2.json').read_text());assert good(natural)
    source=sm if a.mode=='small' else natural
    tp=Path(source['engine_args'][source['engine_args'].index('--tokens-file')+1])
    expected_hash=sm['tokens_sha256'] if a.mode=='small' else natural['input']['tokens_sha256']
    assert sha(tp)==expected_hash;tokens=list(map(int,tp.read_text().strip().split(',')))
    limit=8 if a.mode=='small' else 128;assert len(tokens)==(513 if a.mode=='small' else 34603)
    expected_ids=json.loads((D/'prompt-direct-small-v2.json').read_text())['comparison']['generated_ids'] if a.mode=='small' else prior['requests'][0]['generated_ids']
    if a.mode=='long':
        sg=json.loads((D/'cpu-gu-reuse-small-v1.json').read_text());assert good(sg) and not sg['fresh_faults'] and sg['binary_sha256']==build['binaries']['strata']
    assert sha(OWNER)=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    spec=importlib.util.spec_from_file_location('owner',OWNER);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        exe=Path(build['build_path'])/'strata';assert sha(exe)==build['binaries']['strata']
        assert all(sha(P/n)==h for n,h in build['source_pins'].items())
        assert (DEVICE/'driver').resolve().name=='xe' and not (DEVICE/'devcoredump').exists()
        output=B/('main-primary-cpu-gu-reuse-'+a.mode+'-root-v1');assert not output.exists();output.mkdir(mode=0o700)
        env=dict(prior['environment'],STRATA_SYCL_PROMPT_DIRECT='1',STRATA_PREFILL_LAYER_MAJOR_R_GPU='auto' if a.mode=='small' else '4096')
        for k in ['STRATA_PREFILL_LOGITS','STRATA_PREFILL_LOGITS_POSITIONS','STRATA_DUMP_FIRST_LOGITS']:env.pop(k,None)
        assert not any(k.startswith(('UNITRACE_','PTI_')) or k=='LD_PRELOAD' or 'TOKEN' in k or 'API_KEY' in k for k in env)
        args=prior['argv'][prior['argv'].index(str(exe))+1:]
        args=list(args);args[args.index('--prefill')+1]='257' if a.mode=='small' else '4096'
        record=dict(active=True,passed=False,complete=False,GPU_executed=True,mode=a.mode,commands=[],requests=[],environment=env,
            argv=[str(exe),*args],source_pins=build['source_pins'],binary_sha256=sha(exe),build_receipt_sha256=sha(bp),
            recipe_sha256=sha(Path(__file__)),reference_sha256=sha(D/'model-device-profile-v2.json'),input_sha256=sha(tp),
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),performance_qualified=False,adopted=False,
            quality_qualified=False,physical_262144_qualified=False,restore_qualified=False,
            limits=dict(wall_seconds=1800,request_seconds=1500,RSS_bytes=88<<30,text_bytes=64<<20,job_bytes=256<<20),
            retention_consumer='Compact OFF/ON protocol parity, joined ledger and exact bounded recurrence receipt; retire duplicate successful logs after verified extraction.')
        def persist():(output/'record.json').write_text(json.dumps(record,indent=2)+'\n')
        owner=m.Owner(output/'owned',persist);record['commands']=owner.commands;persist()
        def utility(label,argv):
            e,o,s=owner.run(label,argv,build['environment'],P,wall=30,total_cap=256<<20)
            assert m.closed(e) and e['exit_code']==0,(label,e.get('error'));return o.read_text()
        cursor=None
        try:
            cursor=re.search(r'-- cursor: (\S+)',utility('journal-before',['journalctl','-k','-b','-n','0','--show-cursor','--no-pager']))[1]
            record['journal_before_cursor']=cursor
            for enabled in ([False,True] if a.mode=='small' else [True]):
                arm_env=dict(env,STRATA_CPU_GU_REUSE_OBSERVER=str(int(enabled)),XDG_CACHE_HOME=str(output/('cache-'+str(int(enabled)))))
                def interaction(io,entry):
                    while True:
                        line=io.line(480)
                        if line.startswith('READY '):assert line=='READY 262144 stop';break
                    wire=('GEN '+str(limit)+' temperature=0 seed=42 '+','.join(map(str,tokens))+'\n').encode();io.send(wire)
                    generated=[];progress=[];deadline=time.monotonic()+1500
                    while True:
                        line=io.line(deadline-time.monotonic())
                        if re.fullmatch(r'T \d+',line):generated.append(int(line[2:]));assert len(generated)<=limit
                        elif line.startswith('DONE '):
                            f=line.split();assert len(f)==15
                            done=dict(generated=int(f[1]),prompt=int(f[2]),prompt_ms=float(f[3]),decode_ms=float(f[4]),finish=f[5],accepted=int(f[6]),offered=int(f[7]),reused=int(f[8]),read=int(f[14]))
                            assert done['generated']==len(generated)==limit and done['finish']=='length'
                            assert done['prompt']==done['read']==len(tokens) and done['reused']==0
                            assert all(math.isfinite(done[k]) and done[k]>=0 for k in ['prompt_ms','decode_ms'])
                            assert generated==expected_ids
                            record['requests'].append(dict(observer_on=enabled,environment=arm_env,done=done,generated_ids=generated,reference_ids_equal=True,progress=progress,wire_sha256=hashlib.sha256(wire).hexdigest()))
                            persist();print('GU observer request DONE '+json.dumps(dict(enabled=enabled,done=done)),flush=True);break
                        elif re.fullmatch(r'(?:RESUME \d+|REUSED \d+|PP \d+ \d+ [\d.]+ [\d.]+)',line):progress.append(line)
                        else:raise RuntimeError('unexpected protocol '+line[:200])
                    io.send(b'QUIT\n');io.exit(60)
                print('GU observer '+a.mode+' enabled='+str(enabled),flush=True)
                e,out,err=owner.run('observer-'+str(int(enabled)),[str(exe),*args],arm_env,P,interaction=interaction,wall=1800,
                    cpu=1800*len(os.sched_getaffinity(0)),rss_cap=88<<30,text_cap=64<<20,total_cap=256<<20,file_cap=64<<20)
                assert m.closed(e) and e['exit_code']==0,(e.get('error'),err.read_text()[-3000:])
                text=err.read_text();assert not re.search(r'(?:UR|ZE)_RESULT_ERROR_|terminate called|what\(\):|Strata SYCL cleanup:|retirement failed|stopping with owners live',text)
                blocks=re.findall(r'^strata decode joined begin .*?^strata decode joined close [^\n]*\n?',text,re.M|re.S);assert len(blocks)==1
                req=record['requests'][-1];req['ledger']=ledger(blocks[0],req['done'],0);req['format_service']=format_service(req['ledger'])
                gu=re.findall(r'^strata cpu_gu_reuse_v2 .*?^expert_jobs_uleb128_base64 [^\n]*\n?',text,re.M|re.S)
                assert len(gu)==int(enabled)
                if enabled:req['recurrence']=recurrence(gu[0],req['ledger'])
                req['stdout_sha256'],req['stderr_sha256']=sha(out),sha(err);persist()
            if a.mode=='small':
                x,y=record['requests'];assert x['generated_ids']==y['generated_ids']
                assert x['ledger']['coverage']==y['ledger']['coverage']
                assert [s for s in x['ledger']['lines'] if s.startswith('cpu_job_nt ')]==[s for s in y['ledger']['lines'] if s.startswith('cpu_job_nt ')]
            assert all(sha(P/n)==h for n,h in build['source_pins'].items()) and sha(exe)==record['binary_sha256']
            record.update(passed=True,complete=True)
        except BaseException as error:record['error']=repr(error);raise
        finally:
            try:
                if cursor:
                    entries=[json.loads(s) for s in utility('journal-final',['journalctl','-k','-b','--after-cursor',cursor,'--no-pager','-o','json']).splitlines() if s.strip()]
                    record['fresh_faults']=[s for s in entries if FAULT.search(str(s.get('MESSAGE','')))];record['driver_dump_exists']=(DEVICE/'devcoredump').exists()
                    record['boot_unchanged']=Path('/proc/sys/kernel/random/boot_id').read_text().strip()==record['boot_id']
                    if record['fresh_faults'] or record['driver_dump_exists'] or not record['boot_unchanged']:record.update(passed=False,complete=False)
            finally:
                record['active']=False;persist();(D/('cpu-gu-reuse-'+a.mode+'-v1.json')).write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(dict(passed=record['passed'],fresh_faults=len(record['fresh_faults']))),flush=True)


if __name__=='__main__':main()
