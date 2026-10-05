import array,hashlib,json,math,os,re,selectors,statistics,subprocess,time
from pathlib import Path
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
exe=Path('/tmp/strata-upstream-arc-thp-jit')
out=Path('/tmp/strata-upstream-arc-thp-decode');out.mkdir(exist_ok=True)
ref=json.loads(Path('/tmp/strata-upstream-arc-attn-paired-wall/base-p1/run.json').read_text())
env=dict(os.environ,**ref['env'])
for k in list(env):
    if k.startswith('STRATA_'):env.pop(k)
env.update(STRATA_IO_THREADS='16',STRATA_PREFILL_RING='8',STRATA_PREFILL_ATTN_BATCH='128',
    STRATA_PREFILL_ATTN_LAYOUT='4',STRATA_PREFILL_TOPK_TUNED='1',STRATA_GDN_KEYHEAD='0',
    STRATA_GDN_KEYHEAD_TUNED='1',STRATA_DECODE_TIMING='1',SYCL_CACHE_PERSISTENT='1')
base=Path.home()/'.local/share/strata-sycl'
args=['--pack',str(base/'packs/qwen3.8-flash-next-iq3_s'),'--native',str(base/'models/qwen3.8-flash-next-iq3_s/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf'),
 '--serve','--max-context','2048','--kv','int8','--spec','4','--spec-min-p','0',
 '--mtp',str(base/'mtp/rt'),'--suffix-draft','0','--prefill','32','--no-prefill-borrow',
 '--expert-cache','600','--expert-profile','data/expert-profile.bin','--expert-cache-per-layer',
 '--pool-workers','5','--pcie-frac','0.55','--spec-split','--adapt-swaps','0',
 '--conversation-cache-mib','256','--conversation-cache-slots','4','--greedy']
prompt=','.join((root/'bench/results/2026-10-05-sycl-upstream-arc/writing-tokens.txt').read_text().strip().split(','))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report=dict(binary_sha256=sha(exe),source_sha256=sha(root/'sycl/src/core/pinned.dp.cpp'),
 args=args,env={k:v for k,v in env.items() if k.startswith(('STRATA_','SYCL_','ONEAPI_')) or k=='LD_LIBRARY_PATH'},
 thp_enabled=Path('/sys/kernel/mm/transparent_hugepage/enabled').read_text().strip(),
 thp_defrag=Path('/sys/kernel/mm/transparent_hugepage/defrag').read_text().strip(),runs=[],
 measurement='Three same-binary 4K-page/THP pairs, reversed order in pair two. Per engine, 16-token warmup then two repeated 64-token writing requests. Full warmup first heads, all IDs and all printed logprobs compared. Decode diagnostics are enabled identically; no GPU phase instrumentation or concurrent build.')
def save(): (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
def mem(pid):
    smaps=Path(f'/proc/{pid}/smaps').read_text(); regions=[]
    for part in re.split(r'(?m)(?=^[0-9a-f]+-[0-9a-f]+ )',smaps):
        fields={k:int(v)*1024 for k,v in re.findall(r'^(Size|Rss|AnonHugePages|Anonymous|Swap):\s*(\d+) kB',part,re.M)}
        if fields.get('Size',0)>=1024**3:
            fields['mapping']=part.splitlines()[0].split()[0];fields['THPeligible']=re.findall(r'^THPeligible:\s*(\d+)',part,re.M)
            regions.append(fields)
    return dict(large_regions=regions,anon_huge_bytes=sum(int(x)*1024 for x in re.findall(r'^AnonHugePages:\s*(\d+) kB',smaps,re.M)),
        process_swap_bytes=sum(int(x)*1024 for x in re.findall(r'^Swap:\s*(\d+) kB',smaps,re.M)))
def service():return subprocess.check_output(['systemctl','--user','show','llama-server-qwen3embed.service','-p','ActiveState'],text=True).strip()
report['service_before']=service();assert report['service_before']=='ActiveState=inactive';save()
expected=None
for pair in range(1,4):
    order=[0,1] if pair!=2 else [1,0]
    for enabled in order:
        name=f'p{pair}-'+('thp' if enabled else 'base');dest=out/name;dest.mkdir(exist_ok=True)
        dump=dest/'first-head.bin';run_env=dict(env,STRATA_SYCL_ARENA_THP=str(enabled),STRATA_DUMP_FIRST_LOGITS=str(dump))
        record=dict(name=name,pair=pair,thp=enabled,start=time.time(),requests=[])
        print(name+' start',flush=True);start=time.monotonic()
        with (dest/'engine.log').open('w') as log:
            child=subprocess.Popen([str(exe)]+args,cwd=root,env=run_env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log)
            sel=selectors.DefaultSelector();sel.register(child.stdout,selectors.EVENT_READ);pending=bytearray()
            def line():
                deadline=time.monotonic()+300
                while True:
                    if b'\n' in pending:
                        raw,_,tail=pending.partition(b'\n');pending[:]=tail;return raw.decode().strip()
                    remaining=deadline-time.monotonic()
                    if remaining<=0 or not sel.select(remaining):raise TimeoutError(name+' protocol timeout')
                    b=os.read(child.stdout.fileno(),65536)
                    if not b:raise RuntimeError(name+' engine exited')
                    pending.extend(b)
            try:
                startup=[]
                while True:
                    s=line();startup.append(s)
                    if s.startswith('ERR'):raise RuntimeError(s)
                    if s.startswith('READY '):break
                record.update(startup=startup,startup_seconds=time.monotonic()-start,memory_after_startup=mem(child.pid))
                for request_index,max_new in enumerate([16,64,64]):
                    request=f'GEN {max_new} logprobs=5 {prompt}'
                    child.stdin.write((request+'\n').encode());child.stdin.flush()
                    lines=[];ids=[];lp=[]
                    while True:
                        s=line();lines.append(s)
                        if s.startswith('ERR'):raise RuntimeError(s)
                        if s.startswith('T '):ids.append(int(s.split()[1]))
                        if s.startswith('LP '):
                            assert all(math.isfinite(float(v.rsplit(':',1)[-1])) for v in s.split()[1:]);lp.append(s)
                        if s.startswith('DONE '):break
                    done=s.split();count=int(done[1]);ms=float(done[4])
                    assert len(ids)==len(lp)==count and count==max_new,(name,done)
                    record['requests'].append(dict(warmup=request_index==0,request=request,ids=ids,logprobs=lp,
                        protocol=lines,generated=count,decode_ms=ms,tokens_per_second=count*1000/ms))
                    record['memory_after_requests']=mem(child.pid)
                    print(name+f' request {request_index} {count*1000/ms:.3f} tok/s',flush=True)
                head=dump.read_bytes();values=array.array('f');values.frombytes(head)
                assert len(values)==248320 and all(map(math.isfinite,values))
                record.update(logits_sha256=hashlib.sha256(head).hexdigest(),logits_count=len(values),logits_finite=True)
                signature=(record['logits_sha256'],[(r['ids'],r['logprobs']) for r in record['requests']])
                if expected is None:expected=signature
                else:assert signature==expected,name+' differs from baseline head/IDs/logprobs'
                record['all_head_ids_logprobs_identical']=True
                child.stdin.write(b'QUIT\n');child.stdin.flush();child.wait(timeout=60)
                assert child.returncode==0
                record.update(exit_code=child.returncode,process_wall_seconds=time.monotonic()-start)
            finally:
                sel.close()
                if child.poll() is None:
                    # Let the finite in-flight GPU work drain before stopping.
                    child.stdin.write(b'QUIT\n');child.stdin.flush();child.wait()
        assert service()=='ActiveState=inactive'
        report['runs'].append(record);save();print(name+' PASS full head/IDs/logprobs',flush=True)
report['median_tokens_per_second']={str(k):statistics.median(r['tokens_per_second'] for x in report['runs'] if x['thp']==k for r in x['requests'] if not r['warmup']) for k in [0,1]}
report['complete']=True;report['service_after']=service();save();print(json.dumps(report['median_tokens_per_second']),flush=True)
