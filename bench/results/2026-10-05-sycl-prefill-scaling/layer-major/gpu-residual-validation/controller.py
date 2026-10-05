import array, hashlib, json, math, os, re, shutil, subprocess, sys, threading, time
from pathlib import Path

root = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
out = Path('/tmp/strata-upstream-arc-layer-major-gpu-initial')
out.mkdir(exist_ok=True)
refpath = Path('/tmp/strata-upstream-arc-attn-paired-wall/base-p1/run.json')
ref = json.loads(refpath.read_text())
env = dict(os.environ, **ref['env'])
for k in list(env):
    if k.startswith('STRATA_GDN_') or k in ('STRATA_PREFILL_TIMING', 'STRATA_PREFILL_TRANSFER_TIMING',
        'STRATA_PREFILL_PRELOAD_PLE', 'STRATA_SERVE_NO_MTP', 'STRATA_DUMP_FIRST_LOGITS',
        'STRATA_PREFILL_DUMP_STATE', 'STRATA_PREFILL_DUMP_R', 'STRATA_PREFILL_DUMP_R_ALL'):
        env.pop(k)
env.update(SYCL_CACHE_PERSISTENT='0', STRATA_PREFILL_COMPACT='0', STRATA_PREFILL_LAYER_MAJOR='0')
env['STRATA_PLE_GGUF'] = str(Path.home()/'.local/share/strata-sycl/models/qwen3.8-flash-next-q2_0/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf')
with (out/'ctest.log').open('w') as log:
    subprocess.run(['ctest','--test-dir',str(root/'build-sycl-upstream-jit'),'--output-on-failure',
                    '--timeout','180','-j','1'],env=env,cwd=root,stdout=log,stderr=subprocess.STDOUT,
                   check=True,timeout=900)
exe = Path('/tmp/strata-upstream-arc-layer-major-gpu-jit')
shutil.copy2(root/'build-sycl-upstream-jit/strata', exe)
source = out/'source'
source.mkdir(exist_ok=True)
for name in ['sycl/src/prefill/prefill.cpp','sycl/src/program/generate.cpp',
             'include/strata/prefill/prefill.hpp','sycl/tools/prefill_profile.py']:
    dest = source/name
    dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(root/name,dest)
(source/'changes.patch').write_bytes(subprocess.check_output(['git','diff'],cwd=root))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
report = dict(binary_sha256=sha(exe), source_sha256={str(p.relative_to(source)):sha(p)
              for p in source.rglob('*') if p.is_file()}, runs=[], memory_note='1 Hz samples of this engine process, /proc status and DRM fdinfo deduplicated by client ID; sampled peaks can miss transients. Diagnostics are equality checks, not performance estimates.')
def save():
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
save()
print('JIT all 30 tests passed; frozen binary', report['binary_sha256'],flush=True)
env.update(SYCL_CACHE_PERSISTENT='1', STRATA_PREFILL_ATTN_BATCH='128', STRATA_PREFILL_ATTN_LAYOUT='4',
           STRATA_PREFILL_TOPK_TUNED='1', STRATA_GDN_KEYHEAD='0', STRATA_GDN_KEYHEAD_TUNED='1')
args0 = ref['runs'][0]['args']
def value(key): return args0[args0.index(key)+1]
tokens = [int(t) for t in Path(ref['fixture']).read_text().split()]
def memory(pid):
    proc = Path('/proc')/str(pid)
    result = {}
    status = (proc/'status').read_text()
    for field in ['VmRSS','VmHWM','VmSize','VmSwap']:
        m = re.search(r'^'+field+r':\s*(\d+) kB$',status,re.M)
        if m: result[field+'_bytes'] = int(m[1])*1024
    seen = set()
    for fd in (proc/'fd').iterdir():
        try:
            if '/dev/dri/' not in os.readlink(fd): continue
            info = (proc/'fdinfo'/fd.name).read_text()
            identity = re.search(r'^drm-client-id:\s*(\d+)$',info,re.M)
            if not identity or identity[1] in seen: continue
            seen.add(identity[1])
            for key,num,unit in re.findall(r'^(drm-(?:total|resident|shared|active|purgeable)-[\w]+):\s*(\d+)(?:\s*(KiB|MiB|GiB))?$',info,re.M):
                result[key+'_bytes'] = result.get(key+'_bytes',0)+int(num)*{'':1,'KiB':1024,'MiB':1024**2,'GiB':1024**3}[unit]
        except OSError: pass
    return result
def equal_files(a,b):
    assert a.stat().st_size == b.stat().st_size, (a,b,'sizes')
    with a.open('rb') as f,b.open('rb') as g:
        offset = 0
        while True:
            x,y = f.read(1024*1024),g.read(1024*1024)
            if x != y:
                first = next(i for i,(u,v) in enumerate(zip(x,y)) if u!=v)
                raise AssertionError((str(a),str(b),'first differing byte',offset+first))
            if not x: return
            offset += len(x)
for n,chunk in [(257,128),(4096,4096),(8087,4096)]:
    original = None
    cases = [('baseline',0,0,0)] + ([] if n==4096 else [('gpu-mixed',2,2,128 if n==257 else 4096)]) + [('gpu-all',2,2,n)]
    base_record = None
    for label,compact,major,gpu in cases:
        name = f'{label}-{n}-c{chunk}'
        dest = out/name
        dest.mkdir(exist_ok=True)
        fixture = dest/'tokens.txt'
        fixture.write_text(' '.join(map(str,tokens[:n+1]))+'\n')
        head,state,residual = [dest/k for k in ('head.bin','state.bin','residual.bin')]
        run_env = dict(env,STRATA_PREFILL_COMPACT=str(compact),STRATA_PREFILL_LAYER_MAJOR=str(major),
                       STRATA_DUMP_FIRST_LOGITS=str(head),STRATA_PREFILL_DUMP_STATE=str(state),
                       STRATA_PREFILL_DUMP_R=str(residual),STRATA_PREFILL_DUMP_R_ALL='1',
                       STRATA_PREFILL_LAYER_MAJOR_R_GPU=str(gpu),STRATA_PREFILL_TRANSFER_TIMING='1')
        if label=='gpu-mixed' and n==257: run_env['STRATA_PREFILL_TIMING']='1'
        if label=='gpu-mixed' and n==8087: run_env['STRATA_PREFILL_RING']='16'
        args = [str(exe),'--pack',value('--pack'),'--native',value('--native'),'--tokens-file',str(fixture),
                '--max-new','1','--spec','2','--max-context','8194','--prefill',str(chunk),
                '--no-prefill-borrow','--expert-cache','128','--expert-profile',value('--expert-profile'),
                '--expert-cache-per-layer','--pool-workers','5','--pcie-frac','0','--adapt-swaps','0',
                '--ple-io','direct','--suffix-draft','0','--greedy','--stats']
        print(name+' full residual/state/head check start',flush=True)
        with (dest/'engine.log').open('w') as log,(dest/'memory-samples.jsonl').open('w') as samples:
            child = subprocess.Popen(args,cwd=root,env=run_env,stdout=log,stderr=subprocess.STDOUT)
            stop = threading.Event()
            peaks = {}; count = [0]
            def observe():
                while not stop.is_set():
                    try:
                        data = memory(child.pid)
                        count[0] += 1
                        samples.write(json.dumps(dict(time_ns=time.time_ns(),pid=child.pid,bytes=data))+'\n')
                        samples.flush()
                        for k,v in data.items(): peaks[k] = max(peaks.get(k,0),v)
                    except OSError: pass
                    stop.wait(1)
            observer = threading.Thread(target=observe)
            observer.start()
            try: rc = child.wait(timeout=600)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait(); raise
            finally:
                stop.set(); observer.join()
        text = (dest/'engine.log').read_text()
        record = dict(name=name,tokens=n,chunk=chunk,compact=compact,layer_major=major,gpu_tokens=gpu,args=args,
                      env={k:v for k,v in run_env.items() if k.startswith(('STRATA_','SYCL_','ONEAPI_','NEO_')) or k=='LD_LIBRARY_PATH'},
                      exit_code=rc,memory_samples=count[0],sampled_peak_bytes=peaks)
        report['runs'].append(record);save()
        assert rc == 0,(name,rc)
        if compact == 2: assert f'compact-hc layout for {chunk} tokens' in text,name
        if major: assert 'strata prefill: layer-major,' in text,name
        values = array.array('f'); values.frombytes(head.read_bytes())
        assert len(values)==248320 and all(map(math.isfinite,values)),name
        assert residual.stat().st_size == n*(8+10240*4), name
        match = re.search(r'^output\s*:\s*(.*)$',text,re.M)
        record.update(output_ids=[int(x) for x in match[1].split()],head_sha256=sha(head),
                      state_sha256=sha(state),residual_sha256=sha(residual),
                      state_bytes=state.stat().st_size,residual_bytes=residual.stat().st_size,head_finite=True)
        if label=='baseline':
            original = dest; base_record = record
            prior = Path('/tmp/strata-upstream-arc-layer-major-ple-initial')/name
            for key in ('head.bin','state.bin','residual.bin'): equal_files(prior/key,dest/key)
            record['all_bytes_identical_to_prior_original'] = True
        else:
            for key in ('head.bin','state.bin','residual.bin'): equal_files(original/key,dest/key)
            assert record['output_ids'] == base_record['output_ids']
            record['all_bytes_identical_to_baseline'] = True
        t = json.loads(re.search(r'^strata prefill transfer: (.*)$',text,re.M)[1])
        record['transfer'] = t
        if major:
            assert t['residual_gpu_tokens']==gpu
            assert t['expert_copies']==48*512
            assert t['activation_bytes']==2*47*(n-gpu)*10240*4
            assert t['device_residual_bytes']==2*47*gpu*10240*4
            if label=='gpu-mixed' and n==257:
                lines = re.findall(r'^strata prefill phases: (.*)$',text,re.M)
                assert len(lines)==1
                record['phases']=json.loads(lines[0])
                assert record['phases']['phase_ms']['layer weight preload']>0
                assert record['phases']['phase_ms']['residual copy']>0
        save()
        print(name+' PASS full head, ALL residual rows and ALL persistent state bytes',flush=True)
print('PASS layer-major initial model/state equivalence',flush=True)
