import ast, hashlib, json, os, re, subprocess, sys, threading, time
from pathlib import Path

root = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
exe = Path('/tmp/strata-upstream-arc-layer-major-ple-jit')
out = Path('/tmp/strata-upstream-arc-layer-major-ple-confirm')
out.mkdir(exist_ok=True)
refpath = Path('/tmp/strata-upstream-arc-attn-paired-wall/base-p1/run.json')
ref = json.loads(refpath.read_text()); a = ref['runs'][0]['args']
env = dict(os.environ, **ref['env'])
for k in list(env):
    if k in ('STRATA_PREFILL_TIMING','STRATA_PREFILL_TRANSFER_TIMING','STRATA_PREFILL_PRELOAD_PLE',
             'STRATA_SERVE_NO_MTP','STRATA_DUMP_FIRST_LOGITS','STRATA_PREFILL_DUMP_STATE',
             'STRATA_PREFILL_DUMP_R','STRATA_PREFILL_DUMP_R_ALL'): env.pop(k)
env.update(STRATA_PREFILL_ATTN_BATCH='128',STRATA_PREFILL_ATTN_LAYOUT='4',STRATA_PREFILL_TOPK_TUNED='1',
           STRATA_GDN_KEYHEAD='0',STRATA_GDN_KEYHEAD_TUNED='1',STRATA_PREFILL_COMPACT='2',STRATA_PREFILL_LAYER_MAJOR='1')
proof = json.loads(Path('/tmp/strata-upstream-arc-layer-major-ple-initial/summary.json').read_text())
assert len(proof['runs']) == 9 and all(r.get('all_bytes_identical_to_baseline') for r in proof['runs'] if r['compact'])
assert hashlib.sha256(exe.read_bytes()).hexdigest() == proof['binary_sha256']
for script,name,reference in [('bench/results/2026-10-05-sycl-upstream-arc/serve_check.py','normal-mtp',False),
                              ('bench/results/2026-10-05-sycl-prefill-scaling/gdn/serve_long_check.py','long-mtp',True)]:
    dest = out/(name+'.json')
    args = [sys.executable,str(root/script),'--engine',str(exe),'--out',str(dest)]
    if reference: args += ['--reference',str(refpath)]
    print(name+' layer-major start',flush=True)
    subprocess.run(args,cwd=root,env=env,check=True,timeout=900)
    previous = Path('/tmp/strata-upstream-arc-attn-order-normal-serve.json') if not reference else Path('/tmp/strata-upstream-arc-attn-order-long-serve/layout4.json')
    old,new = [json.loads(p.read_text()) for p in (previous,dest)]
    assert len(old['requests']) == len(new['requests']) == 4
    for r,s in zip(old['requests'],new['requests']):
        assert r['ids']==s['ids'] and r['logprobs']==s['logprobs'],s['name']
        if 'logits_sha256' in r: assert r['logits_sha256']==s['logits_sha256'] and s['logits_finite']
    print(name+' all four IDs, logprobs and available full heads match original',flush=True)

# Reuse the same read-only /proc sampler used by the initial equality checks.
tree = ast.parse(Path('/tmp/strata-upstream-arc-layer-major-ple-validate-jit.py').read_text())
node = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='memory')
exec(compile(ast.Module(body=[node],type_ignores=[]),'<memory sampler>','exec'))
stop = threading.Event(); peaks = {}
def observe():
    with (out/'memory-samples.jsonl').open('w') as handle:
        while not stop.is_set():
            for proc in Path('/proc').iterdir():
                if not proc.name.isdigit(): continue
                try:
                    cmd = (proc/'cmdline').read_bytes().split(b'\0')
                    if not cmd or cmd[0] != os.fsencode(exe) or b'--tokens-file' not in cmd: continue
                    name = Path(os.fsdecode(cmd[cmd.index(b'--tokens-file')+1])).stem
                    data = memory(int(proc.name))
                    handle.write(json.dumps(dict(time_ns=time.time_ns(),pid=int(proc.name),run=name,bytes=data))+'\n')
                    handle.flush()
                    record = peaks.setdefault(name,dict(samples=0,sampled_peak_bytes={}))
                    record['samples'] += 1
                    for k,v in data.items(): record['sampled_peak_bytes'][k] = max(record['sampled_peak_bytes'].get(k,0),v)
                except (OSError,ValueError): pass
            (out/'memory-summary.json').write_text(json.dumps(dict(interval_seconds=1,
                note='Monitoring began before ALL listed CLI runs. /proc status and process DRM fdinfo, deduplicated by DRM client ID; sampled process peaks, not total-card usage. MTP checks above are excluded.',runs=peaks),indent=2)+'\n')
            stop.wait(1)
observer = threading.Thread(target=observe); observer.start()
common = [sys.executable,str(root/'sycl/tools/prefill_profile.py'),'--exe',str(exe),
          '--pack',a[a.index('--pack')+1],'--native',a[a.index('--native')+1],
          '--expert-profile',a[a.index('--expert-profile')+1],'--expert-cache','128','--cwd',str(root),
          '--repeats','1','--chunk','4096','--max-context','65538','--kv','int8','--multiple-chunks',
          '--tokens-file','/tmp/strata-upstream-arc-long-context-fixture/coding-context-64k-tokens.txt','--timeout','1200']
try:
    for n in (16384,32768,65536):
        heads = []
        for label,compact,major in [('baseline',0,0),('layer-major',2,1)]:
            name = f'{label}-k8v8-{n}-c4096'
            dest = out/name
            args = common + ['--lengths',str(n),'--mode','transfer','--label',name,'--output',str(dest)]
            print(name+' transfer/memory metrics start',flush=True)
            subprocess.run(args,cwd=root,env=dict(env,STRATA_PREFILL_COMPACT=str(compact),
                           STRATA_PREFILL_LAYER_MAJOR=str(major)),check=True,timeout=1500)
            r = json.loads((dest/'run.json').read_text())['runs'][0]
            assert r['logits_finite'] and r['logits_count']==248320 and r['chunks']==n//4096
            t = r['transfer']
            if major:
                assert t['expert_copies']==48*512 and t['activation_copies']==2*47*(n//4096)
                assert t['activation_bytes']==2*47*n*10240*4
            else: assert t.get('activation_bytes',0)==0
            heads.append((r['logits_sha256'],r['output_ids']))
            print(name+' metrics validated',flush=True)
        assert heads[0]==heads[1],(n,heads)
        print(str(n)+' K8V8 complete heads bit-identical with same chunk size',flush=True)
finally:
    stop.set(); observer.join()
print('PASS layer-major normal/long MTP and 16K/32K/64K K8V8 transfer/memory suite',flush=True)
