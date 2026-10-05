import ast, hashlib, json, os, re, subprocess, sys, threading, time
from pathlib import Path

root = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
exe = Path('/tmp/strata-upstream-arc-layer-major-gpu-jit')
out = Path('/tmp/strata-upstream-arc-layer-major-c8-confirm')
out.mkdir(exist_ok=True)
refpath = Path('/tmp/strata-upstream-arc-attn-paired-wall/base-p1/run.json')
ref = json.loads(refpath.read_text()); a = ref['runs'][0]['args']
env = dict(os.environ, **ref['env'])
for k in list(env):
    if k in ('STRATA_PREFILL_TIMING','STRATA_PREFILL_TRANSFER_TIMING','STRATA_PREFILL_PRELOAD_PLE',
             'STRATA_SERVE_NO_MTP','STRATA_DUMP_FIRST_LOGITS','STRATA_PREFILL_DUMP_STATE',
             'STRATA_PREFILL_DUMP_R','STRATA_PREFILL_DUMP_R_ALL'): env.pop(k)
env.update(STRATA_PREFILL_ATTN_BATCH='128',STRATA_PREFILL_ATTN_LAYOUT='4',STRATA_PREFILL_TOPK_TUNED='1',
           STRATA_GDN_KEYHEAD='0',STRATA_GDN_KEYHEAD_TUNED='1',STRATA_PREFILL_COMPACT='2',STRATA_PREFILL_LAYER_MAJOR='2',STRATA_PREFILL_LAYER_MAJOR_R_GPU='8194')
proof = json.loads(Path('/tmp/strata-upstream-arc-layer-major-gpu-initial/summary.json').read_text())
assert len(proof['runs']) == 8 and all(r.get('all_bytes_identical_to_baseline') for r in proof['runs'] if r['compact'])
assert hashlib.sha256(exe.read_bytes()).hexdigest() == proof['binary_sha256']
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
          '--repeats','1','--chunk','8192','--max-context','65538','--kv','int8','--multiple-chunks',
          '--tokens-file','/tmp/strata-upstream-arc-long-context-fixture/coding-context-64k-tokens.txt','--timeout','1200']
try:
    for n in (32768,65536):
        control=None
        for label,compact,major,gpu in [('baseline',0,0,0),('gpu32',2,1,32768)]:
            name=f'{label}-k8v8-{n}-c8192'
            dest=out/name
            args=common+['--lengths',str(n),'--mode','transfer','--label',name,'--output',str(dest)]
            print(name+' metrics start',flush=True)
            child=subprocess.run(args,cwd=root,env=dict(env,STRATA_PREFILL_COMPACT=str(compact),
                           STRATA_PREFILL_LAYER_MAJOR=str(major),STRATA_PREFILL_LAYER_MAJOR_R_GPU=str(gpu)),timeout=1500)
            if child.returncode!=0:
                # Preserve failed allocation evidence. If only the extra GPU
                # residual rows do not fit, use one fewer complete GPU chunk.
                if major:
                    assert 'requested GPU residual rows exceed free VRAM' in ''.join(p.read_text() for p in dest.glob('*.log'))
                    gpu=24576;name=f'gpu24-k8v8-{n}-c8192';dest=out/name
                    args=common+['--lengths',str(n),'--mode','transfer','--label',name,'--output',str(dest)]
                    print(name+' smaller residual allocation start',flush=True)
                    subprocess.run(args,cwd=root,env=dict(env,STRATA_PREFILL_COMPACT='2',
                        STRATA_PREFILL_LAYER_MAJOR='1',STRATA_PREFILL_LAYER_MAJOR_R_GPU=str(gpu)),check=True,timeout=1500)
                else: raise RuntimeError('Original C8192 control failed; evidence preserved')
            r=json.loads((dest/'run.json').read_text())['runs'][0]
            assert r['logits_finite'] and r['logits_count']==248320 and r['chunks']==n//8192
            t=r['transfer']
            if major:
                assert t['expert_copies']==48*512 and t['residual_gpu_tokens']==gpu
                assert t['activation_bytes']==2*47*(n-gpu)*10240*4
                assert t['device_residual_bytes']==2*47*gpu*10240*4
                assert (r['logits_sha256'],r['output_ids'])==control
                print(name+' complete head bit-identical to original SAME C8192 control',flush=True)
            else:
                control=(r['logits_sha256'],r['output_ids'])
                assert t['activation_bytes']==0
finally:
    stop.set();observer.join()
print('PASS C8192 paired original/layer-major 32K/64K full-head and memory measurements',flush=True)
