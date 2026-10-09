from pathlib import Path
p=Path('/tmp/strata-upstream-arc-layer-major-ple-confirm-and-long.py').read_text()
p=p.replace('layer-major-ple-jit','layer-major-gpu-jit').replace('layer-major-ple-confirm','layer-major-gpu-confirm').replace('layer-major-ple-initial','layer-major-gpu-initial')
p=p.replace("STRATA_PREFILL_COMPACT='2',STRATA_PREFILL_LAYER_MAJOR='1')", "STRATA_PREFILL_COMPACT='2',STRATA_PREFILL_LAYER_MAJOR='2',STRATA_PREFILL_LAYER_MAJOR_R_GPU='8194')")
p=p.replace("len(proof['runs']) == 9", "len(proof['runs']) == 8")
# The AST loads only the memory function from the old script; no old experiments execute.
start=p.index('try:\n    for n in (16384,32768,65536):')
p=p[:start]+'''try:
    # Reserve the real 256K K8V8 context, all dense weights, decode cache, prefill
    # scratch and the complete largest layer cache. Small input isolates capacity.
    # A successful allocation says nothing about full 256K input throughput.
    capacity=[]
    for chunk in (4096,1024):
        name=f'capacity-256k-c{chunk}'
        dest=out/name
        args=common+['--lengths','2','--mode','transfer','--label',name,'--output',str(dest),
                     '--chunk',str(chunk),'--max-context','262146',
                     '--tokens-file','/tmp/strata-upstream-arc-256k-context-fixture/coding-context-256k-tokens.txt']
        print(name+' actual allocations start',flush=True)
        child=subprocess.run(args,cwd=root,env=dict(env,STRATA_PREFILL_LAYER_MAJOR='2',
                             STRATA_PREFILL_LAYER_MAJOR_R_GPU='0'),timeout=1500)
        row=dict(chunk=chunk,exit_code=child.returncode,report=str(dest/'run.json'))
        if child.returncode==0:
            r=json.loads((dest/'run.json').read_text())['runs'][0]
            assert r['transfer']['expert_copies']==48*512 and r['transfer']['residual_gpu_tokens']==0
            row['actual_full_layer_cache_used']=True
            row['head_finite']=r['logits_finite']
        capacity.append(row)
        (out/'capacity-summary.json').write_text(json.dumps(capacity,indent=2)+'\\n')
        if child.returncode==0: break
    for n,mode in [(32768,'transfer'),(65536,'transfer'),(32768,'profile')]:
        name=f'gpu32768-k8v8-{n}-c4096-{mode}'
        dest=out/name
        args=common+['--lengths',str(n),'--mode',mode,'--label',name,'--output',str(dest)]
        print(name+' metrics start',flush=True)
        subprocess.run(args,cwd=root,env=dict(env,STRATA_PREFILL_LAYER_MAJOR='1',
                       STRATA_PREFILL_LAYER_MAJOR_R_GPU='32768'),check=True,timeout=1500)
        r=json.loads((dest/'run.json').read_text())['runs'][0]
        t=r['transfer']
        assert r['logits_finite'] and r['logits_count']==248320 and r['chunks']==n//4096
        assert t['expert_copies']==48*512 and t['residual_gpu_tokens']==32768
        assert t['activation_bytes']==2*47*(n-32768)*10240*4
        assert t['device_residual_bytes']==2*47*32768*10240*4
        reference=Path('/tmp/strata-upstream-arc-layer-major-ple-confirm')/f'baseline-k8v8-{n}-c4096/run.json'
        original=json.loads(reference.read_text())['runs'][0]
        assert r['logits_sha256']==original['logits_sha256'] and r['output_ids']==original['output_ids']
        print(name+' complete head bit-identical to original K8V8 control',flush=True)
finally:
    stop.set(); observer.join()
print('PASS GPU-residual MTP, capacity observations and 32K/64K K8V8 metrics',flush=True)
'''
Path('/tmp/strata-upstream-arc-layer-major-gpu-confirm-and-metrics.py').write_text(p)
compile(p,'<gpu-suite>','exec')
