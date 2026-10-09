from pathlib import Path
old=Path('/tmp/strata-upstream-arc-layer-major-gpu-confirm-and-metrics.py').read_text()
prefix=old[:old.index('for script,name,reference in')]
observe=old[old.index('# Reuse the same read-only'):old.index('try:\n    # Reserve')]
prefix=prefix.replace("out = Path('/tmp/strata-upstream-arc-layer-major-gpu-confirm')", "out = Path('/tmp/strata-upstream-arc-layer-major-c8-confirm')")
observe=observe.replace("'--chunk','4096'","'--chunk','8192'")
p=prefix+observe+'''try:
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
'''
Path('/tmp/strata-upstream-arc-layer-major-c8-confirm-and-metrics.py').write_text(p)
compile(p,'<c8-suite>','exec')
