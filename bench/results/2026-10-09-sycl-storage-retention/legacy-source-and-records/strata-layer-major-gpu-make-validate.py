from pathlib import Path
p=Path('/tmp/strata-upstream-arc-layer-major-ple-validate-jit.py').read_text()
p=p.replace('layer-major-ple-initial','layer-major-gpu-initial').replace('layer-major-ple-jit','layer-major-gpu-jit')
p=p.replace("for n,chunk in [(257,128),(4096,4096),(8087,4096)]:", "for n,chunk in [(257,128),(4096,4096),(8087,4096)]:")
p=p.replace("    for label,compact,major in [('baseline',0,0),('compact-hc',2,0),('layer-major',2,1)]:", "    cases = [('baseline',0,0,0)] + ([] if n==4096 else [('gpu-mixed',2,2,128 if n==257 else 4096)]) + [('gpu-all',2,2,n)]\n    base_record = None\n    for label,compact,major,gpu in cases:")
p=p.replace("STRATA_PREFILL_DUMP_R=str(residual),STRATA_PREFILL_DUMP_R_ALL='1')", "STRATA_PREFILL_DUMP_R=str(residual),STRATA_PREFILL_DUMP_R_ALL='1',\n                       STRATA_PREFILL_LAYER_MAJOR_R_GPU=str(gpu),STRATA_PREFILL_TRANSFER_TIMING='1')\n        if label=='gpu-mixed' and n==257: run_env['STRATA_PREFILL_TIMING']='1'\n        if label=='gpu-mixed' and n==8087: run_env['STRATA_PREFILL_RING']='16'")
p=p.replace('compact=compact,layer_major=major,args=args,', 'compact=compact,layer_major=major,gpu_tokens=gpu,args=args,')
p=p.replace("        if label=='baseline': original = dest", "        if label=='baseline':\n            original = dest; base_record = record\n            prior = Path('/tmp/strata-upstream-arc-layer-major-ple-initial')/name\n            for key in ('head.bin','state.bin','residual.bin'): equal_files(prior/key,dest/key)\n            record['all_bytes_identical_to_prior_original'] = True")
p=p.replace("report['runs'][-(3 if major else 2)]['output_ids']", "base_record['output_ids']")
p=p.replace("        save()\n        print(name+' PASS full head, ALL residual rows and ALL persistent state bytes',flush=True)", """        t = json.loads(re.search(r'^strata prefill transfer: (.*)$',text,re.M)[1])
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
        print(name+' PASS full head, ALL residual rows and ALL persistent state bytes',flush=True)""")
Path('/tmp/strata-upstream-arc-layer-major-gpu-validate-jit.py').write_text(p)
compile(p,'<gpu-validator>','exec')
