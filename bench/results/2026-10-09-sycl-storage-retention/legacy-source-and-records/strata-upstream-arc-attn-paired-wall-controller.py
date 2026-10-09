import json, os, subprocess, sys
from pathlib import Path
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
profile=json.loads(Path('/tmp/strata-upstream-arc-attn-paired-warm/base-p1/run.json').read_text())
accepted=json.loads((root/'bench/results/2026-10-05-sycl-prefill-scaling/wall/run.json').read_text())
expected={r['tokens']:(r['logits_sha256'],r['logits_count'],r['output_ids']) for r in accepted['runs']}
a=profile['runs'][0]['args']
common=[sys.executable,str(root/'sycl/tools/prefill_profile.py'),'--exe',profile['exe'],'--pack',a[a.index('--pack')+1],'--native',a[a.index('--native')+1],'--expert-profile',a[a.index('--expert-profile')+1],'--tokens-file',profile['fixture'],'--lengths','4096,8087','--max-context','8192','--chunk','8192','--expert-cache','128','--mode','wall','--repeats','1','--cwd',str(root)]
for pair in range(1,4):
    for label,batch,layout in ([('base',32,0),('fast',128,3)] if pair%2 else [('fast',128,3),('base',32,0)]):
        dest=Path('/tmp/strata-upstream-arc-attn-paired-wall')/f'{label}-p{pair}'
        env=dict(os.environ,**profile['env'])
        env.update(STRATA_PREFILL_ATTN_BATCH=str(batch),STRATA_PREFILL_ATTN_LAYOUT=str(layout))
        args=common+['--label',f'{label}-p{pair}','--output',str(dest)]
        if pair==1: args+=['--warmup']
        if pair==2: args+=['--reverse-order']
        print(f'pair={pair} label={label} start',flush=True)
        subprocess.run(args,cwd=root,env=env,check=True)
        result=json.loads((dest/'run.json').read_text())
        for r in result['runs']:
            assert (r['logits_sha256'],r['logits_count'],r['output_ids'])==expected[r['tokens']],r['name']
            assert r['logits_finite'] and not any(k in r for k in ('transfer','phases','preload'))
        print(f'pair={pair} label={label} all-heads-bit-identical',flush=True)
