import ast, hashlib, json, os, re, subprocess, sys, threading, time
from pathlib import Path

root = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
exe = Path('/tmp/strata-upstream-arc-layer-major-aot')
out = Path('/tmp/strata-upstream-arc-layer-major-aot-confirm')
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
proof = json.loads(Path('/tmp/strata-upstream-arc-layer-major-aot-initial/summary.json').read_text())
assert len(proof['runs']) == 6 and all(r.get('all_bytes_identical_to_baseline') for r in proof['runs'] if r['compact'])
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


common=[sys.executable,str(root/'sycl/tools/prefill_profile.py'),'--exe',str(exe),
        '--pack',a[a.index('--pack')+1],'--native',a[a.index('--native')+1],
        '--expert-profile',a[a.index('--expert-profile')+1],'--expert-cache','128','--cwd',str(root),
        '--repeats','1','--chunk','8192','--max-context','65538','--kv','int8','--multiple-chunks',
        '--tokens-file','/tmp/strata-upstream-arc-long-context-fixture/coding-context-64k-tokens.txt',
        '--lengths','32768','--mode','wall','--timeout','1200',
        '--label','aot-gpu32-k8v8-32768-c8192','--output',str(out/'long-cli')]
subprocess.run(common,cwd=root,env=dict(env,STRATA_PREFILL_LAYER_MAJOR='1',
               STRATA_PREFILL_LAYER_MAJOR_R_GPU='32768'),check=True,timeout=1500)
original=json.loads(Path('/tmp/strata-upstream-arc-layer-major-c8-confirm/baseline-k8v8-32768-c8192/run.json').read_text())['runs'][0]
actual=json.loads((out/'long-cli/run.json').read_text())['runs'][0]
assert actual['logits_finite'] and actual['chunks']==4
assert (actual['logits_sha256'],actual['logits_count'],actual['output_ids']) == (original['logits_sha256'],original['logits_count'],original['output_ids'])
assert not any(k in actual for k in ('phases','transfer','preload'))
text=Path(actual['log']).read_text() if 'log' in actual else next((out/'long-cli').glob('*.log')).read_text()
assert 'layer-major, 32768 tokens, host residual bytes 0, GPU residual bytes 1342177280' in text
(out/'summary.json').write_text(json.dumps(dict(binary_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
    all_normal_and_long_mtp_ids_logprobs_heads_identical=True,
    long_cli_full_head_identical=True,full_model_tests=len(proof['runs']),
    registered_tests_passed=30),indent=2)+'\n')
print('PASS AOT MTP restoration and complete 32K C8192 head comparison',flush=True)
