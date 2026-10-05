import ast, hashlib, json, os, re, subprocess, sys, threading, time
from pathlib import Path

root = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
exe = Path('/tmp/strata-upstream-arc-layer-major-aot')
out = Path('/tmp/strata-upstream-arc-ram-ple-shared-gpu')
out.mkdir(exist_ok=True)
original_report = json.loads((root/'bench/results/2026-10-05-sycl-prefill-scaling/layer-major/host-residual-long/baseline-k8v8-32768-c4096/run.json').read_text())
original = original_report['runs'][0]
proof = json.loads(Path('/tmp/strata-upstream-arc-layer-major-aot-initial/summary.json').read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert len(proof['runs']) == 6 and all(r.get('all_bytes_identical_to_jit') for r in proof['runs'])
assert sha(exe) == proof['binary_sha256']
report = dict(binary_sha256=sha(exe),runs=[],
    measurement='Single 32K observations on the GPU shared with the resident embedding server. Fixed 65,538-position K8/V8 context; 4K chunks, first 4K residual rows in VRAM. Production RAM PLE includes table load at startup, outside prefill time but inside process time. Profile phase intervals include gaps and instrumentation overhead. Separate wall case disables markers and transfer timers.')
support = Path('/tmp/strata-upstream-arc-layer-major-ram-ple.py')
tree = ast.parse(support.read_text())
selected = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('save','memory','run_observed')]
assert len(selected) == 3
exec(compile(ast.Module(body=selected,type_ignores=[]),str(support),'exec'))
report['support_sha256'] = sha(support)
assert memory(os.getpid())['VmRSS_bytes'] > 0
peer = Path('/proc/6151')
if peer.exists(): report['peer_before'] = dict(pid=6151,unit='llama-server-qwen3embed.service',memory=memory(6151))
save()
env = dict(os.environ,**original_report['env'])
for k in ('STRATA_PREFILL_TIMING','STRATA_PREFILL_TRANSFER_TIMING','STRATA_PREFILL_PRELOAD_PLE',
          'STRATA_SERVE_NO_MTP','STRATA_DUMP_FIRST_LOGITS','STRATA_PREFILL_DUMP_STATE',
          'STRATA_PREFILL_DUMP_R','STRATA_PREFILL_DUMP_R_ALL'):
    env.pop(k,None)
env.update(STRATA_PREFILL_ATTN_BATCH='128',STRATA_PREFILL_ATTN_LAYOUT='4',STRATA_PREFILL_TOPK_TUNED='1',
    STRATA_GDN_KEYHEAD='0',STRATA_GDN_KEYHEAD_TUNED='1',STRATA_PREFILL_COMPACT='2',
    STRATA_PREFILL_LAYER_MAJOR='1',STRATA_PREFILL_LAYER_MAJOR_R_GPU='4096')
a = original['args']
common = [sys.executable,str(root/'sycl/tools/prefill_profile.py'),'--exe',str(exe),
    '--pack',a[a.index('--pack')+1],'--native',a[a.index('--native')+1],
    '--expert-profile',a[a.index('--expert-profile')+1],'--expert-cache','128','--cwd',str(root),
    '--repeats','1','--chunk','4096','--max-context','65538','--kv','int8','--multiple-chunks',
    '--tokens-file','/tmp/strata-upstream-arc-long-context-fixture/coding-context-64k-tokens.txt',
    '--lengths','32768','--ple-io','ram','--timeout','1800']
expected = (original['logits_sha256'],original['logits_count'],original['output_ids'])
for mode in ('profile','wall'):
    name = 'ram-k8v8-32768-c4096-gpu4096-'+mode
    dest,record = run_observed(name,common+['--mode',mode,'--label',name,'--output',str(out/name)],env)
    actual = json.loads((dest/'run.json').read_text())['runs'][0]
    assert actual['logits_finite'] and actual['chunks'] == 8
    assert (actual['logits_sha256'],actual['logits_count'],actual['output_ids']) == expected
    assert 'preload' not in actual and 'ple_table_locked' in actual
    record.update(prefill_wall_ms=actual['wall_ms'],tokens_per_second=actual['tokens_per_second'],
        ple_table_locked=actual['ple_table_locked'],ple_table_startup_seconds=actual['ple_table_startup_seconds'],
        full_head_and_ids_identical_to_original=True)
    if mode == 'profile':
        record.update(transfer=actual['transfer'],phases=actual['phases'])
        assert actual['transfer']['residual_gpu_tokens'] == 4096
    else: assert 'transfer' not in actual and 'phases' not in actual
    save()
    print(name+' PASS complete head and IDs',flush=True)
if peer.exists(): report['peer_after'] = dict(pid=6151,memory=memory(6151))
save()
print('PASS shared-GPU RAM PLE diagnostic',flush=True)
