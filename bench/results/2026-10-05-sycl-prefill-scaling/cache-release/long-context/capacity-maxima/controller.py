import ast,hashlib,json,os,re,subprocess,sys,threading,time
from pathlib import Path
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
exe=Path('/tmp/strata-upstream-arc-cache-policy4-jit')
out=Path('/tmp/strata-upstream-arc-cache-policy4-capacity-maxima');out.mkdir(exist_ok=True)
ref=json.loads(Path('/tmp/strata-upstream-arc-attn-paired-wall/base-p1/run.json').read_text());a=ref['runs'][0]['args']
env=dict(os.environ,**ref['env'])
for k in list(env):
 if k.startswith('STRATA_'):env.pop(k)
env.update(STRATA_IO_THREADS='16',STRATA_PREFILL_RING='8',STRATA_PREFILL_FIRST='0',
 STRATA_PREFILL_ATTN_BATCH='128',STRATA_PREFILL_ATTN_LAYOUT='4',STRATA_PREFILL_TOPK_TUNED='1',
 STRATA_GDN_KEYHEAD='0',STRATA_GDN_KEYHEAD_TUNED='1',SYCL_CACHE_PERSISTENT='1')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(exe)=='df046f89f1ffe0437240d5202844f21986f8358bf4bfe10dfbfbbf2856580115'
report=dict(binary_sha256=sha(exe),runs=[],measurement='Fixed 65,538-position K8/V8 context, 65,536 input tokens, 4K chunks, 128 decode cache slots. Compare the existing layer-major 32K GPU prefix with full 64K GPU residual storage under cache release policies. CLI initializes the verifier after prefill; actual graph recreation is measured separately in server tests. Single transfer-timed observations, not repeated medians.')
support=Path('/tmp/strata-upstream-arc-layer-major-ram-ple.py');tree=ast.parse(support.read_text())
ns=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('save','memory','run_observed')]
exec(compile(ast.Module(body=ns,type_ignores=[]),str(support),'exec'))
report['support_sha256']=sha(support);save()
def service():return subprocess.check_output(['systemctl','--user','show','llama-server-qwen3embed.service','-p','ActiveState'],text=True).strip()
assert service()=='ActiveState=inactive'
common=[sys.executable,str(root/'sycl/tools/prefill_profile.py'),'--exe',str(exe),
 '--pack',a[a.index('--pack')+1],'--native',a[a.index('--native')+1],
 '--expert-profile',a[a.index('--expert-profile')+1],'--expert-cache','128','--cwd',str(root),
 '--repeats','1','--chunk','4096','--max-context','65538','--kv','int8','--multiple-chunks',
 '--tokens-file','/tmp/strata-upstream-arc-long-context-fixture/coding-context-64k-tokens.txt',
 '--lengths','65536','--ple-io','direct','--mode','transfer','--timeout','1800']
heads=[]
while True:
 previous=json.loads(Path('/tmp/strata-upstream-arc-cache-policy4-capacity-controls/summary.json').read_text())
 if previous.get('terminal_error'):raise RuntimeError(previous['terminal_error'])
 if previous.get('completed'):break
 time.sleep(5)
try:
 for name,gpu,release,source,allocation in [('retain-prefix56',57344,0,'ram','vmm'),('ram-half-prefix60',61440,1,'ram','vmm')]:
  assert service()=='ActiveState=inactive'
  try:
   dest,record=run_observed(name,common+['--label',name,'--output',str(out/name)],
    dict(env,STRATA_PREFILL_COMPACT='2',STRATA_PREFILL_LAYER_MAJOR='1',
    STRATA_PREFILL_LAYER_MAJOR_R_GPU=str(gpu),STRATA_PREFILL_RELEASE_CACHE=str(release),
    STRATA_PREFILL_CACHE_RESTORE=source,STRATA_PREFILL_CACHE_ALLOC=allocation,
    STRATA_PREFILL_CACHE_RELEASE_FRAC='.5'))
  except AssertionError:
   record=report['runs'][-1]
   logs=list((out/name).glob('*.log'))
   text='\n'.join(p.read_text() for p in logs)
   if record['exit_code']!=0 and 'requested GPU residual rows exceed free VRAM' in text:
    record['expected_capacity_failure']=True;save();print(name+' capacity refusal; cache restoration completed without abort',flush=True);continue
   raise
  actual=json.loads((dest/'run.json').read_text())['runs'][0]
  assert actual['logits_finite'] and actual['chunks']==16 and 'phases' not in actual and 'preload' not in actual
  signature=(actual['logits_sha256'],actual['logits_count'],actual['output_ids']);heads.append(signature)
  record.update(prefill_wall_ms=actual['wall_ms'],tokens_per_second=actual['tokens_per_second'],transfer=actual['transfer'])
  assert actual['transfer']['activation_bytes']==2*47*(65536-gpu)*10240*4 and actual['transfer']['residual_gpu_tokens']==gpu
  control=json.loads(Path('/tmp/strata-upstream-arc-cache-policy4-full64/retain-prefix32/run.json').read_text())['runs'][0]
  assert signature==(control['logits_sha256'],control['logits_count'],control['output_ids']), 'Complete head/IDs differ from same-cache control'
  save();print(name+' complete head finite; '+str(actual['tokens_per_second'])+' tok/s',flush=True)
 report['all_head_ids_identical']=True;report['completed']=True;report['service_after']=service();save()
except BaseException as e:
 report['terminal_error']=repr(e);report['completed']=False;report['service_after']=service();save();raise
