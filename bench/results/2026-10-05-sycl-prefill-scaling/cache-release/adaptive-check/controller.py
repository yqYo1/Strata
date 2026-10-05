import hashlib,json,os,re,subprocess,sys,time
from pathlib import Path
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
exe=Path('/tmp/strata-upstream-arc-cache-policy6-jit')
out=Path('/tmp/strata-upstream-arc-cache-adaptive-check');out.mkdir(exist_ok=True)
refpath=Path('/tmp/strata-upstream-arc-attn-paired-wall/base-p1/run.json')
ref=json.loads(refpath.read_text());env=dict(os.environ,**ref['env'])
for k in list(env):
    if k.startswith('STRATA_'):env.pop(k)
env.update(STRATA_IO_THREADS='16',STRATA_PREFILL_RING='8',STRATA_PREFILL_FIRST='0',
    STRATA_PREFILL_ATTN_BATCH='128',STRATA_PREFILL_ATTN_LAYOUT='4',STRATA_PREFILL_TOPK_TUNED='1',
    STRATA_GDN_KEYHEAD='0',STRATA_GDN_KEYHEAD_TUNED='1',SYCL_CACHE_PERSISTENT='1')
report=dict(runs=[],completed=False,measurement='Actual adaptive cache swaps every window (4 slots per round), four 16-token normal MTP requests per policy. Verify all occupied released-tail weight payloads before and after RAM restoration; compare head, IDs/logprobs and final dumped state across policies. Diagnostic copies enabled; no speed estimates from these runs.')
def save():(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
save()
while not exe.exists():time.sleep(5)
report['binary_sha256']=hashlib.sha256(exe.read_bytes()).hexdigest();save()
control=None
try:
 for name,release,source,fraction,allocation in [('control',0,'ram',1,'vmm'),
  ('snapshot-full',1,'snapshot',1,'vmm'),('ram-full',1,'ram',1,'vmm'),
  ('ram-half',1,'ram',.5,'vmm'),('rebuild-ram',1,'ram',1,'rebuild')]:
  assert subprocess.check_output(['systemctl','--user','show','llama-server-qwen3embed.service','-p','ActiveState'],text=True).strip()=='ActiveState=inactive'
  dest=out/name;dest.mkdir(exist_ok=True)
  run_env=dict(env,STRATA_PREFILL_COMPACT='2',STRATA_PREFILL_LAYER_MAJOR='1',
   STRATA_PREFILL_LAYER_MAJOR_R_GPU='1024',STRATA_PREFILL_RELEASE_CACHE=str(release),
   STRATA_PREFILL_CACHE_RESTORE=source,STRATA_PREFILL_CACHE_RELEASE_FRAC=str(fraction),
   STRATA_PREFILL_CACHE_ALLOC=allocation,STRATA_PREFILL_CACHE_VERIFY='1',STRATA_PREFILL_DUMP_STATE=str(dest/'state.bin'))
  cmd=[sys.executable,'/tmp/strata-upstream-arc-cache-adaptive-serve-helper.py','--engine',str(exe),
   '--out',str(dest/'result.json'),'--reference',str(refpath)]
  print(name+' start',flush=True)
  with (dest/'controller.log').open('w') as log:
   child=subprocess.Popen(cmd,cwd=root,env=run_env,stdout=log,stderr=subprocess.STDOUT);rc=child.wait()
  assert rc==0,(name,rc)
  result=json.loads((dest/'result.json').read_text())
  signature=[(r['ids'],r['logprobs']) for r in result['requests']]
  head=result['requests'][0]['logits_sha256'];state=hashlib.sha256((dest/'state.bin').read_bytes()).hexdigest()
  if control is None:control=(signature,head,state)
  assert (signature,head,state)==control,(name,'adaptive head/output/logprobs/state differs')
  text=(dest/'result.log').read_text()
  checks=re.findall(r'strata prefill cache verify: (\d+) occupied tail bytes matched before and after restoration; (\d+) experts differ from the initial admission map',text)
  if release:
   assert len(checks)>=2 and all(int(b)>0 for b,n in checks),(name,'missing complete-cache weight verification')
   if source=='ram':assert any(int(n)>0 for b,n in checks),(name,'no actual changed cache mapping exercised')
  report['runs'].append(dict(name=name,result=result,all_heads_ids_logprobs_state_equal=True,payload_verification=[dict(bytes=int(b),changed_experts=int(n)) for b,n in checks]));save()
  print(name+' PASS adaptive outputs/state; all occupied cache tail bytes verified',flush=True)
 report['completed']=True;save()
except BaseException as e:
 report['terminal_error']=repr(e);save();raise
