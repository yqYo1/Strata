import hashlib, json, os, subprocess, sys, time
from pathlib import Path

root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
out=Path('/tmp/strata-upstream-arc-cache-policy4-serve');out.mkdir(exist_ok=True)
exe=Path('/tmp/strata-upstream-arc-cache-policy4-jit')
refpath=Path('/tmp/strata-upstream-arc-attn-paired-wall/base-p1/run.json')
ref=json.loads(refpath.read_text());env=dict(os.environ,**ref['env'])
for k in list(env):
    if k.startswith('STRATA_'):env.pop(k)
env.update(STRATA_IO_THREADS='16',STRATA_PREFILL_RING='8',STRATA_PREFILL_FIRST='0',
    STRATA_PREFILL_ATTN_BATCH='128',STRATA_PREFILL_ATTN_LAYOUT='4',STRATA_PREFILL_TOPK_TUNED='1',
    STRATA_GDN_KEYHEAD='0',STRATA_GDN_KEYHEAD_TUNED='1',SYCL_CACHE_PERSISTENT='1')
report=dict(runs=[],completed=False,measurement='Six cache policies, four normal-MTP requests each (1025-token prompts / 128-token chunks), includes repeat and checkpoint restoration. A later fresh prompt forces cache release and rebuild after decode graphs already exist. Single correctness observations, not performance medians.')
def save():(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
save()
def run(name,cmd,run_env):
    print(name+' start',flush=True)
    with (out/(name+'.controller.log')).open('w') as log:
        child=subprocess.Popen(cmd,cwd=root,env=run_env,stdout=log,stderr=subprocess.STDOUT)
        rc=child.wait()
    assert rc==0,(name,rc)
    print(name+' complete',flush=True)
try:
    # The previous finite controller owns the GPU until it completes.
    prior=Path('/tmp/strata-upstream-arc-cache-policy3-check/summary.json')
    while True:
        previous=json.loads(prior.read_text())
        if previous.get('terminal_error'):raise RuntimeError(previous['terminal_error'])
        if previous.get('completed'):break
        time.sleep(5)
    while not exe.exists():time.sleep(5)
    report['binary_sha256']=hashlib.sha256(exe.read_bytes()).hexdigest();save()
    run('ctest',['ctest','--test-dir',str(root/'build-sycl-upstream-jit'),'--output-on-failure','--timeout','180','-j','1','-R','^ple_parity$'],dict(env,STRATA_PLE_GGUF=str(Path.home()/'.local/share/strata-sycl/models/qwen3.8-flash-next-q2_0/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf')))
    report['ctest_passed']=True;report['ctest_initial_attempt']='../strata-upstream-arc-cache-policy4-ctest-missing-fixture';report['ctest_note']='Initial 29/30 passed; ple_parity required STRATA_PLE_GGUF. Only that fixture-dependent test rerun with the required GGUF path.';save()
    run('full-state',[sys.executable,'/tmp/strata-upstream-arc-cache-policy4-check.py'],env)
    control=None
    for name,release,source,fraction,allocation in [('control',0,'snapshot',1,'vmm'),
        ('snapshot-full',1,'snapshot',1,'vmm'),('ram-full',1,'ram',1,'vmm'),
        ('snapshot-half',1,'snapshot',.5,'vmm'),('ram-half',1,'ram',.5,'vmm'),
        ('rebuild-ram',1,'ram',1,'rebuild')]:
        dest=out/name;dest.mkdir(exist_ok=True)
        assert subprocess.check_output(['systemctl','--user','show','llama-server-qwen3embed.service','-p','ActiveState'],text=True).strip()=='ActiveState=inactive'
        run_env=dict(env,STRATA_PREFILL_COMPACT='2',STRATA_PREFILL_LAYER_MAJOR='1',
            STRATA_PREFILL_LAYER_MAJOR_R_GPU='1024',STRATA_PREFILL_RELEASE_CACHE=str(release),
            STRATA_PREFILL_CACHE_RESTORE=source,STRATA_PREFILL_CACHE_RELEASE_FRAC=str(fraction),
            STRATA_PREFILL_CACHE_ALLOC=allocation,STRATA_PREFILL_DUMP_STATE=str(dest/'state.bin'))
        cmd=[sys.executable,'/tmp/strata-upstream-arc-cache-policy-serve-helper.py','--engine',str(exe),
            '--out',str(dest/'result.json'),'--reference',str(refpath)]
        run(name,cmd,run_env)
        result=json.loads((dest/'result.json').read_text())
        signature=[(r['ids'],r['logprobs']) for r in result['requests']]
        head=result['requests'][0]['logits_sha256']
        state=hashlib.sha256((dest/'state.bin').read_bytes()).hexdigest()
        if control is None:control=(signature,head,state)
        assert (signature,head,state)==control,(name,'head, decode IDs/logprobs or post-restoration state differs')
        record=dict(name=name,all_ids_logprobs_head_state_identical=True,result=result)
        report['runs'].append(record);save()
    report['completed']=True;save()
except BaseException as e:
    report['terminal_error']=repr(e);save();raise
