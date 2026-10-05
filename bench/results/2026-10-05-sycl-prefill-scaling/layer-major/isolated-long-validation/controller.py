import ast,hashlib,json,os,re,subprocess,sys,threading,time
from pathlib import Path

root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
exe=Path('/tmp/strata-upstream-arc-layer-major-aot')
out=Path('/tmp/strata-upstream-arc-isolated-validation')
out.mkdir(exist_ok=True)
original_path=Path('/tmp/strata-upstream-arc-layer-major-c8-confirm/baseline-k8v8-32768-c8192/run.json')
original_report=json.loads(original_path.read_text());original=original_report['runs'][0]
expected=(original['logits_sha256'],original['logits_count'],original['output_ids'])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
proof=json.loads(Path('/tmp/strata-upstream-arc-layer-major-aot-initial/summary.json').read_text())
assert len(proof['runs'])==6 and all(r.get('all_bytes_identical_to_jit') for r in proof['runs'])
assert sha(exe)==proof['binary_sha256']
report=dict(binary_sha256=sha(exe),runs=[],measurement='Embedding service stopped by user before this finite validation sequence. Fixed 65,538-position K8/V8 context, 32K input, 8K chunks; complete first-head/ID comparisons. AOT single observations and three JIT wall pairs are recorded separately. Production RAM PLE loads at startup, outside prefill time but inside process time.')
support=Path('/tmp/strata-upstream-arc-layer-major-ram-ple.py')
tree=ast.parse(support.read_text());selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('save','memory','run_observed')]
assert len(selected)==3
exec(compile(ast.Module(body=selected,type_ignores=[]),str(support),'exec'))
report['support_sha256']=sha(support)
def service_state():
    return subprocess.check_output(['systemctl','--user','show','llama-server-qwen3embed.service','-p','ActiveState','-p','SubState','-p','MainPID'],text=True).strip()
report['service_before']=service_state()
assert 'ActiveState=inactive' in report['service_before'],report['service_before']
save()
env=dict(os.environ,**original_report['env'])
for k in ('STRATA_PREFILL_TIMING','STRATA_PREFILL_TRANSFER_TIMING','STRATA_PREFILL_PRELOAD_PLE',
          'STRATA_SERVE_NO_MTP','STRATA_DUMP_FIRST_LOGITS','STRATA_PREFILL_DUMP_STATE',
          'STRATA_PREFILL_DUMP_R','STRATA_PREFILL_DUMP_R_ALL'):
    env.pop(k,None)
env.update(STRATA_PREFILL_ATTN_BATCH='128',STRATA_PREFILL_ATTN_LAYOUT='4',STRATA_PREFILL_TOPK_TUNED='1',
    STRATA_GDN_KEYHEAD='0',STRATA_GDN_KEYHEAD_TUNED='1')
a=original['args']
common=[sys.executable,str(root/'sycl/tools/prefill_profile.py'),'--exe',str(exe),
    '--pack',a[a.index('--pack')+1],'--native',a[a.index('--native')+1],
    '--expert-profile',a[a.index('--expert-profile')+1],'--expert-cache','128','--cwd',str(root),
    '--repeats','1','--chunk','8192','--max-context','65538','--kv','int8','--multiple-chunks',
    '--tokens-file','/tmp/strata-upstream-arc-long-context-fixture/coding-context-64k-tokens.txt',
    '--lengths','32768','--timeout','1800']
def cli(name,compact,major,ple,mode):
    assert 'ActiveState=inactive' in service_state(),'Peer restarted; equivalent-condition run would be invalid'
    args=common+['--ple-io',ple,'--mode',mode,'--label',name,'--output',str(out/name)]
    dest,record=run_observed(name,args,dict(env,STRATA_PREFILL_COMPACT=str(compact),
        STRATA_PREFILL_LAYER_MAJOR=str(major),STRATA_PREFILL_LAYER_MAJOR_R_GPU='32768'))
    actual=json.loads((dest/'run.json').read_text())['runs'][0]
    assert actual['logits_finite'] and actual['chunks']==4
    assert (actual['logits_sha256'],actual['logits_count'],actual['output_ids'])==expected
    assert 'preload' not in actual and 'phases' not in actual
    assert 'ActiveState=inactive' in service_state(),'Peer became active during measurement'
    record.update(prefill_wall_ms=actual['wall_ms'],tokens_per_second=actual['tokens_per_second'],
        full_head_and_ids_identical_to_original=True,ple_io=ple,mode=mode)
    if ple=='ram':
        record.update(ple_table_locked=actual['ple_table_locked'],ple_table_startup_seconds=actual['ple_table_startup_seconds'])
    if mode=='transfer':record['transfer']=actual['transfer']
    else:assert 'transfer' not in actual
    save();print(name+' PASS complete head and IDs; '+str(actual['tokens_per_second'])+' tok/s',flush=True)
try:
    cli('aot-direct-gpu32-wall',2,1,'direct','wall')
    print('Three alternating JIT original/layer-major wall pairs start',flush=True)
    args=[sys.executable,'/tmp/strata-upstream-arc-layer-major-paired-wall.py']
    with (out/'paired-controller.log').open('w') as log:
        subprocess.run(args,cwd=root,check=True,stdout=log,stderr=subprocess.STDOUT)
    paired=Path('/tmp/strata-upstream-arc-layer-major-paired-wall/summary.json')
    p=json.loads(paired.read_text());assert len(p['pairs'])==6
    report['jit_paired_wall']=p;save()
    print('Three JIT wall pairs PASS full heads, IDs and memory observations',flush=True)
    cli('aot-ram-gpu32-transfer',2,1,'ram','transfer')
    cli('aot-ram-original-wall',0,0,'ram','wall')
    cli('aot-ram-gpu32-wall',2,1,'ram','wall')
    report['completed']=True;save()
    print('PASS isolated long-input validation and RAM PLE observations',flush=True)
finally:
    # Restore the temporarily paused service after this finite measurement set.
    # Do not start it beside a still-running engine if the controller is interrupted.
    active=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            cmd=(proc/'cmdline').read_bytes().split(b'\0')
            if cmd and cmd[0] in (os.fsencode(exe),b'/tmp/strata-upstream-arc-layer-major-gpu-jit'):active.append(proc.name)
        except OSError:pass
    if active:
        report['service_restore_deferred_for_active_engine_pids']=active
    else:
        result=subprocess.run(['systemctl','--user','start','llama-server-qwen3embed.service'],capture_output=True,text=True)
        report.update(service_restore_exit_code=result.returncode,service_restore_stderr=result.stderr,
                      service_after=service_state())
        print('Embedding service restore:',report['service_after'],flush=True)
    save()
