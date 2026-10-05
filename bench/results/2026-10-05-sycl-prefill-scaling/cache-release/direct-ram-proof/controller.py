import array, hashlib, json, math, os, re, subprocess, time
from pathlib import Path

root = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
exe = Path('/tmp/strata-upstream-arc-cache-policy4-jit')
out = Path('/tmp/strata-upstream-arc-cache-policy4-check'); out.mkdir(exist_ok=True)
ref = json.loads(Path('/tmp/strata-upstream-arc-attn-paired-wall/base-p1/run.json').read_text())
a = ref['runs'][0]['args']
env = dict(os.environ, **ref['env'])
for k in list(env):
    if k.startswith('STRATA_'): env.pop(k)
env.update(STRATA_IO_THREADS='16', STRATA_PREFILL_RING='8', STRATA_PREFILL_FIRST='0',
    STRATA_PREFILL_ATTN_BATCH='128', STRATA_PREFILL_ATTN_LAYOUT='4',
    STRATA_PREFILL_TOPK_TUNED='1', STRATA_GDN_KEYHEAD='0', STRATA_GDN_KEYHEAD_TUNED='1',
    SYCL_CACHE_PERSISTENT='1')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
report = dict(binary_sha256=sha(exe), runs=[], completed=False,
    measurement='Full head, all FP32 residual rows and persistent state equality at 257 tokens / 128-token chunks; diagnostic timings are not paired performance medians.')
def save(): (out/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
def value(k): return a[a.index(k)+1]
def service(): return subprocess.check_output(['systemctl','--user','show','llama-server-qwen3embed.service','-p','ActiveState'],text=True).strip()
save()
try:
    for name, release, mode, fraction, allocation in [('control',0,'snapshot',1,'vmm'), ('snapshot-full',1,'snapshot',1,'vmm'),
        ('ram-full',1,'ram',1,'vmm'), ('snapshot-half',1,'snapshot',.5,'vmm'), ('ram-half',1,'ram',.5,'vmm'), ('ram-retain',1,'ram',0,'vmm'), ('rebuild-ram',1,'ram',1,'rebuild'), ('rebuild-snapshot',1,'snapshot',1,'rebuild')]:
        assert service() == 'ActiveState=inactive'
        dest = out/name; dest.mkdir(exist_ok=True)
        head, state, residual = [dest/k for k in ('head.bin','state.bin','residual.bin')]
        run_env = dict(env, STRATA_PREFILL_COMPACT='2', STRATA_PREFILL_LAYER_MAJOR='1',
            STRATA_PREFILL_LAYER_MAJOR_R_GPU='257', STRATA_PREFILL_RELEASE_CACHE=str(release),
            STRATA_PREFILL_CACHE_RESTORE=mode, STRATA_PREFILL_CACHE_ALLOC=allocation, STRATA_PREFILL_CACHE_RELEASE_FRAC=str(fraction),
            STRATA_DUMP_FIRST_LOGITS=str(head), STRATA_PREFILL_DUMP_STATE=str(state),
            STRATA_PREFILL_DUMP_R=str(residual), STRATA_PREFILL_DUMP_R_ALL='1',
            STRATA_PREFILL_TRANSFER_TIMING='1')
        args = [str(exe),'--pack',value('--pack'),'--native',value('--native'),
            '--tokens-file','/tmp/strata-upstream-arc-layer-major-gpu-initial/baseline-257-c128/tokens.txt',
            '--max-new','1','--spec','2','--max-context','8194','--prefill','128','--no-prefill-borrow',
            '--expert-cache','128','--expert-profile',value('--expert-profile'),'--expert-cache-per-layer',
            '--pool-workers','5','--pcie-frac','0','--adapt-swaps','0','--ple-io','direct',
            '--suffix-draft','0','--greedy','--stats']
        print(name+' start',flush=True); start=time.monotonic()
        with (dest/'engine.log').open('w') as log:
            child=subprocess.Popen(args,cwd=root,env=run_env,stdout=log,stderr=subprocess.STDOUT)
            rc=child.wait()  # Finite max-new; never terminate an active GPU graph.
        record=dict(name=name,args=args,env={k:v for k,v in run_env.items() if k.startswith(('STRATA_','SYCL_','ONEAPI_')) or k=='LD_LIBRARY_PATH'},
            exit_code=rc,process_wall_seconds=time.monotonic()-start)
        report['runs'].append(record); save(); assert rc==0, name
        values=array.array('f'); values.frombytes(head.read_bytes())
        assert len(values)==248320 and all(map(math.isfinite,values))
        for key in ('head.bin','state.bin','residual.bin'):
            p=dest/key
            baseline=Path('/tmp/strata-upstream-arc-layer-major-gpu-initial/baseline-257-c128')/key
            assert p.stat().st_size==baseline.stat().st_size and sha(p)==sha(baseline),(name,key)
            record[key]=dict(bytes=p.stat().st_size,sha256=sha(p))
        text=(dest/'engine.log').read_text()
        record['cache_timing']=re.findall(r'^strata prefill cache release: (.*)$',text,re.M)
        record['head_residual_state_equal']=True
        save(); print(name+' PASS all head/residual/state bytes',flush=True)
    report.update(completed=True,service_after=service()); save()
except BaseException as e:
    report.update(terminal_error=repr(e),service_after=service()); save(); raise
