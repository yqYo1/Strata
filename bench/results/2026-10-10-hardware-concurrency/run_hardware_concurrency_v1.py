"""Root-only serial capacity controls; safe flags unchanged, no model/retry."""
from pathlib import Path
import datetime, fcntl, hashlib, json, math, os, re, subprocess, sys, types

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/docs-sycl-storage-retention-20261009')
SOURCE = Path(__file__).with_name('hardware_concurrency_v1.cpp')
OWNER = B / 'run_gdn_gate_factor_probe_v2.py'
OWNER_SHA = '7df012ee4d9bd047a6094ccedb37fddb3d56d4b054d9fdf468534b0fb6e94e1f'
BINARY = B / 'hardware-concurrency-v1'
STAGE = sys.argv[1]
assert STAGE in ['build', 'qualify', 'r1', 'r2', 'r3', 'profile']

def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def ident(p): return dict(bytes=Path(p).stat().st_size, sha256=sha(p))
def closed(stage):
    p = B / f'hardware-concurrency-v1-{stage}/record.json'
    d = json.loads(p.read_text())
    assert d['complete'] and d['passed'] and not d['active']
    return p, d

with (B / 'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert sha(OWNER) == OWNER_SHA
    assert not Path('/sys/bus/pci/devices/0000:05:00.0/devcoredump').exists()
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=W, text=True).strip()
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=W, text=True)
    module = types.ModuleType('hardware_concurrency_owner')
    owner_source = OWNER.read_text().split('\ndef parse_probe(', 1)[0]
    if STAGE == 'profile':
        # Enforce the total trace budget during supervision and again at exit,
        # in addition to the original stdout/stderr and per-file limits.
        needle = 'so.stat().st_size + se.stat().st_size <= text_cap'
        assert owner_source.count(needle) == 2
        owner_source = owner_source.replace(needle, needle + ' and sum(p.stat().st_size for p in self.output.rglob("*") if p.is_file()) <= (64 << 20)')
    exec(compile(owner_source, str(OWNER), 'exec'), module.__dict__)
    out = B / f'hardware-concurrency-v1-{STAGE}'
    assert not out.exists(); out.mkdir(mode=0o700); module.W = out
    owner = module.Owner(out)
    record = dict(active=True, complete=False, passed=False, stage=STAGE, started_utc=module.utc(),
                  boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                  source=dict(path=str(SOURCE), **ident(SOURCE)), controller=dict(path=__file__, **ident(__file__)),
                  parent_owner=dict(path=str(OWNER), **ident(OWNER)), source_commit=head,
                  owner_source_sha256=hashlib.sha256(owner_source.encode()).hexdigest(),
                  commands=owner.commands, adopted=False, model_executed=False,
                  scope='Independent copy/GEMM concurrent capacity, no production critical-path or physical-engine claim',
                  limits=dict(AS_each_bytes=16<<30, RSS_session_bytes=2<<30, CPU_each_seconds=[120,121], wall_seconds=240, text_bytes=8<<20, total_profile_bytes=64<<20))
    def save():
        tmp = out / 'record.json.tmp'; tmp.write_text(json.dumps(record, indent=2)+'\n'); tmp.replace(out/'record.json')
    owner.persist = save
    def run(label, args, env, wall=120, file_cap=16<<20):
        entry, stdout, stderr = owner.run(label, args, env, wall=wall, text_cap=8<<20, file_cap=file_cap)
        save(); assert module.completed(entry) and entry['exit_code']==0, (label, entry)
        return stdout, stderr
    try:
        clean = dict(PATH='/usr/bin:/bin', LANG='C.UTF-8', LC_ALL='C.UTF-8')
        original_build = B/'hardware-ceilings-v1-build-gpu/record.json'
        original = json.loads(original_build.read_text()); assert original['passed'] and original['complete'] and not original['active']
        env = dict(original['environment'])
        if STAGE == 'build':
            assert not BINARY.exists()
            args = ['/opt/intel/oneapi/compiler/2026.1/bin/icpx','-std=c++20','-O3','-fsycl','-fp-model=precise','-fsycl-default-sub-group-size=32','-fsycl-device-code-split=per_kernel','-qmkl=sequential','-Xsycl-target-backend=spir64','-cl-fp32-correctly-rounded-divide-sqrt',str(SOURCE),'-lze_loader','-o',str(BINARY)]
            run('compile', args, env, wall=240)
        else:
            p, build = closed('build')
            assert build['source'] == record['source'] and build['binary'] == dict(path=str(BINARY), **ident(BINARY))
            record['build_receipt'] = dict(path=str(p), **ident(p))
            if STAGE != 'qualify':
                p, qualification = closed('qualify'); assert qualification['binary'] == build['binary']
                record['qualification_receipt'] = dict(path=str(p), **ident(p))
            if STAGE in ['r2', 'r3', 'profile']:
                for prior in (['r1','r2','r3'] if STAGE=='profile' else ['r1'] if STAGE=='r2' else ['r1','r2']):
                    p, previous = closed(prior); assert previous['binary']==build['binary']
            env.update(UR_ADAPTERS_FORCE_LOAD='/opt/intel/oneapi/compiler/2026.1/lib/libur_adapter_level_zero_v2.so.0',
                       UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1', ONEAPI_DEVICE_SELECTOR='level_zero:gpu', SYCL_CACHE_PERSISTENT='0', EnableDirectSubmission='0', NEOReadDebugKeys='1')
            if STAGE == 'qualify':
                env.update(ZEL_ENABLE_LOADER_LOGGING='1',ZEL_LOADER_LOG_CONSOLE='1',ZEL_LOADER_LOGGING_LEVEL='warn',ZEL_LOADER_LOGGING_ENABLE_SUCCESS_PRINT='0',ZE_ENABLE_VALIDATION_LAYER='1',ZE_ENABLE_PARAMETER_VALIDATION='1',UR_LOG_LOADER='level:warning;flush:warning;output:stderr',UR_LOG_LEVEL_ZERO='level:warning;flush:warning;output:stderr')
            before, _ = run('kernel-cursor',['/usr/bin/journalctl','-k','-n','0','--show-cursor','--no-pager'],clean,wall=10)
            cursor = re.search(r'^-- cursor: (.+)$', before.read_text(), re.M); assert cursor
            record['kernel_cursor']=cursor[1]
            record['cpu_affinity']=sorted(os.sched_getaffinity(0))
            record['cpu_frequency_policy']={str(p):p.read_text().strip() for name in ['scaling_governor','scaling_min_freq','scaling_max_freq','energy_performance_preference'] for p in Path('/sys/devices/system/cpu/cpufreq').glob('policy*/'+name)}
            mode = '--qualify' if STAGE in ['qualify','profile'] else '--measure'
            args = [str(BINARY), mode]
            if STAGE == 'profile':
                profiler = B/'unitrace-build-system-cc/build/unitrace'
                library = profiler.with_name('libunitrace_tool.so')
                assert sha(profiler)=='5f90c453fbaf0b90a357a065cf4392f1ce3b157d5269af165c7cfe565cef1362'
                assert sha(library)=='c544dd2f6d2f6531cd529a9cc4ab9076b705eca8d63df56505e500942d128266'
                record['profiler']={str(p):ident(p) for p in [profiler,library]}
                args=[str(profiler),'-d','--chrome-kernel-logging','--chrome-call-logging','--output-dir-path',str(out),'-o',str(out/'device-summary.txt')]+args
                record['profiling_perturbation']=True
            record['environment']=env; save()
            stdout, stderr = run('benchmark',args,env,wall=240,file_cap=32<<20)
            rows=[json.loads(line) for line in stdout.read_text().splitlines()]
            assert rows[0]['kind']=='configuration' and rows[0]['pci']=='0000:05:00.0' and rows[0]['same_context'] and not rows[0]['profiling']
            assert rows[-1]==dict(kind='PASS',normal_release=True)
            sample_count=1 if mode=='--qualify' else 7
            samples=[x for x in rows if x['kind']=='sample']; validation=[x for x in rows if x['kind']=='validation']
            expected={(role,m,case,s) for role in ['GU','Down'] for m in [80,160,8192] for case in ['copy_only','compute_only','serialized','concurrent'] for s in range(sample_count)}
            keys=[(x['role'],x['M'],x['case'],x['sample']) for x in samples]
            assert len(keys)==len(set(keys))==len(expected) and set(keys)==expected
            assert len(validation)==6 and all(x['passed'] for x in validation)
            assert {(x['role'],x['M']) for x in validation}=={(role,m) for role in ['GU','Down'] for m in [80,160,8192]}
            for row in samples:
                n,k=(1280,2560) if row['role']=='GU' else (2560,640)
                calls=8 if row['M']==8192 else 128
                assert row['N']==n and row['K']==k and row['gemm_calls']==(0 if row['case']=='copy_only' else calls)
                assert row['dense_flops']==2*n*k*row['M']*row['gemm_calls']
                assert row['copy_calls']==(0 if row['case']=='compute_only' else 4) and row['logical_copy_bytes']==row['copy_calls']*(8<<20)
                assert math.isfinite(row['seconds']) and row['seconds']>0
            if STAGE in ['r1','r2','r3']: assert not stderr.read_text()
            record['result_rows']=rows; record['benchmark_stderr']=dict(path=str(stderr),**ident(stderr))
            after, _ = run('kernel-interval',['/usr/bin/journalctl','-k','--after-cursor='+cursor[1],'--no-pager','-o','short-monotonic'],clean,wall=10)
            record['visible_kernel_GPU_entries']=[line for line in after.read_text().splitlines() if re.search(r'\bxe\b|i915|GPU HANG|devcoredump',line,re.I)]
            record['devcoredump_after']=Path('/sys/bus/pci/devices/0000:05:00.0/devcoredump').exists()
            assert not record['visible_kernel_GPU_entries'] and not record['devcoredump_after']
            if STAGE=='profile':
                files=[p for p in out.rglob('*') if p.is_file()]
                assert sum(p.stat().st_size for p in files)<=64<<20
                record['profile_files']={str(p):ident(p) for p in files if p.name not in ['record.json','record.json.tmp']}
        record['binary']=dict(path=str(BINARY),**ident(BINARY)); record['environment']=env
        assert record['source']==dict(path=str(SOURCE),**ident(SOURCE)) and ident(__file__)=={k:record['controller'][k] for k in ['bytes','sha256']}
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip()==head and not subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True)
        record.update(passed=True,complete=True)
    except BaseException as error: record['error']=type(error).__name__+': '+str(error)
    finally:
        record.update(active=owner.active is not None,finished_utc=module.utc())
        record['passed']=record['passed'] and not record['active'] and all(module.completed(entry) for entry in owner.commands)
        save(); print(json.dumps({k:record.get(k) for k in ['stage','passed','complete','active','error']}),flush=True)
    if not record['passed']: sys.exit(1)
