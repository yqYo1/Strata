"""Root-owned CPU build of real GDN factor qualification; no GPU/model run."""
from pathlib import Path
import datetime,fcntl,hashlib,json,os,shlex,subprocess,sys,types

B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/perf-sycl-gdn-prefill-gate-factor-20261010')
G=Path('/home/yayoi/orca/workspaces/Strata/ggml-sycl-pinned')
BUILD=W/'build-sycl-gdn-factor-real-v1'
OUT=B/'gdn-factor-real-cpu-build-v1'
HEAD='cbc77580050a327538be48b0ba7a759c14f7a49f'
PARENT_OWNER=B/'run_gdn_gate_factor_probe_v2.py'
PARENT_HASH='7df012ee4d9bd047a6094ccedb37fddb3d56d4b054d9fdf468534b0fb6e94e1f'

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def git(root,*args):return subprocess.check_output(['/usr/bin/git','-C',str(root),*args],text=True,timeout=20).strip()
def ident(path):return dict(bytes=Path(path).stat().st_size,sha256=sha(path))

def main():
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert git(W,'rev-parse','HEAD')==HEAD and not git(W,'status','--porcelain')
        assert git(G,'rev-parse','HEAD')=='3cf03257f219afbe7334045ff7c6a06ac68c627d' and not git(G,'status','--porcelain')
        assert sha(PARENT_OWNER)==PARENT_HASH
        host=json.loads((B/'gdn-owner-controller-host-checks-v2/record.json').read_text())
        assert host['passed'] and host['complete'] and not host['active']
        assert not OUT.exists() and not BUILD.exists();OUT.mkdir(mode=0o700)
        source=PARENT_OWNER.read_text().split('\ndef parse_probe(',1)[0]
        # Keep the proven ownership/finalization code; enlarge only compiler limits.
        for old,new in [('(16 << 30, 16 << 30)','(32 << 30, 32 << 30)'),('(120, 121)','(600, 601)'),('rss <= 2 << 30','rss <= 12 << 30')]:
            assert source.count(old)==1,old;source=source.replace(old,new)
        owner_module=types.ModuleType('bounded_gdn_factor_build_owner')
        exec(compile(source,str(PARENT_OWNER)+'[build-limits]','exec'),owner_module.__dict__)
        owner_module.W=W
        owner=owner_module.Owner(OUT)
        files=['sycl/CMakeLists.txt','sycl/src/prefill/kernels.dp.cpp','sycl/src/prefill/prefill.cpp','sycl/src/prefill/gdn_gate_factor_parity.cpp','sycl/include/strata/prefill/gdn_gate_factor.hpp','sycl/include/strata/prefill/gdn_variant.hpp','sycl/include/dpct/device.hpp','sycl/include/strata/sycl_error.hpp']
        pins={rel:ident(W/rel) for rel in files}
        record=dict(active=True,complete=False,passed=False,GPU_executed=False,model_opened=False,adopted=False,
            scope='Production-flag full dependency CPU build, real GDN factor parity target only',root=str(W),source_head=HEAD,
            build=str(BUILD),source_pins=pins,controller_sha256=sha(__file__),parent_owner_sha256=PARENT_HASH,
            host_owner_receipt=ident(B/'gdn-owner-controller-host-checks-v2/record.json'),
            limits=dict(AS_each_bytes=32<<30,RSS_session_sampled_bytes=12<<30,CPU_each_seconds=[600,601],stage_wall_seconds=1200,text_cap_bytes=32<<20,file_cap_each_bytes=128<<20,owned_cleanup_grace_seconds=6),
            commands=owner.commands,started_utc=owner_module.utc(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
        def save():
            temp=OUT/'record.json.tmp'
            with temp.open('w') as stream:json.dump(record,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
            temp.replace(OUT/'record.json')
        owner.persist=save
        def run(label,argv,env,wall=120):
            entry,out,err=owner.run(label,argv,env,wall=wall,text_cap=32<<20,file_cap=128<<20)
            save();assert owner_module.completed(entry) and entry['exit_code']==0,(label,entry)
            return out
        try:
            clean=dict(PATH='/usr/bin:/bin',LANG='C.UTF-8',LC_ALL='C.UTF-8')
            run('owner-cpu-closure',['/usr/bin/true'],clean,wall=10)
            raw=run('toolchain-environment',['/bin/bash','-c','source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 && env -0'],clean,wall=30).read_bytes()
            discovered=dict(x.decode().split('=',1) for x in raw.split(b'\0') if b'=' in x)
            env=dict(clean)
            for key in ['PATH','LD_LIBRARY_PATH','LIBRARY_PATH','CPATH','ONEAPI_ROOT','CMAKE_PREFIX_PATH','PKG_CONFIG_PATH','CPLUS_INCLUDE_PATH','C_INCLUDE_PATH','MKLROOT']:
                if key in discovered:env[key]=discovered[key]
            compiler=Path('/opt/intel/oneapi/compiler/2026.1/bin')
            record.update(environment=env,compiler_pins={name:ident(compiler/name) for name in ['icx','icpx']});save()
            options=['-DCMAKE_CXX_COMPILER='+str(compiler/'icpx'),'-DCMAKE_C_COMPILER='+str(compiler/'icx'),'-DCMAKE_BUILD_TYPE=Release','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON','-DSTRATA_GGML_DIR='+str(G),'-DSTRATA_NATIVE_EXPERTS=ON','-DSTRATA_SYCL_PARITY=OFF','-DSTRATA_IQ2S_GCC=OFF','-DSTRATA_SYCL_AOT=','-DSTRATA_SYCL_SPIN_MAX=','-DSTRATA_NATIVE_POOL_TASK_FACTOR=0','-DSTRATA_GDN_PREFILL_GATE_FACTOR=ON','-DSTRATA_GDN_GATE_FACTOR_PARITY=ON']
            run('configure',['/usr/bin/cmake','-S',str(W/'sycl'),'-B',str(BUILD),'-G','Ninja',*options],env,wall=120)
            run('build',['/usr/bin/cmake','--build',str(BUILD),'--target','gdn_gate_factor_parity','--parallel','2'],env,wall=1200)
            commands=json.loads((BUILD/'compile_commands.json').read_text())
            selected={}
            for entry in commands:
                if Path(entry['file']).name in ['kernels.dp.cpp','prefill.cpp','gdn_gate_factor_parity.cpp']:
                    args=shlex.split(entry['command'])
                    for flag in ['-fsycl','-fp-model=precise','-fsycl-default-sub-group-size=32','-fsycl-device-code-split=per_kernel']:assert flag in args,(entry['file'],flag)
                    assert not set(args)&{'-ffast-math','-Ofast','-fassociative-math','-ffinite-math-only'}
                    selected[entry['file']]=args
            assert len(selected)==3
            imports=run('direct-imports',['/usr/bin/readelf','-d',str(BUILD/'gdn_gate_factor_parity')],clean,wall=30)
            assert 'libze_loader.so' in imports.read_text()
            target_commands=run('target-commands',['/usr/bin/ninja','-C',str(BUILD),'-t','commands','gdn_gate_factor_parity'],clean,wall=30)
            assert '-cl-fp32-correctly-rounded-divide-sqrt' in target_commands.read_text()
            record.update(passed=True,complete=True,binary=dict(path=str(BUILD/'gdn_gate_factor_parity'),**ident(BUILD/'gdn_gate_factor_parity')),compile_commands=ident(BUILD/'compile_commands.json'),selected_compile_commands=selected,target_commands=ident(target_commands))
        except BaseException as error:
            record['error']=type(error).__name__+': '+str(error)
        finally:
            try:
                assert git(W,'rev-parse','HEAD')==HEAD and not git(W,'status','--porcelain')
                assert git(G,'rev-parse','HEAD')=='3cf03257f219afbe7334045ff7c6a06ac68c627d' and not git(G,'status','--porcelain')
                assert {rel:ident(W/rel) for rel in files}==pins
                assert sha(PARENT_OWNER)==PARENT_HASH
                assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==record['boot_id']
            except BaseException as error:
                record['final_pin_error']=repr(error);record['passed']=False
            record.update(active=owner.active is not None,finished_utc=owner_module.utc())
            record['passed']=record['passed'] and not record['active'] and all(owner_module.completed(entry) for entry in owner.commands)
            save()
            print(json.dumps(dict(receipt=str(OUT/'record.json'),passed=record['passed'],complete=record['complete'],active=record['active'],error=record.get('error'))),flush=True)
        return 0 if record['passed'] else 1
if __name__=='__main__':sys.exit(main())
