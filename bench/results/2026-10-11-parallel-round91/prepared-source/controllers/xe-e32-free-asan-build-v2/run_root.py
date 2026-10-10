import argparse, fcntl, importlib.util, json, os, shutil, time
from pathlib import Path
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
HERE=Path(__file__).resolve().parent
WX=Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011')
LLVM=WX/'.tools/intel-llvm-v711/install'
GGML_REPO=Path('/home/yayoi/ghq/github.com/ggml-org/llama.cpp')
GGML=GGML_REPO/'.worktree/dep-ggml-3cf-xe-e32-20261011'
OWNER=B/'xe-llvm-v711-source-v3/direct_owner.py'
spec=importlib.util.spec_from_file_location('owner',OWNER);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.require(m.sha(OWNER)=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f','qualified long owner')
BASE=dict(PATH=str(LLVM/'bin')+':/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8',LD_LIBRARY_PATH=str(LLVM/'lib'))
MODES=['free']
CPU_TARGETS=['prefill_publication_test','commit_transaction_test','full_context_probe_test','mtp_completion_test','pool_tasks_test','pool_stress','iq_avx2_parity','q8k_quant_parity','q2_bitplane_parity','expert_multi_test']

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=MODES,required=True);a=p.parse_args()
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        proof=B/'xe-llvm-v711-source-root-v3/record.json';toolchain=json.loads(proof.read_text())
        m.require(toolchain['passed'] and not toolchain['active'] and toolchain['complete'],'closed qualified source compiler')
        out=B/('xe-e32-free-asan-full-build-root-v2');m.require(not out.exists(),'immutable output');out.mkdir()
        r=dict(active=True,complete=False,passed=False,commands=[],started_utc=m.utc(),controller_sha256=m.sha(__file__),owner_sha256=m.sha(OWNER),mode=a.mode,
               source_base='e32b8b0594095352365db0e140a68d5bf56d8b0f',source_tree=str(WX),compiler_receipt_sha256=m.sha(proof),
               model_executed=False,gpu_executed=False,scope='Full corrected latest Xe engine, explicit native-expert CPU tests; no device/runtime tests in this build phase',
               raw_log_budget_bytes=64<<20,total_wall_budget_seconds=7200,artifact_file_cap_bytes=16<<30,build_jobs=4,
               upstream_pristine=False,full262144_runtime_qualified=False,adopted=False)
        def save(): (out/'record.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
        owner=m.Owner(out/'commands',save);r['commands']=owner.commands;begin=time.monotonic();save()
        def cmd(label,args,cwd=WX,env=BASE,wall=60,rss=40<<30):
            m.require(time.monotonic()-begin<7200,'whole build deadline')
            wall=min(wall,7200-(time.monotonic()-begin))
            e,so,se=owner.run(label,args,env,cwd,wall=wall,cpu=max(10,int(wall)*4),file_cap=16<<30,text_cap=64<<20,total_cap=64<<20,rss_cap=rss)
            m.require(m.closed(e) and e['exit_code']==0,'normal closed '+label)
            return so.read_text()
        try:
            m.require(shutil.disk_usage(WX).free>40<<30,'build disk headroom')
            for f,item in toolchain['installed_files'].items(): m.require(m.sha(Path(item['path']))==item['sha256'],'compiler installation identity '+f)
            r['compiler_version']=cmd('compiler-version',[str(LLVM/'bin/clang++'),'--version'])
            m.require(cmd('source-head',['/usr/bin/git','-C',str(WX),'rev-parse','HEAD']).strip()==r['source_base'],'current base matches intended candidate')
            paths=cmd('modified-paths',['/usr/bin/git','-C',str(WX),'diff','--name-only','HEAD']).splitlines()+cmd('untracked-source-paths',['/usr/bin/git','-C',str(WX),'ls-files','--others','--exclude-standard']).splitlines()
            paths=sorted(set(paths+['CMakeLists.txt','AGENTS.md','src/prefill/gemm.cpp','src/core/gpu.cpp','src/core/expert_source.cpp','src/core/pinned.cpp']))
            m.require(all((WX/f).is_file() for f in paths),'all candidate paths exist')
            r['source_pins']={f:m.sha(WX/f) for f in paths}
            patch=cmd('source-patch',['/usr/bin/git','-C',str(WX),'diff','HEAD']);(out/'source.patch').write_text(patch);r['source_patch_sha256']=m.sha(out/'source.patch')
            if not GGML.exists():
                cmd('add-bound-ggml',['/usr/bin/git','-C',str(GGML_REPO),'worktree','add','-b','dep/ggml-3cf-xe-e32-20261011',str(GGML),'3cf03257f219afbe7334045ff7c6a06ac68c627d'])
            m.require(cmd('ggml-head',['/usr/bin/git','-C',str(GGML),'rev-parse','HEAD']).strip()=='3cf03257f219afbe7334045ff7c6a06ac68c627d','exact required ggml dependency')
            m.require(not cmd('ggml-status',['/usr/bin/git','-C',str(GGML),'status','--porcelain','--untracked-files=no']).strip(),'ggml source unchanged')
            r['ggml_tree']=str(GGML);r['ggml_revision']='3cf03257f219afbe7334045ff7c6a06ac68c627d'
            build=WX/'build'/'free-asan';m.require(not build.exists(),'new mode build directory')
            options=['-DCMAKE_BUILD_TYPE=RelWithDebInfo','-DSTRATA_SANITIZE=address,undefined','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON','-DSTRATA_ENABLE_XE=ON',
                     '-DSTRATA_LICENSE='+('free' if a.mode=='free' else 'contrib'),'-DSTRATA_CUDA_ARCHS=','-DSTRATA_HIP_ARCHS=',
                     '-DSTRATA_ONEMKL='+('OFF' if a.mode=='free' else 'ON'),'-DSTRATA_NATIVE_EXPERTS=ON',
                     '-DSTRATA_BUILD_TESTS=ON','-DSTRATA_WERROR=ON','-DSTRATA_GGML_DIR='+str(GGML),
                     '-DCMAKE_C_COMPILER='+str(LLVM/'bin/clang'),'-DCMAKE_CXX_COMPILER='+str(LLVM/'bin/clang++')]
            if a.mode!='free':options+=['-DMKL_ROOT=/opt/intel/oneapi/mkl/2026.1']
            cmd('configure',['/usr/bin/cmake','-S',str(WX),'-B',str(build),'-G','Ninja',*options],wall=900)
            for f in ['CMakeCache.txt','compile_commands.json','BUILD.json']:
                if (build/f).exists():r.setdefault('build_recipe',{})[f]=dict(path=str(build/f),sha256=m.sha(build/f),bytes=(build/f).stat().st_size)
            save()
            cmd('compile-full',['/usr/bin/nice','-n','10','/usr/bin/cmake','--build',str(build),'--parallel','4','--target','strata',*CPU_TARGETS],wall=5400,rss=80<<30)
            r['sanitizer']='address,undefined';r['host_code_only']=True
            r['binary']=dict(path=str(build/'strata'),sha256=m.sha(build/'strata'),bytes=(build/'strata').stat().st_size)
            r['ninja_compile_link_recipe']=cmd('ninja-recipe',['/usr/bin/ninja','-C',str(build),'-t','commands','strata'],wall=60)
            testenv=dict(BASE,ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1',LD_LIBRARY_PATH=str(LLVM/'lib')+(':/opt/intel/oneapi/mkl/2026.1/lib' if a.mode!='free' else ''))
            r['cpu_test_output']=cmd('cpu-ctest',['/usr/bin/ctest','--test-dir',str(build),'--output-on-failure','--timeout','180','-R','^('+'|'.join(CPU_TARGETS)+')$'],wall=900,env=testenv)
            r['source_stable']=all(m.sha(WX/f)==h for f,h in r['source_pins'].items());m.require(r['source_stable'],'candidate source stable during full build/tests')
            r['compiler_stable']=all(m.sha(Path(x['path']))==x['sha256'] for x in toolchain['installed_files'].values());m.require(r['compiler_stable'],'compiler installation stable')
            r['complete']=True
        except BaseException as exc:r['error']=repr(exc)
        finally:
            r['active']=owner.active is not None;r['passed']=bool(r['complete'] and not r['active'] and not r.get('error'));r['finished_utc']=m.utc();save()
        print(json.dumps({k:r.get(k) for k in ['mode','passed','error','binary','source_stable','cpu_test_output']}))
        return 0 if r['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
