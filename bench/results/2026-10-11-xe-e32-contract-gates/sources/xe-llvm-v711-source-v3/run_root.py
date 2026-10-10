import fcntl, hashlib, importlib.util, json, os, shutil, subprocess, time
from pathlib import Path

B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
HERE=Path(__file__).resolve().parent
WX=Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011')
REPO=Path('/home/yayoi/ghq/github.com/intel/llvm')
SRC=REPO/'.worktree/toolchain-sycl-v711-b570-20261011'
DEST=WX/'.tools/intel-llvm-v711'
OWNER=HERE/'direct_owner.py'
spec=importlib.util.spec_from_file_location('owner',OWNER);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.require(m.sha(OWNER)=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f','qualified finite owner')
BASE=dict(PATH='/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8')

def main():
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        out=B/'xe-llvm-v711-source-root-v3';m.require(not out.exists(),'immutable new output');out.mkdir()
        record=dict(active=True,complete=False,passed=False,commands=[],started_utc=m.utc(),controller_sha256=m.sha(__file__),
                    scope='Private source-built Intel LLVM7.1.1 compiler/runtime for latest Xe free and Intel-only contrib-llvm qualification',
                    model_executed=False,gpu_executed=False,queue_created=False,system_packages_installed=False,
                    clone_manager='ghq',source_tag='v7.1.1',source_commit_expected='504366f4b82ff00bbac7b0c956635eed8bc599d8',
                    install_consumer='latest e32 corrected free/contrib-llvm full engine builds',build_jobs=6,link_jobs=1,
                    raw_log_budget_bytes=64<<20,total_wall_budget_seconds=7200,compile_wall_budget_seconds=5400,owner_sha256=m.sha(OWNER),live_identity_cap=256,cumulative_identity_cap=200000,
                    build_storage_review='After successful install/hash and engine configure, retire compiler build intermediates; keep installed compiler and pinned source as current rebuild consumer.',
                    prereq_scope='lld18/hwloc/zstd development files extracted privately; no sudo/global install; no automatic xmx probe')
        def save(): (out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
        owner=m.Owner(out/'commands',save);record['commands']=owner.commands;begin=time.monotonic();save()
        def cmd(label,args,cwd=HERE,env=BASE,wall=60,rss=8<<30):
            m.require(time.monotonic()-begin<7200,'whole bootstrap deadline')
            wall=min(wall,7200-(time.monotonic()-begin))
            e,so,se=owner.run(label,args,env,cwd,wall=wall,text_cap=64<<20,rss_cap=rss,cpu=max(10,int(wall)*6),total_cap=64<<20,file_cap=16<<30)
            m.require(m.closed(e) and e['exit_code']==0,'closed normal '+label)
            return so.read_text()
        try:
            m.require(shutil.disk_usage(B).free>80<<30,'bounded toolchain storage headroom')
            previous=B/'xe-llvm-v711-source-root-v2/record.json'
            prior=json.loads(previous.read_text())
            m.require(prior['active'] is False and not prior['passed'] and prior['configuration_complete'],'closed configured prior failure')
            m.require(prior['error']=="RuntimeError('closed normal compile-install')",'exact prior compile failure')
            m.require(all(c.get('session_empty') and c.get('direct_child_reaped') for c in prior['commands']),'all prior descendants closed')
            lc=B/'xe-llvm-owner-lifecycle-root-v1/record.json';lr=json.loads(lc.read_text())
            m.require(lr['passed'] and not lr['active'] and lr['owner_sha256']==m.sha(OWNER),'actual long-build owner lifecycle qualification')
            record['owner_lifecycle_receipt_sha256']=m.sha(lc)
            record['prerequisite_packages']=prior['prerequisite_packages'];record['previous_failed_receipt_sha256']=m.sha(previous)
            record['artifact_file_cap_bytes']=16<<30
            record['previous_failure_diagnosis']='Prior compiler build hit cumulative256 child-pidfd count after86 of3952 objects, despite small simultaneous population. New owner retires confirmed-dead non-leader descriptors while retaining identity evidence; live256/cumulative200000 are finite independent caps. Prior status unchanged.'
            for package in record['prerequisite_packages']:
                m.require(m.sha(Path(package['path']))==package['sha256'],'reuse exact catalog-verified private dependency')
            deps=DEST/'deps';private_bin=DEST/'bin';lib=deps/'usr/lib/x86_64-linux-gnu'
            m.require((private_bin/'ld.lld').exists() and (deps/'usr/include/hwloc.h').exists() and (deps/'usr/include/zstd.h').exists(),'extracted prerequisites present')
            env=dict(BASE,PATH=str(private_bin)+':/usr/bin:/bin',CMAKE_PREFIX_PATH=str(deps/'usr'),PKG_CONFIG_PATH=str(lib/'pkgconfig'))
            record['build_environment']=env;save()
            cmd('lld-version',[str(private_bin/'ld.lld'),'--version'],env=env)
            record['mirror_default_branch']=cmd('default-branch',['/usr/bin/git','-C',str(REPO),'symbolic-ref','--short','HEAD']).strip()
            m.require(record['mirror_default_branch']==prior['mirror_default_branch'],'mirror branch unchanged')
            m.require(cmd('source-head',['/usr/bin/git','-C',str(SRC),'rev-parse','HEAD']).strip()==record['source_commit_expected'],'exact existing pinned source')
            record['source_commit']=record['source_commit_expected'];record['source_worktree']=str(SRC);record['source_patches']=prior['source_patches']
            for patch in record['source_patches']:
                m.require(m.sha(Path(patch['path']))==patch['sha256'],'patch identity stable')
            build=DEST/'build';install=DEST/'install'
            m.require((build/'build.ninja').exists() and (build/'CMakeCache.txt').exists(),'prior configured build present')
            m.require(m.sha(SRC/'buildbot/configure.py')==prior['configure_script_sha256'] and m.sha(SRC/'buildbot/compile.py')==prior['compile_script_sha256'],'build scripts unchanged')
            record['configuration_complete']=True;record['cache_sha256']=m.sha(build/'CMakeCache.txt');record['source_diff_sha256']=hashlib.sha256(cmd('source-diff',['/usr/bin/git','-C',str(SRC),'diff','HEAD']).encode()).hexdigest()
            record['configure_script_sha256']=prior['configure_script_sha256'];record['compile_script_sha256']=prior['compile_script_sha256'];save()
            cmd('compile-install',['/usr/bin/nice','-n','10','/usr/bin/python3',str(SRC/'buildbot/compile.py'),'-o',str(build),'-j','6'],env=env,wall=5400,rss=80<<30)
            record['installed_files']={}
            for f in ['bin/clang','bin/clang++','lib/libsycl.so']:
                p=install/f;m.require(p.exists(),'installed '+f);record['installed_files'][f]=dict(path=str(p),sha256=m.sha(p),bytes=p.stat().st_size)
            record['compiler_version']=cmd('installed-version',[str(install/'bin/clang++'),'--version'],env=dict(env,LD_LIBRARY_PATH=str(install/'lib')))
            record['complete']=True
        except BaseException as exc:record['error']=repr(exc)
        finally:
            record['active']=owner.active is not None;record['passed']=bool(record['complete'] and not record['active'] and not record.get('error'));record['finished_utc']=m.utc();save()
        print(json.dumps(dict(record=str(out/'record.json'),passed=record['passed'],error=record.get('error'),compiler_version=record.get('compiler_version'))))
        return 0 if record['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
