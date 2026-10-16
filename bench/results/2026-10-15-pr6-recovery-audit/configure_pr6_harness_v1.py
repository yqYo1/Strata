#!/usr/bin/env python3
"""Main owned offline PR6 configuration; no legacy engine or GPU execution."""
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode=True
A=Path(__file__).resolve().parent
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
C=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/verify-pr6-prefill-restore')
L=Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011/.tools/intel-llvm-v711/install')


def pin(p):
    p=Path(p)
    with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=str(p),realpath=str(p.resolve()),bytes=p.stat().st_size,sha256=h)


def main():
    root=B/'main-pr6-harness-configure-v1';target=A/'own-pr6-harness-configure-v1.json'
    assert not root.exists() and not target.exists()
    dependency=A/'own-pr6-dependency-preparation-v1.json'
    assert pin(dependency)['sha256']=='9709a3168804ef812073d71b75bdcb35af6a3b7cbe6e604061a2770812152fdd'
    dep=json.loads(dependency.read_text());assert dep['passed'] and dep['complete']
    owner_path=A.parents[2]/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'
    assert pin(owner_path)['sha256']=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    spec=importlib.util.spec_from_file_location('pr6_configure_owner',owner_path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    head=subprocess.check_output(['/usr/bin/git','-C',str(C),'rev-parse','HEAD'],text=True,timeout=10).strip()
    assert head.startswith('4aa4072b')
    assert not subprocess.check_output(['/usr/bin/git','-C',str(C),'diff','--','validation/pr6-restore'],timeout=10)
    names=subprocess.check_output(['/usr/bin/git','-C',str(C),'ls-files','-z','validation/pr6-restore'],timeout=10).split(b'\0')
    source_pins={str(p):pin(C/p.decode()) for p in names if p}
    for group,path in [('vendor_files','vendor_root'),('llama_files','llama_snapshot')]:
        for v in dep[group]:
            p=Path(dep[path])/v['path'];actual=pin(p)
            assert actual['bytes']==v['bytes'] and actual['sha256']==v['sha256']
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);root.mkdir(mode=0o700)
        record=dict(schema='strata-pr6-harness-configure-v1',passed=False,complete=False,active=False,
            commands=[],GPU_executed=False,legacy_engine_built=False,source_commit=head,source_pins=source_pins,
            dependency_preparation=pin(dependency),compilers=[pin(L/'bin/clang'),pin(L/'bin/clang++')],
            oneMKL_config=pin('/opt/intel/oneapi/mkl/2026.1/lib/cmake/mkl/MKLConfig.cmake'),
            scope='Offline CMake configuration/compiler host probes only. Native/provider/GPU/legacy engine build/numerical600/physical262144/RESTORE are not qualified.')
        owner=None
        def persist():
            record['active']=bool(owner and owner.active is not None)
            (root/'record.json').write_text(json.dumps(record,indent=2)+'\n')
        owner=m.Owner(root/'owned',persist);record['commands']=owner.commands
        env=dict(PATH=str(L/'bin')+':/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8',
            PYTHONDONTWRITEBYTECODE='1',LD_LIBRARY_PATH=str(L/'lib')+':/opt/intel/oneapi/mkl/2026.1/lib',
            MKLROOT='/opt/intel/oneapi/mkl/2026.1',OMP_NUM_THREADS='1',CPLUS_INCLUDE_PATH=str(L/'include'),
            GIT_TERMINAL_PROMPT='0',GIT_SSH_COMMAND='ssh -oBatchMode=yes')
        try:
            build=root/'build'
            argv=['/usr/bin/cmake','-S',str(C/'validation/pr6-restore'),'-B',str(build),'-G','Ninja',
                '-DPR6_MAIN_REVIEWED_BUILD=ON','-DPR6_BUILD_PREFILL_API_HARNESS=ON','-DPR6_OPEN_SOURCE_LLVM_ONEMKL=ON',
                '-DPR6_VENDOR_ROOT='+dep['vendor_root'],'-DSTRATA_GGML_DIR='+dep['llama_snapshot'],
                '-DCMAKE_C_COMPILER='+str(L/'bin/clang'),'-DCMAKE_CXX_COMPILER='+str(L/'bin/clang++'),
                '-DCMAKE_BUILD_TYPE=Release','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
                '-DSTRATA_NATIVE_EXPERTS=ON','-DSTRATA_SYCL_PARITY=OFF','-DSTRATA_SYCL_AOT=',
                '-DFETCHCONTENT_FULLY_DISCONNECTED=ON','-DGGML_CCACHE=OFF']
            c,out,err=owner.run('cmake-configure',argv,env,C,wall=180,cpu=240,rss_cap=8<<30,
                text_cap=8<<20,file_cap=64<<20,total_cap=256<<20)
            record.update(exit_code=c['exit_code'],stdout=out.read_text(),stderr=err.read_text(),build_path=str(build))
            assert m.closed(c) and c['exit_code']==0,(c['exit_code'],err.read_text()[-4096:])
            for v in source_pins.values():assert pin(v['path'])==v
            record['generated_inputs']=[pin(build/name) for name in ('CMakeCache.txt','compile_commands.json','build.ninja','pr6-checked-dpct.hpp','pr6-checked-dpct.sha256')]
            record.update(passed=True,complete=True,compile_and_link_command_review_pending=True)
        except BaseException as e:
            import traceback
            record.update(error=repr(e),traceback=traceback.format_exc())
        finally:
            record['ownership_closed']=owner.active is None and all(m.closed(c) for c in owner.commands)
            for c in owner.commands:
                for old in c['owners']+[c['direct_identity']]:
                    now=m.identity(old['pid']);assert now is None or now['start_ticks']!=old['start_ticks']
            persist();target.write_bytes((root/'record.json').read_bytes())
        print(json.dumps(dict(receipt=pin(target),passed=record['passed'],exit_code=record.get('exit_code'),GPU_executed=False,error=record.get('error'))))
        return 0 if record['passed'] and record['ownership_closed'] else 1


if __name__=='__main__':sys.exit(main())
