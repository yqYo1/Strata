#!/usr/bin/env python3
"""Main-owned strict standalone MoE accumulation build and serialized GPU cases."""
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
W = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/verify-moe-window')
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
L = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011/.tools/intel-llvm-v711/install')
DEV = Path('/sys/bus/pci/devices/0000:05:00.0')
BOOT = 'cfda0b6a-7c13-4fe2-bdf5-8f9202770840'
FAULT = re.compile(r'(?i)(\bxe\b.*(fault|hang|reset|error|wedg|timeout|stuck)|GPU.*(fault|hang|wedg|reset)|GuC.*(fault|hang|reset|error))')
CASES = ('smoke','numeric','short-weights','short-down','legacy-plan','negative-source','negative-slot','unwind','hot65536','balanced65536','hot81920','balanced81920','partial78452')
def pin(path):
    path=Path(path)
    with path.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=str(path),realpath=str(path.resolve()),bytes=path.stat().st_size,sha256=digest)
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path); value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

def main():
    assert len(sys.argv)==2 and sys.argv[1] in ('build',)+CASES
    mode=sys.argv[1]; root=B/f'main-moe-accum-device-{mode}-v2'; dst=A/f'moe-accum-device-{mode}-v2.json'
    candidate_path=A/'moe-window10-candidate-v1.json'; candidate=json.loads(candidate_path.read_text())
    assert all(pin(W/name)['sha256']==value['candidate_sha256'] for name,value in candidate['source_pins'].items())
    assert not root.exists() and not dst.exists()
    owner_path=A/'direct_owner_transient_artifacts_v1.py'
    assert pin(owner_path)['sha256']=='bddd0c5e5020e882ca3404679420faec11e820dabce2ed4c72ef82b4c9c0874a'
    owned=load('accum_owner',owner_path)
    package=A/'own-vllm-xe2-qualification-v1'; metadata=package/'manifest.json'
    assert pin(metadata)['sha256']=='7027faaa150d65f8581aebef1e731afe858288792dcedf8fb4409991806e1b58'
    meta=json.loads(metadata.read_text()); helper=package/'main_controller.py'
    assert pin(helper)['sha256']==meta['package_files']['main_controller.py']['sha256']
    guard=load('accum_guard',helper)
    buildpath=A/'command-service-collector-engine-build-v2.json'; base=json.loads(buildpath.read_text())
    assert base['passed'] and base['complete'] and base['ownership_closed'] and not base['active']
    archive=Path(base['products']['strata']['path']).parent/'libstrata_sycl_runtime.a'
    # Every runtime implementation/header is unchanged from this archive's build.
    runtime_inputs=[]
    for name,digest in base['source_pins'].items():
        relative=name.split('/verify-moe-window/',1)[-1]
        if relative.startswith(('src/sycl/','include/strata/sycl/')):
            assert pin(W/relative)['sha256']==digest,relative
            runtime_inputs.append(relative)
    assert 'src/sycl/runtime.cpp' in runtime_inputs
    cohortpath=A/'moe-accum-model-clean-cohort-v1.json'; cohort=json.loads(cohortpath.read_text())
    assert cohort['passed'] and cohort['complete'] and cohort['active_sample'] is None and len(cohort['completed'])==6
    for item in cohort['completed']:
        assert pin(item['path'])['sha256']==item['sha256']
        receipt=json.loads(Path(item['path']).read_text())
        assert receipt['ownership_closed'] and not receipt['active'] and not receipt['fresh_faults']
        for c in receipt['commands']:
            assert owned.closed(c)
            for old in c['owners']+[c['direct_identity']]:
                now=owned.identity(old['pid']);assert now is None or now['start_ticks']!=old['start_ticks']
    compiler=L/'bin/clang++';assert pin(compiler)['sha256']=='1290881b4e12a5e002e5b0de0787f85328ee5fe19f253a54be1cdd97dc2f26eb'
    elf_manifest=A/'own-vllm-resource-timeline-v3/manifest.json'
    assert pin(elf_manifest)['sha256']=='61da9469e3f975c72d770beee77d45118c988eb6d3d82b6dbe1b568cd9ea3f1c'
    allowed=json.loads(elf_manifest.read_text())['elf_pins']
    sources=[W/'tests/sycl/moe_accum.cpp',W/'src/prefill/sycl/native16.cpp']
    inputs=[*sources,archive,compiler,Path(__file__),owner_path,metadata,helper,buildpath,cohortpath,candidate_path,
            W/'include/strata/prefill/native16.hpp',W/'include/strata/prefill/moe_window_routes.hpp',
            W/'include/strata/kernels/bf16_bits.hpp',W/'include/strata/kernels/f16_bits.hpp']
    inputs += [W/name for name in runtime_inputs]
    frozen={str(path):pin(path) for path in inputs}
    binary_receipt=A/'moe-accum-device-build-v2.json'
    build=None
    if mode!='build':
        build=json.loads(binary_receipt.read_text()); assert build['passed'] and build['complete'] and build['ownership_closed'] and not build['active']
        for source in sources+[W/'include/strata/prefill/native16.hpp',W/'include/strata/prefill/moe_window_routes.hpp']:
            assert pin(source)==build['frozen_files'][str(source)]
        if mode!='smoke':
            smoke=json.loads((A/'moe-accum-device-smoke-v2.json').read_text()); assert smoke['passed'] and smoke['complete'] and smoke['ownership_closed'] and not smoke['fresh_faults']
        frozen[str(binary_receipt)]=pin(binary_receipt)
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        resource=guard.no_model_owners(meta['known_model_files']);guard.driver_guard()
        assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==BOOT
        root.mkdir(mode=0o700);(root/'tmp').mkdir()
        host=dict(PATH='/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8',PYTHONDONTWRITEBYTECODE='1')
        compile_env={**host,'PATH':str(L/'bin')+':/usr/bin:/bin','LD_LIBRARY_PATH':str(L/'lib'),'TMPDIR':str(root/'tmp'),'XDG_CACHE_HOME':str(root/'cache')}
        env={**host,'LD_LIBRARY_PATH':str(L/'lib')+':/opt/intel/oneapi/mkl/2026.1/lib','ONEAPI_DEVICE_SELECTOR':'level_zero:gpu',
             'UR_ADAPTERS_FORCE_LOAD':str(L/'lib/libur_adapter_level_zero_v2.so.0'),'UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD':'1',
             'NEOReadDebugKeys':'1','EnableDirectSubmission':'0','STRATA_SYCL_PREFILL_SAME_QUEUE':'1','STRATA_SYCL_VERIFY_NATIVE_CAPTURE':'0',
             'SYCL_CACHE_PERSISTENT':'0','ZES_ENABLE_SYSMAN':'0','XDG_CACHE_HOME':str(root/'native-cache'),'STRATA_TEST_MAIN_GATES':'1',
             'UR_ENABLE_LAYERS':'UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION','UR_LOG_TRACING':'level:info;flush:info;output:stderr',
             'UR_LOG_VALIDATION':'level:warning;flush:warning;output:stderr','UR_LOG_LOADER':'level:warning;flush:warning;output:stderr',
             'UR_LOG_LEVEL_ZERO':'level:warning;flush:warning;output:stderr'}
        rec=dict(schema='strata-moe-accum-device-v2',mode=mode,passed=False,complete=False,active=False,GPU_executed=False,
                 commands=[],boot_id=BOOT,native_environment=env,compile_environment=compile_env,resource_guard=resource,
                 frozen_files=frozen,snapshots=[],binary=None,runtime_archive_input_paths=runtime_inputs,
                 archive_provenance='Unchanged runtime sources/headers from completed collector-engine-build-v2. Main-owned native16.cpp and fixture freshly compiled; no XeStrata source used.',
                 scope='Synthetic expert-output routing/FP32 accumulation/BF16 epilogue. No GEMM/model/PP/quality/physical262144/RESTORE qualification.',
                 production_selected=False,performance_adopted=False,retention='64MiB raw text per job; retain compact results then retire successful duplicate API traces.')
        owner=None;cursor=None
        def save():
            rec['active']=bool(owner and owner.active is not None)
            rec['ownership_closed']=bool(owner and owner.active is None and all(owned.closed(c) for c in owner.commands))
            dst.write_text(json.dumps(rec,indent=2)+'\n')
        owner=owned.Owner(root/'owned',save);rec['commands']=owner.commands
        def run(label,args,current=host,interaction=None,wall=60):
            c,out,err=owner.run(label,args,current,W,interaction=interaction,wall=wall,cpu=600,rss_cap=6<<30,
                              text_cap=64<<20,file_cap=128<<20,total_cap=256<<20)
            assert owned.closed(c) and c['exit_code']==0,(label,c.get('exit_code'),str(err))
            return out.read_text(),err.read_text()
        def verify():
            assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==BOOT
            assert DEV.joinpath('driver').resolve().name=='xe' and not list(DEV.glob('devcoredump*'))
            for name,value in frozen.items():assert pin(name)==value,name
        def snapshot(io,entry,phase):
            identity=owned.identity(io.proc.pid);assert identity and identity['start_ticks']==entry['direct_identity']['start_ticks']
            proc=Path('/proc')/str(io.proc.pid);assert str(proc.joinpath('exe').resolve())==rec['binary']['realpath']
            observed=dict(x.decode().split('=',1) for x in proc.joinpath('environ').read_bytes().split(b'\0') if x);assert observed==env
            seen={}
            for line in proc.joinpath('maps').read_text().splitlines():
                fields=line.split(None,5)
                if len(fields)!=6 or 'x' not in fields[1] or not fields[5].startswith('/'):continue
                path=Path(fields[5]).resolve(strict=True);assert not fields[5].endswith(' (deleted)') and str(path) in allowed,str(path)
                value=allowed[str(path)];assert pin(path)==value
                st=path.stat();major,minor=[int(x,16) for x in fields[3].split(':')]
                assert st.st_ino==int(fields[4]) and os.major(st.st_dev)==major and os.minor(st.st_dev)==minor
                seen[str(path)]=value['sha256']
            assert rec['binary']['realpath'] in seen
            if phase!='before-runtime':
                for path in [L/'lib/libsycl.so.9',L/'lib/libur_adapter_level_zero_v2.so.0',Path('/usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1.17.39395')]:assert str(path.resolve()) in seen
            rec['snapshots'].append(dict(phase=phase,identity=identity,exec_environment=observed,executed_ELF_sha256=seen));io.send(b'X')
        save()
        try:
            verify()
            raw,_=run('journal-before',['/usr/bin/journalctl','-k','-b','-n','0','--show-cursor','--no-pager']);cursor=re.search(r'-- cursor: (\S+)',raw)[1];rec['journal_before_cursor']=cursor
            if mode=='build':
                binary=root/'sycl_moe_accum'
                argv=[str(compiler),'-std=c++20','-O3','-DNDEBUG','-DSTRATA_ENABLE_SYCL=1','-I'+str(W/'include'),'-fno-fast-math','-ffp-contract=off','-fsycl','-fsycl-id-queries-range=size_t','-Wall','-Wextra','-Werror',*map(str,sources),str(archive),'-o',str(binary)]
                rec['compile_argv']=argv;run('compile-link',argv,compile_env,wall=300)
            else:binary=Path(build['binary']['path']);assert pin(binary)==build['binary']
            rec['binary']=pin(binary);allowed[rec['binary']['realpath']]=rec['binary'];frozen[str(binary)]=rec['binary']
            dynamic,_=run('readelf',['/usr/bin/readelf','-d',str(binary)]);rec['dynamic']=dynamic
            assert set(re.findall(r'Shared library: \[([^\]]+)\]',dynamic))=={'libsycl.so.9','libstdc++.so.6','libm.so.6','libgcc_s.so.1','libc.so.6'}
            if mode!='build':
                holds=[]
                def interaction(io,entry):
                    for _ in range(100):
                        line=io.line(seconds=60)
                        if line.startswith('ACCUM_READY '):
                            phase=line.split(' ',1)[1];assert phase==['before-runtime','runtime-ready','after-cleanup'][len(holds)]
                            snapshot(io,entry,phase);holds.append(phase)
                        if line.startswith('fixture_status=PASS '):assert len(holds)==3;io.exit(seconds=10);return
                    raise RuntimeError('missing bounded completion')
                rec['GPU_executed']=True
                out,err=run('accumulator-case',[str(binary),'--case',mode],env,interaction,wall=180)
                rec['stdout']=out
                assert any(re.fullmatch(r'device=.*B570.* driver=1\.17\.39395\+14',line) for line in out.splitlines())
                assert not re.search(r'(?:UR|ZE)_RESULT_ERROR_|accumulator fixture failed|Strata SYCL cleanup:',err)
                kernels=len(re.findall(r'^\s*---> urEnqueueKernelLaunch(?:WithArgsExp)?\s*$',err,re.M))
                rec['UR_kernel_submissions']=kernels
                result,=[s for s in out.splitlines() if s.startswith(('case ','refusal=','unwind='))]
                expected=int(re.search(r'expected_kernels=(\d+)',result)[1]);assert kernels==expected
                assert result.endswith('PASS') or 'unwind=PASS numeric=UNRUN' in result
                if mode in ('short-weights','short-down','legacy-plan'):assert kernels==0
                if mode.startswith('negative-'):assert 'full_output=PASS' in result
                rec['case_result']=result
            verify();rec.update(passed=True,complete=True)
        except BaseException as error:
            import traceback
            rec.update(error=repr(error),traceback=traceback.format_exc(),passed=False,complete=False)
        finally:
            try:
                assert owner.active is None
                if cursor:
                    raw,_=run('journal-final',['/usr/bin/journalctl','-k','-b','--after-cursor',cursor,'--no-pager','-o','json'])
                    rec['fresh_faults']=[x for x in [json.loads(s) for s in raw.splitlines() if s.strip()] if FAULT.search(x.get('MESSAGE',''))]
                rec['boot_unchanged']=Path('/proc/sys/kernel/random/boot_id').read_text().strip()==BOOT
                rec['driver_dump_entries']=[str(x) for x in DEV.glob('devcoredump*')]
                assert cursor and rec['fresh_faults']==[] and rec['boot_unchanged'] and not rec['driver_dump_entries'];guard.driver_guard()
            except BaseException as error:rec.update(closure_error=repr(error),passed=False,complete=False)
            save()
        print(json.dumps(dict(passed=rec['passed'],complete=rec['complete'],GPU_executed=rec['GPU_executed'],error=rec.get('error'),closure_error=rec.get('closure_error'))),flush=True)
        return 0 if rec['passed'] else 1
if __name__=='__main__':sys.exit(main())
