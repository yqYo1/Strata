#!/usr/bin/env python3
"""Owned strict build and traced ordinary-host copy/queued-command timestamp diagnostic."""
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
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
L = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011/.tools/intel-llvm-v711/install')
DEV = Path('/sys/bus/pci/devices/0000:05:00.0')
BOOT = 'cfda0b6a-7c13-4fe2-bdf5-8f9202770840'
FAULT = re.compile(r'(?i)(\bxe\b.*(fault|hang|reset|error|wedg|timeout|stuck)|GPU.*(fault|hang|wedg|reset)|GuC.*(fault|hang|reset|error))')


def pin(path):
    path = Path(path)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path),realpath=str(path.resolve()),bytes=path.stat().st_size,sha256=digest)


def load(name,path):
    spec = importlib.util.spec_from_file_location(name,path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def main():
    assert len(sys.argv)==2 and sys.argv[1] in ('baseline','aged')
    mode=sys.argv[1]
    root = B/f'main-event-service-clock-age-{mode}-v1'
    dst = A/f'event-service-clock-age-{mode}-v1.json'
    assert not root.exists() and not dst.exists()
    owner_path = A/'direct_owner_transient_artifacts_v1.py'
    assert pin(owner_path)['sha256']=='bddd0c5e5020e882ca3404679420faec11e820dabce2ed4c72ef82b4c9c0874a'
    owned = load('event_calibration_owner',owner_path)
    package = A/'own-vllm-xe2-qualification-v1'
    metadata = package/'manifest.json'
    assert pin(metadata)['sha256']=='7027faaa150d65f8581aebef1e731afe858288792dcedf8fb4409991806e1b58'
    meta = json.loads(metadata.read_text())
    helper = package/'main_controller.py'
    assert pin(helper)['sha256']==meta['package_files']['main_controller.py']['sha256']
    guard = load('event_calibration_guard',helper)
    buildpath = A/'qsa-rowmajor-subgroup-engine-build-v1.json'
    build = json.loads(buildpath.read_text())
    assert build['passed'] and build['complete'] and build['ownership_closed'] and not build['active']
    gatepath = A/'qsa-subgroup-engine-model-clean-cohort-v1.json'
    gate = json.loads(gatepath.read_text())
    assert gate['passed'] and gate['complete'] and gate['active_sample'] is None
    for item in gate['completed']:
        assert pin(item['path'])['sha256']==item['sha256']
        value=json.loads(Path(item['path']).read_text())
        assert value['passed'] and value['ownership_closed'] and not value['active'] and not value['fresh_faults']
        for command in value['commands']:
            assert owned.closed(command) and command['exit_code']==0
            for old in command['owners']+[command['direct_identity']]:
                now=owned.identity(old['pid'])
                assert now is None or now['start_ticks']!=old['start_ticks']
    pairpath=A/'command-service-model-diagnostic-cohort-v1.json'
    pair=json.loads(pairpath.read_text())
    assert pair['active_sample'] is None and len(pair['completed'])==2
    for item in pair['completed']:
        assert pin(item['path'])['sha256']==item['sha256']
        value=json.loads(Path(item['path']).read_text())
        assert value['ownership_closed'] and not value['active'] and value['fresh_faults']==[] and not value['driver_dump_exists']
        assert value['result']['real_engine_exit']['exit_code']==0
    rawpath=A/'command-service-model-diagnostic-on-r0-v2.json'
    raw=json.loads(rawpath.read_text())
    assert raw['raw_capture_complete'] and raw['complete'] and raw['ownership_closed'] and not raw['active']
    assert not raw['fresh_faults'] and not raw['driver_dump_exists'] and raw['result']['real_engine_exit']['exit_code']==0
    baselinepath=A/'event-service-clock-age-baseline-v1.json'
    baseline=None
    if mode=='aged':
        baseline=json.loads(baselinepath.read_text())
        assert baseline['tensor_checks_passed'] and baseline['capture_complete'] and baseline['complete']
        assert baseline['ownership_closed'] and not baseline['active'] and not baseline['fresh_faults'] and not baseline['driver_dump_entries']
    source = P/'tests/sycl/event_service_clock_age.cpp'
    archive = Path(build['products']['strata']['path']).parent/'libstrata_sycl_runtime.a'
    compiler = L/'bin/clang++'
    assert pin(compiler)['sha256']=='1290881b4e12a5e002e5b0de0787f85328ee5fe19f253a54be1cdd97dc2f26eb'
    baseline = A/'own-vllm-resource-timeline-v3/manifest.json'
    assert pin(baseline)['sha256']=='61da9469e3f975c72d770beee77d45118c988eb6d3d82b6dbe1b568cd9ea3f1c'
    allowed = json.loads(baseline.read_text())['elf_pins']
    frozen={str(path):pin(path) for path in [source,archive,compiler,Path(__file__),owner_path,metadata,helper,buildpath,gatepath,pairpath,rawpath,
            P/'include/strata/sycl/runtime.hpp',P/'include/strata/sycl/checked_cleanup.hpp',P/'src/sycl/runtime.cpp']}
    for path in ['include/strata/sycl/runtime.hpp','include/strata/sycl/checked_cleanup.hpp','src/sycl/runtime.cpp']:
        matching=[digest for name,digest in build['source_pins'].items() if name.endswith('/'+path)]
        assert matching==[pin(P/path)['sha256']]
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        resource=guard.no_model_owners(meta['known_model_files'])
        guard.driver_guard()
        assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==BOOT
        root.mkdir(mode=0o700)
        (root/'tmp').mkdir()
        host=dict(PATH='/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8',PYTHONDONTWRITEBYTECODE='1')
        compile_env={**host,'PATH':str(L/'bin')+':/usr/bin:/bin','LD_LIBRARY_PATH':str(L/'lib'),
                     'TMPDIR':str(root/'tmp'),'XDG_CACHE_HOME':str(root/'cache')}
        env={**host,'LD_LIBRARY_PATH':str(L/'lib')+':/opt/intel/oneapi/mkl/2026.1/lib',
             'ONEAPI_DEVICE_SELECTOR':'level_zero:gpu','UR_ADAPTERS_FORCE_LOAD':str(L/'lib/libur_adapter_level_zero_v2.so.0'),
             'UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD':'1','NEOReadDebugKeys':'1','EnableDirectSubmission':'0',
             'STRATA_SYCL_PREFILL_SAME_QUEUE':'1','STRATA_SYCL_VERIFY_NATIVE_CAPTURE':'0','SYCL_CACHE_PERSISTENT':'0',
             'ZES_ENABLE_SYSMAN':'0','XDG_CACHE_HOME':str(root/'native-cache'),'STRATA_TEST_MAIN_GATES':'1',
             'UR_ENABLE_LAYERS':'UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
             'UR_LOG_TRACING':'level:info;flush:info;output:stderr',
             'UR_LOG_VALIDATION':'level:warning;flush:warning;output:stderr',
             'UR_LOG_LOADER':'level:warning;flush:warning;output:stderr','UR_LOG_LEVEL_ZERO':'level:warning;flush:warning;output:stderr'}
        rec=dict(schema='strata-event-service-clock-age-v1',mode=mode,passed=False,complete=False,active=False,
            GPU_executed=False,commands=[],boot_id=BOOT,native_environment=env,compile_environment=compile_env,
            resource_guard=resource,frozen_files=frozen,snapshots=[],binary=None,
            archive_provenance='Freshly hashed runtime archive from the completed f685b16a127-step build. Runtime/header inputs match current primary bytes.',
            scope='Capture all141 selected raw event timestamps after a3075-kernel burst and two640KiB ordinary-host H2D copies on the existing profiled queue. Data/capture PASS is distinct from timestamp_qualified; rejected values remain serialized. No model PP, hardware busy, transfer bandwidth or production collector qualification.',
            retention='64MiB raw-text cap; retain compact timestamps/guards/fault/owner facts; retire successful full API trace after review.')
        owner=None

        def save():
            rec['active']=bool(owner and owner.active is not None)
            rec['ownership_closed']=bool(owner and owner.active is None and all(owned.closed(c) for c in owner.commands))
            dst.write_text(json.dumps(rec,indent=2)+'\n')

        owner=owned.Owner(root/'owned',save)
        rec['commands']=owner.commands

        def run(label,args,current=host,interaction=None,wall=60):
            c,stdout,stderr=owner.run(label,args,current,P,interaction=interaction,wall=wall,cpu=600,
                rss_cap=6<<30,text_cap=64<<20,file_cap=128<<20,total_cap=256<<20)
            assert owned.closed(c) and c['exit_code']==0,(label,c.get('exit_code'),str(stderr))
            return stdout.read_text(),stderr.read_text()

        def verify():
            assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==BOOT
            assert DEV.joinpath('driver').resolve().name=='xe' and not list(DEV.glob('devcoredump*'))
            for name,value in frozen.items():assert pin(name)==value,name

        def snapshot(io,entry,phase):
            identity=owned.identity(io.proc.pid)
            assert identity and identity['start_ticks']==entry['direct_identity']['start_ticks']
            proc=Path('/proc')/str(io.proc.pid)
            assert str(proc.joinpath('exe').resolve())==rec['binary']['realpath']
            observed=dict(x.decode().split('=',1) for x in proc.joinpath('environ').read_bytes().split(b'\0') if x)
            assert observed==env
            seen={}
            for line in proc.joinpath('maps').read_text().splitlines():
                fields=line.split(None,5)
                if len(fields)!=6 or 'x' not in fields[1] or not fields[5].startswith('/'):continue
                path=Path(fields[5]).resolve(strict=True)
                assert not fields[5].endswith(' (deleted)') and str(path) in allowed
                value=allowed[str(path)]
                assert pin(path)==value
                stat=path.stat();major,minor=[int(x,16) for x in fields[3].split(':')]
                assert stat.st_ino==int(fields[4]) and os.major(stat.st_dev)==major and os.minor(stat.st_dev)==minor
                seen[str(path)]=value['sha256']
            assert rec['binary']['realpath'] in seen
            if phase!='before-runtime':
                for path in [L/'lib/libsycl.so.9',L/'lib/libur_adapter_level_zero_v2.so.0',Path('/usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1.17.39395')]:
                    assert str(path.resolve()) in seen
            rec['snapshots'].append(dict(phase=phase,identity=identity,exec_environment=observed,executed_ELF_sha256=seen))
            io.send(b'X')

        cursor=None
        save()
        try:
            verify()
            if mode=='baseline':
                binary=root/'sycl_event_service_clock_age'
                argv=[str(compiler),'-std=c++20','-O3','-DNDEBUG','-DSTRATA_ENABLE_SYCL=1','-I'+str(P/'include'),
                      '-fno-fast-math','-ffp-contract=off','-fsycl','-fsycl-id-queries-range=size_t',
                      '-Wall','-Wextra','-Werror',str(source),str(archive),'-o',str(binary)]
                run('compile-link',argv,compile_env,wall=300)
            else:
                binary=Path(baseline['binary']['path'])
                assert pin(binary)==baseline['binary']
                frozen[str(baselinepath)]=pin(baselinepath)
            rec['binary']=pin(binary)
            allowed[rec['binary']['realpath']]=rec['binary']
            frozen[str(binary)]=rec['binary']
            dynamic,_=run('readelf',['/usr/bin/readelf','-d',str(binary)])
            rec['dynamic']=dynamic
            assert set(re.findall(r'Shared library: \[([^\]]+)\]',dynamic))=={'libsycl.so.9','libstdc++.so.6','libm.so.6','libgcc_s.so.1','libc.so.6'}
            raw,_=run('journal-before',['/usr/bin/journalctl','-k','-b','-n','0','--show-cursor','--no-pager'])
            cursor=re.search(r'-- cursor: (\S+)',raw)[1]
            rec['journal_before_cursor']=cursor
            holds=[]

            def interaction(io,entry):
                for _ in range(200):
                    line=io.line(seconds=90)
                    if line.startswith('EVENT_PROFILE_READY '):
                        phase=line.split(' ',1)[1]
                        assert phase==['before-runtime','runtime-ready','after-cleanup'][len(holds)]
                        snapshot(io,entry,phase);holds.append(phase)
                    if line.startswith('fixture_status=PASS '):
                        assert len(holds)==3
                        io.exit(seconds=10)
                        return
                raise RuntimeError('missing bounded completion')

            rec['GPU_executed']=True
            stdout,stderr=run('timestamp-stress',[str(binary),'--'+mode],env,interaction,wall=180)
            rec['stdout']=stdout
            assert any(re.fullmatch(r'device=.*B570.* driver=1\.17\.39395\+14',line) for line in stdout.splitlines())
            assert not re.search(r'(?:UR|ZE)_RESULT_ERROR_|event stress failed|Strata SYCL cleanup:',stderr)
            rec['UR_kernel_submissions']=len(re.findall(r'^\s*---> urEnqueueKernelLaunch(?:WithArgsExp)?\s*$',stderr,re.M))
            assert rec['UR_kernel_submissions']==3075
            samples=[dict(x.split('=',1) for x in line.split()) for line in stdout.splitlines() if line.startswith('sample=')]
            case,=[line for line in stdout.splitlines() if line.startswith('case ')]
            assert len(samples)==141 and all(x in case for x in ['status=PASS','guards=PASS','inputs=PASS','capture=COMPLETE','kernels=3075','copy_bytes_each=655360'])
            values=dict(item.split('=',1) for item in case.removeprefix('case ').split())
            invalid=sum(row['valid']=='0' for row in samples)
            assert invalid==int(values['invalid_timestamps'])
            qualified=invalid==0
            assert values['timestamp_qualified']==str(int(qualified))
            rec['host_clock_query']=[line for line in stdout.splitlines() if line.startswith('host_clock_query ')]
            assert len(rec['host_clock_query'])==1
            assert int(values['age_delay_ms'])==(20000 if mode=='aged' else 0)
            rec.update(samples=samples,case=case,tensor_checks_passed=True,capture_complete=True,
                       invalid_timestamps=invalid,timestamp_qualified=qualified,passed=qualified,complete=True)
            verify()
        except BaseException as error:
            import traceback
            rec.update(error=repr(error),traceback=traceback.format_exc(),passed=False,complete=False)
        finally:
            try:
                assert owner.active is None
                if cursor:
                    raw,_=run('journal-final',['/usr/bin/journalctl','-k','-b','--after-cursor',cursor,'--no-pager','-o','json'])
                    rows=[json.loads(x) for x in raw.splitlines() if x.strip()]
                    rec['fresh_faults']=[x for x in rows if FAULT.search(x.get('MESSAGE',''))]
                rec['boot_unchanged']=Path('/proc/sys/kernel/random/boot_id').read_text().strip()==BOOT
                rec['driver_dump_entries']=[str(x) for x in DEV.glob('devcoredump*')]
                assert cursor and rec['fresh_faults']==[] and rec['boot_unchanged'] and not rec['driver_dump_entries']
                guard.driver_guard()
            except BaseException as error:
                rec.update(closure_error=repr(error),passed=False,complete=False)
            save()
        print(json.dumps(dict(passed=rec['passed'],complete=rec['complete'],GPU_executed=rec['GPU_executed'],error=rec.get('error'),closure_error=rec.get('closure_error'))),flush=True)
        return 0 if rec['passed'] else 1


if __name__=='__main__':
    sys.exit(main())
