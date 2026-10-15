#!/usr/bin/env python3
"""Serial owned actual-SYCL correctness/timing arm; immutable process environment."""
import fcntl
import json
import math
from pathlib import Path
import re
import sys
sys.dont_write_bytecode=True
A=Path(__file__).resolve().parent;P=A.parents[2]
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
DEVICE=Path('/sys/bus/pci/devices/0000:05:00.0')
FAULT=re.compile(r'(?i)(\bxe\b.*(fault|hang|reset|error|wedg|timeout|stuck)|GPU.*(fault|hang|wedg|reset)|GuC.*(fault|hang|reset|error))')
sys.path.insert(0,str(A/'owned-stall-stack-probe-v1'))
from owned_stack_probe import load_owner,require,sha,boot_id


def main(mode,direct,arm,index):
    require(mode in ('full','bench') and direct in (0,1) and arm in (0,1) and 0<=index<3,'declared arm')
    require((mode=='full' and index==0) or (mode=='bench' and direct==1),'declared geometry')
    module=load_owner(P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        build_path=A/'own-prompt-k-cellmajor-build-v3.json';build=json.loads(build_path.read_text())
        require(build['passed'] and build['complete'] and not build['active'],'actual build closed')
        require(all(sha(P/n)==h for n,h in build['source_pins'].items()),'compiled source stable')
        require(all(module.closed(c) for c in build['commands']),'build owners closed')
        exe=Path(build['component_binary']);require(sha(exe)==build['component_binary_sha256'],'fixture exact')
        prior_path=A/'qsa-subprofile-model-v2.json';prior=json.loads(prior_path.read_text())
        require(prior['passed'] and prior['complete'] and not prior['active'] and not prior['fresh_faults'] and prior['boot_unchanged'] and not prior['driver_dump_exists'] and prior['boot_id']==boot_id(),'healthy reference closure')
        if mode=='bench':
            for d in (0,1):
                for a in (0,1):
                    gate=json.loads((A/f'own-prompt-k-cellmajor-full-d{d}-a{a}-r0-v2.json').read_text())
                    require(gate['passed'] and gate['complete'] and not gate['active'] and not gate['fresh_faults'],'preceding full correctness')
        require((DEVICE/'driver').resolve().name=='xe' and not (DEVICE/'devcoredump').exists(),'GPU preflight')
        tag=f'{mode}-d{direct}-a{arm}-r{index}'
        root=B/('main-own-prompt-k-cellmajor-'+tag+'-root-v2');target=A/('own-prompt-k-cellmajor-'+tag+'-v2.json')
        require(not root.exists() and not target.exists(),'fresh independent process')
        root.mkdir(mode=0o700);dump=root/'output.f32'
        allowed={'PATH','HOME','LANG','LC_ALL','LD_LIBRARY_PATH','MKLROOT','OMP_NUM_THREADS','ONEAPI_DEVICE_SELECTOR','UR_ADAPTERS_FORCE_LOAD','SYCL_CACHE_PERSISTENT','NEOReadDebugKeys','EnableDirectSubmission','UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD','STRATA_SYCL_VERIFY_NATIVE_CAPTURE','STRATA_SYCL_PREFILL_SAME_QUEUE'}
        env={k:v for k,v in prior['environment'].items() if k in allowed}
        env.update(PATH='/usr/bin:/bin',OMP_NUM_THREADS='1',XDG_CACHE_HOME=str(root/'cache'),STRATA_SYCL_PROMPT_DIRECT=str(direct),STRATA_SYCL_PROMPT_K_CELL_MAJOR=str(arm),STRATA_SYCL_PROMPT_NATIVE_KV_MIN='0')
        if mode=='full':
            env.update(UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',UR_LOG_TRACING='level:info;flush:info;output:stderr',UR_LOG_VALIDATION='level:warning;flush:warning;output:stderr',UR_LOG_LOADER='level:warning;flush:warning;output:stderr',UR_LOG_LEVEL_ZERO='level:warning;flush:warning;output:stderr')
        for k,v in dict(UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1',NEOReadDebugKeys='1',EnableDirectSubmission='0',SYCL_CACHE_PERSISTENT='0').items():require(env[k]==v,'fixed runtime setting')
        argv=[str(exe),'--dump',str(dump)] if mode=='full' else [str(exe),'--bench','256','20','--dump',str(dump)]
        record=dict(passed=False,complete=False,active=True,GPU_executed=False,commands=[],mode=mode,direct=direct,arm=arm,index=index,argv=argv,environment=env,
            references={build_path.name:sha(build_path),prior_path.name:sha(prior_path)},recipe_sha256=sha(Path(__file__)),binary_sha256=sha(exe),source_pins=build['source_pins'],boot_id=boot_id(),
            limits=dict(wall_seconds=180,RSS_bytes=8<<30,text_bytes=64<<20,total_bytes=128<<20,file_bytes=64<<20),
            performance_qualified=False,quality_qualified=False,physical_262144_qualified=False,restore_qualified=False,
            scope='Actual own SYCL component: full16format/page/scaled correctness or clean nq256x20 timing with4query scalar oracle/all finite/guards. Cross-process exact dump comparison separate. Not model PP/quality/physical-context qualification.')
        def persist():(root/'record.json').write_text(json.dumps(record,indent=2)+'\n')
        owner=module.Owner(root/'owned',persist);record['commands']=owner.commands;persist()
        hostenv=dict(PATH='/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8')
        def utility(label,argv):
            c,out,err=owner.run(label,argv,hostenv,P,wall=30,cpu=30,rss_cap=1<<30,text_cap=4<<20,total_cap=128<<20,file_cap=4<<20)
            require(module.closed(c) and c['exit_code']==0,label+': '+err.read_text()[-2000:]);return out.read_text()
        cursor=None
        try:
            cursor=re.search(r'-- cursor: (\S+)',utility('journal-before',['journalctl','-k','-b','-n','0','--show-cursor','--no-pager']))[1]
            c,out,err=owner.run('component',argv,env,P,wall=180,cpu=600,rss_cap=8<<30,text_cap=64<<20,total_cap=128<<20,file_cap=64<<20)
            record['GPU_executed']=c.get('direct_identity') is not None
            record.update(stdout=out.read_text(),stderr_sha256=sha(err),stderr_bytes=err.stat().st_size,exit_code=c['exit_code'])
            require(module.closed(c) and c['exit_code']==0,'component exit: '+err.read_text()[-3000:]+'\n'+out.read_text()[-3000:])
            require(not re.search(r'(?:UR|ZE)_RESULT_ERROR_|terminate called|what\(\):|Strata SYCL cleanup:',err.read_text()),'runtime error')
            require(dump.stat().st_size==(1180672 if mode=='full' else 6291520),'complete bounded dump')
            record['dump']=dict(path=str(dump),bytes=dump.stat().st_size,sha256=sha(dump),consumer='Main OFF/ON exact component comparator',budget_bytes=8<<20,next_review='after full and timing cross-arm comparison')
            if mode=='full':require(len(out.read_text().splitlines())==16 and all(l.startswith('split format=') for l in out.read_text().splitlines()),'all16 full cases')
            else:
                m=re.search(r'bounded_component nq=256 cap=2051 .*?repeats=20 total_ms=([0-9.eE+-]+)',out.read_text());require(m is not None,'actual timing shape')
                value=float(m[1]);require(math.isfinite(value) and value>0,'finite measured time');record['component_total_ms']=value
                require('full_finite=PASS tail_canaries=PASS cross_arm_exact=EXTERNAL_REQUIRED' in out.read_text(),'benchmark correctness scope')
            require(all(sha(P/n)==h for n,h in build['source_pins'].items()) and sha(exe)==build['component_binary_sha256'],'source/binary stable')
            record.update(passed=True,complete=True)
        except BaseException as error:record['error']=repr(error);raise
        finally:
            try:
                if cursor and owner.active is None:
                    entries=[json.loads(s) for s in utility('journal-final',['journalctl','-k','-b','--after-cursor',cursor,'--no-pager','-o','json']).splitlines() if s.strip()]
                    record.update(fresh_faults=[v for v in entries if FAULT.search(str(v.get('MESSAGE','')))],boot_unchanged=boot_id()==record['boot_id'],driver_dump_exists=(DEVICE/'devcoredump').exists())
            finally:
                record.update(active=owner.active is not None,finished_utc=module.utc())
                if record['active'] or not all(module.closed(c) for c in owner.commands) or 'fresh_faults' not in record or record.get('fresh_faults') or not record.get('boot_unchanged') or record.get('driver_dump_exists'):record.update(passed=False,complete=False)
                persist()
                with target.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
        print(json.dumps(dict(passed=record['passed'],tag=tag,component_total_ms=record.get('component_total_ms'),dump_sha256=record['dump']['sha256'])),flush=True)


if __name__=='__main__':
    require(len(sys.argv)==5,'mode direct arm index')
    main(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]))
