#!/usr/bin/env python3
"""Main-owned serial actual-loader/owner gates; exactly one mode per process."""
import fcntl
import json
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

MODES=('native-normal','normal','prepare-oom-null','prepare-second-oom-null',
       'restore-oom-null','native-reserve-oom-null',
       'prepare-unknown-null','prepare-unknown-nonnull','prepare-oom-nonnull','prepare-success-null',
       'restore-unknown-null','restore-unknown-nonnull','restore-oom-nonnull','restore-success-null')


def main(mode):
    require(mode in MODES,'declared component mode')
    module=load_owner(P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        paths=dict(build=A/'canonical-bank-profile-build-v1.json',host=A/'coarse-profile-host-v1.json',
                   seed=A/'qsa-device-batch-small-v1.json',model=A/'selective-iq-clean-bench-full-0-v1.json')
        for previous in MODES[:MODES.index(mode)]:
            paths['preceding-'+previous]=A/f'canonical-bank-component-{previous}-v1.json'
        refs={k:json.loads(p.read_text()) for k,p in paths.items()}
        build,host,seed,model=(refs[k] for k in ('build','host','seed','model'))
        for key,r in refs.items():
            require(r['passed'] and r['complete'] and not r['active'],'closed required gate '+key)
            for c in r['commands']:
                require(module.closed(c),'preceding owner closed')
                for old in c['owners']:
                    now=module.identity(old['pid'])
                    require(now is None or now['start_ticks']!=old['start_ticks'],'preceding process remains')
            if key.startswith('preceding-'):
                require(r['boot_id']==boot_id() and r['boot_unchanged'] and not r['fresh_faults'] and
                        not r['driver_dump_exists'],'preceding healthy sameboot GPU gate')
        require(host['host_arms']==13,'host lifetime/status/output gates')
        require(all(sha(P/n)==h for n,h in build['source_pins'].items()),'fresh actual compiled source')
        root=B/f'main-canonical-bank-component-{mode}-root-v1';output=A/f'canonical-bank-component-{mode}-v1.json'
        require(not root.exists() and not output.exists(),'fresh component mode root')
        if mode=='native-normal':name='sycl_native_dense_bank_test'
        elif mode=='normal':name='sycl_canonical_dense_bank_test'
        elif mode=='native-reserve-oom-null':name='sycl_canonical_native_reserve_fault_test'
        else:name='sycl_canonical_dense_bank_fault_test'
        exe=Path(build['build_path'])/name
        require(sha(exe)==build['binaries'][name],'actual fixture binary identity')
        shards=sorted(model['model_identity']);require(len(shards)==2,'actual model shards')
        for n,v in model['model_identity'].items():
            st=Path(n).stat();require(st.st_size==v['bytes'] and st.st_mtime_ns==v['mtime_ns'],'actual shard identity')
        require(all(sha(Path(n))==h for n,h in model['pack_metadata_sha256'].items()),'actual pack metadata')
        require((DEVICE/'driver').resolve().name=='xe' and not (DEVICE/'devcoredump').exists(),'driver preflight')
        pack=model['engine_args'][model['engine_args'].index('--pack')+1]
        argv=[str(exe)]+([] if mode in ('native-normal','normal') else [mode])+[pack,*shards]
        root.mkdir(mode=0o700)
        env=dict(seed['environment'],XDG_CACHE_HOME=str(root/'cache'))
        if mode not in ('native-normal','normal'):
            env.pop('UR_LOG_TRACING',None);env['UR_ENABLE_LAYERS']='UR_LAYER_PARAMETER_VALIDATION'
        for k,v in dict(UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1',NEOReadDebugKeys='1',EnableDirectSubmission='0',
                        STRATA_SYCL_PREFILL_SAME_QUEUE='1',SYCL_CACHE_PERSISTENT='0',STRATA_SYCL_VERIFY_NATIVE_CAPTURE='0').items():
            require(env[k]==v,'unchanged runtime safety '+k)
        require(not any('TOKEN' in k or 'API_KEY' in k or k=='LD_PRELOAD' or
                        k.startswith(('UNITRACE_','PTI_','STRATA_MEASUREMENT_')) for k in env),'bounded debug environment')
        record=dict(passed=False,complete=False,active=True,GPU_executed=False,mode=mode,commands=[],
                    recipe_sha256=sha(Path(__file__)),references={p.name:sha(p) for p in paths.values()},
                    environment=env,argv=argv,binary_sha256=sha(exe),source_pins=build['source_pins'],
                    model_identity=model['model_identity'],pack_metadata_sha256=model['pack_metadata_sha256'],boot_id=boot_id(),
                    performance_qualified=False,quality_qualified=False,physical_262144_qualified=False,restore_qualified=False,
                    scope='Actual loader and allocator-owned native/canonical component. Canonical payload reread shares '
                          'loader descriptors; explicit PLE pointers,48 overlays and full restoration; one declared '
                          'allocation arm. No generator graphs/MTP, session state/task-quality or physical model proof.')
        def persist():(root/'record.json').write_text(json.dumps(record,indent=2)+'\n')
        owner=module.Owner(root/'owned',persist);record['commands']=owner.commands;persist()
        hostenv=dict(PATH='/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8')
        def utility(label,args):
            c,out,err=owner.run(label,args,hostenv,P,wall=60,cpu=60,rss_cap=1<<30,text_cap=4<<20,total_cap=128<<20,file_cap=4<<20)
            require(module.closed(c) and c['exit_code']==0,label+': '+err.read_text()[-2000:])
            return out.read_text()
        cursor=None
        try:
            cursor=re.search(r'-- cursor: (\S+)',utility('journal-before',['journalctl','-k','-b','-n','0','--show-cursor','--no-pager']))[1]
            record['GPU_executed']=True
            c,out,err=owner.run('canonical-'+mode,argv,env,P,wall=360,cpu=4320,rss_cap=16<<30,
                                text_cap=64<<20,total_cap=128<<20,file_cap=64<<20)
            stdout,stderr=out.read_text(),err.read_text()
            record.update(stdout=stdout,stderr_sha256=sha(err),exit_code=c['exit_code'])
            record['runtime_errors']=[s for s in stderr.splitlines() if re.search(r'(?:UR|ZE)_RESULT_ERROR_|terminate called|Strata SYCL cleanup:|FAIL',s)]
            require(module.closed(c) and not record['runtime_errors'],'owned component/runtime failure: '+stderr[-3000:])
            uncertain=mode not in ('native-normal','normal','prepare-oom-null','prepare-second-oom-null','restore-oom-null','native-reserve-oom-null')
            if uncertain:
                expected='canonical dense bank: uncertain allocation result; owners retained'
                require(c['exit_code']==-6 and stderr.count(expected)==1 and
                        'unexpected ordinary' not in stderr and 'COMPONENT_ONLY' not in stdout,
                        'exact allocator uncertainty fail-stop: '+stderr[-3000:])
                record.update(expected_fail_stop=True,mechanism_excerpt=expected)
            else:
                require(c['exit_code']==0,'normal component exit')
                if mode=='native-normal':
                    require(stdout.strip()=='PASS matrices=300 layers=48 snapshot=2116951040 bank=50918400','native all-payload result')
                else:
                    match=re.fullmatch(r'COMPONENT_ONLY canonical=1538035200 bank=32284672 persistent=82821120 rows=(\d+); no model/runtime-quality claim\n',stdout)
                    require(match is not None and int(match[1])>0,'canonical all-payload and PLE result')
                    record['canonical_rows_checked']=int(match[1])
            require(all(sha(P/n)==h for n,h in build['source_pins'].items()),'candidate source stable')
            record.update(passed=True,complete=True)
        except BaseException as error:
            record['error']=repr(error);raise
        finally:
            try:
                if cursor and owner.active is None:
                    entries=[json.loads(s) for s in utility('journal-final',['journalctl','-k','-b','--after-cursor',cursor,'--no-pager','-o','json']).splitlines() if s.strip()]
                    record.update(fresh_faults=[e for e in entries if FAULT.search(str(e.get('MESSAGE','')))],
                                  boot_unchanged=boot_id()==record['boot_id'],driver_dump_exists=(DEVICE/'devcoredump').exists())
            finally:
                record.update(active=owner.active is not None,finished_utc=module.utc())
                if record['active'] or not all(module.closed(c) for c in owner.commands) or 'fresh_faults' not in record or record.get('fresh_faults') or not record.get('boot_unchanged') or record.get('driver_dump_exists'):
                    record.update(passed=False,complete=False)
                persist()
                with output.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
        print(json.dumps(dict(passed=record['passed'],mode=mode,exit_code=record['exit_code'],fresh_faults=len(record['fresh_faults']))))


if __name__=='__main__':
    require(len(sys.argv)==2,'one component mode argument');main(sys.argv[1])
