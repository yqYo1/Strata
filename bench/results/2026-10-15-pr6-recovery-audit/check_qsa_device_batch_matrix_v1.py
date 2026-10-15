#!/usr/bin/env python3
"""Main serial representative QSA device-batch component matrix after first-check PASS."""
import fcntl
import json
import os
from pathlib import Path
import re
import sys
sys.dont_write_bytecode=True
A=Path(__file__).resolve().parent
sys.path.insert(0,str(A))
from analyze_prefill_window_profile_v2 import B,P,load_owner,require,sha
from run_prefill_window_profile_v4 import DEVICE,FAULT,boot_id


def main():
    root=B/'main-qsa-device-batch-matrix-root-v1'
    target=A/'qsa-device-batch-matrix-v1.json'
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(not root.exists() and not target.exists(),'fresh first GPU gate')
        build_path=A/'qsa-device-batch-build-v1.json';build=json.loads(build_path.read_text())
        prior_path=A/'qsa-device-batch-small-v1.json';prior=json.loads(prior_path.read_text())
        require(all(r['passed'] and r['complete'] and not r['active'] for r in (build,prior)),'actual closed build/model gates')
        require(not prior['fresh_faults'] and prior['boot_unchanged'] and not prior['driver_dump_exists'] and
            prior['boot_id']==boot_id(),'latest GPU closure')
        module=load_owner(P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
        for r in (build,prior):
            for c in r['commands']:
                require(module.closed(c),'closed previous commands')
                for old in c['owners']:
                    now=module.identity(old['pid'])
                    require(now is None or now['start_ticks']!=old['start_ticks'],'previous owner still present')
        require(all(sha(P/n)==h for n,h in build['source_pins'].items()),'exact built production source')
        exe=Path(build['fixture']);require(sha(exe)==build['fixture_sha256'],'actual linked fixture')
        require((DEVICE/'driver').resolve().name=='xe' and not (DEVICE/'devcoredump').exists(),'driver/dump preflight')
        root.mkdir(mode=0o700)
        env=dict(prior['environment'])
        for key in list(env):
            if key.startswith(('STRATA_MEASUREMENT_','STRATA_PREFILL_BOUNDARY_')):
                env.pop(key)
        env.pop('UR_LOG_TRACING',None)
        env.update(STRATA_ROPE_TABLE='1',UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION',
            UR_LOG_VALIDATION='level:warning;flush:warning;output:stderr',
            XDG_CACHE_HOME=str(root/'cache'))
        for k,v in dict(UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1',NEOReadDebugKeys='1',EnableDirectSubmission='0',
            STRATA_SYCL_PREFILL_SAME_QUEUE='1',SYCL_CACHE_PERSISTENT='0',STRATA_SYCL_VERIFY_NATIVE_CAPTURE='0').items():
            require(env[k]==v,'unchanged safety option '+k)
        require(not any('TOKEN' in k or 'API_KEY' in k or k=='LD_PRELOAD' or
            k.startswith(('UNITRACE_','PTI_')) for k in env),'bounded first-check environment')
        plans=[['small','yarn','short','mrope','dispatch'],
            ['invalid','none','analytic','plain','direct'],
            ['invalid','yarn','full','mrope','dispatch'],
            ['large','linear','full','plain','direct'],
            ['large','yarn','short','mrope','dispatch'],
            ['physical','none','analytic','plain','direct'],
            ['physical','yarn','short','mrope','dispatch']]
        r=dict(passed=False,complete=False,active=True,GPU_executed=False,commands=[],
            recipe_sha256=sha(Path(__file__)),references={p.name:sha(p) for p in (build_path,prior_path)},
            environment=env,plans=plans,samples=[],binary_sha256=sha(exe),source_pins=build['source_pins'],boot_id=boot_id(),
            limits=dict(wall_seconds_per_case=600,RSS_bytes=12<<30,text_bytes=64<<20,job_bytes=512<<20),
            quality_qualified=False,performance_qualified=False,restore_qualified=False,
            physical_model_262144_qualified=False,
            scope='Representative7 native processes: scaling/RoPE/M-RoPE direct and production dispatch, invalid nonmutation,4096/8192 tails and two real component262144 last-block allocations. Not full model physical context, conversation RESTORE, task quality or speed',
            retention_consumer='Individual component results and fault/normal ownership closure; retire verbose successful duplicate logs after compact evidence.')
        def persist():(root/'record.json').write_text(json.dumps(r,indent=2)+'\n')
        owner=module.Owner(root/'owned',persist);r['commands']=owner.commands;persist()
        util_env=dict(PATH='/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8')
        def utility(label,args):
            c,out,err=owner.run(label,args,util_env,P,wall=60,cpu=60,rss_cap=2<<30,
                text_cap=4<<20,total_cap=256<<20,file_cap=4<<20)
            require(module.closed(c) and c['exit_code']==0,label+': '+err.read_text()[-2000:])
            return out.read_text()
        cursor=None
        try:
            cursor=re.search(r'-- cursor: (\S+)',utility('journal-before',
                ['journalctl','-k','-b','-n','0','--show-cursor','--no-pager']))[1]
            r.update(journal_before_cursor=cursor,GPU_executed=True);persist()
            for i,plan in enumerate(plans):
                case_env=dict(env,STRATA_SYCL_QSA_INDEXER_DEVICE_BATCH='1' if plan[-1]=='dispatch' else '0')
                argv=[str(exe),*plan]
                c,out,err=owner.run('QSA-matrix-'+str(i),argv,case_env,P,wall=600,
                    cpu=600*len(os.sched_getaffinity(0)),rss_cap=12<<30,text_cap=64<<20,
                    total_cap=512<<20,file_cap=64<<20)
                sample=dict(plan=plan,exit_code=c.get('exit_code'),stdout=out.read_text(),
                    stderr_sha256=sha(err),stderr_bytes=err.stat().st_size)
                sample['runtime_errors']=[line for line in err.read_text().splitlines() if
                    re.search(r'(?:UR|ZE)_RESULT_ERROR_|terminate called|cleanup fail|Strata SYCL cleanup:',line)]
                r['samples'].append(sample);persist()
                require(module.closed(c) and c['exit_code']==0,'actual QSA matrix: '+err.read_text()[-6000:])
                result=json.loads(sample['stdout']);sample['component']=result
                require(result['status']=='complete' and result['suite']==plan[0] and result['api']==plan[-1]
                    and result['state_comparisons']>0,'complete exact component assertions')
                if plan[0]=='small': require(result['fresh_zero_multicell_batches']==3,'fresh n2/3/4')
                if plan[0]=='invalid':
                    require(result['invalid_rejections']>0 and
                        result['invalid_rejections']==result['invalid_unchanged_checks'],'every rejection had exact nonmutation check')
                else: require(result['selection_comparisons']>0,'causal score and ID comparisons')
                require(not sample['runtime_errors'] and not (DEVICE/'devcoredump').exists(),'runtime/fault gate')
                sample['passed']=True;persist()
                print('PASS '+str(i)+' '+' '.join(plan),flush=True)
            r['physical_component_262144_passed']=len([s for s in r['samples'] if s['plan'][0]=='physical' and s['passed']])==2
            require(all(sha(P/n)==h for n,h in r['source_pins'].items()),'source stability')
            r.update(passed=True,complete=True)
        except BaseException as e:
            r['error']=repr(e);raise
        finally:
            try:
                if cursor and owner.active is None:
                    entries=[json.loads(s) for s in utility('journal-final',
                        ['journalctl','-k','-b','--after-cursor',cursor,'--no-pager','-o','json']).splitlines() if s.strip()]
                    r.update(fresh_faults=[x for x in entries if FAULT.search(str(x.get('MESSAGE','')))],
                        boot_unchanged=boot_id()==r['boot_id'],driver_dump_exists=(DEVICE/'devcoredump').exists())
            finally:
                r.update(active=owner.active is not None,finished_utc=module.utc())
                if r['active'] or not all(module.closed(c) for c in r['commands']) or r.get('fresh_faults') or not r.get('boot_unchanged') or r.get('driver_dump_exists'):
                    r.update(passed=False,complete=False)
                persist();target.write_text(json.dumps(r,indent=2)+'\n')
        print(json.dumps(dict(passed=r['passed'],samples=len(r['samples']),physical_component_262144_passed=r.get('physical_component_262144_passed'),fresh_faults=len(r.get('fresh_faults',[])))))


if __name__=='__main__':main()
