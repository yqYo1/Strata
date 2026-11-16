#!/usr/bin/env python3
"""Isolate wait placement in the qualified 1024-row mapped-source component.

Main-authored; repository LICENSE. No rebuild or production/model adoption.
"""
import fcntl
import json
import math
from pathlib import Path
import re
import subprocess
import sys
sys.dont_write_bytecode = True
import prefill_mapped_source_reuse as reuse

c = reuse.component
P, B, RESULT = c.P, c.B, c.RESULT
ROOT = B/'prefill-mapped-wait-order-v2'
KEY = 'prefill_mapped_wait_order_v2'
PREFIXES = [*c.PREFIXES, 'tests/core/test_mapped_source_reuse.py']
require, sha, signature = c.require, c.sha, c.fingerprint


def compact(row):
    out = {k:v for k,v in row.items() if k not in ['project','commands']}
    out['project'] = {k:v for k,v in row['project'].items() if k!='source_sha256'}
    return out


def save(row):
    c.write(ROOT/'record.json',row)
    data=json.loads(RESULT.read_text());data[KEY]=compact(row);c.write(RESULT,data)


def native_metadata_gate(row,native):
    # These are consumed transitively by the actual provider/first-init reader.
    for key in ['binaries','native_source_commit','native_validation_probe',
                'runtime_manifest','validation_provider','driver','boot']:
        require(key in row and signature(row[key])==signature(native[key]),
                'original native metadata missing/changed:'+key)


def check(row, committed=True):
    if committed:
        require(signature(compact(row))==signature(c.ur_validation.committed(RESULT,(KEY,))),
                'commit exact typed wait-order plan and previous outcome')
    require(c.original.coherent.build.common.source.pin(P,PREFIXES)['source_sha256']
            ==row['project']['source_sha256'], 'frozen wait-order inputs changed')
    native,_=reuse.verify(expected=row['reuse'])
    native_metadata_gate(row,native)
    for case in row['cases']:
        require(case['rows']==1024 and case['source']=='mapped-neo'
            and case['arm'] in ['serial','overlap']
            and case['argv']==[row['binaries']['benchmark']['path'],case['arm'],'1024','512','mapped-neo'],
            'wait placement is the sole independent native variable;4096 is ineligible')
    return native


def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()),
            'existing wait-order attempt:inspect its owner;no retry')
    native,proof=reuse.verify()
    failed=c.ur_validation.committed(RESULT,('prefill_mapped_wait_order_v1',))
    require(failed['stage']=='failed' and failed['active'] is False
        and failed['failure_closure']['physical_process_closed'] is True
        and failed['failure_closure']['application_GPU_commands']==0,
        'preserve original pre-runtime controller failure and positive process closure')
    project=c.original.coherent.build.common.source.pin(P,PREFIXES)
    ROOT.mkdir(mode=0o700);(ROOT/'owned').mkdir(mode=0o700)
    cases=[]
    for label,arm,trace,pair in [('validated-serial','serial',True,None),
                                ('validated-overlap','overlap',True,None)]+[
            (f'clean-pair{pair}-{arm}',arm,False,pair) for pair in range(3)
            for arm in (['serial','overlap'] if pair%2==0 else ['overlap','serial'])]:
        root=ROOT/label
        template=next(v for v in native['cases'] if v.get('rows')==1024
                      and v['source']=='mapped-neo' and v['trace'] is trace)
        env={k:(v.replace(template['root'],str(root))) for k,v in template['environment'].items()}
        cases.append(dict(label=label,root=str(root),arm=arm,rows=1024,source='mapped-neo',
            trace=trace,pair=pair,environment=env,
            argv=[native['binaries']['benchmark']['path'],arm,'1024','512','mapped-neo']))
    row=dict(stage='prepared',passed=True,complete=False,active=False,ownership_closed=True,
        project=project,reuse=proof,native_source_commit=reuse.NATIVE,binaries=native['binaries'],
        native_validation_probe=native['native_validation_probe'],runtime_manifest=native['runtime_manifest'],
        validation_provider=native['validation_provider'],
        driver=native['driver'],boot=native['boot'],cases=cases,results=[],commands=[],
        model_trial_allowed=False,performance_qualified=False,
        plan=dict(unit='Only move lane.wait_ready(copied) from before dense.beta0 to after it. Same coherent original Runtime,source adapter,payload,1024-row FP16dense,tile,queues,allocation and512epochs. No native source/build change. Previous source-admission/producer/consumer checks were one necessary coherent ownership unit and are reused only after exact historical recovery.',
            evidence='Original0a3295b89 five-TU build/401source-view files/928compiler/25link/7object pins and both ELFs;closed9cc26a79b15safety+16timing cases.1024source-cost passes,4096fails and stays excluded. Original E2USM1024speed failure remains failed;this is a new mapped-source scoped comparison.',
            order='Two traced independent processes:serial,overlap. Then three clean independent-process pairs:serial/overlap,overlap/serial,serial/overlap. No retries,reordering or exclusions. Existing512epoch batch,exact dense and one event epoch warmup,dual-drain and128timer-pair checks unchanged.',
            gates='Committed typed inputs,unchanged original code/recipe/ELFs/provider/boot/safety environment,actual argv/environment/provider gates. Positive HOST1/import ID;full exact dyadic dense and payload/edge/source guards. Traced513payload copies/514dense launches;quiet validation,normal0,closed owners,no fresh fault. Batch>=100ms;max timer pair<=0.001batch.',
            decision='Keep all six clean individual samples and three overlap/serial ratios. Readiness-order component gain requires all three steady-wall ratios<1 and geometric mean<1. Lifecycle(preparation+wall+retirement) ratios must each<=1.10. Import is already part of preparation. This only permits a new production-consumer design/qualification plan;no model trial or adoption. Failed or variable gain is preserved without reclassifying samples.',
            stop='First source/admission/oracle/guard/validation/deadline/ownership/fault failure closes the attempt as failed. Complete all clean samples even if slower. No GPU reset,foreign process action,PTI collector or environment mutation.',
            retention='Reuse the same two component ELFs as the named temporary third generation;no fourth build. Keep compact useful samples,census/reuse proof and decision. Retire routine raw traces after closure and the component generation if no named next consumer remains. Existing independent phase references remain unchanged.',
            limits='Explicit component entrypoint only. Resident dense A is math input;copied next-bank B is not an operand. Same2MBsource is reused,not production50GBRAM or physicalPCIe evidence. Hardware saturation,modelPP/TTFT/TG,decode/handoff,agent/task,physical262144/RESTORE remain unqualified. Production prefill and decode selectors are unchanged.',
            provenance='Main-authored controller/reuse of own historical code under repository LICENSE;native original14508owned dense with previously checked Intel TLA provenance. No XeStrata code/design.'))
    native_metadata_gate(row,native)
    save(row);print('Prepared sole-variable mapped1024wait-order plan;commit before execution.',flush=True)


def execute():
    row=json.loads((ROOT/'record.json').read_text())
    try:check(row)
    except BaseException as error:
        row.update(stage='failed',passed=False,complete=False,prelaunch_error=repr(error))
        save(row);raise
    require(row['stage'] in ['prepared','testing'] and len(row['results'])<len(row['cases']),
            'fresh ordered wait-order stage required')
    supervisor=c.owner_api.Owner(ROOT/'owned',lambda:c.write(ROOT/'ownership.json',supervisor.commands))
    cursor=None
    try:
        case=row['cases'][len(row['results'])];root=Path(case['root'])
        require(not root.exists(),'existing case:inspect owner;no retry')
        root.mkdir(mode=0o700)
        for name in ['tmp','cache']:(root/name).mkdir(mode=0o700)
        cursor=subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15).split('-- cursor: ',1)[1].strip()
        row.update(stage='testing',active=True,ownership_closed=False);save(row)
        expected=['before-runtime','runtime-ready','prewarm-ready','before-timing','before-exit']
        gates,records,providers=[],[],{}
        env=c.engine_once.native_environment(row,case['environment'])
        def interaction(channel,_entry):
            while True:
                item=json.loads(channel.line(45))
                if item['kind']=='gate':
                    name=item['name'];require(len(gates)<len(expected) and name==expected[len(gates)],'gate order')
                    pid=str(channel.proc.pid)
                    actual=dict(v.split('=',1) for v in Path('/proc',pid,'environ').read_text().strip('\0').split('\0'))
                    require(all(actual.get(k)==v for k,v in env.items())
                        and set(actual)-set(env)<= {'PWD','OLDPWD','SHLVL','_'},'actual startup environment differs')
                    require(Path('/proc',pid,'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')==case['argv'],
                            'actual launch differs')
                    providers[name]=c.event.mappings(channel.proc.pid,dict(row,binary=row['binaries']['benchmark']))
                    if name!='before-runtime':
                        needed=[c.event.LOADER,env['ZE_ENABLE_ALT_DRIVERS'],env['UR_ADAPTERS_FORCE_LOAD'],
                                '/opt/intel/oneapi/compiler/2026.2/lib/libsycl.so']
                        if case['trace']:needed.append(c.event.VALIDATION)
                        require(all(str(Path(p).resolve()) in providers[name] for p in needed),'initialized provider missing')
                    gates.append(name);channel.send(b'continue\n')
                else:
                    records.append(item)
                    if item['kind']=='done':channel.exit(30);break
        entry,out,err=supervisor.run(case['label'],case['argv'],env,P,interaction=interaction,**c.RUN)
        require(c.owner_api.closed(entry) and entry['exit_code']==0,'native failed:'+str(err))
        validation=c.ur_validation.preload_evidence(err.read_text(),entry['pid'])
        require(not (root/'validation.log').exists() or (root/'validation.log').stat().st_size==0,'UR validation output')
        require(not re.search(r'\[(?:error|critical)\]|upload (?:lane|overlap) failure:|Strata SYCL cleanup:',err.read_text(),re.I),'ZE validation/cleanup error')
        require(gates==expected and len(records)==3 and records[0]['kind']=='batch'
            and records[1]['kind']=='source' and records[2]==dict(kind='done',passed=True),'incomplete batch')
        batch,source=records[:2]
        require(batch['arm']==case['arm'] and batch['gemm_rows']==1024 and batch['gemm_n']==1280
            and batch['gemm_k']==2560 and batch['iterations']==512 and batch['requested_H2D_bytes']==512*1971200
            and batch['lookahead']==1 and batch['oracle_and_guards'] is True,'actual timed dispatch/workload differs')
        require(batch['wall_s']>=0.1 and 0<=batch['timer_pair_max_s']<=batch['wall_s']*.001,'ineligible batch/clock overhead')
        require(source['source']=='mapped-neo' and type(source['import_property_type']) is int
            and source['import_property_type']==1 and type(source['import_allocation_id']) is int
            and source['import_allocation_id']>0,'actual source lease differs')
        require(all(type(source[k]) in [int,float] and math.isfinite(source[k]) and source[k]>=0
                    for k in ['preparation_s','import_s','retirement_s'])
            and source['preparation_s']>=source['import_s'],'invalid source lifecycle')
        result=dict(label=case['label'],arm=case['arm'],pair=case['pair'],clean=not case['trace'],batch=batch,
            source_lifecycle=source,lifecycle_plus_batch_s=source['preparation_s']+batch['wall_s']+source['retirement_s'],
            passed=True,native_exit_code=0,ownership_closed=True,validation=validation,
            providers_sha256=signature(providers))
        if case['trace']:
            trace=(root/'ur.trace').read_text()
            copies=sum('SUCCESS (ZE_RESULT_SUCCESS) in zeCommandListAppendMemoryCopy(' in v and 'size=1971200,' in v for v in err.read_text().splitlines())
            kernels=len(re.findall(r'---> (urEnqueueKernelLaunch\w*)',trace))
            require(copies==513 and kernels==514,'actual copy/dense count differs')
            result['validated_counts']=dict(payload_copies=copies,dense_launches=kernels)
        check(row,False);row['results'].append(result)
        row.update(passed=True,active=False,ownership_closed=True)
        if len(row['results'])==len(row['cases']):
            ratios=[];lifecycle=[]
            for pair in range(3):
                values={x['arm']:x for x in row['results'] if x['clean'] and x['pair']==pair}
                require(set(values)=={'serial','overlap'},'complete clean pair required')
                ratios.append(values['overlap']['batch']['wall_s']/values['serial']['batch']['wall_s'])
                lifecycle.append(values['overlap']['lifecycle_plus_batch_s']/values['serial']['lifecycle_plus_batch_s'])
            gm=math.prod(ratios)**(1/3)
            gain=all(x<1 for x in ratios) and gm<1 and all(x<=1.10 for x in lifecycle)
            row.update(stage='closed',complete=True,analysis=dict(paired_overlap_over_serial=ratios,
                geometric_mean_ratio=gm,paired_lifecycle_ratios=lifecycle,
                repeated_component_wall_gain=gain,model_trial_allowed=False))
    except BaseException as error:
        row.update(stage='failed',passed=False,complete=False,error=repr(error));raise
    finally:
        row['commands'].extend(supervisor.commands);row['active']=supervisor.active is not None
        row['ownership_closed']=supervisor.active is None and all(c.owner_api.closed(v) for v in row['commands'])
        if cursor:
            try:
                after=subprocess.check_output(['journalctl','-k','-b','--after-cursor',cursor,'--no-pager'],text=True,timeout=15)
                row['fresh_faults']=[v for v in after.splitlines() if re.search(r'gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout)',v,re.I)]
                row['boot_unchanged']=row['boot']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
                row['driver_after']=c.guards.driver_guard()
                if row['fresh_faults'] or not row['boot_unchanged'] or row['driver_after']!=row['driver']:
                    row.update(stage='failed',passed=False,complete=False)
            except BaseException as error:row.update(stage='failed',passed=False,complete=False,final_health_error=repr(error))
        save(row)
    print(json.dumps({k:row.get(k) for k in ['stage','passed','complete','ownership_closed','analysis']},sort_keys=True),flush=True)
    return 0 if row['passed'] else 1


if __name__=='__main__':
    require(sys.argv[1:] in [['prepare'],['next'],['verify']], 'prepare|next|verify')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);c.guards.no_model_owners([])
        if sys.argv[1]=='prepare':prepare()
        elif sys.argv[1]=='verify':_,proof=reuse.verify();print(json.dumps(proof,sort_keys=True))
        else:raise SystemExit(execute())
