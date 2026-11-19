#!/usr/bin/env python3
"""Bounded coherent long-prefill producer/consumer checks;repository LICENSE.

Main-authored from owned eager_fullfit_device.py at04634f850. New checks use
the original owned workspace/routes and Down fixtures,never XeStrata sources.
No component timing,model admission,phase adoption or retry is implicit.
"""
import copy
import fcntl
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode=True
import prefill_window_build as build
import qsa_cached_subgroup as diagnostics

P,B,RESULT=build.P,build.B,build.RESULT
ROOT,KEY=B/'prefill-window-device-v1','prefill_window_device_v1'
require,sha,write=build.require,build.sha,build.write
STAGES=['host','bank-smoke','long-smoke','bank-validate','long-validate','route0','route1']
HOST=dict(wall=90,cpu=120,rss_cap=2<<30,text_cap=128<<10,file_cap=1<<20,total_cap=4<<20)
SMALL=dict(wall=180,cpu=360,rss_cap=4<<30,text_cap=32<<20,file_cap=64<<20,total_cap=80<<20)
LARGE=dict(wall=900,cpu=2400,rss_cap=16<<30,text_cap=4<<20,file_cap=64<<20,total_cap=128<<20)
WHOLE_BYTES=640<<20
FIXTURE_ENV=dict(STRATA_SYCL_PREFILL_NATIVE16='bf16',STRATA_SYCL_PREFILL_NATIVE16_BACKEND='vllm',
    STRATA_SYCL_PREFILL_NATIVE16_VLLM_TILE='auto',STRATA_SYCL_PREFILL_GU_SWIGLU='0',
    STRATA_SYCL_PREFILL_DOWN_SCATTER='0',STRATA_SYCL_PREFILL_DOWN_SPLIT_KERNEL='original')


def compact(row):
    out={k:v for k,v in row.items() if k not in ['sources','commands']}
    out['sources']=dict(files=len(row['sources']),sha256=build.original.fingerprint(row['sources']))
    return out


def save(row):
    write(ROOT/'record.json',row);data=json.loads(RESULT.read_text());data[KEY]=compact(row);write(RESULT,data)


def native():
    value=json.loads((build.ROOT/'record.json').read_text());build.check(value)
    require(value['stage']=='built' and value['passed'] and value['complete']
        and value['ownership_closed'] and not value['active'],'closed coherent candidate build required')
    for name in ['compiled','generated','links','objects','produced']:
        require(all(sha(p)==h for p,h in value['actual_inputs'][name].items()),'actual build input changed:'+name)
    require(all(sha(v['path'])==v['sha256'] and Path(v['path']).stat().st_size==v['bytes']
        for v in value['binaries'].values()),'candidate executable changed')
    return value


def verify(row):
    require(build.ur_validation.committed(RESULT,(KEY,))==compact(row),'commit preceding device stage')
    require(build.source.pin(P,list(row['sources']))['source_sha256']==row['sources'],'device controls changed')
    value=native()
    require(sha(build.ROOT/'record.json')==row['native_record_sha256'] and value['binaries']==row['binaries'],
        'exact native generation changed')
    require(build.guards.driver_guard()==row['driver'] and
        Path('/proc/sys/kernel/random/boot_id').read_text().strip()==row['boot'],'platform changed')
    require(row['fixture_environment']==FIXTURE_ENV,'immutable fixture backend/layout changed')
    require(build.ownership.artifact_bytes(ROOT)<=row['whole_bytes'],'whole diagnostic byte budget exceeded')
    return value


def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()),'fresh plan only;no retry')
    value=native();sources=build.source.pin(P,['tools/sycl'])
    manifest=value['foundation']['runtime_manifest']
    require(sha(manifest['path'])==manifest['sha256'],'runtime manifest changed')
    ROOT.mkdir(mode=0o700)
    row=dict(stage='prepared',source_commit=sources['revision'],sources=sources['source_sha256'],
        native_base_commit=value['native_base_commit'],native_overlay_commit=value['source_commit'],
        native_record_sha256=sha(build.ROOT/'record.json'),binaries=value['binaries'],
        environment=value['recipe']['environment'],fixture_environment=FIXTURE_ENV,runtime_manifest=manifest,
        native_validation_probe=value['native_validation_probe'],driver=build.guards.driver_guard(),
        boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),results=[],commands=[],
        passed=False,complete=False,active=False,ownership_closed=True,
        budgets=dict(host=HOST,traced_small=SMALL,quiet_large=LARGE),whole_bytes=WHOLE_BYTES,
        plan=dict(order=STAGES,
            host='Fresh same-build CPUfixtures:48stable/inverse/unique/bounded-job routes through131072,49152workspace layouts and134admission/terminal+9459KVhash span checks. Host proofs do not qualify physical GPUcontext.',
            small='Traced+parameter-validated U1normal/cancellation4expert cases,then long-prefill3/1and17/8producer/consumer cases. Exact same-PID pre-main loader proof and actual enqueues required.',
            large='Quiet parameter validation:U1zero/one/127/512experts and cancellation;long-prefill117489/131071/131072with contiguous/irregular maps,primary65536,guards,bitwise original FP32FMA/address checks,and full weight/gather dataflow at131072irregular.',
            routes='Separate immutable pre-exec DownScatter0andlarge processes,each fixed131072window/65536split with2333/2334/2335/8192job rows and contiguous/irregular maps. Validate actual production dispatch,threshold,fallback,unchanged inputs and false-route no-write.',
            providers='Diagnostic-only LD_DEBUG=libs must prove initialized SYCL,URv2,LevelZero GPU,IGC and pre-main providers against original runtime manifest. Safety environment remains fixed before exec.',
            acceptance='Commit each stage before next. All ordered checks normal0,reaped/empty,quiet validation,unchanged boot/driver and no fresh fault. Stop first input/provider/owner/case/byte/budget failure. No component performance gate or model auto-launch.',
            next='A separate committed model plan must isolate old-path coherence,98Krelocation at equal bank/window,128Kwindow at fixed bank1,and the combination against256/window10. Preserve the independent bank loss and all infeasible cells. Both prefill strategies,decode implementation/configuration/placement and actual handoff require independent preservation gates.',
            limits='Component numerical/address/lifetime evidence only. Actual full-model GPUadmission,PP/TTFT,quality,agent,thinking1Kgeneration with observed strong/weak MTP,decode preservation,physical262144/tails/RESTORE and ordinary runtime teardown remain unqualified.',
            retention='Keep case counts/conservation/closure/provider evidence until named model admission consumes it. Retire successful raw API/loader/JIT output after original source recovery;no new binary generation.'))
    save(row);print('Prepared seven ordered long-prefill component checks;commit before execution.',flush=True)


def loaded_providers(errors,row):
    providers=json.loads(Path(row['runtime_manifest']['path']).read_text())
    probe=row['native_validation_probe'];providers[probe['library']['path']]=probe['library']['sha256']
    allowed={str(Path(p).resolve()):h for p,h in providers.items()}
    loaded={str(Path(p.strip()).resolve()) for p in re.findall(r'^\s*\d+:\s+calling init:\s*(/[^\n]+)$',errors,re.M)}
    require(loaded and all(p in allowed and sha(p)==allowed[p] for p in loaded),'unexpected initialized ELF/provider')
    required=[probe['library']['path'],probe['loader']['path'],row['environment']['UR_ADAPTERS_FORCE_LOAD']]
    require(all(str(Path(p).resolve()) in loaded for p in required),'required runtime/probe not initialized')
    require(any('libsycl.so' in p for p in loaded) and any('libze_intel_gpu.so' in p for p in loaded)
        and any('/libigc.so' in p for p in loaded),'actual SYCL/GPU/compiler provider absent')
    return dict(initialized_ELFs=len(loaded),sha256=build.original.fingerprint({p:allowed[p] for p in sorted(loaded)}),
        required_providers_verified=True,source='glibc constructors;not speed or hardware utilization')


def cpu_cases(label,text):
    if label=='prefill_moe_routes_test':
        pattern=r'^routes rows=(\d+) hot=([01]) split=([01]) long_prefill=([01]) inverse=PASS unique=PASS bounded_jobs=PASS legacy=PASS$'
        cases=[tuple(map(int,m)) for m in re.findall(pattern,text,re.M)]
        expected=[(n,hot,split,long) for lengths,split,long in [
            ([1,8192,8193,16384,32768,65536,65537,78452,81920],0,0),
            ([2,17,8193,65537,81921,117488],1,0),
            ([2,17,8193,65537,81921,117488,117489,131071,131072],1,1)]
            for n in lengths for hot in [0,1]]
        require(cases==expected and text.count('moe_accum_routes PASS GPU=UNRUN')==1,'route coverage differs')
        return dict(cases=len(cases),maximum_rows=131072,stable_inverse_unique_bounded_jobs=True)
    if label=='prefill_moe_workspace_test':
        require(text.strip()=='prefill_moe_workspace CPU_PASS rows_checked=49152 moved_planes=2 extra_131071=729799424 historical_margin=22492160 GPU=UNRUN runtime_admission=UNRUN model_math=UNRUN model_performance=UNRUN physical262144_model=UNRUN','workspace coverage differs')
        return dict(rows_checked=49152,relocated_planes=2,extra_bytes_at131071=729799424,actual_GPU_admission=False)
    require(label=='full_context_diagnostic_host_test' and text.strip()==
        'cases=134 full_context_request_and_terminal_bounds=PASS\nbounds_cases=134 kv_hash_cases=9459 byte_order_and_bounds=PASS full_extent_spans=1 GPU=UNRUN',
        'host-only admission/span coverage differs')
    return dict(bounds_cases=134,kv_hash_cases=9459,physical_model_GPU_qualified=False)


def gpu_cases(stage,text):
    if stage.startswith('bank-'):
        cases=[tuple(map(int,m)) for m in re.findall(r'^prefill_fullfit_eager PASS experts=(\d+) source_stride=(\d+) layers=(\d+) cancellations=(\d+) rejects=(\d+)$',text,re.M)]
        expected=[(4,256,2,0,4),(4,512,2,2,4)]
        if stage=='bank-validate':expected += [(0,256,2,0,4),(1,256,3,3,4),(127,512,2,0,4),(512,256,2,0,4)]
        require(cases==expected and text.count('prefill_fullfit_eager_fixture PASS model=UNRUN physical262144_RESTORE=UNQUALIFIED')==1,'bank fixture coverage differs')
        return dict(cases=[dict(zip(['experts','source_stride','layers','cancellations','refusals'],c)) for c in cases])
    values=[json.loads(line) for line in text.splitlines() if line.startswith('{')]
    require(values and all(v.get('status')=='PASS' for v in values)
        and text.splitlines().count('fixture_status=PASS')==1,'missing/failed Down fixture evidence')
    if stage.startswith('route'):
        enabled=int(stage[-1]);require(len(values)==8,'eight route cases required')
        for value,(irregular,rows) in zip(values,[(i,r) for i in [0,1] for r in [2333,2334,2335,8192]]):
            first=(131072+rows-1)//rows-1 if irregular else 0
            last=131071 if irregular else rows-1
            require(value==dict(kind='selected_route',rows=rows,window_rows=131072,split_rows=65536,
                source_first=first,source_last=last,enabled=enabled,selected=int(enabled and rows>=2334),
                expected_mask=0,inputs_immutable=True,false_route_no_write=True,status='PASS'),'actual production route differs')
        return dict(cases=values,actual_dispatch_verified=True)
    cases=[v for v in values if v['kind']=='long_prefill_case']
    expected=[(3,1,0,0,0),(17,8,3,1,1)] if stage=='long-smoke' else [
        (n,65536,17,i,0) for i in [0,1] for n in [117489,131071,131072]]
    require(len(cases)==len(expected),'long-prefill case count differs')
    for value,condition in zip(cases,expected):
        n,primary,target,irregular,edges=condition
        require(all(value[k]==v for k,v in dict(tokens=n,primary_rows=primary,target_rows=target,
            irregular=irregular,edges=edges,physical_final_row=True,status='PASS').items())
            and value['jobs']>0,'long-prefill boundary coverage differs')
    dataflow=[v for v in values if v['kind']=='split_dataflow']
    expected_tokens=[17] if stage=='long-smoke' else [131072]
    require([v['tokens'] for v in dataflow]==expected_tokens and all(v['full_weight_map']
        and v['full_gather'] and v['inputs_immutable'] and v['guards'] for v in dataflow),'full producer/consumer dataflow absent')
    details=[v for v in values if v['kind']=='split_correctness']
    require(len(details)==sum(v['jobs'] for v in cases) and all(v['full_FP32_host_oracle']
        and v['full_BF16_capture'] and v['guards'] and v['expected_mask']==0
        and 0<v['job_rows']<=8192 for v in details),'complete bitwise job/guard coverage absent')
    pending=[]
    for value in values:
        if value['kind']=='split_correctness':pending.append(value)
        elif value['kind']=='long_prefill_case':
            require(len(pending)==value['jobs'] and len({v['job'] for v in pending})==len(pending)
                and all(v['tokens']==value['tokens'] and v['primary_rows']==value['primary_rows']
                    for v in pending),'job proof/summary ordering differs')
            pending=[]
    require(not pending and all(v['kind'] in ('split_dataflow','long_prefill_case','split_correctness') for v in values),
        'unexpected long-prefill fixture output')
    return dict(cases=cases,dataflow=dataflow,bitwise_original_FP32_FMA_and_address_oracle=True)


def run(stage):
    row=json.loads((ROOT/'record.json').read_text());verify(row);frozen=copy.deepcopy(row)
    require(row['stage'] in ['prepared','in_progress'] and len(row['results'])==STAGES.index(stage)
        and all(v['passed'] for v in row['results']),'one ordered stage;no retry')
    root=ROOT/stage;root.mkdir(mode=0o700)
    for name in ['tmp','cache']:(root/name).mkdir(mode=0o700)
    cursor=subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15).strip().split('-- cursor: ')[-1]
    require(cursor and '\n' not in cursor,'kernel cursor required')
    owner=None;result=dict(mode=stage,passed=False,complete=False,GPU_executed=False,host_probe_accepted=False)
    try:
        owner=build.ownership.Owner(root/'owned',lambda:write(root/'ownership.json',owner.commands))
        env=dict(row['environment'],TMPDIR=str(root/'tmp'),XDG_CACHE_HOME=str(root/'cache'))
        if stage=='host':
            require(not env.get('LD_PRELOAD'),'host fixtures must not inherit a preload')
            cases={}
            for name in ['prefill_moe_routes_test','prefill_moe_workspace_test','full_context_diagnostic_host_test']:
                entry,out,err=owner.run(name,[row['binaries'][name]['path']],env,P,**HOST)
                require(build.ownership.closed(entry) and entry['exit_code']==0 and not err.read_text(),'normal quiet CPUfixture required')
                cases[name]=cpu_cases(name,out.read_text())
            result.update(cases=cases,normal_exit_codes=[0,0,0])
        else:
            probe=row['native_validation_probe'];trace=stage.endswith('smoke')
            env.update(row['fixture_environment'])
            env.update(STRATA_SYCL_PREFILL_FULLFIT_UPLOAD='eager',STRATA_TRACE='1',LD_DEBUG='libs',
                LD_PRELOAD=probe['library']['path'],STRATA_UR_PROBE_EXPECTED_LOADER=probe['loader']['path'],
                UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION' if trace else 'UR_LAYER_PARAMETER_VALIDATION',
                UR_LOG_VALIDATION='level:warning;flush:warning;output:file,'+str(root/'validation.log'))
            if trace:env['UR_LOG_TRACING']='level:info;flush:info;output:stderr'
            else:env.pop('UR_LOG_TRACING',None)
            if stage.startswith('bank-'):
                args=[row['binaries']['sycl_expert_bank_test']['path'],'--fullfit-eager-'+stage.split('-')[1]]
            else:
                args=[row['binaries']['sycl_moe_down_scatter_bench']['path']]
                if stage.startswith('route'):
                    arm=stage[-1];env['STRATA_SYCL_PREFILL_DOWN_SCATTER']='large' if arm=='1' else '0'
                    args+=['--long-prefill-route',arm]
                else:args+=['--long-prefill-'+stage.split('-')[1]]
            result['native_execution_attempted']=True
            entry,out,err=owner.run(stage,args,env,P,**(SMALL if trace else LARGE))
            errors=err.read_text();proof=build.ur_validation.preload_evidence(errors,entry['pid'])
            diagnostics.diagnostics(errors,entry['pid'])
            require(build.ownership.closed(entry) and entry['exit_code']==0,'normal native closure required')
            require(proof['loader']==probe['loader']['path'] and (not trace or 'urEnqueue' in errors),
                'actual loader/traced submissions absent')
            require(not (root/'validation.log').exists() or (root/'validation.log').stat().st_size==0,'parameter validation warning')
            result.update(GPU_executed=True,native_exit_code=0,validation=proof,
                providers=loaded_providers(errors,row),**gpu_cases(stage,out.read_text()))
        verify(frozen);result.update(passed=True,complete=True,host_probe_accepted=True)
    except BaseException as exc:result['error']=repr(exc)
    finally:
        if owner:build.ur_validation.finish(result,owner,cursor,row['boot'])
        result['passed']=bool(result['passed'] and result['complete'] and result.get('driver_after')==row['driver'])
        row['results'].append(result);row['commands']+=owner.commands if owner else []
        row['active']=bool(owner and owner.active);row['ownership_closed']=not row['active'] and all(build.ownership.closed(v) for v in row['commands'])
        row['complete']=len(row['results'])==len(STAGES) and all(v['passed'] for v in row['results'])
        row['passed']=row['complete'];row['stage']='failed' if not result['passed'] else 'closed' if row['complete'] else 'in_progress';save(row)
    require(result['passed'] and row['ownership_closed'],result.get('error','native component check failed'))
    print('Long-prefill component '+stage+' closed0;model/speed/full context unqualified.',flush=True)


if __name__=='__main__':
    require(len(sys.argv)==2 and sys.argv[1] in ['prepare',*STAGES],'prepare|'+ '|'.join(STAGES))
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);build.guards.no_model_owners([])
        if sys.argv[1]=='prepare':prepare()
        else:run(sys.argv[1])
