#!/usr/bin/env python3
"""Fresh two-strategy candidate CLI/device checks; repository LICENSE.

Main-only serialized execution;no model,adoption or2048-call GU batch.
The planned owner runs each case once,stopping at its first failed gate.
"""
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import select
import subprocess
import sys
import time

sys.dont_write_bytecode=True
import build_prefill_candidate as build
import engine_once
import qualify_official_model as original
import measure_gdn_scheduling as gdn
import runtime_probe as runtime

P,B,RESULT=build.P,build.B,build.RESULT
ROOT,KEY=build.ROOT/'device-checks-v2','official_prefill_strategies_device_v2'
require,digest,write=build.require,build.digest,build.write


def pins(revision):return build.base.controls(revision,controller=Path(__file__).resolve())


def cases(loader):
    cli=[dict(name='cli-'+name,target='strata',args=args,cli=True,trace=False,expected_exit=code,needle=needle)
         for name,args,code,needle in (
             ('ttft',['--prefill-strategy','ttft','--help'],0,'--prefill-strategy ttft|throughput'),
             ('throughput',['--prefill-strategy','throughput','--help'],0,'--prefill-strategy ttft|throughput'),
             ('invalid',['--prefill-strategy','invalid'],1,'--prefill-strategy requires ttft or throughput'),
             ('case',['--prefill-strategy','Throughput'],1,'--prefill-strategy requires ttft or throughput'),
             ('missing',['--prefill-strategy'],2,'--prefill-strategy needs a value'))]
    return cli+original.cases(loader)+[
        dict(name='GDN-smoke',target='sycl_gdn_columns_bench',args=['--barriers3-smoke'],trace=True,gdn_mode='smoke'),
        dict(name='GDN-tails',target='sycl_gdn_columns_bench',args=['--barriers3-validate'],trace=False,gdn_mode='validate'),
        dict(name='GDN-context',target='sycl_gdn_columns_bench',args=['--barriers3-context'],trace=False,gdn_mode='context')]


def compact(row):
    out={k:v for k,v in row.items() if k!='inputs'}
    out['input_manifest']=dict(files=len(row['inputs']),sha256=hashlib.sha256(json.dumps(row['inputs'],sort_keys=True).encode()).hexdigest())
    return out


def save(row):
    write(ROOT/'record.json',row);d=json.loads(RESULT.read_text());d[KEY]=compact(row);write(RESULT,d)


def prepare():
    require(KEY not in build.base.committed(RESULT) and not ROOT.exists(),'fresh device/CLI cohort only')
    native,_=build.current();require(native['passed'] and native['stage']=='built-not-executed' and native['ownership_closed'] and not native['active'],'closed candidate build required')
    providers=gdn.inventory(native['runtime_basis']);probe=native['native_validation_probe']
    env=dict(native['runtime_basis']['environment'],STRATA_SYCL_PREFILL_NATIVE16='bf16',
        STRATA_SYCL_PREFILL_NATIVE16_BACKEND='vllm',STRATA_SYCL_PREFILL_NATIVE16_VLLM_TILE='auto',
        STRATA_SYCL_SELECT_RADIX4='1',STRATA_SYCL_SELECT_CYCLIC='0',STRATA_SYCL_PREFILL_COMMAND_PROFILE='0',
        STRATA_SYCL_PREFILL_FUSED_COMMAND_PROFILE='0',STRATA_SYCL_PREFILL_DOWN_SPLIT_KERNEL='original',
        ZE_ENABLE_ALT_DRIVERS=next(p for p in providers if 'libze_intel_gpu.so.' in p))
    require(env.get('LD_PRELOAD') in (None,probe['library']['path']),'unexpected inherited native preload')
    env.pop('LD_PRELOAD',None)
    env['CUDA_MODULE_LOADING']='EAGER'
    env=engine_once.native_environment(native,env)
    loader=next(p for p in providers if Path(p).name.startswith('libze_loader.so.'))
    revision=build.base.basis.build.git(P,'rev-parse','HEAD').decode().strip()
    ROOT.mkdir(mode=0o700)
    row=dict(stage='prepared',source_commit=revision,inputs=pins(revision),native_source_commit=native['source_commit'],
        implementation_commit=native['implementation_commit'],binaries=native['binaries'],
        runtime_manifest=native['runtime_basis'],native_validation_probe=probe,
        canary=native['runtime_basis']['canary'],environment=env,property_loader=loader,
        boot=native['boot'],driver=native['driver'],cases=cases(loader),processes=[],active=False,ownership_closed=True,
        complete=False,passed=False,GPU_executed=False,model_executed=False,inference_adopted=False,
        budgets=dict(wall=360,cpu=1080,rss_cap=6<<30,text_cap=8<<20,file_cap=8<<20,total_cap=64<<20),
        whole_bytes=64<<20,whole_wall_seconds=1800,
        criteria='Five real CLI parser/help/refusal cases plus13fresh official cases and3GDN cases,once in fixed order;stop first failure,no retries. Same frozen eight ELFs/providers,pre-main validation,quiet final log,normal ownership and unchanged driver/no fresh fault. GDN includes exact tails and262144carry/restore/existing decode tail. Public GU64calls in four16-call groups with stage conservation;never repeat the stalled2048batch. CLI source checks alone do not qualify runtime strategy/default/override or model speed. GDN/public-GU success is not a universal progress or model qualification.',
        retention='Keep compact individual case outcomes/coverage/provenance. Retire successful full API traces,JIT/polling output after verification. Keep only current official reference and this candidate for imminent model comparison;old6e40executables already retired. Separate legacy provenance holds remain.')
    save(row);print('Prepared21fresh CLI/device cases;commit before one serialized cohort.',flush=True)


def current(row):
    require(build.base.committed(RESULT)[KEY]==compact(row),'commit exact device cohort plan/result')
    native,allowed=build.current()
    require(native['binaries']==row['binaries'] and native['source_commit']==row['native_source_commit']
            and pins(row['source_commit'])==row['inputs'],'native/controller inputs changed')
    require(row['cases']==cases(row['property_loader']),'case plan changed')
    require(digest(build.ROOT/'build/compile_commands.json')==native['compile_command_attestation']['database_sha256'],
            'attested compiler recipe changed')
    if native.get('original_build_failure'):
        evidence=native['original_build_failure']['record']
        require(digest(evidence['path'])==evidence['sha256'],'original failed build receipt changed')
    providers=gdn.inventory(row['runtime_manifest']);return native,allowed,providers


def run():
    row=json.loads((ROOT/'record.json').read_text());native,allowed,providers=current(row)
    frozen_plan=copy.deepcopy(row)
    require(row['stage']=='prepared' and not (ROOT/'owned').exists(),'one fresh cohort only')
    cursor=next(s.split(': ',1)[1] for s in subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15).splitlines() if s.startswith('-- cursor: '))
    deadline=time.monotonic()+row['whole_wall_seconds'];owner=None;active_case=None
    def persist():write(ROOT/'execution.json',dict(processes=row['processes'],active_case=active_case,
        commands=owner.commands if owner else [],active=bool(owner and owner.active)))
    try:
        owner=build.base.supervision.Owner(ROOT/'owned',persist)
        for case in row['cases']:
            active_case=case['name'];folder=ROOT/case['name'];folder.mkdir(mode=0o700)
            for sub in ('tmp','cache'):(folder/sub).mkdir()
            env=dict(row['environment'],TMPDIR=str(folder/'tmp'),XDG_CACHE_HOME=str(folder/'cache'),
                STRATA_SYCL_PREFILL_GU_SWIGLU='large' if case.get('gu_arm',1) else '0',
                STRATA_SYCL_PREFILL_DOWN_SCATTER='large',STRATA_TEST_MAIN_GATES='1',
                STRATA_SYCL_PREFILL_TOPK_EARLY=str(case.get('topk',0)),
                UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION' if case['trace'] else 'UR_LAYER_PARAMETER_VALIDATION',
                UR_LOG_TRACING='level:info;flush:info;output:file,'+str(folder/'ur.trace'),
                UR_LOG_VALIDATION='level:warning;flush:warning;output:file,'+str(folder/'validation.log'))
            artifact=row['binaries'][case['target']] if case['target'] else row['canary']
            mapped,gates,env_seen=set(),[],False
            def observe(pid):
                nonlocal env_seen
                actual=dict(v.split(b'=',1) for v in Path('/proc',str(pid),'environ').read_bytes().split(b'\0') if b'=' in v)
                require(all(actual.get(k.encode())==v.encode() for k,v in env.items()) and
                        set(actual)-{k.encode() for k in env}<={b'PWD',b'OLDPWD',b'SHLVL',b'_'},'native startup environment differs')
                seen=runtime.executable_mappings(Path('/proc',str(pid),'maps').read_text())
                require(seen<=set(providers)|{str(Path(artifact['path']).resolve())},'unplanned native provider')
                mapped.update(seen);env_seen=True
            def monitor(io,entry):
                next_size=0.
                while not io.exit_watch.poll(0) or not io.eof:
                    io.poll();require(time.monotonic()<deadline,'whole cohort wall budget')
                    try:observe(io.proc.pid)
                    except (FileNotFoundError,ProcessLookupError):pass
                    except PermissionError:require(io.exit_watch.poll(20),'live native /proc access denied')
                    if time.monotonic()>=next_size:
                        require(build.base.supervision.artifact_bytes(ROOT)<=row['whole_bytes'],'whole cohort byte budget');next_size=time.monotonic()+.25
                    if select.select([io.proc.stdout],[],[],.01)[0]:
                        io.read()
                        while b'\n' in io.pending:
                            line,_,tail=io.pending.partition(b'\n');io.pending[:]=tail;text=line.decode()
                            if text.startswith(('GATE stage=','SPLIT_READY ','COMPONENT_READY ')):
                                phase=text.split('=',1)[1] if text.startswith('GATE ') else text.split(' ',1)[1]
                                require(len(gates)<3 and phase==['before-runtime','runtime-ready','before-exit'][len(gates)],'native gate order')
                                observe(io.proc.pid);gates.append(phase);io.send(b'continue\n' if text.startswith('GATE ') else b'X')
                io.pending.clear();io.exit(1)
            entry,out,err=owner.run(case['name'],[artifact['path'],*case['args']],env,P,interaction=monitor,**row['budgets'])
            require(build.base.supervision.closed(entry) and entry['exit_code']==case.get('expected_exit',0),'native exit differs: '+case['name'])
            activation=runtime.validation.preload_evidence(err.read_text(),entry['pid'])
            require(activation['loader']==env['STRATA_UR_PROBE_EXPECTED_LOADER'],'wrong pre-main loader')
            require(not (folder/'validation.log').exists() or (folder/'validation.log').stat().st_size==0,'final validation is not quiet')
            stdout,stderr=out.read_text(),err.read_text()
            if case.get('cli'):
                require(case['needle'] in stdout+stderr,'CLI diagnostic/help missing');details=dict(exit_code=entry['exit_code'],expected_message=True,GPU_body_requested=False)
            elif case.get('gdn_mode'):
                require('fixture_status=PASS' in stdout and gates==['before-runtime','runtime-ready','before-exit'],'GDN completion/gates missing')
                details=dict(events=gdn.events_for(stdout,case['gdn_mode'],0))
            else:details=original.analyze(case,stdout,entry['pid'])
            if case['target'] and not case.get('cli'):
                require(env_seen and {env['LD_PRELOAD'],env['STRATA_UR_PROBE_EXPECTED_LOADER'],env['ZE_ENABLE_ALT_DRIVERS'],
                    '/opt/intel/oneapi/compiler/2026.2/lib/libsycl.so.9.0.0'}<=mapped,'native provider observation incomplete')
                if case['trace'] and case['name']!='Topk-properties':require('urEnqueue' in (folder/'ur.trace').read_text(),'GPU API submission trace missing')
            if case['target'] in ('sycl_submission_test','sycl_qsa_topk_prefill_entry_test','sycl_moe_down_scatter_bench'):
                require(gates==['before-runtime','runtime-ready','before-exit'],'native gates incomplete')
            check=dict(host_probe_accepted=True);runtime.validation.finish(check,owner,cursor,row['boot'])
            require(check['passed'] and check['driver_after']==row['driver'],'driver/fault/ownership gate failed')
            row['processes'].append(dict(name=case['name'],passed=True,ownership_closed=True,native_exit_code=entry['exit_code'],
                pre_main_parameter_validation=True,quiet_final_validation=True,gates=gates,details=details))
            persist();print('Passed '+case['name'],flush=True)
        current(frozen_plan)
        row.update(stage='closed',complete=True,host_probe_accepted=True,GPU_executed=True)
    except BaseException as error:row.update(stage='failed',complete=False,error=repr(error),failed_case=active_case,host_probe_accepted=False)
    finally:
        if owner:runtime.validation.finish(row,owner,cursor,row['boot'])
        if row.get('driver_after')!=row['driver']:row.update(passed=False,final_driver_mismatch=True)
        row['passed']=bool(row.get('passed') and row['complete']);row['active']=bool(owner and owner.active);save(row)
    require(row['passed'] and len(row['processes'])==21,row.get('error','cohort incomplete'))
    print('All21CLI/device processes passed;model/TTFT/TG/profile adoption unqualified.',flush=True)


if __name__=='__main__':
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);build.base.guards.no_model_owners([])
        if sys.argv[1:]==['prepare']:prepare()
        elif sys.argv[1:]==['run']:run()
        else:raise RuntimeError('qualify_prefill_candidate.py prepare|run')
